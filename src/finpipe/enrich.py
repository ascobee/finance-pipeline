"""Spark glue: categorize the merchants in silver and join the result back on.

``categorize_warehouse`` collects only the *distinct merchants* to the driver (bounded by the
number of merchants, not transactions), runs the categorizer, and writes:

* ``<warehouse>/categorization/merchant_categories``  one row per merchant (parquet)
* ``<warehouse>/categorization/review_queue.csv``      low-confidence / fallback merchants
* ``<warehouse>/categorization/cache.jsonl``           LLM answers (written by the categorizer)

``with_categories`` is a broadcast left join: the merchant table is tiny, so silver is never
shuffled, and a merchant with no category stays visible as NULL instead of vanishing.
"""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType, DoubleType, StringType, StructField, StructType,
)

from .categories import Category
from .categorize import (
    DEFAULT_REVIEW_THRESHOLD, Categorizer, LLMClient, MerchantCategory, load_corrections,
    write_review_queue,
)
from .spark_session import get_spark

CATEGORY_SCHEMA = StructType([
    StructField("merchant", StringType(), False),
    StructField("category", StringType(), False),
    StructField("confidence", DoubleType(), False),
    StructField("category_source", StringType(), False),
    StructField("model", StringType(), True),
    StructField("needs_review", BooleanType(), False),
    StructField("categorized_at", StringType(), True),
])


def with_categories(silver: DataFrame, categories: DataFrame) -> DataFrame:
    """Left-join merchant categories onto transactions (broadcast, no shuffle of silver)."""
    cats = categories.select(
        F.col("merchant").alias("_cat_merchant"),
        "category", "confidence", "category_source", "needs_review",
    )
    joined = silver.join(
        F.broadcast(cats), silver["merchant_norm"] == cats["_cat_merchant"], "left"
    )
    return joined.drop("_cat_merchant")


def categorize_warehouse(
    warehouse_dir: str | Path,
    client: LLMClient,
    corrections_path: str | Path | None = None,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
    batch_size: int = 25,
) -> dict:
    spark = get_spark()
    warehouse = Path(warehouse_dir)
    silver = spark.read.parquet(str(warehouse / "silver" / "transactions"))

    # A merchant is a rule-based Transfer only if EVERY one of its rows is flagged as a transfer.
    per_merchant = (
        silver.where(F.col("merchant_norm").isNotNull())
        .groupBy("merchant_norm")
        .agg(F.min(F.col("is_transfer").cast("int")).alias("all_transfer"))
        .collect()
    )
    merchants = sorted(r["merchant_norm"] for r in per_merchant)
    rules = {r["merchant_norm"]: Category.TRANSFER for r in per_merchant if r["all_transfer"] == 1}

    out_dir = warehouse / "categorization"
    categorizer = Categorizer(
        client,
        cache_path=out_dir / "cache.jsonl",
        corrections=load_corrections(corrections_path),
        rules=rules,
        batch_size=batch_size,
        review_threshold=review_threshold,
    )
    results: list[MerchantCategory] = categorizer.categorize(merchants)

    rows = [
        (r.merchant, r.category, float(r.confidence), r.source, r.model,
         bool(r.needs_review), r.categorized_at)
        for r in results
    ]
    spark.createDataFrame(rows, CATEGORY_SCHEMA).write.mode("overwrite").parquet(
        str(out_dir / "merchant_categories")
    )
    queued = write_review_queue(results, out_dir / "review_queue.csv")

    summary = {"merchants": len(results), "review_queue": queued, **dict(categorizer.stats)}
    for attr in ("input_tokens", "output_tokens"):
        if hasattr(client, attr):
            summary[attr] = getattr(client, attr)
    return summary

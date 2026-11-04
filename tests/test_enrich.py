"""Spark-backed tests for the categorization glue. Skipped when pyspark is not installed."""

import csv

import pytest
from fakes import ScriptedClient

pytest.importorskip("pyspark")

from finpipe import pipeline  # noqa: E402
from finpipe.enrich import categorize_warehouse, with_categories  # noqa: E402
from finpipe.normalize import normalize_merchant  # noqa: E402
from finpipe.synthetic import write_sample  # noqa: E402


def test_with_categories_is_a_left_join_that_keeps_unmatched_rows(spark):
    silver = spark.createDataFrame(
        [("t1", "Netflix"), ("t2", "Mystery Shop"), ("t3", "Netflix")],
        ["txn_id", "merchant_norm"],
    )
    cats = spark.createDataFrame(
        [("Netflix", "Subscriptions", 0.95, "llm", False)],
        ["merchant", "category", "confidence", "category_source", "needs_review"],
    )
    out = with_categories(silver, cats)
    assert out.count() == 3
    rows = {r["txn_id"]: r for r in out.collect()}
    assert rows["t1"]["category"] == "Subscriptions"
    assert rows["t3"]["category"] == "Subscriptions"
    assert rows["t2"]["category"] is None
    assert "_cat_merchant" not in out.columns


def test_categorize_warehouse_end_to_end(spark, tmp_path):
    raw, warehouse = tmp_path / "raw", tmp_path / "warehouse"
    write_sample(raw, seed=42)
    pipeline.run(raw, warehouse)

    truth = {}
    with (raw / "truth" / "merchants.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            truth[normalize_merchant(row["description_raw"])] = row["category"]
    client = ScriptedClient({m: (c, 0.95) for m, c in truth.items()})

    summary = categorize_warehouse(warehouse, client)
    assert summary["merchants"] == len(truth) == 28
    assert summary["rule"] == 3                    # the three transfer merchants never hit the LLM
    assert summary["llm"] == 25
    assert summary["review_queue"] == 0
    assert not any("transfer" in m.lower() or "payment thank you" in m.lower()
                   for m in client.requested)

    cats = spark.read.parquet(str(warehouse / "categorization" / "merchant_categories"))
    assert {r["merchant"]: r["category"] for r in cats.collect()} == truth

    silver = spark.read.parquet(str(warehouse / "silver" / "transactions"))
    enriched = with_categories(silver, cats)
    assert enriched.count() == silver.count()
    assert enriched.filter("category IS NULL").count() == 0

    again = ScriptedClient()
    second = categorize_warehouse(warehouse, again)    # everything now comes from the cache
    assert again.calls == []
    assert second["cache"] == 25 and second.get("llm", 0) == 0


def test_corrections_file_overrides_the_model(spark, tmp_path):
    raw, warehouse = tmp_path / "raw", tmp_path / "warehouse"
    write_sample(raw, seed=42)
    pipeline.run(raw, warehouse)
    corrections = tmp_path / "corrections.csv"
    corrections.write_text("merchant,category\nBlue Bottle Coffee,Entertainment\n")

    client = ScriptedClient({"Blue Bottle Coffee": ("Dining", 0.95)})
    summary = categorize_warehouse(warehouse, client, corrections_path=corrections)
    assert summary["correction"] == 1
    assert "Blue Bottle Coffee" not in client.requested

    cats = spark.read.parquet(str(warehouse / "categorization" / "merchant_categories"))
    row = cats.filter("merchant = 'Blue Bottle Coffee'").collect()[0]
    assert (row["category"], row["category_source"]) == ("Entertainment", "correction")

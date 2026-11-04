"""Bronze layer: land raw bank exports exactly as received.

Bronze rules:
* Values stay raw strings. No parsing, no cleanup, nothing that can fail on a weird row.
* Column *names* are sanitized to snake_case (Parquet rejects spaces in names).
* Each file is read on its own and the files are unioned by column name, so exports with
  different layouts (e.g. a signed ``Amount`` column vs ``Debit``/``Credit`` columns) can sit in
  one source folder without their columns being matched up by position.
* The one deliberate exception to "raw": a ``card_no`` column is cut down to its last four
  digits, so a full card number never lands in the warehouse.
* Every row carries lineage: ``_source_file`` and ``_load_date``.
* Append-only. A file that is already in bronze is skipped, so re-running the pipeline
  over the same folder does not grow bronze.
"""

from __future__ import annotations

import re
from functools import reduce
from pathlib import Path

from pyspark.errors import AnalysisException
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

SOURCES = ("checking", "credit_card")


def _snake(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()


def _read_file(spark: SparkSession, path: Path) -> DataFrame:
    df = (
        spark.read.option("header", True)
        .option("inferSchema", False)  # bronze is raw strings by design
        .csv(str(path))
    )
    df = df.toDF(*[_snake(c) for c in df.columns])
    if "card_no" in df.columns:
        digits = F.regexp_replace(F.col("card_no"), r"\D", "")
        df = df.withColumn("card_no", F.regexp_extract(digits, r"(\d{0,4})$", 1))
    return df.withColumn("_source_file", F.input_file_name())


def read_raw(spark: SparkSession, raw_dir: str | Path, source: str) -> DataFrame | None:
    """Read every CSV under ``<raw_dir>/<source>/`` as strings, or None if there are none."""
    files = sorted((Path(raw_dir) / source).glob("*.csv"))
    if not files:
        return None
    frames = [_read_file(spark, f) for f in files]
    df = reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), frames)
    return df.withColumn("_load_date", F.current_date().cast("string"))


def loaded_files(spark: SparkSession, bronze_path: str) -> set[str]:
    """Source files already present in a bronze table (empty set if the table is new)."""
    try:
        rows = spark.read.parquet(bronze_path).select("_source_file").distinct().collect()
    except AnalysisException:
        return set()
    return {r["_source_file"] for r in rows}


def append_bronze(spark: SparkSession, raw: DataFrame, bronze_path: str) -> int:
    """Append rows from files not yet in bronze. Returns the number of rows written."""
    seen = loaded_files(spark, bronze_path)
    new = raw.filter(~F.col("_source_file").isin(list(seen))) if seen else raw
    new = new.cache()
    n = new.count()
    if n:
        new.write.mode("append").partitionBy("_load_date").parquet(bronze_path)
    new.unpersist()
    return n

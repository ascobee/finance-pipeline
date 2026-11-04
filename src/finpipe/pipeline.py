"""Orchestrates raw CSVs -> bronze -> silver.

    python -m finpipe.pipeline --raw data/sample --warehouse data/warehouse
    python -m finpipe.pipeline --categorize        # also categorize merchants (API key needed)
    python -m finpipe.pipeline --categorize --gold # ...and build the gold tables
    python -m finpipe.pipeline --categorize --gold --check   # ...then verify (exit 1 on error)

Running it twice is safe: bronze skips files it has already loaded, and silver is rebuilt
from bronze with deterministic IDs, so the result is identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import reduce
from pathlib import Path

from pyspark.sql import DataFrame

from .ingest import SOURCES, append_bronze, read_raw
from .silver import build_silver, split_valid, parse_file
from .spark_session import get_spark


def run(raw_dir: str | Path, warehouse_dir: str | Path) -> dict:
    spark = get_spark()
    warehouse = Path(warehouse_dir)

    bronze_loaded: dict[str, int] = {}
    parsed_frames: list[DataFrame] = []
    for source in SOURCES:
    # for source in SOURCES2:
        raw = read_raw(spark, raw_dir, source)
        bronze_path = str(warehouse / "bronze" / source)
        if raw is not None:
            bronze_loaded[source] = append_bronze(spark, raw, bronze_path)
        if (warehouse / "bronze" / source).exists():
            # mergeSchema: files appended at different times may have different columns
            bronze = spark.read.option("mergeSchema", True).parquet(bronze_path)
            # parsed_frames.append(PARSERS[source](bronze))
            parsed_frames.append(parse_file(bronze, source))

    if not parsed_frames:
        raise SystemExit(f"No CSV files found under {raw_dir}/<source>/")

    parsed = reduce(DataFrame.unionByName, parsed_frames)
    valid, rejects = split_valid(parsed)

    silver = build_silver(valid).cache()
    silver_rows = silver.count()
    silver.write.mode("overwrite").parquet(str(warehouse / "silver" / "transactions"))
    silver.unpersist()

    rejects = rejects.cache()
    reject_rows = rejects.count()
    rejects.write.mode("overwrite").parquet(str(warehouse / "quarantine" / "transactions"))
    rejects.unpersist()

    return {
        "bronze_rows_loaded": bronze_loaded,
        "silver_rows": silver_rows,
        "quarantined_rows": reject_rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Run the finance pipeline (bronze -> silver).")
    p.add_argument("--raw", type=Path, default=Path("data/sample"))
    p.add_argument("--warehouse", type=Path, default=Path("data/warehouse"))
    p.add_argument("--categorize", action="store_true",
                   help="also categorize merchants with the Anthropic API (needs ANTHROPIC_API_KEY)")
    p.add_argument("--corrections", type=Path, default=Path("data/private/corrections.csv"),
                   help="merchant,category overrides that always win (default: data/private/)")
    p.add_argument("--gold", action="store_true",
                   help="also build the gold tables (uses merchant categories if they exist)")
    p.add_argument("--check", action="store_true",
                   help="run data quality checks last; exit 1 if any error-level check fails")
    p.add_argument("--model", default=None,
                   help="override the model (default: FINPIPE_MODEL env var, else the cheapest)")
    args = p.parse_args()

    client = None
    if args.categorize:  # build the client first so a missing key fails before any Spark work
        from .llm import AnthropicClient

        client = AnthropicClient(model=args.model)

    summary = run(args.raw, args.warehouse)
    if client is not None:
        from .enrich import categorize_warehouse

        summary["categorization"] = {
            "model": client.model,
            **categorize_warehouse(args.warehouse, client, args.corrections),
        }
    if args.gold:
        from .gold import write_gold

        summary["gold"] = write_gold(args.warehouse)
    report = None
    if args.check:
        from .quality import run_checks

        report = run_checks(args.warehouse, args.raw)
        summary["quality"] = {"ok": report.ok, **report.counts()}
    print(json.dumps(summary, indent=2))
    if report is not None:
        print(report.format(), file=sys.stderr)
        if not report.ok:
            sys.exit(1)


if __name__ == "__main__":
    main()

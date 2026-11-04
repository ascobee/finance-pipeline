"""Quality checks against a real warehouse. Skipped when pyspark is not installed."""

import csv
import json
import shutil
from datetime import date

import pytest
from fakes import ScriptedClient

pytest.importorskip("pyspark")

from finpipe import gold, ingest, pipeline, quality  # noqa: E402
from finpipe.enrich import categorize_warehouse  # noqa: E402
from finpipe.normalize import normalize_merchant  # noqa: E402
from finpipe.synthetic import write_sample  # noqa: E402

TODAY = date(2025, 7, 1)   # the sample data ends on 2025-06-30


def build(tmp_path, *, categorize=True, with_gold=True):
    raw, warehouse = tmp_path / "raw", tmp_path / "warehouse"
    write_sample(raw, seed=42, card_layout="mixed")  # exercises both card layouts
    pipeline.run(raw, warehouse)
    if categorize:
        truth = {}
        with (raw / "truth" / "merchants.csv").open(newline="") as fh:
            for r in csv.DictReader(fh):
                truth[normalize_merchant(r["description_raw"])] = r["category"]
        categorize_warehouse(warehouse, ScriptedClient({m: (c, 0.95) for m, c in truth.items()}))
    if with_gold:
        gold.write_gold(warehouse)
    return raw, warehouse


def test_sources_constant_matches_ingest():
    assert quality.SOURCES == ingest.SOURCES


def test_clean_pipeline_passes_every_check_with_no_warnings_or_skips(spark, tmp_path):
    raw, warehouse = build(tmp_path)
    report = quality.run_checks(warehouse, raw, today=TODAY)
    assert report.ok, report.format()
    assert report.counts()["warnings"] == 0 and report.counts()["skipped"] == 0, report.format()

    saved = json.loads((warehouse / "quality" / "report.json").read_text())
    assert saved["ok"] is True and saved["counts"] == report.counts()
    accounting = next(c for c in saved["checks"] if c["name"] == "rows_accounted_for")
    assert "duplicates collapsed 63" in accounting["detail"]    # 432 exported - 369 unique


def test_metrics_match_what_the_pipeline_produced(spark, tmp_path):
    raw, warehouse = build(tmp_path)
    m = quality.collect_metrics(warehouse)
    assert m["silver"]["rows"] == m["silver"]["distinct_ids"] == 369
    assert sum(m["bronze_rows_by_file"].values()) == 432
    assert set(m["bronze_rows_by_file"]) == set(quality.scan_raw(raw)["files"])
    assert m["quarantine_rows"] == 0
    assert m["categorization"]["merchants"] == 28 and m["categorization"]["missing"] == 0
    assert m["gold"]["category_spend_cents"] == m["gold"]["spend_cents"]
    tx = m["gold"]["transactions"]
    assert tx["rows"] == 369 and tx["spend_cents"] == m["gold"]["spend_cents"]
    assert tx["income_cents"] == m["gold"]["income_cents"]
    assert tx["transfer_net_cents"] == m["gold"]["transfer_net_cents"]


def test_tampered_transactions_table_is_caught(spark, tmp_path):
    """Dropping a row from gold.transactions must fail the reconciliation (and exit 1)."""
    raw, warehouse = build(tmp_path)
    path = str(warehouse / "gold" / "transactions")
    kept = spark.read.parquet(path).orderBy("txn_id").limit(368).collect()
    spark.createDataFrame(kept, spark.read.parquet(path).schema).write.mode("overwrite").parquet(
        str(tmp_path / "tampered"))
    import shutil
    shutil.rmtree(path)
    shutil.copytree(tmp_path / "tampered", path)
    report = quality.run_checks(warehouse, raw, today=TODAY)
    failed = {r.name for r in report.results if r.status == quality.FAIL}
    assert "gold_transactions_match" in failed and not report.ok


def test_layers_that_have_not_run_are_skipped_not_passed(spark, tmp_path):
    raw, warehouse = build(tmp_path, categorize=False, with_gold=False)
    report = quality.run_checks(warehouse, raw, today=TODAY)
    skipped = {r.name for r in report.results if r.status == quality.SKIP}
    assert skipped == {
        "merchants_categorized", "no_fallback_categories", "review_queue_size",
        "gold_reconciles_to_silver", "category_spend_matches_summary",
        "gold_transactions_match",
    }
    assert report.ok


def test_tampered_raw_file_is_caught_by_the_balance_check(spark, tmp_path):
    raw, warehouse = build(tmp_path)
    path = sorted((raw / "checking").glob("*.csv"))[0]
    rows = list(csv.reader(path.open(newline="")))
    rows[4][2] = "1.23"
    with path.open("w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    report = quality.run_checks(warehouse, raw, today=TODAY)
    assert not report.ok
    assert [r.name for r in report.errors] == ["running_balance_continuity"]


def test_raw_file_that_was_never_ingested_is_caught(spark, tmp_path):
    raw, warehouse = build(tmp_path)
    first = sorted((raw / "credit_card").glob("*.csv"))[0]
    shutil.copy(first, raw / "credit_card" / "credit_card_new_export.csv")
    report = quality.run_checks(warehouse, raw, today=TODAY)
    assert "raw_files_ingested" in [r.name for r in report.errors]


def test_unparseable_rows_are_quarantined_and_reported(spark, tmp_path):
    raw, warehouse = tmp_path / "raw", tmp_path / "warehouse"
    write_sample(raw, seed=42)
    path = sorted((raw / "checking").glob("*.csv"))[0]
    with path.open("a", newline="") as fh:
        csv.writer(fh).writerow(["13/45/2025", "GARBLED ROW", "oops", ""])
    pipeline.run(raw, warehouse)

    cfg = quality.QualityConfig(check_running_balance=False)
    report = quality.run_checks(warehouse, raw, cfg, today=TODAY)
    by_name = {r.name: r for r in report.results}
    assert by_name["quarantine_empty"].status == quality.FAIL    # warns...
    assert by_name["quarantine_rate"].status == quality.PASS     # ...but 1 of 433 is under 1%
    assert report.ok

    strict = quality.QualityConfig(check_running_balance=False, max_quarantine_rate=0.0)
    assert not quality.run_checks(warehouse, raw, strict, today=TODAY).ok


def test_cli_exit_code_reflects_the_result(spark, tmp_path, monkeypatch):
    raw, warehouse = build(tmp_path)
    args = ["--raw", str(raw), "--warehouse", str(warehouse)]
    assert quality.main(args) == 0

    path = sorted((raw / "checking").glob("*.csv"))[0]
    rows = list(csv.reader(path.open(newline="")))
    rows[4][2] = "1.23"
    with path.open("w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    assert quality.main(args) == 1

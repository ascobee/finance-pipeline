"""Quality rules and the raw-file scan: pure Python, no Spark needed."""

import csv
from copy import deepcopy
from datetime import date

import pytest

from finpipe.quality import (
    ERROR, FAIL, PASS, SKIP, WARN, QualityConfig, QualityReport, balance_breaks, card_sides,
    evaluate, scan_raw,
)
from finpipe.synthetic import write_sample

TODAY = date(2025, 7, 10)


def good_metrics():
    return {
        "bronze_rows_by_file": {"checking/a.csv": 50, "credit_card/b.csv": 150},
        "silver": {
            "rows": 190, "distinct_ids": 190, "min_date": "2025-01-01", "max_date": "2025-06-30",
            "zero_amount_rows": 0, "max_abs_amount_cents": 250_000, "total_cents": -12_345,
            "nulls": {c: 0 for c in
                      ("txn_id", "source", "txn_date", "amount_cents", "description_raw",
                       "merchant_norm")},
            "by_source": {"checking": {"rows": 50, "max_date": "2025-06-30"},
                          "credit_card": {"rows": 140, "max_date": "2025-06-29"}},
        },
        "quarantine_rows": 0,
        "categorization": {"merchants": 28, "missing": 0, "fallbacks": 0, "review_queue": 2},
        "gold": {"income_cents": 600_000, "spend_cents": 400_000, "transfer_net_cents": -212_345,
                 "category_spend_cents": 400_000,
                 "transactions": {"rows": 190, "income_cents": 600_000, "spend_cents": 400_000,
                                  "transfer_net_cents": -212_345}},
    }


def good_raw():
    return {"files": {
        "checking/a.csv": {"rows": 50, "balance": {"breaks": 0, "checked": 49, "examples": []}},
        "credit_card/b.csv": {"rows": 150, "balance": None, "sides": {
            "debit_only": 100, "credit_only": 50, "bad": 0, "examples": []}},
    }}


def by_name(results):
    return {r.name: r for r in results}


def run(metrics=None, raw="default", cfg=QualityConfig()):
    raw = good_raw() if raw == "default" else raw
    return by_name(evaluate(metrics or good_metrics(), raw, cfg, TODAY))


# ---------------------------------------------------------------- a healthy run


def test_healthy_data_passes_everything():
    results = evaluate(good_metrics(), good_raw(), QualityConfig(), TODAY)
    assert all(r.status == PASS for r in results), [
        (r.name, r.status, r.detail) for r in results if r.status != PASS]
    report = QualityReport(results)
    assert report.ok and report.counts()["errors"] == 0 and report.counts()["warnings"] == 0
    assert "OK" in report.format()


def test_duplicates_collapsed_are_reported_in_the_accounting_detail():
    r = run()["rows_accounted_for"]
    assert r.status == PASS and "duplicates collapsed 10" in r.detail   # 200 - 190 - 0


# ---------------------------------------------------------------- source checks


def test_raw_file_missing_from_bronze_is_an_error():
    raw = good_raw()
    raw["files"]["checking/new.csv"] = {"rows": 5, "balance": None}
    r = run(raw=raw)["raw_files_ingested"]
    assert (r.status, r.severity) == (FAIL, ERROR) and "checking/new.csv" in r.detail


def test_row_count_mismatch_between_raw_and_bronze_is_an_error():
    raw = good_raw()
    raw["files"]["credit_card/b.csv"]["rows"] = 151
    r = run(raw=raw)["raw_bronze_row_counts"]
    assert r.status == FAIL and "raw 151, bronze 150" in r.detail


def test_balance_breaks_are_an_error_with_an_example():
    raw = good_raw()
    raw["files"]["checking/a.csv"]["balance"] = {
        "breaks": 2, "checked": 49, "examples": ["2025-03-02: amount 5.00 but balance moved 6.00"]}
    r = run(raw=raw)["running_balance_continuity"]
    assert (r.status, r.observed) == (FAIL, 2) and "balance moved 6.00" in r.detail


def test_source_checks_skip_without_raw_files_and_never_silently_pass():
    results = run(raw=None)
    for name in ("raw_files_ingested", "raw_bronze_row_counts", "running_balance_continuity"):
        assert results[name].status == SKIP


def test_balance_check_can_be_disabled():
    r = run(cfg=QualityConfig(check_running_balance=False))["running_balance_continuity"]
    assert r.status == SKIP


def test_no_balance_column_anywhere_is_a_skip():
    raw = good_raw()
    raw["files"]["checking/a.csv"]["balance"] = None
    assert run(raw=raw)["running_balance_continuity"].status == SKIP


# ---------------------------------------------------------------- silver checks


def test_rows_appearing_from_nowhere_is_an_error():
    m = good_metrics()
    m["silver"]["rows"] = 250
    m["silver"]["distinct_ids"] = 250
    assert run(m)["rows_accounted_for"].status == FAIL


def test_duplicate_ids_are_an_error():
    m = good_metrics()
    m["silver"]["distinct_ids"] = 188
    r = run(m)["txn_id_unique"]
    assert (r.status, r.observed) == (FAIL, 2)


def test_nulls_in_required_columns_are_an_error_and_named():
    m = good_metrics()
    m["silver"]["nulls"]["txn_date"] = 3
    r = run(m)["required_fields_present"]
    assert r.status == FAIL and "txn_date (3)" in r.detail


def test_small_quarantine_warns_but_does_not_fail_the_run():
    m = good_metrics()
    m["bronze_rows_by_file"]["credit_card/b.csv"] = 151
    m["quarantine_rows"] = 1
    results = run(m, raw=None)
    assert results["quarantine_rate"].status == PASS          # 1 / 191 is under 1%
    assert (results["quarantine_empty"].status, results["quarantine_empty"].severity) == (FAIL, WARN)
    assert QualityReport(list(results.values())).ok


def test_large_quarantine_rate_is_an_error():
    m = good_metrics()
    m["bronze_rows_by_file"]["credit_card/b.csv"] = 160
    m["quarantine_rows"] = 10
    r = run(m, raw=None)["quarantine_rate"]
    assert (r.status, r.severity) == (FAIL, ERROR)
    assert not QualityReport([r]).ok


@pytest.mark.parametrize("lo, hi", [("1999-12-31", "2025-06-30"), ("2025-01-01", "2026-03-01")])
def test_implausible_dates_are_an_error(lo, hi):
    m = good_metrics()
    m["silver"]["min_date"], m["silver"]["max_date"] = lo, hi
    assert run(m, raw=None)["dates_plausible"].status == FAIL


def test_tomorrow_is_allowed_for_timezone_slack():
    m = good_metrics()
    m["silver"]["max_date"] = "2025-07-11"
    assert run(m, raw=None)["dates_plausible"].status == PASS


def test_empty_silver_skips_the_date_check():
    m = good_metrics()
    m["silver"].update(rows=0, distinct_ids=0, min_date=None, max_date=None, by_source={})
    m["bronze_rows_by_file"] = {}
    assert run(m, raw=None)["dates_plausible"].status == SKIP


def test_huge_and_zero_amounts_warn():
    m = good_metrics()
    m["silver"]["max_abs_amount_cents"] = 90_000_000
    m["silver"]["zero_amount_rows"] = 2
    results = run(m, raw=None)
    assert results["amounts_plausible"].status == FAIL
    assert results["no_zero_amounts"].status == FAIL
    assert results["amounts_plausible"].severity == WARN


def test_missing_source_and_stale_source_warn():
    m = good_metrics()
    del m["silver"]["by_source"]["credit_card"]
    m["silver"]["by_source"]["checking"]["max_date"] = "2025-05-01"
    results = run(m, raw=None)
    assert "credit_card" in results["sources_present"].detail
    assert results["data_freshness"].status == FAIL and "checking" in results["data_freshness"].detail


# ---------------------------------------------------------------- categorization and gold


def test_uncategorized_merchants_are_an_error_and_fallbacks_warn():
    m = good_metrics()
    m["categorization"].update(missing=3, fallbacks=2, review_queue=40)
    results = run(m, raw=None)
    assert results["merchants_categorized"].status == FAIL
    assert results["no_fallback_categories"].status == FAIL
    assert results["review_queue_size"].status == FAIL
    assert not QualityReport(list(results.values())).ok


def test_layers_that_have_not_run_are_skipped():
    m = good_metrics()
    m["categorization"] = None
    m["gold"] = None
    results = run(m, raw=None)
    assert all(results[n].status == SKIP for n in (
        "merchants_categorized", "gold_reconciles_to_silver", "category_spend_matches_summary",
        "gold_transactions_match"))


def test_gold_that_does_not_reconcile_is_an_error_with_the_difference():
    m = good_metrics()
    m["gold"]["spend_cents"] = 399_000
    m["gold"]["category_spend_cents"] = 399_000
    r = run(m, raw=None)["gold_reconciles_to_silver"]
    assert r.status == FAIL and "difference 1000" in r.detail


def test_category_table_disagreeing_with_summary_is_an_error():
    m = good_metrics()
    m["gold"]["category_spend_cents"] = 390_000
    assert run(m, raw=None)["category_spend_matches_summary"].status == FAIL


# ---------------------------------------------------------------- report


def test_report_serializes_and_formats_failures():
    m = good_metrics()
    m["silver"]["distinct_ids"] = 189
    report = QualityReport(evaluate(m, good_raw(), QualityConfig(), TODAY), "2026-10-20T00:00:00")
    data = report.to_dict()
    assert data["ok"] is False and data["counts"]["errors"] == 1
    assert {c["name"] for c in data["checks"]} >= {"txn_id_unique", "gold_reconciles_to_silver"}
    text = report.format()
    assert "FAILED" in text and "[ERROR] silver/txn_id_unique" in text


# ---------------------------------------------------------------- raw scan on real exports


def test_scan_raw_counts_rows_and_balances_are_continuous_on_the_sample(tmp_path):
    summary = write_sample(tmp_path, seed=42)
    raw = scan_raw(tmp_path)
    assert sum(f["rows"] for f in raw["files"].values()) == summary["exported_rows"]["total"]
    checking = [f for k, f in raw["files"].items() if k.startswith("checking/")]
    assert len(checking) == 2 and all(f["balance"]["breaks"] == 0 for f in checking)
    assert all(f["balance"]["checked"] > 0 for f in checking)
    assert all(f["balance"] is None for k, f in raw["files"].items() if k.startswith("credit_card/"))


def test_balance_check_catches_a_corrupted_amount(tmp_path):
    write_sample(tmp_path, seed=42)
    path = sorted((tmp_path / "checking").glob("*.csv"))[0]
    rows = list(csv.reader(path.open(newline="")))
    rows[5][2] = "999.99"          # tamper with one amount
    with path.open("w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    result = balance_breaks(path)
    assert result["breaks"] == 1   # only the tampered row; balances around it still agree
    assert "999.99" in result["examples"][0]


def test_balance_check_catches_a_dropped_row(tmp_path):
    write_sample(tmp_path, seed=42)
    path = sorted((tmp_path / "checking").glob("*.csv"))[0]
    rows = list(csv.reader(path.open(newline="")))
    del rows[7]                    # the bank row vanished
    with path.open("w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    assert balance_breaks(path)["breaks"] == 1


def test_balance_check_handles_newest_first_exports(tmp_path):
    write_sample(tmp_path, seed=42)
    path = sorted((tmp_path / "checking").glob("*.csv"))[0]
    rows = list(csv.reader(path.open(newline="")))
    flipped = [rows[0]] + rows[:0:-1]          # header, then data reversed
    with path.open("w", newline="") as fh:
        csv.writer(fh).writerows(flipped)
    assert balance_breaks(path)["breaks"] == 0


def test_file_without_a_balance_column_is_not_checked(tmp_path):
    write_sample(tmp_path, seed=42)
    card = sorted((tmp_path / "credit_card").glob("*.csv"))[0]
    assert balance_breaks(card) is None


def test_unparseable_balance_rows_are_reported_not_crashed_on(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("Date,Description,Amount,Balance\n01/01/2025,A,1.00,1.00\nbad,B,x,y\n")
    assert balance_breaks(path)["breaks"] == 1


def test_scan_raw_does_not_mutate_between_calls(tmp_path):
    write_sample(tmp_path, seed=42)
    assert deepcopy(scan_raw(tmp_path)) == scan_raw(tmp_path)


# ---------------------------------------------------------------- debit / credit card exports

CARD_HEADER = "Transaction Date,Posted Date,Card No.,Description,Category,Debit,Credit\n"


def card_file(tmp_path, *lines, header=CARD_HEADER):
    path = tmp_path / "card.csv"
    path.write_text(header + "".join(line + "\n" for line in lines))
    return path


def test_card_sides_counts_well_formed_rows(tmp_path):
    path = card_file(
        tmp_path,
        "2025-01-05,2025-01-06,1234,STARBUCKS,Dining,5.45,",
        "2025-01-26,2025-01-27,1234,PAYMENT THANK YOU,Payment,,900.00",
        "2025-02-01,2025-02-02,1234,BIG BUY,Shopping,\"1,234.56\",",
    )
    assert card_sides(path) == {"debit_only": 2, "credit_only": 1, "bad": 0, "examples": []}


@pytest.mark.parametrize("line,why", [
    ("2025-01-05,2025-01-06,1234,X,Dining,5.00,5.00", "both sides filled"),
    ("2025-01-05,2025-01-06,1234,X,Dining,,", "neither side filled"),
    ("2025-01-05,2025-01-06,1234,X,Dining,abc,", "unreadable or negative"),
    ("2025-01-05,2025-01-06,1234,X,Dining,-5.00,", "unreadable or negative"),
])
def test_card_sides_flags_malformed_rows(tmp_path, line, why):
    result = card_sides(card_file(tmp_path, line))
    assert result["bad"] == 1 and why in result["examples"][0] and "line 2" in result["examples"][0]


def test_card_sides_ignores_files_with_a_signed_amount_column(tmp_path):
    path = card_file(tmp_path, "2025-01-05,2025-01-06,X,Dining,Sale,5.00",
                     header="Transaction Date,Post Date,Description,Category,Type,Amount\n")
    assert card_sides(path) is None


def test_scan_raw_reports_sides_for_split_layout_cards_only(tmp_path):
    write_sample(tmp_path, seed=42, card_layout="mixed")
    files = scan_raw(tmp_path)["files"]
    cards = sorted(k for k in files if k.startswith("credit_card/"))
    assert files[cards[0]]["sides"] is None
    assert files[cards[1]]["sides"]["bad"] == 0 and files[cards[1]]["sides"]["debit_only"] > 0
    assert all(f["sides"] is None for k, f in files.items() if k.startswith("checking/"))


def raw_with_sides(**sides):
    raw = good_raw()
    raw["files"]["credit_card/b.csv"]["sides"] = {
        "debit_only": 100, "credit_only": 50, "bad": 0, "examples": [], **sides}
    return raw


def test_one_sided_check_passes_when_every_row_has_one_side():
    r = run(raw=raw_with_sides())["card_rows_one_sided"]
    assert (r.status, r.severity) == (PASS, WARN) and "150 rows" in r.detail


def test_one_sided_check_warns_with_an_example():
    r = run(raw=raw_with_sides(bad=2, examples=["line 7: both sides filled"]))["card_rows_one_sided"]
    assert (r.status, r.observed) == (FAIL, 2) and "line 7: both sides filled" in r.detail


def test_one_sided_check_is_skipped_without_split_layout_files():
    raw = good_raw()
    raw["files"]["credit_card/b.csv"]["sides"] = None
    assert run(raw=raw)["card_rows_one_sided"].status == SKIP
    assert run(raw=None)["card_rows_one_sided"].status == SKIP


# ---------------------------------------------------------------- gold transactions table


def test_transactions_table_that_matches_passes():
    r = run()["gold_transactions_match"]
    assert (r.status, r.severity) == (PASS, ERROR) and "190 transactions" in r.detail


@pytest.mark.parametrize("field, value, label", [
    ("rows", 189, "rows"), ("income_cents", 1, "income"), ("spend_cents", 399_999, "spend"),
    ("transfer_net_cents", 0, "transfers")])
def test_transactions_table_that_drifts_is_an_error_naming_the_field(field, value, label):
    m = good_metrics()
    m["gold"]["transactions"][field] = value
    r = run(m)["gold_transactions_match"]
    assert (r.status, r.observed) == (FAIL, 1) and label in r.detail


def test_old_warehouse_without_a_transactions_table_skips_the_check():
    m = good_metrics()
    m["gold"]["transactions"] = None
    r = run(m)["gold_transactions_match"]
    assert r.status == SKIP and "--gold" in r.detail

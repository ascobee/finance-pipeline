import csv
import json
from datetime import date

import pytest
from support import gold_records, row, sample_silver_rows

from finpipe import dashboard_data as dd
from finpipe.synthetic import write_sample


def make_gold(tmp_path):
    write_sample(tmp_path, seed=42)
    return gold_records(sample_silver_rows(tmp_path))


# ---------------------------------------------------------------- formatting


@pytest.mark.parametrize("cents, text", [(0, "$0.00"), (123456, "$1,234.56"),
                                         (-5, "-$0.05"), (-1_850_00, "-$1,850.00")])
def test_money(cents, text):
    assert dd.money(cents) == text


def test_pct_and_month_helpers():
    assert dd.pct(0.1256) == "12.6%" and dd.pct(None) is None
    assert dd.month_label("2025-03") == "March 2025" and dd.month_label("2025-12-01") == "December 2025"
    assert dd.previous_month("2025-03") == "2025-02"
    assert dd.previous_month("2025-01") == "2024-12"


def test_iso_accepts_dates_datetimes_and_strings():
    from datetime import datetime
    assert dd.iso(date(2025, 3, 1)) == dd.iso(datetime(2025, 3, 1, 9)) == dd.iso("2025-03-01")


@pytest.mark.parametrize("cadence, cents, monthly", [
    ("monthly", -1549, -1549), ("weekly", 1000, 4333), ("biweekly", 1200, 2600),
    ("quarterly", 3000, 1000), ("yearly", 12000, 1000)])
def test_charges_are_put_on_a_monthly_scale(cadence, cents, monthly):
    assert dd.monthly_equivalent_cents(cadence, cents) == monthly


# ---------------------------------------------------------------- months


def test_months_are_listed_oldest_first_and_latest_prefers_complete(tmp_path):
    gold = make_gold(tmp_path)
    assert dd.available_months(gold) == [f"2025-0{i}" for i in range(1, 7)]
    last = gold["monthly_summary"][-1]
    expected = "2025-06" if not last["is_partial_month"] else "2025-05"
    assert dd.latest_month(gold) == expected
    assert dd.latest_month(gold, prefer_complete=False) == "2025-06"


def test_latest_month_falls_back_to_a_partial_month_and_handles_empty():
    only_partial = {"monthly_summary": [{"month": date(2025, 1, 1), "is_partial_month": True}]}
    assert dd.latest_month(only_partial) == "2025-01"
    assert dd.latest_month({}) is None


# ---------------------------------------------------------------- tables


def test_overview_is_in_dollars_with_savings_rate_as_a_percentage(tmp_path):
    gold = make_gold(tmp_path)
    rows = dd.overview(gold)
    raw = {dd.month_key(r): r for r in gold["monthly_summary"]}
    assert [r["month"] for r in rows] == dd.available_months(gold)
    mar = next(r for r in rows if r["month"] == "2025-03")
    assert mar["income"] == raw["2025-03"]["income_cents"] / 100
    assert mar["net"] == pytest.approx(mar["income"] - mar["spend"])
    assert mar["savings_rate"] == pytest.approx(raw["2025-03"]["savings_rate"] * 100)


def test_category_breakdown_is_sorted_and_shares_add_up(tmp_path):
    gold = make_gold(tmp_path)
    rows = dd.category_breakdown(gold, "2025-03")
    assert [r["spend"] for r in rows] == sorted((r["spend"] for r in rows), reverse=True)
    assert sum(r["share_pct"] for r in rows) == pytest.approx(100)
    assert all(r["spend"] > 0 for r in rows)
    first = dd.category_breakdown(gold, "2025-01")
    assert all(r["change"] is None for r in first)       # no previous month to compare with


def test_category_trend_filters_and_orders_months(tmp_path):
    gold = make_gold(tmp_path)
    trend = dd.category_trend(gold, ["Groceries", "Dining"])
    assert list(trend) == dd.available_months(gold)
    assert all(set(v) <= {"Groceries", "Dining"} for v in trend.values())
    assert set(dd.category_trend(gold)["2025-03"]) > {"Groceries", "Dining"}


def test_spend_categories_start_with_the_biggest(tmp_path):
    gold = make_gold(tmp_path)
    cats = dd.spend_categories(gold)
    assert cats[0] == "Other" and "Groceries" in cats and "Income" not in cats


def test_recurring_view_totals_outflows_and_lists_biggest_first(tmp_path):
    gold = make_gold(tmp_path)
    view = dd.recurring_view(gold)
    outflows = [r for r in view["rows"] if not r["is_income"]]
    assert len(view["rows"]) == 7 and len(outflows) == 6
    assert outflows[0]["merchant"] == "Zelle Payment To J Martinez"      # other, largest outflow
    assert view["rows"][-1]["is_income"] is True                           # income listed last
    assert view["monthly_outflow"] == pytest.approx(-sum(r["monthly_cost"] for r in outflows))
    assert view["annual_outflow"] == pytest.approx(view["monthly_outflow"] * 12)
    assert {r["merchant"]: r["amount"] for r in outflows}["Xcel Energy Autopay"] == "varies"


def test_anomaly_table_is_sorted_by_severity_and_in_positive_dollars():
    gold = {"anomalies": [
        {"txn_date": date(2025, 2, 1), "merchant": "A", "category": "Shopping",
         "amount_cents": -30_000, "typical_cents": 4_000, "ratio": 7.5, "z_score": 9.0},
        {"txn_date": date(2025, 3, 1), "merchant": "B", "category": "Dining",
         "amount_cents": -90_000, "typical_cents": 5_000, "ratio": 18.0, "z_score": 40.0},
    ]}
    table = dd.anomaly_table(gold)
    assert [r["merchant"] for r in table] == ["B", "A"]
    assert table[0]["amount"] == 900.0 and table[0]["date"] == "2025-03-01"


# ---------------------------------------------------------------- warehouse files


def test_quality_status_summarizes_failures(tmp_path):
    assert dd.quality_status(tmp_path) is None
    (tmp_path / "quality").mkdir()
    (tmp_path / "quality" / "report.json").write_text(json.dumps({
        "ok": False, "generated_at": "2026-10-20T12:00:00+00:00",
        "counts": {"passed": 10, "errors": 1, "warnings": 1, "skipped": 0},
        "checks": [
            {"layer": "silver", "name": "txn_id_unique", "severity": "error", "status": "fail",
             "detail": "2 duplicate txn_id value(s)"},
            {"layer": "silver", "name": "quarantine_empty", "severity": "warn", "status": "fail",
             "detail": "1 unparseable row(s)"},
            {"layer": "gold", "name": "x", "severity": "error", "status": "pass", "detail": ""},
        ]}))
    status = dd.quality_status(tmp_path)
    assert status["ok"] is False and status["errors"] == 1 and status["warnings"] == 1
    assert [p["check"] for p in status["problems"]] == ["silver/txn_id_unique",
                                                        "silver/quarantine_empty"]


def test_review_queue_and_saved_summary_are_optional(tmp_path):
    assert dd.review_queue(tmp_path) == [] and dd.saved_summary(tmp_path, "2025-03") is None
    (tmp_path / "categorization").mkdir()
    with (tmp_path / "categorization" / "review_queue.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["merchant", "suggested_category", "confidence", "source"])
        w.writerow(["Xyz Holdings", "Shopping", "0.40", "llm"])
    assert dd.review_queue(tmp_path)[0]["merchant"] == "Xyz Holdings"
    (tmp_path / "summaries").mkdir()
    (tmp_path / "summaries" / "2025-03.json").write_text(json.dumps({"text": "hello"}))
    assert dd.saved_summary(tmp_path, "2025-03")["text"] == "hello"


# ---------------------------------------------------------------- transactions view


def txn(txn_id, d, merchant, category, cents, flow="spend", source="checking", desc=None):
    return {"txn_id": txn_id, "txn_date": d, "merchant": merchant, "category": category,
            "amount_cents": cents, "flow": flow, "source": source,
            "description_raw": desc if desc is not None else merchant.upper() + " 123"}


def small_gold():
    return {
        "transactions": [
            txn("a", date(2025, 3, 1), "Acme Payroll", "Income", 300_000, "income"),
            txn("b", date(2025, 3, 5), "Target", "Shopping", -10_000, source="credit_card"),
            txn("c", date(2025, 3, 5), "Target", "Shopping", 2_500, source="credit_card",
                desc="TARGET REFUND"),
            txn("d", date(2025, 3, 26), "Credit Card Payment", "Transfer", -90_000, "transfer"),
            txn("e", date(2025, 4, 2), "=SUM(A1)", "Other", -500, desc="-evil"),
        ],
        "monthly_summary": [
            {"month": date(2025, 3, 1), "income_cents": 300_000, "spend_cents": 7_500,
             "net_cents": 292_500, "transfer_net_cents": -90_000, "savings_rate": 0.97,
             "is_partial_month": False},
            {"month": date(2025, 4, 1), "income_cents": 0, "spend_cents": 500, "net_cents": -500,
             "transfer_net_cents": 0, "savings_rate": None, "is_partial_month": True},
        ],
    }


def test_transactions_view_filters_to_a_month_newest_first_with_id_tiebreak():
    view = dd.transactions_view(small_gold(), "2025-03")
    assert [r["txn_id"] for r in view["rows"]] == ["d", "b", "c", "a"]
    assert view["count"] == 4 and view["rows"][1]["amount"] == -100.0
    assert set(view["rows"][0]) == set(dd.TRANSACTION_COLUMNS)


def test_all_months_when_no_month_is_given():
    view = dd.transactions_view(small_gold())
    assert view["count"] == 5 and view["rows"][0]["txn_id"] == "e"


def test_totals_use_the_monthly_summary_signs():
    view = dd.transactions_view(small_gold(), "2025-03")
    assert (view["income_cents"], view["spend_cents"], view["transfer_net_cents"]) == (
        300_000, 7_500, -90_000)                          # the refund reduces spend
    assert view["net_cents"] == 300_000 - 10_000 + 2_500 - 90_000


def test_hide_transfers_removes_them_from_rows_and_totals():
    view = dd.transactions_view(small_gold(), "2025-03", hide_transfers=True)
    assert [r["txn_id"] for r in view["rows"]] == ["b", "c", "a"]
    assert view["transfer_net_cents"] == 0


def test_category_source_and_search_filters():
    g = small_gold()
    assert [r["txn_id"] for r in dd.transactions_view(g, categories=["Shopping"])["rows"]] == ["b", "c"]
    assert [r["txn_id"] for r in dd.transactions_view(g, sources=["checking"])["rows"]] == [
        "e", "d", "a"]
    assert dd.transactions_view(g, categories=[])["count"] == 0          # nothing selected
    assert [r["txn_id"] for r in dd.transactions_view(g, search="refund")["rows"]] == ["c"]
    assert [r["txn_id"] for r in dd.transactions_view(g, search="  TARGET ")["rows"]] == ["b", "c"]
    assert dd.transactions_view(g, "2025-03", search="nothing like this")["rows"] == []


def test_filter_options_and_months():
    g = small_gold()
    assert dd.transaction_months(g) == ["2025-03", "2025-04"]
    assert dd.transaction_filter_options(g) == {
        "categories": ["Income", "Other", "Shopping", "Transfer"],
        "sources": ["checking", "credit_card"]}
    assert dd.transaction_months({}) == [] and dd.transactions_view({})["count"] == 0


def test_transactions_match_summary_detects_agreement_and_drift():
    g = small_gold()
    assert dd.transactions_match_summary(g, "2025-03") is True
    g["monthly_summary"][0]["spend_cents"] = 7_000
    assert dd.transactions_match_summary(g, "2025-03") is False
    assert dd.transactions_match_summary({"monthly_summary": g["monthly_summary"]}, "2025-03") is None
    assert dd.transactions_match_summary(g, "2030-01") is None


def test_csv_download_neutralises_formula_text_but_keeps_numbers():
    import csv as csvmod
    import io

    view = dd.transactions_view(small_gold(), "2025-04")
    rows = list(csvmod.DictReader(io.StringIO(dd.transactions_csv(view["rows"]))))
    assert rows[0]["merchant"] == "'=SUM(A1)" and rows[0]["description"] == "'-evil"
    assert rows[0]["amount"] == "-5.0"                 # a real number, not text-prefixed
    assert list(rows[0]) == list(dd.TRANSACTION_COLUMNS)


def test_sample_data_transactions_reconcile_with_the_summary(tmp_path):
    gold = make_gold(tmp_path)
    assert len(gold["transactions"]) == 369
    for month in dd.available_months(gold):
        assert dd.transactions_match_summary(gold, month) is True, month
    everything = dd.transactions_view(gold)
    assert everything["count"] == 369 and everything["rows"][0]["source"] in ("checking", "credit_card")
    assert all(r["description"] for r in everything["rows"])

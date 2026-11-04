"""Spark gold tables checked against the pure-Python reference. Skipped without pyspark."""

import csv
import math
from datetime import date

import pytest
import reference_gold as ref
from fakes import ScriptedClient
from support import row, sample_silver_rows

pytest.importorskip("pyspark")

from pyspark.sql.types import (  # noqa: E402
    BooleanType, DateType, LongType, StringType, StructField, StructType,
)

from finpipe import gold, pipeline  # noqa: E402
from finpipe.enrich import categorize_warehouse  # noqa: E402
from finpipe.normalize import normalize_merchant  # noqa: E402
from finpipe.synthetic import write_sample  # noqa: E402

D = date

SCHEMA = StructType([
    StructField("txn_id", StringType(), False),
    StructField("txn_date", DateType(), False),
    StructField("amount_cents", LongType(), False),
    StructField("merchant_norm", StringType(), False),
    StructField("is_transfer", BooleanType(), False),
    StructField("category", StringType(), True),
])


def frame(spark, rows):
    return spark.createDataFrame(
        [(r["txn_id"], r["txn_date"], r["amount_cents"], r["merchant_norm"], r["is_transfer"],
          r["category"]) for r in rows],
        SCHEMA,
    )


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def assert_matches(spark_rows, key, expected: dict, fields):
    got = {key(r): r for r in spark_rows}
    assert set(got) == set(expected), (set(got) ^ set(expected))
    for k, exp in expected.items():
        for f in fields:
            assert same(got[k][f], exp[f]), f"{k} {f}: spark={got[k][f]!r} reference={exp[f]!r}"


def scenario_rows():
    """Refund, transfer, income, a gap month, a year boundary, a recurring bill, a spike."""
    rows = [
        row("pay1", D(2024, 12, 1), 300_000, "Acme Payroll", "Income"),
        row("t1", D(2024, 12, 5), -10_000, "Target", "Shopping"),
        row("t2", D(2024, 12, 9), 2_500, "Target", "Shopping"),
        row("cc", D(2024, 12, 26), -90_000, "Credit Card Payment", "Transfer", True),
        row("ks", D(2025, 2, 3), -7_000, "King Soopers", "Groceries"),
        row("none", D(2025, 2, 4), -1_234, "Mystery Shop", None),
        row("pay2", D(2025, 2, 28), 300_000, "Acme Payroll", "Income"),
    ]
    for i in range(5):   # monthly subscription, Jan-May
        rows.append(row(f"nf{i}", D(2025, 1 + i, 8), -1_549, "Netflix", "Subscriptions"))
    rows.append(row("nf_dup", D(2025, 2, 8), -1_549, "Netflix", "Subscriptions"))  # same-day repeat
    normal = [4_000, 4_400, 3_800, 4_200, 4_100, 3_900, 4_300, 4_000, 400_000]
    for i, c in enumerate(normal):
        rows.append(row(f"sm{i}", D.fromordinal(D(2024, 12, 2).toordinal() + 4 * i), -c,
                        "Spike Mart", "Shopping"))
    return rows


def test_monthly_summary_matches_reference(spark):
    rows = scenario_rows()
    out = gold.monthly_summary(gold.prepare(frame(spark, rows))).collect()
    assert_matches(
        out, lambda r: r["month"], ref.monthly_summary(rows),
        ["income_cents", "spend_cents", "net_cents", "savings_rate", "transfer_net_cents",
         "txn_count"],
    )


def test_monthly_category_spend_matches_reference(spark):
    rows = scenario_rows()
    out = gold.monthly_category_spend(gold.prepare(frame(spark, rows))).collect()
    assert_matches(
        out, lambda r: (r["month"], r["category"]), ref.monthly_category_spend(rows),
        ["spend_cents", "txn_count", "prev_month_spend_cents", "mom_change_cents",
         "mom_change_pct"],
    )


def test_recurring_charges_match_reference(spark):
    rows = scenario_rows()
    out = gold.recurring_charges(gold.prepare(frame(spark, rows))).collect()
    expected = ref.recurring_charges(rows)
    assert "Netflix" in expected
    assert_matches(
        out, lambda r: r["merchant"], expected,
        ["cadence", "occurrences", "gap_share", "typical_amount_cents", "amount_stable",
         "is_income", "first_date", "last_date", "next_expected_date", "category"],
    )


def test_anomalies_match_reference_and_find_the_spike(spark):
    rows = scenario_rows()
    out = gold.anomalies(gold.prepare(frame(spark, rows))).collect()
    expected = ref.anomalies(rows)
    assert list(expected) == ["sm8"]
    assert_matches(out, lambda r: r["txn_id"], expected,
                   ["merchant", "amount_cents", "typical_cents", "ratio", "z_score"])


def test_refund_nets_and_transfer_is_excluded(spark):
    rows = [
        row("a", D(2025, 1, 5), -10_000, "Target", "Shopping"),
        row("b", D(2025, 1, 9), 2_500, "Target", "Shopping"),
        row("c", D(2025, 1, 25), -90_000, "Credit Card Payment", "Transfer", True),
    ]
    jan = gold.monthly_summary(gold.prepare(frame(spark, rows))).collect()[0]
    assert (jan["spend_cents"], jan["income_cents"], jan["transfer_net_cents"]) == (
        7_500, 0, -90_000)


def test_gap_months_are_filled_and_partial_months_flagged(spark):
    rows = [row("a", D(2025, 1, 15), -1_000, "Shop", "Shopping"),
            row("b", D(2025, 3, 20), -2_000, "Shop", "Shopping")]
    out = {r["month"]: r for r in gold.monthly_summary(gold.prepare(frame(spark, rows))).collect()}
    assert sorted(out) == [D(2025, 1, 1), D(2025, 2, 1), D(2025, 3, 1)]
    assert out[D(2025, 2, 1)]["spend_cents"] == 0 and out[D(2025, 2, 1)]["txn_count"] == 0
    assert out[D(2025, 1, 1)]["is_partial_month"] is True    # data starts on the 15th
    assert out[D(2025, 2, 1)]["is_partial_month"] is False
    assert out[D(2025, 3, 1)]["is_partial_month"] is True    # data ends on the 20th


def test_fully_covered_months_are_not_partial(spark):
    rows = [row("a", D(2025, 1, 1), -1_000, "Shop", "Shopping"),
            row("b", D(2025, 1, 31), -2_000, "Shop", "Shopping")]
    (jan,) = gold.monthly_summary(gold.prepare(frame(spark, rows))).collect()
    assert jan["is_partial_month"] is False


def test_prepare_works_without_a_category_column(spark):
    df = frame(spark, [row("a", D(2025, 1, 5), -1_000, "Shop", "Shopping")]).drop("category")
    prepared = gold.prepare(df)
    assert prepared.collect()[0]["category"] == "Uncategorized"


def test_build_gold_returns_all_tables(spark):
    tables = gold.build_gold(frame(spark, scenario_rows()))
    assert tuple(tables) == gold.TABLES


def test_transactions_table_matches_reference_and_nets_to_the_summary(spark):
    rows = scenario_rows()
    df = frame(spark, rows)
    out = gold.transactions(df).collect()
    assert_matches(out, lambda r: r["txn_id"], ref.transactions(rows),
                   ["txn_date", "merchant", "category", "amount_cents", "flow"])
    assert [r["txn_date"] for r in out] == sorted((r["txn_date"] for r in out), reverse=True)

    for m in gold.monthly_summary(gold.prepare(df)).collect():
        month = [r for r in out if (r["txn_date"].year, r["txn_date"].month)
                 == (m["month"].year, m["month"].month)]
        assert m["spend_cents"] == -sum(r["amount_cents"] for r in month if r["flow"] == "spend")
        assert m["income_cents"] == sum(r["amount_cents"] for r in month if r["flow"] == "income")
        assert m["transfer_net_cents"] == sum(
            r["amount_cents"] for r in month if r["flow"] == "transfer")


def test_transactions_table_tolerates_frames_without_source_or_raw_description(spark):
    (only,) = gold.transactions(frame(spark, [row("a", D(2025, 1, 5), -1_000, "Shop")])).collect()
    assert only["source"] is None and only["description_raw"] is None
    assert only["category"] == "Uncategorized" and only["flow"] == "spend"


def test_write_gold_end_to_end_matches_reference(spark, tmp_path):
    raw, warehouse = tmp_path / "raw", tmp_path / "warehouse"
    write_sample(raw, seed=42)
    pipeline.run(raw, warehouse)

    truth = {}
    with (raw / "truth" / "merchants.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            truth[normalize_merchant(r["description_raw"])] = r["category"]
    categorize_warehouse(warehouse, ScriptedClient({m: (c, 0.95) for m, c in truth.items()}))

    summary = gold.write_gold(warehouse)
    assert summary["categories_applied"] is True
    assert summary["monthly_summary"] == 6
    assert summary["recurring_charges"] == 7
    assert summary["anomalies"] == 0

    rows = sample_silver_rows(raw)   # independent pure-Python build of the same ledger
    months = spark.read.parquet(str(warehouse / "gold" / "monthly_summary")).collect()
    assert_matches(
        months, lambda r: r["month"], ref.monthly_summary(rows),
        ["income_cents", "spend_cents", "net_cents", "transfer_net_cents", "txn_count"],
    )
    total = sum(r["income_cents"] - r["spend_cents"] + r["transfer_net_cents"] for r in months)
    assert total == sum(r["amount_cents"] for r in rows)   # gold reconciles to silver

    recurring = spark.read.parquet(str(warehouse / "gold" / "recurring_charges")).collect()
    assert_matches(recurring, lambda r: r["merchant"], ref.recurring_charges(rows),
                   ["cadence", "occurrences", "typical_amount_cents", "amount_stable"])

    assert summary["transactions"] == len(rows)
    txns = spark.read.parquet(str(warehouse / "gold" / "transactions")).collect()
    assert_matches(txns, lambda r: r["txn_id"], ref.transactions(rows),
                   ["txn_date", "merchant", "category", "amount_cents", "flow"])
    expected = {r["txn_id"]: r for r in rows}
    for t in txns:   # provenance columns come straight from silver
        assert t["source"] == expected[t["txn_id"]]["source"]
        assert t["description_raw"] == expected[t["txn_id"]]["description_raw"]


def test_write_gold_without_categories_reports_uncategorized(spark, tmp_path):
    raw, warehouse = tmp_path / "raw", tmp_path / "warehouse"
    write_sample(raw, seed=42)
    pipeline.run(raw, warehouse)

    summary = gold.write_gold(warehouse)   # no categorization has run
    assert summary["categories_applied"] is False
    cats = spark.read.parquet(str(warehouse / "gold" / "monthly_category_spend"))
    assert [r["category"] for r in cats.select("category").distinct().collect()] == [
        "Uncategorized"]
    assert summary["recurring_charges"] == 7   # transfers are still excluded via is_transfer

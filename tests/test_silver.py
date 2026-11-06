"""Spark-backed tests. Skipped automatically when pyspark is not installed."""

import json

import pytest

pytest.importorskip("pyspark")

from finpipe import pipeline  # noqa: E402
from finpipe.ids import description_key, transaction_id  # noqa: E402
from finpipe.silver import (  # noqa: E402
    build_silver,
    split_valid,
    parse_file
)
from finpipe.synthetic import write_sample  # noqa: E402

CHECKING_COLS = ["date", "description", "amount", "balance", "_source_file"]
CARD_COLS = [
    "transaction_date", "post_date", "description", "category", "type", "amount",
    "_source_file",
]


def checking_df(spark, rows):
    return spark.createDataFrame(rows, CHECKING_COLS)


def card_df(spark, rows):
    return spark.createDataFrame(rows, CARD_COLS)


# ---------------------------------------------------------------- parsers


def test_checking_parses_signed_money_with_commas_and_dates(spark):
    df = checking_df(spark, [
        ("01/02/2025", "PAYROLL ACME", "2,468.24", "6,468.24", "f1"),
        ("01/03/2025", "OTHER", "-1,850.00", "4,618.24", "f1"),
        ("01/04/2025", "COFFEE", "$-4.75", "4,613.49", "f1"),
    ])
    rows = {r["description_raw"]: r for r in parse_file(df, "checking").collect()}
    assert rows["PAYROLL ACME"]["amount_cents"] == 246824
    assert rows["OTHER"]["amount_cents"] == -185000
    assert rows["COFFEE"]["amount_cents"] == -475
    assert str(rows["OTHER"]["txn_date"]) == "2025-01-03"


def test_card_purchases_become_negative_and_payments_positive(spark):
    df = card_df(spark, [
        ("2025-01-05", "2025-01-06", "STARBUCKS #1", "Food & Drink", "Sale", "5.45", "f1"),
        ("2025-01-26", "2025-01-27", "PAYMENT THANK YOU", "Payment", "Payment", "-900.00", "f1"),
    ])
    rows = {r["description_raw"]: r["amount_cents"] for r in parse_file(df, "credit_card").collect()}
    assert rows["STARBUCKS #1"] == -545
    assert rows["PAYMENT THANK YOU"] == 90000


SPLIT_COLS = [
    "transaction_date", "posted_date", "card_no", "description", "category", "debit", "credit",
    "_source_file",
]


def split_df(spark, rows, columns=SPLIT_COLS):
    """All-string frame; an explicit schema lets a column hold only NULLs."""
    from pyspark.sql.types import StringType, StructField, StructType

    return spark.createDataFrame(rows, StructType([StructField(c, StringType()) for c in columns]))


def split_row(desc, debit=None, credit=None, date="2025-01-05", f="f1"):
    return date, date, "1234", desc, "Dining", debit, credit, f


def test_debit_credit_card_purchases_negative_and_credits_positive(spark):
    df = split_df(spark, [
        split_row("STARBUCKS #1", debit="5.45"),
        split_row("PAYMENT THANK YOU", credit="900.00", date="2025-01-26"),
        split_row("AMAZON", debit="$1,234.56", date="2025-01-07"),
        split_row("REFUND TARGET", credit="12.10", date="2025-01-08"),
    ])
    rows = {r["description_raw"]: r["amount_cents"] for r in parse_file(df, "credit_card").collect()}
    assert rows == {
        "STARBUCKS #1": -545, "PAYMENT THANK YOU": 90000, "AMAZON": -123456,
        "REFUND TARGET": 1210,
    }


def test_debit_credit_blank_cells_may_be_empty_strings_or_nulls(spark):
    df = split_df(spark, [
        split_row("A", debit="5.00", credit=""),
        split_row("B", debit=" ", credit="7.00"),
        split_row("C", debit="1.00", credit=None),
    ])
    rows = {r["description_raw"]: r["amount_cents"] for r in parse_file(df, "credit_card").collect()}
    assert rows == {"A": -500, "B": 700, "C": -100}


def test_debit_credit_rows_with_both_neither_or_bad_values_are_quarantined(spark):
    df = split_df(spark, [
        split_row("GOOD", debit="5.00"),
        split_row("BOTH", debit="5.00", credit="5.00"),
        split_row("NEITHER"),
        split_row("BLANKS", debit=" ", credit=""),
        split_row("TEXT", debit="abc"),
        split_row("NEGATIVE", debit="-5.00"),
    ])
    valid, rejects = split_valid(parse_file(df, "credit_card"))
    assert [r["description_raw"] for r in valid.collect()] == ["GOOD"]
    assert {r["description_raw"] for r in rejects.collect()} == {
        "BOTH", "NEITHER", "BLANKS", "TEXT", "NEGATIVE"}


def test_zero_amount_debit_is_kept(spark):
    df = split_df(spark, [split_row("FREE TRIAL", debit="0.00")])
    assert [r["amount_cents"] for r in parse_file(df, "credit_card").collect()] == [0]


def test_bronze_holding_both_card_layouts_parses_each_row_by_its_own_columns(spark):
    columns = ["transaction_date", "description", "category", "amount", "debit", "credit",
               "_source_file"]
    df = split_df(spark, [
        ("2025-01-05", "OLD SALE", "Dining", "5.45", None, None, "old.csv"),
        ("2025-01-26", "OLD PAYMENT", "Payment", "-900.00", None, None, "old.csv"),
        ("2025-02-05", "NEW SALE", "Dining", None, "6.00", None, "new.csv"),
        ("2025-02-26", "NEW PAYMENT", "Payment", None, None, "900.00", "new.csv"),
        ("2025-02-27", "NEW BAD", "Dining", None, "1.00", "1.00", "new.csv"),
    ], columns)
    rows = {r["description_raw"]: r["amount_cents"] for r in parse_file(df, "credit_card").collect()}
    assert rows == {"OLD SALE": -545, "OLD PAYMENT": 90000, "NEW SALE": -600,
                    "NEW PAYMENT": 90000, "NEW BAD": None}


def test_card_export_with_no_recognised_amount_columns_fails_loudly(spark):
    df = split_df(spark, [("2025-01-05", "X", "Dining", "f1")],
                  ["transaction_date", "description", "category", "_source_file"])
    with pytest.raises(ValueError, match="neither an 'amount' column nor 'debit'/'credit'"):
        parse_file(df, "credit_card")


def test_unparseable_rows_are_quarantined_not_dropped_silently(spark):
    df = checking_df(spark, [
        ("01/02/2025", "GOOD ROW", "-1.00", "0", "f1"),
        ("not-a-date", "BAD DATE", "-1.00", "0", "f1"),
        ("01/02/2025", "BAD AMOUNT", "abc", "0", "f1"),
        ("01/02/2025", "   ", "-1.00", "0", "f1"),
    ])
    valid, rejects = split_valid(parse_file(df, "checking"))
    assert valid.count() == 1
    assert rejects.count() == 3


# ---------------------------------------------------------------- identity / dedupe


def test_overlapping_exports_collapse_to_one_row_per_transaction(spark):
    row_a = ("01/10/2025", "POS DEBIT KING SOOPERS #1 DENVER CO", "-50.00", "0")
    row_b = ("01/11/2025", "XCEL ENERGY AUTOPAY 1", "-90.00", "0")
    df = checking_df(spark, [
        (*row_a, "file_jan_feb"), (*row_b, "file_jan_feb"),
        (*row_a, "file_jan_mar"), (*row_b, "file_jan_mar"),  # same transactions, later file
    ])
    valid, _ = split_valid(parse_file(df, "checking"))
    assert build_silver(valid).count() == 2


def test_identical_same_day_purchases_in_one_file_are_kept_separate(spark):
    same = ("01/10/2025", "SQ *BLUE BOTTLE COFFEE 1 DENVER CO", "-5.25", "0")
    df = checking_df(spark, [(*same, "f1"), (*same, "f1")])
    valid, _ = split_valid(parse_file(df, "checking"))
    assert build_silver(valid).count() == 2


def test_same_day_duplicates_survive_overlap_between_files(spark):
    same = ("01/10/2025", "SQ *BLUE BOTTLE COFFEE 1 DENVER CO", "-5.25", "0")
    df = checking_df(spark, [(*same, "f1"), (*same, "f1"), (*same, "f2"), (*same, "f2")])
    valid, _ = split_valid(parse_file(df, "checking"))
    assert build_silver(valid).count() == 2  # two coffees, not four and not one


def test_latest_file_wins_on_conflict(spark):
    row = ("01/10/2025", "OTHER", "-10.00", "0")
    df = checking_df(spark, [(*row, "a_older"), (*row, "z_newer")])
    valid, _ = split_valid(parse_file(df, "checking"))
    out = build_silver(valid).collect()
    assert len(out) == 1 and out[0]["source_file"] == "z_newer"


def test_spark_id_matches_python_reference_implementation(spark):
    df = checking_df(spark, [("01/10/2025", "  sq *Blue   Bottle 1 ", "-5.25", "0", "f1")])
    valid, _ = split_valid(parse_file(df, "checking"))
    row = build_silver(valid).collect()[0]
    expected = transaction_id(
        "checking", "2025-01-10", -525, description_key("  sq *Blue   Bottle 1 "), 1
    )
    assert row["txn_id"] == expected


def test_transfers_are_flagged(spark):
    df = checking_df(spark, [
        ("01/25/2025", "CREDIT CARD PAYMENT - THANK YOU", "-900.00", "0", "f1"),
        ("01/12/2025", "XCEL ENERGY AUTOPAY 1", "-90.00", "0", "f1"),
    ])
    valid, _ = split_valid(parse_file(df, "checking"))
    flags = {r["description_raw"]: r["is_transfer"] for r in build_silver(valid).collect()}
    assert flags["CREDIT CARD PAYMENT - THANK YOU"] is True
    assert flags["XCEL ENERGY AUTOPAY 1"] is False


# ---------------------------------------------------------------- end to end


def test_end_to_end_matches_ground_truth_and_is_idempotent(spark, tmp_path):
    raw, warehouse = tmp_path / "raw", tmp_path / "warehouse"
    summary = write_sample(raw, seed=42)
    expected = summary["unique_transactions"]["total"]
    exported = summary["exported_rows"]

    first = pipeline.run(raw, warehouse)
    assert first["bronze_rows_loaded"] == {
        "checking": exported["checking"], "credit_card": exported["credit_card"],
    }
    assert first["silver_rows"] == expected
    assert first["quarantined_rows"] == 0

    second = pipeline.run(raw, warehouse)  # nothing new to load, nothing changes
    assert second["bronze_rows_loaded"] == {"checking": 0, "credit_card": 0}
    assert second["silver_rows"] == expected

    silver = spark.read.parquet(str(warehouse / "silver" / "transactions"))
    assert silver.select("txn_id").distinct().count() == expected
    assert json.loads((raw / "truth" / "summary.json").read_text()) == summary


# ---------------------------------------------------------------- card layouts end to end


def silver_rows(spark, warehouse):
    df = spark.read.parquet(str(warehouse / "silver" / "transactions"))
    return {r["txn_id"]: (r["amount_cents"], r["source"]) for r in df.collect()}


def test_every_card_layout_produces_identical_silver(spark, tmp_path):
    """Same transactions exported three ways must give the same transaction IDs and amounts,
    including when two overlapping exports use different layouts."""
    results = {}
    for layout in ("amount", "debit_credit", "mixed"):
        raw, warehouse = tmp_path / f"raw_{layout}", tmp_path / f"wh_{layout}"
        summary = write_sample(raw, seed=42, card_layout=layout)
        result = pipeline.run(raw, warehouse)
        assert result["quarantined_rows"] == 0, layout
        assert result["silver_rows"] == summary["unique_transactions"]["total"], layout
        results[layout] = silver_rows(spark, warehouse)
    assert results["amount"] == results["debit_credit"] == results["mixed"]


def test_card_number_is_cut_to_four_digits_in_bronze(spark, tmp_path):
    raw, warehouse = tmp_path / "raw", tmp_path / "wh"
    write_sample(raw, seed=42, card_layout="debit_credit")
    for f in (raw / "credit_card").glob("*.csv"):  # pretend the bank exports full numbers
        f.write_text(f.read_text().replace(",1234,", ",4111 1111 1111 9876,"))
    pipeline.run(raw, warehouse)
    bronze = spark.read.option("mergeSchema", True).parquet(str(warehouse / "bronze" / "credit_card"))
    assert {r["card_no"] for r in bronze.select("card_no").distinct().collect()} == {"9876"}
    silver = spark.read.parquet(str(warehouse / "silver" / "transactions"))
    assert "card_no" not in silver.columns


def test_second_run_with_a_new_layout_file_adds_only_that_file(spark, tmp_path):
    """Bronze already holds old-layout files; a new debit/credit file arrives later."""
    raw, warehouse = tmp_path / "raw", tmp_path / "wh"
    write_sample(raw, seed=42, card_layout="amount")
    first = pipeline.run(raw, warehouse)

    newer = tmp_path / "other"
    write_sample(newer, seed=42, card_layout="debit_credit")
    extra = sorted((newer / "credit_card").glob("*.csv"))[-1]
    (raw / "credit_card" / "zz_newer_layout.csv").write_text(extra.read_text())

    second = pipeline.run(raw, warehouse)
    assert second["bronze_rows_loaded"]["checking"] == 0
    assert second["bronze_rows_loaded"]["credit_card"] > 0
    assert second["silver_rows"] == first["silver_rows"]  # same transactions, just re-exported
    assert second["quarantined_rows"] == 0

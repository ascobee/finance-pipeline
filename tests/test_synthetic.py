import csv
import json
from datetime import date

import pytest

from finpipe.synthetic import generate_transactions, write_sample


def test_same_seed_is_reproducible(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_sample(a, seed=7)
    write_sample(b, seed=7)
    for rel in ("checking", "credit_card"):
        for f in sorted((a / rel).glob("*.csv")):
            assert f.read_bytes() == (b / rel / f.name).read_bytes()


def test_different_seed_differs(tmp_path):
    a = generate_transactions(seed=1)
    b = generate_transactions(seed=2)
    assert [t.description for t in a] != [t.description for t in b]


def test_overlapping_windows_produce_duplicate_rows(tmp_path):
    summary = write_sample(tmp_path, seed=42)
    exported = summary["exported_rows"]["total"]
    unique = summary["unique_transactions"]["total"]
    assert exported > unique  # overlap means the same transaction shows up in two files


def test_two_files_per_source_and_both_layouts_differ(tmp_path):
    write_sample(tmp_path, seed=42)
    checking = sorted((tmp_path / "checking").glob("*.csv"))
    cards = sorted((tmp_path / "credit_card").glob("*.csv"))
    assert len(checking) == 2 and len(cards) == 2
    with checking[0].open() as fh:
        assert next(csv.reader(fh)) == ["Date", "Description", "Amount", "Balance"]
    with cards[0].open() as fh:
        assert next(csv.reader(fh))[0] == "Transaction Date"


def test_same_day_identical_purchases_exist():
    txns = generate_transactions(seed=42)
    keys = [(t.source, t.txn_date, t.description, t.ledger_cents) for t in txns]
    assert len(keys) > len(set(keys))


def test_card_file_uses_inverted_sign_convention(tmp_path):
    write_sample(tmp_path, seed=42)
    rows = []
    for f in (tmp_path / "credit_card").glob("*.csv"):
        with f.open() as fh:
            rows.extend(csv.DictReader(fh))
    sales = [r for r in rows if r["Type"] == "Sale"]
    payments = [r for r in rows if r["Type"] == "Payment"]
    assert sales and all(float(r["Amount"]) > 0 for r in sales)
    assert payments and all(float(r["Amount"]) < 0 for r in payments)


def test_summary_is_written_and_consistent(tmp_path):
    summary = write_sample(tmp_path, seed=42)
    on_disk = json.loads((tmp_path / "truth" / "summary.json").read_text())
    assert on_disk == summary
    u = summary["unique_transactions"]
    assert u["total"] == u["checking"] + u["credit_card"]


def test_respects_date_range(tmp_path):
    txns = generate_transactions(seed=3, start=date(2025, 2, 1), end=date(2025, 2, 28))
    assert txns and all(date(2025, 2, 1) <= t.txn_date <= date(2025, 2, 28) for t in txns)


def read_card_rows(folder):
    rows = []
    for f in sorted(folder.glob("*.csv")):
        with f.open() as fh:
            rows.append(list(csv.DictReader(fh)))
    return rows


def signed_cents(row):
    """Silver-convention cents for a card row in either layout."""
    from decimal import Decimal

    if "Amount" in row:
        return -int(Decimal(row["Amount"]) * 100)
    debit, credit = row["Debit"], row["Credit"]
    return int(Decimal(credit) * 100) if credit else -int(Decimal(debit) * 100)


def test_debit_credit_layout_has_positive_values_in_exactly_one_column(tmp_path):
    write_sample(tmp_path, seed=42, card_layout="debit_credit")
    files = read_card_rows(tmp_path / "credit_card")
    with sorted((tmp_path / "credit_card").glob("*.csv"))[0].open() as fh:
        assert next(csv.reader(fh)) == [
            "Transaction Date", "Posted Date", "Card No.", "Description", "Category", "Debit",
            "Credit",
        ]
    rows = [r for f in files for r in f]
    assert rows and all(bool(r["Debit"]) != bool(r["Credit"]) for r in rows)
    assert all(float(r["Debit"] or r["Credit"]) > 0 for r in rows)
    assert any(r["Debit"] for r in rows) and any(r["Credit"] for r in rows)


def test_every_card_layout_describes_the_same_transactions(tmp_path):
    from collections import Counter

    seen = {}
    for layout in ("amount", "debit_credit", "mixed"):
        out = tmp_path / layout
        summary = write_sample(out, seed=42, card_layout=layout)
        rows = [r for f in read_card_rows(out / "credit_card") for r in f]
        seen[layout] = (
            summary["exported_rows"],
            Counter((r["Transaction Date"], r["Description"], signed_cents(r)) for r in rows),
        )
    assert seen["amount"] == seen["debit_credit"] == seen["mixed"]


def test_mixed_layout_uses_one_of_each(tmp_path):
    write_sample(tmp_path, seed=42, card_layout="mixed")
    headers = []
    for f in sorted((tmp_path / "credit_card").glob("*.csv")):
        with f.open() as fh:
            headers.append(next(csv.reader(fh)))
    assert "Amount" in headers[0] and "Debit" in headers[1]


def test_unknown_card_layout_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="card_layout"):
        write_sample(tmp_path, card_layout="nope")

"""Pure-Python helpers shared by tests: build silver-equivalent rows without Spark."""

from __future__ import annotations

import csv
import re
from collections import Counter
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from finpipe.ids import description_key, transaction_id
from finpipe.normalize import normalize_merchant

_TRANSFER = re.compile(r"CREDIT CARD PAYMENT|PAYMENT THANK YOU|ONLINE TRANSFER")


def _cents(text: str) -> int:
    return int((Decimal(text.replace("$", "").replace(",", "")) * 100).to_integral_value())


def sample_silver_rows(root: Path) -> list[dict]:
    """Parse + dedupe the synthetic exports in ``root`` (same algorithm as silver.py),
    attaching the ground-truth category. Mirrors what the Spark pipeline should produce."""
    parsed = []
    for f in sorted((root / "checking").glob("*.csv")):
        with f.open(newline="") as fh:
            for r in csv.DictReader(fh):
                d = datetime.strptime(r["Date"], "%m/%d/%Y").date()
                parsed.append(("checking", f.name, d, _cents(r["Amount"]), r["Description"]))
    for f in sorted((root / "credit_card").glob("*.csv")):
        with f.open(newline="") as fh:
            for r in csv.DictReader(fh):
                d = date.fromisoformat(r["Transaction Date"])
                parsed.append(("credit_card", f.name, d, -_cents(r["Amount"]), r["Description"]))

    seen: Counter = Counter()
    best: dict = {}
    for src, fname, d, cents, desc in parsed:
        key = description_key(desc)
        seen[(src, fname, d, cents, key)] += 1
        tid = transaction_id(src, d.isoformat(), cents, key, seen[(src, fname, d, cents, key)])
        if tid not in best or fname > best[tid][1]:
            best[tid] = (src, fname, d, cents, desc)

    truth = {}
    with (root / "truth" / "merchants.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            truth[normalize_merchant(r["description_raw"])] = r["category"]

    rows = []
    for tid, (src, _fname, d, cents, desc) in best.items():
        merchant = normalize_merchant(desc)
        rows.append({
            "txn_id": tid, "txn_date": d, "amount_cents": cents, "merchant_norm": merchant,
            "is_transfer": bool(_TRANSFER.search(description_key(desc))),
            "category": truth[merchant],
            "source": src, "description_raw": desc,
        })
    return rows


def row(txn_id, txn_date, amount_cents, merchant, category=None, is_transfer=False) -> dict:
    return {"txn_id": txn_id, "txn_date": txn_date, "amount_cents": amount_cents,
            "merchant_norm": merchant, "is_transfer": is_transfer, "category": category}


def gold_records(rows) -> dict:
    """Gold-table-shaped records (as read_gold returns them), built with the pure reference."""
    import calendar

    import reference_gold as ref

    summary = ref.monthly_summary(rows)
    lo, hi = min(r["txn_date"] for r in rows), max(r["txn_date"] for r in rows)
    monthly = []
    for month, v in summary.items():
        last_day = date(month.year, month.month, calendar.monthrange(month.year, month.month)[1])
        partial = (month == lo.replace(day=1) and lo > month) or (
            month == hi.replace(day=1) and hi < last_day)
        monthly.append({"month": month, **v, "is_partial_month": partial})

    categories = [{"month": m, "category": c, **cell}
                  for (m, c), cell in ref.monthly_category_spend(rows).items()]
    recurring = [{"merchant": k, **v} for k, v in ref.recurring_charges(rows).items()]
    by_id = {r["txn_id"]: r for r in rows}
    anomalies = [
        {"txn_id": t, "txn_date": by_id[t]["txn_date"], "category": by_id[t]["category"], **v}
        for t, v in ref.anomalies(rows).items()
    ]
    transactions = [
        {"txn_id": t, **v, "source": by_id[t].get("source"),
         "description_raw": by_id[t].get("description_raw")}
        for t, v in ref.transactions(rows).items()
    ]
    return {"monthly_summary": monthly, "monthly_category_spend": categories,
            "recurring_charges": recurring, "anomalies": anomalies,
            "transactions": transactions}

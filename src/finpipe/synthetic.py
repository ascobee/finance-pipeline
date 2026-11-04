"""Synthetic bank exports with realistic mess (stdlib only, fully deterministic per seed).

Generates two sources with *different* CSV layouts, noisy merchant descriptions, overlapping
export windows (so the same transaction appears in two files), and same-day identical
purchases (so the deterministic-ID occurrence index is exercised).

Besides the CSVs it writes ground truth under ``truth/``:

* ``merchants.csv``  raw description -> canonical merchant + category. This is the seed of
  the labeled "golden set" for measuring LLM categorization accuracy.
* ``summary.json``  expected unique-transaction counts, so tests can assert that the pipeline
  collapses overlapping exports to exactly the right number of rows.

Usage: ``python -m finpipe.synthetic --out data/sample --seed 42``
"""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import random
import re
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

CITIES = ["DENVER", "BOULDER", "ARVADA", "GOLDEN", "LAKEWOOD", "AURORA", "LITTLETON",
          "CENTRAL CITY", "BLACK HAWK"]

DEFAULT_START = date(2025, 1, 1)
DEFAULT_END = date(2025, 6, 30)


@dataclass(frozen=True)
class Merchant:
    canonical: str
    category: str
    source: str  # "checking" or "credit_card"
    templates: tuple[str, ...]
    amount_range: tuple[str, str] = ("0", "0")  # magnitude, ignored when fixed_prices is set
    per_month: tuple[int, int] = (1, 1)  # random-day merchants only
    days: tuple[int, ...] | None = None  # fixed days of month (recurring charges)
    fixed_prices: tuple[str, ...] | None = None
    money_in: bool = False  # True for payroll, card payments, refunds


@dataclass(frozen=True)
class Txn:
    source: str
    txn_date: date
    description: str
    ledger_cents: int  # negative = money out, positive = money in
    canonical: str
    category: str


CATALOG: tuple[Merchant, ...] = (
    # --- checking account ---
    Merchant("Acme Corp Payroll", "Income", "checking",
             ("PAYROLL ACME CORP DIRECT DEP {n6}",), ("2450.00", "2550.00"),
             days=(1, 15), money_in=True),
    Merchant("Landlord Other", "Other", "checking",
             ("ZELLE PAYMENT TO J MARTINEZ {n6}",), fixed_prices=("1850.00",), days=(1,)),
    Merchant("Xcel Energy", "Utilities", "checking",
             ("XCEL ENERGY AUTOPAY {n6}",), ("60.00", "140.00"), days=(12,)),
    Merchant("Xfinity", "Utilities", "checking",
             ("XFINITY INTERNET {n6}",), fixed_prices=("79.99",), days=(20,)),
    Merchant("Credit Card Payment", "Transfer", "checking",
             ("CREDIT CARD PAYMENT - THANK YOU",), ("900.00", "1600.00"), days=(25,)),
    Merchant("Savings Transfer", "Transfer", "checking",
             ("ONLINE TRANSFER TO SAVINGS {n4}",), fixed_prices=("400.00",), days=(2,)),
    Merchant("King Soopers", "Groceries", "checking",
             ("POS DEBIT KING SOOPERS #{n4} {city} CO",), ("40.00", "120.00"), per_month=(2, 3)),
    Merchant("ATM Withdrawal", "Other", "checking",
             ("ATM WITHDRAWAL {city} CO {n4}",), ("40.00", "100.00"), per_month=(0, 1)),
    # --- credit card ---
    Merchant("Blue Bottle Coffee", "Dining", "credit_card",
             ("SQ *BLUE BOTTLE COFFEE {n4} {city} CO", "BLUE BOTTLE COFFEE {city}"),
             fixed_prices=("4.75", "5.25", "6.00"), per_month=(6, 10)),
    Merchant("Starbucks", "Dining", "credit_card",
             ("STARBUCKS STORE {n5} {city} CO", "STARBUCKS #{n5}"),
             fixed_prices=("5.45", "6.15"), per_month=(3, 6)),
    Merchant("King Soopers", "Groceries", "credit_card",
             ("KING SOOPERS #{n4} {city} CO",), ("35.00", "140.00"), per_month=(3, 5)),
    Merchant("Whole Foods", "Groceries", "credit_card",
             ("WHOLEFDS DEN {n5}", "WHOLE FOODS MARKET #{n5}"), ("25.00", "90.00"),
             per_month=(1, 3)),
    Merchant("Safeway", "Groceries", "credit_card",
             ("SAFEWAY #{n4} {city} CO",), ("20.00", "80.00"), per_month=(1, 2)),
    Merchant("Amazon", "Shopping", "credit_card",
             ("AMZN Mktp US*{a8}", "AMAZON.COM*{a8} AMZN.COM/BILL WA"), ("8.00", "120.00"),
             per_month=(4, 7)),
    Merchant("Target", "Shopping", "credit_card",
             ("TARGET {n8}", "TARGET.COM  *"), ("15.00", "110.00"), per_month=(1, 3)),
    Merchant("Uber", "Transportation", "credit_card",
             ("UBER   *TRIP HELP.UBER.COM CA", "UBER *TRIP"), ("9.00", "45.00"),
             per_month=(1, 4)),
    Merchant("Shell", "Fuel", "credit_card",
             ("SHELL OIL {n10} {city} CO", "SHELL SERVICE STATION {n6}"), ("32.00", "68.00"),
             per_month=(2, 4)),
    Merchant("Conoco", "Fuel", "credit_card",
             ("CONOCO #{n5} {city} CO",), ("30.00", "60.00"), per_month=(0, 2)),
    Merchant("Netflix", "Subscriptions", "credit_card",
             ("NETFLIX.COM LOS GATOS CA",), fixed_prices=("15.49",), days=(8,)),
    Merchant("Spotify", "Subscriptions", "credit_card",
             ("SPOTIFY USA {n6}",), fixed_prices=("11.99",), days=(14,)),
    Merchant("Apple", "Subscriptions", "credit_card",
             ("APPLE.COM/BILL CUPERTINO CA",), fixed_prices=("2.99",), days=(3,)),
    Merchant("Chipotle", "Dining", "credit_card",
             ("CHIPOTLE {n4}", "CHIPOTLE ONLINE"), ("11.00", "26.00"), per_month=(2, 4)),
    Merchant("The Pizza Place", "Dining", "credit_card",
             ("TST* THE PIZZA PLACE {city} CO",), ("18.00", "48.00"), per_month=(1, 2)),
    Merchant("DoorDash", "Dining", "credit_card",
             ("DOORDASH*{a6}", "DD *DOORDASH {a6}"), ("22.00", "55.00"), per_month=(1, 4)),
    Merchant("CVS Pharmacy", "Health", "credit_card",
             ("CVS/PHARMACY #{n5}",), ("6.00", "45.00"), per_month=(1, 2)),
    Merchant("Airbnb", "Travel", "credit_card",
             ("AIRBNB * {a10}",), ("180.00", "520.00"), per_month=(0, 1)),
    Merchant("Delta Air Lines", "Travel", "credit_card",
             ("DELTA AIR 006{n10}",), ("220.00", "480.00"), per_month=(0, 1)),
    Merchant("AMC Theatres", "Entertainment", "credit_card",
             ("AMC {n4} ONLINE",), ("14.00", "34.00"), per_month=(0, 2)),
    Merchant("Card Payment", "Transfer", "credit_card",
             ("PAYMENT THANK YOU",), ("900.00", "1600.00"), days=(26,), money_in=True),
)

# Labels a card issuer might put in its own "Category" column (not our taxonomy).
_BANK_CATEGORY = {
    "Dining": "Food & Drink", "Groceries": "Groceries", "Shopping": "Shopping",
    "Transportation": "Travel", "Fuel": "Gas", "Subscriptions": "Bills & Utilities",
    "Health": "Health & Wellness", "Travel": "Travel", "Entertainment": "Entertainment",
    "Transfer": "Payment", "Other": "Miscellaneous",
}

_ALNUM = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def _fill(template: str, rng: random.Random) -> str:
    def repl(m: re.Match[str]) -> str:
        kind = m.group(1)
        if kind == "city":
            return rng.choice(CITIES)
        width = int(kind[1:])
        alphabet = "0123456789" if kind[0] == "n" else _ALNUM
        return "".join(rng.choice(alphabet) for _ in range(width))

    return re.sub(r"\{(\w+)\}", repl, template)


def _cents(value: str) -> int:
    return int((Decimal(value) * 100).to_integral_value())


def _month_starts(start: date, end: date) -> list[date]:
    out, cur = [], date(start.year, start.month, 1)
    while cur <= end:
        out.append(cur)
        cur = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
    return out


def generate_transactions(
    seed: int = 42, start: date = DEFAULT_START, end: date = DEFAULT_END
) -> list[Txn]:
    """All ground-truth transactions for the date range, in deterministic order."""
    rng = random.Random(seed)
    txns: list[Txn] = []

    for month in _month_starts(start, end):
        last_day = calendar.monthrange(month.year, month.month)[1]
        for m in CATALOG:
            if m.days is not None:
                days = list(m.days)
            else:
                days = [rng.randint(1, last_day) for _ in range(rng.randint(*m.per_month))]
            for d in days:
                when = date(month.year, month.month, min(d, last_day))
                if not start <= when <= end:
                    continue
                if m.fixed_prices:
                    cents = _cents(rng.choice(m.fixed_prices))
                else:
                    lo, hi = (_cents(x) for x in m.amount_range)
                    cents = rng.randint(lo, hi)
                txns.append(Txn(
                    source=m.source,
                    txn_date=when,
                    description=_fill(rng.choice(m.templates), rng),
                    ledger_cents=cents if m.money_in else -cents,
                    canonical=m.canonical,
                    category=m.category,
                ))

        # One exact same-day duplicate purchase per month (two identical coffees).
        coffees = [t for t in txns
                   if t.canonical == "Blue Bottle Coffee"
                   and (t.txn_date.year, t.txn_date.month) == (month.year, month.month)]
        if coffees:
            txns.append(rng.choice(coffees))

    txns.sort(key=lambda t: (t.txn_date, t.source, t.description, t.ledger_cents))
    return txns


def _money(cents: int) -> str:
    return f"{Decimal(cents) / 100:,.2f}"


def _write_checking(path: Path, txns: list[Txn], balances: dict[int, int]) -> int:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Date", "Description", "Amount", "Balance"])
        for t in txns:  # ascending
            w.writerow([t.txn_date.strftime("%m/%d/%Y"), t.description,
                        _money(t.ledger_cents), _money(balances[id(t)])])
    return len(txns)


CARD_LAYOUTS = ("amount", "debit_credit", "mixed")


def _write_card(path: Path, txns: list[Txn], rng: random.Random, layout: str = "amount") -> int:
    """Write a card export, newest first.

    ``amount``: one signed column where purchases are positive and payments negative.
    ``debit_credit``: two always-positive columns; purchases are debits, payments credits.
    """
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        if layout == "debit_credit":
            w.writerow(["Transaction Date", "Posted Date", "Card No.", "Description",
                        "Category", "Debit", "Credit"])
        else:
            w.writerow(["Transaction Date", "Post Date", "Description", "Category", "Type",
                        "Amount"])
        for t in reversed(txns):  # this issuer exports newest first
            posted = t.txn_date + timedelta(days=rng.randint(1, 2))
            category = _BANK_CATEGORY.get(t.category, "Miscellaneous")
            if layout == "debit_credit":
                money = f"{Decimal(abs(t.ledger_cents)) / 100:.2f}"
                debit, credit = ("", money) if t.ledger_cents > 0 else (money, "")
                w.writerow([t.txn_date.isoformat(), posted.isoformat(), "1234", t.description,
                            category, debit, credit])
            else:
                kind = "Payment" if t.ledger_cents > 0 else "Sale"
                # Card exports show purchases as positive numbers, payments as negative.
                w.writerow([t.txn_date.isoformat(), posted.isoformat(), t.description,
                            category, kind, f"{Decimal(-t.ledger_cents) / 100:.2f}"])
    return len(txns)


def write_sample(
    out_dir: Path,
    seed: int = 42,
    start: date = DEFAULT_START,
    end: date = DEFAULT_END,
    card_layout: str = "amount",
) -> dict:
    """Write raw exports and ground truth under ``out_dir``; return the summary dict.

    ``card_layout`` picks the card export format: ``amount`` (signed), ``debit_credit`` (split),
    or ``mixed`` (first file signed, second file split, like an issuer changing its format).
    The underlying transactions are identical in every case.
    """
    if card_layout not in CARD_LAYOUTS:
        raise ValueError(f"card_layout must be one of {CARD_LAYOUTS}, got {card_layout!r}")
    txns = generate_transactions(seed, start, end)
    rng = random.Random(seed + 1)  # post-date jitter, independent of the ledger itself

    # Two export windows per source that overlap by ~a month.
    mid = start + (end - start) // 2
    windows = [(start, mid), (mid - timedelta(days=30), end)]

    (out_dir / "checking").mkdir(parents=True, exist_ok=True)
    (out_dir / "credit_card").mkdir(parents=True, exist_ok=True)
    (out_dir / "truth").mkdir(parents=True, exist_ok=True)

    # Running balance over the full checking ledger so overlapping files agree.
    balance, balances = 400_000, {}
    for t in txns:
        if t.source == "checking":
            balance += t.ledger_cents
            balances[id(t)] = balance

    exported = {"checking": 0, "credit_card": 0}
    for source in ("checking", "credit_card"):
        source_txns = [t for t in txns if t.source == source]
        for n, (lo, hi) in enumerate(windows):
            chunk = [t for t in source_txns if lo <= t.txn_date <= hi]
            name = f"{source}_{lo:%Y%m%d}_{hi:%Y%m%d}.csv"
            path = out_dir / source / name
            if source == "checking":
                exported[source] += _write_checking(path, chunk, balances)
            else:
                layout = (("amount", "debit_credit")[n] if card_layout == "mixed"
                          else card_layout)
                exported[source] += _write_card(path, chunk, rng, layout)

    truth: dict[str, tuple[str, str, str]] = {}
    for t in txns:
        truth[t.description] = (t.canonical, t.category, t.source)
    with (out_dir / "truth" / "merchants.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["description_raw", "merchant_canonical", "category", "source"])
        for desc in sorted(truth):
            w.writerow([desc, *truth[desc]])

    unique = {s: sum(1 for t in txns if t.source == s) for s in exported}
    summary = {
        "seed": seed,
        "unique_transactions": {**unique, "total": sum(unique.values())},
        "exported_rows": {**exported, "total": sum(exported.values())},
        "windows": [[lo.isoformat(), hi.isoformat()] for lo, hi in windows],
    }
    (out_dir / "truth" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic bank exports.")
    p.add_argument("--out", type=Path, default=Path("data/sample"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start", type=date.fromisoformat, default=DEFAULT_START)
    p.add_argument("--end", type=date.fromisoformat, default=DEFAULT_END)
    p.add_argument("--card-layout", choices=CARD_LAYOUTS, default="amount",
                   help="card export format: signed amount, debit/credit columns, or one of each")
    args = p.parse_args()
    summary = write_sample(args.out, args.seed, args.start, args.end, args.card_layout)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

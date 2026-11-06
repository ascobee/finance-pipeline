"""Pure-Python data prep for the dashboard and the monthly summary.

Everything here takes the gold tables as ``{name: [row dicts]}`` (see ``gold_io.read_gold``) and
returns plain lists and dicts, so it is unit-tested without Spark, pandas or Streamlit. The
Streamlit app only turns these into DataFrames and widgets.

Money is integer cents in gold; helpers here convert to dollars only for display.
"""

from __future__ import annotations

import csv
import json
from datetime import date, datetime
from pathlib import Path

# How many times a cadence charges per month, for putting every recurring charge on one scale.
CHARGES_PER_MONTH = {
    "weekly": 52 / 12,
    "biweekly": 26 / 12,
    "monthly": 1.0,
    "quarterly": 1 / 3,
    "yearly": 1 / 12,
}

MONTH_NAMES = ("January", "February", "March", "April", "May", "June", "July", "August",
               "September", "October", "November", "December")


# --------------------------------------------------------------------------- formatting


def iso(value) -> str:
    """ISO string for a date / datetime / string."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def money(cents: int | float) -> str:
    """``-123456`` -> ``-$1,234.56``."""
    sign = "-" if cents < 0 else ""
    return f"{sign}${abs(cents) / 100:,.2f}"


def pct(ratio: float | None) -> str | None:
    return None if ratio is None else f"{ratio * 100:.1f}%"


def month_label(month: str) -> str:
    """``2026-09`` or ``2026-09-01`` -> ``September 2026``."""
    year, mon = int(month[:4]), int(month[5:7])
    return f"{MONTH_NAMES[mon - 1]} {year}"


def previous_month(month: str) -> str:
    year, mon = int(month[:4]), int(month[5:7])
    return f"{year - (mon == 1)}-{(mon - 2) % 12 + 1:02d}"


# --------------------------------------------------------------------------- gold lookups


def month_key(row: dict) -> str:
    return iso(row["month"])[:7]


def summary_by_month(gold: dict) -> dict[str, dict]:
    return {month_key(r): r for r in gold.get("monthly_summary", [])}


def available_months(gold: dict) -> list[str]:
    return sorted(summary_by_month(gold))


def latest_month(gold: dict, prefer_complete: bool = True) -> str | None:
    """Newest month, preferring the newest month the data covers fully."""
    by_month = summary_by_month(gold)
    months = sorted(by_month)
    if not months:
        return None
    if prefer_complete:
        complete = [m for m in months if not by_month[m].get("is_partial_month")]
        if complete:
            return complete[-1]
    return months[-1]


def monthly_equivalent_cents(cadence: str, amount_cents: int) -> int:
    """What a recurring charge costs per month, whatever its cadence."""
    return round(amount_cents * CHARGES_PER_MONTH[cadence])


# --------------------------------------------------------------------------- dashboard tables


def overview(gold: dict) -> list[dict]:
    """One row per month, in dollars, for the income / spend / net chart."""
    rows = []
    for month in available_months(gold):
        r = summary_by_month(gold)[month]
        rows.append({
            "month": month,
            "income": r["income_cents"] / 100,
            "spend": r["spend_cents"] / 100,
            "net": r["net_cents"] / 100,
            "savings_rate": None if r["savings_rate"] is None else r["savings_rate"] * 100,
            "partial": bool(r.get("is_partial_month")),
        })
    return rows


def category_breakdown(gold: dict, month: str) -> list[dict]:
    """Spend by category for one month, largest first, with the change from the month before."""
    total = summary_by_month(gold).get(month, {}).get("spend_cents", 0)
    rows = []
    for r in gold.get("monthly_category_spend", []):
        if month_key(r) != month or r["spend_cents"] == 0:
            continue
        rows.append({
            "category": r["category"],
            "spend": r["spend_cents"] / 100,
            "share_pct": (r["spend_cents"] / total * 100) if total > 0 else None,
            "change": None if r["mom_change_cents"] is None else r["mom_change_cents"] / 100,
            "change_pct": None if r["mom_change_pct"] is None else r["mom_change_pct"] * 100,
            "transactions": r["txn_count"],
        })
    return sorted(rows, key=lambda r: -r["spend"])


def category_trend(gold: dict, categories: list[str] | None = None) -> dict[str, dict[str, float]]:
    """``{month: {category: dollars}}`` for the selected categories (all when None)."""
    out: dict[str, dict[str, float]] = {m: {} for m in available_months(gold)}
    for r in gold.get("monthly_category_spend", []):
        if categories is None or r["category"] in categories:
            out.setdefault(month_key(r), {})[r["category"]] = r["spend_cents"] / 100
    return dict(sorted(out.items()))


def spend_categories(gold: dict) -> list[str]:
    """Categories ordered by total spend, so the biggest are selected first."""
    totals: dict[str, int] = {}
    for r in gold.get("monthly_category_spend", []):
        totals[r["category"]] = totals.get(r["category"], 0) + r["spend_cents"]
    return [c for c, _ in sorted(totals.items(), key=lambda kv: -kv[1])]


def recurring_view(gold: dict) -> dict:
    """Recurring charges on a common monthly scale, plus outflow totals."""
    rows = []
    for r in gold.get("recurring_charges", []):
        monthly = monthly_equivalent_cents(r["cadence"], r["typical_amount_cents"])
        rows.append({
            "merchant": r["merchant"],
            "category": r["category"],
            "cadence": r["cadence"],
            "typical_amount": r["typical_amount_cents"] / 100,
            "monthly_cost": monthly / 100,
            "annual_cost": monthly * 12 / 100,
            "amount": "stable" if r["amount_stable"] else "varies",
            "is_income": bool(r["is_income"]),
            "last_charge": iso(r["last_date"]),
            "next_expected": iso(r["next_expected_date"]),
        })
    rows.sort(key=lambda r: (r["is_income"], r["monthly_cost"]))   # biggest outflows first
    outflows = [r for r in rows if not r["is_income"]]
    return {
        "rows": rows,
        "monthly_outflow": -sum(r["monthly_cost"] for r in outflows),
        "annual_outflow": -sum(r["annual_cost"] for r in outflows),
    }


def anomaly_table(gold: dict) -> list[dict]:
    return [
        {
            "date": iso(r["txn_date"]),
            "merchant": r["merchant"],
            "category": r["category"],
            "amount": -r["amount_cents"] / 100,
            "typical": r["typical_cents"] / 100,
            "times_typical": r["ratio"],
        }
        for r in sorted(gold.get("anomalies", []), key=lambda r: -r["z_score"])
    ]


# --------------------------------------------------------------------------- warehouse files


def quality_status(warehouse_dir: str | Path) -> dict | None:
    """Counts and age of the latest quality report, or None if checks have not run."""
    path = Path(warehouse_dir) / "quality" / "report.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return {
        "ok": data["ok"],
        "generated_at": data.get("generated_at", ""),
        **data["counts"],
        "problems": [
            {"check": f"{c['layer']}/{c['name']}", "severity": c["severity"], "detail": c["detail"]}
            for c in data["checks"] if c["status"] == "fail"
        ],
    }


def review_queue(warehouse_dir: str | Path) -> list[dict]:
    path = Path(warehouse_dir) / "categorization" / "review_queue.csv"
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def saved_summary(warehouse_dir: str | Path, month: str) -> dict | None:
    path = Path(warehouse_dir) / "summaries" / f"{month}.json"
    return json.loads(path.read_text()) if path.exists() else None


# --------------------------------------------------------------------------- transactions

TRANSACTION_COLUMNS = (
    "date", "merchant", "category", "amount", "flow", "source", "description", "txn_id",
)


def transaction_months(gold: dict) -> list[str]:
    return sorted({iso(r["txn_date"])[:7] for r in gold.get("transactions", [])})


def transaction_filter_options(gold: dict) -> dict:
    """Values to offer in the filter widgets."""
    rows = gold.get("transactions", [])
    return {
        "categories": sorted({r["category"] for r in rows}),
        "sources": sorted({r["source"] for r in rows if r.get("source")}),
    }


def transactions_view(
    gold: dict,
    month: str | None = None,
    categories: list[str] | None = None,
    sources: list[str] | None = None,
    search: str = "",
    hide_transfers: bool = False,
) -> dict:
    """Filtered, newest-first transactions plus totals, for the Transactions tab.

    ``month`` is ``YYYY-MM`` or None for every month. ``categories`` / ``sources`` of None mean
    no filter (an empty list matches nothing). ``search`` is a case-insensitive substring of the
    merchant or the raw bank description. Totals use the same signs as ``monthly_summary``
    (``spend_cents`` is positive, refunds reduce it) so they can be compared to it directly.
    """
    needle = search.strip().casefold()
    kept = []
    for r in gold.get("transactions", []):
        if month and iso(r["txn_date"])[:7] != month:
            continue
        if hide_transfers and r["flow"] == "transfer":
            continue
        if categories is not None and r["category"] not in categories:
            continue
        if sources is not None and (r.get("source") or "") not in sources:
            continue
        if needle and needle not in f"{r['merchant']} {r.get('description_raw') or ''}".casefold():
            continue
        kept.append(r)

    kept.sort(key=lambda r: r["txn_id"])
    kept.sort(key=lambda r: iso(r["txn_date"]), reverse=True)   # stable: ties keep id order

    by_flow = {"income": 0, "spend": 0, "transfer": 0}
    for r in kept:
        by_flow[r["flow"]] += r["amount_cents"]
    return {
        "rows": [
            {
                "date": iso(r["txn_date"]),
                "merchant": r["merchant"],
                "category": r["category"],
                "amount": r["amount_cents"] / 100,
                "flow": r["flow"],
                "source": r.get("source") or "",
                "description": r.get("description_raw") or "",
                "txn_id": r["txn_id"],
            }
            for r in kept
        ],
        "count": len(kept),
        "income_cents": by_flow["income"],
        "spend_cents": -by_flow["spend"],
        "transfer_net_cents": by_flow["transfer"],
        "net_cents": sum(by_flow.values()),
    }


def transactions_match_summary(gold: dict, month: str) -> bool | None:
    """Do a month's unfiltered transactions add up to ``monthly_summary``? None if unknown."""
    summary = summary_by_month(gold).get(month)
    if summary is None or "transactions" not in gold:
        return None
    view = transactions_view(gold, month)
    return (view["income_cents"], view["spend_cents"], view["transfer_net_cents"]) == (
        summary["income_cents"], summary["spend_cents"], summary["transfer_net_cents"])


_FORMULA_STARTS = ("=", "+", "-", "@", "\t", "\r")


def _safe_cell(value):
    """Stop spreadsheets from running text that starts like a formula (CSV injection)."""
    return "'" + value if isinstance(value, str) and value.startswith(_FORMULA_STARTS) else value


def transactions_csv(rows: list[dict]) -> str:
    """CSV text for the download button. Text cells are neutralized; amounts stay numbers."""
    import io

    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=TRANSACTION_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for r in rows:
        writer.writerow({
            k: (r[k] if k in ("amount", "flow") else _safe_cell(r[k]))
            for k in TRANSACTION_COLUMNS
        })
    return out.getvalue()


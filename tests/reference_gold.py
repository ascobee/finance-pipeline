"""Independent pure-Python reference ("oracle") for the gold-layer algorithms.

Deliberately written differently from the Spark code (plain loops, ``statistics``) so the two
can disagree. Spark tests compare their output to this on the same data, and it runs without
Spark so the algorithm and thresholds can be sanity-checked anywhere.

Input rows are dicts with: txn_id, txn_date (datetime.date), amount_cents, merchant_norm,
is_transfer, category (or None).
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from datetime import date, timedelta

from finpipe.gold_config import CADENCES, MAD_TO_SIGMA, GoldConfig


def half_up(x: float) -> int:
    """Round half away from zero, like Spark's round(); Python's round() is banker's."""
    return int(math.copysign(math.floor(abs(x) + 0.5), x))


def kind(row) -> str:
    category = row.get("category") or "Uncategorized"
    if row["is_transfer"] or category == "Transfer":
        return "transfer"
    if category == "Income":
        return "income"
    return "spend"


def transactions(rows) -> dict[str, dict]:
    """One record per transaction, keyed by txn_id: the browsing table's logical content."""
    return {
        r["txn_id"]: {
            "txn_date": r["txn_date"], "merchant": r["merchant_norm"],
            "category": r.get("category") or "Uncategorized",
            "amount_cents": r["amount_cents"], "flow": kind(r),
        }
        for r in rows
    }


def month_start(d: date) -> date:
    return d.replace(day=1)


def next_month(d: date) -> date:
    return date(d.year + (d.month == 12), d.month % 12 + 1, 1)


def month_axis(rows) -> list[date]:
    if not rows:
        return []
    lo = month_start(min(r["txn_date"] for r in rows))
    hi = month_start(max(r["txn_date"] for r in rows))
    out = []
    while lo <= hi:
        out.append(lo)
        lo = next_month(lo)
    return out


def monthly_summary(rows) -> dict[date, dict]:
    out = {m: {"income_cents": 0, "spend_cents": 0, "transfer_net_cents": 0, "txn_count": 0}
           for m in month_axis(rows)}
    for r in rows:
        m = out[month_start(r["txn_date"])]
        k = kind(r)
        m["txn_count"] += 1
        if k == "income":
            m["income_cents"] += r["amount_cents"]
        elif k == "spend":
            m["spend_cents"] -= r["amount_cents"]
        else:
            m["transfer_net_cents"] += r["amount_cents"]
    for m in out.values():
        m["net_cents"] = m["income_cents"] - m["spend_cents"]
        m["savings_rate"] = m["net_cents"] / m["income_cents"] if m["income_cents"] > 0 else None
    return out


def monthly_category_spend(rows) -> dict[tuple[date, str], dict]:
    months = month_axis(rows)
    spend = [r for r in rows if kind(r) == "spend"]
    cats = sorted({r.get("category") or "Uncategorized" for r in spend})
    grid = {(m, c): {"spend_cents": 0, "txn_count": 0} for m in months for c in cats}
    for r in spend:
        cell = grid[(month_start(r["txn_date"]), r.get("category") or "Uncategorized")]
        cell["spend_cents"] -= r["amount_cents"]
        cell["txn_count"] += 1
    for (m, c), cell in grid.items():
        prev = grid.get((month_start(m - timedelta(days=1)), c))
        cell["prev_month_spend_cents"] = prev["spend_cents"] if prev else None
        if prev is None:
            cell["mom_change_cents"] = cell["mom_change_pct"] = None
        else:
            cell["mom_change_cents"] = cell["spend_cents"] - prev["spend_cents"]
            cell["mom_change_pct"] = (
                cell["mom_change_cents"] / prev["spend_cents"] if prev["spend_cents"] > 0 else None
            )
    return grid


def _cadence(median_gap: float):
    for name, center, tol in CADENCES:
        if abs(median_gap - center) <= tol:
            return name, center, tol
    return None


def recurring_charges(rows, cfg: GoldConfig = GoldConfig()) -> dict[str, dict]:
    daily: dict[str, dict[date, int]] = defaultdict(lambda: defaultdict(int))
    categories: dict[str, set] = defaultdict(set)
    for r in rows:
        if kind(r) == "transfer":
            continue
        daily[r["merchant_norm"]][r["txn_date"]] += r["amount_cents"]
        categories[r["merchant_norm"]].add(r.get("category") or "Uncategorized")

    out = {}
    for merchant, by_day in daily.items():
        dates = sorted(by_day)
        if len(dates) < cfg.recurring_min_occurrences:
            continue
        gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
        cadence = _cadence(statistics.median(gaps))
        if cadence is None:
            continue
        name, center, tol = cadence
        share = sum(abs(g - center) <= tol for g in gaps) / len(gaps)
        if share < cfg.recurring_min_gap_share:
            continue
        amounts = [by_day[d] for d in dates]
        mean = statistics.fmean(amounts)
        cv = statistics.pstdev(amounts) / abs(mean) if mean else float("inf")
        median_amount = statistics.median(amounts)
        out[merchant] = {
            "cadence": name,
            "occurrences": len(dates),
            "gap_share": share,
            "typical_amount_cents": half_up(median_amount),
            "amount_stable": cv <= cfg.stable_amount_cv,
            "is_income": median_amount > 0,
            "first_date": dates[0],
            "last_date": dates[-1],
            "next_expected_date": dates[-1] + timedelta(days=half_up(statistics.median(gaps))),
            "category": max(categories[merchant]),
        }
    return out


def anomalies(rows, cfg: GoldConfig = GoldConfig()) -> dict[str, dict]:
    charges: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if kind(r) == "spend" and r["amount_cents"] < 0:
            charges[r["merchant_norm"]].append(r)

    out = {}
    for merchant, items in charges.items():
        if len(items) < cfg.anomaly_min_charges:
            continue
        xs = [-r["amount_cents"] for r in items]
        med = statistics.median(xs)
        mad = statistics.median([abs(x - med) for x in xs])
        scale = max(MAD_TO_SIGMA * mad, cfg.anomaly_scale_floor * med)
        for r, x in zip(items, xs):
            z = (x - med) / scale
            if z >= cfg.anomaly_min_z and x >= cfg.anomaly_min_ratio * med \
                    and x - med >= cfg.anomaly_min_excess_cents:
                out[r["txn_id"]] = {
                    "merchant": merchant, "amount_cents": r["amount_cents"],
                    "typical_cents": med, "ratio": x / med, "z_score": z,
                }
    return out

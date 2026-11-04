"""Monthly summary: gold facts in, a short narrative out, with every number verified.

The model is a writer, not a calculator. ``build_facts`` computes every figure in plain code
and pre-formats it ("$4,031.00", "12.5%"); the model only turns that bundle into prose. Then
``verify_numbers`` checks the draft: every number in the text must appear in the facts (whole
dollars and one-decimal roundings are allowed). A draft containing a number that is not in the
facts is rejected and the model is told which numbers were wrong. If it still fails after
``max_attempts``, a deterministic template summary is used instead, so a summary always exists
and is never wrong.

What is sent to the API: the aggregate facts only (totals, category totals, recurring-charge
and anomaly details). No individual transactions. Merchant names appear only for the top
recurring charges and anomalies; ``--no-merchants`` removes them.

    python -m finpipe.summary --month 2026-09
    python -m finpipe.summary --template-only        # no API call at all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Protocol

from .categorize import LLMConfigError
from .dashboard_data import (
    iso, latest_month, money, month_key, month_label, monthly_equivalent_cents, pct,
    previous_month, summary_by_month,
)

log = logging.getLogger(__name__)

PROMPT_VERSION = "1"
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


class SummaryClient(Protocol):
    model: str

    def write_summary(self, facts: dict, feedback: str | None = None) -> str:
        """Return prose for ``facts``; ``feedback`` says why the previous draft was rejected."""


@dataclass
class Summary:
    month: str
    text: str
    source: str            # "llm", "cache" or "template"
    model: str | None
    generated_at: str
    attempts: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- facts


def build_facts(gold: dict, month: str, include_merchants: bool = True) -> dict:
    """Everything the summary may mention, computed and formatted in code."""
    by_month = summary_by_month(gold)
    if month not in by_month:
        raise ValueError(f"no data for {month}; available: {', '.join(sorted(by_month))}")
    cur = by_month[month]
    prev = by_month.get(previous_month(month))

    facts: dict = {
        "month": month,
        "month_label": month_label(month),
        "partial_month": bool(cur.get("is_partial_month")),
        "income": money(cur["income_cents"]),
        "spend": money(cur["spend_cents"]),
        "net": money(cur["net_cents"]),
        "savings_rate": pct(cur["savings_rate"]),
        "previous_month": None,
    }
    if prev is not None:
        delta = cur["spend_cents"] - prev["spend_cents"]
        facts["previous_month"] = {
            "label": month_label(previous_month(month)),
            "spend": money(prev["spend_cents"]),
            "spend_direction": "up" if delta > 0 else "down" if delta < 0 else "unchanged",
            "spend_change": money(abs(delta)),
            "spend_change_pct": pct(abs(delta) / prev["spend_cents"])
            if prev["spend_cents"] > 0 else None,
        }

    cats = sorted(
        (r for r in gold.get("monthly_category_spend", [])
         if month_key(r) == month and r["spend_cents"] > 0),
        key=lambda r: -r["spend_cents"],
    )
    total = cur["spend_cents"]
    facts["top_categories"] = [
        {
            "category": r["category"],
            "spend": money(r["spend_cents"]),
            "share_of_spend": pct(r["spend_cents"] / total) if total > 0 else None,
        }
        for r in cats[:5]
    ]
    changed = sorted(
        (r for r in cats_with_change(gold, month)),
        key=lambda r: -abs(r["mom_change_cents"]),
    )[:3]
    facts["biggest_changes"] = [
        {
            "category": r["category"],
            "direction": "up" if r["mom_change_cents"] > 0 else "down",
            "change": money(abs(r["mom_change_cents"])),
        }
        for r in changed
    ]

    outflows = [r for r in gold.get("recurring_charges", []) if not r["is_income"]]
    monthly_costs = sorted(
        ((r, -monthly_equivalent_cents(r["cadence"], r["typical_amount_cents"])) for r in outflows),
        key=lambda pair: -pair[1],
    )
    recurring: dict = {
        "count": len(outflows),
        "monthly_cost": money(sum(cost for _, cost in monthly_costs)),
    }
    if include_merchants:
        recurring["largest"] = [
            {"merchant": r["merchant"], "cadence": r["cadence"], "monthly_cost": money(cost)}
            for r, cost in monthly_costs[:3]
        ]
    facts["recurring"] = recurring

    odd = [r for r in gold.get("anomalies", []) if iso(r["txn_date"])[:7] == month]
    anomalies: dict = {"count": len(odd)}
    if odd:
        anomalies["items"] = [
            {
                ("merchant" if include_merchants else "category"):
                    r["merchant"] if include_merchants else r["category"],
                "date": iso(r["txn_date"]),
                "amount": money(-r["amount_cents"]),
                "typical": money(r["typical_cents"]),
            }
            for r in sorted(odd, key=lambda r: -r["z_score"])[:3]
        ]
    facts["anomalies"] = anomalies
    return facts


def cats_with_change(gold: dict, month: str):
    return (r for r in gold.get("monthly_category_spend", [])
            if month_key(r) == month and r["mom_change_cents"] not in (None, 0))


# --------------------------------------------------------------------------- verification


def _tokens(text: str) -> list[tuple[str, Decimal]]:
    out = []
    for m in _NUMBER.finditer(text):
        token = m.group().rstrip(",")
        out.append((token, Decimal(token.replace(",", ""))))
    return out


def allowed_numbers(facts: dict) -> set[Decimal]:
    """Every number in the facts, plus whole-dollar and one-decimal roundings of decimals."""
    allowed: set[Decimal] = set()
    for token, value in _tokens(json.dumps(facts)):
        allowed.add(value)
        if "." in token:
            allowed.add(value.quantize(Decimal(1), rounding=ROUND_HALF_UP))
            allowed.add(value.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
    return allowed


def verify_numbers(text: str, facts: dict) -> list[str]:
    """Numbers in ``text`` that are not in ``facts`` (empty list means the text is grounded)."""
    allowed = allowed_numbers(facts)
    bad: list[str] = []
    for token, value in _tokens(text):
        if value not in allowed and token not in bad:
            bad.append(token)
    return bad


# --------------------------------------------------------------------------- template


def template_summary(facts: dict) -> str:
    """Deterministic fallback; uses only numbers that are in the facts."""
    sentences = [
        f"In {facts['month_label']}, income was {facts['income']} and spending was "
        f"{facts['spend']}, a net of {facts['net']}"
        + (f" (savings rate {facts['savings_rate']})." if facts["savings_rate"] else ".")
    ]
    if facts["partial_month"]:
        sentences.append("The data only partly covers this month.")
    prev = facts["previous_month"]
    if prev:
        if prev["spend_direction"] == "unchanged":
            sentences.append(f"Spending was unchanged from {prev['label']}.")
        else:
            rel = f" ({prev['spend_change_pct']})" if prev["spend_change_pct"] else ""
            sentences.append(
                f"Spending was {prev['spend_direction']} {prev['spend_change']}{rel} "
                f"from {prev['label']}."
            )
    if facts["top_categories"]:
        top = facts["top_categories"][0]
        share = f", {top['share_of_spend']} of spending" if top["share_of_spend"] else ""
        sentences.append(f"The largest category was {top['category']} at {top['spend']}{share}.")
    if facts["recurring"]["count"]:
        sentences.append(
            f"Recurring charges cost about {facts['recurring']['monthly_cost']} per month."
        )
    if facts["anomalies"]["count"]:
        n = facts["anomalies"]["count"]
        sentences.append(f"{n} charge{'s' if n != 1 else ''} looked unusually large.")
    return " ".join(sentences)


# --------------------------------------------------------------------------- generation


def facts_key(facts: dict, model: str | None) -> str:
    payload = json.dumps({"facts": facts, "model": model, "v": PROMPT_VERSION}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def generate_summary(
    facts: dict,
    client: SummaryClient | None = None,
    cache_path: str | Path | None = None,
    max_attempts: int = 3,
) -> Summary:
    """Verified LLM summary, cached by a hash of the facts; template fallback when needed."""
    month = facts["month"]
    path = Path(cache_path) if cache_path else None

    def finish(text: str, source: str, model: str | None, attempts: int, key: str | None) -> Summary:
        summary = Summary(month, text, source, model, _now(), attempts)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({**summary.to_dict(), "key": key}, indent=2) + "\n")
        return summary

    if client is None:
        return finish(template_summary(facts), "template", None, 0, None)

    key = facts_key(facts, client.model)
    if path and path.exists():
        saved = json.loads(path.read_text())
        if saved.get("key") == key and saved.get("source") in ("llm", "cache"):
            return Summary(month, saved["text"], "cache", saved.get("model"),
                           saved.get("generated_at", ""), 0)

    feedback: str | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            text = (client.write_summary(facts, feedback) or "").strip()
        except LLMConfigError:
            raise
        except Exception as exc:  # transient API failure: try again
            log.warning("summary attempt %d failed: %s", attempt, exc)
            continue
        if not text:
            feedback = "The draft was empty."
            continue
        bad = verify_numbers(text, facts)
        if bad:
            log.warning("summary attempt %d used numbers not in the facts: %s", attempt, bad)
            feedback = (
                "These numbers are not in the facts: " + ", ".join(bad)
                + ". Use only numbers exactly as written in the facts."
            )
            continue
        return finish(text, "llm", client.model, attempt, key)

    log.warning("no verified LLM summary after %d attempts; using the template", max_attempts)
    return finish(template_summary(facts), "template", None, max_attempts, None)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Write a verified monthly summary from gold tables.")
    p.add_argument("--warehouse", type=Path, default=Path("data/warehouse"))
    p.add_argument("--month", help="YYYY-MM (default: the latest fully covered month)")
    p.add_argument("--no-merchants", action="store_true",
                   help="leave merchant names out of what is sent to the API")
    p.add_argument("--template-only", action="store_true", help="no API call; use the template")
    p.add_argument("--model", default=None)
    p.add_argument("--max-attempts", type=int, default=3)
    args = p.parse_args(argv)

    from .gold_io import read_gold

    gold = read_gold(args.warehouse)
    if not gold.get("monthly_summary"):
        print(f"no gold tables in {args.warehouse}; run the pipeline with --gold first",
              file=sys.stderr)
        return 2
    month = args.month or latest_month(gold)
    facts = build_facts(gold, month, include_merchants=not args.no_merchants)

    client = None
    if not args.template_only:
        from .llm import AnthropicClient

        client = AnthropicClient(model=args.model)
        print("Sending aggregate facts (no individual transactions) to the Anthropic API"
              + (" without merchant names." if args.no_merchants else "."), file=sys.stderr)

    path = args.warehouse / "summaries" / f"{month}.json"
    summary = generate_summary(facts, client, path, args.max_attempts)
    print(f"{month_label(month)} ({summary.source}"
          + (f", {summary.model}" if summary.model else "") + ")\n")
    print(summary.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

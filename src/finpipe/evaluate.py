"""Measure categorizer accuracy against a labeled golden set.

The golden set is a CSV of ``merchant,category`` where ``merchant`` is the *normalized* name
(what ``normalize_merchant`` produces and what the categorizer receives).

The evaluation deliberately runs the model with **no corrections and no rules**, so the number
reflects the model alone. It uses its own cache file so a re-run is free, and so evaluating
never pollutes the cache the pipeline uses.

    # synthetic golden set (28 merchants), derived from the generator's ground truth
    python -m finpipe.evaluate build-golden --truth data/sample/truth/merchants.csv \\
        --out data/golden/synthetic_merchants.csv

    # score the model (needs ANTHROPIC_API_KEY)
    python -m finpipe.evaluate run --golden data/golden/synthetic_merchants.csv

For a meaningful number, label ~100 of your *real* merchants by hand and keep that file under
``data/private/`` (git-ignored). Synthetic merchants are all well-known brands, so they are an
easy test; the real long tail is where the model gets things wrong.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from .categories import CATEGORY_VALUES
from .categorize import Categorizer, MerchantCategory, SOURCE_FALLBACK
from .normalize import normalize_merchant


def load_golden(path: str | Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with Path(path).open(newline="") as fh:
        for line_no, row in enumerate(csv.DictReader(fh), start=2):
            merchant, category = row["merchant"].strip(), row["category"].strip()
            if category not in CATEGORY_VALUES:
                raise ValueError(f"{path}:{line_no}: unknown category {category!r}")
            if merchant in out and out[merchant] != category:
                raise ValueError(f"{path}:{line_no}: conflicting labels for {merchant!r}")
            out[merchant] = category
    return out


def golden_from_truth(truth_csv: str | Path) -> dict[str, str]:
    """Collapse the generator's raw-description truth to normalized-merchant labels."""
    out: dict[str, str] = {}
    with Path(truth_csv).open(newline="") as fh:
        for row in csv.DictReader(fh):
            merchant = normalize_merchant(row["description_raw"])
            category = row["category"]
            if out.setdefault(merchant, category) != category:
                raise ValueError(f"{merchant!r} maps to more than one category in the truth file")
    return out


def write_golden(golden: Mapping[str, str], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["merchant", "category"])
        for merchant in sorted(golden):
            w.writerow([merchant, golden[merchant]])


@dataclass
class EvalReport:
    total: int
    correct: int
    errors: list[dict] = field(default_factory=list)
    per_category: dict[str, tuple[int, int]] = field(default_factory=dict)  # (correct, support)
    flagged: int = 0
    correct_unflagged: int = 0
    unflagged: int = 0
    fallbacks: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def unflagged_accuracy(self) -> float:
        return self.correct_unflagged / self.unflagged if self.unflagged else 0.0


def evaluate(golden: Mapping[str, str], results: Sequence[MerchantCategory]) -> EvalReport:
    by_merchant = {r.merchant: r for r in results}
    support: Counter[str] = Counter()
    hits: Counter[str] = Counter()
    report = EvalReport(total=0, correct=0)
    for merchant, expected in golden.items():
        got = by_merchant.get(merchant)
        if got is None:
            raise KeyError(f"no prediction for golden merchant {merchant!r}")
        report.total += 1
        support[expected] += 1
        ok = got.category == expected
        if ok:
            report.correct += 1
            hits[expected] += 1
        else:
            report.errors.append({
                "merchant": merchant, "expected": expected,
                "predicted": got.category, "confidence": got.confidence,
            })
        if got.needs_review:
            report.flagged += 1
        else:
            report.unflagged += 1
            report.correct_unflagged += int(ok)
        report.fallbacks += int(got.source == SOURCE_FALLBACK)
    report.per_category = {c: (hits[c], support[c]) for c in sorted(support)}
    return report


def format_report(report: EvalReport) -> str:
    lines = [
        f"accuracy: {report.correct}/{report.total} = {report.accuracy:.1%}",
        f"flagged for review: {report.flagged}  "
        f"(accuracy on the unflagged rest: {report.unflagged_accuracy:.1%})",
        f"fallbacks (LLM gave no usable answer): {report.fallbacks}",
        "",
        "per category (correct/support):",
    ]
    lines += [f"  {c:<15} {h}/{s}" for c, (h, s) in report.per_category.items()]
    if report.errors:
        lines += ["", "errors:"]
        for e in sorted(report.errors, key=lambda e: -e["confidence"]):
            lines.append(
                f"  {e['merchant']!r}: expected {e['expected']}, got {e['predicted']} "
                f"(confidence {e['confidence']:.2f})"
            )
    return "\n".join(lines)


def write_errors(report: EvalReport, path: str | Path) -> None:
    with Path(path).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["merchant", "expected", "predicted", "confidence"])
        w.writeheader()
        w.writerows(report.errors)


def run_eval(golden: Mapping[str, str], client, cache_path: str | Path | None) -> EvalReport:
    categorizer = Categorizer(client, cache_path=cache_path)  # no corrections, no rules
    return evaluate(golden, categorizer.categorize(sorted(golden)))


def main() -> None:
    p = argparse.ArgumentParser(description="Categorizer accuracy evaluation.")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-golden", help="derive a golden set from synthetic ground truth")
    b.add_argument("--truth", type=Path, default=Path("data/sample/truth/merchants.csv"))
    b.add_argument("--out", type=Path, default=Path("data/golden/synthetic_merchants.csv"))

    r = sub.add_parser("run", help="score the model against a golden set (calls the API)")
    r.add_argument("--golden", type=Path, default=Path("data/golden/synthetic_merchants.csv"))
    r.add_argument("--cache", type=Path, default=Path("data/warehouse/categorization/eval_cache.jsonl"))
    r.add_argument("--model", default=None)
    r.add_argument("--errors-out", type=Path, default=None)

    args = p.parse_args()
    if args.cmd == "build-golden":
        golden = golden_from_truth(args.truth)
        write_golden(golden, args.out)
        print(f"wrote {len(golden)} labeled merchants to {args.out}")
        return

    from .llm import AnthropicClient

    client = AnthropicClient(model=args.model)
    report = run_eval(load_golden(args.golden), client, args.cache)
    print(f"model: {client.model}\n")
    print(format_report(report))
    print(f"\nAPI usage this run: {client.calls} calls, "
          f"{client.input_tokens} input tokens, {client.output_tokens} output tokens")
    if args.errors_out:
        write_errors(report, args.errors_out)


if __name__ == "__main__":
    main()

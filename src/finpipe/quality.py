"""Data quality checks: does the pipeline's output still agree with its inputs?

The checks are split in two so the rules are testable without Spark:

1. ``scan_raw`` (pure Python) reads the original CSV exports and ``collect_metrics`` (Spark)
   reads the warehouse. Both produce plain dicts of facts.
2. ``evaluate`` (pure Python) turns those facts into pass / fail / skip results.

What is checked, and why these checks rather than generic null counts:

source
    * every raw file made it into bronze, with the same number of rows (silent row loss);
    * running-balance continuity in files that carry a balance column: each row's balance must
      equal the previous balance plus the row's amount. This is the strongest check available,
      because it independently verifies the amount parsing, the sign convention, and that no
      row went missing, using only the bank's own numbers;
    * in card exports with separate debit and credit columns, every row has exactly one side
      filled in (a warning; such rows are quarantined by silver rather than guessed at).
silver
    * rows are accounted for (bronze = silver + quarantined + duplicates collapsed);
    * ``txn_id`` is unique, required fields are present, quarantine stays small;
    * dates are plausible (a swapped day/month format shows up as future dates), amounts are
      plausible, every expected source is present, and the data is fresh.
categorization
    * every merchant has a category; fallbacks and the review queue stay small.
gold
    * income - spend + transfers equals the silver total (nothing double counted or lost), and
      the two spend tables agree with each other; the transactions table has silver's row
      count and the same income, spend and transfer totals as the monthly summary.

Each failed check has a severity. ``error`` failures make the run (and the CLI exit code) fail
so an orchestrator can stop; ``warn`` failures are reported but do not fail the run. A check
that cannot run (raw files not provided, a layer not built yet) is ``skip``, never a silent pass.

    python -m finpipe.quality --raw data/sample --warehouse data/warehouse
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import unquote

# Mirrors ingest.SOURCES (kept here so this module's pure parts import without Spark).
SOURCES = ("checking", "credit_card")
REQUIRED_SILVER_COLUMNS = (
    "txn_id", "source", "txn_date", "amount_cents", "description_raw", "merchant_norm",
)

PASS, FAIL, SKIP = "pass", "fail", "skip"
ERROR, WARN = "error", "warn"


@dataclass(frozen=True)
class QualityConfig:
    max_quarantine_rate: float = 0.01          # more than 1% unparseable rows is an error
    min_plausible_date: date = date(2000, 1, 1)
    stale_after_days: int = 45                 # newest transaction older than this: warn
    max_abs_amount_cents: int = 50_000_000     # $500,000 in a single transaction: warn
    max_review_queue: int = 25                 # merchants awaiting human review: warn
    check_running_balance: bool = True


@dataclass
class CheckResult:
    name: str
    layer: str
    severity: str
    status: str
    observed: Any = None
    expected: Any = None
    detail: str = ""

    @property
    def failed_error(self) -> bool:
        return self.status == FAIL and self.severity == ERROR


@dataclass
class QualityReport:
    results: list[CheckResult] = field(default_factory=list)
    generated_at: str = ""

    @property
    def errors(self) -> list[CheckResult]:
        return [r for r in self.results if r.failed_error]

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == FAIL and r.severity == WARN]

    @property
    def ok(self) -> bool:
        return not self.errors

    def counts(self) -> dict[str, int]:
        return {
            "passed": sum(r.status == PASS for r in self.results),
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "skipped": sum(r.status == SKIP for r in self.results),
        }

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "ok": self.ok,
            "counts": self.counts(),
            "checks": [asdict(r) for r in self.results],
        }

    def format(self) -> str:
        c = self.counts()
        lines = [
            f"data quality: {'OK' if self.ok else 'FAILED'}  "
            f"({c['passed']} passed, {c['errors']} errors, {c['warnings']} warnings, "
            f"{c['skipped']} skipped)"
        ]
        for r in self.results:
            if r.status == FAIL:
                tag = "ERROR" if r.severity == ERROR else "WARN "
                lines.append(f"  [{tag}] {r.layer}/{r.name}: {r.detail}")
            elif r.status == SKIP:
                lines.append(f"  [skip ] {r.layer}/{r.name}: {r.detail}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- raw scan (pure)


def _snake(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()


def _cents(text: str) -> int | None:
    try:
        return int((Decimal(re.sub(r"[$,\s]", "", text)) * 100).to_integral_value())
    except (InvalidOperation, ValueError):
        return None


def _parse_date(text: str) -> date | None:
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            continue
    return None


def balance_breaks(path: Path) -> dict | None:
    """Running-balance check for one export.

    Returns None when the file has no balance column. Otherwise ``{"breaks", "checked",
    "examples"}``. Works for ascending and descending exports (decided from the first and last
    dates); same-day order is whatever order the bank wrote, which is what balances follow.
    """
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return None
        cols = {_snake(h): i for i, h in enumerate(header)}
        if not {"date", "amount", "balance"} <= set(cols):
            return None
        rows = []
        for raw in reader:
            if not raw:
                continue
            d = _parse_date(raw[cols["date"]])
            a, b = _cents(raw[cols["amount"]]), _cents(raw[cols["balance"]])
            if d is None or a is None or b is None:
                return {"breaks": 1, "checked": 0,
                        "examples": [f"unparseable row: {raw!r}"[:120]]}
            rows.append((d, a, b))

    if len(rows) >= 2 and rows[0][0] > rows[-1][0]:
        rows.reverse()
    breaks, examples = 0, []
    for prev, cur in zip(rows, rows[1:]):
        delta = cur[2] - prev[2]
        if delta != cur[1]:
            breaks += 1
            if len(examples) < 3:
                examples.append(f"{cur[0]}: amount {cur[1] / 100:.2f} but balance moved "
                                f"{delta / 100:.2f}")
    return {"breaks": breaks, "checked": max(len(rows) - 1, 0), "examples": examples}


def card_sides(path: Path) -> dict | None:
    """For a debit/credit card export, count rows by which side is filled in.

    Returns None for files without both ``debit`` and ``credit`` columns. A well-formed row has
    exactly one side filled in with a non-negative amount; anything else is counted under
    ``bad`` (both sides, neither side, an unreadable number, or a negative one), and the
    pipeline quarantines those rows.
    """
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        cols = {_snake(h): i for i, h in enumerate(header or [])}
        if not {"debit", "credit"} <= set(cols):
            return None
        out = {"debit_only": 0, "credit_only": 0, "bad": 0, "examples": []}
        for n, raw in enumerate(reader, start=2):
            if not raw:
                continue
            cell = {k: (raw[cols[k]].strip() if cols[k] < len(raw) else "")
                    for k in ("debit", "credit")}
            filled = [k for k, v in cell.items() if v]
            amount = _cents(cell[filled[0]]) if len(filled) == 1 else None
            if len(filled) == 1 and amount is not None and amount >= 0:
                out[f"{filled[0]}_only"] += 1
            else:
                out["bad"] += 1
                if len(out["examples"]) < 3:
                    why = ("both sides filled" if len(filled) == 2 else "neither side filled"
                           if not filled else "unreadable or negative amount")
                    out["examples"].append(f"line {n}: {why}")
    return out


def scan_raw(raw_dir: str | Path, check_balance: bool = True) -> dict:
    """Facts about the original exports: data rows per file and balance continuity."""
    files: dict[str, dict] = {}
    for source in SOURCES:
        for path in sorted((Path(raw_dir) / source).glob("*.csv")):
            with path.open(newline="") as fh:
                rows = sum(1 for r in list(csv.reader(fh))[1:] if r)
            entry: dict = {"rows": rows, "balance": None, "sides": card_sides(path)}
            if check_balance:
                entry["balance"] = balance_breaks(path)
            files[f"{source}/{path.name}"] = entry
    return {"files": files}


# --------------------------------------------------------------------------- warehouse metrics


def collect_metrics(warehouse_dir: str | Path) -> dict:
    """Facts about the warehouse, computed with a handful of Spark aggregations."""
    from pyspark.sql import functions as F

    from .spark_session import get_spark

    spark = get_spark()
    wh = Path(warehouse_dir)
    silver_path = wh / "silver" / "transactions"
    if not silver_path.exists():
        raise FileNotFoundError(f"no silver table at {silver_path}; run the pipeline first")

    bronze_by_file: dict[str, int] = {}
    for source in SOURCES:
        path = wh / "bronze" / source
        if path.exists():
            counts = spark.read.parquet(str(path)).groupBy("_source_file").count().collect()
            for r in counts:
                name = unquote(r["_source_file"]).rsplit("/", 1)[-1]
                bronze_by_file[f"{source}/{name}"] = r["count"]

    silver = spark.read.parquet(str(silver_path))
    nulls = [F.sum(F.col(c).isNull().cast("int")).alias(f"null_{c}")
             for c in REQUIRED_SILVER_COLUMNS]
    agg = silver.agg(
        F.count(F.lit(1)).alias("rows"),
        F.countDistinct("txn_id").alias("distinct_ids"),
        F.min("txn_date").alias("min_date"),
        F.max("txn_date").alias("max_date"),
        F.sum((F.col("amount_cents") == 0).cast("int")).alias("zero_amount_rows"),
        F.max(F.abs(F.col("amount_cents"))).alias("max_abs_amount_cents"),
        F.sum("amount_cents").alias("total_cents"),
        *nulls,
    ).collect()[0]
    by_source = {
        r["source"]: {"rows": r["rows"], "max_date": r["max_date"].isoformat()}
        for r in silver.groupBy("source")
        .agg(F.count(F.lit(1)).alias("rows"), F.max("txn_date").alias("max_date"))
        .collect()
    }

    quarantine_path = wh / "quarantine" / "transactions"
    metrics: dict = {
        "bronze_rows_by_file": bronze_by_file,
        "silver": {
            "rows": agg["rows"],
            "distinct_ids": agg["distinct_ids"],
            "min_date": agg["min_date"].isoformat() if agg["min_date"] else None,
            "max_date": agg["max_date"].isoformat() if agg["max_date"] else None,
            "zero_amount_rows": agg["zero_amount_rows"] or 0,
            "max_abs_amount_cents": agg["max_abs_amount_cents"] or 0,
            "total_cents": agg["total_cents"] or 0,
            "nulls": {c: agg[f"null_{c}"] or 0 for c in REQUIRED_SILVER_COLUMNS},
            "by_source": by_source,
        },
        "quarantine_rows": spark.read.parquet(str(quarantine_path)).count()
        if quarantine_path.exists() else 0,
        "categorization": None,
        "gold": None,
    }

    cats_path = wh / "categorization" / "merchant_categories"
    if cats_path.exists():
        cats = spark.read.parquet(str(cats_path))
        merchants = silver.select("merchant_norm").where("merchant_norm IS NOT NULL").distinct()
        metrics["categorization"] = {
            "merchants": merchants.count(),
            "missing": merchants.join(
                cats.select(F.col("merchant").alias("merchant_norm")), "merchant_norm", "left_anti"
            ).count(),
            "fallbacks": cats.where(F.col("category_source") == "fallback").count(),
            "review_queue": cats.where("needs_review").count(),
        }

    summary_path = wh / "gold" / "monthly_summary"
    category_path = wh / "gold" / "monthly_category_spend"
    if summary_path.exists() and category_path.exists():
        s = spark.read.parquet(str(summary_path)).agg(
            F.sum("income_cents").alias("income"), F.sum("spend_cents").alias("spend"),
            F.sum("transfer_net_cents").alias("transfer"),
        ).collect()[0]
        c = (spark.read.parquet(str(category_path))
             .agg(F.sum("spend_cents").alias("spend")).collect()[0])
        metrics["gold"] = {
            "income_cents": s["income"] or 0, "spend_cents": s["spend"] or 0,
            "transfer_net_cents": s["transfer"] or 0, "category_spend_cents": c["spend"] or 0,
            "transactions": None,
        }
        tx_path = wh / "gold" / "transactions"
        if tx_path.exists():   # absent in warehouses built before this table existed
            tx = spark.read.parquet(str(tx_path))

            def flow_sum(flow):
                return F.sum(F.when(F.col("flow") == flow, F.col("amount_cents")).otherwise(0))

            t = tx.agg(
                F.count(F.lit(1)).alias("rows"), flow_sum("income").alias("income"),
                flow_sum("spend").alias("spend"), flow_sum("transfer").alias("transfer"),
            ).collect()[0]
            metrics["gold"]["transactions"] = {
                "rows": t["rows"], "income_cents": t["income"] or 0,
                "spend_cents": -(t["spend"] or 0), "transfer_net_cents": t["transfer"] or 0,
            }
    return metrics


# --------------------------------------------------------------------------- rules (pure)


def _result(name, layer, severity, ok, observed=None, expected=None, detail="") -> CheckResult:
    return CheckResult(name, layer, severity, PASS if ok else FAIL, observed, expected, detail)


def _skip(name, layer, severity, why) -> CheckResult:
    return CheckResult(name, layer, severity, SKIP, detail=why)


def evaluate(
    metrics: dict,
    raw: dict | None = None,
    cfg: QualityConfig = QualityConfig(),
    today: date | None = None,
) -> list[CheckResult]:
    today = today or datetime.now(timezone.utc).date()
    out: list[CheckResult] = []

    # ---- source: raw files vs bronze
    if raw is None:
        for name in ("raw_files_ingested", "raw_bronze_row_counts", "running_balance_continuity",
                     "card_rows_one_sided"):
            out.append(_skip(name, "source", ERROR, "raw directory not provided"))
    else:
        bronze = metrics["bronze_rows_by_file"]
        files = raw["files"]
        missing = sorted(f for f in files if f not in bronze)
        out.append(_result(
            "raw_files_ingested", "source", ERROR, not missing, len(missing), 0,
            f"{len(missing)} raw file(s) not in bronze: {', '.join(missing[:5])}"
            if missing else f"all {len(files)} raw files are in bronze",
        ))
        mismatched = sorted(
            f"{f} (raw {info['rows']}, bronze {bronze[f]})"
            for f, info in files.items() if f in bronze and bronze[f] != info["rows"]
        )
        out.append(_result(
            "raw_bronze_row_counts", "source", ERROR, not mismatched, len(mismatched), 0,
            "row count differs for: " + "; ".join(mismatched[:5]) if mismatched
            else "every file has the same row count in raw and bronze",
        ))
        if not cfg.check_running_balance:
            out.append(_skip("running_balance_continuity", "source", ERROR, "disabled in config"))
        else:
            with_balance = {f: i["balance"] for f, i in files.items() if i.get("balance")}
            if not with_balance:
                out.append(_skip("running_balance_continuity", "source", ERROR,
                                 "no export has a balance column"))
            else:
                bad = {f: b for f, b in with_balance.items() if b["breaks"] > 0}
                detail = "; ".join(f"{f}: {b['breaks']} break(s), e.g. {b['examples'][0]}"
                                   for f, b in list(bad.items())[:3]) if bad else (
                    f"balance continuous across {sum(b['checked'] for b in with_balance.values())}"
                    f" row pairs in {len(with_balance)} file(s)")
                out.append(_result("running_balance_continuity", "source", ERROR, not bad,
                                   sum(b["breaks"] for b in bad.values()), 0, detail))

        split = {f: i["sides"] for f, i in files.items() if i.get("sides")}
        if not split:
            out.append(_skip("card_rows_one_sided", "source", WARN,
                             "no export has debit/credit columns"))
        else:
            bad = {f: x for f, x in split.items() if x["bad"] > 0}
            out.append(_result(
                "card_rows_one_sided", "source", WARN, not bad,
                sum(x["bad"] for x in bad.values()), 0,
                "; ".join(f"{f}: {x['bad']} row(s), e.g. {x['examples'][0]}"
                          for f, x in list(bad.items())[:3]) if bad else
                f"every debit/credit row has exactly one side filled "
                f"({sum(x['debit_only'] + x['credit_only'] for x in split.values())} rows "
                f"in {len(split)} file(s))",
            ))

    # ---- silver
    s = metrics["silver"]
    bronze_total = sum(metrics["bronze_rows_by_file"].values())
    collapsed = bronze_total - metrics["quarantine_rows"] - s["rows"]
    out.append(_result(
        "rows_accounted_for", "silver", ERROR, collapsed >= 0, collapsed, ">= 0",
        f"bronze {bronze_total} = silver {s['rows']} + quarantined {metrics['quarantine_rows']}"
        f" + duplicates collapsed {collapsed}" if collapsed >= 0 else
        f"silver ({s['rows']}) + quarantine ({metrics['quarantine_rows']}) exceed bronze "
        f"({bronze_total}): rows appeared from nowhere",
    ))
    out.append(_result(
        "txn_id_unique", "silver", ERROR, s["rows"] == s["distinct_ids"],
        s["rows"] - s["distinct_ids"], 0,
        f"{s['rows'] - s['distinct_ids']} duplicate txn_id value(s)"
        if s["rows"] != s["distinct_ids"] else f"{s['rows']} rows, all ids distinct",
    ))
    null_cols = {c: n for c, n in s["nulls"].items() if n}
    out.append(_result(
        "required_fields_present", "silver", ERROR, not null_cols, sum(null_cols.values()), 0,
        "nulls in " + ", ".join(f"{c} ({n})" for c, n in null_cols.items()) if null_cols
        else "no nulls in required columns",
    ))

    total_in = s["rows"] + metrics["quarantine_rows"]
    rate = metrics["quarantine_rows"] / total_in if total_in else 0.0
    out.append(_result(
        "quarantine_rate", "silver", ERROR, rate <= cfg.max_quarantine_rate, round(rate, 4),
        f"<= {cfg.max_quarantine_rate}",
        f"{metrics['quarantine_rows']} of {total_in} rows quarantined ({rate:.2%})",
    ))
    out.append(_result(
        "quarantine_empty", "silver", WARN, metrics["quarantine_rows"] == 0,
        metrics["quarantine_rows"], 0,
        f"{metrics['quarantine_rows']} unparseable row(s) in quarantine/ need a look"
        if metrics["quarantine_rows"] else "nothing quarantined",
    ))

    if s["rows"] == 0:
        out.append(_skip("dates_plausible", "silver", ERROR, "silver is empty"))
    else:
        lo, hi = date.fromisoformat(s["min_date"]), date.fromisoformat(s["max_date"])
        ok = lo >= cfg.min_plausible_date and hi <= today + timedelta(days=1)
        out.append(_result(
            "dates_plausible", "silver", ERROR, ok, f"{lo}..{hi}",
            f"{cfg.min_plausible_date}..{today}",
            f"dates {lo}..{hi}" + ("" if ok else " fall outside the plausible range; a day/month "
                                   "format mix-up produces dates like this"),
        ))

    out.append(_result(
        "amounts_plausible", "silver", WARN, s["max_abs_amount_cents"] <= cfg.max_abs_amount_cents,
        s["max_abs_amount_cents"], f"<= {cfg.max_abs_amount_cents}",
        f"largest amount is ${s['max_abs_amount_cents'] / 100:,.2f}",
    ))
    out.append(_result(
        "no_zero_amounts", "silver", WARN, s["zero_amount_rows"] == 0, s["zero_amount_rows"], 0,
        f"{s['zero_amount_rows']} zero-amount transaction(s)" if s["zero_amount_rows"]
        else "no zero-amount transactions",
    ))
    absent = [src for src in SOURCES if src not in s["by_source"]]
    out.append(_result(
        "sources_present", "silver", WARN, not absent, len(absent), 0,
        f"no rows for source(s): {', '.join(absent)}" if absent else "every source has rows",
    ))
    stale = {
        src: (today - date.fromisoformat(v["max_date"])).days
        for src, v in s["by_source"].items()
        if (today - date.fromisoformat(v["max_date"])).days > cfg.stale_after_days
    }
    out.append(_result(
        "data_freshness", "silver", WARN, not stale, max(stale.values(), default=0),
        f"<= {cfg.stale_after_days} days",
        "newest transaction is old for: " + ", ".join(f"{k} ({v} days)" for k, v in stale.items())
        if stale else "every source has a recent transaction",
    ))

    # ---- categorization
    cat = metrics.get("categorization")
    if cat is None:
        for name in ("merchants_categorized", "no_fallback_categories", "review_queue_size"):
            out.append(_skip(name, "categorization", ERROR, "categorization has not run"))
    else:
        out.append(_result(
            "merchants_categorized", "categorization", ERROR, cat["missing"] == 0, cat["missing"],
            0, f"{cat['missing']} of {cat['merchants']} merchants have no category"
            if cat["missing"] else f"all {cat['merchants']} merchants have a category",
        ))
        out.append(_result(
            "no_fallback_categories", "categorization", WARN, cat["fallbacks"] == 0,
            cat["fallbacks"], 0,
            f"{cat['fallbacks']} merchant(s) fell back to Other (the LLM gave no usable answer)"
            if cat["fallbacks"] else "no fallback categories",
        ))
        out.append(_result(
            "review_queue_size", "categorization", WARN, cat["review_queue"] <= cfg.max_review_queue,
            cat["review_queue"], f"<= {cfg.max_review_queue}",
            f"{cat['review_queue']} merchant(s) awaiting review",
        ))

    # ---- gold
    g = metrics.get("gold")
    if g is None:
        for name in ("gold_reconciles_to_silver", "category_spend_matches_summary",
                     "gold_transactions_match"):
            out.append(_skip(name, "gold", ERROR, "gold has not been built"))
    else:
        recon = g["income_cents"] - g["spend_cents"] + g["transfer_net_cents"]
        out.append(_result(
            "gold_reconciles_to_silver", "gold", ERROR, recon == s["total_cents"], recon,
            s["total_cents"],
            "income - spend + transfers equals the silver total" if recon == s["total_cents"]
            else f"gold nets to {recon} cents but silver totals {s['total_cents']} "
                 f"(difference {recon - s['total_cents']})",
        ))
        out.append(_result(
            "category_spend_matches_summary", "gold", ERROR,
            g["category_spend_cents"] == g["spend_cents"], g["category_spend_cents"],
            g["spend_cents"],
            "category spend equals summary spend" if g["category_spend_cents"] == g["spend_cents"]
            else f"category table {g['category_spend_cents']} vs summary {g['spend_cents']}",
        ))
        tx = g.get("transactions")
        if tx is None:
            out.append(_skip("gold_transactions_match", "gold", ERROR,
                             "gold transactions table not built; rerun the pipeline with --gold"))
        else:
            diffs = [
                f"{label}: transactions {tx[key]} vs {expected}"
                for label, key, expected in (
                    ("rows", "rows", s["rows"]), ("income", "income_cents", g["income_cents"]),
                    ("spend", "spend_cents", g["spend_cents"]),
                    ("transfers", "transfer_net_cents", g["transfer_net_cents"]))
                if tx[key] != expected
            ]
            out.append(_result(
                "gold_transactions_match", "gold", ERROR, not diffs, len(diffs), 0,
                "; ".join(diffs) if diffs else
                f"{tx['rows']} transactions; income, spend and transfers equal the summary "
                f"and the row count equals silver",
            ))
    return out


# --------------------------------------------------------------------------- entry points


def run_checks(
    warehouse_dir: str | Path,
    raw_dir: str | Path | None = None,
    cfg: QualityConfig = QualityConfig(),
    today: date | None = None,
    write: bool = True,
) -> QualityReport:
    metrics = collect_metrics(warehouse_dir)
    raw = scan_raw(raw_dir, cfg.check_running_balance) if raw_dir else None
    report = QualityReport(
        evaluate(metrics, raw, cfg, today),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    if write:
        out = Path(warehouse_dir) / "quality" / "report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), indent=2, default=str) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run data quality checks; exit 1 on any error.")
    p.add_argument("--raw", type=Path, default=Path("data/sample"))
    p.add_argument("--warehouse", type=Path, default=Path("data/warehouse"))
    p.add_argument("--no-raw", action="store_true", help="skip checks that need the raw CSVs")
    args = p.parse_args(argv)
    report = run_checks(args.warehouse, None if args.no_raw else args.raw)
    print(report.format())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())

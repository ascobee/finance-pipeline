"""Gold layer: analysis tables built from categorized silver.

Everything here is a pure ``DataFrame -> DataFrame`` function over the *enriched* silver frame
(silver joined to merchant categories, see ``enrich.with_categories``), so each table is easy
to test on a handful of hand-built rows. Gold keeps no state: it is rebuilt from silver on every
run, which keeps reruns safe.

Cash-flow classification (``flow``), applied to every transaction:

* ``transfer``  ``is_transfer`` or category ``Transfer``. Moves between your own accounts and
                card payments. Excluded from spend and income so nothing is counted twice, but
                reported as ``transfer_net_cents`` so totals still reconcile to silver.
* ``income``    category ``Income``.
* ``spend``     everything else. **Refunds net against spend**: a returned item is a positive
                amount in the same category, so it reduces that category's spend.

Other conventions:

* Integer cents everywhere. Months are calendar months by ``txn_date``.
* A merchant with no category is reported as ``Uncategorized`` rather than dropped.
* The first and last months are flagged ``is_partial_month`` when the data does not cover them
  fully (the end is inferred from the last transaction date, so it is a heuristic).
* Month axes are densified: a month or category with no activity shows up as zero, so the
  month-over-month change always compares adjacent calendar months.

Tables
------
``monthly_summary``         income, spend, net, savings rate, transfer net, per month
``monthly_category_spend``  month x category spend with month-over-month change
``recurring_charges``       merchants charging on a regular cadence (weekly ... yearly)
``anomalies``               charges far above that merchant's norm
``transactions``            every transaction with its merchant, category and flow (the
                            dashboard's transaction browser); nets to the other tables

Anomaly scoring uses the median and median absolute deviation (MAD) per merchant rather than
mean and standard deviation: one huge charge inflates the mean and the standard deviation and
can hide itself, while the median and MAD barely move. Thresholds are deliberately
conservative (see ``gold_config.GoldConfig``).

``tests/reference_gold.py`` re-implements these algorithms in plain Python as an independent
oracle; the Spark tests check that both agree.
"""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from .gold_config import CADENCES, MAD_TO_SIGMA, GoldConfig
from .spark_session import get_spark

TABLES = (
    "monthly_summary", "monthly_category_spend", "recurring_charges", "anomalies", "transactions",
)


def _with_flow(enriched: DataFrame) -> DataFrame:
    """Fill missing categories and add ``flow`` (transfer / income / spend)."""
    df = enriched
    if "category" not in df.columns:
        df = df.withColumn("category", F.lit(None).cast("string"))
    df = df.withColumn("category", F.coalesce(F.col("category"), F.lit("Uncategorized")))
    is_transfer = F.coalesce(F.col("is_transfer"), F.lit(False)) | (F.col("category") == "Transfer")
    flow = (
        F.when(is_transfer, "transfer")
        .when(F.col("category") == "Income", "income")
        .otherwise("spend")
    )
    return df.withColumn("flow", flow)


def prepare(enriched: DataFrame) -> DataFrame:
    """Add ``flow`` and ``month``; fill missing categories. Works with or without categories."""
    return _with_flow(enriched).select(
        "txn_id", "txn_date", "amount_cents", "merchant_norm", "category", "flow",
        F.trunc("txn_date", "month").alias("month"),
    )


def transactions(enriched: DataFrame) -> DataFrame:
    """One row per transaction for browsing: what it was, how it is classed, where it came from.

    ``merchant`` is the cleaned name and ``description_raw`` the bank's original text (useful
    when a merchant looks wrong). Amounts keep silver's sign (negative = money out). Because it
    uses the same ``flow`` logic as the other tables, its per-flow sums equal the monthly summary.
    Newest first, with ``txn_id`` as the tie-breaker so reruns write identical files.
    """
    df = _with_flow(enriched)
    for optional in ("source", "description_raw"):   # absent in hand-built test frames
        if optional not in df.columns:
            df = df.withColumn(optional, F.lit(None).cast("string"))
    return df.select(
        "txn_id", "txn_date",
        F.col("merchant_norm").alias("merchant"),
        "category", "amount_cents", "flow", "source", "description_raw",
    ).orderBy(F.col("txn_date").desc(), "txn_id")


def _month_axis(df: DataFrame) -> DataFrame:
    """One row per calendar month between the first and last transaction (inclusive)."""
    bounds = df.agg(F.min("txn_date").alias("min_date"), F.max("txn_date").alias("max_date"))
    return bounds.select(
        "min_date",
        "max_date",
        F.explode(
            F.sequence(
                F.trunc("min_date", "month"), F.trunc("max_date", "month"),
                F.expr("interval 1 month"),
            )
        ).alias("month"),
    )


def _amount_where(flow: str) -> Column:
    return F.sum(F.when(F.col("flow") == flow, F.col("amount_cents")).otherwise(0))


def monthly_summary(df: DataFrame) -> DataFrame:
    axis = _month_axis(df)
    agg = df.groupBy("month").agg(
        _amount_where("income").alias("income_cents"),
        (-_amount_where("spend")).alias("spend_cents"),
        _amount_where("transfer").alias("transfer_net_cents"),
        F.count(F.lit(1)).alias("txn_count"),
    )
    out = axis.join(agg, "month", "left").fillna(
        0, subset=["income_cents", "spend_cents", "transfer_net_cents", "txn_count"]
    )
    first_month, last_month = F.trunc("min_date", "month"), F.trunc("max_date", "month")
    partial = ((F.col("month") == first_month) & (F.col("min_date") > F.col("month"))) | (
        (F.col("month") == last_month) & (F.col("max_date") < F.last_day(F.col("month")))
    )
    return (
        out.withColumn("net_cents", F.col("income_cents") - F.col("spend_cents"))
        .withColumn(
            "savings_rate",
            F.when(F.col("income_cents") > 0, F.col("net_cents") / F.col("income_cents")),
        )
        .withColumn("is_partial_month", partial)
        .select(
            "month", "income_cents", "spend_cents", "net_cents", "savings_rate",
            "transfer_net_cents", "txn_count", "is_partial_month",
        )
        .orderBy("month")
    )


def monthly_category_spend(df: DataFrame) -> DataFrame:
    spend = df.filter(F.col("flow") == "spend")
    agg = spend.groupBy("month", "category").agg(
        (-F.sum("amount_cents")).alias("spend_cents"),
        F.count(F.lit(1)).alias("txn_count"),
    )
    grid = (
        _month_axis(df).select("month")
        .crossJoin(spend.select("category").distinct())
        .join(agg, ["month", "category"], "left")
        .fillna(0, subset=["spend_cents", "txn_count"])
    )
    by_category = Window.partitionBy("category").orderBy("month")
    return (
        grid.withColumn("prev_month_spend_cents", F.lag("spend_cents").over(by_category))
        .withColumn("mom_change_cents", F.col("spend_cents") - F.col("prev_month_spend_cents"))
        .withColumn(
            "mom_change_pct",
            F.when(
                F.col("prev_month_spend_cents") > 0,
                F.col("mom_change_cents") / F.col("prev_month_spend_cents"),
            ),
        )
        .select(
            "month", "category", "spend_cents", "txn_count",
            "prev_month_spend_cents", "mom_change_cents", "mom_change_pct",
        )
        .orderBy("month", "category")
    )


def _cadence_case(position: int) -> Column:
    """CASE over the median gap: CADENCES[i][position] for the first band that contains it."""
    expr: Column | None = None
    for spec in CADENCES:
        cond = F.abs(F.col("median_gap") - spec[1]) <= spec[2]
        expr = F.when(cond, F.lit(spec[position])) if expr is None else expr.when(
            cond, F.lit(spec[position])
        )
    assert expr is not None
    return expr


def recurring_charges(df: DataFrame, cfg: GoldConfig = GoldConfig()) -> DataFrame:
    """Merchants charging on a regular schedule.

    Several charges on one day count as one occurrence (their amounts are summed), so a
    same-day duplicate never produces a zero-day gap.
    """
    non_transfer = df.filter(F.col("flow") != "transfer")
    daily = non_transfer.groupBy("merchant_norm", "txn_date").agg(
        F.sum("amount_cents").alias("amount_cents")
    )
    by_merchant = Window.partitionBy("merchant_norm").orderBy("txn_date")
    gaps = daily.withColumn("gap_days", F.datediff("txn_date", F.lag("txn_date").over(by_merchant)))

    cadence = (
        gaps.groupBy("merchant_norm")
        .agg(F.median("gap_days").alias("median_gap"))
        .withColumn("cadence", _cadence_case(0))
        .withColumn("center", _cadence_case(1))
        .withColumn("tol", _cadence_case(2))
        .filter(F.col("cadence").isNotNull())
    )
    share = (
        gaps.filter(F.col("gap_days").isNotNull())
        .join(cadence, "merchant_norm")
        .groupBy("merchant_norm")
        .agg(
            F.avg(
                (F.abs(F.col("gap_days") - F.col("center")) <= F.col("tol")).cast("double")
            ).alias("gap_share")
        )
    )
    amounts = daily.groupBy("merchant_norm").agg(
        F.count(F.lit(1)).alias("occurrences"),
        F.min("txn_date").alias("first_date"),
        F.max("txn_date").alias("last_date"),
        F.median("amount_cents").alias("median_amount"),
        F.avg("amount_cents").alias("mean_amount"),
        F.stddev_pop("amount_cents").alias("sd_amount"),
    )
    categories = non_transfer.groupBy("merchant_norm").agg(F.max("category").alias("category"))

    return (
        amounts.join(cadence, "merchant_norm")
        .join(share, "merchant_norm")
        .join(categories, "merchant_norm")
        .filter(
            (F.col("occurrences") >= cfg.recurring_min_occurrences)
            & (F.col("gap_share") >= cfg.recurring_min_gap_share)
        )
        .select(
            F.col("merchant_norm").alias("merchant"),
            "category",
            "cadence",
            "occurrences",
            "gap_share",
            F.round("median_amount").cast("long").alias("typical_amount_cents"),
            F.when(
                F.col("mean_amount") != 0,
                F.col("sd_amount") / F.abs(F.col("mean_amount")) <= cfg.stable_amount_cv,
            ).otherwise(F.lit(False)).alias("amount_stable"),
            (F.col("median_amount") > 0).alias("is_income"),
            "first_date",
            "last_date",
            F.date_add("last_date", F.round("median_gap").cast("int")).alias("next_expected_date"),
        )
        .orderBy("merchant")
    )


def anomalies(df: DataFrame, cfg: GoldConfig = GoldConfig()) -> DataFrame:
    """Charges far above the merchant's typical charge (conservative; see GoldConfig)."""
    charges = (
        df.filter((F.col("flow") == "spend") & (F.col("amount_cents") < 0))
        .withColumn("x", -F.col("amount_cents"))
    )
    typical = (
        charges.groupBy("merchant_norm")
        .agg(F.count(F.lit(1)).alias("n"), F.median("x").alias("med"))
        .filter(F.col("n") >= cfg.anomaly_min_charges)
    )
    scored = charges.join(typical, "merchant_norm")
    mad = scored.groupBy("merchant_norm").agg(
        F.median(F.abs(F.col("x") - F.col("med"))).alias("mad")
    )
    scale = F.greatest(
        F.lit(MAD_TO_SIGMA) * F.col("mad"), F.lit(cfg.anomaly_scale_floor) * F.col("med")
    )
    return (
        scored.join(mad, "merchant_norm")
        .withColumn("z_score", (F.col("x") - F.col("med")) / scale)
        .filter(
            (F.col("z_score") >= cfg.anomaly_min_z)
            & (F.col("x") >= cfg.anomaly_min_ratio * F.col("med"))
            & (F.col("x") - F.col("med") >= cfg.anomaly_min_excess_cents)
        )
        .select(
            "txn_id", "txn_date",
            F.col("merchant_norm").alias("merchant"),
            "category",
            "amount_cents",
            F.col("med").alias("typical_cents"),
            (F.col("x") / F.col("med")).alias("ratio"),
            "z_score",
        )
        .orderBy(F.col("z_score").desc())
    )


def build_gold(enriched: DataFrame, cfg: GoldConfig = GoldConfig()) -> dict[str, DataFrame]:
    df = prepare(enriched).cache()
    return {
        "monthly_summary": monthly_summary(df),
        "monthly_category_spend": monthly_category_spend(df),
        "recurring_charges": recurring_charges(df, cfg),
        "anomalies": anomalies(df, cfg),
        "transactions": transactions(enriched),
    }


def write_gold(warehouse_dir: str | Path, cfg: GoldConfig = GoldConfig()) -> dict:
    """Read silver (+ merchant categories when present), write gold tables, return row counts."""
    spark = get_spark()
    warehouse = Path(warehouse_dir)
    silver = spark.read.parquet(str(warehouse / "silver" / "transactions"))

    categories_path = warehouse / "categorization" / "merchant_categories"
    has_categories = categories_path.exists()
    if has_categories:
        from .enrich import with_categories

        enriched = with_categories(silver, spark.read.parquet(str(categories_path)))
    else:
        enriched = silver  # every merchant will show as "Uncategorized"

    summary: dict = {"categories_applied": has_categories}
    for name, table in build_gold(enriched, cfg).items():
        target = str(warehouse / "gold" / name)
        table.write.mode("overwrite").parquet(target)
        summary[name] = spark.read.parquet(target).count()
    return summary

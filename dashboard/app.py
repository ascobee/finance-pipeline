"""Personal finance dashboard (Streamlit), read-only over the gold tables.

    streamlit run dashboard/app.py
    streamlit run dashboard/app.py -- --warehouse data/warehouse

The warehouse folder can also come from the FINPIPE_WAREHOUSE environment variable. The app
reads Parquet directly with PyArrow (no Spark), keeps no state, and never writes. All data
preparation lives in ``finpipe.dashboard_data`` so it is unit-tested without a browser; this
file only turns those results into widgets.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from finpipe import dashboard_data as dd
from finpipe.gold_io import read_gold

st.set_page_config(page_title="Personal finance", layout="wide")


def default_warehouse() -> str:
    if os.environ.get("FINPIPE_WAREHOUSE"):
        return os.environ["FINPIPE_WAREHOUSE"]
    args = sys.argv[1:]
    if "--warehouse" in args and args.index("--warehouse") + 1 < len(args):
        return args[args.index("--warehouse") + 1]
    return "data/warehouse"


def _stamp(warehouse: str) -> float:
    """Newest modification time under gold/, so the cache refreshes after a pipeline run."""
    files = (Path(warehouse) / "gold").rglob("*")
    return max((f.stat().st_mtime for f in files if f.is_file()), default=0.0)


@st.cache_data(show_spinner=False)
def load_gold(warehouse: str, stamp: float) -> dict:
    return read_gold(warehouse)


def show_quality(warehouse: str) -> None:
    status = dd.quality_status(warehouse)
    if status is None:
        st.info("No data quality report yet. Run `python -m finpipe.quality` or add `--check`.")
    elif status["ok"] and not status["warnings"]:
        st.success(f"Data quality: all {status['passed']} checks passed.")
    else:
        message = (f"Data quality: {status['errors']} error(s), {status['warnings']} warning(s).")
        (st.error if not status["ok"] else st.warning)(message)
        for problem in status["problems"]:
            st.markdown(f"- **{problem['check']}** ({problem['severity']}): {problem['detail']}")


def show_overview(gold: dict, warehouse: str, month: str) -> None:
    by_month = dd.summary_by_month(gold)
    cur = by_month[month]
    prev = by_month.get(dd.previous_month(month))

    cols = st.columns(4)
    cols[0].metric("Income", dd.money(cur["income_cents"]))
    spend_delta = None
    if prev is not None:
        spend_delta = dd.money(cur["spend_cents"] - prev["spend_cents"])
    cols[1].metric("Spend", dd.money(cur["spend_cents"]), delta=spend_delta, delta_color="inverse")
    cols[2].metric("Net", dd.money(cur["net_cents"]))
    cols[3].metric("Savings rate", dd.pct(cur["savings_rate"]) or "n/a")
    if cur.get("is_partial_month"):
        st.caption("The data only partly covers this month.")

    st.subheader("Summary")
    saved = dd.saved_summary(warehouse, month)
    if saved:
        st.write(saved["text"])
        origin = "template" if saved["source"] == "template" else f"written by {saved.get('model')}"
        st.caption(f"{origin}; every number is checked against the data.")
    else:
        st.caption(f"No summary yet. Create one with `python -m finpipe.summary --month {month}`.")

    st.subheader("Income and spend by month")
    frame = pd.DataFrame(dd.overview(gold)).set_index("month")
    st.bar_chart(frame[["income", "spend"]])
    st.line_chart(frame[["savings_rate"]])
    partial = [m for m, flag in frame["partial"].items() if flag]
    if partial:
        st.caption("Partly covered months: " + ", ".join(dd.month_label(m) for m in partial))


def show_categories(gold: dict, month: str) -> None:
    rows = dd.category_breakdown(gold, month)
    st.subheader(f"Spend by category, {dd.month_label(month)}")
    if not rows:
        st.info("No spending recorded this month.")
    else:
        frame = pd.DataFrame(rows)
        st.bar_chart(frame.set_index("category")["spend"])
        st.dataframe(
            frame,
            hide_index=True,
            column_config={
                "category": "Category",
                "spend": st.column_config.NumberColumn("Spend ($)", format="%.2f"),
                "share_pct": st.column_config.NumberColumn("Share (%)", format="%.1f"),
                "change": st.column_config.NumberColumn("vs prior month ($)", format="%.2f"),
                "change_pct": st.column_config.NumberColumn("vs prior month (%)", format="%.1f"),
                "transactions": "Transactions",
            },
        )

    st.subheader("Trend")
    options = dd.spend_categories(gold)
    chosen = st.multiselect("Categories", options, default=options[:4])
    if chosen:
        trend = pd.DataFrame.from_dict(dd.category_trend(gold, chosen), orient="index")
        st.line_chart(trend.reindex(columns=chosen).fillna(0))


def show_recurring(gold: dict) -> None:
    view = dd.recurring_view(gold)
    if not view["rows"]:
        st.info("No recurring charges detected yet.")
        return
    cols = st.columns(2)
    cols[0].metric("Recurring outflow per month", dd.money(round(view["monthly_outflow"] * 100)))
    cols[1].metric("Per year", dd.money(round(view["annual_outflow"] * 100)))

    config = {
        "merchant": "Merchant", "category": "Category", "cadence": "Cadence",
        "typical_amount": st.column_config.NumberColumn("Typical ($)", format="%.2f"),
        "monthly_cost": st.column_config.NumberColumn("Per month ($)", format="%.2f"),
        "annual_cost": st.column_config.NumberColumn("Per year ($)", format="%.2f"),
        "amount": "Amount", "last_charge": "Last", "next_expected": "Next expected",
    }
    columns = list(config)
    charges = [r for r in view["rows"] if not r["is_income"]]
    income = [r for r in view["rows"] if r["is_income"]]
    st.subheader("Charges")
    st.dataframe(pd.DataFrame(charges)[columns], hide_index=True, column_config=config)
    if income:
        st.subheader("Recurring income")
        st.dataframe(pd.DataFrame(income)[columns], hide_index=True, column_config=config)


def show_attention(gold: dict, warehouse: str) -> None:
    st.subheader("Unusually large charges")
    anomalies = dd.anomaly_table(gold)
    if anomalies:
        st.dataframe(
            pd.DataFrame(anomalies),
            hide_index=True,
            column_config={
                "amount": st.column_config.NumberColumn("Amount ($)", format="%.2f"),
                "typical": st.column_config.NumberColumn("Typical ($)", format="%.2f"),
                "times_typical": st.column_config.NumberColumn("x typical", format="%.1f"),
            },
        )
    else:
        st.caption("Nothing flagged. Detection is deliberately conservative.")

    st.subheader("Merchants awaiting review")
    queue = dd.review_queue(warehouse)
    if queue:
        st.dataframe(pd.DataFrame(queue), hide_index=True)
        st.caption(
            "Low-confidence categories. To fix one, add a line `merchant,category` to "
            "`data/private/corrections.csv` and rerun the pipeline."
        )
    else:
        st.caption("No merchants need review.")


def show_transactions(gold: dict, month: str) -> None:
    if not gold.get("transactions"):
        st.info(
            "This warehouse has no transactions table yet. Rebuild the gold tables with "
            "`python -m finpipe.pipeline --categorize --gold`."
        )
        return
    options = dd.transaction_filter_options(gold)

    top = st.columns([1, 1, 2])
    all_months = top[0].checkbox("All months", value=False)
    hide_transfers = top[1].checkbox("Hide transfers", value=False)
    search = top[2].text_input("Search merchant or description")
    pick = st.columns(2)
    categories = pick[0].multiselect("Categories", options["categories"],
                                     default=options["categories"])
    sources = pick[1].multiselect("Source", options["sources"], default=options["sources"])

    view = dd.transactions_view(
        gold,
        None if all_months else month,
        # every option selected means "no filter", so rows with a blank source are not dropped
        None if set(categories) == set(options["categories"]) else categories,
        None if set(sources) == set(options["sources"]) else sources,
        search,
        hide_transfers,
    )
    label = "all months" if all_months else dd.month_label(month)

    cols = st.columns(4)
    cols[0].metric("Transactions", f"{view['count']:,}")
    cols[1].metric("Income", dd.money(view["income_cents"]))
    cols[2].metric("Spend", dd.money(view["spend_cents"]))
    cols[3].metric("Transfers (net)", dd.money(view["transfer_net_cents"]))
    st.caption(f"{label.capitalize()}. Spend is net of refunds, as on the Overview tab.")
    if not all_months and dd.transactions_match_summary(gold, month) is False:
        st.warning("This month's transactions do not add up to the monthly summary. "
                   "Rebuild the gold tables and run the data quality checks.")

    if not view["rows"]:
        st.info("No transactions match these filters.")
    else:
        st.dataframe(
            pd.DataFrame(view["rows"], columns=list(dd.TRANSACTION_COLUMNS)),
            hide_index=True,
            column_config={
                "date": "Date", "merchant": "Merchant", "category": "Category",
                "amount": st.column_config.NumberColumn("Amount ($)", format="%.2f"),
                "flow": "Type", "source": "Source", "description": "Bank description",
                "txn_id": st.column_config.TextColumn("Transaction ID", width="small"),
            },
        )
    st.download_button(
        "Download as CSV",
        dd.transactions_csv(view["rows"]),
        file_name=f"transactions_{'all' if all_months else month}.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title("Personal finance")
    warehouse = st.sidebar.text_input("Warehouse folder", value=default_warehouse())
    if st.sidebar.button("Reload data"):
        st.cache_data.clear()

    gold = load_gold(warehouse, _stamp(warehouse))
    if not gold.get("monthly_summary"):
        st.warning(
            f"No gold tables found in `{warehouse}`. "
            "Build them with `python -m finpipe.pipeline --categorize --gold`."
        )
        return

    months = dd.available_months(gold)
    default = dd.latest_month(gold)
    month = st.sidebar.selectbox(
        "Month", months, index=months.index(default), format_func=dd.month_label
    )
    st.caption(f"Data covers {dd.month_label(months[0])} to {dd.month_label(months[-1])}.")
    show_quality(warehouse)

    overview_tab, categories_tab, recurring_tab, transactions_tab, attention_tab = st.tabs(
        ["Overview", "Categories", "Recurring", "Transactions", "Needs attention"]
    )
    with overview_tab:
        show_overview(gold, warehouse, month)
    with categories_tab:
        show_categories(gold, month)
    with recurring_tab:
        show_recurring(gold)
    with transactions_tab:
        show_transactions(gold, month)
    with attention_tab:
        show_attention(gold, warehouse)


main()

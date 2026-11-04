"""Reading gold without Spark, the summary CLI, and the dashboard app. Needs pyarrow (and
streamlit for the app test); skipped otherwise. No Spark involved: gold tables are written
straight to Parquet from the pure reference."""

import json
from pathlib import Path

import pytest
from support import gold_records, sample_silver_rows

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from finpipe import dashboard_data as dd  # noqa: E402
from finpipe.gold_io import read_gold  # noqa: E402
from finpipe.summary import main as summary_main  # noqa: E402
from finpipe.synthetic import write_sample  # noqa: E402


def write_warehouse(tmp_path):
    raw = tmp_path / "raw"
    write_sample(raw, seed=42)
    gold = gold_records(sample_silver_rows(raw))
    warehouse = tmp_path / "warehouse"
    for name, rows in gold.items():
        if rows:   # an empty table has no columns to write
            target = warehouse / "gold" / name
            target.mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist(rows), target / "part-00000.parquet")
            (target / "_SUCCESS").write_text("")      # Spark leaves marker files like this
    return warehouse, gold


def test_read_gold_round_trips_the_tables(tmp_path):
    warehouse, gold = write_warehouse(tmp_path)
    back = read_gold(warehouse)
    assert set(back) == {n for n, rows in gold.items() if rows}
    assert len(back["monthly_summary"]) == len(gold["monthly_summary"]) == 6
    assert len(back["transactions"]) == len(gold["transactions"]) == 369
    assert {r["txn_id"] for r in back["transactions"]} == {r["txn_id"] for r in gold["transactions"]}
    assert dd.transactions_view(back)["rows"] == dd.transactions_view(gold)["rows"]
    assert dd.available_months(back) == dd.available_months(gold)
    assert back["recurring_charges"][0]["next_expected_date"] is not None
    assert dd.recurring_view(back)["rows"] == dd.recurring_view(gold)["rows"]


def test_read_gold_of_an_empty_warehouse_is_empty(tmp_path):
    assert read_gold(tmp_path) == {}


def test_summary_cli_writes_a_template_summary_without_any_api_call(tmp_path, capsys=None):
    warehouse, _ = write_warehouse(tmp_path)
    assert summary_main(["--warehouse", str(warehouse), "--month", "2025-03",
                         "--template-only"]) == 0
    saved = json.loads((warehouse / "summaries" / "2025-03.json").read_text())
    assert saved["source"] == "template" and "March 2025" in saved["text"]
    assert dd.saved_summary(warehouse, "2025-03")["text"] == saved["text"]


def test_summary_cli_defaults_to_the_latest_complete_month(tmp_path):
    warehouse, gold = write_warehouse(tmp_path)
    assert summary_main(["--warehouse", str(warehouse), "--template-only"]) == 0
    assert (warehouse / "summaries" / f"{dd.latest_month(read_gold(warehouse))}.json").exists()


def test_summary_cli_without_gold_tables_exits_2(tmp_path):
    assert summary_main(["--warehouse", str(tmp_path), "--template-only"]) == 2


# ---------------------------------------------------------------- the Streamlit app

APP = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"


def run_app(warehouse, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("pandas")
    monkeypatch.setenv("FINPIPE_WAREHOUSE", str(warehouse))
    return testing.AppTest.from_file(str(APP), default_timeout=60).run()


def test_app_renders_every_section_without_errors(tmp_path, monkeypatch):
    warehouse, _ = write_warehouse(tmp_path)
    summary_main(["--warehouse", str(warehouse), "--month", "2025-03", "--template-only"])
    app = run_app(warehouse, monkeypatch)
    assert not app.exception, [e.value for e in app.exception]
    assert app.title[0].value == "Personal finance"
    assert {m.label for m in app.metric} >= {"Income", "Spend", "Net", "Savings rate"}
    assert len(app.tabs) == 5
    assert [t.label for t in app.tabs][3] == "Transactions"


def test_app_explains_what_to_do_when_there_is_no_data(tmp_path, monkeypatch):
    app = run_app(tmp_path, monkeypatch)
    assert not app.exception, [e.value for e in app.exception]
    assert any("No gold tables" in w.value for w in app.warning)


def test_transactions_tab_filters_and_handles_a_warehouse_without_the_table(tmp_path, monkeypatch):
    warehouse, _ = write_warehouse(tmp_path)
    app = run_app(warehouse, monkeypatch)
    assert not app.exception, [e.value for e in app.exception]
    labels = {c.label for c in app.checkbox}
    assert {"All months", "Hide transfers"} <= labels
    assert any(m.label == "Transactions" for m in app.metric)

    next(c for c in app.checkbox if c.label == "All months").check()
    next(c for c in app.checkbox if c.label == "Hide transfers").check()
    app.run()
    assert not app.exception, [e.value for e in app.exception]

    import shutil
    shutil.rmtree(warehouse / "gold" / "transactions")        # a warehouse from before this table
    old = run_app(warehouse, monkeypatch)
    assert not old.exception, [e.value for e in old.exception]
    assert any("no transactions table" in i.value for i in old.info)

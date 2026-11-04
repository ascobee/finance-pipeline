import pytest
from fakes import ScriptedClient

from finpipe.categorize import Categorizer
from finpipe.evaluate import (
    evaluate,
    format_report,
    golden_from_truth,
    load_golden,
    run_eval,
    write_golden,
)
from finpipe.synthetic import write_sample


def test_golden_from_synthetic_truth_has_one_label_per_normalized_merchant(tmp_path):
    write_sample(tmp_path, seed=42)
    golden = golden_from_truth(tmp_path / "truth" / "merchants.csv")
    assert len(golden) == 28
    assert golden["Blue Bottle Coffee"] == "Dining"
    assert golden["Online Transfer To Savings"] == "Transfer"


def test_golden_round_trips_through_csv(tmp_path):
    golden = {"Shell": "Fuel", "Netflix": "Subscriptions"}
    write_golden(golden, tmp_path / "g.csv")
    assert load_golden(tmp_path / "g.csv") == golden


def test_load_golden_rejects_bad_categories_and_conflicts(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("merchant,category\nShell,Petrol\n")
    with pytest.raises(ValueError, match="unknown category"):
        load_golden(bad)
    conflict = tmp_path / "conflict.csv"
    conflict.write_text("merchant,category\nShell,Fuel\nShell,Shopping\n")
    with pytest.raises(ValueError, match="conflicting"):
        load_golden(conflict)


def test_accuracy_errors_and_flag_stats():
    golden = {"A": "Dining", "B": "Fuel", "C": "Shopping", "D": "Health"}
    client = ScriptedClient({
        "A": ("Dining", 0.95),     # right, confident
        "B": ("Fuel", 0.50),       # right, flagged
        "C": ("Dining", 0.90),     # wrong, confident
        "D": ("Shopping", 0.40),   # wrong, flagged
    })
    report = run_eval(golden, client, cache_path=None)
    assert (report.total, report.correct, report.accuracy) == (4, 2, 0.5)
    assert report.flagged == 2 and report.unflagged == 2
    assert report.unflagged_accuracy == 0.5
    assert {e["merchant"] for e in report.errors} == {"C", "D"}
    assert report.per_category["Dining"] == (1, 1)
    assert report.per_category["Shopping"] == (0, 1)
    text = format_report(report)
    assert "2/4 = 50.0%" in text and "'C': expected Shopping, got Dining" in text


def test_eval_ignores_corrections_and_rules_so_it_measures_the_model():
    client = ScriptedClient({"Shell": ("Fuel", 0.9)})
    run_eval({"Shell": "Fuel"}, client, cache_path=None)
    assert client.requested == ["Shell"]


def test_fallbacks_count_as_errors_and_are_reported():
    client = ScriptedClient(script=[[]] * 3)
    report = run_eval({"Shell": "Fuel"}, client, cache_path=None)
    assert (report.correct, report.fallbacks, report.flagged) == (0, 1, 1)


def test_missing_prediction_is_an_error_not_a_silent_skip():
    results = Categorizer(ScriptedClient()).categorize(["A"])
    with pytest.raises(KeyError):
        evaluate({"A": "Other", "B": "Other"}, results)

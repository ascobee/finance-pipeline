import csv
import json

import pytest
from fakes import ScriptedClient

from finpipe.categories import Category
from finpipe.categorize import (
    Categorizer,
    LLMConfigError,
    load_corrections,
    write_review_queue,
)


def by_name(results):
    return {r.merchant: r for r in results}


# ---------------------------------------------------------------- precedence


def test_correction_beats_rule_cache_and_llm(tmp_path):
    cache = tmp_path / "cache.jsonl"
    cache.write_text(json.dumps({
        "merchant": "Netflix", "category": "Dining", "confidence": 0.9,
        "model": "old", "categorized_at": "2026-01-01T00:00:00+00:00",
    }) + "\n")
    client = ScriptedClient({"Netflix": ("Shopping", 0.99)})
    cat = Categorizer(
        client, cache_path=cache,
        corrections={"netflix": Category.ENTERTAINMENT},
        rules={"Netflix": Category.TRANSFER},
    )
    (r,) = cat.categorize(["Netflix"])
    assert (r.category, r.source, r.needs_review) == ("Entertainment", "correction", False)
    assert client.calls == []


def test_rule_beats_cache_and_llm(tmp_path):
    client = ScriptedClient()
    cat = Categorizer(client, rules={"Online Transfer To Savings": Category.TRANSFER})
    (r,) = cat.categorize(["Online Transfer To Savings"])
    assert (r.category, r.source) == ("Transfer", "rule")
    assert client.calls == []


def test_cache_beats_llm(tmp_path):
    cache = tmp_path / "cache.jsonl"
    cache.write_text(json.dumps({
        "merchant": "Chipotle", "category": "Dining", "confidence": 0.95, "model": "m",
        "categorized_at": "2026-01-01T00:00:00+00:00",
    }) + "\n")
    client = ScriptedClient()
    (r,) = Categorizer(client, cache_path=cache).categorize(["Chipotle"])
    assert (r.category, r.source) == ("Dining", "cache")
    assert client.calls == []


def test_unknown_merchant_is_other_without_calling_the_llm():
    client = ScriptedClient()
    (r,) = Categorizer(client).categorize(["Unknown"])
    assert (r.category, r.source) == ("Other", "rule")
    assert client.calls == []


# ---------------------------------------------------------------- the LLM path


def test_llm_answers_are_used_and_cached_so_a_second_run_is_free(tmp_path):
    cache = tmp_path / "cache.jsonl"
    client = ScriptedClient({"Blue Bottle Coffee": ("Dining", 0.97), "Shell": ("Fuel", 0.95)})
    first = Categorizer(client, cache_path=cache).categorize(["Blue Bottle Coffee", "Shell"])
    assert {r.merchant: (r.category, r.source) for r in first} == {
        "Blue Bottle Coffee": ("Dining", "llm"), "Shell": ("Fuel", "llm"),
    }
    assert len(client.calls) == 1

    again = ScriptedClient()
    second = Categorizer(again, cache_path=cache).categorize(["Blue Bottle Coffee", "Shell"])
    assert again.calls == []
    assert {r.merchant: (r.category, r.source) for r in second} == {
        "Blue Bottle Coffee": ("Dining", "cache"), "Shell": ("Fuel", "cache"),
    }


def test_duplicates_and_blank_input_are_collapsed():
    client = ScriptedClient()
    results = Categorizer(client).categorize(["Shell", "shell", " Shell ", None, ""])
    assert len(results) == 1
    assert client.requested == ["Shell"]


def test_batching_splits_requests():
    client = ScriptedClient()
    Categorizer(client, batch_size=25).categorize([f"Merchant {i}" for i in range(60)])
    assert [len(c) for c in client.calls] == [25, 25, 10]


def test_merchant_matching_in_the_response_ignores_case_and_spacing():
    client = ScriptedClient(script=[[
        {"merchant": "  BLUE   bottle coffee ", "category": "Dining", "confidence": 0.9}
    ]])
    (r,) = Categorizer(client).categorize(["Blue Bottle Coffee"])
    assert (r.merchant, r.category, r.source) == ("Blue Bottle Coffee", "Dining", "llm")


def test_hallucinated_category_is_rejected_then_retried():
    bad = [{"merchant": "Shell", "category": "Gasoline Stuff", "confidence": 0.9}]
    client = ScriptedClient({"Shell": ("Fuel", 0.95)}, script=[bad])
    cat = Categorizer(client)
    (r,) = cat.categorize(["Shell"])
    assert (r.category, r.source) == ("Fuel", "llm")
    assert len(client.calls) == 2
    assert cat.stats["invalid_items"] == 1


@pytest.mark.parametrize(
    "bad_entry",
    [
        {"merchant": "Shell", "category": "Fuel", "confidence": 1.7},
        {"merchant": "Shell", "category": "Fuel", "confidence": -0.1},
        {"merchant": "Shell", "category": "Fuel"},
        {"merchant": "", "category": "Fuel", "confidence": 0.9},
        {"category": "Fuel", "confidence": 0.9},
        "not a dict",
    ],
)
def test_malformed_items_are_never_accepted(bad_entry):
    client = ScriptedClient(script=[[bad_entry]] * 3)
    (r,) = Categorizer(client, max_attempts=3).categorize(["Shell"])
    assert (r.category, r.source, r.needs_review) == ("Other", "fallback", True)


def test_merchants_missing_from_a_response_are_retried():
    client = ScriptedClient(
        {"A": ("Dining", 0.9), "B": ("Fuel", 0.9)},
        script=[[{"merchant": "A", "category": "Dining", "confidence": 0.9}]],
    )
    results = by_name(Categorizer(client).categorize(["A", "B"]))
    assert results["A"].source == "llm" and results["B"].source == "llm"
    assert client.calls == [["A", "B"], ["B"]]


def test_extra_and_duplicate_merchants_in_a_response_are_ignored():
    client = ScriptedClient(script=[[
        {"merchant": "A", "category": "Dining", "confidence": 0.9},
        {"merchant": "A", "category": "Fuel", "confidence": 0.9},
        {"merchant": "Not Requested", "category": "Fuel", "confidence": 0.9},
    ]])
    cat = Categorizer(client)
    results = cat.categorize(["A"])
    assert [(r.merchant, r.category) for r in results] == [("A", "Dining")]
    assert cat.stats["ignored_items"] == 2


def test_exhausted_retries_fall_back_to_other_and_are_not_cached(tmp_path):
    cache = tmp_path / "cache.jsonl"
    client = ScriptedClient(script=[RuntimeError("boom")] * 3)
    cat = Categorizer(client, cache_path=cache, max_attempts=3)
    (r,) = cat.categorize(["Shell"])
    assert (r.category, r.source, r.needs_review, r.confidence) == ("Other", "fallback", True, 0.0)
    assert len(client.calls) == 3
    assert not cache.exists() or cache.read_text() == ""

    healthy = ScriptedClient({"Shell": ("Fuel", 0.95)})
    (retry,) = Categorizer(healthy, cache_path=cache).categorize(["Shell"])
    assert (retry.category, retry.source) == ("Fuel", "llm")  # a fallback is retried next run


def test_low_confidence_is_flagged_but_cached(tmp_path):
    cache = tmp_path / "cache.jsonl"
    client = ScriptedClient({"Xyz Holdings": ("Shopping", 0.4)})
    (r,) = Categorizer(client, cache_path=cache).categorize(["Xyz Holdings"])
    assert (r.category, r.needs_review) == ("Shopping", True)

    again = ScriptedClient()
    (cached,) = Categorizer(again, cache_path=cache).categorize(["Xyz Holdings"])
    assert again.calls == []
    assert (cached.source, cached.needs_review) == ("cache", True)


def test_threshold_boundary():
    client = ScriptedClient({"A": ("Dining", 0.7), "B": ("Dining", 0.69)})
    results = by_name(Categorizer(client, review_threshold=0.7).categorize(["A", "B"]))
    assert results["A"].needs_review is False
    assert results["B"].needs_review is True


def test_progress_survives_a_crash_midway(tmp_path):
    cache = tmp_path / "cache.jsonl"
    merchants = [f"M{i}" for i in range(4)]
    # batch 1 succeeds, batch 2 fails on every attempt
    client = ScriptedClient(script=[None, RuntimeError("x"), RuntimeError("x")])
    client.script[0] = lambda ms: [client.entry(m) for m in ms]
    cat = Categorizer(client, cache_path=cache, batch_size=2, max_attempts=2)
    results = by_name(cat.categorize(merchants))
    assert results["M0"].source == "llm" and results["M1"].source == "llm"
    assert results["M2"].source == "fallback" and results["M3"].source == "fallback"
    cached = [json.loads(line)["merchant"] for line in cache.read_text().splitlines()]
    assert cached == ["M0", "M1"]


def test_config_errors_are_not_swallowed():
    client = ScriptedClient(script=[LLMConfigError("bad key")])
    with pytest.raises(LLMConfigError):
        Categorizer(client).categorize(["Shell"])


def test_corrupt_cache_lines_are_skipped(tmp_path):
    cache = tmp_path / "cache.jsonl"
    good = {"merchant": "Shell", "category": "Fuel", "confidence": 0.9, "model": "m",
            "categorized_at": "t"}
    cache.write_text("not json\n\n" + json.dumps({**good, "category": "Nope"}) + "\n"
                     + json.dumps(good) + "\n")
    client = ScriptedClient()
    (r,) = Categorizer(client, cache_path=cache).categorize(["Shell"])
    assert (r.category, r.source) == ("Fuel", "cache")


def test_result_order_follows_input_order():
    results = Categorizer(ScriptedClient()).categorize(["Zeta", "Alpha", "Mid"])
    assert [r.merchant for r in results] == ["Zeta", "Alpha", "Mid"]


# ---------------------------------------------------------------- corrections file


def test_load_corrections_reads_and_validates(tmp_path):
    path = tmp_path / "corrections.csv"
    path.write_text("merchant,category\nAcme Gym,health\nCorner Cafe,Dining\n,\n")
    assert load_corrections(path) == {
        "Acme Gym": Category.HEALTH, "Corner Cafe": Category.DINING,
    }


def test_load_corrections_missing_file_is_empty(tmp_path):
    assert load_corrections(tmp_path / "nope.csv") == {}
    assert load_corrections(None) == {}


def test_load_corrections_rejects_unknown_category_with_line_number(tmp_path):
    path = tmp_path / "corrections.csv"
    path.write_text("merchant,category\nAcme Gym,Health\nBad Row,Fitness\n")
    with pytest.raises(ValueError, match=r":3:"):
        load_corrections(path)


def test_load_corrections_requires_header(tmp_path):
    path = tmp_path / "corrections.csv"
    path.write_text("name,cat\nx,y\n")
    with pytest.raises(ValueError, match="header"):
        load_corrections(path)


# ---------------------------------------------------------------- review queue


def test_review_queue_lists_only_flagged_merchants_lowest_confidence_first(tmp_path):
    client = ScriptedClient({"A": ("Dining", 0.95), "B": ("Shopping", 0.5), "C": ("Fuel", 0.3)})
    results = Categorizer(client).categorize(["A", "B", "C"])
    path = tmp_path / "review" / "queue.csv"
    assert write_review_queue(results, path) == 2
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert [(r["merchant"], r["suggested_category"]) for r in rows] == [
        ("C", "Fuel"), ("B", "Shopping"),
    ]

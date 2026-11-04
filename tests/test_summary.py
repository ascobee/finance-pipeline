import json
from datetime import date

import pytest
from fakes import ScriptedSummaryClient
from support import gold_records, row, sample_silver_rows

from finpipe.categorize import LLMConfigError
from finpipe.summary import (
    PROMPT_VERSION, allowed_numbers, build_facts, facts_key, generate_summary, template_summary,
    verify_numbers,
)
from finpipe.synthetic import write_sample


def make_gold(tmp_path, extra_rows=()):
    write_sample(tmp_path, seed=42)
    rows = sample_silver_rows(tmp_path) + list(extra_rows)
    return gold_records(rows), rows


# ---------------------------------------------------------------- facts


def test_facts_report_the_month_in_formatted_strings(tmp_path):
    gold, _ = make_gold(tmp_path)
    facts = build_facts(gold, "2025-03")
    by_month = {r["month"].isoformat()[:7]: r for r in gold["monthly_summary"]}
    cur, prev = by_month["2025-03"], by_month["2025-02"]
    assert facts["month_label"] == "March 2025" and facts["partial_month"] is False
    assert facts["income"] == f"${cur['income_cents'] / 100:,.2f}"
    assert facts["spend"] == f"${cur['spend_cents'] / 100:,.2f}"
    assert facts["previous_month"]["label"] == "February 2025"
    assert facts["previous_month"]["spend"] == f"${prev['spend_cents'] / 100:,.2f}"
    assert facts["previous_month"]["spend_direction"] in ("up", "down", "unchanged")


def test_first_month_has_no_previous_comparison(tmp_path):
    gold, _ = make_gold(tmp_path)
    facts = build_facts(gold, "2025-01")
    assert facts["previous_month"] is None
    assert facts["biggest_changes"] == []


def test_top_categories_are_largest_first_and_capped_at_five(tmp_path):
    gold, _ = make_gold(tmp_path)
    cats = build_facts(gold, "2025-03")["top_categories"]
    amounts = [float(c["spend"].replace("$", "").replace(",", "")) for c in cats]
    assert 1 <= len(cats) <= 5 and amounts == sorted(amounts, reverse=True)
    assert all(c["share_of_spend"].endswith("%") for c in cats)


def test_merchant_names_can_be_left_out(tmp_path):
    gold, _ = make_gold(tmp_path)
    with_names = json.dumps(build_facts(gold, "2025-03", include_merchants=True))
    without = json.dumps(build_facts(gold, "2025-03", include_merchants=False))
    assert "Zelle Payment To J Martinez" in with_names
    assert "Zelle" not in without and "Netflix" not in without and "Xfinity" not in without


def test_anomalies_appear_for_their_month_only(tmp_path):
    spike = [row(f"s{i}", date.fromordinal(date(2025, 1, 2).toordinal() + 5 * i),
                 -c, "Spike Mart", "Shopping")
             for i, c in enumerate([4_000, 4_400, 3_800, 4_200, 4_100, 3_900, 4_300, 400_000])]
    gold, _ = make_gold(tmp_path, spike)
    spike_month = next(a["txn_date"].isoformat()[:7] for a in gold["anomalies"])
    facts = build_facts(gold, spike_month)
    assert facts["anomalies"]["count"] == 1
    assert facts["anomalies"]["items"][0]["merchant"] == "Spike Mart"
    assert facts["anomalies"]["items"][0]["amount"] == "$4,000.00"
    anon = build_facts(gold, spike_month, include_merchants=False)
    assert "merchant" not in anon["anomalies"]["items"][0]
    assert anon["anomalies"]["items"][0]["category"] == "Shopping"
    other = "2025-06" if spike_month != "2025-06" else "2025-05"
    assert build_facts(gold, other)["anomalies"]["count"] == 0


def test_recurring_summary_counts_outflows_only(tmp_path):
    gold, _ = make_gold(tmp_path)
    rec = build_facts(gold, "2025-03")["recurring"]
    assert rec["count"] == 6      # seven recurring merchants, one of them payroll (income)
    assert rec["largest"][0]["merchant"] == "Zelle Payment To J Martinez"
    assert rec["monthly_cost"].startswith("$")


def test_unknown_month_is_a_clear_error(tmp_path):
    gold, _ = make_gold(tmp_path)
    with pytest.raises(ValueError, match="available: 2025-01"):
        build_facts(gold, "2031-01")


# ---------------------------------------------------------------- number verification


FACTS = {"month": "2025-03", "month_label": "March 2025", "income": "$4,982.48",
         "spend": "$4,031.07", "savings_rate": "19.1%", "anomalies": {"count": 2},
         "items": [{"date": "2025-03-14"}]}


def test_numbers_copied_from_the_facts_are_accepted():
    text = ("In March 2025 you earned $4,982.48 and spent $4,031.07, a savings rate of 19.1%. "
            "Two charges stood out, 2 in total, the largest on March 14 (2025-03-14).")
    assert verify_numbers(text, FACTS) == []


def test_whole_dollar_and_one_decimal_roundings_are_accepted():
    assert verify_numbers("Spending was about $4,031 and income roughly $4,982.5.", FACTS) == []
    assert verify_numbers("Spending was about $4,032.", FACTS) == ["4,032"]


def test_invented_numbers_are_reported_once_each():
    bad = verify_numbers("You spent $4,500.00 and saved 25%, after $4,500.00 of bills.", FACTS)
    assert bad == ["4,500.00", "25"]


def test_trailing_punctuation_does_not_break_matching():
    assert verify_numbers("Income was $4,982.48, spending $4,031.07.", FACTS) == []


def test_allowed_numbers_include_dates_and_counts():
    allowed = allowed_numbers(FACTS)
    assert {2025, 3, 14, 2} <= {int(a) for a in allowed if a == int(a)}


def test_a_prose_count_without_digits_is_never_flagged():
    assert verify_numbers("Three categories dominated.", FACTS) == []


# ---------------------------------------------------------------- template


def test_template_summary_is_always_grounded(tmp_path):
    gold, _ = make_gold(tmp_path)
    for month in ("2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"):
        facts = build_facts(gold, month)
        text = template_summary(facts)
        assert verify_numbers(text, facts) == [], (month, text)
        assert facts["month_label"] in text and facts["spend"] in text


# ---------------------------------------------------------------- generation


def facts_for(tmp_path, month="2025-03"):
    gold, _ = make_gold(tmp_path)
    return build_facts(gold, month)


def test_grounded_draft_is_accepted_and_cached(tmp_path):
    facts = facts_for(tmp_path)
    cache = tmp_path / "summaries" / "2025-03.json"
    client = ScriptedSummaryClient()
    first = generate_summary(facts, client, cache)
    assert (first.source, first.attempts, first.model) == ("llm", 1, "fake-summary-model")
    assert json.loads(cache.read_text())["key"] == facts_key(facts, client.model)

    healthy = ScriptedSummaryClient([RuntimeError("must not be called")])
    again = generate_summary(facts, healthy, cache)
    assert again.source == "cache" and again.text == first.text and healthy.calls == 0


def test_changed_facts_or_model_invalidate_the_cache(tmp_path):
    facts = facts_for(tmp_path)
    cache = tmp_path / "s.json"
    generate_summary(facts, ScriptedSummaryClient(), cache)

    changed = {**facts, "income": "$1.00"}
    client = ScriptedSummaryClient()
    assert generate_summary(changed, client, cache).source == "llm" and client.calls == 1

    other_model = ScriptedSummaryClient()
    other_model.model = "different-model"
    generate_summary(changed, other_model, cache)
    assert other_model.calls == 1


def test_hallucinated_number_is_rejected_and_the_model_is_told_which(tmp_path):
    facts = facts_for(tmp_path)
    client = ScriptedSummaryClient(["You spent $99,999.99 this month."])
    result = generate_summary(facts, client, tmp_path / "s.json")
    assert (result.source, result.attempts) == ("llm", 2)
    assert client.feedback[0] is None and "99,999.99" in client.feedback[1]
    assert verify_numbers(result.text, facts) == []


def test_persistent_hallucination_falls_back_to_the_template_and_is_not_cached(tmp_path):
    facts = facts_for(tmp_path)
    cache = tmp_path / "s.json"
    client = ScriptedSummaryClient(["Saved $12,345."] * 3)
    result = generate_summary(facts, client, cache, max_attempts=3)
    assert (result.source, result.attempts, result.model) == ("template", 3, None)
    assert result.text == template_summary(facts)

    healthy = ScriptedSummaryClient()
    assert generate_summary(facts, healthy, cache).source == "llm"   # retried, not stuck


def test_empty_drafts_and_transient_errors_are_retried(tmp_path):
    facts = facts_for(tmp_path)
    client = ScriptedSummaryClient(["   ", RuntimeError("timeout")])
    result = generate_summary(facts, client, None)
    assert (result.source, result.attempts) == ("llm", 3)
    assert client.feedback[1] == "The draft was empty."


def test_credential_errors_are_not_swallowed(tmp_path):
    facts = facts_for(tmp_path)
    with pytest.raises(LLMConfigError):
        generate_summary(facts, ScriptedSummaryClient([LLMConfigError("bad key")]), None)


def test_no_client_means_the_template_without_any_api_call(tmp_path):
    facts = facts_for(tmp_path)
    cache = tmp_path / "s.json"
    result = generate_summary(facts, None, cache)
    assert result.source == "template" and result.text == template_summary(facts)
    assert json.loads(cache.read_text())["source"] == "template"
    assert generate_summary(facts, ScriptedSummaryClient(), cache).source == "llm"


def test_prompt_version_is_part_of_the_cache_key():
    assert facts_key({"a": 1}, "m") != facts_key({"a": 2}, "m")
    assert PROMPT_VERSION in json.dumps({"v": PROMPT_VERSION})

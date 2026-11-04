"""Behavior of the gold algorithms, specified against the pure-Python reference.

These run without Spark. tests/test_gold.py then checks that the Spark implementation agrees
with this reference on the same inputs.
"""

from datetime import date

import reference_gold as ref
from support import row, sample_silver_rows

from finpipe.gold_config import GoldConfig
from finpipe.synthetic import write_sample

D = date


# ---------------------------------------------------------------- cash-flow classification


def test_refunds_net_against_category_spend():
    rows = [
        row("a", D(2025, 1, 5), -10_000, "Target", "Shopping"),
        row("b", D(2025, 1, 9), 2_500, "Target", "Shopping"),   # refund
    ]
    cell = ref.monthly_category_spend(rows)[(D(2025, 1, 1), "Shopping")]
    assert cell["spend_cents"] == 7_500
    assert cell["txn_count"] == 2
    assert ref.monthly_summary(rows)[D(2025, 1, 1)]["spend_cents"] == 7_500


def test_transfers_are_not_spend_or_income_but_still_reconcile():
    rows = [
        row("pay", D(2025, 1, 1), 300_000, "Acme Payroll", "Income"),
        row("food", D(2025, 1, 2), -5_000, "King Soopers", "Groceries"),
        row("cc", D(2025, 1, 25), -90_000, "Credit Card Payment", "Transfer", is_transfer=True),
        row("llm", D(2025, 1, 26), -1_000, "Venmo", "Transfer"),  # labeled Transfer, not flagged
    ]
    jan = ref.monthly_summary(rows)[D(2025, 1, 1)]
    assert (jan["income_cents"], jan["spend_cents"]) == (300_000, 5_000)
    assert jan["transfer_net_cents"] == -91_000
    assert jan["income_cents"] - jan["spend_cents"] + jan["transfer_net_cents"] == sum(
        r["amount_cents"] for r in rows
    )
    assert jan["savings_rate"] == (300_000 - 5_000) / 300_000


def test_missing_category_is_uncategorized_not_dropped():
    rows = [row("a", D(2025, 1, 5), -2_000, "Mystery Shop", None)]
    cell = ref.monthly_category_spend(rows)[(D(2025, 1, 1), "Uncategorized")]
    assert cell["spend_cents"] == 2_000


def test_savings_rate_is_null_without_income():
    rows = [row("a", D(2025, 1, 5), -2_000, "Shop", "Shopping")]
    assert ref.monthly_summary(rows)[D(2025, 1, 1)]["savings_rate"] is None


# ---------------------------------------------------------------- month axis


def test_months_without_activity_appear_as_zero():
    rows = [
        row("a", D(2025, 1, 5), -1_000, "Shop", "Shopping"),
        row("b", D(2025, 3, 5), -3_000, "Shop", "Shopping"),
    ]
    summary = ref.monthly_summary(rows)
    assert sorted(summary) == [D(2025, 1, 1), D(2025, 2, 1), D(2025, 3, 1)]
    assert summary[D(2025, 2, 1)]["spend_cents"] == 0

    grid = ref.monthly_category_spend(rows)
    assert grid[(D(2025, 2, 1), "Shopping")]["spend_cents"] == 0
    # March compares to February (zero), not to January
    assert grid[(D(2025, 3, 1), "Shopping")]["prev_month_spend_cents"] == 0
    assert grid[(D(2025, 3, 1), "Shopping")]["mom_change_cents"] == 3_000
    assert grid[(D(2025, 3, 1), "Shopping")]["mom_change_pct"] is None  # no base to divide by


def test_first_month_has_no_previous_and_pct_uses_previous_spend():
    rows = [
        row("a", D(2024, 12, 20), -10_000, "Shop", "Shopping"),
        row("b", D(2025, 1, 20), -15_000, "Shop", "Shopping"),
    ]
    grid = ref.monthly_category_spend(rows)
    assert grid[(D(2024, 12, 1), "Shopping")]["mom_change_cents"] is None
    jan = grid[(D(2025, 1, 1), "Shopping")]   # crosses a year boundary
    assert (jan["mom_change_cents"], jan["mom_change_pct"]) == (5_000, 0.5)


# ---------------------------------------------------------------- recurring charges


def monthly(merchant, amounts, day=10, start=(2025, 1), category="Subscriptions"):
    y, m = start
    out = []
    for i, cents in enumerate(amounts):
        out.append(row(f"{merchant}{i}", D(y + (m - 1 + i) // 12, (m - 1 + i) % 12 + 1, day),
                       cents, merchant, category))
    return out


def test_monthly_fixed_charge_is_recurring_and_stable():
    rec = ref.recurring_charges(monthly("Netflix", [-1_549] * 5))["Netflix"]
    assert (rec["cadence"], rec["occurrences"], rec["amount_stable"]) == ("monthly", 5, True)
    assert rec["typical_amount_cents"] == -1_549 and rec["is_income"] is False
    assert rec["last_date"] == D(2025, 5, 10)
    # gaps are 31, 28, 31, 30 days -> median 30.5, rounded half up to 31 -> 10 June
    assert rec["next_expected_date"] == D(2025, 6, 10)


def test_variable_monthly_bill_is_recurring_but_not_stable():
    rec = ref.recurring_charges(monthly("Utility", [-6_000, -14_000, -9_000, -11_000]))["Utility"]
    assert rec["cadence"] == "monthly" and rec["amount_stable"] is False


def test_two_occurrences_are_not_enough():
    assert ref.recurring_charges(monthly("Netflix", [-1_549] * 2)) == {}
    cfg = GoldConfig(recurring_min_occurrences=2)
    assert "Netflix" in ref.recurring_charges(monthly("Netflix", [-1_549] * 2), cfg)


def test_irregular_charges_are_not_recurring():
    days = [1, 4, 19, 22, 40, 41]
    rows = [row(f"x{i}", D(2025, 1, 1).fromordinal(D(2025, 1, 1).toordinal() + d), -500, "Cafe",
                "Dining") for i, d in enumerate(days)]
    assert ref.recurring_charges(rows) == {}


def test_same_day_duplicates_count_once_and_sum():
    rows = monthly("Gym", [-3_000] * 4)
    rows.append(row("dup", D(2025, 2, 10), -3_000, "Gym", "Health"))   # second charge same day
    rec = ref.recurring_charges(rows)["Gym"]
    assert rec["occurrences"] == 4   # still four distinct dates
    assert rec["typical_amount_cents"] == -3_000   # median of 4 daily totals: -3000,-6000,-3000,-3000


def test_biweekly_payroll_is_recurring_income():
    rows = []
    for i in range(6):
        rows.append(row(f"p{i}", D(2025, 1, 3).fromordinal(D(2025, 1, 3).toordinal() + 14 * i),
                        250_000, "Acme Payroll", "Income"))
    rec = ref.recurring_charges(rows)["Acme Payroll"]
    assert (rec["cadence"], rec["is_income"]) == ("biweekly", True)


def test_transfers_are_never_recurring_charges():
    rows = monthly("Savings Transfer", [-40_000] * 5, category="Transfer")
    assert ref.recurring_charges(rows) == {}


def test_half_up_rounds_away_from_zero():
    assert [ref.half_up(x) for x in (2.5, -2.5, 3.4, -3.4, 0.5)] == [3, -3, 3, -3, 1]


# ---------------------------------------------------------------- anomalies


def charges(merchant, amounts_cents, category="Shopping"):
    return [row(f"{merchant}{i}", D(2025, 1, 1).fromordinal(D(2025, 1, 1).toordinal() + 3 * i),
                -c, merchant, category) for i, c in enumerate(amounts_cents)]


NORMAL = [4_000, 4_400, 3_800, 4_200, 4_100, 3_900, 4_300, 4_000]


def test_large_spike_is_flagged_with_context():
    rows = charges("Target", NORMAL + [30_000])
    found = ref.anomalies(rows)
    assert list(found) == ["Target8"]
    assert found["Target8"]["amount_cents"] == -30_000
    assert found["Target8"]["typical_cents"] == 4_100
    assert round(found["Target8"]["ratio"], 2) == round(30_000 / 4_100, 2)


def test_moderate_increase_is_not_flagged_by_the_conservative_defaults():
    assert ref.anomalies(charges("Target", NORMAL + [7_000])) == {}


def test_small_dollar_jump_is_ignored_by_the_dollar_floor():
    rows = charges("Cafe", [300, 310, 295, 305, 300, 290, 315] + [900])   # 3x, but only $6 more
    assert ref.anomalies(rows) == {}


def test_merchants_with_few_charges_are_not_scored():
    assert ref.anomalies(charges("Rare", [4_000, 4_200, 3_900, 50_000])) == {}


def test_fixed_price_merchant_with_zero_mad_still_scores_sanely():
    rows = charges("Gym", [4_000] * 7 + [40_000])
    assert list(ref.anomalies(rows)) == ["Gym7"]
    assert ref.anomalies(charges("Gym", [4_000] * 8)) == {}   # no variation, no anomaly


def test_one_huge_charge_cannot_hide_itself():
    # With mean/std a 100x outlier inflates both and scores low; median/MAD keeps it obvious.
    rows = charges("Target", NORMAL + [400_000])
    assert "Target8" in ref.anomalies(rows)


def test_refunds_and_income_are_never_anomalies():
    rows = charges("Target", NORMAL) + [row("r", D(2025, 3, 1), 500_000, "Target", "Shopping")]
    assert ref.anomalies(rows) == {}


# ---------------------------------------------------------------- on the synthetic sample


def test_sample_data_yields_exactly_the_known_recurring_merchants(tmp_path):
    write_sample(tmp_path, seed=42)
    rows = sample_silver_rows(tmp_path)
    assert len(rows) == 369
    found = ref.recurring_charges(rows)
    assert set(found) == {
        "Apple", "Netflix", "Spotify", "Xfinity Internet", "Xcel Energy Autopay",
        "Zelle Payment To J Martinez", "Payroll Acme Corp Direct Dep",
    }
    assert {m for m, v in found.items() if not v["amount_stable"]} == {"Xcel Energy Autopay"}
    assert {m for m, v in found.items() if v["is_income"]} == {"Payroll Acme Corp Direct Dep"}
    assert ref.anomalies(rows) == {}   # uniformly random amounts: nothing should be flagged


def test_sample_data_totals_reconcile_to_silver(tmp_path):
    write_sample(tmp_path, seed=42)
    rows = sample_silver_rows(tmp_path)
    summary = ref.monthly_summary(rows)
    assert len(summary) == 6
    assert sum(
        m["income_cents"] - m["spend_cents"] + m["transfer_net_cents"] for m in summary.values()
    ) == sum(r["amount_cents"] for r in rows)
    assert sum(m["txn_count"] for m in summary.values()) == len(rows)

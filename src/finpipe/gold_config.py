"""Gold-layer settings (pure Python, no Spark) so tests and the Spark code share one source.

Defaults are deliberately conservative: a short, trustworthy list beats a long noisy one.
Tune them on real data, not on the synthetic sample (its amounts are uniformly random).
"""

from __future__ import annotations

from dataclasses import dataclass

# (name, center gap in days, tolerance in days). Bands must not overlap.
CADENCES: tuple[tuple[str, int, int], ...] = (
    ("weekly", 7, 2),
    ("biweekly", 14, 3),
    ("monthly", 30, 5),
    ("quarterly", 91, 10),
    ("yearly", 365, 15),
)


@dataclass(frozen=True)
class GoldConfig:
    # --- recurring charges ---
    recurring_min_occurrences: int = 3     # distinct charge dates (so at least 2 gaps)
    recurring_min_gap_share: float = 0.8   # share of gaps that sit within the cadence band
    stable_amount_cv: float = 0.10         # std/|mean| of per-day amounts at or below this
    # --- anomalies (charges far above a merchant's norm) ---
    anomaly_min_charges: int = 6           # merchants with fewer charges are not scored
    anomaly_min_z: float = 6.0             # robust z-score (typical textbook cutoff is 3.5)
    anomaly_min_ratio: float = 2.0         # and at least this multiple of the typical charge
    anomaly_min_excess_cents: int = 5_000  # and at least this much above it ($50)
    anomaly_scale_floor: float = 0.05      # scale never below 5% of the median (MAD can be 0)


MAD_TO_SIGMA = 1.4826  # scales MAD to a standard deviation for normally distributed data

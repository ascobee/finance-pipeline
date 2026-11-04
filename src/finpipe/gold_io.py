"""Read gold tables without Spark.

The dashboard and the summary command only need the small gold tables, so they read the
Parquet files directly with PyArrow instead of starting a JVM. Rows come back as plain dicts,
which keeps everything downstream (facts, summaries, dashboard prep) pure Python.
"""

from __future__ import annotations

from pathlib import Path

GOLD_TABLES = (
    "monthly_summary", "monthly_category_spend", "recurring_charges", "anomalies", "transactions",
)


def read_gold(warehouse_dir: str | Path) -> dict[str, list[dict]]:
    """Return ``{table_name: [row dicts]}`` for every gold table found in the warehouse."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Reading gold tables needs pyarrow. Install it with: pip install -e '.[dashboard]'"
        ) from exc

    gold_dir = Path(warehouse_dir) / "gold"
    out: dict[str, list[dict]] = {}
    for name in GOLD_TABLES:
        path = gold_dir / name
        if path.exists():
            out[name] = pq.read_table(str(path)).to_pylist()   # ignores _SUCCESS / .crc files
    return out

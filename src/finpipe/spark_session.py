"""Local SparkSession factory."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession

# Parent directory of the ``finpipe`` package. Python UDF workers are separate processes that
# unpickle our functions by module name, so they must be able to import ``finpipe`` too.
_PACKAGE_PARENT = str(Path(__file__).resolve().parent.parent)


def _expose_package_to_python_workers() -> None:
    paths = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if _PACKAGE_PARENT not in paths:
        os.environ["PYTHONPATH"] = os.pathsep.join([_PACKAGE_PARENT, *paths])
    # Make workers use the same interpreter (e.g. the active venv) as the driver.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)


def get_spark(app_name: str = "finance-pipeline", shuffle_partitions: int = 4) -> SparkSession:
    """Local session tuned for small data: few shuffle partitions, UI off, UTC dates."""
    _expose_package_to_python_workers()
    return (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

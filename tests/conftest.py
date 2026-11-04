import pytest


@pytest.fixture(scope="session")
def spark():
    pytest.importorskip("pyspark")
    from finpipe.spark_session import get_spark

    session = get_spark("finpipe-tests", shuffle_partitions=2)
    yield session
    session.stop()

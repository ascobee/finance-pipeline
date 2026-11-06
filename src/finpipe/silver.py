"""Silver layer: one common schema, validated, deduplicated, with stable IDs.

Sign convention in silver: ``amount_cents`` is negative for money out and positive for money
in, for every source. Banks disagree (the card export shows purchases as positive numbers),
so each parser is responsible for converting to this convention.

Identity and dedupe
-------------------
``txn_id`` = sha256(source | date | cents | description_key | occurrence). The occurrence index
counts identical rows *within one source file*, so:

* two identical coffees on the same day in one file -> occurrence 1 and 2 -> two rows;
* the same file loaded twice, or the same transaction in two overlapping exports -> the same
  IDs -> collapsed to one row (the row from the latest file name wins).

Computing the index across files instead would number the overlapping copies 1..4 and
quietly defeat the dedupe, which is the classic bug in this kind of pipeline.
"""

from __future__ import annotations

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from .common import existing_cols_from_list, one_col_exists
from .ids import ID_SEPARATOR
from .normalize import normalize_merchant

# Money moving between the user's own accounts is not spending.
TRANSFER_PATTERN = r"CREDIT CARD PAYMENT|PAYMENT THANK YOU|ONLINE TRANSFER"

PARSED_COLUMNS = [
    "source", "source_file", "txn_date", "amount_cents", "description_raw", "bank_category",
]

SILVER_COLUMNS = [
    "txn_id", "source", "txn_date", "amount_cents", "description_raw", "description_key",
    "merchant_norm", "bank_category", "is_transfer", "source_file",
]

DATE_COLUMNS = ["transaction_date", "processed_date", "date"]

_normalize_udf = F.udf(normalize_merchant, StringType(), useArrow=False)


def _cents(col: Column) -> Column:
    """Parse '$1,234.56' / '-12.30' into integer cents via decimal (never float)."""
    cleaned = F.regexp_replace(col, r"[$,\s]", "")
    return F.round(cleaned.try_cast("decimal(18,2)") * 100).cast("long")


def description_raw(desc_col_names: list[str], df_cols: list[str]) -> Column:
    desc_cols = existing_cols_from_list(desc_col_names, df_cols)
    if not desc_cols:
        raise ValueError(
            "Could not identify raw description column in bronze. "
            f"found: {', '.join(df_cols)}"
        )
    else:
        return F.coalesce(*desc_cols).alias("description_raw")


def description_key_col(col_name: str) -> Column:
    """Spark twin of ``ids.description_key``: trim, collapse whitespace, upper-case."""
    trimmed = F.regexp_replace(F.col(col_name), r"^\s+|\s+$", "")
    return F.upper(F.regexp_replace(trimmed, r"\s+", " "))


def txn_date_col(columns: list[str]) -> Column:
    date_col_names = ["transaction_date", "processed_date", "date"]
    date_cols = existing_cols_from_list(date_col_names, columns)
    if not date_cols:
        raise ValueError(
            "Could not identify date column in bronze. "
            f"found: {', '.join(columns)}"
        )
    date_formats = ["MM/dd/yyyy", "yyyy-MM-dd", "MM/dd/yy"]
    date_exprs = [F.try_to_date(F.coalesce(*date_cols), date_format) for date_format in date_formats]
    return F.coalesce(*date_exprs).alias("txn_date")


def parse_file(bronze: DataFrame, source_account: str) -> DataFrame:
    is_checking = source_account == "checking"
    bronze_cols = bronze.columns
    desc_col_names = ["description", "transaction_description"]
    category_col = F.col("category") if not is_checking and "category" in bronze_cols else F.lit(None).cast("string")

    return bronze.select(
        F.lit(source_account).alias("source"),
        F.col("_source_file").alias("source_file"),
        txn_date_col(bronze_cols),
        _amount_cents(bronze, is_checking).alias("amount_cents"),
        description_raw(desc_col_names, bronze_cols),
        category_col.alias("bank_category"),
    )


def _present(col_name: str) -> Column:
    """True when the cell holds something other than whitespace (NULL counts as blank)."""
    return F.coalesce(F.length(F.trim(F.col(col_name))) > 0, F.lit(False))


def _amount_cents(df: DataFrame, is_checking: bool) -> Column:
    columns = df.columns
    amt_col_names = ["amount", "transaction_amount"]
    type_col_names = ["credit_or_debit", "transaction_type"]
    layouts: list[Column] = []

    if one_col_exists(amt_col_names, columns):
        amt_col = F.coalesce(*existing_cols_from_list(amt_col_names, columns))
        amt = _cents(amt_col)
        if one_col_exists(type_col_names, columns):
            type_col = F.coalesce(*existing_cols_from_list(type_col_names, columns))
            layouts.append(
                F.when(type_col == "Debit", -amt)
                .otherwise(amt)
            )
        else:
            layouts.append(amt if is_checking else -amt)
    if {"debit", "credit"} <= set(columns):
        debit, credit = _cents(F.col("debit")), _cents(F.col("credit"))
        layouts.append(
            F.when(_present("debit") & ~_present("credit") & (debit >= 0), -debit)
            .when(_present("credit") & ~_present("debit") & (credit >= 0), credit)
        )
    if not layouts:
        raise ValueError(
            "credit card export has neither an 'amount' column nor 'debit'/'credit' columns; "
            f"found: {', '.join(columns)}"
        )
    return F.coalesce(*layouts)


def split_valid(parsed: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Separate rows that parsed cleanly from rows to quarantine (bad date/amount/text)."""
    ok = (
        F.col("txn_date").isNotNull()
        & F.col("amount_cents").isNotNull()
        & F.col("description_raw").isNotNull()
        & (F.length(F.trim(F.col("description_raw"))) > 0)
    )
    return parsed.filter(ok), parsed.filter(~ok)


def with_transaction_id(parsed: DataFrame) -> DataFrame:
    """Add description_key, occurrence (per source file), and the deterministic txn_id."""
    keyed = parsed.withColumn("description_key", description_key_col("description_raw"))
    identical_in_file = Window.partitionBy(
        "source", "source_file", "txn_date", "amount_cents", "description_key"
    ).orderBy(F.lit(1))  # rows in a group are identical, so the order is irrelevant
    with_occurrence = keyed.withColumn("occurrence", F.row_number().over(identical_in_file))
    return with_occurrence.withColumn(
        "txn_id",
        F.sha2(
            F.concat_ws(
                ID_SEPARATOR,
                F.col("source"),
                F.date_format("txn_date", "yyyy-MM-dd"),
                F.col("amount_cents").cast("string"),
                F.col("description_key"),
                F.col("occurrence").cast("string"),
            ),
            256,
        ),
    )


def build_silver(valid: DataFrame) -> DataFrame:
    """Valid parsed rows -> deduplicated silver transactions."""
    ided = with_transaction_id(valid)
    latest_file_wins = Window.partitionBy("txn_id").orderBy(F.col("source_file").desc())
    deduped = (
        ided.withColumn("_rn", F.row_number().over(latest_file_wins))
        .filter(F.col("_rn") == 1)
        .drop("_rn", "occurrence")
    )
    return deduped.select(
        "txn_id",
        "source",
        "txn_date",
        "amount_cents",
        "description_raw",
        "description_key",
        _normalize_udf(F.col("description_raw")).alias("merchant_norm"),
        "bank_category",
        F.col("description_key").rlike(TRANSFER_PATTERN).alias("is_transfer"),
        "source_file",
    )

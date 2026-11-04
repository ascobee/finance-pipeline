from __future__ import annotations

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F


def check_column_exists(col_name: str, df_cols: list[str]) -> bool:
    return col_name in df_cols

def one_col_exists(col_names: list[str], df_cols: list[str]) -> bool:
    return not set(col_names).isdisjoint(df_cols)

def all_cols_exist(col_names: list[str], df_cols: list[str]) -> bool:
    return set(col_names) <= set(df_cols)


# def existing_cols_from_list(col_names: list[str], df: DataFrame) -> list[Column]:
#     return [F.col(c) for c in col_names if c in df.columns]

def existing_cols_from_list(col_names: list[str], df_cols: list[str]) -> list[Column]:
    return [F.col(c) for c in col_names if c in df_cols]

def existing_col_names_from_list(col_names: list[str], df_cols: list[str]) -> list[str]:
    return [c for c in col_names if c in df_cols]


def first_real_col(col_names: list[str], df_cols: list[str]) -> str:
    return [c for c in col_names if c in df_cols][0]


def is_null_col(df: DataFrame, col_name: str) -> bool:
    return df.filter(F.col(col_name).isNotNull()).count() == 0


def target_col_name(df: DataFrame, col_names: list[str]) -> str:
    df_cols = df.columns
    existing_col_names = existing_col_names_from_list(col_names, df_cols)
    non_null_cols = [c for c in existing_col_names if not is_null_col(df, c)]
    return non_null_cols[0]


def target_col(df: DataFrame, col_names: list[str]) -> Column:
    df_cols = df.columns
    existing_col_names = existing_col_names_from_list(col_names, df_cols)
    non_null_cols = [c for c in existing_col_names if not is_null_col(df, c)]
    return F.col(non_null_cols[0])
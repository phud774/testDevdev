import json
from datetime import datetime
from pathlib import Path

import polars as pl

from .constants import (
    BILL_COL,
    BRAND_COL,
    CAT_COL,
    CAT_L1_COL,
    CAT_L2_COL,
    CAT_L3_COL,
    DATE_COL,
    DISCOUNT_COL,
    ITEM_CATALOG_PRICE_COL,
    ITEM_COL,
    ITEM_DESC_LEN_COL,
    LOC_COL,
    PRICE_COL,
    QTY_COL,
    SALE_STATUS_COL,
    USER_COL,
)
from .time_utils import MonthWindow


def load_target_users_from_ground_truth(path: Path) -> pl.DataFrame | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        ground_truth = json.load(f)
    users = [int(user_id) for user_id in ground_truth.keys()]
    return pl.DataFrame({USER_COL: users}).unique()


def scan_transactions(path: Path) -> pl.LazyFrame:
    return (
        pl.scan_parquet(path)
        .select(
            pl.col(BILL_COL).cast(pl.Int64),
            pl.col(USER_COL).cast(pl.Int64),
            pl.col(ITEM_COL).cast(pl.Utf8),
            pl.col(DATE_COL).cast(pl.Datetime),
            pl.col(PRICE_COL).cast(pl.Float64),
            pl.col(DISCOUNT_COL).cast(pl.Float64),
            pl.col(QTY_COL).cast(pl.Float64),
            pl.col(LOC_COL).cast(pl.Int64),
        )
        .filter(pl.col(USER_COL).is_not_null() & pl.col(ITEM_COL).is_not_null())
    )


def scan_items(path: Path) -> pl.LazyFrame | None:
    if not path.exists():
        return None
    return (
        pl.scan_parquet(path)
        .select(
            pl.col(ITEM_COL).cast(pl.Utf8),
            pl.col(PRICE_COL).cast(pl.Float64).alias(ITEM_CATALOG_PRICE_COL),
            pl.col(CAT_L1_COL).cast(pl.Utf8),
            pl.col(CAT_L2_COL).cast(pl.Utf8),
            pl.col(CAT_L3_COL).cast(pl.Utf8),
            pl.col(CAT_COL).cast(pl.Utf8),
            pl.col(BRAND_COL).cast(pl.Utf8),
            pl.col(SALE_STATUS_COL).cast(pl.Int8).alias("item_sale_status"),
            pl.col("description").cast(pl.Utf8).str.len_chars().alias(ITEM_DESC_LEN_COL),
        )
        .unique(subset=[ITEM_COL], keep="first")
    )


def target_users_for_month(
    lf: pl.LazyFrame,
    target: MonthWindow,
    max_users: int | None,
    ground_truth_path: Path | None = None,
) -> pl.DataFrame:
    if ground_truth_path is not None:
        users = load_target_users_from_ground_truth(ground_truth_path)
        if users is not None:
            return users.head(max_users) if max_users else users

    users = (
        lf.filter((pl.col(DATE_COL) >= target.start) & (pl.col(DATE_COL) < target.end))
        .select(USER_COL)
        .unique()
        .collect()
    )
    if max_users:
        users = users.sample(n=min(max_users, users.height), seed=42)
    return users


def users_before_cutoff(
    lf: pl.LazyFrame,
    cutoff: datetime,
    max_users: int | None = None,
) -> pl.DataFrame:
    users = (
        lf.filter(pl.col(DATE_COL) < cutoff)
        .select(USER_COL)
        .unique()
        .collect(engine="streaming")
    )
    if max_users:
        users = users.sample(n=min(max_users, users.height), seed=42)
    return users


def iter_user_chunks(users: pl.DataFrame, chunk_size: int):
    for start in range(0, users.height, chunk_size):
        yield start // chunk_size + 1, users.slice(start, chunk_size)


def history_before(lf: pl.LazyFrame, cutoff: datetime) -> pl.LazyFrame:
    return lf.filter(pl.col(DATE_COL) < cutoff)

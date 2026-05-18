from datetime import datetime

import polars as pl

from .constants import (
    BILL_COL,
    CANDIDATE_SOURCE_COLS,
    DATE_COL,
    ITEM_COL,
    LOC_COL,
    USER_COL,
)
from .data import history_before


def _mark_source(frame: pl.LazyFrame, source_col: str) -> pl.LazyFrame:
    return frame.select(
        USER_COL,
        ITEM_COL,
        *[pl.lit(1 if col == source_col else 0).alias(col) for col in CANDIDATE_SOURCE_COLS],
    )


def build_candidates(
    lf: pl.LazyFrame,
    target_users: pl.DataFrame,
    cutoff: datetime,
    personal_days: int,
    personal_top: int,
    global_top: int,
    location_top: int,
    recent_days: int,
    recent_global_top: int,
    recent_location_top: int,
) -> pl.DataFrame:
    hist = history_before(lf, cutoff)
    user_lf = target_users.lazy()

    recent_hist = hist.filter(pl.col(DATE_COL) >= pl.lit(cutoff).dt.offset_by(f"-{recent_days}d"))
    personal_hist = hist.filter(pl.col(DATE_COL) >= pl.lit(cutoff).dt.offset_by(f"-{personal_days}d"))

    personal = (
        personal_hist.join(user_lf, on=USER_COL, how="inner")
        .group_by([USER_COL, ITEM_COL])
        .agg(
            pl.len().alias("candidate_personal_count"),
            pl.max(DATE_COL).alias("candidate_personal_last_date"),
        )
        .sort(
            [USER_COL, "candidate_personal_count", "candidate_personal_last_date"],
            descending=[False, True, True],
        )
        .group_by(USER_COL)
        .head(personal_top)
    )
    personal = _mark_source(personal, "candidate_personal")

    repeat_all = (
        hist.join(user_lf, on=USER_COL, how="inner")
        .group_by([USER_COL, ITEM_COL])
        .agg(
            pl.len().alias("repeat_count"),
            pl.max(DATE_COL).alias("repeat_last_date"),
        )
        .sort([USER_COL, "repeat_count", "repeat_last_date"], descending=[False, True, True])
        .group_by(USER_COL)
        .head(personal_top)
    )
    repeat_all = _mark_source(repeat_all, "candidate_repeat_all")

    global_items = (
        hist.group_by(ITEM_COL)
        .agg(pl.len().alias("global_count"))
        .sort("global_count", descending=True)
        .head(global_top)
        .select(ITEM_COL)
    )
    global_candidates = _mark_source(user_lf.join(global_items, how="cross"), "candidate_global")

    recent_global_items = (
        recent_hist.group_by(ITEM_COL)
        .agg(pl.len().alias("recent_global_count"))
        .sort("recent_global_count", descending=True)
        .head(recent_global_top)
        .select(ITEM_COL)
    )
    recent_global_candidates = _mark_source(
        user_lf.join(recent_global_items, how="cross"),
        "candidate_recent_global",
    )

    user_location = (
        hist.join(user_lf, on=USER_COL, how="inner")
        .sort([USER_COL, DATE_COL])
        .group_by(USER_COL)
        .agg(pl.last(LOC_COL).alias(LOC_COL))
    )
    location_items = (
        hist.group_by([LOC_COL, ITEM_COL])
        .agg(pl.len().alias("location_count"))
        .with_columns(
            pl.col("location_count")
            .rank(method="ordinal", descending=True)
            .over(LOC_COL)
            .alias("location_rank")
        )
        .filter(pl.col("location_rank") <= location_top)
        .select(LOC_COL, ITEM_COL)
    )
    location_candidates = _mark_source(
        user_location.join(location_items, on=LOC_COL, how="inner"),
        "candidate_location",
    )

    recent_location_items = (
        recent_hist.group_by([LOC_COL, ITEM_COL])
        .agg(pl.len().alias("recent_location_count"))
        .with_columns(
            pl.col("recent_location_count")
            .rank(method="ordinal", descending=True)
            .over(LOC_COL)
            .alias("recent_location_rank")
        )
        .filter(pl.col("recent_location_rank") <= recent_location_top)
        .select(LOC_COL, ITEM_COL)
    )
    recent_location_candidates = _mark_source(
        user_location.join(recent_location_items, on=LOC_COL, how="inner"),
        "candidate_recent_location",
    )

    recent_user_items = (
        recent_hist.join(user_lf, on=USER_COL, how="inner")
        .group_by([USER_COL, ITEM_COL])
        .agg(pl.len().alias("anchor_count"), pl.max(DATE_COL).alias("anchor_last_date"))
        .sort([USER_COL, "anchor_count", "anchor_last_date"], descending=[False, True, True])
        .group_by(USER_COL)
        .head(min(personal_top, 40))
        .rename({ITEM_COL: "anchor_item"})
        .select(USER_COL, "anchor_item")
    )
    basket_items = personal_hist.select(BILL_COL, ITEM_COL).unique()
    anchor_items = recent_user_items.select(pl.col("anchor_item").alias(ITEM_COL)).unique()
    anchor_bills = basket_items.join(anchor_items, on=ITEM_COL, how="inner").rename(
        {ITEM_COL: "anchor_item"}
    )
    basket_co_items = basket_items.rename({ITEM_COL: "co_item"})
    co_items = (
        anchor_bills.join(basket_co_items, on=BILL_COL, how="inner")
        .filter(pl.col("anchor_item") != pl.col("co_item"))
        .group_by(["anchor_item", "co_item"])
        .agg(pl.len().alias("cobuy_count"))
        .with_columns(
            pl.col("cobuy_count")
            .rank(method="ordinal", descending=True)
            .over("anchor_item")
            .alias("cobuy_rank")
        )
        .filter(pl.col("cobuy_rank") <= max(recent_location_top, 20))
        .select("anchor_item", "co_item", "cobuy_count")
    )
    cobuy_candidates = (
        recent_user_items.join(co_items, on="anchor_item", how="inner")
        .group_by([USER_COL, "co_item"])
        .agg(pl.sum("cobuy_count").alias("cobuy_score"))
        .sort([USER_COL, "cobuy_score"], descending=[False, True])
        .group_by(USER_COL)
        .head(personal_top)
        .rename({"co_item": ITEM_COL})
    )
    cobuy_candidates = _mark_source(cobuy_candidates, "candidate_cobuy")

    return (
        pl.concat(
            [
                personal,
                repeat_all,
                global_candidates,
                location_candidates,
                recent_global_candidates,
                recent_location_candidates,
                cobuy_candidates,
            ]
        )
        .group_by([USER_COL, ITEM_COL])
        .agg(*[pl.max(col).alias(col) for col in CANDIDATE_SOURCE_COLS])
        .with_columns(
            sum(pl.col(col) for col in CANDIDATE_SOURCE_COLS).alias("candidate_source_count")
        )
        .collect(engine="streaming")
    )

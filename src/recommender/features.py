from datetime import datetime

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
    FEATURE_COLS,
    ITEM_COL,
    LOC_COL,
    PRICE_COL,
    QTY_COL,
    USER_COL,
)
from .data import history_before


def add_features(
    lf: pl.LazyFrame,
    candidates: pl.DataFrame,
    cutoff: datetime,
    item_lf: pl.LazyFrame | None = None,
) -> pl.DataFrame:
    hist = history_before(lf, cutoff)
    cand_lf = candidates.lazy()
    candidate_pairs = cand_lf.select([USER_COL, ITEM_COL])

    def recent(days: int) -> pl.LazyFrame:
        return hist.filter(pl.col(DATE_COL) >= pl.lit(cutoff).dt.offset_by(f"-{days}d"))

    spend = pl.col(PRICE_COL).fill_null(0) * pl.col(QTY_COL).fill_null(0)
    discount = pl.col(DISCOUNT_COL).fill_null(0)
    net_spend = pl.max_horizontal(spend - discount, pl.lit(0.0))

    pair_features = (
        hist.join(candidate_pairs, on=[USER_COL, ITEM_COL], how="inner")
        .group_by([USER_COL, ITEM_COL])
        .agg(
            pl.len().alias("ui_tx_count"),
            pl.sum(QTY_COL).alias("ui_qty_sum"),
            spend.sum().alias("ui_spend_sum"),
            discount.sum().alias("ui_discount_sum"),
            net_spend.sum().alias("ui_net_spend_sum"),
            pl.mean(PRICE_COL).alias("ui_avg_price"),
            pl.mean(DISCOUNT_COL).alias("ui_avg_discount"),
            pl.mean(QTY_COL).alias("ui_avg_qty"),
            ((pl.lit(cutoff) - pl.max(DATE_COL)).dt.total_days()).alias("ui_last_days"),
            ((pl.lit(cutoff) - pl.min(DATE_COL)).dt.total_days()).alias("ui_first_days"),
        )
        .with_columns(
            (pl.col("ui_discount_sum") / (pl.col("ui_spend_sum") + 1.0)).alias(
                "ui_discount_rate"
            ),
            (pl.col("ui_first_days") - pl.col("ui_last_days")).alias("ui_active_span_days"),
        )
    )

    pair_recent_7 = (
        recent(7)
        .join(candidate_pairs, on=[USER_COL, ITEM_COL], how="inner")
        .group_by([USER_COL, ITEM_COL])
        .agg(pl.len().alias("ui_tx_7d"))
    )

    pair_recent_30 = (
        recent(30)
        .join(candidate_pairs, on=[USER_COL, ITEM_COL], how="inner")
        .group_by([USER_COL, ITEM_COL])
        .agg(pl.len().alias("ui_tx_30d"))
    )
    pair_recent_90 = (
        recent(90)
        .join(candidate_pairs, on=[USER_COL, ITEM_COL], how="inner")
        .group_by([USER_COL, ITEM_COL])
        .agg(pl.len().alias("ui_tx_90d"), pl.sum(QTY_COL).alias("ui_qty_90d"))
    )

    user_features = (
        hist.join(cand_lf.select(USER_COL).unique(), on=USER_COL, how="inner")
        .group_by(USER_COL)
        .agg(
            pl.len().alias("u_tx_count"),
            pl.n_unique(BILL_COL).alias("u_bill_nunique"),
            pl.n_unique(ITEM_COL).alias("u_item_nunique"),
            pl.sum(QTY_COL).alias("u_qty_sum"),
            spend.sum().alias("u_spend_sum"),
            discount.sum().alias("u_discount_sum"),
            net_spend.sum().alias("u_net_spend_sum"),
            pl.mean(PRICE_COL).alias("u_avg_price"),
            pl.mean(DISCOUNT_COL).alias("u_avg_discount"),
            ((pl.lit(cutoff) - pl.max(DATE_COL)).dt.total_days()).alias("u_last_days"),
        )
        .with_columns(
            (pl.col("u_discount_sum") / (pl.col("u_spend_sum") + 1.0)).alias(
                "u_discount_rate"
            )
        )
    )
    user_recent_30 = (
        recent(30)
        .join(cand_lf.select(USER_COL).unique(), on=USER_COL, how="inner")
        .group_by(USER_COL)
        .agg(pl.len().alias("u_tx_30d"))
    )
    user_recent_90 = (
        recent(90)
        .join(cand_lf.select(USER_COL).unique(), on=USER_COL, how="inner")
        .group_by(USER_COL)
        .agg(pl.len().alias("u_tx_90d"))
    )

    item_features = (
        hist.join(cand_lf.select(ITEM_COL).unique(), on=ITEM_COL, how="inner")
        .group_by(ITEM_COL)
        .agg(
            pl.len().alias("i_tx_count"),
            pl.n_unique(USER_COL).alias("i_user_nunique"),
            pl.sum(QTY_COL).alias("i_qty_sum"),
        )
    )
    item_recent_30 = (
        recent(30)
        .join(cand_lf.select(ITEM_COL).unique(), on=ITEM_COL, how="inner")
        .group_by(ITEM_COL)
        .agg(pl.len().alias("i_tx_30d"))
    )
    item_recent_90 = (
        recent(90)
        .join(cand_lf.select(ITEM_COL).unique(), on=ITEM_COL, how="inner")
        .group_by(ITEM_COL)
        .agg(pl.len().alias("i_tx_90d"), pl.n_unique(USER_COL).alias("i_user_90d"))
    )

    if item_lf is not None:
        item_meta = item_lf.select(
            ITEM_COL,
            CAT_L1_COL,
            CAT_L2_COL,
            CAT_L3_COL,
            CAT_COL,
            BRAND_COL,
        )
        candidate_meta = cand_lf.select([USER_COL, ITEM_COL]).join(
            item_meta, on=ITEM_COL, how="left"
        )
        hist_meta = hist.join(
            item_meta.select([ITEM_COL, CAT_L1_COL, CAT_L2_COL, CAT_L3_COL, CAT_COL, BRAND_COL]),
            on=ITEM_COL,
            how="left",
        )
        recent_90_meta = recent(90).join(
            item_meta.select([ITEM_COL, CAT_L1_COL, BRAND_COL]),
            on=ITEM_COL,
            how="left",
        )

        user_cat_l1 = (
            hist_meta.join(
                candidate_meta.select([USER_COL, CAT_L1_COL]).drop_nulls().unique(),
                on=[USER_COL, CAT_L1_COL],
                how="inner",
            )
            .group_by([USER_COL, CAT_L1_COL])
            .agg(pl.len().alias("u_cat_l1_tx_count"))
        )
        user_cat_l2 = (
            hist_meta.join(
                candidate_meta.select([USER_COL, CAT_L2_COL]).drop_nulls().unique(),
                on=[USER_COL, CAT_L2_COL],
                how="inner",
            )
            .group_by([USER_COL, CAT_L2_COL])
            .agg(pl.len().alias("u_cat_l2_tx_count"))
        )
        user_cat_l3 = (
            hist_meta.join(
                candidate_meta.select([USER_COL, CAT_L3_COL]).drop_nulls().unique(),
                on=[USER_COL, CAT_L3_COL],
                how="inner",
            )
            .group_by([USER_COL, CAT_L3_COL])
            .agg(pl.len().alias("u_cat_l3_tx_count"))
        )
        user_category = (
            hist_meta.join(
                candidate_meta.select([USER_COL, CAT_COL]).drop_nulls().unique(),
                on=[USER_COL, CAT_COL],
                how="inner",
            )
            .group_by([USER_COL, CAT_COL])
            .agg(pl.len().alias("u_category_tx_count"))
        )
        user_brand = (
            hist_meta.join(
                candidate_meta.select([USER_COL, BRAND_COL]).drop_nulls().unique(),
                on=[USER_COL, BRAND_COL],
                how="inner",
            )
            .group_by([USER_COL, BRAND_COL])
            .agg(pl.len().alias("u_brand_tx_count"))
        )
        user_cat_l1_90 = (
            recent_90_meta.join(
                candidate_meta.select([USER_COL, CAT_L1_COL]).drop_nulls().unique(),
                on=[USER_COL, CAT_L1_COL],
                how="inner",
            )
            .group_by([USER_COL, CAT_L1_COL])
            .agg(pl.len().alias("u_cat_l1_tx_90d"))
        )
        user_brand_90 = (
            recent_90_meta.join(
                candidate_meta.select([USER_COL, BRAND_COL]).drop_nulls().unique(),
                on=[USER_COL, BRAND_COL],
                how="inner",
            )
            .group_by([USER_COL, BRAND_COL])
            .agg(pl.len().alias("u_brand_tx_90d"))
        )

    else:
        candidate_meta = cand_lf.select([USER_COL, ITEM_COL]).with_columns(
            pl.lit("").alias(CAT_L1_COL),
            pl.lit("").alias(CAT_L2_COL),
            pl.lit("").alias(CAT_L3_COL),
            pl.lit("").alias(CAT_COL),
            pl.lit("").alias(BRAND_COL),
        )
        user_cat_l1 = candidate_meta.select([USER_COL, CAT_L1_COL]).with_columns(
            pl.lit(0).alias("u_cat_l1_tx_count")
        )
        user_cat_l2 = candidate_meta.select([USER_COL, CAT_L2_COL]).with_columns(
            pl.lit(0).alias("u_cat_l2_tx_count")
        )
        user_cat_l3 = candidate_meta.select([USER_COL, CAT_L3_COL]).with_columns(
            pl.lit(0).alias("u_cat_l3_tx_count")
        )
        user_category = candidate_meta.select([USER_COL, CAT_COL]).with_columns(
            pl.lit(0).alias("u_category_tx_count")
        )
        user_brand = candidate_meta.select([USER_COL, BRAND_COL]).with_columns(
            pl.lit(0).alias("u_brand_tx_count")
        )
        user_cat_l1_90 = candidate_meta.select([USER_COL, CAT_L1_COL]).with_columns(
            pl.lit(0).alias("u_cat_l1_tx_90d")
        )
        user_brand_90 = candidate_meta.select([USER_COL, BRAND_COL]).with_columns(
            pl.lit(0).alias("u_brand_tx_90d")
        )

    user_location = (
        hist.join(cand_lf.select(USER_COL).unique(), on=USER_COL, how="inner")
        .sort([USER_COL, DATE_COL])
        .group_by(USER_COL)
        .agg(pl.last(LOC_COL).alias(LOC_COL))
    )
    candidate_with_location = cand_lf.select([USER_COL, ITEM_COL]).join(
        user_location, on=USER_COL, how="left"
    )
    loc_item_pairs = candidate_with_location.select([LOC_COL, ITEM_COL]).drop_nulls().unique()
    loc_item_features = (
        hist.join(loc_item_pairs, on=[LOC_COL, ITEM_COL], how="inner")
        .group_by([LOC_COL, ITEM_COL])
        .agg(
            pl.len().alias("loc_item_tx_count"),
            pl.n_unique(USER_COL).alias("loc_item_user_nunique"),
        )
    )
    loc_item_recent_30 = (
        recent(30)
        .join(loc_item_pairs, on=[LOC_COL, ITEM_COL], how="inner")
        .group_by([LOC_COL, ITEM_COL])
        .agg(pl.len().alias("loc_item_tx_30d"))
    )
    loc_item_recent_90 = (
        recent(90)
        .join(loc_item_pairs, on=[LOC_COL, ITEM_COL], how="inner")
        .group_by([LOC_COL, ITEM_COL])
        .agg(pl.len().alias("loc_item_tx_90d"))
    )

    out = (
        cand_lf.join(pair_features, on=[USER_COL, ITEM_COL], how="left")
        .join(pair_recent_7, on=[USER_COL, ITEM_COL], how="left")
        .join(pair_recent_30, on=[USER_COL, ITEM_COL], how="left")
        .join(pair_recent_90, on=[USER_COL, ITEM_COL], how="left")
        .join(user_features, on=USER_COL, how="left")
        .join(user_recent_30, on=USER_COL, how="left")
        .join(user_recent_90, on=USER_COL, how="left")
        .join(item_features, on=ITEM_COL, how="left")
        .join(item_recent_30, on=ITEM_COL, how="left")
        .join(item_recent_90, on=ITEM_COL, how="left")
        .join(candidate_meta, on=[USER_COL, ITEM_COL], how="left")
        .join(user_cat_l1, on=[USER_COL, CAT_L1_COL], how="left")
        .join(user_cat_l2, on=[USER_COL, CAT_L2_COL], how="left")
        .join(user_cat_l3, on=[USER_COL, CAT_L3_COL], how="left")
        .join(user_category, on=[USER_COL, CAT_COL], how="left")
        .join(user_brand, on=[USER_COL, BRAND_COL], how="left")
        .join(user_cat_l1_90, on=[USER_COL, CAT_L1_COL], how="left")
        .join(user_brand_90, on=[USER_COL, BRAND_COL], how="left")
        .join(candidate_with_location, on=[USER_COL, ITEM_COL], how="left")
        .join(loc_item_features, on=[LOC_COL, ITEM_COL], how="left")
        .join(loc_item_recent_30, on=[LOC_COL, ITEM_COL], how="left")
        .join(loc_item_recent_90, on=[LOC_COL, ITEM_COL], how="left")
        .with_columns(
            (pl.col("ui_tx_count") / (pl.col("u_tx_count") + 1.0)).alias("ui_share_of_user_tx"),
            (pl.col("ui_tx_count") / (pl.col("i_tx_count") + 1.0)).alias("ui_share_of_item_tx"),
            (pl.col("ui_qty_sum") / (pl.col("u_qty_sum") + 1.0)).alias("ui_share_of_user_qty"),
            (pl.col("ui_qty_sum") / (pl.col("i_qty_sum") + 1.0)).alias("ui_share_of_item_qty"),
            (pl.col("ui_avg_price") / (pl.col("u_avg_price") + 1.0)).alias(
                "user_item_price_ratio"
            ),
            (pl.col("u_cat_l1_tx_count") / (pl.col("u_tx_count") + 1.0)).alias(
                "u_cat_l1_share"
            ),
            (pl.col("u_brand_tx_count") / (pl.col("u_tx_count") + 1.0)).alias(
                "u_brand_share"
            ),
            (pl.col("loc_item_tx_90d") / (pl.col("loc_item_tx_count") + 1.0)).alias(
                "loc_item_recent_share"
            ),
            (pl.col("loc_item_tx_count") / (pl.col("i_tx_count") + 1.0)).alias(
                "loc_item_share_of_item_tx"
            ),
        )
        .with_columns([pl.col(col).fill_null(0) for col in FEATURE_COLS])
    )

    return out.select([USER_COL, ITEM_COL, *FEATURE_COLS]).collect(engine="streaming")

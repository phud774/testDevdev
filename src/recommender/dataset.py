import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from .candidates import build_candidates
from .constants import DATE_COL, FEATURE_COLS, ITEM_COL, USER_COL
from .data import iter_user_chunks, target_users_for_month
from .features import add_features
from .time_utils import MonthWindow, parse_month


def add_labels(lf: pl.LazyFrame, features: pl.DataFrame, target: MonthWindow) -> pl.DataFrame:
    positives = (
        lf.filter((pl.col(DATE_COL) >= target.start) & (pl.col(DATE_COL) < target.end))
        .select(USER_COL, ITEM_COL)
        .unique()
        .with_columns(pl.lit(1).alias("label"))
        .collect(engine="streaming")
    )
    return (
        features.join(positives, on=[USER_COL, ITEM_COL], how="left")
        .with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
    )


def downsample_negatives(df: pd.DataFrame, negative_ratio: float, seed: int) -> pd.DataFrame:
    positives = df[df["label"] == 1]
    negatives = df[df["label"] == 0]
    if positives.empty or negatives.empty:
        return df

    n_neg = min(len(negatives), int(len(positives) * negative_ratio))
    sampled_negatives = negatives.sample(n=n_neg, random_state=seed)
    return (
        pd.concat([positives, sampled_negatives], ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def build_dataset_for_users(
    lf: pl.LazyFrame,
    target: MonthWindow,
    users: pl.DataFrame,
    args: argparse.Namespace,
    item_lf: pl.LazyFrame | None = None,
) -> pd.DataFrame:
    candidates = build_candidates(
        lf,
        users,
        cutoff=target.start,
        personal_days=args.history_days,
        personal_top=args.candidate_top,
        global_top=args.popular_top,
        location_top=args.location_top,
        recent_days=args.recent_days,
        recent_global_top=args.popular_top,
        recent_location_top=args.location_top,
        cobuy_top=args.cobuy_top,
        category_top=args.category_top,
        brand_top=args.brand_top,
        item_lf=item_lf,
        cache_dir=None if args.no_candidate_cache else Path(args.candidate_cache_dir),
        refresh_cache=args.refresh_candidate_cache,
    )
    features = add_features(lf, candidates, cutoff=target.start, item_lf=item_lf)
    labeled = add_labels(lf, features, target)
    pdf = labeled.to_pandas()
    pdf[FEATURE_COLS] = pdf[FEATURE_COLS].astype(np.float32)
    pdf["label"] = pdf["label"].astype(np.int8)
    return pdf


def build_training_dataset_for_month(
    lf: pl.LazyFrame,
    target_month: str,
    max_users: int | None,
    args: argparse.Namespace,
    seed: int,
    downsample: bool = True,
    item_lf: pl.LazyFrame | None = None,
) -> pd.DataFrame:
    target = parse_month(target_month)
    users = target_users_for_month(lf, target, max_users)
    parts = []
    total_rows = 0
    total_pos = 0
    n_chunks = (users.height + args.user_chunk_size - 1) // args.user_chunk_size

    print(f"\nBuilding training dataset for {target_month}")
    print(f"Target users: {users.height:,} | chunk size: {args.user_chunk_size:,}")
    for chunk_id, user_chunk in iter_user_chunks(users, args.user_chunk_size):
        chunk = build_dataset_for_users(lf, target, user_chunk, args, item_lf=item_lf)
        if chunk.empty:
            print(f"  chunk {chunk_id:,}/{n_chunks:,}: rows=0, skipped")
            continue

        total_rows += len(chunk)
        total_pos += int(chunk["label"].sum())
        sampled = (
            downsample_negatives(chunk, args.negative_ratio, seed=seed + chunk_id)
            if downsample
            else chunk
        )
        parts.append(sampled)
        print(
            f"  chunk {chunk_id:,}/{n_chunks:,}: users={user_chunk.height:,}, "
            f"rows={len(chunk):,}, positives={int(chunk['label'].sum()):,}, "
            f"kept={len(sampled):,}"
        )
        del chunk

    if not parts:
        print(f"Month {target_month}: no training rows, skipped")
        return pd.DataFrame(columns=[USER_COL, ITEM_COL, *FEATURE_COLS, "label"])

    out = pd.concat(parts, ignore_index=True)
    print(
        f"Month {target_month}: scanned rows={total_rows:,}, positives={total_pos:,}, "
        f"training rows={len(out):,}"
    )
    return out

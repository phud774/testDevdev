import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import log_loss

from .candidates import build_candidates
from .constants import DATE_COL, FEATURE_COLS, ITEM_COL, USER_COL
from .data import iter_user_chunks, target_users_for_month
from .dataset import add_labels, build_dataset_for_users
from .evaluation import CandidateCoverageStats, load_candidate_coverage_stats, precision_at_10_from_scores
from .features import add_features
from .time_utils import parse_month


def top_k_from_scored(df: pd.DataFrame, k: int = 10) -> dict[str, list[str]]:
    scored = df.sort_values([USER_COL, "score"], ascending=[True, False])
    top = scored.groupby(USER_COL, sort=False).head(k)
    return {
        str(user_id): group[ITEM_COL].astype(str).tolist()
        for user_id, group in top.groupby(USER_COL, sort=False)
    }


def build_scored_chunk(
    model,
    lf: pl.LazyFrame,
    user_chunk: pl.DataFrame,
    target_month: str,
    args: argparse.Namespace,
    item_lf: pl.LazyFrame | None = None,
) -> pd.DataFrame:
    target = parse_month(target_month)
    candidates = build_candidates(
        lf,
        user_chunk,
        cutoff=target.start,
        personal_days=args.history_days,
        personal_top=args.candidate_top,
        global_top=args.popular_top,
        location_top=args.location_top,
        recent_days=args.recent_days,
        recent_global_top=args.popular_top,
        recent_location_top=args.location_top,
    )
    features = add_features(lf, candidates, cutoff=target.start, item_lf=item_lf)
    pdf = features.to_pandas()
    pdf[FEATURE_COLS] = pdf[FEATURE_COLS].astype(np.float32)
    pdf["score"] = model.predict_proba(pdf[FEATURE_COLS])[:, 1]
    return pdf


def predict_month_chunked(
    model,
    lf: pl.LazyFrame,
    target_month: str,
    args: argparse.Namespace,
    output_path: Path,
    ground_truth_path: Path | None = None,
    item_lf: pl.LazyFrame | None = None,
) -> dict[str, list[str]]:
    target = parse_month(target_month)
    use_ground_truth_users = (
        getattr(args, "use_ground_truth_users", False)
        or getattr(args, "test_users_from_ground_truth", False)
    )
    users = target_users_for_month(
        lf,
        target,
        args.max_test_users,
        ground_truth_path=ground_truth_path if use_ground_truth_users else None,
    )
    n_chunks = (users.height + args.user_chunk_size - 1) // args.user_chunk_size
    submission: dict[str, list[str]] = {}
    coverage_stats = (
        load_candidate_coverage_stats(ground_truth_path)
        if use_ground_truth_users or getattr(args, "eval_ground_truth", False)
        else None
    )

    print(f"\nPredicting {target_month} by chunks")
    print(f"Target users: {users.height:,} | chunk size: {args.user_chunk_size:,}")
    for chunk_id, user_chunk in iter_user_chunks(users, args.user_chunk_size):
        scored = build_scored_chunk(model, lf, user_chunk, target_month, args, item_lf=item_lf)
        if coverage_stats is not None:
            coverage_stats.update_from_frame(scored[[USER_COL, ITEM_COL]])
        submission.update(top_k_from_scored(scored, k=10))
        print(
            f"  chunk {chunk_id:,}/{n_chunks:,}: users={user_chunk.height:,}, "
            f"candidates={len(scored):,}, saved_users={len(submission):,}"
        )
        del scored

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False)
    if coverage_stats is not None:
        coverage_stats.print_summary()
    return submission


def validation_truth_for_users(
    lf: pl.LazyFrame,
    user_chunk: pl.DataFrame,
    target_month: str,
) -> dict[str, list[str]]:
    target = parse_month(target_month)
    truth = (
        lf.filter((pl.col(DATE_COL) >= target.start) & (pl.col(DATE_COL) < target.end))
        .join(user_chunk.lazy(), on=USER_COL, how="inner")
        .select(USER_COL, ITEM_COL)
        .unique()
        .collect(engine="streaming")
        .to_pandas()
    )
    return {
        str(user_id): group[ITEM_COL].astype(str).tolist()
        for user_id, group in truth.groupby(USER_COL, sort=False)
    }


def evaluate_month_chunked(
    model,
    lf: pl.LazyFrame,
    target_month: str,
    args: argparse.Namespace,
    output_path: Path,
    item_lf: pl.LazyFrame | None = None,
) -> None:
    target = parse_month(target_month)
    users = target_users_for_month(lf, target, args.max_val_users)
    n_chunks = (users.height + args.user_chunk_size - 1) // args.user_chunk_size
    precisions = []
    losses = []
    submission: dict[str, list[str]] = {}
    coverage_stats = CandidateCoverageStats(ground_truth={})

    print(f"\nValidation for {target_month}")
    print(f"Target users: {users.height:,} | chunk size: {args.user_chunk_size:,}")
    for chunk_id, user_chunk in iter_user_chunks(users, args.user_chunk_size):
        candidates = build_candidates(
            lf,
            user_chunk,
            cutoff=target.start,
            personal_days=args.history_days,
            personal_top=args.candidate_top,
            global_top=args.popular_top,
            location_top=args.location_top,
            recent_days=args.recent_days,
            recent_global_top=args.popular_top,
            recent_location_top=args.location_top,
        )
        coverage_stats.ground_truth.update(validation_truth_for_users(lf, user_chunk, target_month))
        coverage_stats.update_from_frame(candidates[[USER_COL, ITEM_COL]].to_pandas())

        features = add_features(lf, candidates, cutoff=target.start, item_lf=item_lf)
        labeled = add_labels(lf, features, target)
        chunk = labeled.to_pandas()
        chunk[FEATURE_COLS] = chunk[FEATURE_COLS].astype(np.float32)
        chunk["score"] = model.predict_proba(chunk[FEATURE_COLS])[:, 1]
        precisions.append(precision_at_10_from_scores(chunk))
        losses.append(log_loss(chunk["label"], np.clip(chunk["score"], 1e-6, 1 - 1e-6)))
        submission.update(top_k_from_scored(chunk, k=10))
        print(
            f"  chunk {chunk_id:,}/{n_chunks:,}: users={user_chunk.height:,}, "
            f"rows={len(chunk):,}, precision@10={precisions[-1]:.6f}"
        )
        del chunk

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False)
    coverage_stats.print_summary()
    print("\nValidation metrics")
    print(f"Mean chunk log loss:      {float(np.mean(losses)):.6f}")
    print(f"Mean chunk Precision@10:  {float(np.mean(precisions)):.6f}")

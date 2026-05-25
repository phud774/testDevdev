import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from recommender.constants import FEATURE_COLS, ITEM_COL, USER_COL
from recommender.data import scan_items, scan_transactions
from recommender.dataset import build_training_dataset_for_month


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure feature-label association on monthly recommendation datasets."
    )
    parser.add_argument("--data", default="data/transaction_full_2025.parquet")
    parser.add_argument("--items", default="data/items.parquet")
    parser.add_argument(
        "--months",
        default="2025-10",
        help="Comma-separated label months to analyze, for example: 2025-09,2025-10.",
    )
    parser.add_argument("--output", default="feature_correlation.csv")
    parser.add_argument("--max-users", type=int, default=20_000)
    parser.add_argument("--user-chunk-size", type=int, default=5_000)

    parser.add_argument("--history-days", type=int, default=270)
    parser.add_argument("--recent-days", type=int, default=90)
    parser.add_argument("--candidate-top", type=int, default=80)
    parser.add_argument("--popular-top", type=int, default=120)
    parser.add_argument("--location-top", type=int, default=50)
    parser.add_argument(
        "--cobuy-top",
        type=int,
        default=20,
        help="Co-buy items kept per anchor item. Use 0 to disable this memory-heavy source.",
    )
    parser.add_argument("--candidate-cache-dir", default="cache/candidates")
    parser.add_argument("--no-candidate-cache", action="store_true")
    parser.add_argument("--refresh-candidate-cache", action="store_true")
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=6.0,
        help="Negative samples kept per positive. Use a large value to keep more negatives.",
    )
    parser.add_argument(
        "--sort-by",
        default="abs_spearman",
        choices=["abs_pearson", "abs_spearman", "auc_lift", "ap_lift", "positive_rate_lift"],
    )
    parser.add_argument(
        "--min-nonzero-rate",
        type=float,
        default=0.001,
        help="Minimum non-zero rate used to flag extremely sparse features.",
    )
    parser.add_argument(
        "--min-abs-corr",
        type=float,
        default=0.01,
        help="Minimum absolute Pearson/Spearman correlation for a feature to be treated as usable.",
    )
    parser.add_argument(
        "--min-auc-lift",
        type=float,
        default=0.01,
        help="Minimum directionless AUC lift over 0.5 for a feature to be treated as usable.",
    )
    parser.add_argument(
        "--min-positive-rate-lift",
        type=float,
        default=0.005,
        help="Minimum positive-rate lift when the feature is non-zero.",
    )
    return parser.parse_args()


def safe_corr(feature: pd.Series, label: pd.Series, method: str) -> float:
    if feature.nunique(dropna=False) <= 1 or label.nunique(dropna=False) <= 1:
        return 0.0
    value = feature.corr(label, method=method)
    return 0.0 if pd.isna(value) else float(value)


def safe_auc(feature: pd.Series, label: pd.Series) -> float:
    if feature.nunique(dropna=False) <= 1 or label.nunique(dropna=False) <= 1:
        return 0.5
    try:
        auc = float(roc_auc_score(label, feature))
    except ValueError:
        return 0.5
    return max(auc, 1.0 - auc)


def safe_average_precision(feature: pd.Series, label: pd.Series) -> float:
    if feature.nunique(dropna=False) <= 1 or label.nunique(dropna=False) <= 1:
        return 0.0
    try:
        return float(average_precision_score(label, feature))
    except ValueError:
        return 0.0


def summarize_feature(df: pd.DataFrame, feature_name: str) -> dict[str, float | int | str]:
    label = df["label"].astype(np.float32)
    if feature_name not in df.columns:
        return {
            "feature": feature_name,
            "pearson": 0.0,
            "spearman": 0.0,
            "abs_pearson": 0.0,
            "abs_spearman": 0.0,
            "auc_directionless": 0.5,
            "auc_lift": 0.0,
            "average_precision": 0.0,
            "ap_lift": 0.0,
            "positive_rate_when_nonzero": 0.0,
            "global_positive_rate": float(label.mean()) if len(label) else 0.0,
            "positive_rate_lift": 0.0,
            "mean_positive": 0.0,
            "mean_negative": 0.0,
            "mean_delta": 0.0,
            "null_rate": 1.0,
            "nonzero_rate": 0.0,
            "unique_values": 0,
        }

    raw_feature = df[feature_name]
    null_rate = float(raw_feature.isna().mean()) if len(raw_feature) else 0.0
    feature = raw_feature.fillna(0).astype(np.float32)
    positive_mask = label == 1
    negative_mask = label == 0
    nonzero_mask = feature != 0

    global_positive_rate = float(label.mean()) if len(label) else 0.0
    feature_positive_rate = (
        float(label[nonzero_mask].mean()) if int(nonzero_mask.sum()) > 0 else 0.0
    )
    mean_pos = float(feature[positive_mask].mean()) if int(positive_mask.sum()) > 0 else 0.0
    mean_neg = float(feature[negative_mask].mean()) if int(negative_mask.sum()) > 0 else 0.0
    ap = safe_average_precision(feature, label)

    return {
        "feature": feature_name,
        "pearson": safe_corr(feature, label, "pearson"),
        "spearman": safe_corr(feature, label, "spearman"),
        "abs_pearson": abs(safe_corr(feature, label, "pearson")),
        "abs_spearman": abs(safe_corr(feature, label, "spearman")),
        "auc_directionless": safe_auc(feature, label),
        "auc_lift": safe_auc(feature, label) - 0.5,
        "average_precision": ap,
        "ap_lift": ap - global_positive_rate,
        "positive_rate_when_nonzero": feature_positive_rate,
        "global_positive_rate": global_positive_rate,
        "positive_rate_lift": feature_positive_rate - global_positive_rate,
        "mean_positive": mean_pos,
        "mean_negative": mean_neg,
        "mean_delta": mean_pos - mean_neg,
        "null_rate": null_rate,
        "nonzero_rate": float(nonzero_mask.mean()) if len(feature) else 0.0,
        "unique_values": int(feature.nunique(dropna=False)),
    }


def classify_feature(row: pd.Series, args: argparse.Namespace) -> tuple[str, str]:
    if row["unique_values"] == 0:
        return "missing", "column was not produced by the pipeline"
    if row["unique_values"] <= 1:
        return "constant", "only one value after filling nulls"
    if row["nonzero_rate"] == 0:
        return "all_zero", "feature is always zero"

    has_corr_signal = max(row["abs_pearson"], row["abs_spearman"]) >= args.min_abs_corr
    has_auc_signal = row["auc_lift"] >= args.min_auc_lift
    has_nonzero_signal = abs(row["positive_rate_lift"]) >= args.min_positive_rate_lift
    has_ap_signal = row["ap_lift"] > 0

    if has_corr_signal or has_auc_signal or has_nonzero_signal or has_ap_signal:
        if row["nonzero_rate"] < args.min_nonzero_rate:
            return "usable_sparse", "has label signal but appears in very few rows"
        return "usable", "passes at least one signal threshold"

    if row["nonzero_rate"] < args.min_nonzero_rate:
        return "too_sparse", "too sparse and no clear label signal"
    return "weak_signal", "varies but does not pass the signal thresholds"


def summarize_all_features(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = [summarize_feature(df, feature) for feature in FEATURE_COLS]
    result = pd.DataFrame(rows)
    labels = result.apply(lambda row: classify_feature(row, args), axis=1, result_type="expand")
    result["status"] = labels[0]
    result["reason"] = labels[1]
    return result.sort_values(args.sort_by, ascending=False)


def build_analysis_frame(lf, months: list[str], args: argparse.Namespace, item_lf=None) -> pd.DataFrame:
    parts = []
    for month_id, month in enumerate(months):
        month_df = build_training_dataset_for_month(
            lf,
            month,
            args.max_users,
            args,
            seed=2026 + month_id * 10_000,
            item_lf=item_lf,
        )
        if not month_df.empty:
            month_df["label_month"] = month
            parts.append(month_df)

    if not parts:
        raise ValueError("No rows were created. Try a different month or increase --max-users.")
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    args = parse_args()
    months = [month.strip() for month in args.months.split(",") if month.strip()]
    lf = scan_transactions(Path(args.data))
    item_lf = scan_items(Path(args.items))
    df = build_analysis_frame(lf, months, args, item_lf=item_lf)

    result = summarize_all_features(df, args)
    result.to_csv(args.output, index=False)
    usable = result[result["status"].isin(["usable", "usable_sparse"])]
    unusable = result[~result["status"].isin(["usable", "usable_sparse"])]

    print("\nFeature-label association")
    print(f"Rows: {len(df):,}")
    print(f"Users: {df[USER_COL].nunique():,}")
    print(f"Items: {df[ITEM_COL].nunique():,}")
    print(f"Positives: {int(df['label'].sum()):,}")
    print(f"Positive rate: {float(df['label'].mean()):.6f}")
    print(f"Features checked: {len(result):,}")
    print(f"Usable features:  {len(usable):,}")
    print(f"Weak/unusable:    {len(unusable):,}")
    print(f"Saved: {args.output}")

    print("\nFeature status counts")
    print(result["status"].value_counts().to_string())

    print("\nTop usable features")
    display_cols = [
        "feature",
        "status",
        "spearman",
        "pearson",
        "auc_lift",
        "ap_lift",
        "positive_rate_lift",
        "nonzero_rate",
        "unique_values",
    ]
    if usable.empty:
        print("No feature passed the current thresholds.")
    else:
        print(usable[display_cols].head(30).to_string(index=False))

    print("\nFeatures to inspect/drop")
    inspect_cols = ["feature", "status", "reason", "nonzero_rate", "unique_values"]
    if unusable.empty:
        print("No weak or unusable features under the current thresholds.")
    else:
        print(unusable[inspect_cols].head(50).to_string(index=False))


if __name__ == "__main__":
    main()

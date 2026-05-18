import argparse
from pathlib import Path

import pandas as pd

from recommender.data import scan_items, scan_transactions
from recommender.dataset import build_training_dataset_for_month
from recommender.evaluation import evaluate_submission_dict
from recommender.inference import evaluate_month_chunked, predict_month_chunked
from recommender.model import train_xgboost
from recommender.time_utils import month_minus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature engineering + XGBoost training pipeline for monthly item recommendation."
    )
    parser.add_argument("--data", default="data/transaction_full_2025.parquet")
    parser.add_argument("--items", default="data/items.parquet")
    parser.add_argument("--ground-truth", default="ground_truth.json")
    parser.add_argument("--submission", default="submission_xgb.json")
    parser.add_argument("--test-month", default="2025-11")
    parser.add_argument("--val-month", default="2025-10")
    parser.add_argument(
        "--train-months",
        default=None,
        help="Comma-separated label months. Default: two months before validation.",
    )
    parser.add_argument("--max-train-users", type=int, default=None)
    parser.add_argument("--max-val-users", type=int, default=None)
    parser.add_argument("--max-test-users", type=int, default=None)
    parser.add_argument("--user-chunk-size", type=int, default=5_000)

    parser.add_argument("--history-days", type=int, default=270)
    parser.add_argument("--recent-days", type=int, default=90)
    parser.add_argument("--candidate-top", type=int, default=80)
    parser.add_argument("--popular-top", type=int, default=60)
    parser.add_argument("--location-top", type=int, default=50)

    parser.add_argument("--negative-ratio", type=float, default=6.0)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=0)
    parser.add_argument(
        "--use-ground-truth-users",
        action="store_true",
        help="Use users from --ground-truth during test prediction.",
    )
    parser.add_argument(
        "--test-users-from-ground-truth",
        action="store_true",
        help="Use users from --ground-truth during test prediction.",
    )
    parser.add_argument(
        "--eval-ground-truth",
        action="store_true",
        help="Evaluate final submission against --ground-truth. Use only when it matches --test-month.",
    )
    parser.add_argument(
        "--no-ground-truth-users",
        action="store_true",
        help="Deprecated alias kept for old commands; ground-truth users are off by default.",
    )
    parser.add_argument(
        "--evaluate-test-month",
        action="store_true",
        help="Evaluate the final model on --test-month labels from the transaction parquet.",
    )
    parser.add_argument(
        "--no-refit-on-validation",
        action="store_true",
        help="Use the validation-trained model for test prediction instead of refitting on train + validation.",
    )
    return parser.parse_args()


def resolve_train_months(args: argparse.Namespace) -> list[str]:
    if args.train_months:
        return [month.strip() for month in args.train_months.split(",") if month.strip()]
    return [month_minus(args.val_month, 2), month_minus(args.val_month, 1)]


def build_train_frame(lf, train_months: list[str], args: argparse.Namespace, item_lf=None) -> pd.DataFrame:
    train_parts = []
    for i, month in enumerate(train_months):
        part = build_training_dataset_for_month(
            lf,
            month,
            args.max_train_users,
            args,
            seed=42 + i * 10_000,
            item_lf=item_lf,
        )
        if not part.empty:
            train_parts.append(part)

    if not train_parts:
        raise ValueError("No training rows were created. Use months with history before their cutoff.")

    train_df = pd.concat(train_parts, ignore_index=True)
    print(
        f"\nTraining rows: {len(train_df):,} | "
        f"positives: {int(train_df['label'].sum()):,}"
    )
    return train_df


def best_tree_count(model, default_n_estimators: int) -> int:
    if hasattr(model, "best_iteration") and model.best_iteration is not None:
        return int(model.best_iteration) + 1
    return default_n_estimators


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    gt_path = Path(args.ground_truth)
    submission_path = Path(args.submission)

    lf = scan_transactions(data_path)
    item_lf = scan_items(Path(args.items))
    if item_lf is not None:
        print(f"Using item metadata: {args.items}")
    else:
        print(f"Item metadata not found, continuing without item features: {args.items}")

    train_df = build_train_frame(lf, resolve_train_months(args), args, item_lf=item_lf)
    val_df = build_training_dataset_for_month(
        lf,
        args.val_month,
        args.max_val_users,
        args,
        seed=99,
        downsample=False,
        item_lf=item_lf,
    )

    model = train_xgboost(train_df, val_df, args)
    evaluate_month_chunked(
        model,
        lf,
        args.val_month,
        args,
        Path(f"validation_{args.val_month}_submission.json"),
        item_lf=item_lf,
    )

    if args.no_refit_on_validation:
        print("\nSkipping final refit. Using validation-trained model for prediction.")
        final_model = model
    else:
        n_estimators = best_tree_count(model, args.n_estimators)
        final_train = pd.concat([train_df, val_df], ignore_index=True)
        print(
            f"\nRefitting on train + validation rows: {len(final_train):,} "
            f"with n_estimators={n_estimators}"
        )
        final_model = train_xgboost(
            final_train,
            None,
            args,
            n_estimators=n_estimators,
            early_stopping=False,
        )

    submission = predict_month_chunked(
        final_model,
        lf,
        args.test_month,
        args,
        submission_path,
        ground_truth_path=gt_path if gt_path.exists() else None,
        item_lf=item_lf,
    )
    print(f"\nSaved {len(submission):,} users to {submission_path}")
    if args.evaluate_test_month:
        evaluate_month_chunked(
            final_model,
            lf,
            args.test_month,
            args,
            Path(f"test_{args.test_month}_submission.json"),
            item_lf=item_lf,
        )
    if args.eval_ground_truth:
        evaluate_submission_dict(gt_path, submission, k=10)
    else:
        print(
            "\nSkipped ground-truth evaluation. Pass --eval-ground-truth only when "
            "--ground-truth matches --test-month."
        )


if __name__ == "__main__":
    main()

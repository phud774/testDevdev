import argparse

import numpy as np
import pandas as pd

from .constants import FEATURE_COLS, USER_COL


def xgboost_major_version(version: str) -> int:
    try:
        return int(version.split(".", 1)[0])
    except (ValueError, IndexError):
        return 0


def uses_ranking_objective(args: argparse.Namespace) -> bool:
    return getattr(args, "xgb_objective", "binary") != "binary"


def sort_by_user(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(USER_COL, kind="mergesort").reset_index(drop=True)


def group_sizes(df: pd.DataFrame) -> np.ndarray:
    return df.groupby(USER_COL, sort=False).size().to_numpy(dtype=np.uint32)


def keep_groups_with_positive_label(df: pd.DataFrame, name: str) -> pd.DataFrame:
    positive_by_user = df.groupby(USER_COL, sort=False)["label"].transform("sum") > 0
    filtered = df[positive_by_user].copy()
    dropped_rows = len(df) - len(filtered)
    dropped_groups = df.loc[~positive_by_user, USER_COL].nunique()
    if dropped_rows:
        print(
            f"{name}: dropped {dropped_rows:,} rows from {dropped_groups:,} "
            "all-negative ranking groups"
        )
    if filtered.empty:
        raise ValueError(f"{name}: no ranking groups with positive labels were created.")
    return filtered


def prepare_rank_frame(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, np.ndarray]:
    ranked = sort_by_user(keep_groups_with_positive_label(df, name))
    return ranked, group_sizes(ranked)


def train_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    args: argparse.Namespace,
    n_estimators: int | None = None,
    early_stopping: bool = True,
):
    try:
        import xgboost as xgb
        from xgboost import XGBClassifier, XGBRanker
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: xgboost. Install it with `pip install xgboost` "
            "or run `pip install -r requirements.txt`."
        ) from exc

    train_pos = max(int(train_df["label"].sum()), 1)
    train_neg = max(len(train_df) - train_pos, 1)
    use_eval = val_df is not None and not val_df.empty
    use_ranker = uses_ranking_objective(args)
    xgb_objective = getattr(args, "xgb_objective", "binary")

    model_kwargs = {
        "n_estimators": n_estimators or args.n_estimators,
        "max_depth": 5,
        "learning_rate": 0.06,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 10.0,
        "reg_lambda": 2.0,
        "tree_method": "hist",
        "n_jobs": args.n_jobs,
        "random_state": 42,
    }
    if use_ranker:
        model_kwargs["objective"] = (
            "rank:pairwise" if xgb_objective == "rank_pairwise" else "rank:ndcg"
        )
        model_kwargs["eval_metric"] = "ndcg@10"
    else:
        model_kwargs["objective"] = "binary:logistic"
        model_kwargs["eval_metric"] = "aucpr"
        model_kwargs["scale_pos_weight"] = train_neg / train_pos

    if args.xgb_device == "cuda":
        if xgboost_major_version(xgb.__version__) >= 2:
            model_kwargs["device"] = "cuda"
        else:
            model_kwargs["tree_method"] = "gpu_hist"
            model_kwargs["predictor"] = "gpu_predictor"
    elif xgboost_major_version(xgb.__version__) >= 2:
        model_kwargs["device"] = "cpu"

    if early_stopping and use_eval and args.early_stopping_rounds > 0:
        model_kwargs["early_stopping_rounds"] = args.early_stopping_rounds

    model = XGBRanker(**model_kwargs) if use_ranker else XGBClassifier(**model_kwargs)
    print(
        f"Training XGBoost {xgb_objective} on {args.xgb_device} "
        f"(tree_method={model_kwargs['tree_method']})"
    )
    if use_ranker:
        rank_train_df, train_group = prepare_rank_frame(train_df, "Train")
        if use_eval:
            rank_val_df, val_group = prepare_rank_frame(val_df, "Validation")
            eval_set = [(rank_val_df[FEATURE_COLS], rank_val_df["label"])]
            eval_group = [val_group]
        else:
            eval_set = None
            eval_group = None
        model.fit(
            rank_train_df[FEATURE_COLS],
            rank_train_df["label"],
            group=train_group,
            eval_set=eval_set,
            eval_group=eval_group,
            verbose=True,
        )
    else:
        eval_set = [(val_df[FEATURE_COLS], val_df["label"])] if use_eval else None
        model.fit(train_df[FEATURE_COLS], train_df["label"], eval_set=eval_set, verbose=True)

    if early_stopping and use_eval and hasattr(model, "best_iteration"):
        print(f"Best iteration: {model.best_iteration}")
        print(f"Best score:     {model.best_score:.6f}")
    return model

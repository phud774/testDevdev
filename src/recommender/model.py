import argparse

import pandas as pd

from .constants import FEATURE_COLS


def train_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    args: argparse.Namespace,
    n_estimators: int | None = None,
    early_stopping: bool = True,
):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: xgboost. Install it with `pip install xgboost` "
            "or run `pip install -r requirements.txt`."
        ) from exc

    train_pos = max(int(train_df["label"].sum()), 1)
    train_neg = max(len(train_df) - train_pos, 1)
    use_eval = val_df is not None and not val_df.empty

    model_kwargs = {
        "n_estimators": n_estimators or args.n_estimators,
        "max_depth": 5,
        "learning_rate": 0.06,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 10.0,
        "reg_lambda": 2.0,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "n_jobs": args.n_jobs,
        "random_state": 42,
        "scale_pos_weight": train_neg / train_pos,
    }
    if early_stopping and use_eval and args.early_stopping_rounds > 0:
        model_kwargs["early_stopping_rounds"] = args.early_stopping_rounds

    model = XGBClassifier(**model_kwargs)
    eval_set = [(val_df[FEATURE_COLS], val_df["label"])] if use_eval else None
    model.fit(train_df[FEATURE_COLS], train_df["label"], eval_set=eval_set, verbose=True)

    if early_stopping and use_eval and hasattr(model, "best_iteration"):
        print(f"Best iteration: {model.best_iteration}")
        print(f"Best score:     {model.best_score:.6f}")
    return model

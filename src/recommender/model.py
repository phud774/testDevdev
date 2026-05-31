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


def selected_model(args: argparse.Namespace) -> str:
    return getattr(args, "model", "xgboost")


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


def train_lightgbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    args: argparse.Namespace,
    n_estimators: int | None = None,
    early_stopping: bool = True,
):
    try:
        from lightgbm import LGBMClassifier, LGBMRanker, early_stopping as lgb_early_stopping
        from lightgbm import log_evaluation
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: lightgbm. Install it with `pip install lightgbm` "
            "or run `pip install -r requirements.txt`."
        ) from exc

    train_pos = max(int(train_df["label"].sum()), 1)
    train_neg = max(len(train_df) - train_pos, 1)
    use_eval = val_df is not None and not val_df.empty
    use_ranker = uses_ranking_objective(args)
    xgb_objective = getattr(args, "xgb_objective", "binary")
    n_jobs = args.n_jobs if args.n_jobs != 0 else None

    model_kwargs = {
        "n_estimators": n_estimators or args.n_estimators,
        "learning_rate": 0.06,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 40,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_lambda": 2.0,
        "n_jobs": n_jobs,
        "random_state": 42,
        "verbosity": -1,
    }
    if args.xgb_device == "cuda":
        model_kwargs["device_type"] = "gpu"

    callbacks = []
    if early_stopping and use_eval and args.early_stopping_rounds > 0:
        callbacks.append(lgb_early_stopping(args.early_stopping_rounds, verbose=True))
    callbacks.append(log_evaluation(period=50))

    if use_ranker:
        model_kwargs["objective"] = "lambdarank"
        model_kwargs["metric"] = "ndcg"
        model = LGBMRanker(**model_kwargs)
        print(f"Training LightGBM {xgb_objective} on {args.xgb_device}")
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
            eval_at=[10],
            callbacks=callbacks,
        )
    else:
        model_kwargs["objective"] = "binary"
        model_kwargs["scale_pos_weight"] = train_neg / train_pos
        model = LGBMClassifier(**model_kwargs)
        print(f"Training LightGBM binary on {args.xgb_device}")
        eval_set = [(val_df[FEATURE_COLS], val_df["label"])] if use_eval else None
        model.fit(
            train_df[FEATURE_COLS],
            train_df["label"],
            eval_set=eval_set,
            callbacks=callbacks,
        )

    best_iteration = getattr(model, "best_iteration_", None)
    if early_stopping and use_eval and best_iteration:
        print(f"Best iteration: {best_iteration}")
    return model


def train_linear_regression(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    args: argparse.Namespace,
    n_estimators: int | None = None,
    early_stopping: bool = True,
):
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: scikit-learn. Install it with `pip install scikit-learn` "
            "or run `pip install -r requirements.txt`."
        ) from exc

    n_jobs = args.n_jobs if args.n_jobs != 0 else None
    model = LinearRegression(n_jobs=n_jobs)
    print("Training LinearRegression scorer")
    model.fit(train_df[FEATURE_COLS], train_df["label"])
    return model


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    args: argparse.Namespace,
    n_estimators: int | None = None,
    early_stopping: bool = True,
):
    model_name = selected_model(args)
    if model_name == "xgboost":
        return train_xgboost(train_df, val_df, args, n_estimators, early_stopping)
    if model_name == "lightgbm":
        return train_lightgbm(train_df, val_df, args, n_estimators, early_stopping)
    if model_name == "linear_regression":
        return train_linear_regression(train_df, val_df, args, n_estimators, early_stopping)
    raise ValueError(f"Unknown model: {model_name}")


def predict_model_scores(model, feature_frame: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    values = feature_frame.to_numpy(dtype=np.float32, copy=False)
    model_name = selected_model(args)
    use_ranker = uses_ranking_objective(args)

    if model_name == "xgboost":
        if args.xgb_device == "cuda":
            try:
                import cupy as cp
            except ImportError:
                if use_ranker:
                    return model.predict(values)
                return model.predict_proba(values)[:, 1]

            gpu_values = cp.asarray(values)
            if use_ranker:
                scores = model.predict(gpu_values)
            else:
                scores = model.predict_proba(gpu_values)[:, 1]
            return cp.asnumpy(scores)

        if use_ranker:
            return model.predict(values)
        return model.predict_proba(values)[:, 1]

    if model_name == "lightgbm":
        if use_ranker:
            return model.predict(feature_frame)
        return model.predict_proba(feature_frame)[:, 1]

    if model_name == "linear_regression":
        return model.predict(feature_frame)

    raise ValueError(f"Unknown model: {model_name}")


def supports_log_loss(args: argparse.Namespace) -> bool:
    return selected_model(args) in {"xgboost", "lightgbm"} and not uses_ranking_objective(args)

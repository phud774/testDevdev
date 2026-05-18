import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss

from .constants import FEATURE_COLS, ITEM_COL, USER_COL


@dataclass
class CandidateCoverageStats:
    ground_truth: dict[str, list[str]]
    common_users: int = 0
    users_with_hit: int = 0
    total_true_items: int = 0
    total_hits: int = 0
    total_candidates: int = 0
    user_recalls: list[float] = field(default_factory=list)

    def update_from_frame(self, candidates: pd.DataFrame) -> None:
        for user_id, group in candidates.groupby(USER_COL, sort=False):
            true_items = self.ground_truth.get(str(user_id))
            if not true_items:
                continue

            true_set = {str(item) for item in true_items}
            candidate_set = set(group[ITEM_COL].astype(str))
            hits = len(true_set & candidate_set)

            self.common_users += 1
            self.users_with_hit += int(hits > 0)
            self.total_true_items += len(true_set)
            self.total_hits += hits
            self.total_candidates += len(candidate_set)
            self.user_recalls.append(hits / len(true_set) if true_set else 0.0)

    def results(self) -> dict[str, float]:
        return {
            "candidate_recall": self.total_hits / self.total_true_items if self.total_true_items else 0.0,
            "candidate_precision": self.total_hits / self.total_candidates if self.total_candidates else 0.0,
            "user_hit_rate": self.users_with_hit / self.common_users if self.common_users else 0.0,
            "mean_user_recall": float(np.mean(self.user_recalls)) if self.user_recalls else 0.0,
        }

    def print_summary(self) -> None:
        results = self.results()
        print("\nCandidate pool coverage")
        print(f"Common users:        {self.common_users:,}")
        print(f"Candidate pairs:     {self.total_candidates:,}")
        print(f"GT item hits:        {self.total_hits:,}/{self.total_true_items:,}")
        print(f"Candidate recall:    {results['candidate_recall']:.6f}")
        print(f"Candidate precision: {results['candidate_precision']:.6f}")
        print(f"User hit rate:       {results['user_hit_rate']:.6f}")
        print(f"Mean user recall:    {results['mean_user_recall']:.6f}")


def load_candidate_coverage_stats(ground_truth_path: Path | None) -> CandidateCoverageStats | None:
    if ground_truth_path is None or not ground_truth_path.exists():
        return None
    with ground_truth_path.open("r", encoding="utf-8") as f:
        return CandidateCoverageStats(ground_truth=json.load(f))


def precision_at_10_from_scores(df: pd.DataFrame) -> float:
    scored = df.sort_values([USER_COL, "score"], ascending=[True, False])
    top = scored.groupby(USER_COL, sort=False).head(10)
    per_user = top.groupby(USER_COL)["label"].sum() / 10.0
    return float(per_user.mean()) if len(per_user) else 0.0


def predictions_to_submission(df: pd.DataFrame, output_path: Path, k: int = 10) -> dict[str, list[str]]:
    scored = df.sort_values([USER_COL, "score"], ascending=[True, False])
    top = scored.groupby(USER_COL, sort=False).head(k)
    submission = {
        str(user_id): group[ITEM_COL].astype(str).tolist()
        for user_id, group in top.groupby(USER_COL, sort=False)
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False)
    return submission


def evaluate_submission_dict(
    ground_truth_path: Path,
    submission: dict[str, list[str]],
    k: int = 10,
) -> dict[str, float]:
    if not ground_truth_path.exists():
        print(f"\nGround truth not found, skip final evaluation: {ground_truth_path}")
        return {}

    with ground_truth_path.open("r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    ious = []
    mrrs = []
    precisions = []
    maps = []
    common_users = set(ground_truth.keys()) & set(submission.keys())
    missing_users = set(ground_truth.keys()) - set(submission.keys())

    for user_id in common_users:
        true_items = ground_truth[user_id]
        pred_items = submission[user_id][:k]
        true_set = set(true_items)
        pred_set = set(pred_items)

        union = true_set | pred_set
        ious.append(len(true_set & pred_set) / len(union) if union else 0.0)

        rr = 0.0
        hits = 0
        ap = 0.0
        for rank, item in enumerate(pred_items, start=1):
            if item in true_set:
                if rr == 0.0:
                    rr = 1.0 / rank
                hits += 1
                ap += hits / rank

        mrrs.append(rr)
        precisions.append(hits / k)
        denom = min(len(true_set), k)
        maps.append(ap / denom if denom else 0.0)

    results = {
        "IoU": float(np.mean(ious)) if ious else 0.0,
        "MRR": float(np.mean(mrrs)) if mrrs else 0.0,
        f"Precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"MAP@{k}": float(np.mean(maps)) if maps else 0.0,
    }
    user_coverage = len(common_users) / len(ground_truth) if ground_truth else 0.0
    all_user_results = {
        metric: value * user_coverage
        for metric, value in results.items()
    }

    print("\nFinal submission evaluation")
    print(f"Ground-truth users: {len(ground_truth):,}")
    print(f"Submission users:   {len(submission):,}")
    print(f"Common users:       {len(common_users):,}")
    print(f"Missing GT users:   {len(missing_users):,}")
    for metric, value in results.items():
        print(f"{metric:<15}: {value:.6f}")
    if missing_users:
        print("\nAll-ground-truth-user metrics, counting missing users as zero")
        for metric, value in all_user_results.items():
            print(f"{metric:<15}: {value:.6f}")
    return results


def evaluate_model(model, val_df: pd.DataFrame) -> pd.DataFrame:
    val_df = val_df.copy()
    val_df["score"] = model.predict_proba(val_df[FEATURE_COLS])[:, 1]
    ap = average_precision_score(val_df["label"], val_df["score"])
    loss = log_loss(val_df["label"], np.clip(val_df["score"], 1e-6, 1 - 1e-6))
    p10 = precision_at_10_from_scores(val_df)
    print("\nValidation metrics")
    print(f"Average precision: {ap:.6f}")
    print(f"Log loss:          {loss:.6f}")
    print(f"Precision@10:     {p10:.6f}")
    return val_df

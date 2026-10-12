"""Spread-model candidates shared by training and benchmarking."""

from sklearn.base import RegressorMixin
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)

BASELINE_MODEL_NAME = "random_forest_baseline"


def build_spread_candidate_regressors(random_state: int = 42) -> dict[str, RegressorMixin]:
    """Build the supported spread-regression candidates."""
    return {
        BASELINE_MODEL_NAME: RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1,
        ),
        "extra_trees_ensemble": ExtraTreesRegressor(
            n_estimators=160,
            max_depth=12,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
        "rf_high_capacity": RandomForestRegressor(
            n_estimators=260,
            max_depth=18,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        ),
        "et_high_capacity": ExtraTreesRegressor(
            n_estimators=320,
            max_depth=16,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gbr_resilient": GradientBoostingRegressor(
            n_estimators=260,
            learning_rate=0.03,
            max_depth=2,
            subsample=0.9,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_depth=8,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=random_state,
        ),
        "voting_ensemble": VotingRegressor(
            estimators=[
                (
                    "rf",
                    RandomForestRegressor(
                        n_estimators=120,
                        max_depth=12,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
                (
                    "et",
                    ExtraTreesRegressor(
                        n_estimators=180,
                        max_depth=12,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
                (
                    "gbr",
                    GradientBoostingRegressor(
                        n_estimators=180,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "adaptive_voting_ensemble": VotingRegressor(
            estimators=[
                (
                    "rfhc",
                    RandomForestRegressor(
                        n_estimators=240,
                        max_depth=16,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
                (
                    "ethc",
                    ExtraTreesRegressor(
                        n_estimators=280,
                        max_depth=14,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
                (
                    "gbr",
                    GradientBoostingRegressor(
                        n_estimators=260,
                        learning_rate=0.03,
                        max_depth=2,
                        subsample=0.9,
                        random_state=random_state,
                    ),
                ),
            ],
            weights=[0.4, 0.35, 0.25],
        ),
    }


def build_spread_regressor(model_name: str, random_state: int = 42) -> RegressorMixin:
    """Build one named spread-regression candidate."""
    candidates = build_spread_candidate_regressors(random_state)
    if model_name not in candidates:
        raise ValueError(f"Unknown spread model: {model_name}")
    return candidates[model_name]

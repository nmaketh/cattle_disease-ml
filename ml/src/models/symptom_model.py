"""Symptom model: training, evaluation, and inference helpers.

Key design decisions (v2):
  - HistGradientBoostingClassifier is now the primary candidate.
    It has built-in L2 regularisation, leaf-level min-sample constraints,
    and is ~50-200Ã— smaller on disk than the previous 591 MB Random Forest.
  - RandomForest / ExtraTrees are kept as fallback candidates but with a
    corrected hyperparameter search that prevents train-F1=1.0 overfitting:
      * bootstrap=True  (was False â€” each tree now sees only a bootstrap sample)
      * max_depth  âˆˆ [8, 10, 12, 15]   (was up to 24 / None)
      * min_samples_leaf âˆˆ [4, 6, 8, 10]   (was 1)
      * min_samples_split âˆˆ [8, 12, 16, 20] (was 2)
  - All three candidates use class_weight="balanced" / "balanced_subsample"
    to handle ECF/CBPP under-prediction.
  - Feature engineering (fe_* columns) is applied by the caller before passing
    DataFrames to train_symptom_model().  This keeps this module stateless.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold


# â”€â”€ Config dataclass â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@dataclass
class SymptomModelConfig:
    n_estimators: int = 300
    max_depth: int = 12            # Safe upper bound; search explores lower values
    min_samples_leaf: int = 6      # Leaf regularisation; prevents single-sample leaves
    random_state: int = 42
    model_candidates: Tuple[str, ...] = (
        "hist_gradient_boosting",
        "random_forest",
        "extra_trees",
    )
    search_n_iter: int = 40        # RandomizedSearchCV iterations per candidate
    cv_folds: int = 5              # Stratified K-fold splits
    max_overfit_gap: float = 0.06
    overfit_penalty: float = 0.75
    cv_max_overfit_gap: float = 0.06
    cv_overfit_penalty: float = 0.75


# â”€â”€ Internal helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    if not labels:
        return 0.0
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def _random_search(
    estimator: BaseEstimator,
    param_distributions: Dict[str, List[object]],
    x_train: np.ndarray,
    y_train: np.ndarray,
    cfg: SymptomModelConfig,
) -> Tuple[BaseEstimator, float, Dict[str, object], float]:
    """Run a stratified randomised hyperparameter search and return the best model."""
    cv = StratifiedKFold(
        n_splits=max(3, int(cfg.cv_folds)),
        shuffle=True,
        random_state=cfg.random_state,
    )
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=max(8, int(cfg.search_n_iter)),
        scoring="f1_macro",
        n_jobs=-1,
        cv=cv,
        random_state=cfg.random_state,
        refit=False,
        return_train_score=True,
        verbose=1,
        error_score="raise",
    )
    search.fit(x_train, y_train)

    results = search.cv_results_
    mean_test = np.asarray(results["mean_test_score"], dtype=float)
    mean_train = np.asarray(results.get("mean_train_score", np.zeros_like(mean_test)), dtype=float)
    if mean_train.shape != mean_test.shape:
        mean_train = np.zeros_like(mean_test)
    cv_gap = np.maximum(0.0, mean_train - mean_test)

    cv_max_gap = float(cfg.cv_max_overfit_gap)
    cv_penalty = float(cfg.cv_overfit_penalty)
    adjusted_score = mean_test - cv_penalty * np.maximum(0.0, cv_gap - cv_max_gap)

    best_idx = 0
    best_adj = float("-inf")
    best_test = float("-inf")
    for idx in range(len(mean_test)):
        adj = float(adjusted_score[idx])
        test = float(mean_test[idx])
        if adj > best_adj or (abs(adj - best_adj) < 1e-12 and test > best_test):
            best_idx = idx
            best_adj = adj
            best_test = test

    best_params = dict(results["params"][best_idx])
    tuned_model = clone(estimator).set_params(**best_params)
    tuned_model.fit(x_train, y_train)

    return tuned_model, float(mean_test[best_idx]), best_params, float(cv_gap[best_idx])


def _build_candidate(
    name: str, cfg: SymptomModelConfig
) -> Tuple[BaseEstimator, Dict[str, List[object]]]:
    """Instantiate a model candidate with its hyperparameter search space."""
    rs = cfg.random_state

    # â”€â”€ HistGradientBoostingClassifier â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if name == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            max_iter=240,
            learning_rate=0.06,
            max_leaf_nodes=20,       # Equivalent to max_depthâ‰ˆ5 for balanced trees
            max_depth=4,
            min_samples_leaf=30,     # Strong leaf regularisation
            l2_regularization=0.50,
            class_weight="balanced",
            random_state=rs,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
        )
        param_space: Dict[str, List[object]] = {
            "learning_rate":      [0.02, 0.04, 0.06, 0.08],
            "max_iter":           [120, 180, 240, 320],
            "max_leaf_nodes":     [12, 16, 20, 24, 31],
            "max_depth":          [2, 3, 4, 5],
            "min_samples_leaf":   [20, 30, 40, 60, 80],
            "l2_regularization":  [0.1, 0.5, 1.0, 2.0, 5.0],
            "early_stopping":     [True],
            "validation_fraction": [0.12, 0.15, 0.20],
            "n_iter_no_change":   [10, 15, 20],
        }
        return model, param_space

    # â”€â”€ ExtraTreesClassifier â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if name == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=6,
            bootstrap=True,          # Was False in v1 â€” enforces diversity
            random_state=rs,
            class_weight="balanced",
            n_jobs=-1,
        )
        param_space = {
            "n_estimators":      [200, 300, 400],
            "max_depth":         [6, 8, 10, 12],
            "min_samples_leaf":  [6, 8, 10, 12, 16],
            "min_samples_split": [12, 16, 20, 24],
            "max_features":      ["sqrt", "log2", 0.30, 0.40],
            "bootstrap":         [True],             # Forced True
            "ccp_alpha":         [0.0, 0.0005, 0.001, 0.002],
            "max_samples":       [0.70, 0.85, 1.0],
        }
        return model, param_space

    # â”€â”€ RandomForestClassifier (default) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=6,
        bootstrap=True,              # Was False in v1
        random_state=rs,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    param_space = {
        "n_estimators":      [200, 300, 400],
        "max_depth":         [6, 8, 10, 12],
        "min_samples_leaf":  [6, 8, 10, 12, 16],
        "min_samples_split": [12, 16, 20, 24],
        "max_features":      ["sqrt", "log2", 0.30, 0.40],
        "bootstrap":         [True],             # Forced True
        "ccp_alpha":         [0.0, 0.0005, 0.001, 0.002],
        "max_samples":       [0.70, 0.85, 1.0],
    }
    return model, param_space


# â”€â”€ Public training function â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def train_symptom_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    cfg: SymptomModelConfig,
) -> Tuple[BaseEstimator, Dict[str, object]]:
    """Train all model candidates, return the best model and its metrics.

    The caller is responsible for applying feature engineering to both
    train_df and val_df before calling this function.

    Args:
        train_df:     Training split (must contain 'Disease' column).
        val_df:       Validation split (must contain 'Disease' column).
        feature_cols: Ordered list of feature column names (incl. fe_* cols).
        cfg:          Training configuration.

    Returns:
        (best_model, metrics_dict)
    """
    x_train = train_df[feature_cols].astype(float).values
    y_train = train_df["Disease"].astype(str).values
    x_val   = val_df[feature_cols].astype(float).values
    y_val   = val_df["Disease"].astype(str).values

    candidate_names = tuple(dict.fromkeys(cfg.model_candidates or ("random_forest",)))
    candidate_results: List[Dict[str, object]] = []
    best_gap_model: Optional[BaseEstimator] = None
    best_gap_result: Optional[Dict[str, object]] = None
    best_gap_val = -1.0
    best_fallback_model: Optional[BaseEstimator] = None
    best_fallback_result: Optional[Dict[str, object]] = None
    best_fallback_score = -1.0
    best_fallback_val = -1.0

    for name in candidate_names:
        print(f"\n[symptom_model] --- Training candidate: {name} ---")
        estimator, param_space = _build_candidate(name, cfg)
        fitted_model, cv_macro, best_params, cv_gap = _random_search(
            estimator, param_space, x_train, y_train, cfg
        )

        train_pred = fitted_model.predict(x_train)
        val_pred   = fitted_model.predict(x_val)
        train_f1   = _macro_f1(y_train, train_pred)
        val_f1     = _macro_f1(y_val,   val_pred)
        gap        = train_f1 - val_f1
        overfit_excess = max(0.0, gap - float(cfg.max_overfit_gap))
        selection_score = val_f1 - float(cfg.overfit_penalty) * overfit_excess

        print(
            f"[symptom_model]   cv_f1={cv_macro:.4f}  "
            f"cv_gap={cv_gap:.4f}  "
            f"train_f1={train_f1:.4f}  val_f1={val_f1:.4f}  "
            f"overfit_gap={gap:.4f}  adjusted_score={selection_score:.4f}  "
            f"params={best_params}"
        )

        candidate_result = {
            "name":            name,
            "cv_macro_f1":     round(cv_macro, 6),
            "cv_overfit_gap":  round(cv_gap, 6),
            "train_macro_f1":  round(train_f1, 6),
            "val_macro_f1":    round(val_f1, 6),
            "overfit_gap":     round(gap, 6),
            "overfit_excess":  round(overfit_excess, 6),
            "selection_score": round(selection_score, 6),
            "best_params":     best_params,
        }
        candidate_results.append(candidate_result)

        if gap <= float(cfg.max_overfit_gap) and val_f1 > best_gap_val:
            best_gap_val = val_f1
            best_gap_model = fitted_model
            best_gap_result = candidate_result

        if (selection_score > best_fallback_score) or (
            abs(selection_score - best_fallback_score) < 1e-12 and val_f1 > best_fallback_val
        ):
            best_fallback_score = selection_score
            best_fallback_val = val_f1
            best_fallback_model = fitted_model
            best_fallback_result = candidate_result

    if best_gap_model is not None and best_gap_result is not None:
        best_model = best_gap_model
        selected_result = best_gap_result
        selection_reason = "best_val_within_gap"
    else:
        best_model = best_fallback_model
        selected_result = best_fallback_result
        selection_reason = "best_penalized_score"

    if best_model is None or selected_result is None:
        raise RuntimeError("No symptom model candidate was successfully trained.")

    # â”€â”€ Final metrics on best model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    y_train_pred = best_model.predict(x_train)
    y_val_pred   = best_model.predict(x_val)
    y_val_prob   = best_model.predict_proba(x_val)
    train_macro  = _macro_f1(y_train, y_train_pred)
    val_macro    = _macro_f1(y_val,   y_val_pred)

    rep_train = classification_report(y_train, y_train_pred, zero_division=0, output_dict=True)
    rep_val   = classification_report(y_val,   y_val_pred,   zero_division=0, output_dict=True)

    # Feature importances (native if available, else permutation-based)
    if hasattr(best_model, "feature_importances_"):
        importances = np.asarray(best_model.feature_importances_, dtype=float)
    else:
        perm = permutation_importance(
            best_model, x_val, y_val,
            n_repeats=5,
            random_state=cfg.random_state,
            scoring="f1_macro",
            n_jobs=-1,
        )
        importances = np.asarray(perm.importances_mean, dtype=float)

    top_idx = np.argsort(importances)[::-1][:20]
    top_features = [
        {"feature": feature_cols[i], "importance": round(float(importances[i]), 6)}
        for i in top_idx
    ]

    selected_name = type(best_model).__name__
    selected_candidate_name = str(selected_result["name"])

    print(
        f"\n[symptom_model] --- Best model: {selected_name} ({selected_candidate_name})  "
        f"train_f1={train_macro:.4f}  val_f1={val_macro:.4f}  "
        f"gap={train_macro - val_macro:.4f}  reason={selection_reason} ---"
    )

    metrics: Dict[str, object] = {
        "selected_model":            selected_name,
        "selected_candidate":        selected_candidate_name,
        "selection_reason":          selection_reason,
        "selection_max_overfit_gap": float(cfg.max_overfit_gap),
        "selection_overfit_penalty": float(cfg.overfit_penalty),
        "cv_selection_max_overfit_gap": float(cfg.cv_max_overfit_gap),
        "cv_selection_overfit_penalty": float(cfg.cv_overfit_penalty),
        "macro_f1":                  val_macro,
        "train_macro_f1":            train_macro,
        "val_macro_f1":              val_macro,
        "overfit_gap_train_val":     train_macro - val_macro,
        "classification_report_train": rep_train,
        "classification_report_val":   rep_val,
        "labels":                    sorted(set(y_train.tolist()) | set(y_val.tolist())),
        "top_features":              top_features,
        "candidate_results":         candidate_results,
        "val_pred_probs_shape":      list(y_val_prob.shape),
    }
    return best_model, metrics


# â”€â”€ Persistence helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def save_symptom_model(model: BaseEstimator, path: str) -> None:
    joblib.dump(model, path, compress=3)  # Compression reduces disk size significantly


def load_symptom_model(path: str) -> BaseEstimator:
    return joblib.load(path)


# â”€â”€ Inference helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def symptom_top_features(
    model: BaseEstimator,
    input_row: pd.Series,
    feature_cols: List[str],
    top_k: int = 10,
) -> List[Dict[str, float]]:
    """Return top-k features sorted by (importance Ã— presence) contribution.

    Engineered fe_* features are excluded from the returned list so that only
    clinically meaningful symptom names are shown to end users.

    Args:
        model:        Fitted model with feature_importances_ attribute.
        input_row:    Single-row Series with feature values.
        feature_cols: Feature names in model input order.
        top_k:        Number of top features to return.

    Returns:
        List of {"symptom": ..., "importance": ...} dicts.
    """
    if not hasattr(model, "feature_importances_"):
        return []

    importances = np.asarray(model.feature_importances_, dtype=float)
    scored: List[Tuple[str, float]] = []

    for idx, feature in enumerate(feature_cols):
        # Skip engineered interaction features â€” they are internal signals
        if feature.startswith("fe_"):
            continue
        raw_value = input_row.get(feature, 0)
        try:
            value = float(raw_value)
        except Exception:
            value = 0.0
        if value <= 0:
            continue
        contribution = float(importances[idx]) * value
        scored.append((feature, contribution))

    scored.sort(key=lambda item: item[1], reverse=True)
    return [{"symptom": name, "importance": score} for name, score in scored[:top_k]]

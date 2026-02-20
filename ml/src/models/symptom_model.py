from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report


@dataclass
class SymptomModelConfig:
    n_estimators: int = 500
    max_depth: int = 24
    min_samples_leaf: int = 1
    random_state: int = 42


def train_symptom_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    cfg: SymptomModelConfig,
) -> Tuple[RandomForestClassifier, Dict[str, object]]:
    model = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        random_state=cfg.random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    x_train = train_df[feature_cols].values
    y_train = train_df["Disease"].values
    x_val = val_df[feature_cols].values
    y_val = val_df["Disease"].values

    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    y_prob = model.predict_proba(x_val)

    report = classification_report(y_val, y_pred, zero_division=0, output_dict=True)
    macro_f1 = float(report.get("macro avg", {}).get("f1-score", 0.0))

    perm = permutation_importance(model, x_val, y_val, n_repeats=5, random_state=cfg.random_state, scoring="f1_macro")
    top_idx = np.argsort(perm.importances_mean)[::-1][:12]
    top_features = [
        {"feature": feature_cols[i], "importance": float(perm.importances_mean[i])}
        for i in top_idx
    ]

    metrics = {
        "macro_f1": macro_f1,
        "classification_report": report,
        "labels": list(model.classes_),
        "top_features": top_features,
        "val_pred_probs_shape": list(y_prob.shape),
    }
    return model, metrics


def save_symptom_model(model: RandomForestClassifier, path: str) -> None:
    joblib.dump(model, path)


def load_symptom_model(path: str) -> RandomForestClassifier:
    return joblib.load(path)


def symptom_top_features(
    model: RandomForestClassifier,
    input_row: pd.Series,
    feature_cols: List[str],
    top_k: int = 10,
) -> List[Dict[str, float]]:
    importances = model.feature_importances_
    active = []
    for idx, feat in enumerate(feature_cols):
        if int(input_row.get(feat, 0)) == 1:
            active.append((feat, float(importances[idx])))
    active.sort(key=lambda x: x[1], reverse=True)
    return [{"symptom": k, "importance": v} for k, v in active[:top_k]]

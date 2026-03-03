from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from src.data.feature_engineering import add_engineered_features
from src.data.symptom_bootstrap import build_bootstrap_symptom_df
from src.models.symptom_model import SymptomModelConfig, save_symptom_model, train_symptom_model
from src.utils.io import ensure_dir, read_yaml, write_json
from src.utils.viz import save_confusion_matrix


def _macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return 0.0
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def main() -> int:
    cfg = read_yaml("ml/configs/config.yaml")
    csv_path = Path(cfg["paths"]["symptoms_merged_csv"]).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing symptom csv: {csv_path}. Run clean_merge_symptoms.py first.")

    real_df = pd.read_csv(csv_path)
    real_df["__source"] = "real"
    unique_rows = int(len(real_df.drop_duplicates()))
    print(f"[paths] symptoms_merged={csv_path}")
    print(f"[count] rows={len(real_df)} class_counts={real_df['Disease'].value_counts().to_dict()}")
    print(f"[count] unique_rows={unique_rows}")

    # Symptom model is intentionally 4-class (LSD / FMD / ECF / CBPP).
    # Normal is handled exclusively by the image model; the fusion layer
    # down-weights the symptom branch for Normal image predictions.
    if "Normal" in set(real_df["Disease"].unique().tolist()):
        real_df = real_df[real_df["Disease"] != "Normal"].copy()
        print("[info] Dropped 'Normal' rows from symptom dataset — Normal is image-only.")

    feature_cols = [column for column in real_df.columns if column not in {"Disease", "__source"}]
    if not feature_cols:
        raise ValueError("No symptom feature columns found.")

    symptom_cfg = cfg.get("symptom", {}) if isinstance(cfg.get("symptom"), dict) else {}
    min_unique_rows = int(symptom_cfg.get("min_unique_rows", 100))
    bootstrap_enabled = bool(symptom_cfg.get("bootstrap_if_insufficient", False))
    training_mode = "real_only"
    synthetic_df = pd.DataFrame(columns=feature_cols + ["Disease", "__source"])
    train_df_all = real_df.copy()

    if unique_rows < min_unique_rows:
        if not bootstrap_enabled:
            raise ValueError(
                "Symptom dataset has too few unique rows for reliable ML training. "
                "Collect more diverse symptom records or enable bootstrap_if_insufficient."
            )

        catalog = cfg.get("rules", {}).get("disease_symptom_catalog", {}) if isinstance(cfg.get("rules"), dict) else {}
        bootstrap_labels = list(symptom_cfg.get("bootstrap_labels", ["Normal", "LSD", "FMD"]))
        synthetic_df = build_bootstrap_symptom_df(
            feature_cols=feature_cols,
            catalog=catalog,
            labels=bootstrap_labels,
            samples_per_label=int(symptom_cfg.get("bootstrap_samples_per_label", 300)),
            seed=int(cfg["seed"]),
            core_prob=float(symptom_cfg.get("bootstrap_core_prob", 0.78)),
            support_prob=float(symptom_cfg.get("bootstrap_support_prob", 0.42)),
            noise_prob=float(symptom_cfg.get("bootstrap_noise_prob", 0.03)),
        )
        synthetic_df["__source"] = "synthetic"
        train_df_all = pd.concat([real_df, synthetic_df], ignore_index=True)
        training_mode = "bootstrap_weak"
        print(
            f"[warn] Using synthetic bootstrap for symptom model due to low unique real rows ({unique_rows}). "
            f"Synthetic rows added={len(synthetic_df)}"
        )

    if len(train_df_all["Disease"].unique()) < 2:
        raise ValueError("Symptom dataset has fewer than 2 classes after preparation.")

    # ── Feature engineering ───────────────────────────────────────────────────
    # Applied to the full dataset BEFORE splitting so that train, val, and test
    # all have identical fe_* interaction features.  The updated feature_cols
    # list (which now includes fe_* columns) is passed to the model and saved
    # as symptom_features.json so that inference can replicate the same step.
    _base_feature_count = len(feature_cols)
    print(f"[feature_eng] Applying feature engineering to {len(train_df_all)} rows ...")
    train_df_all = add_engineered_features(train_df_all)
    feature_cols = [c for c in train_df_all.columns if c not in {"Disease", "__source"}]
    _eng_count = len(feature_cols) - _base_feature_count
    print(f"[feature_eng] Features: {_base_feature_count} base + {_eng_count} engineered = {len(feature_cols)} total")

    train_df, hold_df = train_test_split(
        train_df_all,
        test_size=0.3,
        stratify=train_df_all["Disease"],
        random_state=int(cfg["seed"]),
    )
    val_df, test_df = train_test_split(
        hold_df,
        test_size=0.5,
        stratify=hold_df["Disease"],
        random_state=int(cfg["seed"]),
    )

    model_cfg = SymptomModelConfig(
        n_estimators=int(symptom_cfg.get("n_estimators", 300)),
        max_depth=int(symptom_cfg.get("max_depth", 12)),
        min_samples_leaf=int(symptom_cfg.get("min_samples_leaf", 6)),
        random_state=int(cfg["seed"]),
        model_candidates=tuple(
            symptom_cfg.get("model_candidates", ["hist_gradient_boosting", "random_forest", "extra_trees"])
        ),
        search_n_iter=int(symptom_cfg.get("search_n_iter", 40)),
        cv_folds=int(symptom_cfg.get("cv_folds", 5)),
        max_overfit_gap=float(symptom_cfg.get("max_overfit_gap", 0.06)),
        overfit_penalty=float(symptom_cfg.get("selection_overfit_penalty", 0.75)),
        cv_max_overfit_gap=float(symptom_cfg.get("cv_max_overfit_gap", symptom_cfg.get("max_overfit_gap", 0.06))),
        cv_overfit_penalty=float(
            symptom_cfg.get("cv_overfit_penalty", symptom_cfg.get("selection_overfit_penalty", 0.75))
        ),
    )
    model, metrics = train_symptom_model(train_df, val_df, feature_cols, model_cfg)

    artifact_root = Path(cfg["paths"]["artifacts_dir"]).resolve()
    ensure_dir(artifact_root)
    labels = sorted(train_df_all["Disease"].unique().tolist())

    model_path = artifact_root / "symptom_model.pkl"
    save_symptom_model(model, str(model_path))
    write_json(artifact_root / "symptom_features.json", feature_cols)
    write_json(
        artifact_root / "label_map.json",
        {
            "final_labels": cfg["labels"]["final_labels"],
            "symptom_train_labels": labels,
            "image_labels": cfg["labels"]["image_labels"],
            "symptom_training_mode": training_mode,
        },
    )

    x_test = test_df[feature_cols].astype(float).values
    y_true_test = test_df["Disease"].astype(str).tolist()
    y_pred_test = model.predict(x_test).tolist()
    test_macro = _macro_f1(y_true_test, y_pred_test)
    test_report = classification_report(y_true_test, y_pred_test, zero_division=0, output_dict=True)

    reports_dir = ensure_dir(artifact_root / "reports")
    save_confusion_matrix(y_true_test, y_pred_test, labels, reports_dir / "confusion_matrix_symptom.png", "Symptom Model CM")

    summary_lines = [
        "Symptom Training Summary",
        "=" * 30,
        f"selected_model: {metrics.get('selected_model')}",
        f"selected_candidate: {metrics.get('selected_candidate')}",
        f"train_macro_f1: {float(metrics.get('train_macro_f1', 0.0)):.4f}",
        f"val_macro_f1: {float(metrics.get('val_macro_f1', 0.0)):.4f}",
        f"test_macro_f1: {test_macro:.4f}",
        f"overfit_gap_train_val: {float(metrics.get('overfit_gap_train_val', 0.0)):.4f}",
        "",
        "candidate_results:",
    ]
    for item in metrics.get("candidate_results", []):
        summary_lines.append(
            f"  - {item.get('name')}: cv={item.get('cv_macro_f1')} train={item.get('train_macro_f1')} val={item.get('val_macro_f1')}"
        )
    (reports_dir / "symptom_report.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    test_df.assign(pred=y_pred_test).to_csv(artifact_root / "symptom_test_predictions.csv", index=False)
    split_source_counts = {
        "train": train_df["__source"].value_counts().to_dict(),
        "val": val_df["__source"].value_counts().to_dict(),
        "test": test_df["__source"].value_counts().to_dict(),
    }

    max_overfit_gap = float(symptom_cfg.get("max_overfit_gap", 0.08))
    target_macro_f1 = float(symptom_cfg.get("target_macro_f1", 0.90))
    overfit_gap = float(metrics.get("overfit_gap_train_val", 0.0))
    warnings: list[str] = []
    if overfit_gap > max_overfit_gap:
        warnings.append(
            f"train/val gap {overfit_gap:.4f} exceeds max_overfit_gap={max_overfit_gap:.4f}; possible overfitting."
        )
    if test_macro < target_macro_f1:
        warnings.append(
            f"test macro_f1 {test_macro:.4f} is below target_macro_f1={target_macro_f1:.4f}; requires more signal or data."
        )

    write_json(
        artifact_root / "symptom_model_metadata.json",
        {
            "training_mode": training_mode,
            "real_unique_rows": unique_rows,
            "real_rows": int(len(real_df)),
            "synthetic_rows_added": int(len(synthetic_df)),
            "labels_trained": labels,
            "split_source_counts": split_source_counts,
            "eval_has_synthetic_rows": bool(int(split_source_counts["test"].get("synthetic", 0)) > 0),
            "warning": (
                "Symptom model includes synthetic bootstrap examples and should be treated as weak evidence."
                if training_mode != "real_only"
                else ""
            ),
            "selected_model": metrics.get("selected_model"),
            "selected_candidate": metrics.get("selected_candidate"),
            "train_macro_f1": float(metrics.get("train_macro_f1", 0.0)),
            "val_macro_f1": float(metrics.get("val_macro_f1", 0.0)),
            "test_macro_f1": test_macro,
            "overfit_gap_train_val": overfit_gap,
            "generalization_gap_val_test": float(metrics.get("val_macro_f1", 0.0)) - test_macro,
            "target_macro_f1": target_macro_f1,
            "max_overfit_gap": max_overfit_gap,
            "candidate_results": metrics.get("candidate_results", []),
            "top_features": metrics.get("top_features", []),
            "test_classification_report": test_report,
            "warnings": warnings,
        },
    )

    print(f"[save] symptom_model={model_path}")
    print(f"[save] symptom_features={artifact_root / 'symptom_features.json'}")
    print(f"[save] symptom_report={reports_dir / 'symptom_report.txt'}")
    print(f"[save] symptom_metadata={artifact_root / 'symptom_model_metadata.json'}")
    if warnings:
        for warning in warnings:
            print(f"[warn] {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

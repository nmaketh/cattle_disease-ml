from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.symptom_bootstrap import build_bootstrap_symptom_df
from src.models.symptom_model import SymptomModelConfig, save_symptom_model, train_symptom_model
from src.utils.io import ensure_dir, read_yaml, write_json
from src.utils.viz import save_confusion_matrix


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

    if "Normal" not in set(real_df["Disease"].unique().tolist()):
        print("[warn] Symptom dataset lacks Normal class. Hybrid fusion will down-weight symptom branch for Normal image predictions.")

    feature_cols = [c for c in real_df.columns if c not in {"Disease", "__source"}]
    if not feature_cols:
        raise ValueError("No symptom feature columns found.")

    min_unique_rows = int(cfg["symptom"].get("min_unique_rows", 100))
    bootstrap_enabled = bool(cfg["symptom"].get("bootstrap_if_insufficient", False))
    training_mode = "real_only"
    synthetic_df = pd.DataFrame(columns=feature_cols + ["Disease", "__source"])
    train_df_all = real_df.copy()

    if unique_rows < min_unique_rows:
        if not bootstrap_enabled:
            raise ValueError(
                "Symptom dataset has too few unique rows for reliable ML training. "
                "Collect more diverse symptom records or enable bootstrap_if_insufficient."
            )

        catalog = cfg.get("rules", {}).get("disease_symptom_catalog", {})
        bootstrap_labels = list(cfg["symptom"].get("bootstrap_labels", ["Normal", "LSD", "FMD"]))
        synthetic_df = build_bootstrap_symptom_df(
            feature_cols=feature_cols,
            catalog=catalog,
            labels=bootstrap_labels,
            samples_per_label=int(cfg["symptom"].get("bootstrap_samples_per_label", 300)),
            seed=int(cfg["seed"]),
            core_prob=float(cfg["symptom"].get("bootstrap_core_prob", 0.78)),
            support_prob=float(cfg["symptom"].get("bootstrap_support_prob", 0.42)),
            noise_prob=float(cfg["symptom"].get("bootstrap_noise_prob", 0.03)),
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
        n_estimators=int(cfg["symptom"]["n_estimators"]),
        max_depth=int(cfg["symptom"]["max_depth"]),
        min_samples_leaf=int(cfg["symptom"]["min_samples_leaf"]),
        random_state=int(cfg["seed"]),
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

    y_true = test_df["Disease"].tolist()
    y_pred = model.predict(test_df[feature_cols].values).tolist()

    reports_dir = ensure_dir(artifact_root / "reports")
    save_confusion_matrix(y_true, y_pred, labels, reports_dir / "confusion_matrix_symptom.png", "Symptom Model CM")
    (reports_dir / "symptom_report.txt").write_text(str(metrics), encoding="utf-8")

    test_df.assign(pred=y_pred).to_csv(artifact_root / "symptom_test_predictions.csv", index=False)
    split_source_counts = {
        "train": train_df["__source"].value_counts().to_dict(),
        "val": val_df["__source"].value_counts().to_dict(),
        "test": test_df["__source"].value_counts().to_dict(),
    }
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
        },
    )

    print(f"[save] symptom_model={model_path}")
    print(f"[save] symptom_features={artifact_root / 'symptom_features.json'}")
    print(f"[save] symptom_report={reports_dir / 'symptom_report.txt'}")
    print(f"[save] symptom_metadata={artifact_root / 'symptom_model_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.symptom_model import SymptomModelConfig, save_symptom_model, train_symptom_model
from src.utils.io import ensure_dir, read_yaml, write_json
from src.utils.viz import save_confusion_matrix


def main() -> int:
    cfg = read_yaml("ml/configs/config.yaml")
    csv_path = Path(cfg["paths"]["symptoms_merged_csv"]).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing symptom csv: {csv_path}. Run clean_merge_symptoms.py first.")

    df = pd.read_csv(csv_path)
    print(f"[paths] symptoms_merged={csv_path}")
    print(f"[count] rows={len(df)} class_counts={df['Disease'].value_counts().to_dict()}")
    if "Normal" not in set(df["Disease"].unique().tolist()):
        print("[warn] Symptom dataset lacks Normal class. Hybrid fusion will down-weight symptom branch for Normal image predictions.")

    feature_cols = [c for c in df.columns if c != "Disease"]
    if not feature_cols:
        raise ValueError("No symptom feature columns found.")

    train_df, hold_df = train_test_split(df, test_size=0.3, stratify=df["Disease"], random_state=int(cfg["seed"]))
    val_df, test_df = train_test_split(hold_df, test_size=0.5, stratify=hold_df["Disease"], random_state=int(cfg["seed"]))

    model_cfg = SymptomModelConfig(
        n_estimators=int(cfg["symptom"]["n_estimators"]),
        max_depth=int(cfg["symptom"]["max_depth"]),
        min_samples_leaf=int(cfg["symptom"]["min_samples_leaf"]),
        random_state=int(cfg["seed"]),
    )
    model, metrics = train_symptom_model(train_df, val_df, feature_cols, model_cfg)

    artifact_root = Path(cfg["paths"]["artifacts_dir"]).resolve()
    ensure_dir(artifact_root)

    model_path = artifact_root / "symptom_model.pkl"
    save_symptom_model(model, str(model_path))
    write_json(artifact_root / "symptom_features.json", feature_cols)

    y_true = test_df["Disease"].tolist()
    y_pred = model.predict(test_df[feature_cols].values).tolist()
    labels = sorted(df["Disease"].unique().tolist())

    reports_dir = ensure_dir(artifact_root / "reports")
    save_confusion_matrix(y_true, y_pred, labels, reports_dir / "confusion_matrix_symptom.png", "Symptom Model CM")
    (reports_dir / "symptom_report.txt").write_text(str(metrics), encoding="utf-8")

    test_df.assign(pred=y_pred).to_csv(artifact_root / "symptom_test_predictions.csv", index=False)

    print(f"[save] symptom_model={model_path}")
    print(f"[save] symptom_features={artifact_root / 'symptom_features.json'}")
    print(f"[save] symptom_report={reports_dir / 'symptom_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

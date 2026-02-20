from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

from src.models.symptom_model import load_symptom_model
from src.utils.io import read_json, read_yaml
from src.utils.viz import save_confusion_matrix


def main() -> int:
    cfg = read_yaml("ml/configs/config.yaml")
    artifact_root = Path(cfg["paths"]["artifacts_dir"]).resolve()

    test_csv = artifact_root / "symptom_test_predictions.csv"
    model_path = artifact_root / "symptom_model.pkl"
    feature_path = artifact_root / "symptom_features.json"
    merged_csv = Path(cfg["paths"]["symptoms_merged_csv"]).resolve()

    if not model_path.exists() or not feature_path.exists() or not merged_csv.exists():
        raise FileNotFoundError("Missing symptom model artifacts. Run train_symptoms.py first.")

    model = load_symptom_model(str(model_path))
    features = read_json(feature_path)
    df = pd.read_csv(merged_csv)

    # If explicit test split exists from training run, prefer that.
    if test_csv.exists():
        eval_df = pd.read_csv(test_csv)
        y_true = eval_df["Disease"].tolist()
        y_pred = eval_df["pred"].tolist()
    else:
        y_true = df["Disease"].tolist()
        y_pred = model.predict(df[features].values).tolist()

    labels = sorted(df["Disease"].unique().tolist())
    rep = classification_report(y_true, y_pred, labels=labels, zero_division=0)

    reports = artifact_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "symptom_report.txt").write_text(rep, encoding="utf-8")
    save_confusion_matrix(y_true, y_pred, labels, reports / "confusion_matrix_symptom.png", "Symptom Model CM")

    print(rep)
    print(f"[save] {reports / 'symptom_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

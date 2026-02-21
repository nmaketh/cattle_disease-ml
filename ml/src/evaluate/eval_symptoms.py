from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, f1_score

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

    labels = sorted(set(y_true) | set(y_pred))
    rep = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    macro_f1_all = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    meta_path = artifact_root / "symptom_model_metadata.json"
    meta = read_json(meta_path) if meta_path.exists() else {}
    training_mode = str(meta.get("training_mode", "unknown"))
    warning = str(meta.get("warning", ""))

    reports = artifact_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    header = [
        "Symptom Evaluation Report",
        "=" * 30,
        f"training_mode: {training_mode}",
        f"macro_f1_all_rows: {macro_f1_all:.4f}",
    ]
    if warning:
        header.append(f"warning: {warning}")
    header.append("note: Symptom-only metrics are not equivalent to validated 5-class clinical diagnostic performance.")
    if test_csv.exists() and "eval_df" in locals() and "__source" in eval_df.columns:
        real_mask = eval_df["__source"].astype(str).str.lower() == "real"
        n_real = int(real_mask.sum())
        n_total = int(len(eval_df))
        header.append(f"test_rows_total: {n_total}")
        header.append(f"test_rows_real: {n_real}")
        header.append(f"test_rows_synthetic: {n_total - n_real}")
        if n_real < 30:
            header.append("warning: real-only symptom test sample is very small; real-only metrics are unstable.")
        if n_real > 0:
            y_true_real = eval_df.loc[real_mask, "Disease"].tolist()
            y_pred_real = eval_df.loc[real_mask, "pred"].tolist()
            labels_real = sorted(set(y_true_real) | set(y_pred_real))
            rep_real = classification_report(y_true_real, y_pred_real, labels=labels_real, zero_division=0)
            macro_f1_real = float(f1_score(y_true_real, y_pred_real, labels=labels_real, average="macro", zero_division=0))
            header.append(f"macro_f1_real_only: {macro_f1_real:.4f}")
            header.append("")
            header.append("Real-only report:")
            header.append(rep_real)
        else:
            header.append("macro_f1_real_only: n/a (no real rows in symptom test split)")
    header.append("")
    header.append("All evaluated rows report:")
    header.append(rep)
    (reports / "symptom_report.txt").write_text("\n".join(header), encoding="utf-8")
    save_confusion_matrix(y_true, y_pred, labels, reports / "confusion_matrix_symptom.png", "Symptom Model CM")

    print(rep)
    print(f"[save] {reports / 'symptom_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

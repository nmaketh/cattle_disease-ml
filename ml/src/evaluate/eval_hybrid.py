import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score

from src.models.fusion import FusionConfig, fuse_predictions
from src.models.rules_engine import RulesConfig, rules_predict
from src.models.symptom_model import load_symptom_model, symptom_top_features
from src.utils.io import read_json, read_yaml, write_json
from src.utils.viz import save_confusion_matrix

LABEL_TO_ID = {"Normal": 0, "LSD": 1, "FMD": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def make_ds(df: pd.DataFrame, img_size: int, batch_size: int) -> tf.data.Dataset:
    paths = df["filepath"].astype(str).tolist()
    labels = df["label"].map(LABEL_TO_ID).astype("int32").tolist()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32)
        return img, label

    return ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main() -> int:
    cfg = read_yaml("ml/configs/config.yaml")
    artifact_root = Path(cfg["paths"]["artifacts_dir"]).resolve()

    manifest = Path(cfg["paths"]["splits_manifest_csv"]).resolve()
    symptom_csv = Path(cfg["paths"]["symptoms_merged_csv"]).resolve()
    image_dir = artifact_root / "image_model"
    symptom_model_path = artifact_root / "symptom_model.pkl"
    symptom_features = read_json(artifact_root / "symptom_features.json") if (artifact_root / "symptom_features.json").exists() else []

    if not manifest.exists() or not symptom_csv.exists() or not image_dir.exists() or not symptom_model_path.exists():
        raise FileNotFoundError("Missing required artifacts for hybrid evaluation.")

    img_layer = tf.keras.layers.TFSMLayer(str(image_dir), call_endpoint="serve")
    sym_model = load_symptom_model(str(symptom_model_path))

    split_df = pd.read_csv(manifest)
    test_df = split_df[split_df["split"] == "test"].copy().reset_index(drop=True)
    symptom_df_full = pd.read_csv(symptom_csv)

    # Build class-aware pseudo-pairing for ablation when true case IDs are unavailable:
    # FMD/LSD rows are sampled from matching symptom labels; Normal receives zero-symptom row.
    symptom_rows = []
    grouped = {k: v.reset_index(drop=True) for k, v in symptom_df_full.groupby("Disease")}
    counters = {k: 0 for k in grouped}
    zero_row = {c: 0 for c in symptom_df_full.columns if c != "Disease"}
    zero_row["Disease"] = "Normal"

    for _, row in test_df.iterrows():
        lbl = row["label"]
        if lbl in grouped and len(grouped[lbl]) > 0:
            idx = counters[lbl] % len(grouped[lbl])
            symptom_rows.append(grouped[lbl].iloc[idx].to_dict())
            counters[lbl] += 1
        else:
            symptom_rows.append(dict(zero_row))

    symptom_df = pd.DataFrame(symptom_rows).reset_index(drop=True)
    n = len(test_df)

    ds_test = make_ds(test_df, cfg["image"]["img_size"], cfg["image"]["batch_size"])

    image_probs = []
    for xb, _ in ds_test:
        out = img_layer(xb)
        if isinstance(out, dict):
            out = list(out.values())[0]
        image_probs.append(np.array(out))
    image_probs = np.concatenate(image_probs, axis=0)

    sym_x = symptom_df[symptom_features].values if symptom_features else np.zeros((n, 0))
    sym_prob = sym_model.predict_proba(sym_x)
    sym_labels = list(sym_model.classes_)

    fusion_cfg = FusionConfig(
        image_weight=float(cfg["fusion"]["image_weight"]),
        symptom_weight=float(cfg["fusion"]["symptom_weight"]),
        temperature=float(cfg["fusion"]["temperature"]),
        image_confidence_hi=float(cfg["image"]["image_confidence_hi"]),
        rule_threshold=float(cfg["fusion"]["rule_threshold"]),
        contradiction_threshold=float(cfg["fusion"]["contradiction_threshold"]),
        urgent_rule_score=float(cfg["thresholds"]["urgent_rule_score"]),
        low_confidence=float(cfg["thresholds"]["low_confidence"]),
    )
    rule_cfg = RulesConfig(ecf_weights=cfg["rules"]["ecf_weights"], cbpp_weights=cfg["rules"]["cbpp_weights"])

    y_true = test_df["label"].tolist()
    y_img = []
    y_hybrid = []
    y_hybrid_nlf = []
    payloads = []

    for i in range(n):
        img_dict = {"Normal": float(image_probs[i, 0]), "LSD": float(image_probs[i, 1]), "FMD": float(image_probs[i, 2])}
        y_img.append(max(img_dict, key=img_dict.get))

        sprob = {lbl: float(sym_prob[i, j]) for j, lbl in enumerate(sym_labels)}
        symptoms = symptom_df.iloc[i].to_dict()
        rule_obj = rules_predict(symptoms, rule_cfg)
        top_sym = symptom_top_features(sym_model, symptom_df.iloc[i], symptom_features, top_k=8) if symptom_features else []

        out = fuse_predictions(
            image_probs=img_dict,
            symptom_probs=sprob,
            rule_scores=rule_obj["scores"],
            rule_triggers=rule_obj["explanation"],
            cfg=fusion_cfg,
            gradcam_path=None,
            top_symptoms=top_sym,
        )
        y_hybrid.append(out["final_label"])
        probs_nlf = {k: float(out["probs"].get(k, 0.0)) for k in ["Normal", "LSD", "FMD"]}
        y_hybrid_nlf.append(max(probs_nlf, key=probs_nlf.get))
        payloads.append(out)

    labels_final = ["Normal", "LSD", "FMD", "ECF", "CBPP"]
    labels_img = ["Normal", "LSD", "FMD"]

    image_report = classification_report(y_true, y_img, labels=labels_img, zero_division=0)
    hybrid_report = classification_report(y_true, y_hybrid, labels=labels_final, zero_division=0)
    hybrid_nlf_report = classification_report(y_true, y_hybrid_nlf, labels=labels_img, zero_division=0)

    macro_f1_img = float(f1_score(y_true, y_img, labels=labels_img, average="macro", zero_division=0))
    macro_f1_hyb = float(f1_score(y_true, y_hybrid_nlf, labels=labels_img, average="macro", zero_division=0))

    reports = artifact_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    save_confusion_matrix(y_true, y_img, labels_img, reports / "confusion_matrix_image.png", "Image-only CM")
    save_confusion_matrix(y_true, y_hybrid_nlf, labels_img, reports / "confusion_matrix_hybrid.png", "Hybrid CM (N/L/F)")

    lines = [
        "Hybrid Ablation Report",
        "=" * 30,
        "Pairing strategy: class-aware pseudo pairing (no shared case IDs available).",
        f"Image-only macro F1: {macro_f1_img:.4f}",
        f"Hybrid macro F1 (N/L/F projection): {macro_f1_hyb:.4f}",
        f"Delta (Hybrid - Image): {macro_f1_hyb - macro_f1_img:+.4f}",
        "",
        "Image-only report:",
        image_report,
        "",
        "Hybrid report:",
        hybrid_report,
        "",
        "Hybrid N/L/F report (ablation metric space):",
        hybrid_nlf_report,
    ]
    (reports / "hybrid_report.txt").write_text("\n".join(lines), encoding="utf-8")
    (reports / "image_report.txt").write_text(image_report, encoding="utf-8")

    (artifact_root / "hybrid_predictions.json").write_text(json.dumps(payloads, indent=2), encoding="utf-8")
    write_json(
        artifact_root / "fusion_config.json",
        {
            "image_weight": fusion_cfg.image_weight,
            "symptom_weight": fusion_cfg.symptom_weight,
            "temperature": fusion_cfg.temperature,
            "image_confidence_hi": fusion_cfg.image_confidence_hi,
            "rule_threshold": fusion_cfg.rule_threshold,
            "contradiction_threshold": fusion_cfg.contradiction_threshold,
            "urgent_rule_score": fusion_cfg.urgent_rule_score,
            "low_confidence": fusion_cfg.low_confidence,
            "final_labels": labels_final,
        },
    )

    print("\n".join(lines[:5]))
    print(f"[save] {reports / 'hybrid_report.txt'}")
    print(f"[save] {reports / 'confusion_matrix_hybrid.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

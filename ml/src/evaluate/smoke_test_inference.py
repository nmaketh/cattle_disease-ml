import json
from pathlib import Path

from src.infer.predict import predict_full, predict_image, predict_symptoms


def _assert_prob_dict(probs: dict, expected_labels: list[str]) -> None:
    for lbl in expected_labels:
        if lbl not in probs:
            raise AssertionError(f"Missing probability label: {lbl}")
    s = sum(float(v) for v in probs.values())
    if abs(s - 1.0) > 1e-6:
        raise AssertionError(f"Probabilities must sum to 1.0, got {s}")


def main() -> int:
    image_path = Path("ml/data/processed/images_merged/FMD").glob("*.jpg")
    first = next(image_path, None)
    if first is None:
        first = next(Path("ml/data/processed/images_merged/FMD").glob("*.png"), None)
    if first is None:
        raise FileNotFoundError("No image found for smoke test under ml/data/processed/images_merged/FMD")

    img_bytes = first.read_bytes()
    symptom_payload = {
        "fever": 1,
        "swollen_lymph_nodes": 1,
        "eye_discharge": 1,
        "nasal_discharge": 1,
        "difficulty_breathing": 0,
    }

    img_probs, img_label, img_conf, grad = predict_image(img_bytes)
    if img_label not in {"Normal", "LSD", "FMD"}:
        raise AssertionError(f"Invalid image label: {img_label}")
    if not (0.0 <= img_conf <= 1.0):
        raise AssertionError(f"Invalid image confidence: {img_conf}")

    sym_probs, sym_label, sym_conf, top, sym_meta = predict_symptoms(symptom_payload)
    if not (0.0 <= sym_conf <= 1.0):
        raise AssertionError(f"Invalid symptom confidence: {sym_conf}")

    full = predict_full(image_bytes=img_bytes, symptoms_dict=symptom_payload)
    required_keys = {"final_label", "confidence", "method", "probs", "explain", "recommendation_flags"}
    missing = required_keys.difference(full.keys())
    if missing:
        raise AssertionError(f"Missing full response keys: {missing}")
    explain = full.get("explain")
    if not isinstance(explain, dict):
        raise AssertionError("Expected explain payload to be a dict.")
    explain_required = {"reasoning", "supporting_evidence", "cautionary_evidence", "modality_summary"}
    explain_missing = explain_required.difference(explain.keys())
    if explain_missing:
        raise AssertionError(f"Missing explainability text fields: {explain_missing}")

    _assert_prob_dict(full["probs"], ["Normal", "LSD", "FMD", "ECF", "CBPP"])

    final_label = full["final_label"]
    if final_label not in ["Normal", "LSD", "FMD", "ECF", "CBPP"]:
        raise AssertionError(f"Invalid final label: {final_label}")
    if abs(float(full["confidence"]) - float(full["probs"][final_label])) > 1e-6:
        raise AssertionError("Confidence must match probs[final_label].")

    report = {
        "image_path": str(first),
        "image_label": img_label,
        "image_confidence": img_conf,
        "symptom_label": sym_label,
        "symptom_confidence": sym_conf,
        "symptom_model_training_mode": sym_meta.get("training_mode", "unknown"),
        "symptom_model_warning": sym_meta.get("warning", ""),
        "full_result": full,
        "gradcam_path": grad,
        "top_symptoms_count": len(top) if isinstance(top, list) else 0,
    }

    out = Path("ml/artifacts/reports/smoke_test_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[ok] smoke test passed")
    print(f"[save] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

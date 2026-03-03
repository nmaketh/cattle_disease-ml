from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from pathlib import Path
from src.infer.predict import _ART, _CFG, load_symptom_metadata, predict_full, predict_image, predict_symptoms


class SymptomsRequest(BaseModel):
    symptoms: Dict[str, Any] = Field(default_factory=dict)


class FullRequest(BaseModel):
    symptoms: Dict[str, Any] = Field(default_factory=dict)


app = FastAPI(
    title="Livestock Hybrid Disease Screening API",
    version="1.0.0",
    description="Hybrid inference API for Normal/LSD/FMD/ECF/CBPP with provenance and explainability payloads.",
)

_GRADCAM_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def _gradcam_reports_dir() -> Path:
    return Path(_ART) / "reports" / "gradcam_examples"


@app.get("/health")
def health() -> Dict[str, Any]:
    symptom_meta = load_symptom_metadata()
    return {
        "status": "ok",
        "artifacts_dir": str(_ART),
        "symptom_training_mode": symptom_meta.get("training_mode", "unknown"),
        "symptom_real_rows": symptom_meta.get("real_rows"),
        "symptom_real_unique_rows": symptom_meta.get("real_unique_rows"),
        "symptom_synthetic_rows_added": symptom_meta.get("synthetic_rows_added"),
    }


@app.get("/symptoms/catalog")
def symptom_catalog() -> Dict[str, Any]:
    return {
        "labels": _CFG["labels"]["final_labels"],
        "catalog": _CFG.get("rules", {}).get("disease_symptom_catalog", {}),
    }


@app.get("/artifacts/gradcam/{filename}")
def get_gradcam_artifact(filename: str):
    safe_name = filename.strip()
    if not safe_name or "/" in safe_name or "\\" in safe_name or ".." in safe_name:
        raise HTTPException(status_code=400, detail="Invalid artifact filename")

    base_dir = _gradcam_reports_dir().resolve()
    target = (base_dir / safe_name).resolve()
    try:
        target.relative_to(base_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid artifact path")

    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Grad-CAM artifact not found")

    media_type = _GRADCAM_MEDIA_TYPES.get(target.suffix.lower(), "application/octet-stream")
    return FileResponse(str(target), media_type=media_type, filename=target.name)


@app.post("/predict/image")
async def predict_image_endpoint(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        image_bytes = await file.read()
        probs, label, conf, gradcam_path = predict_image(image_bytes)
        ranked = [
            {
                "label": str(k),
                "probability": round(float(v), 6),
                "percentage": round(float(v) * 100.0, 2),
            }
            for k, v in sorted(probs.items(), key=lambda item: item[1], reverse=True)
        ]
        return {
            "label": label,
            "confidence": conf,
            "method": "image_model",
            "probs": probs,
            "gradcam_path": gradcam_path,
            "explain": {
                "gradcam_path": gradcam_path,
                "probability_ranked": ranked,
                "reasoning": (
                    f"Image model predicted {label} with {conf * 100:.1f}% confidence. "
                    "Use Grad-CAM to review lesion regions that most influenced the CNN output."
                ),
                "evidence_quality": "strong" if conf >= 0.85 else "moderate" if conf >= 0.60 else "limited",
            },
        }
    except Exception as ex:
        raise HTTPException(status_code=400, detail=f"Image prediction failed: {ex}")


@app.post("/predict/symptoms")
def predict_symptoms_endpoint(payload: SymptomsRequest) -> Dict[str, Any]:
    try:
        probs, label, conf, top_features, meta = predict_symptoms(payload.symptoms)
        ranked = [
            {
                "label": str(k),
                "probability": round(float(v), 6),
                "percentage": round(float(v) * 100.0, 2),
            }
            for k, v in sorted(probs.items(), key=lambda item: item[1], reverse=True)
        ]
        top_names: list[str] = []
        if isinstance(top_features, list):
            for item in top_features[:4]:
                if isinstance(item, dict):
                    feat = item.get("display_name") or item.get("feature") or item.get("symptom") or item.get("name")
                    if feat:
                        top_names.append(str(feat).replace("_", " "))
                elif isinstance(item, (list, tuple)) and item:
                    top_names.append(str(item[0]).replace("_", " "))
                elif isinstance(item, str):
                    top_names.append(item.replace("_", " "))
        return {
            "label": label,
            "confidence": conf,
            "method": "symptom_model",
            "probs": probs,
            "top_symptoms": top_features,
            "symptom_model_training_mode": meta.get("training_mode", "unknown"),
            "symptom_model_warning": meta.get("warning", ""),
            "explain": {
                "top_symptoms": top_features,
                "probability_ranked": ranked,
                "reasoning": (
                    f"Symptom model predicted {label} with {conf * 100:.1f}% confidence. "
                    + (
                        f"Top contributing symptoms: {', '.join(top_names)}."
                        if top_names
                        else "Prediction is based on the active symptom pattern."
                    )
                ),
                "symptom_model_training_mode": meta.get("training_mode", "unknown"),
                "symptom_model_warning": meta.get("warning", ""),
                "evidence_quality": "strong" if conf >= 0.85 else "moderate" if conf >= 0.60 else "limited",
            },
        }
    except Exception as ex:
        raise HTTPException(status_code=400, detail=f"Symptom prediction failed: {ex}")


@app.post("/predict/full")
async def predict_full_endpoint(
    payload: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None),
) -> Dict[str, Any]:
    import ast
    import json
    import re

    def _parse_payload(raw_payload: str) -> Dict[str, Any]:
        raw = raw_payload.strip()
        if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
            raw = raw[1:-1]

        # 1) strict JSON
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        # 2) python literal style
        try:
            obj = ast.literal_eval(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        # 3) relaxed json-like fix for unquoted keys: {symptoms:{fever:1}}
        relaxed = re.sub(r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)', r'\1"\2"\3', raw)
        relaxed = re.sub(r"'", '"', relaxed)
        obj = json.loads(relaxed)
        if isinstance(obj, dict):
            return obj
        raise ValueError("Payload is not a JSON object.")

    symptoms: Dict[str, Any] = {}
    if payload:
        try:
            body = _parse_payload(payload)
            symptoms = body.get("symptoms", body if isinstance(body, dict) else {})
            if not isinstance(symptoms, dict):
                symptoms = {}
        except Exception as ex:
            raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {ex}")

    image_bytes = None
    if file is not None:
        image_bytes = await file.read()

    if image_bytes is None and not symptoms:
        raise HTTPException(status_code=400, detail="Provide image file and/or symptoms payload.")

    try:
        return predict_full(image_bytes=image_bytes, symptoms_dict=symptoms)
    except Exception as ex:
        raise HTTPException(status_code=400, detail=f"Hybrid prediction failed: {ex}")

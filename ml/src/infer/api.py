from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.infer.predict import predict_full, predict_image, predict_symptoms


class SymptomsRequest(BaseModel):
    symptoms: Dict[str, Any] = Field(default_factory=dict)


class FullRequest(BaseModel):
    symptoms: Dict[str, Any] = Field(default_factory=dict)


app = FastAPI(
    title="Livestock Hybrid Disease Screening API",
    version="1.0.0",
    description="Hybrid inference API for Normal/LSD/FMD/ECF/CBPP with provenance and explainability payloads.",
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict/image")
async def predict_image_endpoint(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        image_bytes = await file.read()
        probs, label, conf, gradcam_path = predict_image(image_bytes)
        return {
            "label": label,
            "confidence": conf,
            "method": "image_model",
            "probs": probs,
            "gradcam_path": gradcam_path,
        }
    except Exception as ex:
        raise HTTPException(status_code=400, detail=f"Image prediction failed: {ex}")


@app.post("/predict/symptoms")
def predict_symptoms_endpoint(payload: SymptomsRequest) -> Dict[str, Any]:
    try:
        probs, label, conf, top_features = predict_symptoms(payload.symptoms)
        return {
            "label": label,
            "confidence": conf,
            "method": "symptom_model",
            "probs": probs,
            "top_symptoms": top_features,
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

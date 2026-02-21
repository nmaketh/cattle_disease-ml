import io
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from src.models.fusion import FINAL_LABELS, FusionConfig, fuse_predictions
from src.models.gradcam import make_gradcam, save_gradcam
from src.models.rules_engine import RulesConfig, rules_predict
from src.models.symptom_model import symptom_top_features
from src.utils.io import read_json, read_yaml

_THIS_FILE = Path(__file__).resolve()
_SRC_ROOT = _THIS_FILE.parents[1]
_ML_ROOT = _THIS_FILE.parents[2]
_REPO_ROOT = _THIS_FILE.parents[3]


def _resolve_existing(path: Path) -> Optional[Path]:
    return path if path.exists() else None


def _resolve_config_path() -> Path:
    env_path = os.getenv("ML_CONFIG_PATH")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            Path.cwd() / "ml" / "configs" / "config.yaml",
            _REPO_ROOT / "ml" / "configs" / "config.yaml",
            _ML_ROOT / "configs" / "config.yaml",
        ]
    )
    for c in candidates:
        p = c.resolve()
        if p.exists():
            return p
    raise FileNotFoundError("Could not resolve config.yaml. Set ML_CONFIG_PATH to explicit file path.")


def _resolve_path_from_cfg(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    candidates = [Path.cwd() / p, _REPO_ROOT / p, _ML_ROOT / p]
    for c in candidates:
        resolved = c.resolve()
        if resolved.exists():
            return resolved
    return (_REPO_ROOT / p).resolve()


_CFG_PATH = _resolve_config_path()
_CFG = read_yaml(_CFG_PATH)
_ART = _resolve_path_from_cfg(_CFG["paths"]["artifacts_dir"])


@lru_cache(maxsize=1)
def load_image_model() -> tf.keras.layers.TFSMLayer:
    model_dir = _ART / "image_model"
    if not model_dir.exists():
        raise FileNotFoundError(f"Image SavedModel not found: {model_dir}")
    return tf.keras.layers.TFSMLayer(str(model_dir), call_endpoint="serve")


@lru_cache(maxsize=1)
def load_image_decision_calibration() -> np.ndarray:
    p = _ART / "image_model" / "decision_calibration.json"
    if p.exists():
        try:
            obj = read_json(p)
            arr = np.array(obj.get("class_bias", [1.0, 1.0, 1.0]), dtype=float)
            if arr.shape == (3,):
                return arr
        except Exception:
            pass
    return np.array([1.0, 1.0, 1.0], dtype=float)


@lru_cache(maxsize=1)
def load_symptom_model():
    p = _ART / "symptom_model.pkl"
    if not p.exists():
        return None
    return joblib.load(p)


@lru_cache(maxsize=1)
def load_gradcam_model() -> Optional[tf.keras.Model]:
    best_keras = _ART / "image_model" / "best.keras"
    if best_keras.exists():
        try:
            return tf.keras.models.load_model(str(best_keras), compile=False)
        except Exception:
            return None
    return None


@lru_cache(maxsize=1)
def load_symptom_features() -> list:
    p = _ART / "symptom_features.json"
    return read_json(p) if p.exists() else []


@lru_cache(maxsize=1)
def load_symptom_metadata() -> Dict[str, object]:
    p = _ART / "symptom_model_metadata.json"
    if p.exists():
        return read_json(p)
    return {}


def _symptom_reliability_from_metadata(meta: Dict[str, object]) -> float:
    mode = str(meta.get("training_mode", "unknown")).strip().lower()
    if mode == "real_only":
        return float(_CFG["fusion"].get("symptom_reliability_real", 1.0))
    if mode == "bootstrap_weak":
        return float(_CFG["fusion"].get("symptom_reliability_bootstrap", 0.2))
    return float(_CFG["fusion"].get("symptom_reliability_unknown", 0.5))


def preprocess_image(image_bytes: bytes) -> tf.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img)
    arr = tf.image.resize(arr, [_CFG["image"]["img_size"], _CFG["image"]["img_size"]]).numpy().astype("float32")
    return tf.convert_to_tensor(arr[None, ...], dtype=tf.float32)


def _unwrap_output(out):
    if isinstance(out, dict):
        return list(out.values())[0]
    return out


def predict_image(image_bytes: bytes) -> Tuple[Dict[str, float], str, float, Optional[str]]:
    model = load_image_model()
    x = preprocess_image(image_bytes)
    out = _unwrap_output(model(x))
    probs = np.array(out)[0]
    class_bias = load_image_decision_calibration()
    probs = probs * class_bias
    probs = probs / max(float(np.sum(probs)), 1e-8)

    labels = ["Normal", "LSD", "FMD"]
    prob_map = {labels[i]: float(probs[i]) for i in range(3)}
    label = max(prob_map, key=prob_map.get)
    conf = float(prob_map[label])

    # Grad-CAM best-effort for API path.
    grad_path = None
    try:
        grad_model = load_gradcam_model()
        if grad_model is None:
            inp = tf.keras.Input(shape=(_CFG["image"]["img_size"], _CFG["image"]["img_size"], 3), name="image")
            wrapped_out = _unwrap_output(model(inp))
            grad_model = tf.keras.Model(inp, wrapped_out)

        heat = make_gradcam(grad_model, x)
        arr_rgb = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        grad_dir = _ART / "reports" / "gradcam_examples"
        grad_dir.mkdir(parents=True, exist_ok=True)
        grad_path = save_gradcam(arr_rgb, heat, grad_dir / f"predict_{label}.png")
    except Exception:
        grad_path = None
    return prob_map, label, conf, grad_path


def predict_symptoms(symptoms_dict: Dict[str, object]) -> Tuple[Dict[str, float], str, float, list, Dict[str, object]]:
    model = load_symptom_model()
    features = load_symptom_features()
    metadata = load_symptom_metadata()

    if model is None or not features:
        return {"Normal": 1.0}, "Normal", 0.5, [], metadata

    row = {f: int(float(symptoms_dict.get(f, 0)) > 0) for f in features}
    x = pd.DataFrame([row])[features]
    probs = model.predict_proba(x.values)[0]
    labels = list(model.classes_)
    prob_map = {labels[i]: float(probs[i]) for i in range(len(labels))}
    label = max(prob_map, key=prob_map.get)
    conf = float(prob_map[label])
    top = symptom_top_features(model, x.iloc[0], features, top_k=8)
    return prob_map, label, conf, top, metadata


def predict_full(image_bytes: Optional[bytes] = None, symptoms_dict: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    symptoms_dict = symptoms_dict or {}
    image_probs = None
    symptom_probs = None
    gradcam_path = None
    top_sym = None

    if image_bytes is not None:
        image_probs, _, _, gradcam_path = predict_image(image_bytes)

    if symptoms_dict:
        symptom_probs, _, _, top_sym, symptom_meta = predict_symptoms(symptoms_dict)
    else:
        symptom_meta = load_symptom_metadata()
    symptom_reliability = _symptom_reliability_from_metadata(symptom_meta)

    rule_cfg = RulesConfig(
        ecf_weights=_CFG["rules"]["ecf_weights"],
        cbpp_weights=_CFG["rules"]["cbpp_weights"],
        disease_symptom_catalog=_CFG["rules"].get("disease_symptom_catalog", {}),
    )
    rule_obj = rules_predict(symptoms_dict, rule_cfg)

    fusion_cfg = FusionConfig(
        image_weight=float(_CFG["fusion"]["image_weight"]),
        symptom_weight=float(_CFG["fusion"]["symptom_weight"]),
        temperature=float(_CFG["fusion"]["temperature"]),
        symptom_only_min_reliability=float(_CFG["fusion"].get("symptom_only_min_reliability", 0.6)),
        image_confidence_hi=float(_CFG["image"]["image_confidence_hi"]),
        rule_threshold=float(_CFG["fusion"]["rule_threshold"]),
        contradiction_threshold=float(_CFG["fusion"]["contradiction_threshold"]),
        urgent_rule_score=float(_CFG["thresholds"]["urgent_rule_score"]),
        low_confidence=float(_CFG["thresholds"]["low_confidence"]),
    )

    result = fuse_predictions(
        image_probs=image_probs,
        symptom_probs=symptom_probs,
        rule_scores=rule_obj["scores"],
        rule_triggers=rule_obj["explanation"],
        cfg=fusion_cfg,
        gradcam_path=gradcam_path,
        top_symptoms=top_sym,
        symptom_reliability=symptom_reliability,
    )

    # guarantee full label support
    for lbl in FINAL_LABELS:
        result["probs"].setdefault(lbl, 0.0)
    total = sum(result["probs"].values())
    if total > 0:
        result["probs"] = {k: float(v) / total for k, v in result["probs"].items()}

    result.setdefault("explain", {})
    result["explain"]["disease_symptom_catalog"] = rule_obj.get("disease_symptom_catalog", {})
    result["explain"]["catalog_match_scores"] = rule_obj.get("catalog_scores", {})
    result["explain"]["clinical_advisories"] = rule_obj.get("advisories", {})
    result["explain"]["symptom_model_training_mode"] = symptom_meta.get("training_mode", "unknown")
    result["explain"]["symptom_model_warning"] = symptom_meta.get("warning", "")
    result["explain"]["symptom_reliability"] = float(symptom_reliability)
    result["explain"]["symptom_advisory_only"] = bool(
        image_bytes is None and symptom_reliability < float(_CFG["fusion"].get("symptom_only_min_reliability", 0.6))
    )

    return result

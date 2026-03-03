import io
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from src.data.feature_engineering import add_engineered_features
from src.models.fusion import FINAL_LABELS, FusionConfig, fuse_predictions
from src.models.gradcam import make_gradcam, save_gradcam
from src.models.rules_engine import RulesConfig, _symptom_present, rules_predict
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


def _resolve_artifacts_dir() -> Path:
    env_dir = os.getenv("ML_ARTIFACTS_DIR", "").strip()
    if env_dir:
        candidate = Path(env_dir)
        if not candidate.is_absolute():
            candidate = (_REPO_ROOT / candidate).resolve()
        if candidate.exists():
            return candidate
    return _resolve_path_from_cfg(_CFG["paths"]["artifacts_dir"])


_CFG_PATH = _resolve_config_path()
_CFG = read_yaml(_CFG_PATH)
_ART = _resolve_artifacts_dir()


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


def _humanize_key(value: str) -> str:
    return str(value).replace("_", " ").strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_top_symptoms(top_symptoms: Optional[list]) -> list[Dict[str, float | str]]:
    normalized: list[Dict[str, float | str]] = []
    if not isinstance(top_symptoms, list):
        return normalized
    for item in top_symptoms:
        feature = ""
        importance = 0.0
        if isinstance(item, dict):
            feature = str(item.get("symptom") or item.get("feature") or item.get("name") or "").strip()
            importance = _safe_float(item.get("importance") or item.get("score") or 0.0)
        elif isinstance(item, (tuple, list)) and item:
            feature = str(item[0]).strip()
            if len(item) > 1:
                importance = _safe_float(item[1])
        elif isinstance(item, str):
            feature = item.strip()
        if not feature:
            continue
        normalized.append(
            {
                "feature": feature,
                "symptom": feature,
                "importance": float(importance),
                "display_name": _humanize_key(feature),
            }
        )
    return normalized


def _sorted_prob_entries(probs: Optional[Dict[str, float]]) -> list[Dict[str, float | str]]:
    if not isinstance(probs, dict):
        return []
    ranked = sorted(probs.items(), key=lambda kv: float(kv[1]), reverse=True)
    return [
        {
            "label": str(label),
            "probability": round(float(score), 6),
            "percentage": round(float(score) * 100.0, 2),
        }
        for label, score in ranked
    ]


def _top_label_and_confidence(probs: Optional[Dict[str, float]]) -> Tuple[Optional[str], Optional[float]]:
    if not isinstance(probs, dict) or not probs:
        return None, None
    label = max(probs, key=lambda k: float(probs.get(k, 0.0)))
    return str(label), float(probs[label])


def _confidence_band(confidence: float) -> str:
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.60:
        return "moderate"
    return "low"


def _active_symptom_keys(symptoms_dict: Dict[str, object]) -> list[str]:
    out: list[str] = []
    for key, value in (symptoms_dict or {}).items():
        try:
            present = bool(float(value) > 0)
        except Exception:
            if isinstance(value, str):
                present = value.strip().lower() in {"yes", "true", "1", "present", "positive"}
            else:
                present = bool(value)
        if present:
            out.append(str(key))
    return sorted(set(out))


def _rule_score_breakdown(
    symptoms_dict: Dict[str, object],
    rule_cfg: RulesConfig,
) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for disease, weights in {
        "ECF": rule_cfg.ecf_weights,
        "CBPP": rule_cfg.cbpp_weights,
    }.items():
        total_weight = float(sum(float(w) for w in weights.values()))
        matched_items: list[Dict[str, float | str]] = []
        missing_items: list[Dict[str, float | str]] = []
        matched_weight = 0.0
        for symptom, weight in weights.items():
            weight_f = float(weight)
            present = _symptom_present(symptoms_dict, symptom)
            (matched_items if present else missing_items).append(
                {
                    "symptom": str(symptom),
                    "display_name": _humanize_key(symptom),
                    "weight": round(weight_f, 4),
                }
            )
            if present:
                matched_weight += weight_f
        out[disease] = {
            "matched_weight": round(matched_weight, 4),
            "total_weight": round(total_weight, 4),
            "coverage": round((matched_weight / total_weight) if total_weight > 0 else 0.0, 4),
            "matched_symptoms": matched_items,
            "missing_symptoms": missing_items,
        }
    return out


def _catalog_evidence_for_label(
    label: str,
    symptoms_dict: Dict[str, object],
    catalog: Dict[str, Dict[str, List[str]]],
) -> Dict[str, object]:
    if not catalog:
        return {}
    key = str(label or "").strip()
    label_catalog = catalog.get(key) or catalog.get(key.upper()) or catalog.get(key.lower()) or {}
    if not isinstance(label_catalog, dict):
        return {}
    core = [str(s) for s in (label_catalog.get("core") or [])]
    supporting = [str(s) for s in (label_catalog.get("supporting") or [])]
    core_matched = [s for s in core if _symptom_present(symptoms_dict, s)]
    sup_matched = [s for s in supporting if _symptom_present(symptoms_dict, s)]
    return {
        "core_matched": core_matched,
        "core_missing": [s for s in core if s not in core_matched],
        "supporting_matched": sup_matched,
        "supporting_missing": [s for s in supporting if s not in sup_matched],
        "core_coverage": round((len(core_matched) / len(core)) if core else 0.0, 4),
        "supporting_coverage": round((len(sup_matched) / len(supporting)) if supporting else 0.0, 4),
    }


def _build_reasoning_text(
    *,
    result: Dict[str, object],
    image_label: Optional[str],
    image_conf: Optional[float],
    symptom_label: Optional[str],
    symptom_conf: Optional[float],
    top_symptoms: list[Dict[str, float | str]],
    rule_obj: Dict[str, object],
    catalog_evidence: Dict[str, object],
    symptom_reliability: float,
    had_image: bool,
    had_symptoms: bool,
) -> Tuple[str, list[str], list[str], str]:
    final_label = str(result.get("final_label") or "Unknown")
    confidence = _safe_float(result.get("confidence"), 0.0)
    method = str(result.get("method") or "unknown")
    explain = result.get("explain") if isinstance(result.get("explain"), dict) else {}
    rule_triggers = explain.get("rule_triggers") if isinstance(explain, dict) else {}
    top_entries = _sorted_prob_entries(result.get("probs") if isinstance(result.get("probs"), dict) else {})
    runner_up = top_entries[1] if len(top_entries) > 1 else None
    support: list[str] = []
    cautions: list[str] = []

    if had_image and image_label and image_conf is not None:
        image_agrees = image_label == final_label
        support.append(
            f"Image model {'agrees with' if image_agrees else 'suggested'} {image_label} ({image_conf * 100:.1f}%)."
        )
        if not image_agrees and image_conf >= 0.65:
            cautions.append(
                f"Image evidence leaned toward {image_label} ({image_conf * 100:.1f}%), so review photo quality and lesion visibility."
            )
    else:
        cautions.append("No image evidence was provided, so image-based explainability and Grad-CAM are unavailable.")

    if had_symptoms and symptom_label and symptom_conf is not None:
        symptom_agrees = symptom_label == final_label
        support.append(
            f"Symptom model {'supports' if symptom_agrees else 'ranked'} {symptom_label} ({symptom_conf * 100:.1f}%) with reliability {symptom_reliability:.2f}."
        )
    elif not had_symptoms:
        cautions.append("No symptom inputs were provided; decision is driven by image/rules only.")

    top_symptom_names = [str(item.get("display_name") or item.get("feature") or item.get("symptom")) for item in top_symptoms[:3]]
    if top_symptom_names:
        support.append(f"Top symptom evidence: {', '.join(top_symptom_names)}.")

    final_label_triggers: list[str] = []
    if isinstance(rule_triggers, dict):
        raw = rule_triggers.get(final_label) or rule_triggers.get(final_label.upper()) or rule_triggers.get(final_label.lower())
        if isinstance(raw, list):
            final_label_triggers = [str(t) for t in raw]
    if final_label_triggers:
        support.append(
            "Clinical rule triggers for the predicted disease: "
            + ", ".join(_humanize_key(t) for t in final_label_triggers[:4])
            + "."
        )

    catalog_scores = rule_obj.get("catalog_scores") if isinstance(rule_obj, dict) else {}
    if isinstance(catalog_scores, dict):
        score = catalog_scores.get(final_label) or catalog_scores.get(final_label.upper()) or catalog_scores.get(final_label.lower())
        if score is not None:
            support.append(f"Catalog match score for {final_label}: {float(score) * 100:.1f}%.")

    core_missing = catalog_evidence.get("core_missing") if isinstance(catalog_evidence, dict) else []
    if isinstance(core_missing, list) and core_missing:
        cautions.append(
            "Some expected core signs for "
            + final_label
            + " were not reported: "
            + ", ".join(_humanize_key(s) for s in core_missing[:3])
            + "."
        )

    if runner_up and isinstance(runner_up, dict):
        runner_prob = _safe_float(runner_up.get("probability"), 0.0)
        if confidence - runner_prob < 0.15:
            cautions.append(
                f"Differential remains close: {runner_up.get('label')} is also plausible ({runner_prob * 100:.1f}%)."
            )

    if confidence < 0.60:
        cautions.append("Model confidence is low; prioritize re-photo, follow-up observation, and veterinary review.")

    if method == "clinical_rules":
        support.append("Final decision was rule-driven, which improves transparency but may require confirmatory testing.")
    elif method == "hybrid":
        support.append("Final decision fused image, symptom, and clinical-rule evidence.")
    elif method == "image_model":
        support.append("Final decision was primarily driven by image evidence with rules attached for safety checks.")
    elif method == "symptom_model":
        support.append("Final decision was symptom-model driven with rule safeguards.")

    evidence_quality = (
        "strong"
        if confidence >= 0.8 and len(cautions) <= 1
        else "moderate"
        if confidence >= 0.6
        else "limited"
    )

    summary = (
        f"{final_label} predicted with {confidence * 100:.1f}% confidence using {method.replace('_', ' ')}. "
        f"{' '.join(support[:2])}".strip()
    )
    if cautions:
        summary += " " + cautions[0]
    return summary.strip(), support, cautions, evidence_quality


def _build_modality_contributions(
    *,
    method: str,
    had_image: bool,
    had_symptoms: bool,
    image_conf: Optional[float],
    symptom_conf: Optional[float],
    symptom_reliability: float,
    rule_scores: Dict[str, float],
    fusion_cfg: FusionConfig,
) -> Dict[str, object]:
    max_rule = max(_safe_float(rule_scores.get("ECF"), 0.0), _safe_float(rule_scores.get("CBPP"), 0.0))
    if method == "image_model":
        image_share, symptom_share, rule_share = 0.85, 0.0, 0.15
    elif method == "hybrid":
        symptom_share = round(float(fusion_cfg.symptom_weight) * float(max(0.0, min(1.0, symptom_reliability))), 4)
        image_share = round(float(fusion_cfg.image_weight), 4)
        total = max(image_share + symptom_share, 1e-8)
        image_share /= total
        symptom_share /= total
        rule_share = 0.0 if max_rule < 0.05 else min(0.35, round(max_rule, 4))
    elif method == "symptom_model":
        image_share, symptom_share, rule_share = 0.0, max(0.5, float(symptom_reliability)), min(0.4, max_rule)
    else:
        image_share, symptom_share, rule_share = 0.0, 0.0, 1.0 if max_rule > 0 else 0.0

    return {
        "image_available": bool(had_image),
        "symptoms_available": bool(had_symptoms),
        "method": method,
        "config_weights": {
            "image_weight": float(fusion_cfg.image_weight),
            "symptom_weight": float(fusion_cfg.symptom_weight),
            "temperature": float(fusion_cfg.temperature),
        },
        "estimated_contributions": {
            "image": round(float(image_share), 4),
            "symptoms": round(float(symptom_share), 4),
            "rules": round(float(rule_share), 4),
        },
        "modality_confidence": {
            "image": None if image_conf is None else round(float(image_conf), 4),
            "symptoms": None if symptom_conf is None else round(float(symptom_conf), 4),
            "rules_max": round(float(max_rule), 4),
        },
        "symptom_reliability": round(float(symptom_reliability), 4),
    }


def _enrich_explainability(
    *,
    result: Dict[str, object],
    symptoms_dict: Dict[str, object],
    image_bytes_present: bool,
    image_probs: Optional[Dict[str, float]],
    image_label: Optional[str],
    image_conf: Optional[float],
    symptom_probs: Optional[Dict[str, float]],
    symptom_label: Optional[str],
    symptom_conf: Optional[float],
    top_symptoms: Optional[list],
    rule_obj: Dict[str, object],
    rule_cfg: RulesConfig,
    symptom_meta: Dict[str, object],
    symptom_reliability: float,
    fusion_cfg: FusionConfig,
) -> Dict[str, object]:
    explain = result.setdefault("explain", {})
    if not isinstance(explain, dict):
        explain = {}
        result["explain"] = explain

    normalized_top_symptoms = _normalize_top_symptoms(top_symptoms)
    explain["top_symptoms"] = normalized_top_symptoms

    final_label = str(result.get("final_label") or "Unknown")
    final_probs = result.get("probs") if isinstance(result.get("probs"), dict) else {}
    ranked_probs = _sorted_prob_entries(final_probs)
    active_symptoms = _active_symptom_keys(symptoms_dict)
    catalog = rule_obj.get("disease_symptom_catalog") if isinstance(rule_obj.get("disease_symptom_catalog"), dict) else {}
    catalog_evidence = _catalog_evidence_for_label(final_label, symptoms_dict, catalog)
    rule_scores = rule_obj.get("scores") if isinstance(rule_obj.get("scores"), dict) else {}
    rule_breakdown = _rule_score_breakdown(symptoms_dict, rule_cfg)
    reasoning_text, support, cautions, evidence_quality = _build_reasoning_text(
        result=result,
        image_label=image_label,
        image_conf=image_conf,
        symptom_label=symptom_label,
        symptom_conf=symptom_conf,
        top_symptoms=normalized_top_symptoms,
        rule_obj=rule_obj,
        catalog_evidence=catalog_evidence,
        symptom_reliability=symptom_reliability,
        had_image=bool(image_bytes_present and image_probs),
        had_symptoms=bool(symptoms_dict),
    )

    runner_up = ranked_probs[1]["label"] if len(ranked_probs) > 1 else None
    runner_up_prob = ranked_probs[1]["probability"] if len(ranked_probs) > 1 else None
    method = str(result.get("method") or "unknown")
    modality_contributions = _build_modality_contributions(
        method=method,
        had_image=bool(image_bytes_present and image_probs),
        had_symptoms=bool(symptoms_dict),
        image_conf=image_conf,
        symptom_conf=symptom_conf,
        symptom_reliability=symptom_reliability,
        rule_scores={k: _safe_float(v, 0.0) for k, v in rule_scores.items()} if isinstance(rule_scores, dict) else {},
        fusion_cfg=fusion_cfg,
    )

    explain["reasoning"] = reasoning_text
    explain["supporting_evidence"] = support
    explain["cautionary_evidence"] = cautions
    explain["evidence_quality"] = evidence_quality
    explain["confidence_band"] = _confidence_band(_safe_float(result.get("confidence"), 0.0))
    explain["probability_ranked"] = ranked_probs
    explain["input_summary"] = {
        "has_image": bool(image_bytes_present and image_probs),
        "symptom_count": len(active_symptoms),
        "active_symptoms": active_symptoms,
        "active_symptoms_preview": [_humanize_key(s) for s in active_symptoms[:8]],
    }
    explain["modality_outputs"] = {
        "image_model": {
            "available": bool(image_probs),
            "label": image_label,
            "confidence": None if image_conf is None else round(float(image_conf), 4),
            "probs": image_probs or {},
        },
        "symptom_model": {
            "available": bool(symptom_probs),
            "label": symptom_label,
            "confidence": None if symptom_conf is None else round(float(symptom_conf), 4),
            "probs": symptom_probs or {},
            "top_features": normalized_top_symptoms,
            "training_mode": symptom_meta.get("training_mode", "unknown"),
        },
        "clinical_rules": {
            "scores": rule_scores if isinstance(rule_scores, dict) else {},
            "triggers": explain.get("rule_triggers") or {},
            "candidate_labels": rule_obj.get("candidate_labels", []),
            "catalog_match_scores": rule_obj.get("catalog_scores", {}),
        },
    }
    explain["modality_contributions"] = modality_contributions
    explain["rule_score_breakdown"] = rule_breakdown
    explain["predicted_disease_catalog_evidence"] = catalog_evidence
    explain["differential_summary"] = {
        "runner_up_label": runner_up,
        "runner_up_probability": runner_up_prob,
        "close_differential": bool(
            isinstance(runner_up_prob, (int, float))
            and (_safe_float(result.get("confidence"), 0.0) - float(runner_up_prob)) < 0.15
        ),
    }
    explain["modality_summary"] = (
        f"{str(method).replace('_', ' ').title()} decision with estimated contributions "
        f"(image {float(modality_contributions['estimated_contributions']['image']) * 100:.0f}%, "
        f"symptoms {float(modality_contributions['estimated_contributions']['symptoms']) * 100:.0f}%, "
        f"rules {float(modality_contributions['estimated_contributions']['rules']) * 100:.0f}%)."
    )
    explain["explanation_version"] = "2.0"
    return explain


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
    all_features = load_symptom_features()
    metadata = load_symptom_metadata()

    if model is None or not all_features:
        return {"Normal": 1.0}, "Normal", 0.5, [], metadata

    # ── Step 1: Build a row for base (non-engineered) features ───────────────
    # Use actual float values (not binarised) so that numeric features such as
    # days_since_onset, age_months, and days_off_feed are preserved correctly.
    base_features = [f for f in all_features if not f.startswith("fe_")]
    base_row: Dict[str, float] = {f: float(symptoms_dict.get(f, 0)) for f in base_features}

    # ── Step 2: Apply the same feature engineering used at training time ──────
    df_row = pd.DataFrame([base_row])
    df_row = add_engineered_features(df_row)

    # ── Step 3: Build the final feature vector in model-expected order ────────
    # Fill any fe_* column that was not produced (e.g., if a base feature was
    # absent from the incoming dict) with 0.0.
    final_row: Dict[str, float] = {}
    for f in all_features:
        if f in df_row.columns:
            final_row[f] = float(df_row[f].iloc[0])
        else:
            final_row[f] = 0.0

    x = pd.DataFrame([final_row])[all_features]
    probs = model.predict_proba(x.values)[0]
    labels = list(model.classes_)
    prob_map = {labels[i]: float(probs[i]) for i in range(len(labels))}
    label = max(prob_map, key=prob_map.get)
    conf = float(prob_map[label])
    # symptom_top_features already filters out fe_* columns internally
    top = symptom_top_features(model, x.iloc[0], all_features, top_k=8)
    return prob_map, label, conf, top, metadata


def predict_full(image_bytes: Optional[bytes] = None, symptoms_dict: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    symptoms_dict = symptoms_dict or {}
    image_probs = None
    image_label = None
    image_conf = None
    symptom_probs = None
    symptom_label = None
    symptom_conf = None
    gradcam_path = None
    top_sym = None

    if image_bytes is not None:
        image_probs, image_label, image_conf, gradcam_path = predict_image(image_bytes)

    if symptoms_dict:
        symptom_probs, symptom_label, symptom_conf, top_sym, symptom_meta = predict_symptoms(symptoms_dict)
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

    _enrich_explainability(
        result=result,
        symptoms_dict=symptoms_dict,
        image_bytes_present=image_bytes is not None,
        image_probs=image_probs,
        image_label=image_label,
        image_conf=image_conf,
        symptom_probs=symptom_probs,
        symptom_label=symptom_label,
        symptom_conf=symptom_conf,
        top_symptoms=top_sym,
        rule_obj=rule_obj,
        rule_cfg=rule_cfg,
        symptom_meta=symptom_meta,
        symptom_reliability=symptom_reliability,
        fusion_cfg=fusion_cfg,
    )

    return result

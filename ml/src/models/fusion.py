from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

FINAL_LABELS = ["Normal", "LSD", "FMD", "ECF", "CBPP"]
IMAGE_LABELS = ["Normal", "LSD", "FMD"]


@dataclass
class FusionConfig:
    image_weight: float
    symptom_weight: float
    temperature: float
    symptom_only_min_reliability: float
    image_confidence_hi: float
    rule_threshold: float
    contradiction_threshold: float
    urgent_rule_score: float
    low_confidence: float


def _softmax_temperature(probs: Dict[str, float], labels, temp: float) -> Dict[str, float]:
    arr = np.array([float(probs.get(lbl, 0.0)) for lbl in labels], dtype=float)
    arr = np.clip(arr, 1e-8, 1.0)
    logits = np.log(arr)
    scaled = logits / max(temp, 1e-3)
    exp = np.exp(scaled - np.max(scaled))
    out = exp / np.sum(exp)
    return {lbl: float(out[i]) for i, lbl in enumerate(labels)}


def _zeros() -> Dict[str, float]:
    return {lbl: 0.0 for lbl in FINAL_LABELS}


def fuse_predictions(
    image_probs: Optional[Dict[str, float]],
    symptom_probs: Optional[Dict[str, float]],
    rule_scores: Dict[str, float],
    rule_triggers: Dict[str, list],
    cfg: FusionConfig,
    gradcam_path: Optional[str] = None,
    top_symptoms: Optional[list] = None,
    symptom_reliability: float = 1.0,
) -> Dict[str, object]:
    final = _zeros()

    # Path 1: high-confidence image for Normal/LSD/FMD (rules attached but not override)
    if image_probs:
        image_cal = _softmax_temperature(image_probs, IMAGE_LABELS, cfg.temperature)
        best_img = max(image_cal, key=image_cal.get)
        best_img_score = image_cal[best_img]
        if best_img_score >= cfg.image_confidence_hi:
            for lbl in IMAGE_LABELS:
                final[lbl] = image_cal[lbl]
            final["ECF"] = float(rule_scores.get("ECF", 0.0)) * 0.25
            final["CBPP"] = float(rule_scores.get("CBPP", 0.0)) * 0.25
            total = sum(final.values())
            if total > 0:
                final = {k: v / total for k, v in final.items()}
            max_rule = max(rule_scores.get("ECF", 0.0), rule_scores.get("CBPP", 0.0))
            # For very strong image evidence, only raise urgent if rules are also very strong.
            urgent = max_rule >= cfg.urgent_rule_score
            if best_img_score >= 0.95 and max_rule < 0.90:
                urgent = False
            return {
                "final_label": best_img,
                "confidence": float(final.get(best_img, best_img_score)),
                "method": "image_model",
                "probs": final,
                "explain": {
                    "gradcam_path": gradcam_path,
                    "top_symptoms": top_symptoms,
                    "rule_triggers": rule_triggers,
                },
                "recommendation_flags": {
                    "retake_image": float(best_img_score) < 0.9,
                    "contact_vet_urgent": urgent,
                },
            }

    # Path 2: hybrid weighted average where both modalities exist
    if image_probs and symptom_probs:
        image_cal = _softmax_temperature(image_probs, IMAGE_LABELS, cfg.temperature)
        symptom_cal = _softmax_temperature(symptom_probs, list(symptom_probs.keys()), cfg.temperature)
        best_img = max(image_cal, key=image_cal.get)
        best_img_score = float(image_cal[best_img])
        symptom_weight = float(cfg.symptom_weight) * float(max(0.0, min(1.0, symptom_reliability)))
        image_weight = float(cfg.image_weight)

        # If symptom model lacks Normal class, avoid biasing Normal cases toward disease.
        if "Normal" not in symptom_probs:
            symptom_weight = min(symptom_weight, 0.15)
            image_weight = max(image_weight, 0.85)
        if best_img == "Normal":
            symptom_weight = min(symptom_weight, 0.05)
            image_weight = max(image_weight, 0.95)

        for lbl in IMAGE_LABELS:
            final[lbl] += image_weight * image_cal.get(lbl, 0.0)
            final[lbl] += symptom_weight * symptom_cal.get(lbl, 0.0)

        for lbl in ["ECF", "CBPP"]:
            final[lbl] = max(final[lbl], float(rule_scores.get(lbl, 0.0)))

        total = sum(final.values())
        if total > 0:
            final = {k: v / total for k, v in final.items()}

        best_label = max(final, key=final.get)
        confidence = float(final[best_label])

        # Do not override strong LSD/FMD/Normal image signal unless rules are urgent-level.
        max_rule = max(rule_scores.get("ECF", 0.0), rule_scores.get("CBPP", 0.0))
        if best_img_score >= cfg.contradiction_threshold and max_rule < cfg.urgent_rule_score:
            best_label = best_img
            confidence = best_img_score

        return {
            "final_label": best_label,
            "confidence": confidence,
            "method": "hybrid",
            "probs": final,
            "explain": {
                "gradcam_path": gradcam_path,
                "top_symptoms": top_symptoms,
                "rule_triggers": rule_triggers,
            },
            "recommendation_flags": {
                "retake_image": confidence < cfg.low_confidence,
                "contact_vet_urgent": max(rule_scores.get("ECF", 0.0), rule_scores.get("CBPP", 0.0)) >= cfg.urgent_rule_score,
            },
        }

    # Path 3: symptom-only / clinical-rules-driven
    if symptom_probs:
        min_rel = float(max(0.0, min(1.0, cfg.symptom_only_min_reliability)))
        reliability = float(max(0.0, min(1.0, symptom_reliability)))
        max_rule = max(rule_scores.get("ECF", 0.0), rule_scores.get("CBPP", 0.0))

        # If symptom model is weak (e.g., bootstrap_weak), treat symptoms as advisory only
        # and rely on transparent clinical rules for final decision in symptom-only flow.
        if reliability < min_rel:
            if max_rule >= cfg.rule_threshold:
                best_label = "ECF" if rule_scores.get("ECF", 0.0) >= rule_scores.get("CBPP", 0.0) else "CBPP"
                final = _zeros()
                final["ECF"] = float(rule_scores.get("ECF", 0.0))
                final["CBPP"] = float(rule_scores.get("CBPP", 0.0))
                rem = max(0.0, 1.0 - (final["ECF"] + final["CBPP"]))
                final["Normal"] = rem
                total = sum(final.values())
                if total > 0:
                    final = {k: v / total for k, v in final.items()}
                return {
                    "final_label": best_label,
                    "confidence": float(final[best_label]),
                    "method": "clinical_rules",
                    "probs": final,
                    "explain": {
                        "gradcam_path": gradcam_path,
                        "top_symptoms": top_symptoms,
                        "rule_triggers": rule_triggers,
                    },
                    "recommendation_flags": {
                        "retake_image": True,
                        "contact_vet_urgent": max_rule >= cfg.urgent_rule_score,
                    },
                }

            # No strong rule evidence and weak symptom model: avoid overcalling disease.
            final = _zeros()
            # Keep a deliberately uncertain distribution to avoid false certainty.
            final["Normal"] = 0.5
            final["LSD"] = 0.25
            final["FMD"] = 0.25
            return {
                "final_label": "Normal",
                "confidence": float(final["Normal"]),
                "method": "clinical_rules",
                "probs": final,
                "explain": {
                    "gradcam_path": gradcam_path,
                    "top_symptoms": top_symptoms,
                    "rule_triggers": rule_triggers,
                },
                "recommendation_flags": {
                    "retake_image": True,
                    "contact_vet_urgent": False,
                },
            }

        symptom_cal = _softmax_temperature(symptom_probs, list(symptom_probs.keys()), cfg.temperature)
        for lbl in FINAL_LABELS:
            final[lbl] = reliability * float(symptom_cal.get(lbl, 0.0))
        final["Normal"] += (1.0 - reliability)
        for lbl in ["ECF", "CBPP"]:
            final[lbl] = max(final[lbl], float(rule_scores.get(lbl, 0.0)))

        total = sum(final.values())
        if total > 0:
            final = {k: v / total for k, v in final.items()}

        best_label = max(final, key=final.get)
        confidence = float(final[best_label])
        method = "symptom_model"

        if max(rule_scores.get("ECF", 0.0), rule_scores.get("CBPP", 0.0)) >= cfg.rule_threshold:
            best_label = "ECF" if rule_scores.get("ECF", 0.0) >= rule_scores.get("CBPP", 0.0) else "CBPP"
            confidence = float(max(rule_scores.get("ECF", 0.0), rule_scores.get("CBPP", 0.0)))
            method = "clinical_rules"

        return {
            "final_label": best_label,
            "confidence": confidence,
            "method": method,
            "probs": final,
            "explain": {
                "gradcam_path": gradcam_path,
                "top_symptoms": top_symptoms,
                "rule_triggers": rule_triggers,
            },
            "recommendation_flags": {
                "retake_image": True,
                "contact_vet_urgent": max(rule_scores.get("ECF", 0.0), rule_scores.get("CBPP", 0.0)) >= cfg.urgent_rule_score,
            },
        }

    # Path 4: rules only fallback
    final["ECF"] = float(rule_scores.get("ECF", 0.0))
    final["CBPP"] = float(rule_scores.get("CBPP", 0.0))
    if sum(final.values()) == 0:
        final["Normal"] = 1.0
    else:
        s = sum(final.values())
        final = {k: v / s for k, v in final.items()}

    label = max(final, key=final.get)
    return {
        "final_label": label,
        "confidence": float(final[label]),
        "method": "clinical_rules",
        "probs": final,
        "explain": {
            "gradcam_path": gradcam_path,
            "top_symptoms": top_symptoms,
            "rule_triggers": rule_triggers,
        },
        "recommendation_flags": {
            "retake_image": True,
            "contact_vet_urgent": max(rule_scores.get("ECF", 0.0), rule_scores.get("CBPP", 0.0)) >= cfg.urgent_rule_score,
        },
    }

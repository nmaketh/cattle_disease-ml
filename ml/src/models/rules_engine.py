from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class RulesConfig:
    ecf_weights: Dict[str, float]
    cbpp_weights: Dict[str, float]
    disease_symptom_catalog: Dict[str, Dict[str, List[str]]] | None = None


_SYMPTOM_SYNONYMS = {
    "swollen_lymph_nodes": ["enlarged_lymph_nodes", "lymph_node_swelling", "swollen_lymph_nodes"],
    "difficulty_breathing": ["difficulty_breathing", "laboured_breathing", "labored_breathing", "shortness_of_breath"],
    "nasal_discharge": ["nasal_discharge", "runny_nose"],
    "eye_discharge": ["eye_discharge", "ocular_discharge"],
    "coughing": ["coughing", "cough"],
    "chest_pain_signs": ["chest_pain_signs", "painful_breathing", "pleuritic_pain_signs"],
    "rapid_shallow_breathing": ["rapid_shallow_breathing", "tachypnea", "rapid_respiratory_rate"],
    "extended_neck_posture": ["extended_neck_posture", "neck_extended_posture"],
    "head_lowered": ["head_lowered", "lowered_head"],
    "arched_back": ["arched_back", "back_arched"],
    "grunt_on_expiration": ["grunt_on_expiration", "expiratory_grunt"],
    "recumbency": ["recumbency", "lying_down"],
    "weight_loss": ["weight_loss", "loss_of_condition"],
    "painless_lumps": ["painless_lumps", "skin_nodules", "lumpy_skin"],
    "loss_of_appetite": ["loss_of_appetite", "anorexia", "reduced_feed_intake"],
    "mouth_blisters": ["mouth_blisters", "blisters_on_mouth", "oral_vesicles"],
    "tongue_sores": ["tongue_sores", "sores_on_tongue", "tongue_lesions"],
    "foot_lesions": ["foot_lesions", "hoof_lesions", "interdigital_lesions"],
    "drooling": ["drooling", "excess_salivation", "salivation"],
    "lameness": ["lameness", "difficulty_walking"],
}


def _symptom_value(symptoms: Dict[str, object], key: str) -> float:
    val = symptoms.get(key, 0)
    if isinstance(val, str):
        n = val.strip().lower()
        if n in {"yes", "true", "1", "present", "positive"}:
            return 1.0
        if n in {"no", "false", "0", "absent", "negative", ""}:
            return 0.0
    try:
        return 1.0 if float(val) > 0 else 0.0
    except Exception:
        return 0.0


def _symptom_present(symptoms: Dict[str, object], key: str) -> bool:
    aliases = _SYMPTOM_SYNONYMS.get(key, [key])
    return any(_symptom_value(symptoms, alias) > 0 for alias in aliases)


def _score(weights: Dict[str, float], symptoms: Dict[str, object]) -> Tuple[float, List[str]]:
    total = float(sum(weights.values()))
    if total <= 0:
        return 0.0, []
    acc = 0.0
    triggers = []
    for k, w in weights.items():
        if _symptom_present(symptoms, k):
            acc += float(w)
            triggers.append(k)
    score = max(0.0, min(1.0, acc / total))
    return score, triggers


def score_ecf(symptoms_dict: Dict[str, object], cfg: RulesConfig) -> Tuple[float, List[str]]:
    return _score(cfg.ecf_weights, symptoms_dict)


def score_cbpp(symptoms_dict: Dict[str, object], cfg: RulesConfig) -> Tuple[float, List[str]]:
    return _score(cfg.cbpp_weights, symptoms_dict)


def _catalog_match_scores(symptoms_dict: Dict[str, object], catalog: Dict[str, Dict[str, List[str]]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for disease, groups in catalog.items():
        core = groups.get("core", []) or []
        supporting = groups.get("supporting", []) or []

        core_hits = sum(1 for s in core if _symptom_present(symptoms_dict, s))
        sup_hits = sum(1 for s in supporting if _symptom_present(symptoms_dict, s))

        core_score = (core_hits / len(core)) if core else 0.0
        sup_score = (sup_hits / len(supporting)) if supporting else 0.0
        out[disease] = max(0.0, min(1.0, 0.75 * core_score + 0.25 * sup_score))
    return out


def rules_predict(symptoms_dict: Dict[str, object], cfg: RulesConfig) -> Dict[str, object]:
    ecf_score, ecf_triggers = score_ecf(symptoms_dict, cfg)
    cbpp_score, cbpp_triggers = score_cbpp(symptoms_dict, cfg)

    ranked = sorted([("ECF", ecf_score), ("CBPP", cbpp_score)], key=lambda x: x[1], reverse=True)

    catalog = cfg.disease_symptom_catalog or {}
    catalog_scores = _catalog_match_scores(symptoms_dict, catalog) if catalog else {}

    return {
        "candidate_labels": [x[0] for x in ranked],
        "scores": {"ECF": ecf_score, "CBPP": cbpp_score},
        "catalog_scores": catalog_scores,
        "method": "clinical_rules",
        "explanation": {
            "ECF": ecf_triggers,
            "CBPP": cbpp_triggers,
        },
        "disease_symptom_catalog": catalog,
        "advisories": {
            "ECF": "Tick-borne disease; survivors may remain carriers. Confirm with laboratory diagnostics where available.",
            "CBPP": "Humans are not known to be susceptible. Subacute/carrier cattle can occur; confirm with laboratory diagnostics.",
        },
    }

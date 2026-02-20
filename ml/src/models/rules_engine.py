from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class RulesConfig:
    ecf_weights: Dict[str, float]
    cbpp_weights: Dict[str, float]


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


def _score(weights: Dict[str, float], symptoms: Dict[str, object]) -> Tuple[float, List[str]]:
    total = float(sum(weights.values()))
    if total <= 0:
        return 0.0, []
    acc = 0.0
    triggers = []
    for k, w in weights.items():
        if _symptom_value(symptoms, k) > 0:
            acc += float(w)
            triggers.append(k)
    score = max(0.0, min(1.0, acc / total))
    return score, triggers


def score_ecf(symptoms_dict: Dict[str, object], cfg: RulesConfig) -> Tuple[float, List[str]]:
    return _score(cfg.ecf_weights, symptoms_dict)


def score_cbpp(symptoms_dict: Dict[str, object], cfg: RulesConfig) -> Tuple[float, List[str]]:
    return _score(cfg.cbpp_weights, symptoms_dict)


def rules_predict(symptoms_dict: Dict[str, object], cfg: RulesConfig) -> Dict[str, object]:
    ecf_score, ecf_triggers = score_ecf(symptoms_dict, cfg)
    cbpp_score, cbpp_triggers = score_cbpp(symptoms_dict, cfg)

    ranked = sorted([("ECF", ecf_score), ("CBPP", cbpp_score)], key=lambda x: x[1], reverse=True)
    return {
        "candidate_labels": [x[0] for x in ranked],
        "scores": {"ECF": ecf_score, "CBPP": cbpp_score},
        "method": "clinical_rules",
        "explanation": {
            "ECF": ecf_triggers,
            "CBPP": cbpp_triggers,
        },
    }

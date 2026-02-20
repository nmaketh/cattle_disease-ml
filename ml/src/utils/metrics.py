from typing import Dict, Sequence

from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score


def multiclass_metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> Dict[str, object]:
    report = classification_report(y_true, y_pred, labels=list(labels), zero_division=0, output_dict=True)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)),
        "per_class_f1": {
            label: float(report.get(label, {}).get("f1-score", 0.0))
            for label in labels
        },
        "classification_report": report,
    }

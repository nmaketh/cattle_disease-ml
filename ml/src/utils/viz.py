from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def save_confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str], out_path: str | Path, title: str
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(labels))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30)
    ax.set_title(title)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def save_bar(values: Iterable[float], labels: Sequence[str], out_path: str | Path, title: str) -> None:
    vals = np.array(list(values), dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, vals)
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)

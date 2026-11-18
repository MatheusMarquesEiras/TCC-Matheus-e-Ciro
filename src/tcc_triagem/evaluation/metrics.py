"""Métricas quantitativas e artefatos de diagnóstico.

F1 macro é a métrica principal do trabalho: com classes desbalanceadas — no
MedPT top-10 a maior classe tem quase seis vezes a menor — a acurácia sozinha
esconde o desempenho nas classes minoritárias, que são justamente as de saúde
mental onde está o gargalo do modelo.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sem janela — os scripts rodam via terminal

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    """Métricas macro e F1 por classe."""
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4
        ),
        "precision_macro": round(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4
        ),
        "recall_macro": round(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4
        ),
        "f1_per_class": {
            label: round(float(score), 4)
            for label, score in zip(
                labels,
                f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
            )
        },
        "n_examples": len(y_true),
    }


def save_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    out_path: Path,
    title: str,
) -> Path:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    size = max(8, len(labels))
    fig, ax = plt.subplots(figsize=(size, size - 1))
    sns.heatmap(
        cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues", ax=ax
    )
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_classification_report(
    y_true: list[str], y_pred: list[str], labels: list[str], out_path: Path, header: str
) -> Path:
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"{header}\n\n{report}")
    return out_path


def format_metrics_table(results: dict[str, dict]) -> str:
    """Tabela comparativa dos setups avaliados."""
    header = f"{'setup':<12} {'F1 macro':>10} {'Precision':>10} {'Recall':>10} {'Accuracy':>10}"
    lines = [header, "-" * len(header)]
    for setup, m in results.items():
        lines.append(
            f"{setup:<12} {m['f1_macro']:>10.4f} {m['precision_macro']:>10.4f} "
            f"{m['recall_macro']:>10.4f} {m['accuracy']:>10.4f}"
        )
    return "\n".join(lines)

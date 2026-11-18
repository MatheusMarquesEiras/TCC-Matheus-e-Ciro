"""Consolidação dos resultados de várias execuções em um comparativo único.

Percorre ``results/<modelo>/<dataset>/`` juntando o que cada etapa deixou —
métricas do script 03, escore qualitativo do script 04 e custo de treino do
script 02 — e produz a tabela e o gráfico que vão para o texto do TCC.

Execuções incompletas não são um erro: um modelo que só foi treinado e avaliado
quantitativamente aparece na tabela com o campo qualitativo vazio.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from .model_registry import MODELS  # noqa: E402
from .paths import resolve_path  # noqa: E402


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"  Aviso: {path} não é JSON válido — ignorado.")
        return None


def collect_runs(
    results_root: str | Path = "results",
    models_root: str | Path = "models",
    dataset_key: str | None = None,
) -> list[dict]:
    """Coleta todas as execuções encontradas, opcionalmente de um só dataset."""
    results_path = resolve_path(results_root)
    models_path = resolve_path(models_root)
    runs: list[dict] = []

    for model_key in sorted(MODELS):
        model_dir = results_path / model_key
        if not model_dir.is_dir():
            continue
        for dataset_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            if dataset_key and dataset_dir.name != dataset_key:
                continue

            evaluation = _read_json(dataset_dir / "evaluation" / "results.json")
            if not evaluation:
                continue

            qualitative = _read_json(dataset_dir / "qualitative" / "qualitative_results.json")
            training = _read_json(
                models_path / model_key / dataset_dir.name / "training_summary.json"
            )

            setups = evaluation.get("setups", {})
            finetuned = setups.get("finetuned", {})
            zeroshot = setups.get("zeroshot", {})

            runs.append(
                {
                    "model_key": model_key,
                    "model_label": evaluation.get("model_label", MODELS[model_key].label),
                    "params": MODELS[model_key].params,
                    "variant": MODELS[model_key].variant,
                    "dataset_key": dataset_dir.name,
                    "n_test": evaluation.get("n_test"),
                    "accuracy": finetuned.get("accuracy"),
                    "f1_macro": finetuned.get("f1_macro"),
                    "precision_macro": finetuned.get("precision_macro"),
                    "recall_macro": finetuned.get("recall_macro"),
                    "f1_per_class": finetuned.get("f1_per_class", {}),
                    "zeroshot_f1_macro": zeroshot.get("f1_macro"),
                    "zeroshot_accuracy": zeroshot.get("accuracy"),
                    "qualitative_score": (qualitative or {}).get("overall_score"),
                    "qualitative_by_theme": (qualitative or {}).get("by_theme", {}),
                    "train_hours": (training or {}).get("train_runtime_hours"),
                    "peak_vram_gb": (training or {}).get("peak_vram_gb"),
                    "trainable_params": (training or {}).get("trainable_params"),
                    "trainable_pct": (training or {}).get("trainable_pct"),
                    "final_train_loss": (training or {}).get("final_train_loss"),
                }
            )
    return runs


def _cell(value, fmt: str = "{:.4f}") -> str:
    return "—" if value is None else fmt.format(value)


def build_markdown(runs: list[dict]) -> str:
    """Tabelas em Markdown, prontas para colar no texto do TCC."""
    if not runs:
        return "Nenhuma execução encontrada.\n"

    lines = ["# Comparativo entre modelos\n"]

    lines.append("## Desempenho (modelo ajustado)\n")
    lines.append(
        "| Modelo | Params | Dataset | Accuracy | F1 macro | Precision | Recall | "
        "F1 zero-shot | Δ F1 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for run in runs:
        delta = (
            run["f1_macro"] - run["zeroshot_f1_macro"]
            if run["f1_macro"] is not None and run["zeroshot_f1_macro"] is not None
            else None
        )
        lines.append(
            f"| {run['model_label']} | {run['params']} | {run['dataset_key']} | "
            f"{_cell(run['accuracy'])} | **{_cell(run['f1_macro'])}** | "
            f"{_cell(run['precision_macro'])} | {_cell(run['recall_macro'])} | "
            f"{_cell(run['zeroshot_f1_macro'])} | {_cell(delta, '{:+.4f}')} |"
        )

    lines.append("\n## Custo de treino e avaliação qualitativa\n")
    lines.append(
        "| Modelo | Dataset | Params treináveis | % | Tempo (h) | Pico VRAM (GB) | "
        "Loss final | Score qualitativo |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for run in runs:
        trainable = (
            f"{run['trainable_params']:,}".replace(",", ".")
            if run["trainable_params"]
            else "—"
        )
        lines.append(
            f"| {run['model_label']} | {run['dataset_key']} | {trainable} | "
            f"{_cell(run['trainable_pct'], '{:.2f}%')} | "
            f"{_cell(run['train_hours'], '{:.2f}')} | "
            f"{_cell(run['peak_vram_gb'], '{:.2f}')} | "
            f"{_cell(run['final_train_loss'])} | {_cell(run['qualitative_score'])} |"
        )

    all_classes = sorted({c for run in runs for c in run["f1_per_class"]})
    if all_classes:
        lines.append("\n## F1 por classe (modelo ajustado)\n")
        lines.append("| Classe | " + " | ".join(r["model_label"] for r in runs) + " |")
        lines.append("|---" * (len(runs) + 1) + "|")
        for label in all_classes:
            cells = " | ".join(_cell(run["f1_per_class"].get(label)) for run in runs)
            lines.append(f"| {label} | {cells} |")

    return "\n".join(lines) + "\n"


def save_comparison_chart(runs: list[dict], out_path: Path) -> Path | None:
    """Gráfico de barras F1 macro zero-shot × fine-tuned por modelo."""
    plottable = [r for r in runs if r["f1_macro"] is not None]
    if not plottable:
        return None

    names = [f"{r['model_label']}\n({r['dataset_key']})" for r in plottable]
    finetuned = [r["f1_macro"] for r in plottable]
    zeroshot = [r["zeroshot_f1_macro"] or 0.0 for r in plottable]

    positions = range(len(plottable))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7, 2.2 * len(plottable)), 5))
    ax.bar([p - width / 2 for p in positions], zeroshot, width, label="Zero-shot")
    ax.bar([p + width / 2 for p in positions], finetuned, width, label="Fine-tuned")

    for p, value in zip(positions, finetuned):
        ax.text(p + width / 2, value + 0.012, f"{value:.3f}", ha="center", fontsize=9)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("F1 macro")
    ax.set_ylim(0, 1.0)
    ax.set_title("Triagem por especialidade — F1 macro por modelo")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

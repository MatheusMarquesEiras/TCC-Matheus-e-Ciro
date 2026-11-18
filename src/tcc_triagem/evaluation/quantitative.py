"""Avaliação intrínseca: zero-shot, few-shot e fine-tuned.

Compara o modelo ajustado com o mesmo checkpoint sem ajuste, no mesmo conjunto
de teste e com o mesmo prompt — é essa diferença que sustenta a conclusão do
trabalho sobre o ganho do fine-tuning.
"""

import json
import random
from pathlib import Path

from .. import pausa as pause
from ..data_prep import label_of, labels_of, load_metadata, load_split, question_of
from ..inference import Classifier
from ..model_registry import ModelSpec
from .metrics import (
    compute_metrics,
    format_metrics_table,
    save_classification_report,
    save_confusion_matrix,
)

#: Quantos exemplos few-shot cada setup coloca no prompt.
SETUP_SHOTS = {"zeroshot": 0, "threeshot": 3, "fiveshot": 5, "finetuned": 0}
ALL_SETUPS = tuple(SETUP_SHOTS)


def _few_shot_pool(data_dir: Path) -> list[dict]:
    return [
        {"question": question_of(record), "label": label_of(record)}
        for record in load_split(data_dir, "train")
    ]


def _load_partial(path: Path, n_test: int) -> list[str]:
    """Lê as predições de uma execução interrompida (uma por linha, em ordem).

    Uma linha cortada no meio (corte brusco) é descartada, e o arquivo é regravado só
    com o que está íntegro, para as próximas linhas serem anexadas no ponto certo.
    """
    if not path.exists():
        return []
    preds: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            if row.get("i") != len(preds) or len(preds) >= n_test:
                break
            preds.append(row["pred"])
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for i, pred in enumerate(preds):
            f.write(json.dumps({"i": i, "pred": pred}, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return preds


def run_setup(
    setup: str,
    spec: ModelSpec,
    test_data: list[dict],
    labels: list[str],
    model_dir: str | None,
    output_dir: Path,
    few_shot_pool: list[dict],
    system_template: str | None,
    seed: int,
) -> tuple[dict, list[str]]:
    """Avalia um setup e devolve ``(métricas, predições)``."""
    n_shot = SETUP_SHOTS[setup]
    # Fine-tuned usa os adaptadores; os demais usam o checkpoint original.
    classifier = Classifier.load(
        spec,
        labels,
        model_dir=model_dir if setup == "finetuned" else None,
        system_template=system_template,
    )
    rng = random.Random(seed)

    # Cada predição é anexada ao disco assim que sai: o teste completo leva horas e
    # pode ser pausado (ver pausa.py) ou cortado a qualquer momento sem perder nada.
    partial_file = output_dir / f"partial_{setup}.jsonl"
    y_true = [label_of(item) for item in test_data]
    y_pred = _load_partial(partial_file, len(y_true))
    if y_pred:
        print(f"  Retomando de {len(y_pred)}/{len(test_data)} predições já salvas.")

    with open(partial_file, "a", encoding="utf-8") as partial:
        for i, item in enumerate(test_data):
            few_shot = None
            if n_shot:
                # Exemplos da mesma classe alvo são excluídos para não entregar a resposta.
                # O sorteio roda também nos itens já feitos, para manter a mesma sequência.
                candidates = [ex for ex in few_shot_pool if ex["label"] != y_true[i]]
                few_shot = rng.sample(candidates, min(n_shot, len(candidates)))

            if i < len(y_pred):
                continue

            if pause.requested():
                # Para de mandar exemplos ao modelo: registra onde parou e libera a placa.
                partial.flush()
                pause.write_state(
                    output_dir, status="pausado", etapa=f"predições ({setup})",
                    feito=len(y_pred), total=len(test_data), motivo=pause.reason(),
                    como_retomar="rodar o mesmo comando de novo: ele continua daqui",
                )
                classifier.release()
                pause.consume()
                raise pause.PauseRequested(
                    f"pausado em {len(y_pred)}/{len(test_data)} ({setup})"
                )

            y_pred.append(classifier.predict(question_of(item), few_shot))
            partial.write(json.dumps({"i": i, "pred": y_pred[-1]}, ensure_ascii=False) + "\n")
            partial.flush()

            if (i + 1) % 100 == 0:
                acc = sum(t == p for t, p in zip(y_true, y_pred)) / (i + 1)
                print(f"  [{i + 1}/{len(test_data)}] acurácia parcial: {acc:.3f}")
                pause.write_state(
                    output_dir, status="rodando", etapa=f"predições ({setup})",
                    feito=i + 1, total=len(test_data), acuracia_parcial=round(acc, 4),
                )

    metrics = compute_metrics(y_true, y_pred, labels)
    save_confusion_matrix(
        y_true, y_pred, labels,
        output_dir / f"confusion_matrix_{setup}.png",
        title=f"Matriz de confusão — {spec.label} — {setup}",
    )
    save_classification_report(
        y_true, y_pred, labels,
        output_dir / f"report_{setup}.txt",
        header=f"Modelo: {spec.label} ({spec.hf_id})\nSetup: {setup}",
    )

    partial_file.unlink(missing_ok=True)
    classifier.release()
    return metrics, y_pred


def run_evaluation(
    spec: ModelSpec,
    data_dir: Path,
    model_dir: str | None,
    output_dir: Path,
    setups: list[str],
    max_test: int | None = None,
    seed: int = 42,
) -> dict:
    """Roda os setups pedidos e grava métricas, matrizes e predições."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(data_dir)
    labels = labels_of(metadata)
    system_template = None  # o prompt já está embutido nos splits em ChatML

    test_data = load_split(data_dir, "test")
    if max_test is not None:
        test_data = test_data[:max_test]

    few_shot_pool = _few_shot_pool(data_dir) if any(SETUP_SHOTS[s] for s in setups) else []

    print(f"Modelo:   {spec.label} ({spec.hf_id})")
    print(f"Dataset:  {metadata.get('dataset_label', data_dir.name)}")
    print(f"Teste:    {len(test_data)} exemplos"
          f"{' (limitado por --max_test)' if max_test else ''}")
    print(f"Setups:   {', '.join(setups)}\n")

    results: dict[str, dict] = {}
    finetuned_predictions: list[str] | None = None

    for setup in setups:
        print("=" * 55)
        print(f"Setup: {setup}")
        metrics, predictions = run_setup(
            setup, spec, test_data, labels, model_dir, output_dir,
            few_shot_pool, system_template, seed,
        )
        results[setup] = metrics
        if setup == "finetuned":
            finetuned_predictions = predictions

        print(f"  F1 macro:        {metrics['f1_macro']:.4f}")
        print(f"  Precision macro: {metrics['precision_macro']:.4f}")
        print(f"  Recall macro:    {metrics['recall_macro']:.4f}")
        print(f"  Accuracy:        {metrics['accuracy']:.4f}")

    print(f"\n{format_metrics_table(results)}")

    payload = {
        "model_key": spec.key,
        "model_label": spec.label,
        "model_hf_id": spec.hf_id,
        "dataset_key": metadata.get("dataset_key", data_dir.name),
        "dataset_label": metadata.get("dataset_label", data_dir.name),
        "n_test": len(test_data),
        "labels": labels,
        "setups": results,
    }
    results_file = output_dir / "results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nResultados salvos em {results_file}")

    if finetuned_predictions is not None:
        pred_file = output_dir / "finetuned_predictions.json"
        with open(pred_file, "w", encoding="utf-8") as f:
            json.dump(finetuned_predictions, f, ensure_ascii=False, indent=2)
        print(f"Predições fine-tuned salvas em {pred_file}")
        print("→ use este arquivo como --predictions_file no script 04")
    pause.write_state(
        output_dir, status="concluído", etapa="predições", feito=len(test_data),
        total=len(test_data),
    )

    return payload

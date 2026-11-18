"""Avaliação qualitativa por LLM-grader, no formato HealthBench.

Seleciona uma amostra estratificada do conjunto de teste, aplica a rubrica de
cinco critérios e agrega o escore por tema (rotineiro, saúde mental, urgente).

Sobre a amostragem de casos urgentes: a rodada do Experimento 02 pegou apenas
1 caso urgente em 100, o que não sustenta conclusão nenhuma sobre red flags.
Por isso ``min_urgent`` força uma cota mínima desses casos na amostra — os
escores por tema continuam separados, então a cota não contamina o escore dos
casos rotineiros.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from .. import pausa as pause
from ..data_prep import label_of, question_of
from .rubrics import (
    GRADER_PROMPT,
    GRADER_SYSTEM,
    THEMES,
    assign_theme,
    build_criteria,
    is_urgent,
)


#: Tentativas por exemplo quando a resposta do grader não traz JSON legível.
GRADE_ATTEMPTS = 3


def _parse_grading(raw: str) -> list[dict]:
    """Extrai o JSON de avaliações da resposta do grader.

    Modelos costumam embrulhar o JSON em texto ou cercas de código; recortar do
    primeiro ``{`` ao último ``}`` cobre os dois casos.
    """
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        return json.loads(raw[start:end]).get("avaliacoes", [])
    except (ValueError, json.JSONDecodeError):
        return []


def grade_example(grader, question: str, predicted: str, expected: str) -> dict:
    criteria = build_criteria(expected)
    prompt = GRADER_PROMPT.format(
        question=question,
        expected=expected,
        predicted=predicted,
        criteria_json=json.dumps(criteria, ensure_ascii=False, indent=2),
    )
    # Resposta ilegível (JSON quebrado ou vazio) é falha de formato do grader,
    # não do modelo avaliado: tenta de novo antes de contar o exemplo como 0.
    for _ in range(GRADE_ATTEMPTS):
        raw = grader.grade(GRADER_SYSTEM, prompt)
        evaluations = _parse_grading(raw)
        if evaluations:
            break

    obtained = sum(e.get("pontos", 0) for e in evaluations)
    max_points = sum(c["pontos_max"] for c in criteria if c["pontos_max"] > 0)
    score = max(0.0, min(1.0, obtained / max_points)) if max_points else 0.0

    return {
        "pontos_obtidos": obtained,
        "pontos_max_positivos": max_points,
        "score": round(score, 4),
        "grader_parse_ok": bool(evaluations),
        "avaliacoes": evaluations,
        **({} if evaluations else {"grader_raw": raw[-2000:]}),
    }


def select_sample(
    test_data: list[dict],
    n_samples: int,
    seed: int = 42,
    min_urgent: int = 0,
) -> list[int]:
    """Amostra estratificada por rótulo, com cota mínima de casos urgentes.

    A cota é atendida **trocando** exemplos não urgentes por urgentes, não
    acrescentando: assim a amostra mantém exatamente ``n_samples`` exemplos e a
    cota sobrevive ao embaralhamento final.
    """
    rng = random.Random(seed)
    n_samples = min(n_samples, len(test_data))

    by_label: dict[str, list[int]] = defaultdict(list)
    for i, item in enumerate(test_data):
        by_label[label_of(item)].append(i)

    per_label = max(1, n_samples // max(1, len(by_label)))
    selected: list[int] = []
    for indices in by_label.values():
        selected.extend(rng.sample(indices, min(per_label, len(indices))))

    remaining = list(set(range(len(test_data))) - set(selected))
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, n_samples - len(selected))])
    selected = selected[:n_samples]

    quota = min(min_urgent, n_samples)
    if quota:
        urgent = {i for i, item in enumerate(test_data) if is_urgent(question_of(item))}
        chosen = set(selected)
        missing = min(quota, len(urgent)) - len(chosen & urgent)
        if missing > 0:
            candidates = [i for i in urgent if i not in chosen]
            replaceable = [i for i in selected if i not in urgent]
            rng.shuffle(candidates)
            rng.shuffle(replaceable)
            positions = {value: pos for pos, value in enumerate(selected)}
            for new, old in zip(candidates[:missing], replaceable):
                selected[positions[old]] = new

    rng.shuffle(selected)
    return selected


def _save_partial_grading(path: Path, results: list[dict]) -> None:
    """Grava as notas dadas até agora, sem deixar o arquivo pela metade."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    tmp.replace(path)


def _load_partial_grading(path: Path, valid_indices: set[int]) -> list[dict]:
    """Lê as notas de uma execução interrompida; descarta o que não bate com a amostra."""
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    return [r for r in saved if r.get("idx") in valid_indices]


def run_qualitative_eval(
    grader,
    test_data: list[dict],
    predictions: list[str],
    output_dir: Path,
    n_samples: int = 100,
    seed: int = 42,
    min_urgent: int = 0,
    run_info: dict | None = None,
) -> dict:
    """Roda a avaliação completa e grava os resultados."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(predictions) > len(test_data):
        raise ValueError(
            f"Predições ({len(predictions)}) excedem os exemplos de teste ({len(test_data)})."
        )
    if len(predictions) < len(test_data):
        print(
            f"Aviso: predições parciais ({len(predictions)}/{len(test_data)}) — "
            f"usando os primeiros {len(predictions)} exemplos."
        )
        test_data = test_data[: len(predictions)]

    indices = select_sample(test_data, n_samples, seed=seed, min_urgent=min_urgent)
    print(f"LLM-grader: {grader.label}")
    print(f"Avaliando {len(indices)} exemplos...\n")

    # Notas já dadas ficam em disco: se a execução for interrompida, rodar de novo
    # só consulta o juiz para os exemplos que faltam.
    partial_file = output_dir / "partial_grading.json"
    results = _load_partial_grading(partial_file, set(indices))
    done = {r["idx"] for r in results}
    if results:
        print(f"Retomando: {len(results)}/{len(indices)} notas já salvas.\n")
    scores_by_theme: dict[str, list[float]] = defaultdict(list)
    for r in results:
        scores_by_theme[r["theme"]].append(r["score"])

    for position, idx in enumerate(indices, 1):
        if idx in done:
            continue
        if pause.requested():
            # Para de consultar o juiz: as notas já dadas ficam em disco.
            pause.write_state(
                output_dir, status="pausado", etapa="juiz GLM", feito=len(results),
                total=len(indices), motivo=pause.reason(),
                como_retomar="rodar o mesmo comando de novo: ele continua daqui",
            )
            pause.consume()
            raise pause.PauseRequested(f"pausado em {len(results)}/{len(indices)} notas")
        item = test_data[idx]
        question = question_of(item)
        expected = label_of(item)
        predicted = predictions[idx]
        correct = predicted.strip().lower() == expected.strip().lower()
        theme = assign_theme(question, expected)

        print(
            f"  [{position:3d}/{len(indices)}] {expected} → {predicted} "
            f"{'✓' if correct else '✗'}"
        )

        grading = grade_example(grader, question, predicted, expected)
        results.append(
            {
                "idx": idx,
                "theme": theme,
                "theme_label": THEMES.get(theme, theme),
                "question": question,
                "expected": expected,
                "predicted": predicted,
                "correct": correct,
                **grading,
            }
        )
        scores_by_theme[theme].append(grading["score"])
        _save_partial_grading(partial_file, results)
        pause.write_state(
            output_dir, status="rodando", etapa="juiz GLM", feito=len(results),
            total=len(indices),
        )

    overall_score = sum(r["score"] for r in results) / len(results)
    overall_accuracy = sum(r["correct"] for r in results) / len(results)
    parse_failures = sum(1 for r in results if not r["grader_parse_ok"])

    by_theme = {
        theme: {
            "label": THEMES.get(theme, theme),
            "score": round(sum(scores) / len(scores), 4),
            "n": len(scores),
        }
        for theme, scores in scores_by_theme.items()
    }

    print(f"\n{'=' * 52}")
    print(f"Grader:                  {grader.label}")
    print(f"Score qualitativo geral: {overall_score:.4f}")
    print(f"Acurácia na amostra:     {overall_accuracy:.4f} ({len(results)} exemplos)")
    if parse_failures:
        print(f"Respostas do grader ilegíveis: {parse_failures} (contam como score 0)")
    print("\nScore por tema:")
    for info in by_theme.values():
        print(f"  {info['label']}: {info['score']:.4f} ({info['n']} exemplos)")

    summary = {
        **(run_info or {}),
        "grader": grader.label,
        "overall_score": round(overall_score, 4),
        "overall_accuracy": round(overall_accuracy, 4),
        "n_samples": len(results),
        "grader_parse_failures": parse_failures,
        "by_theme": by_theme,
        "examples": results,
    }

    out_file = output_dir / "qualitative_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    partial_file.unlink(missing_ok=True)
    pause.write_state(
        output_dir, status="concluído", etapa="juiz GLM", feito=len(results),
        total=len(indices),
    )
    print(f"\nResultados salvos em {out_file}")

    errors = [r for r in results if not r["correct"]]
    if errors:
        error_file = output_dir / "error_analysis.json"
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"Análise de erros ({len(errors)} casos) salva em {error_file}")

    return summary

"""
Avaliação qualitativa inspirada no HealthBench.

Seleciona 100 exemplos estratificados do test set, aplica rubricas por 5 eixos
e usa um LLM-grader (Claude via API Anthropic OU modelo local via Ollama).

Exemplos de uso:

  # Com Anthropic (Claude):
  python scripts/04_qualitative_eval.py \\
      --grader anthropic \\
      --api_key $ANTHROPIC_API_KEY \\
      --data_dir data/top10 \\
      --predictions_file results/evaluation/finetuned_predictions.json

  # Com Ollama (modelo local):
  python scripts/04_qualitative_eval.py \\
      --grader ollama \\
      --ollama_model llama3.1:8b \\
      --data_dir data/top10 \\
      --predictions_file results/evaluation/finetuned_predictions.json

  # Ollama com URL customizada:
  python scripts/04_qualitative_eval.py \\
      --grader ollama \\
      --ollama_model qwen2.5:7b \\
      --ollama_url http://localhost:11434 \\
      --data_dir data/top10 \\
      --predictions_file results/evaluation/finetuned_predictions.json
"""

import json
import argparse
import random
import os
import urllib.request
import urllib.error
from pathlib import Path
from collections import defaultdict


RANDOM_SEED = 42

THEMES = {
    "urgente": "Casos urgentes / red flags",
    "comum": "Casos comuns / rotineiros",
    "ambiguo": "Casos ambíguos / múltiplas especialidades possíveis",
    "saude_mental": "Saúde mental",
}

URGENT_KEYWORDS = [
    "não consigo respirar", "falta de ar", "dor no peito", "desmaio", "perda de consciência",
    "convulsão", "sangramento intenso", "vômito com sangue", "febre muito alta",
    "paralisia", "dormência no rosto", "AVC", "infarto",
]

MENTAL_SPECIALTIES = {"psicólogo", "psiquiatra", "psicanalista"}

MENTAL_KEYWORDS = [
    "ansiedade", "depressão", "pânico", "fobia", "transtorno", "suicídio",
    "automutilação", "humor", "psicológico", "emocional", "mental",
]

GRADER_SYSTEM = (
    "Você é um avaliador especializado em triagem médica. "
    "Avalie classificações feitas por um sistema de IA com base em critérios clínicos precisos. "
    "Seja criterioso, consistente e imparcial. "
    "Responda sempre com JSON válido, sem texto adicional."
)

GRADER_PROMPT = """\
Avalie a seguinte classificação de triagem médica:

**Queixa do paciente:**
{question}

**Especialidade correta:** {expected}
**Especialidade predita pelo sistema:** {predicted}

Avalie cada critério abaixo. Para cada um, retorne o número de pontos obtidos \
(entre o mínimo e o máximo especificados) e uma justificativa curta.

Critérios:
{criteria_json}

Retorne um JSON com o formato:
{{
  "avaliacoes": [
    {{"criterio": "nome", "pontos": N, "justificativa": "texto curto"}},
    ...
  ]
}}
"""


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------

class AnthropicGrader:
    """LLM-grader usando a API Anthropic (Claude)."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def grade(self, system: str, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class OllamaGrader:
    """LLM-grader usando Ollama (modelo local via API REST)."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._check_connection()

    def _check_connection(self):
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5):
                pass
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(
                f"Não foi possível conectar ao Ollama em {self.base_url}. "
                f"Certifique-se de que o Ollama está rodando (`ollama serve`). Erro: {e}"
            )

    def grade(self, system: str, prompt: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0},
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["message"]["content"]


# ---------------------------------------------------------------------------
# Lógica de avaliação
# ---------------------------------------------------------------------------

def build_criteria(predicted: str, expected: str) -> list[dict]:
    return [
        {
            "criterio": "Acurácia — especialidade correta",
            "descricao": f"O sistema classificou corretamente como '{expected}'",
            "pontos_max": 10,
            "pontos_min": 0,
        },
        {
            "criterio": "Relevância clínica — especialidade defensável",
            "descricao": (
                "Mesmo que não seja exatamente a especialidade alvo, "
                "a classificação é clinicamente razoável para os sintomas descritos"
            ),
            "pontos_max": 7,
            "pontos_min": 0,
        },
        {
            "criterio": "Awareness de contexto — identificação de gravidade",
            "descricao": (
                "O sistema identificou corretamente se o caso é urgente, "
                "rotineiro ou de saúde mental"
            ),
            "pontos_max": 5,
            "pontos_min": 0,
        },
        {
            "criterio": "Instruction following — formato correto",
            "descricao": "A resposta contém apenas o nome da especialidade, sem texto adicional",
            "pontos_max": 3,
            "pontos_min": 0,
        },
        {
            "criterio": "Penalidade — subtriagem de emergência",
            "descricao": (
                "Aplique -8 se o caso for urgente (red flag) e o sistema "
                "classificou como especialidade rotineira; 0 caso contrário"
            ),
            "pontos_max": 0,
            "pontos_min": -8,
        },
    ]


def assign_theme(question: str, specialty: str) -> str:
    q_lower = question.lower()
    s_lower = specialty.lower()
    if s_lower in MENTAL_SPECIALTIES or any(kw in q_lower for kw in MENTAL_KEYWORDS):
        return "saude_mental"
    if any(kw in q_lower for kw in URGENT_KEYWORDS):
        return "urgente"
    return "comum"


def grade_example(grader, question: str, predicted: str, expected: str) -> dict:
    criteria = build_criteria(predicted, expected)
    prompt = GRADER_PROMPT.format(
        question=question,
        expected=expected,
        predicted=predicted,
        criteria_json=json.dumps(criteria, ensure_ascii=False, indent=2),
    )

    raw = grader.grade(GRADER_SYSTEM, prompt)

    try:
        # Extrai o bloco JSON mesmo que o modelo adicione texto extra
        start = raw.index("{")
        end = raw.rindex("}") + 1
        result = json.loads(raw[start:end])
        avaliacoes = result.get("avaliacoes", [])
    except (ValueError, json.JSONDecodeError):
        avaliacoes = []

    pontos_obtidos = sum(a.get("pontos", 0) for a in avaliacoes)
    pontos_max = sum(c["pontos_max"] for c in criteria if c["pontos_max"] > 0)
    score = max(0.0, min(1.0, pontos_obtidos / pontos_max)) if pontos_max > 0 else 0.0

    return {
        "pontos_obtidos": pontos_obtidos,
        "pontos_max_positivos": pontos_max,
        "score": round(score, 4),
        "avaliacoes": avaliacoes,
    }


def select_stratified_indices(test_data: list, n_samples: int) -> list[int]:
    by_specialty: dict[str, list[int]] = defaultdict(list)
    for i, item in enumerate(test_data):
        spec = item["messages"][2]["content"]
        by_specialty[spec].append(i)

    specialties = list(by_specialty.keys())
    per_spec = max(1, n_samples // len(specialties))

    selected = []
    for spec in specialties:
        indices = by_specialty[spec]
        selected.extend(random.sample(indices, min(per_spec, len(indices))))

    remaining = list(set(range(len(test_data))) - set(selected))
    random.shuffle(remaining)
    selected.extend(remaining[: max(0, n_samples - len(selected))])
    return selected[:n_samples]


def main(args):
    random.seed(RANDOM_SEED)
    data_path = Path(args.data_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(data_path / "test.json", encoding="utf-8") as f:
        test_data = json.load(f)
    with open(args.predictions_file, encoding="utf-8") as f:
        predictions = json.load(f)

    if len(predictions) != len(test_data):
        raise ValueError(
            f"Número de predições ({len(predictions)}) != exemplos de teste ({len(test_data)})"
        )

    # Instanciar o grader correto
    if args.grader == "anthropic":
        if not args.api_key:
            raise ValueError(
                "Forneça a chave via --api_key ou variável de ambiente ANTHROPIC_API_KEY"
            )
        grader = AnthropicGrader(api_key=args.api_key, model=args.anthropic_model)
        grader_label = f"Claude ({args.anthropic_model})"
    else:
        grader = OllamaGrader(model=args.ollama_model, base_url=args.ollama_url)
        grader_label = f"Ollama ({args.ollama_model})"

    selected_indices = select_stratified_indices(test_data, args.n_samples)
    print(f"LLM-grader: {grader_label}")
    print(f"Avaliando {len(selected_indices)} exemplos...\n")

    results = []
    themes_scores: dict[str, list[float]] = defaultdict(list)

    for i, idx in enumerate(selected_indices, 1):
        item = test_data[idx]
        question = item["messages"][1]["content"]
        expected = item["messages"][2]["content"]
        predicted = predictions[idx]
        theme = assign_theme(question, expected)
        correct = predicted.strip().lower() == expected.strip().lower()

        print(f"  [{i:3d}/{len(selected_indices)}] {expected} → {predicted} {'✓' if correct else '✗'}")

        grading = grade_example(grader, question, predicted, expected)

        results.append({
            "idx": idx,
            "theme": theme,
            "theme_label": THEMES.get(theme, theme),
            "question": question,
            "expected": expected,
            "predicted": predicted,
            "correct": correct,
            **grading,
        })
        themes_scores[theme].append(grading["score"])

    overall_score = sum(r["score"] for r in results) / len(results)
    overall_accuracy = sum(r["correct"] for r in results) / len(results)

    print(f"\n{'='*50}")
    print(f"Grader:                  {grader_label}")
    print(f"Score qualitativo geral: {overall_score:.4f}")
    print(f"Acurácia nos {len(results)} exemplos: {overall_accuracy:.4f}")
    print(f"\nScore por tema:")

    themes_summary = {}
    for theme, scores in themes_scores.items():
        avg = sum(scores) / len(scores)
        themes_summary[theme] = {
            "label": THEMES.get(theme, theme),
            "score": round(avg, 4),
            "n": len(scores),
        }
        print(f"  {THEMES.get(theme, theme)}: {avg:.4f} ({len(scores)} exemplos)")

    summary = {
        "grader": grader_label,
        "overall_score": round(overall_score, 4),
        "overall_accuracy": round(overall_accuracy, 4),
        "n_samples": len(results),
        "by_theme": themes_summary,
        "examples": results,
    }

    out_file = output_path / "qualitative_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nResultados salvos em {out_file}")

    errors = [r for r in results if not r["correct"]]
    if errors:
        error_file = output_path / "error_analysis.json"
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"Análise de erros ({len(errors)} casos) salva em {error_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação qualitativa HealthBench-style.")

    # --- Dados e saída ---
    parser.add_argument("--data_dir", type=str, default="data/top10")
    parser.add_argument(
        "--predictions_file", type=str,
        default="results/evaluation/finetuned_predictions.json",
    )
    parser.add_argument("--output_dir", type=str, default="results/qualitative")
    parser.add_argument("--n_samples", type=int, default=100)

    # --- Escolha do grader ---
    parser.add_argument(
        "--grader", type=str, default="anthropic", choices=["anthropic", "ollama"],
        help="Backend do LLM-grader: 'anthropic' (padrão) ou 'ollama' (local)",
    )

    # --- Flags Anthropic ---
    parser.add_argument(
        "--api_key", type=str, default=os.getenv("ANTHROPIC_API_KEY", ""),
        help="[anthropic] Chave da API. Padrão: variável ANTHROPIC_API_KEY",
    )
    parser.add_argument(
        "--anthropic_model", type=str, default="claude-sonnet-4-6",
        help="[anthropic] Modelo a usar. Padrão: claude-sonnet-4-6",
    )

    # --- Flags Ollama ---
    parser.add_argument(
        "--ollama_model", type=str, default="llama3.1:8b",
        help="[ollama] Nome do modelo no Ollama. Ex: llama3.1:8b, qwen2.5:7b, mistral",
    )
    parser.add_argument(
        "--ollama_url", type=str, default="http://localhost:11434",
        help="[ollama] URL base do servidor Ollama. Padrão: http://localhost:11434",
    )

    args = parser.parse_args()
    main(args)

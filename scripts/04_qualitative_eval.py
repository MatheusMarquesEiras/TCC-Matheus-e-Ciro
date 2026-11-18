"""Avaliação qualitativa por LLM-grader, no formato HealthBench.

Aplica a rubrica de cinco critérios sobre uma amostra estratificada do conjunto
de teste e agrega o escore por tema (rotineiro, saúde mental, urgente).

Backends de grader:
    ollama         modelo local (`ollama serve`), sem custo
    ollama-cloud   modelos grandes hospedados pela Ollama, via chave de API
    anthropic      Claude via API

Exemplos de uso:

  # Ollama local
  ollama pull llama3.1:8b
  uv run python scripts/04_qualitative_eval.py --model qwen35-0.8b --grader ollama

  # Ollama Cloud (chave em https://ollama.com/settings/keys)
  $env:OLLAMA_API_KEY = "..."
  uv run python scripts/04_qualitative_eval.py --model qwen35-0.8b `
      --grader ollama-cloud --grader_model gpt-oss:120b --min_urgent 15

  # Anthropic
  uv run python scripts/04_qualitative_eval.py --model qwen35-0.8b `
      --grader anthropic --api_key sk-ant-...
"""

import _bootstrap  # noqa: F401  (efeito colateral: ajusta o sys.path)

import argparse
import sys
import json
import os

from tcc_triagem.data_prep import load_split
from tcc_triagem.dataset_registry import DATASETS, DEFAULT_DATASET, get_dataset
from tcc_triagem.model_registry import MODELS, get_model
from tcc_triagem.paths import resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação qualitativa HealthBench-style.")
    parser.add_argument(
        "--model", type=str, default="qwen35-0.8b", choices=sorted(MODELS),
        help="Modelo avaliado — define os caminhos padrão",
    )
    parser.add_argument(
        "--dataset", type=str, default=DEFAULT_DATASET, choices=sorted(DATASETS),
        help=f"Chave do dataset. Padrão: {DEFAULT_DATASET}",
    )
    parser.add_argument("--data_dir", type=str, default=None, help="Caminho explícito dos splits")
    parser.add_argument(
        "--predictions_file", type=str, default=None,
        help="Predições do script 03. Padrão: results/<modelo>/<dataset>/evaluation/"
             "finetuned_predictions.json",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Saída. Padrão: results/<modelo>/<dataset>/qualitative",
    )
    parser.add_argument("--n_samples", type=int, default=100, help="Tamanho da amostra")
    parser.add_argument(
        "--min_urgent", type=int, default=0,
        help="Cota mínima de casos com red flag na amostra. A rodada do "
             "Experimento 02 pegou só 1 caso urgente, o que não sustenta conclusão",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semente da amostragem")

    parser.add_argument(
        "--grader", type=str, default="ollama",
        choices=["ollama", "ollama-cloud", "anthropic"],
        help="Backend do LLM-grader. Padrão: ollama (local)",
    )
    parser.add_argument(
        "--grader_model", type=str, default=None,
        help="Modelo do grader. Padrão: llama3.1:8b (local), gpt-oss:120b (cloud), "
             "claude-sonnet-4-6 (anthropic)",
    )
    parser.add_argument(
        "--ollama_url", type=str, default=None,
        help="Host do Ollama. Padrão: http://localhost:11434 ou https://ollama.com",
    )
    parser.add_argument(
        "--api_key", type=str, default=None,
        help="Chave de API do grader. Padrão: OLLAMA_API_KEY ou ANTHROPIC_API_KEY",
    )
    parser.add_argument(
        "--timeout", type=int, default=180, help="Timeout por requisição, em segundos",
    )
    args = parser.parse_args()

    spec = get_model(args.model)
    data_dir = resolve_path(args.data_dir or get_dataset(args.dataset).resolved_output_dir)
    predictions_path = resolve_path(
        args.predictions_file
        or f"results/{spec.key}/{args.dataset}/evaluation/finetuned_predictions.json"
    )
    output_dir = resolve_path(
        args.output_dir or f"results/{spec.key}/{args.dataset}/qualitative"
    )

    if not predictions_path.exists():
        parser.error(
            f"Predições não encontradas em {predictions_path}. "
            f"Rode antes: uv run python scripts/03_evaluate.py "
            f"--model {spec.key} --dataset {args.dataset}"
        )

    from tcc_triagem.evaluation.graders import build_grader
    from tcc_triagem.evaluation.qualitative import run_qualitative_eval

    api_key = args.api_key or (
        os.getenv("ANTHROPIC_API_KEY") if args.grader == "anthropic"
        else os.getenv("OLLAMA_API_KEY")
    )
    grader = build_grader(
        backend=args.grader,
        model=args.grader_model,
        host=args.ollama_url,
        api_key=api_key,
        timeout=args.timeout,
    )

    with open(predictions_path, encoding="utf-8") as f:
        predictions = json.load(f)

    run_qualitative_eval(
        grader=grader,
        test_data=load_split(data_dir, "test"),
        predictions=predictions,
        output_dir=output_dir,
        n_samples=args.n_samples,
        seed=args.seed,
        min_urgent=args.min_urgent,
        run_info={
            "model_key": spec.key,
            "model_label": spec.label,
            "dataset_key": args.dataset,
            "predictions_file": str(predictions_path),
        },
    )


if __name__ == "__main__":
    from tcc_triagem import pausa

    pausa.install()
    try:
        main()
    except pausa.PauseRequested as exc:
        print(f"\n[pausa] {exc}. Progresso salvo; rode o mesmo comando para retomar.")
        sys.exit(pausa.PAUSE_EXIT_CODE)

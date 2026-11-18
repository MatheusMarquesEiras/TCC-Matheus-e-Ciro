"""Avaliação quantitativa de um modelo ajustado.

Compara zero-shot, 3-shot, 5-shot e fine-tuned no mesmo conjunto de teste.
Gera métricas (F1 macro, precision, recall, accuracy e F1 por classe), matrizes
de confusão, relatórios de classificação e as predições do modelo ajustado —
entrada do script 04.

Os caminhos padrão seguem a convenção ``<modelo>/<dataset>``, então basta
informar ``--model`` e ``--dataset`` para avaliar o que o script 02 treinou.

Exemplos de uso:
    uv run python scripts/03_evaluate.py --model qwen35-0.8b --dataset medpt-top10
    uv run python scripts/03_evaluate.py --model gemma3-4b --dataset medpt-top10 \
        --setups finetuned --max_test 200
"""

import _bootstrap  # noqa: F401  (efeito colateral: ajusta o sys.path)

import argparse
import sys

from tcc_triagem.dataset_registry import DATASETS, DEFAULT_DATASET, get_dataset
from tcc_triagem.model_registry import MODELS, get_model
from tcc_triagem.paths import resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação quantitativa multi-modelo.")
    parser.add_argument(
        "--model", type=str, default="qwen35-0.8b", choices=sorted(MODELS),
        help="Chave do modelo no registro",
    )
    parser.add_argument(
        "--dataset", type=str, default=DEFAULT_DATASET, choices=sorted(DATASETS),
        help=f"Chave do dataset. Padrão: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Caminho explícito dos splits, ignorando --dataset",
    )
    parser.add_argument(
        "--model_dir", type=str, default=None,
        help="Adaptadores LoRA. Padrão: models/<modelo>/<dataset>/final",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Saída. Padrão: results/<modelo>/<dataset>/evaluation",
    )
    parser.add_argument(
        "--setups", nargs="+", default=["zeroshot", "threeshot", "fiveshot", "finetuned"],
        choices=["zeroshot", "threeshot", "fiveshot", "finetuned"],
        help="Setups a avaliar. Padrão: todos os quatro",
    )
    parser.add_argument(
        "--max_test", type=int, default=None,
        help="Limita o número de exemplos de teste (ex.: --max_test 200)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semente da amostragem few-shot")
    args = parser.parse_args()

    spec = get_model(args.model)
    data_dir = resolve_path(args.data_dir or get_dataset(args.dataset).resolved_output_dir)
    model_dir = resolve_path(
        args.model_dir or f"models/{spec.key}/{args.dataset}/final"
    )
    output_dir = resolve_path(
        args.output_dir or f"results/{spec.key}/{args.dataset}/evaluation"
    )

    if "finetuned" in args.setups and not model_dir.exists():
        parser.error(
            f"Adaptadores não encontrados em {model_dir}. "
            f"Treine antes com: uv run python scripts/02_train.py "
            f"--model {spec.key} --dataset {args.dataset}"
        )

    from tcc_triagem.evaluation.quantitative import run_evaluation

    run_evaluation(
        spec=spec,
        data_dir=data_dir,
        model_dir=str(model_dir),
        output_dir=output_dir,
        setups=args.setups,
        max_test=args.max_test,
        seed=args.seed,
    )


if __name__ == "__main__":
    from tcc_triagem import pausa

    pausa.install()
    try:
        main()
    except pausa.PauseRequested as exc:
        print(f"\n[pausa] {exc}. Progresso salvo; rode o mesmo comando para retomar.")
        sys.exit(pausa.PAUSE_EXIT_CODE)

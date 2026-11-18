"""Prepara os splits de um dataset registrado para fine-tuning.

O dataset é escolhido pela chave do registro (``tcc_triagem/dataset_registry.py``),
que já carrega fonte, colunas, número de classes e diretório de saída. As flags
existem só para variar um experimento sem editar o registro.

Funciona a partir de qualquer diretório.

Exemplos de uso:
    uv run python scripts/01_prepare_data.py --list
    uv run python scripts/01_prepare_data.py --dataset medpt-top10
    uv run python scripts/01_prepare_data.py --dataset medpt-top10-merged
    uv run python scripts/01_prepare_data.py --dataset medpt-top10 --max_per_class 500 \
        --output_dir data/top10_500
"""

import _bootstrap  # noqa: F401  (efeito colateral: ajusta o sys.path)

import argparse

from tcc_triagem.data_prep import prepare_dataset
from tcc_triagem.dataset_registry import (
    DEFAULT_DATASET,
    DATASETS,
    describe_datasets,
    get_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepara os splits de um dataset registrado.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        choices=sorted(DATASETS),
        help=f"Chave do dataset no registro. Padrão: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Sobrescreve o diretório de saída do registro (relativo à raiz do projeto)",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=None,
        help="Sobrescreve o número de classes mais frequentes a incluir",
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=None,
        help="Sobrescreve o limite de exemplos por classe",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Lista os datasets registrados e sai",
    )
    args = parser.parse_args()

    if args.list:
        print(describe_datasets())
        return

    metadata = prepare_dataset(
        get_dataset(args.dataset),
        output_dir=args.output_dir,
        max_per_class=args.max_per_class,
        top_n=args.top_n,
    )

    print(
        f"\nPré-processamento concluído: {len(metadata['labels'])} classes, "
        f"{sum(metadata['split_sizes'].values())} exemplos."
    )


if __name__ == "__main__":
    main()

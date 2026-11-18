"""Linha de comando compartilhada pelos scripts de treino por modelo.

Cada script em ``scripts/train/`` fixa o modelo e delega tudo o mais para cá,
de modo que os quatro modelos do estudo rodam exatamente o mesmo esquema de
fine-tuning — a única diferença entre eles vive no ``model_registry``.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tcc_triagem.console import enable_utf8_output  # noqa: E402
from tcc_triagem.dataset_registry import DATASETS, DEFAULT_DATASET  # noqa: E402
from tcc_triagem.model_registry import get_model  # noqa: E402

enable_utf8_output()


def build_parser(model_key: str) -> argparse.ArgumentParser:
    spec = get_model(model_key)
    parser = argparse.ArgumentParser(
        description=f"Fine-tuning QLoRA do {spec.label} ({spec.hf_id}).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        choices=sorted(DATASETS),
        help=f"Chave do dataset preparado pelo script 01. Padrão: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Caminho explícito dos splits, ignorando --dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=f"Saída dos checkpoints. Padrão: models/{model_key}/<dataset>",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"Sobrescreve o número de épocas do registro ({spec.training.epochs})",
    )
    parser.add_argument("--run_name", type=str, default=None, help="Nome da run no logger")
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        choices=["none", "wandb", "tensorboard"],
        help="Backend de logging. Padrão: none",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Ignora checkpoints existentes e treina do zero",
    )
    return parser


def run(model_key: str, argv: list[str] | None = None) -> dict:
    """Interpreta os argumentos e dispara o treino do modelo informado."""
    args = build_parser(model_key).parse_args(argv)

    from tcc_triagem.training import run_training

    return run_training(
        model_key=model_key,
        dataset_key=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        run_name=args.run_name,
        report_to=args.report_to,
        resume=not args.no_resume,
    )

"""Fine-tuning QLoRA de qualquer modelo registrado.

Ponto de entrada genérico. Para os quatro modelos do estudo há também um script
dedicado em ``scripts/train/``, que fixa o modelo e chama esta mesma rotina —
use aqueles quando quiser um comando por modelo, e este para varreduras.

Funciona a partir de qualquer diretório.

Exemplos de uso:
    uv run python scripts/02_train.py --list
    uv run python scripts/02_train.py --model qwen35-0.8b --dataset medpt-top10
    uv run python scripts/02_train.py --model gemma3-4b --dataset medpt-top10-merged
"""

import _bootstrap  # noqa: F401  (efeito colateral: ajusta o sys.path)

import argparse

from tcc_triagem.dataset_registry import DATASETS, DEFAULT_DATASET
from tcc_triagem.model_registry import MODELS, describe_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning QLoRA multi-modelo.")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen35-0.8b",
        choices=sorted(MODELS),
        help="Chave do modelo no registro. Padrão: qwen35-0.8b",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        choices=sorted(DATASETS),
        help=f"Chave do dataset preparado pelo script 01. Padrão: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Caminho explícito dos splits, ignorando --dataset",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Saída dos checkpoints. Padrão: models/<modelo>/<dataset>",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Sobrescreve as épocas")
    parser.add_argument("--run_name", type=str, default=None, help="Nome da run no logger")
    parser.add_argument(
        "--report_to", type=str, default="none",
        choices=["none", "wandb", "tensorboard"], help="Backend de logging",
    )
    parser.add_argument(
        "--no_resume", action="store_true",
        help="Ignora checkpoints existentes e treina do zero",
    )
    # Sobrescritas pontuais dos hiperparâmetros do registro (ex.: GPU maior no Colab).
    # Mantenha batch_size × grad_accum = 32 para o batch efetivo ficar igual ao dos outros modelos.
    parser.add_argument("--batch_size", type=int, default=None, help="Batch por passo (micro-lote)")
    parser.add_argument("--grad_accum", type=int, default=None, help="Passos de acumulação do gradiente")
    parser.add_argument("--eval_batch_size", type=int, default=None, help="Batch da avaliação")
    parser.add_argument("--save_steps", type=int, default=None, help="Checkpoint a cada N passos")
    parser.add_argument("--save_total_limit", type=int, default=None, help="Checkpoints mantidos")
    parser.add_argument(
        "--max_example_tokens", type=int, default=None,
        help="Descarta do treino exemplos acima de N tokens (0 = usa todos)",
    )
    parser.add_argument("--list", action="store_true", help="Lista os modelos e sai")
    args = parser.parse_args()

    if args.list:
        print(describe_models())
        return

    from tcc_triagem.training import run_training

    run_training(
        model_key=args.model,
        dataset_key=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        run_name=args.run_name,
        report_to=args.report_to,
        resume=not args.no_resume,
        training_overrides={
            key: getattr(args, key)
            for key in (
                "batch_size", "grad_accum", "eval_batch_size", "save_steps",
                "save_total_limit", "max_example_tokens",
            )
            if getattr(args, key) is not None
        } or None,
    )


if __name__ == "__main__":
    main()

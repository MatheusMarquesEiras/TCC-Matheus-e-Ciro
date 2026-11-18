"""Fine-tuning QLoRA do Qwen3.5-0.8B.

Modelo do Experimento 02 (F1 macro 0,7911) — a linha de base do trabalho.
Roda sempre em non-thinking mode: no thinking mode o 0.8B entra em loop infinito.

Exemplos de uso:
    uv run python scripts/train/train_qwen35_0_8b.py --dataset medpt-top10
    uv run python scripts/train/train_qwen35_0_8b.py --dataset medpt-top10-merged
"""

from _cli import run

MODEL_KEY = "qwen35-0.8b"

if __name__ == "__main__":
    run(MODEL_KEY)

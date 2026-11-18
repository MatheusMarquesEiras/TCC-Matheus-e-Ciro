"""Fine-tuning QLoRA do Gemma 3 1B (checkpoint base ``-pt``).

O checkpoint base não traz chat template; o template ``gemma-3`` é aplicado
pelo pipeline antes da formatação dos splits.

Exemplos de uso:
    uv run python scripts/train/train_gemma3_1b.py --dataset medpt-top10
"""

from _cli import run

MODEL_KEY = "gemma3-1b"

if __name__ == "__main__":
    run(MODEL_KEY)

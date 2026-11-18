"""Fine-tuning QLoRA do Qwen3.5-2B.

Mesma família do 0.8B, com batch por dispositivo reduzido a 2 e acumulação de
16 passos para manter o batch efetivo em 32 dentro dos 12 GB da RTX 4070.

Exemplos de uso:
    uv run python scripts/train/train_qwen35_2b.py --dataset medpt-top10
"""

from _cli import run

MODEL_KEY = "qwen35-2b"

if __name__ == "__main__":
    run(MODEL_KEY)

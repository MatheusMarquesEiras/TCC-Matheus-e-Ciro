"""Fine-tuning QLoRA do Gemma 3 4B (checkpoint base ``-pt``).

Maior modelo do estudo. É multimodal: carrega com ``FastModel``, mantém as
camadas de visão congeladas e treina LoRA só nas camadas de linguagem. Usa
batch 1 com acumulação de 32 passos e learning rate menor (1e-4).

Exemplos de uso:
    uv run python scripts/train/train_gemma3_4b.py --dataset medpt-top10
"""

from _cli import run

MODEL_KEY = "gemma3-4b"

if __name__ == "__main__":
    run(MODEL_KEY)

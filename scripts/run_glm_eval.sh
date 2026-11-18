#!/usr/bin/env bash
# Avaliação quantitativa (script 03) + qualitativa com GLM 5.3 como juiz (script 04)
# para os quatro pilotos. Pula etapas já concluídas, então pode ser interrompido
# e rodado de novo.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
# Antivírus com inspeção de HTTPS (Kaspersky) assina os certificados com uma raiz
# própria: usa um pacote com as raízes confiáveis do Windows, se existir.
CA_BUNDLE="$HOME/.cache/ca-bundle-windows.pem"
if [ -f "$CA_BUNDLE" ]; then
  export SSL_CERT_FILE="$CA_BUNDLE" REQUESTS_CA_BUNDLE="$CA_BUNDLE"
fi

DATASET=medpt-top10-500
GRADER_MODEL=glm-5.3-flash:cloud

for MODEL in qwen35-0.8b gemma3-1b qwen35-2b gemma3-4b; do
  EVAL_DIR="results/$MODEL/$DATASET/evaluation"
  QUAL_DIR="results/$MODEL/$DATASET/qualitative_glm"

  if [ ! -f "$EVAL_DIR/finetuned_predictions.json" ]; then
    echo "=== [$MODEL] script 03: predições ==="
    uv run python scripts/03_evaluate.py --model "$MODEL" --dataset "$DATASET" \
      --setups zeroshot finetuned || { echo "FALHOU 03 $MODEL"; continue; }
  else
    echo "=== [$MODEL] predições já existem, pulando 03 ==="
  fi

  if [ ! -f "$QUAL_DIR/qualitative_results.json" ]; then
    echo "=== [$MODEL] script 04: juiz $GRADER_MODEL ==="
    uv run python scripts/04_qualitative_eval.py --model "$MODEL" --dataset "$DATASET" \
      --grader ollama --grader_model "$GRADER_MODEL" \
      --n_samples 100 --min_urgent 15 --timeout 300 \
      --output_dir "$QUAL_DIR" || echo "FALHOU 04 $MODEL"
  else
    echo "=== [$MODEL] avaliação qualitativa já existe, pulando 04 ==="
  fi
done
echo "=== FIM ==="

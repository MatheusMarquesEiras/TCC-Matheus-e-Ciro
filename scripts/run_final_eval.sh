#!/usr/bin/env bash
# Avaliação dos modelos FINAIS (dataset completo, classes fundidas):
# script 03 (predições no teste inteiro, 18.166 exemplos) + script 04 com o GLM 5.3
# como juiz. Pula etapas já concluídas.
#
# Pode ser pausada a qualquer momento sem perder nada: `bash scripts/pausar.sh`
# (ou Ctrl+C). Rodar este mesmo comando de novo retoma de onde parou.
#
# Uso:
#   bash scripts/run_final_eval.sh qwen35-2b
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
CA_BUNDLE="$HOME/.cache/ca-bundle-windows.pem"
if [ -f "$CA_BUNDLE" ]; then
  export SSL_CERT_FILE="$CA_BUNDLE" REQUESTS_CA_BUNDLE="$CA_BUNDLE"
fi

DATASET=medpt-top10-merged
GRADER_MODEL=glm-5.3-flash:cloud
PAUSED=75   # código de saída de "pausado com sucesso" (tcc_triagem/pausa.py)
MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(qwen35-2b)

for MODEL in "${MODELS[@]}"; do
  EVAL_DIR="results/$MODEL/$DATASET/evaluation"
  QUAL_DIR="results/$MODEL/$DATASET/qualitative_glm"

  if [ ! -f "$EVAL_DIR/finetuned_predictions.json" ]; then
    echo "=== [$MODEL] script 03: predições no teste completo — $(date '+%d/%m %H:%M') ==="
    uv run --no-sync python scripts/03_evaluate.py --model "$MODEL" --dataset "$DATASET" \
      --setups finetuned
    status=$?
    [ $status -eq $PAUSED ] && { echo "=== [$MODEL] PAUSADO na etapa 03 — $(date '+%d/%m %H:%M') ==="; exit $PAUSED; }
    [ $status -ne 0 ] && { echo "FALHOU 03 $MODEL (código $status)"; continue; }
  else
    echo "=== [$MODEL] predições já existem, pulando 03 ==="
  fi

  if [ ! -f "$QUAL_DIR/qualitative_results.json" ]; then
    echo "=== [$MODEL] script 04: juiz $GRADER_MODEL — $(date '+%d/%m %H:%M') ==="
    uv run --no-sync python scripts/04_qualitative_eval.py --model "$MODEL" --dataset "$DATASET" \
      --grader ollama --grader_model "$GRADER_MODEL" \
      --n_samples 100 --min_urgent 15 --timeout 300 \
      --output_dir "$QUAL_DIR"
    status=$?
    [ $status -eq $PAUSED ] && { echo "=== [$MODEL] PAUSADO na etapa 04 — $(date '+%d/%m %H:%M') ==="; exit $PAUSED; }
    [ $status -ne 0 ] && echo "FALHOU 04 $MODEL (código $status)"
  else
    echo "=== [$MODEL] avaliação qualitativa já existe, pulando 04 ==="
  fi
done
echo "=== AVALIAÇÃO FINAL CONCLUÍDA — $(date '+%d/%m %H:%M') ==="

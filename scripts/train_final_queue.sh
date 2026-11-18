#!/usr/bin/env bash
# Treino dos modelos finais no dataset completo com Psicólogo e Psicanalista
# fundidos (medpt-top10-merged, 9 classes, 145.321 exemplos de treino).
#
# Roda os modelos em sequência. Pode ser interrompido a qualquer momento
# (Ctrl+C, fechar o terminal, desligar o PC): rodar de novo retoma do último
# checkpoint do modelo em andamento e pula os que já têm `final/`.
#
# Uso:
#   bash scripts/train_final_queue.sh                    # todos, na ordem abaixo
#   bash scripts/train_final_queue.sh qwen35-2b          # só os modelos informados
#   TRAIN_ARGS="--save_steps 50" bash scripts/train_final_queue.sh gemma3-1b
#                                                        # sobrescritas repassadas ao 02_train.py
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
# Antivírus com inspeção de HTTPS (Kaspersky): usa as raízes confiáveis do Windows.
CA_BUNDLE="$HOME/.cache/ca-bundle-windows.pem"
if [ -f "$CA_BUNDLE" ]; then
  export SSL_CERT_FILE="$CA_BUNDLE" REQUESTS_CA_BUNDLE="$CA_BUNDLE"
fi

DATASET=medpt-top10-merged
# Do mais rápido ao mais lento: os resultados úteis chegam primeiro.
MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(qwen35-0.8b qwen35-2b gemma3-1b gemma3-4b)

mkdir -p logs
for MODEL in "${MODELS[@]}"; do
  OUT="models/$MODEL/$DATASET"
  LOG="logs/train_${MODEL}_final.log"
  if [ -d "$OUT/final" ]; then
    echo "=== [$MODEL] já concluído ($OUT/final), pulando ==="
    continue
  fi
  # No Windows o cache de kernels do Triton falha de forma intermitente no 1o passo com
  # "WinError 145: a pasta não está vazia" (pastas tmp.* órfãs, ou duas compilações do
  # mesmo módulo disputando o destino). O cache é recriado sozinho: apaga antes de cada
  # tentativa e tenta de novo, no máximo 3 vezes, só quando o erro é esse.
  # Gemma 3 4B: micro-lote 8 enche os 12 GB e o Windows passa a usar a RAM (treino ~5x mais lento de
  # repente). Teto de 90% da VRAM + coleta de blocos em cache evitam o transbordamento.
  if [ "$MODEL" = "gemma3-4b" ]; then
    export TCC_GPU_MEMORY_FRACTION="${TCC_GPU_MEMORY_FRACTION:-0.90}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-garbage_collection_threshold:0.8,max_split_size_mb:256}"
  fi
  clear_triton_cache() {
    if [ -n "${LOCALAPPDATA:-}" ]; then
      rm -rf "$(cygpath -u "$LOCALAPPDATA" 2>/dev/null || echo "$LOCALAPPDATA")/Temp/torchinductor_${USERNAME:-$USER}/triton"
    fi
  }
  for ATTEMPT in 1 2 3; do
    clear_triton_cache
    echo "=== [$MODEL] treino em $DATASET — $(date '+%d/%m %H:%M') — tentativa $ATTEMPT — log: $LOG ==="
    SIZE_BEFORE=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
    uv run python scripts/02_train.py --model "$MODEL" --dataset "$DATASET"       --output_dir "$OUT" ${TRAIN_ARGS:-} >> "$LOG" 2>&1
    status=$?
    [ $status -eq 0 ] && break
    if tail -c +"$((SIZE_BEFORE + 1))" "$LOG" | grep -aq "WinError 145"; then
      echo "=== [$MODEL] erro intermitente do cache do Triton (WinError 145), tentando de novo ==="
      continue
    fi
    break
  done
  if [ $status -ne 0 ]; then
    echo "=== [$MODEL] FALHOU (código $status) — veja $LOG. Fila interrompida. ==="
    exit $status
  fi
  echo "=== [$MODEL] concluído — $(date '+%d/%m %H:%M') ==="
done
echo "=== FILA CONCLUÍDA ==="

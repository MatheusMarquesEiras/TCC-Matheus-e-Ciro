#!/usr/bin/env bash
# Pausa limpa da avaliação em andamento (scripts 03 e 04).
#
# Cria o arquivo PAUSAR. A avaliação, ao ver o arquivo, para de mandar exemplos ao
# modelo, deixa o exemplo atual terminar, grava tudo em disco, registra onde parou
# em results/<modelo>/<dataset>/.../estado_avaliacao.json e encerra o processo,
# o que devolve a placa de vídeo. Este script espera isso acontecer e mostra o estado.
#
# Para retomar, rode de novo o mesmo comando da avaliação.
#
# Uso:
#   bash scripts/pausar.sh
set -u
cd "$(dirname "$0")/.."

running() {
  powershell -NoProfile -Command '(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "03_evaluate|04_qualitative_eval" -and $_.CommandLine -notmatch "Get-CimInstance" } | Measure-Object).Count' | tr -d '\r'
}

if [ "$(running)" = "0" ]; then
  echo "Nenhuma avaliação em andamento."
  rm -f PAUSAR
  exit 0
fi

touch PAUSAR
echo "Pedido de pausa enviado (arquivo PAUSAR). Esperando a avaliação salvar e encerrar..."
for _ in $(seq 1 120); do
  [ "$(running)" = "0" ] && break
  sleep 2
done

if [ "$(running)" != "0" ]; then
  echo "A avaliação ainda não encerrou (talvez esteja carregando o modelo). O pedido continua valendo."
  exit 1
fi
rm -f PAUSAR
echo "Pausada. Estado registrado:"
for f in results/*/*/evaluation/estado_avaliacao.json results/*/*/qualitative_glm/estado_avaliacao.json; do
  [ -f "$f" ] && { echo "--- $f"; cat "$f"; echo; }
done
nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | sed 's/^/VRAM em uso agora: /'

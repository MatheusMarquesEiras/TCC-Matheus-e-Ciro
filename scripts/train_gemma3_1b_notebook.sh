#!/usr/bin/env bash
# Gemma 3 1B no dataset completo fundido (medpt-top10-merged, 9 classes) no notebook
# com RTX 4060 Laptop (8 GB). Mesmos hiperparâmetros do registro (batch 1×32, 3 épocas),
# só com checkpoints mais frequentes.
#
# Medido em 29/10/2026 na 4060 Laptop: batch 1×32 usa ~2,9 GB de VRAM no início (pico de 7,08 GB
# no treino completo) e leva ~36 s/passo
# (2.112 passos, ~21 h). Batch 4×8 não acelera (~35 s/passo, GPU já a 100%) e encosta nos
# 8 GB, então fica o do registro. save_steps=50 é um checkpoint a cada ~30 min: no notebook
# interrupções (suspensão, bateria, atualização do Windows) são mais prováveis.
#
# Núcleos do processador (afeta muito o tempo total): o passo é limitado por uma única
# thread, e o i7-13620H tem núcleos rápidos (lógicos 0-11) e lentos (12-15). Medido em
# 30/10/2026 neste treino: ao desbloquear o notebook com o treino em segundo plano, o ritmo
# saltou de ~38 para ~61 s/passo; limitar o processo aos núcleos 0-11 voltou a ~40 s/passo,
# sem interromper o treino. A VRAM não mudou com a queda (~3,5 GB), então não era falta de
# memória de vídeo. No ritmo lento, os 2.112 passos inteiros levariam cerca de 13 h a mais.
#
# Por isso este script se limita aos núcleos 0-11 antes de chamar a fila: o Windows faz os
# processos filhos herdarem o limite, inclusive numa retomada. A máscara 0x0FFF é específica
# deste processador; em outro, o script só avisa e treina sem limite.
#
# Pode ser interrompido a qualquer momento; rodar de novo retoma do último checkpoint.
#   bash scripts/train_gemma3_1b_notebook.sh
cd "$(dirname "$0")/.."
CPU=$(powershell -NoProfile -Command '(Get-CimInstance Win32_Processor).Name')
case "$CPU" in
  *i7-13620H*)
    powershell -NoProfile -Command "(Get-Process -Id $(cat /proc/$$/winpid)).ProcessorAffinity = 0x0FFF"       && echo "=== Treino limitado aos núcleos rápidos (0-11) ==="       || echo "=== Aviso: não foi possível limitar os núcleos; o treino pode ficar mais lento ==="
    ;;
  *) echo "=== Aviso: processador $CPU sem máscara definida; treinando sem limite de núcleos ===" ;;
esac
TRAIN_ARGS="--save_steps 50" exec bash scripts/train_final_queue.sh gemma3-1b

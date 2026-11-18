"""Pausa limpa para execuções longas (avaliação do teste completo).

A execução é pausada de três formas equivalentes:

* criar o arquivo ``PAUSAR`` na raiz do projeto (``bash scripts/pausar.sh``);
* Ctrl+C no terminal (SIGINT) ou ``kill``/``taskkill`` sem força (SIGTERM, SIGBREAK);
* um segundo Ctrl+C, que corta na hora (para quando o exemplo atual travou).

Ao receber o pedido, o laço **para de mandar exemplos ao modelo**, deixa o exemplo
em andamento terminar, grava tudo em disco, registra onde parou em
``estado_avaliacao.json`` e encerra com o código ``PAUSE_EXIT_CODE``. Encerrar o
processo é o que devolve a VRAM da placa de vídeo.
"""

import json
import signal
import time
from pathlib import Path

from .paths import PROJECT_ROOT

#: Código de saída de "pausado com sucesso" (EX_TEMPFAIL); os scripts de fila param nele.
PAUSE_EXIT_CODE = 75

#: Arquivo-sentinela: se existir, a execução em andamento pausa.
PAUSE_FILE = PROJECT_ROOT / "PAUSAR"

STATE_FILE_NAME = "estado_avaliacao.json"

_reason: str | None = None


class PauseRequested(Exception):
    """Levantada pelo laço depois de salvar tudo, para o script encerrar."""


def _on_signal(signum, _frame) -> None:
    global _reason
    if _reason is not None:
        # Segundo pedido: o usuário quer sair já.
        raise KeyboardInterrupt
    _reason = f"sinal {signal.Signals(signum).name}"
    print(
        f"\n[pausa] {_reason} recebido: termino o exemplo atual, salvo e desligo o modelo. "
        "(Ctrl+C de novo corta na hora.)",
        flush=True,
    )


def install() -> None:
    """Registra os sinais e descarta um ``PAUSAR`` esquecido de uma pausa anterior."""
    global _reason
    _reason = None
    if PAUSE_FILE.exists():
        PAUSE_FILE.unlink()
        print(f"[pausa] removido {PAUSE_FILE.name} que sobrou de uma pausa anterior.")
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            signal.signal(sig, _on_signal)


def requested() -> bool:
    """Há um pedido de pausa (sinal ou arquivo ``PAUSAR``)?"""
    global _reason
    if _reason is None and PAUSE_FILE.exists():
        _reason = f"arquivo {PAUSE_FILE.name}"
        print(f"\n[pausa] {_reason} encontrado: termino o exemplo atual, salvo e desligo o modelo.",
              flush=True)
    return _reason is not None


def reason() -> str:
    return _reason or "desconhecido"


def consume() -> None:
    """Apaga o arquivo-sentinela depois que a pausa foi cumprida."""
    PAUSE_FILE.unlink(missing_ok=True)


def write_state(output_dir: Path, **fields) -> None:
    """Registra o estado da execução (rodando, pausado, concluído) em disco."""
    path = output_dir / STATE_FILE_NAME
    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError):
        state = {}
    state.update(fields)
    if state.get("status") != "pausado":
        # Campos de uma pausa anterior não valem mais depois que a execução retoma.
        state.pop("motivo", None)
        state.pop("como_retomar", None)
    state["atualizado_em"] = time.strftime("%d/%m/%Y %H:%M:%S")
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

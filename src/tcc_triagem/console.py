"""Saída de terminal em UTF-8.

O console do Windows usa cp1252 por padrão, que não codifica os caracteres do
progresso das avaliações (``→``, ``✓``, ``✗``) nem os acentos dos nomes das
especialidades. Sem isto, uma avaliação de 100 exemplos morre com
``UnicodeEncodeError`` depois de dezenas de chamadas ao grader — o trabalho é
perdido no meio.

Chamado explicitamente pelos scripts; o pacote não altera nada na importação.
"""

import sys


def enable_utf8_output() -> None:
    """Reconfigura stdout e stderr para UTF-8, sem falhar onde não se aplica."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # Stream redirecionado ou já fechado: seguir sem reconfigurar.
            pass

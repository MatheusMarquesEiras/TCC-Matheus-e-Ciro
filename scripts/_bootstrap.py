"""Coloca ``src/`` no ``sys.path`` para os scripts acharem o pacote ``tcc_triagem``.

Importado como primeira linha de cada script. Evita depender de instalação do
pacote no ambiente — o que manteria o `uv sync` mais frágil neste projeto, onde
o ambiente Windows já exige ajustes manuais (causal-conv1d e patch do TRL).

Também força a saída do terminal em UTF-8, senão o console cp1252 do Windows
derruba os scripts ao imprimir acentos e os símbolos de progresso.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcc_triagem.console import enable_utf8_output  # noqa: E402

enable_utf8_output()

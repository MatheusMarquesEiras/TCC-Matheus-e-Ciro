"""Resolução de caminhos relativos à raiz do repositório.

Todos os scripts aceitam caminhos relativos (``data/top10``) e funcionam a
partir de qualquer diretório de trabalho.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


def resolve_path(p: str | Path) -> Path:
    """Converte um caminho relativo à raiz do projeto em caminho absoluto."""
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_dir(base: str | Path, model_key: str, dataset_key: str) -> Path:
    """Diretório canônico de uma execução: ``base/<modelo>/<dataset>``.

    Mantém os artefatos de cada par (modelo, dataset) isolados, o que é o que
    permite comparar várias execuções lado a lado no script 06.
    """
    return resolve_path(base) / model_key / dataset_key

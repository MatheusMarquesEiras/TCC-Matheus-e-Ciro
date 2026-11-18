"""Leitura das fontes brutas de dados e limpeza dos rótulos.

Isola do resto do pipeline a única parte que depende do formato de cada corpus.
Qualquer fonte que produza pares ``(texto, rótulo)`` serve para o restante.
"""

from pathlib import Path

from datasets import load_dataset

from .dataset_registry import DatasetSpec
from .paths import resolve_path

_LOCAL_LOADERS = {
    ".csv": "csv",
    ".tsv": "csv",
    ".json": "json",
    ".jsonl": "json",
    ".parquet": "parquet",
}


def _load_hf(spec: DatasetSpec):
    print(f"Baixando {spec.path} (split '{spec.split}') do HuggingFace...")
    return load_dataset(spec.path, split=spec.split)


def _load_local(spec: DatasetSpec):
    path = resolve_path(spec.path)
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo do dataset '{spec.key}' não encontrado: {path}"
        )
    loader = _LOCAL_LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(
            f"Extensão '{path.suffix}' não suportada para dataset local. "
            f"Use uma de: {', '.join(sorted(_LOCAL_LOADERS))}"
        )
    print(f"Lendo dataset local: {path}")
    kwargs = {"delimiter": "\t"} if path.suffix.lower() == ".tsv" else {}
    return load_dataset(loader, data_files=str(path), split="train", **kwargs)


def clean_label(raw: str, spec: DatasetSpec) -> str:
    """Normaliza um rótulo bruto conforme a especificação do dataset.

    Aplica, em ordem: corte no separador de rótulos múltiplos, ``strip`` e
    fusão de classes (``label_aliases``).
    """
    if raw is None:
        return ""
    label = str(raw)
    if spec.label_separator:
        label = label.split(spec.label_separator)[0]
    label = label.strip()
    return spec.label_aliases.get(label, label)


def load_pairs(spec: DatasetSpec) -> tuple[list[str], list[str]]:
    """Carrega a fonte e devolve ``(textos, rótulos)`` já limpos.

    Descarta linhas sem texto, sem rótulo ou cujo rótulo está em ``drop_labels``.
    """
    if spec.source == "huggingface":
        ds = _load_hf(spec)
    elif spec.source == "local":
        ds = _load_local(spec)
    else:
        raise ValueError(
            f"Fonte '{spec.source}' desconhecida no dataset '{spec.key}'. "
            "Use 'huggingface' ou 'local'."
        )

    for column in (spec.text_column, spec.label_column):
        if column not in ds.column_names:
            raise KeyError(
                f"Coluna '{column}' não existe em '{spec.path}'. "
                f"Colunas disponíveis: {', '.join(ds.column_names)}"
            )

    print(f"Total de exemplos brutos: {len(ds)}")

    texts, labels = [], []
    dropped = set(spec.drop_labels)
    for text, raw_label in zip(ds[spec.text_column], ds[spec.label_column]):
        label = clean_label(raw_label, spec)
        if not text or not label or label in dropped:
            continue
        texts.append(str(text).strip())
        labels.append(label)

    print(f"Após limpeza de rótulos: {len(texts)} exemplos")
    return texts, labels


def describe_source(spec: DatasetSpec) -> str:
    if spec.source == "huggingface":
        return f"HuggingFace: {spec.path}"
    return f"arquivo local: {Path(spec.path).as_posix()}"

"""Preparação dos splits de treino, validação e teste.

Recebe um ``DatasetSpec``, aplica seleção de classes e cap por classe, faz o
split estratificado e grava os arquivos em ChatML mais um ``metadata.json``
que descreve exatamente como aquele conjunto foi construído.

O ``metadata.json`` é o contrato entre os scripts: treino, avaliação e demo
leem dele a lista de rótulos e o system prompt, então nenhum deles precisa
saber de qual dataset veio.
"""

import gzip
import json
import random
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from sklearn.model_selection import train_test_split

from .data_sources import describe_source, load_pairs
from .dataset_registry import DatasetSpec
from .paths import resolve_path
from .prompting import build_system_prompt, to_chatml

#: Mínimo absoluto para uma classe sobreviver ao split estratificado 80/10/10.
_MIN_FOR_STRATIFY = 3


def _cap_per_class(
    texts: list[str], labels: list[str], max_per_class: int, seed: int
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    by_class: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        by_class[label].append(i)

    selected: list[int] = []
    for indices in by_class.values():
        selected.extend(rng.sample(indices, min(max_per_class, len(indices))))
    selected.sort()
    return [texts[i] for i in selected], [labels[i] for i in selected]


def _select_labels(counter: Counter, spec: DatasetSpec) -> list[str]:
    """Escolhe as classes que entram no dataset final."""
    eligible = [
        (label, n)
        for label, n in counter.most_common()
        if n >= max(spec.min_per_class, _MIN_FOR_STRATIFY)
    ]
    discarded = len(counter) - len(eligible)
    if discarded:
        print(
            f"  {discarded} classe(s) descartada(s) por terem menos de "
            f"{max(spec.min_per_class, _MIN_FOR_STRATIFY)} exemplos."
        )
    if spec.top_n is not None:
        eligible = eligible[: spec.top_n]
    return [label for label, _ in eligible]


def prepare_dataset(
    spec: DatasetSpec,
    output_dir: str | Path | None = None,
    max_per_class: int | None = None,
    top_n: int | None = None,
) -> dict:
    """Executa a preparação completa e devolve o metadata gravado.

    ``max_per_class`` e ``top_n`` sobrescrevem o que estiver no ``spec``,
    permitindo variações rápidas sem editar o registro.
    """
    if top_n is not None:
        spec = replace(spec, top_n=top_n)
    if max_per_class is not None:
        spec = replace(spec, max_per_class=max_per_class)

    output_path = resolve_path(output_dir or spec.resolved_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {spec.label}")
    print(f"Fonte:   {describe_source(spec)}")
    print(f"Saída:   {output_path}\n")

    texts, labels = load_pairs(spec)
    if spec.label_aliases:
        print(f"Fusões de classe aplicadas: {spec.label_aliases}")

    counter = Counter(labels)
    selected_labels = _select_labels(counter, spec)

    print(f"\n{len(selected_labels)} classe(s) selecionada(s):")
    for i, label in enumerate(selected_labels, 1):
        print(f"  {i:2d}. {label}: {counter[label]} exemplos")

    keep = set(selected_labels)
    filtered = [(t, s) for t, s in zip(texts, labels) if s in keep]
    if not filtered:
        raise ValueError(
            f"Nenhum exemplo restou para o dataset '{spec.key}'. "
            "Verifique as colunas, o separador de rótulos e os filtros."
        )
    texts_f, labels_f = map(list, zip(*filtered))
    print(f"\nTotal após filtragem: {len(texts_f)} exemplos")

    if spec.max_per_class is not None:
        texts_f, labels_f = _cap_per_class(
            texts_f, labels_f, spec.max_per_class, spec.seed
        )
        print(f"Após cap de {spec.max_per_class} por classe: {len(texts_f)} exemplos")

    # Split estratificado: primeiro separa treino, depois divide o resto em
    # validação e teste, preservando a proporção das classes nos três.
    holdout = spec.val_ratio + spec.test_ratio
    x_train, x_temp, y_train, y_temp = train_test_split(
        texts_f, labels_f, test_size=holdout, stratify=labels_f, random_state=spec.seed
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=spec.test_ratio / holdout,
        stratify=y_temp,
        random_state=spec.seed,
    )

    print("\nSplit resultante:")
    print(f"  Treino:    {len(x_train)}")
    print(f"  Validação: {len(x_val)}")
    print(f"  Teste:     {len(x_test)}")

    splits = {
        "train": (x_train, y_train),
        "validation": (x_val, y_val),
        "test": (x_test, y_test),
    }
    for split_name, (xs, ys) in splits.items():
        records = [
            to_chatml(x, y, selected_labels, spec.system_template) for x, y in zip(xs, ys)
        ]
        out_file = output_path / f"{split_name}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"  Salvo: {out_file}")

    label2id = {label: i for i, label in enumerate(sorted(selected_labels))}
    metadata = {
        "dataset_key": spec.key,
        "dataset_label": spec.label,
        "source": spec.source,
        "source_path": spec.path,
        "citation": spec.citation,
        "language": spec.language,
        "labels": selected_labels,
        # Chave legada mantida para não quebrar artefatos e relatórios anteriores.
        "specialties": selected_labels,
        "top_n": spec.top_n,
        "max_per_class": spec.max_per_class,
        "min_per_class": spec.min_per_class,
        "label_aliases": spec.label_aliases,
        "drop_labels": list(spec.drop_labels),
        "seed": spec.seed,
        "system_prompt": build_system_prompt(selected_labels, spec.system_template),
        "label2id": label2id,
        "id2label": {str(i): label for label, i in label2id.items()},
        "split_sizes": {name: len(xs) for name, (xs, _) in splits.items()},
        "class_distribution": {
            name: dict(Counter(ys)) for name, (_, ys) in splits.items()
        },
    }
    meta_file = output_path / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  Salvo: {meta_file}")

    return metadata


# --------------------------------------------------------------------------
# Leitura dos splits já preparados
# --------------------------------------------------------------------------

def load_metadata(data_dir: str | Path) -> dict:
    """Lê o ``metadata.json`` de um diretório preparado pelo script 01."""
    path = resolve_path(data_dir) / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não existe. Rode antes: "
            f"uv run python scripts/01_prepare_data.py --dataset <chave>"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_split(data_dir: str | Path, split: str) -> list[dict]:
    """Lê um split em ChatML (``train``, ``validation`` ou ``test``).

    Aceita também ``<split>.json.gz``: os splits do dataset completo vão para o
    repositório comprimidos, porque o ``train.json`` passa do limite de 100 MB
    por arquivo do GitHub.
    """
    path = resolve_path(data_dir) / f"{split}.json"
    if not path.exists() and path.with_suffix(".json.gz").exists():
        with gzip.open(path.with_suffix(".json.gz"), "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def labels_of(metadata: dict) -> list[str]:
    """Lista de rótulos do metadata, aceitando o formato legado."""
    return metadata.get("labels") or metadata["specialties"]


def question_of(record: dict) -> str:
    """Extrai a queixa do paciente de um registro ChatML."""
    return record["messages"][1]["content"]


def label_of(record: dict) -> str:
    """Extrai o rótulo verdadeiro de um registro ChatML."""
    return record["messages"][2]["content"]

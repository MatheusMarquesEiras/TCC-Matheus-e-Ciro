"""Catálogo de datasets suportados pelo pipeline.

Um dataset é descrito por dados, não por código: quais colunas carregam o texto
e o rótulo, como o rótulo bruto é limpo, quantas classes entram e como o split
é feito. Acrescentar um dataset novo — outro corpus em PT-BR, um CSV do próprio
hospital — é acrescentar uma entrada em ``DATASETS`` ou chamar ``register()``.

Fontes aceitas:
    ``huggingface``  qualquer repositório de dataset do Hub
    ``local``        arquivo ``.csv``, ``.json`` ou ``.jsonl`` dentro do repositório
"""

from dataclasses import dataclass, field

from .prompting import DEFAULT_SYSTEM_TEMPLATE


@dataclass(frozen=True)
class DatasetSpec:
    """Descrição completa de um dataset de classificação de especialidade."""

    key: str
    label: str
    source: str  # "huggingface" | "local"
    path: str  # repo id no Hub, ou caminho relativo à raiz do projeto
    text_column: str
    label_column: str
    split: str = "train"
    output_dir: str | None = None  # padrão: data/<key>

    # Limpeza do rótulo bruto -------------------------------------------------
    #: Separador de rótulos múltiplos no campo original. Só o primeiro é usado.
    #: No MedPT é ";" — NÃO usar "," porque nomes como
    #: "Ortopedista - traumatologista" fazem parte de um único rótulo.
    label_separator: str | None = ";"
    #: Fusão de classes: ``{rótulo_original: rótulo_final}``. Aplicada antes da
    #: contagem de frequência, então classes fundidas somam seus exemplos.
    label_aliases: dict[str, str] = field(default_factory=dict)
    #: Rótulos descartados por completo.
    drop_labels: tuple[str, ...] = ()

    # Seleção de classes e exemplos -------------------------------------------
    top_n: int | None = 10  # None = todas as classes
    max_per_class: int | None = None  # cap por classe, aplicado após o filtro
    min_per_class: int = 0  # descarta classes raras demais para estratificar

    # Split -------------------------------------------------------------------
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42

    # Prompt ------------------------------------------------------------------
    system_template: str = DEFAULT_SYSTEM_TEMPLATE
    language: str = "pt-BR"
    citation: str = ""

    @property
    def resolved_output_dir(self) -> str:
        return self.output_dir or f"data/{self.key}"


# --------------------------------------------------------------------------
# Registro
# --------------------------------------------------------------------------

MEDPT_CITATION = (
    "FÄRBER, F. B. et al. MedPT: a massive medical question answering dataset "
    "for Brazilian-Portuguese speakers. arXiv:2511.11878, 2026."
)

_MEDPT_COMMON = dict(
    source="huggingface",
    path="AKCIT/MedPT",
    text_column="question",
    label_column="medical_specialty",
    label_separator=";",
    citation=MEDPT_CITATION,
)

#: Fusão recomendada no relatório do Experimento 02: `Psicólogo, Psicanalista`
#: ficou com F1 = 0 por sobreposição semântica quase total com `Psicólogo`,
#: e a distinção raramente muda a conduta de triagem.
MEDPT_PSICO_MERGE = {"Psicólogo, Psicanalista": "Psicólogo"}

DATASETS: dict[str, DatasetSpec] = {
    "medpt-top10": DatasetSpec(
        key="medpt-top10",
        label="MedPT — 10 especialidades mais frequentes",
        top_n=10,
        output_dir="data/top10",
        **_MEDPT_COMMON,
    ),
    "medpt-top10-merged": DatasetSpec(
        key="medpt-top10-merged",
        label="MedPT — top-10 com Psicólogo e Psicanalista fundidos (9 classes)",
        # top_n=9, não 10: a fusão libera uma vaga, e manter 10 promoveria uma
        # 11ª especialidade para dentro do conjunto. O resultado deixaria de ser
        # comparável ao Experimento 02, que é justamente o ponto da variante.
        # Com 9, o conjunto é exatamente o top-10 original menos a classe fundida.
        top_n=9,
        label_aliases=MEDPT_PSICO_MERGE,
        output_dir="data/top10_merged",
        **_MEDPT_COMMON,
    ),
    "medpt-top10-500": DatasetSpec(
        key="medpt-top10-500",
        label="MedPT — top-10 com 500 exemplos por classe (piloto)",
        top_n=10,
        max_per_class=500,
        output_dir="data/top10_500",
        **_MEDPT_COMMON,
    ),
    "medpt-top20": DatasetSpec(
        key="medpt-top20",
        label="MedPT — 20 especialidades mais frequentes",
        top_n=20,
        output_dir="data/top20",
        **_MEDPT_COMMON,
    ),
}

#: Dataset usado por padrão quando nenhum é informado.
DEFAULT_DATASET = "medpt-top10"


def register(spec: DatasetSpec) -> DatasetSpec:
    """Adiciona um dataset ao registro em tempo de execução.

    Útil para experimentos pontuais sem editar este arquivo::

        from tcc_triagem.dataset_registry import DatasetSpec, register

        register(DatasetSpec(
            key="triagem-local",
            label="Queixas anotadas do ambulatório",
            source="local",
            path="data/raw/queixas.csv",
            text_column="queixa",
            label_column="especialidade",
            label_separator=None,
            top_n=None,
            min_per_class=20,
        ))
    """
    DATASETS[spec.key] = spec
    return spec


def get_dataset(key: str) -> DatasetSpec:
    try:
        return DATASETS[key]
    except KeyError:
        raise KeyError(
            f"Dataset '{key}' não está registrado. Disponíveis: {', '.join(DATASETS)}"
        ) from None


def list_datasets() -> list[DatasetSpec]:
    return list(DATASETS.values())


def describe_datasets() -> str:
    """Tabela textual dos datasets — usada por ``--list`` nos scripts."""
    header = f"{'chave':<20} {'classes':>7} {'cap/classe':>11}  {'saída':<22} descrição"
    lines = [header, "-" * len(header)]
    for spec in DATASETS.values():
        top_n = "todas" if spec.top_n is None else str(spec.top_n)
        cap = "—" if spec.max_per_class is None else str(spec.max_per_class)
        lines.append(
            f"{spec.key:<20} {top_n:>7} {cap:>11}  "
            f"{spec.resolved_output_dir:<22} {spec.label}"
        )
    return "\n".join(lines)

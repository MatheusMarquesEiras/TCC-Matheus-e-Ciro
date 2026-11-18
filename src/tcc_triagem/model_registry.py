"""Catálogo de modelos base suportados pelo pipeline.

Adicionar um modelo novo é acrescentar uma entrada em ``MODELS`` — nenhum script
precisa ser alterado. Cada entrada carrega o repositório no HuggingFace, o
carregador correto do Unsloth e os hiperparâmetros de treino que cabem na GPU
alvo do trabalho (RTX 4070 12 GB).

Convenção dos hiperparâmetros: o **batch efetivo é sempre 32**
(``batch_size × grad_accum``). Modelos maiores reduzem ``batch_size`` e
aumentam ``grad_accum`` na mesma proporção, de modo que a dinâmica de
otimização se mantém comparável entre os modelos — o que muda é só quanta
VRAM cada passo consome.
"""

from dataclasses import dataclass, field

DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class LoRASpec:
    """Configuração dos adaptadores LoRA."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.0  # 0 é obrigatório para o Unsloth aplicar os fast kernels
    bias: str = "none"
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES
    random_state: int = 42


@dataclass(frozen=True)
class TrainingSpec:
    """Hiperparâmetros de treino por modelo."""

    epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 8
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    optim: str = "adamw_8bit"
    packing: bool = True
    save_steps: int = 500
    save_total_limit: int = 5
    logging_steps: int = 50
    # Descarta do TREINO exemplos acima deste número de tokens (None = usa todos). Serve a
    # modelos cujo micro-lote precisa ser grande para ser rápido: um lote com um exemplo
    # muito longo é preenchido até ele e estoura a VRAM.
    max_example_tokens: int | None = None
    # Batch da avaliação (None = o mesmo do treino). Na avaliação o modelo calcula os
    # logits de todo o vocabulário em todas as posições, então lotes grandes custam caro.
    eval_batch_size: int | None = None

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum


@dataclass(frozen=True)
class ModelSpec:
    """Descrição completa de um modelo base."""

    key: str
    hf_id: str
    label: str
    family: str
    params: str
    variant: str  # "instruct" ou "base"
    loader: str = "language"  # "language" (FastLanguageModel) ou "multimodal" (FastModel)
    max_seq_length: int = 1024
    chat_template: str | None = None  # nome do template Unsloth p/ modelos sem template
    chat_template_kwargs: dict = field(default_factory=dict)
    lora: LoRASpec = field(default_factory=LoRASpec)
    training: TrainingSpec = field(default_factory=TrainingSpec)
    notes: str = ""

    @property
    def is_multimodal(self) -> bool:
        return self.loader == "multimodal"

    def slug(self, dataset_key: str) -> str:
        """Identificador de execução usado nos diretórios de saída."""
        return f"{self.key}/{dataset_key}"


# --------------------------------------------------------------------------
# Registro
# --------------------------------------------------------------------------

MODELS: dict[str, ModelSpec] = {
    # ----------------------------- Qwen 3.5 -----------------------------
    "qwen35-0.8b": ModelSpec(
        key="qwen35-0.8b",
        hf_id="Qwen/Qwen3.5-0.8B",
        label="Qwen3.5-0.8B",
        family="qwen3.5",
        params="758.782.784",
        variant="instruct",
        loader="language",
        max_seq_length=1024,
        chat_template_kwargs={"enable_thinking": False},
        training=TrainingSpec(batch_size=4, grad_accum=8, learning_rate=2e-4),
        notes=(
            "Modelo do Experimento 02 (F1 macro 0,7911). Roda obrigatoriamente em "
            "non-thinking mode: no thinking mode entra em loop infinito."
        ),
    ),
    "qwen35-2b": ModelSpec(
        key="qwen35-2b",
        hf_id="Qwen/Qwen3.5-2B",
        label="Qwen3.5-2B",
        family="qwen3.5",
        params="~2,0 B",
        variant="instruct",
        loader="language",
        max_seq_length=1024,
        chat_template_kwargs={"enable_thinking": False},
        training=TrainingSpec(batch_size=2, grad_accum=16, learning_rate=2e-4),
        notes="Mesma família do 0.8B, com batch menor para caber em 12 GB.",
    ),
    # ------------------------------ Gemma 3 -----------------------------
    "gemma3-1b": ModelSpec(
        key="gemma3-1b",
        hf_id="unsloth/gemma-3-1b-pt",
        label="Gemma 3 1B (base)",
        family="gemma3",
        params="~1,0 B",
        variant="base",
        loader="language",  # gemma3_text — só texto
        max_seq_length=1024,
        chat_template="gemma-3",  # o checkpoint base não traz chat template
        # batch 1, não 4: com packing ativo cada linha vira uma sequência cheia
        # de max_seq_length tokens (não a queixa curta original). batch=4 e
        # depois batch=2 chegaram a ~98% da VRAM da RTX 4070 (quase travando
        # por thrashing de memória) mesmo nesse modelo pequeno — o vocabulário
        # grande do Gemma 3 deixa a camada de embedding/LM-head cara mesmo em
        # 4-bit. Batch efetivo se mantém em 32.
        training=TrainingSpec(batch_size=1, grad_accum=32, learning_rate=2e-4),
        notes=(
            "Checkpoint pré-treinado (-pt), sem instruction tuning. O chat template "
            "gemma-3 é aplicado pelo pipeline antes do treino."
        ),
    ),
    "gemma3-4b": ModelSpec(
        key="gemma3-4b",
        hf_id="unsloth/gemma-3-4b-pt",
        label="Gemma 3 4B (base)",
        family="gemma3",
        params="~4,3 B",
        variant="base",
        loader="multimodal",  # Gemma 3 4B é multimodal — carregado com FastModel
        max_seq_length=1024,
        chat_template="gemma-3",
        # Medido na RTX 4070 (diagnóstico de 25/10/2026, batch efetivo 32 em todos): 2×16 leva
        # ~27,5 s/passo (~104 h); 8×4 leva ~8,6 s/passo (~33 h) com a mesma perda por passo. O
        # custo por micro-lote é quase fixo (pesos 4-bit→bf16, recomputo do checkpointing) e
        # os exemplos são curtos (~140 tokens), então micro-lotes de 2 desperdiçavam a GPU.
        # 16×2 é mais rápido (~19 h) mas passa dos 12 GB de VRAM. Com batch 8 um exemplo muito
        # longo estoura a memória, por isso exemplos acima de 512 tokens (86 de 145.321, 0,06%)
        # ficam fora do treino; a avaliação usa batch 2 e mantém o conjunto inteiro.
        # save_steps=100: com ~8,6 s/passo é um checkpoint a cada ~15 min.
        training=TrainingSpec(
            batch_size=8, grad_accum=4, learning_rate=1e-4, save_steps=100,
            max_example_tokens=512, eval_batch_size=2,
        ),
        notes=(
            "Multimodal: as camadas de visão ficam congeladas e só as de linguagem "
            "recebem LoRA. Maior modelo do estudo — batch pequeno e lr menor."
        ),
    ),
    # Variantes instruction-tuned, para o contraste base × instruct.
    "gemma3-1b-it": ModelSpec(
        key="gemma3-1b-it",
        hf_id="unsloth/gemma-3-1b-it",
        label="Gemma 3 1B (instruct)",
        family="gemma3",
        params="~1,0 B",
        variant="instruct",
        loader="language",
        max_seq_length=1024,
        # batch 1 — mesmo motivo do gemma3-1b (ver nota lá): packing enche
        # cada linha até max_seq_length tokens; batch=4 e batch=2 quase
        # saturaram a VRAM da RTX 4070 no checkpoint base, mesma arquitetura.
        training=TrainingSpec(batch_size=1, grad_accum=32, learning_rate=2e-4),
        notes="Variante instruction-tuned do gemma3-1b; já traz chat template.",
    ),
    "gemma3-4b-it": ModelSpec(
        key="gemma3-4b-it",
        hf_id="unsloth/gemma-3-4b-it",
        label="Gemma 3 4B (instruct)",
        family="gemma3",
        params="~4,3 B",
        variant="instruct",
        loader="multimodal",
        max_seq_length=1024,
        training=TrainingSpec(batch_size=1, grad_accum=32, learning_rate=1e-4),
        notes="Variante instruction-tuned do gemma3-4b.",
    ),
}

#: Modelos comparados no estudo principal do TCC.
STUDY_MODELS = ("qwen35-0.8b", "qwen35-2b", "gemma3-1b", "gemma3-4b")


def get_model(key: str) -> ModelSpec:
    try:
        return MODELS[key]
    except KeyError:
        raise KeyError(
            f"Modelo '{key}' não está registrado. Disponíveis: {', '.join(MODELS)}"
        ) from None


def list_models() -> list[ModelSpec]:
    return list(MODELS.values())


def describe_models() -> str:
    """Tabela textual dos modelos — usada por ``--list`` nos scripts."""
    header = f"{'chave':<14} {'modelo':<26} {'params':<12} {'batch ef.':>9}  repositório"
    lines = [header, "-" * len(header)]
    for spec in MODELS.values():
        lines.append(
            f"{spec.key:<14} {spec.label:<26} {spec.params:<12} "
            f"{spec.training.effective_batch:>9}  {spec.hf_id}"
        )
    return "\n".join(lines)

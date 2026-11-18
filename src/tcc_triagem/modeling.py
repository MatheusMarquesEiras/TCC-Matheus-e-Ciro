"""Carregamento de modelos via Unsloth e anexo dos adaptadores LoRA.

Concentra as diferenças entre as famílias suportadas:

* **Qwen3.5** exige ``enable_thinking=False`` no chat template — no thinking mode
  o 0.8B entra em loop infinito.
* **Gemma 3 base (-pt)** não traz chat template no checkpoint; o template
  ``gemma-3`` é aplicado aqui antes de qualquer formatação.
* **Gemma 3 4B** é multimodal e carrega com ``FastModel``; as camadas de visão
  ficam congeladas e só as de linguagem recebem LoRA.

O import do ``unsloth`` acontece no topo deste módulo de propósito: ele precisa
vir antes de ``transformers``/``trl`` para aplicar os patches de performance.
"""

import unsloth  # noqa: F401  — precisa ser importado antes de transformers/trl
from unsloth import FastLanguageModel

try:  # FastModel cobre os modelos multimodais (Gemma 3 4B)
    from unsloth import FastModel
except ImportError:  # versões antigas do Unsloth
    FastModel = FastLanguageModel

from .model_registry import ModelSpec


def loader_for(spec: ModelSpec):
    """Classe do Unsloth adequada ao modelo."""
    return FastModel if spec.is_multimodal else FastLanguageModel


def text_tokenizer(tokenizer):
    """Devolve o tokenizer de texto.

    Modelos multimodais carregam um ``Processor``; o tokenizer de texto fica
    em ``.tokenizer``. Encoding e decoding devem usar sempre este objeto.
    """
    return getattr(tokenizer, "tokenizer", tokenizer)


def ensure_chat_template(tokenizer, spec: ModelSpec):
    """Garante que o tokenizer tenha chat template.

    Só age quando o modelo declara um template no registro e o checkpoint não
    traz nenhum — caso dos checkpoints base (``-pt``) do Gemma 3.
    """
    if spec.chat_template is None:
        return tokenizer

    inner = text_tokenizer(tokenizer)
    if getattr(inner, "chat_template", None):
        return tokenizer

    from unsloth.chat_templates import get_chat_template

    print(f"  Aplicando chat template '{spec.chat_template}' (checkpoint base sem template)")
    patched = get_chat_template(inner, chat_template=spec.chat_template)
    template = getattr(patched, "chat_template", None)
    if template:
        inner.chat_template = template
        if inner is not tokenizer:
            tokenizer.chat_template = template
    return patched if inner is tokenizer else tokenizer


def _merge_system_into_user(messages: list[dict]) -> list[dict]:
    """Funde o turno ``system`` no primeiro ``user``.

    Alguns chat templates recusam o papel ``system``. Em vez de perder a
    instrução de triagem, ela vira o cabeçalho da primeira mensagem do usuário.
    """
    system = [m["content"] for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]
    if not system or not rest:
        return rest or messages
    merged = dict(rest[0])
    merged["content"] = "\n\n".join([*system, merged["content"]])
    return [merged, *rest[1:]]


def apply_chat_template(
    tokenizer,
    messages: list[dict],
    spec: ModelSpec,
    add_generation_prompt: bool,
) -> str:
    """Formata mensagens no template do modelo, respeitando suas particularidades.

    Trata dois casos que variam entre famílias: kwargs específicos do template
    (``enable_thinking`` no Qwen3.5) e templates que não aceitam papel ``system``.
    """
    kwargs = dict(spec.chat_template_kwargs)

    def render(msgs, extra):
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt, **extra
        )

    # Tentativas em ordem de fidelidade: primeiro o formato pretendido, depois
    # sem os kwargs (template que não os conhece) e por fim com o system fundido
    # no user (template que recusa esse papel).
    merged = _merge_system_into_user(messages)
    attempts = [(messages, kwargs)]
    if kwargs:
        attempts.append((messages, {}))
    attempts.append((merged, kwargs))
    if kwargs:
        attempts.append((merged, {}))

    last_error: Exception | None = None
    for msgs, extra in attempts:
        try:
            text = render(msgs, extra)
        except Exception as error:  # noqa: BLE001 — o próximo formato pode funcionar
            last_error = error
            continue
        if (
            add_generation_prompt
            and not extra
            and kwargs.get("enable_thinking") is False
        ):
            # O template ignorou enable_thinking: força o modo non-thinking com
            # um bloco de raciocínio vazio antes da resposta.
            text += "<think>\n\n</think>\n\n"
        return text

    raise RuntimeError(
        f"Nenhum formato de chat template funcionou para {spec.key}."
    ) from last_error


def load_model(
    spec: ModelSpec,
    model_dir: str | None = None,
    for_inference: bool = False,
    load_in_4bit: bool = True,
):
    """Carrega o modelo base ou um checkpoint com adaptadores já treinados.

    ``model_dir`` aponta para a saída do script 02 (``.../final``); sem ele,
    carrega o checkpoint original do HuggingFace.
    """
    loader = loader_for(spec)
    source = model_dir or spec.hf_id
    kind = "adaptadores LoRA" if model_dir else "modelo base"
    print(f"  Carregando {kind} ({loader.__name__}): {source}")

    model, tokenizer = loader.from_pretrained(
        model_name=str(source),
        max_seq_length=spec.max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=None,  # detecção automática (bf16 em GPUs Ampere+)
    )
    tokenizer = ensure_chat_template(tokenizer, spec)

    if for_inference:
        enable = getattr(loader, "for_inference", None)
        if enable is not None:
            enable(model)
    return model, tokenizer


def attach_lora(model, spec: ModelSpec):
    """Anexa os adaptadores LoRA conforme a configuração do modelo."""
    loader = loader_for(spec)
    lora = spec.lora

    kwargs = dict(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        bias=lora.bias,
        use_gradient_checkpointing="unsloth",  # ~30% menos VRAM
        random_state=lora.random_state,
    )

    if spec.is_multimodal:
        # Tarefa é só texto: as camadas de visão ficam congeladas.
        kwargs.update(
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
        )
    else:
        kwargs["target_modules"] = list(lora.target_modules)

    model = loader.get_peft_model(model, **kwargs)
    model.print_trainable_parameters()
    return model


def trainable_parameter_stats(model) -> dict:
    """Contagem de parâmetros treináveis — vai para o relatório do TCC."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 4) if total else 0.0,
    }

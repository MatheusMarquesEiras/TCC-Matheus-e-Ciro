"""Rotina de fine-tuning QLoRA compartilhada por todos os modelos.

O mesmo esquema de treino roda para Qwen3.5 e Gemma 3: o que muda por modelo
está no ``ModelSpec`` (carregador, batch, learning rate, chat template), e o que
muda por dataset está no ``metadata.json`` produzido pelo script 01.

Os scripts em ``scripts/train/`` são cascas de uma linha sobre ``run_training``,
para que cada modelo tenha um ponto de entrada próprio e rastreável no histórico
do repositório, sem duplicar a lógica.
"""

import json
import math
import os
from dataclasses import replace
import shutil
import time
from pathlib import Path

# modeling importa unsloth, que precisa vir antes de transformers/trl.
from .modeling import apply_chat_template, attach_lora, load_model, trainable_parameter_stats

from .data_prep import load_metadata, load_split
from .dataset_registry import get_dataset
from .model_registry import ModelSpec, get_model
from .paths import resolve_path


def _find_latest_checkpoint(output_path: Path) -> str | None:
    if not output_path.exists():
        return None
    checkpoints = sorted(
        (d for d in output_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")),
        key=lambda d: int(d.name.split("-")[-1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


#: Número mínimo de checkpoints ao longo de uma execução. Garante que dá para
#: interromper o treino e retomar sem perder mais do que ~1/N do trabalho.
MIN_CHECKPOINTS_PER_RUN = 12


def _total_steps(n_train: int, spec: ModelSpec, epochs: int) -> int:
    """Número de passos do otimizador em toda a execução."""
    steps_per_epoch = math.ceil(n_train / spec.training.effective_batch)
    return steps_per_epoch * epochs


def _warmup_steps(total_steps: int, spec: ModelSpec) -> int:
    return max(1, round(total_steps * spec.training.warmup_ratio))


def _save_steps(total_steps: int, spec: ModelSpec) -> int:
    """Intervalo de salvamento que sempre produz checkpoints nesta execução.

    O intervalo configurado no registro (``save_steps``) é pensado para o
    dataset completo, onde a execução tem milhares de passos. Numa execução
    curta — o piloto de 500 exemplos/classe tem 375 passos — um intervalo de
    500 nunca é alcançado e **nenhum** checkpoint intermediário é gravado: o
    treino só salva ao terminar, e interromper no meio perde tudo.

    Aqui o intervalo é reduzido ao necessário para caber pelo menos
    ``MIN_CHECKPOINTS_PER_RUN`` checkpoints na execução, sem nunca ultrapassar
    o valor configurado.
    """
    target = max(1, math.ceil(total_steps / MIN_CHECKPOINTS_PER_RUN))
    return min(spec.training.save_steps, target)


def _limit_gpu_memory() -> None:
    """Teto de memória da GPU para o alocador do PyTorch (variável TCC_GPU_MEMORY_FRACTION).

    No Windows, quando o alocador passa da VRAM física o driver empurra o excedente para a
    RAM em vez de dar erro, e o treino pode ficar várias vezes mais lento de repente (visto no
    Gemma 3 4B: de ~7 s para ~35 s por passo). Com o teto, o PyTorch libera os blocos em cache e
    tenta de novo antes de pedir mais memória, e só falha se o uso real não couber.
    """
    fraction = os.environ.get("TCC_GPU_MEMORY_FRACTION")
    if not fraction:
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(fraction))
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Teto de memória da GPU: {float(fraction):.0%} de {total:.1f} GB")


def _peak_vram_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.max_memory_reserved() / 1024**3, 2)
    except Exception:  # noqa: BLE001 — telemetria opcional, nunca derruba o treino
        return None


def _loss_history(log_history: list[dict], total_steps: int, epochs: int) -> dict:
    """Perdas de treino e validação a partir do histórico do trainer.

    ``train_result.training_loss`` não serve num treino retomado de checkpoint: o
    trainer soma a perda só do trecho executado e divide pelo total de passos, então
    o valor sai muito abaixo do real. O ``log_history`` é restaurado do checkpoint e
    cobre o treino inteiro.
    """
    train = [(e["step"], e["loss"]) for e in log_history if "loss" in e]
    per_epoch = []
    for k in range(epochs):
        lo, hi = total_steps * k / epochs, total_steps * (k + 1) / epochs
        values = [loss for step, loss in train if lo < step <= hi]
        per_epoch.append(round(sum(values) / len(values), 4) if values else None)
    last = [loss for _, loss in train[-10:]]
    evals = [round(e["eval_loss"], 4) for e in log_history if "eval_loss" in e]
    return {
        "train_loss_per_epoch": per_epoch,
        "final_train_loss": round(sum(last) / len(last), 4) if last else None,
        "eval_loss_per_epoch": evals,
    }


def resolve_data_dir(dataset_key: str | None, data_dir: str | None) -> tuple[Path, str]:
    """Resolve o diretório dos splits a partir da chave do dataset ou de um caminho."""
    if data_dir:
        path = resolve_path(data_dir)
        metadata = load_metadata(path)
        return path, metadata.get("dataset_key", path.name)
    key = dataset_key or "medpt-top10"
    return resolve_path(get_dataset(key).resolved_output_dir), key


def run_training(
    model_key: str,
    dataset_key: str | None = None,
    data_dir: str | None = None,
    output_dir: str | None = None,
    epochs: int | None = None,
    run_name: str | None = None,
    report_to: str = "none",
    resume: bool = True,
    training_overrides: dict | None = None,
) -> dict:
    """Executa o fine-tuning QLoRA e devolve o resumo da execução.

    Salva em ``output_dir``: checkpoints intermediários, os adaptadores finais
    em ``final/``, uma cópia do ``metadata.json`` do dataset e um
    ``training_summary.json`` com hiperparâmetros, duração e pico de VRAM —
    os números que alimentam a tabela comparativa do TCC.
    """
    from trl import SFTConfig, SFTTrainer  # importado após o unsloth
    from datasets import Dataset

    _limit_gpu_memory()
    spec = get_model(model_key)
    if training_overrides:
        overrides = dict(training_overrides)
        if overrides.get("max_example_tokens") == 0:
            overrides["max_example_tokens"] = None  # 0 na linha de comando = sem limite
        spec = replace(spec, training=replace(spec.training, **overrides))
        print(f"Sobrescritas de treino: {training_overrides}")
    data_path, resolved_dataset_key = resolve_data_dir(dataset_key, data_dir)
    output_path = resolve_path(
        output_dir or f"models/{spec.key}/{resolved_dataset_key}"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(data_path)
    n_train = metadata["split_sizes"]["train"]
    cfg = spec.training
    total_epochs = epochs if epochs is not None else cfg.epochs

    print("=" * 62)
    print(f"Modelo:  {spec.label} ({spec.hf_id})")
    print(f"Dataset: {metadata.get('dataset_label', resolved_dataset_key)}")
    print(f"Splits:  {data_path}")
    print(f"Saída:   {output_path}")
    print(
        f"Treino:  {n_train} exemplos | {len(metadata['labels'])} classes | "
        f"batch efetivo {cfg.effective_batch} ({cfg.batch_size}×{cfg.grad_accum}) | "
        f"{total_epochs} épocas | lr {cfg.learning_rate}"
    )
    if spec.notes:
        print(f"Nota:    {spec.notes}")
    print("=" * 62)

    model, tokenizer = load_model(spec)
    model = attach_lora(model, spec)
    param_stats = trainable_parameter_stats(model)

    inner_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    inner_tokenizer.padding_side = "right"

    print("\nFormatando os splits no chat template do modelo...")

    def to_text(batch: dict) -> dict:
        return {
            "text": [
                apply_chat_template(tokenizer, msgs, spec, add_generation_prompt=False)
                for msgs in batch["messages"]
            ]
        }

    train_ds = Dataset.from_list(load_split(data_path, "train")).map(
        to_text, batched=True, remove_columns=["messages"]
    )
    val_ds = Dataset.from_list(load_split(data_path, "validation")).map(
        to_text, batched=True, remove_columns=["messages"]
    )

    n_dropped_long = 0
    if cfg.max_example_tokens:
        limit = cfg.max_example_tokens
        lengths = [
            len(ids)
            for ids in inner_tokenizer(list(train_ds["text"]), add_special_tokens=False)["input_ids"]
        ]
        keep = [i for i, n in enumerate(lengths) if n <= limit]
        n_dropped_long = len(lengths) - len(keep)
        train_ds = train_ds.select(keep)
        print(
            f"Exemplos de treino acima de {limit} tokens descartados: {n_dropped_long} "
            f"de {len(lengths)} ({100 * n_dropped_long / len(lengths):.3f}%)"
        )
        n_train = len(train_ds)

    # Estimativa inicial, só para configurar o SFTConfig — o empacotamento de
    # sequências (packing) pode reduzir o número real de linhas de treino
    # bem abaixo de n_train, então essa conta é corrigida mais abaixo usando
    # o dataset já empacotado pelo próprio SFTTrainer.
    total_steps_estimado = _total_steps(n_train, spec, total_epochs)
    warmup = _warmup_steps(total_steps_estimado, spec)
    save_steps = _save_steps(total_steps_estimado, spec)

    sft_config = SFTConfig(
        output_dir=str(output_path),
        run_name=run_name or f"{spec.key}-{resolved_dataset_key}-qlora",
        num_train_epochs=total_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size or cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler,
        warmup_steps=warmup,
        weight_decay=cfg.weight_decay,
        optim=cfg.optim,
        bf16=True,
        # Gradient checkpointing NÃO é ligado aqui: attach_lora() já configura
        # use_gradient_checkpointing="unsloth" via get_peft_model. Ligar de novo
        # no SFTConfig faz o Trainer aplicar o checkpointing padrão do HF por
        # cima do checkpointing próprio do Unsloth — os dois brigam pelo grafo
        # de autograd e a GPU passa a maior parte do tempo ociosa (~27% de uso
        # observado no Gemma 3 4B, ~68 s/passo quando devia ser bem mais rápido).
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=False,
        logging_steps=cfg.logging_steps,
        report_to=report_to,
        dataloader_num_workers=0,
        max_length=spec.max_seq_length,
        dataset_text_field="text",
        packing=cfg.packing,
        seed=spec.lora.random_state,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    # Corrige total_steps/warmup/save_steps com o tamanho real do dataset
    # DEPOIS do empacotamento (SFTTrainer empacota durante a construção acima).
    # Sem isso, um modelo cujo packing funciona (ex.: Gemma 3 1B, que não é
    # multimodal) pode ter bem menos passos reais do que a estimativa baseada
    # em n_train — no piloto isso caiu de 375 para 60, e com save_steps=32
    # calculado sobre 375 só existiria 1 checkpoint em todo o treino.
    n_train_real = len(trainer.train_dataset)
    if n_train_real != n_train:
        total_steps = _total_steps(n_train_real, spec, total_epochs)
        warmup = _warmup_steps(total_steps, spec)
        save_steps = _save_steps(total_steps, spec)
        trainer.args.warmup_steps = warmup
        trainer.args.save_steps = save_steps
        print(
            f"packing reduziu o dataset de {n_train} para {n_train_real} linhas "
            f"— recalculando passos totais/checkpoints com o valor real."
        )
    else:
        total_steps = total_steps_estimado

    print(f"passos totais: {total_steps}")
    print(f"warmup_steps: {warmup} (~{cfg.warmup_ratio:.0%} dos passos totais)")
    print(
        f"save_steps: {save_steps} "
        f"(~{math.floor(total_steps / save_steps)} checkpoints; "
        f"interrupção perde no máximo {save_steps} passos)"
    )

    resume_from = _find_latest_checkpoint(output_path) if resume else None
    if resume_from:
        print(f"\nCheckpoint encontrado — retomando de: {resume_from}")
    else:
        print("\nNenhum checkpoint — iniciando treino do zero.")

    started = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume_from)
    elapsed = time.time() - started

    final_dir = output_path / "final"
    print(f"\nSalvando adaptadores LoRA em {final_dir}")
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    shutil.copy(data_path / "metadata.json", final_dir / "metadata.json")

    summary = {
        "model_key": spec.key,
        "model_label": spec.label,
        "model_hf_id": spec.hf_id,
        "model_variant": spec.variant,
        "dataset_key": resolved_dataset_key,
        "dataset_label": metadata.get("dataset_label", resolved_dataset_key),
        "data_dir": str(data_path),
        "n_train": n_train,
        "n_train_dropped_long": n_dropped_long,
        "n_labels": len(metadata["labels"]),
        "epochs": total_epochs,
        "batch_size": cfg.batch_size,
        "grad_accum": cfg.grad_accum,
        "effective_batch": cfg.effective_batch,
        "learning_rate": cfg.learning_rate,
        "lr_scheduler": cfg.lr_scheduler,
        "total_steps": total_steps,
        "warmup_steps": warmup,
        "save_steps": save_steps,
        "optim": cfg.optim,
        "packing": cfg.packing,
        "max_seq_length": spec.max_seq_length,
        "lora_r": spec.lora.r,
        "lora_alpha": spec.lora.alpha,
        **param_stats,
        "train_runtime_seconds": round(elapsed, 1),
        "train_runtime_hours": round(elapsed / 3600, 2),
        **_loss_history(trainer.state.log_history, total_steps, total_epochs),
        "peak_vram_gb": _peak_vram_gb(),
        "adapters_dir": str(final_dir),
    }
    for path in (output_path / "training_summary.json", final_dir / "training_summary.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"\nTreino concluído em {summary['train_runtime_hours']}h "
        f"(loss final {summary['final_train_loss']}, "
        f"pico de VRAM {summary['peak_vram_gb']} GB)."
    )
    print(f"Resumo salvo em {output_path / 'training_summary.json'}")
    return summary

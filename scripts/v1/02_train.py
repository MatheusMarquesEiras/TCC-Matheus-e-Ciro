"""
Fine-tuning QLoRA do Qwen3.5-0.8B no MedPT para triagem médica.

Compatível com TRL >= 1.x (usa SFTConfig + processing_class).

Execução:
    python scripts/02_train.py --data_dir data/top10 --output_dir models/qwen35-top10
"""

import json
import argparse
import math
import shutil
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


MODEL_ID = "Qwen/Qwen3.5-0.8B"
BATCH_SIZE = 4
GRAD_ACCUM = 8       # batch efetivo = 32
EPOCHS = 3
WARMUP_RATIO = 0.03


def load_split_as_dataset(data_dir: Path, split: str) -> Dataset:
    with open(data_dir / f"{split}.json", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)


def apply_chat_template(examples: dict, tokenizer) -> dict:
    texts = [
        tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,  # non-thinking mode — obrigatório para 0.8B
        )
        for msgs in examples["messages"]
    ]
    return {"text": texts}


def compute_warmup_steps(n_train: int) -> int:
    steps_per_epoch = math.ceil(n_train / (BATCH_SIZE * GRAD_ACCUM))
    total_steps = steps_per_epoch * EPOCHS
    return max(1, round(total_steps * WARMUP_RATIO))


def find_latest_checkpoint(output_path: Path) -> str | None:
    checkpoints = sorted(
        [d for d in output_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[-1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


def main(data_dir: str, output_dir: str, run_name: str, report_to: str):
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(data_path / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    n_train = metadata["split_sizes"]["train"]
    print(f"Top-{metadata['top_n']} especialidades | Treino: {n_train} exemplos")

    # Quantização NF4 (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"\nCarregando tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Carregando modelo em 4-bit NF4: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA — aplicar em atenção e MLP para máxima cobertura de parâmetros
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Carregar e formatar datasets
    print("\nCarregando e formatando datasets...")
    train_ds = load_split_as_dataset(data_path, "train")
    val_ds = load_split_as_dataset(data_path, "validation")

    train_ds = train_ds.map(
        lambda ex: apply_chat_template(ex, tokenizer),
        batched=True,
        remove_columns=["messages"],
    )
    val_ds = val_ds.map(
        lambda ex: apply_chat_template(ex, tokenizer),
        batched=True,
        remove_columns=["messages"],
    )

    warmup_steps = compute_warmup_steps(n_train)
    print(f"warmup_steps calculado: {warmup_steps} (~{WARMUP_RATIO:.0%} dos steps totais)")

    # SFTConfig = TrainingArguments + parâmetros SFT (TRL >= 1.0)
    sft_config = SFTConfig(
        output_dir=str(output_path),
        run_name=run_name,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        optim="paged_adamw_8bit",
        bf16=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=500,            # salva a cada 500 steps, mesmo no meio da epoch
        save_total_limit=5,        # mantém os 5 checkpoints mais recentes
        load_best_model_at_end=False,
        logging_steps=50,
        report_to=report_to,
        dataloader_num_workers=0,
        # Parâmetros SFT-específicos (saíram do SFTTrainer no TRL 1.x)
        max_length=1024,           # era max_seq_length em TRL < 1.0
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,   # era 'tokenizer=' no TRL < 1.0
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    # Sempre busca o checkpoint mais recente automaticamente
    resume_from = find_latest_checkpoint(output_path)
    if resume_from:
        print(f"\nCheckpoint encontrado — retomando de: {resume_from}")
    else:
        print("\nNenhum checkpoint encontrado — iniciando treino do zero.")

    print("Iniciando treino QLoRA...")
    trainer.train(resume_from_checkpoint=resume_from)

    # Salvar adaptadores LoRA finais (não o modelo base — são poucos MB)
    final_dir = output_path / "final"
    print(f"\nSalvando adaptadores LoRA em {final_dir}")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    shutil.copy(data_path / "metadata.json", final_dir / "metadata.json")

    print("Treino concluído.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning QLoRA do Qwen3.5-0.8B no MedPT.")
    parser.add_argument("--data_dir", type=str, default="data/top10",
                        help="Diretório com os splits preparados pelo script 01")
    parser.add_argument("--output_dir", type=str, default="models/qwen35-medpt-top10",
                        help="Diretório para salvar checkpoints e modelo final")
    parser.add_argument("--run_name", type=str, default="qwen35-0.8b-medpt-qlora",
                        help="Nome da run (usado no WandB se ativado)")
    parser.add_argument("--report_to", type=str, default="none", choices=["none", "wandb", "tensorboard"],
                        help="Backend de logging. Padrão: none")
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.run_name, args.report_to)

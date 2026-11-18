"""
Avaliação intrínseca: zero-shot, 3-shot, 5-shot e fine-tuned.

Gera:
  - F1 macro, precision macro, recall macro, accuracy (por setup)
  - F1 por classe
  - Matriz de confusão (PNG)
  - Relatório de classificação (TXT)
  - Predições do modelo fine-tuned salvas para uso no script 04

Execução:
    # Todos os setups
    python scripts/03_evaluate.py --data_dir data/top10 --model_dir models/qwen35-medpt-top10/final

    # Só fine-tuned (mais rápido após já ter zero/few-shot)
    python scripts/03_evaluate.py --data_dir data/top10 --model_dir models/qwen35-medpt-top10/final --setups finetuned
"""

import json
import argparse
import random
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)


MODEL_ID = "Qwen/Qwen3.5-0.8B"
RANDOM_SEED = 42


def build_messages(
    question: str,
    specialties: list[str],
    few_shot_examples: list[dict] | None = None,
) -> list[dict]:
    system = (
        "Você é um sistema de triagem médica. Dada a queixa do paciente, "
        f"classifique a especialidade médica mais apropriada entre: {', '.join(sorted(specialties))}. "
        "Responda APENAS com o nome da especialidade, sem explicações adicionais."
    )
    messages = [{"role": "system", "content": system}]
    if few_shot_examples:
        for ex in few_shot_examples:
            messages.append({"role": "user", "content": ex["question"]})
            messages.append({"role": "assistant", "content": ex["specialty"]})
    messages.append({"role": "user", "content": question})
    return messages


def apply_template(tokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Fallback se o tokenizer não suportar enable_thinking
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # Forçar não-thinking inserindo bloco vazio de raciocínio
        return text + "<think>\n\n</think>\n\n"


def match_specialty(generated: str, specialties: list[str]) -> str:
    """Mapeia o texto gerado para a especialidade mais próxima na lista."""
    gen_lower = generated.strip().lower()
    # Correspondência exata primeiro
    for spec in specialties:
        if spec.lower() == gen_lower:
            return spec
    # Correspondência parcial
    for spec in specialties:
        if spec.lower() in gen_lower or gen_lower in spec.lower():
            return spec
    # Sem correspondência: retorna o texto bruto (será contado como erro)
    return generated.strip()


@torch.inference_mode()
def predict_single(model, tokenizer, messages: list[dict], specialties: list[str], device: str) -> str:
    text = apply_template(tokenizer, messages)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,          # greedy — classificação não precisa de amostragem
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return match_specialty(generated, specialties)


def load_model_and_tokenizer(setup: str, model_dir: str):
    if setup == "finetuned":
        print(f"  Carregando base + adaptadores LoRA de {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_dir)
    else:
        print(f"  Carregando modelo base: {MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    model.eval()
    return model, tokenizer


def compute_metrics(y_true: list, y_pred: list, labels: list) -> dict:
    return {
        "f1_macro": round(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
        "precision_macro": round(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_per_class": {
            label: round(score, 4)
            for label, score in zip(
                labels,
                f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
            )
        },
    }


def save_confusion_matrix(y_true, y_pred, labels, out_path: Path, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig_size = max(8, len(labels))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size - 1))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels,
                cmap="Blues", ax=ax)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Matriz de confusão: {out_path}")


def run_setup(
    setup: str,
    test_data: list,
    specialties: list[str],
    model_dir: str,
    output_path: Path,
    few_shot_pool: list[dict],
    device: str,
) -> tuple[dict, list[str]]:
    n_shot = {"zeroshot": 0, "threeshot": 3, "fiveshot": 5, "finetuned": 0}[setup]

    model, tokenizer = load_model_and_tokenizer(setup, model_dir)

    y_true, y_pred = [], []
    for i, item in enumerate(test_data):
        question = item["messages"][1]["content"]
        true_label = item["messages"][2]["content"]

        if n_shot > 0:
            candidates = [ex for ex in few_shot_pool if ex["specialty"] != true_label]
            examples = random.sample(candidates, min(n_shot, len(candidates)))
        else:
            examples = None

        messages = build_messages(question, specialties, examples)
        pred = predict_single(model, tokenizer, messages, specialties, device)

        y_true.append(true_label)
        y_pred.append(pred)

        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{len(test_data)}] acurácia parcial: {sum(t==p for t,p in zip(y_true,y_pred))/(i+1):.3f}")

    metrics = compute_metrics(y_true, y_pred, specialties)

    # Matriz de confusão
    save_confusion_matrix(
        y_true, y_pred, specialties,
        output_path / f"confusion_matrix_{setup}.png",
        title=f"Matriz de Confusão — {setup}",
    )

    # Relatório por classe
    report = classification_report(y_true, y_pred, labels=specialties, zero_division=0)
    with open(output_path / f"report_{setup}.txt", "w", encoding="utf-8") as f:
        f.write(f"Setup: {setup}\n\n{report}")

    del model
    torch.cuda.empty_cache()

    return metrics, y_pred


def main(data_dir: str, model_dir: str, output_dir: str, setups: list[str]):
    random.seed(RANDOM_SEED)
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(data_path / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    specialties = metadata["specialties"]

    with open(data_path / "test.json", encoding="utf-8") as f:
        test_data = json.load(f)

    # Pool de exemplos few-shot vem do treino (evita contaminação do teste)
    with open(data_path / "train.json", encoding="utf-8") as f:
        train_data = json.load(f)
    few_shot_pool = [
        {"question": d["messages"][1]["content"], "specialty": d["messages"][2]["content"]}
        for d in train_data
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")
    print(f"Exemplos de teste: {len(test_data)}")
    print(f"Setups a avaliar: {setups}\n")

    all_results = {}
    finetuned_predictions = None

    for setup in setups:
        print(f"{'='*55}")
        print(f"Setup: {setup}")
        metrics, y_pred = run_setup(
            setup, test_data, specialties, model_dir, output_path, few_shot_pool, device
        )
        all_results[setup] = metrics

        if setup == "finetuned":
            finetuned_predictions = y_pred

        print(f"  F1 macro:        {metrics['f1_macro']:.4f}")
        print(f"  Precision macro: {metrics['precision_macro']:.4f}")
        print(f"  Recall macro:    {metrics['recall_macro']:.4f}")
        print(f"  Accuracy:        {metrics['accuracy']:.4f}")

    # Tabela comparativa
    print(f"\n{'='*55}")
    print(f"{'Setup':<12} {'F1 macro':>10} {'Precision':>10} {'Recall':>10} {'Accuracy':>10}")
    print("-" * 55)
    for setup, m in all_results.items():
        print(f"{setup:<12} {m['f1_macro']:>10.4f} {m['precision_macro']:>10.4f} "
              f"{m['recall_macro']:>10.4f} {m['accuracy']:>10.4f}")

    # Salvar resultados
    with open(output_path / "results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResultados salvos em {output_path / 'results.json'}")

    # Salvar predições do fine-tuned para uso no script 04
    if finetuned_predictions is not None:
        pred_file = output_path / "finetuned_predictions.json"
        with open(pred_file, "w", encoding="utf-8") as f:
            json.dump(finetuned_predictions, f, ensure_ascii=False, indent=2)
        print(f"Predições fine-tuned salvas em {pred_file}")
        print("→ Use este arquivo como --predictions_file no script 04_qualitative_eval.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação intrínseca: zero-shot, few-shot e fine-tuned.")
    parser.add_argument("--data_dir", type=str, default="data/top10")
    parser.add_argument("--model_dir", type=str, default="models/qwen35-medpt-top10/final",
                        help="Diretório dos adaptadores LoRA (saída do script 02)")
    parser.add_argument("--output_dir", type=str, default="results/evaluation")
    parser.add_argument(
        "--setups", nargs="+",
        default=["zeroshot", "threeshot", "fiveshot", "finetuned"],
        choices=["zeroshot", "threeshot", "fiveshot", "finetuned"],
        help="Quais setups avaliar. Padrão: todos os quatro",
    )
    args = parser.parse_args()
    main(args.data_dir, args.model_dir, args.output_dir, args.setups)

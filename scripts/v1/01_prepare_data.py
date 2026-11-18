"""
Prepara o dataset AKCIT/MedPT para fine-tuning.

Execução:
    python scripts/01_prepare_data.py --top_n 10 --output_dir data/top10
    python scripts/01_prepare_data.py --top_n 10 --max_per_class 5000 --output_dir data/top10_balanced
"""

import json
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict

from datasets import load_dataset
from sklearn.model_selection import train_test_split


SYSTEM_PROMPT_TEMPLATE = (
    "Você é um sistema de triagem médica. Dada a queixa do paciente, "
    "classifique a especialidade médica mais apropriada entre: {specialties}. "
    "Responda APENAS com o nome da especialidade, sem explicações adicionais."
)


def format_chatml(question: str, specialty: str, specialties_list: list[str]) -> dict:
    system = SYSTEM_PROMPT_TEMPLATE.format(specialties=", ".join(sorted(specialties_list)))
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
            {"role": "assistant", "content": specialty},
        ]
    }


def extract_primary_specialty(raw: str) -> str:
    # O campo é sempre string. Casos raros têm múltiplos separados por ";".
    # NÃO dividir por "," pois nomes como "Ortopedista - traumatologista" usam hífen
    # que faz parte do título oficial da especialidade.
    return raw.split(";")[0].strip()


def cap_per_class(
    questions: tuple, specialties: tuple, max_per_class: int, seed: int = 42
) -> tuple[list, list]:
    """Limita cada classe a max_per_class exemplos, mantendo balanceamento."""
    rng = random.Random(seed)
    by_class: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(specialties):
        by_class[s].append(i)

    selected = []
    for spec, indices in by_class.items():
        chosen = rng.sample(indices, min(max_per_class, len(indices)))
        selected.extend(chosen)

    selected.sort()
    q_out = [questions[i] for i in selected]
    s_out = [specialties[i] for i in selected]
    return q_out, s_out


def main(top_n: int, output_dir: str, max_per_class: int | None):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Carregando AKCIT/MedPT do HuggingFace...")
    ds = load_dataset("AKCIT/MedPT", split="train")
    print(f"Total de exemplos brutos: {len(ds)}")

    # Colunas reais do MedPT: "question" e "medical_specialty" (snake_case)
    questions = list(ds["question"])
    raw_specialties = list(ds["medical_specialty"])
    specialties_clean = [extract_primary_specialty(s) for s in raw_specialties]

    # Remover exemplos com especialidade vazia
    pairs = [(q, s) for q, s in zip(questions, specialties_clean) if s]
    questions, specialties_clean = zip(*pairs)

    # Selecionar Top-N especialidades por frequência
    counter = Counter(specialties_clean)
    top_specialties = [spec for spec, _ in counter.most_common(top_n)]

    print(f"\nTop-{top_n} especialidades selecionadas:")
    for i, spec in enumerate(top_specialties, 1):
        print(f"  {i:2d}. {spec}: {counter[spec]} exemplos")

    # Filtrar apenas amostras das Top-N especialidades
    pairs_filtered = [
        (q, s) for q, s in zip(questions, specialties_clean) if s in top_specialties
    ]
    questions_f, specialties_f = zip(*pairs_filtered)
    print(f"\nTotal após filtragem: {len(questions_f)} exemplos")

    # Limitar por classe se solicitado
    if max_per_class is not None:
        questions_f, specialties_f = cap_per_class(questions_f, specialties_f, max_per_class)
        print(f"Após cap de {max_per_class} por classe: {len(questions_f)} exemplos")
        dist = Counter(specialties_f)
        for spec in top_specialties:
            print(f"  {spec}: {dist[spec]}")

    # Split estratificado: 80% treino / 10% validação / 10% teste
    q_train, q_temp, s_train, s_temp = train_test_split(
        questions_f, specialties_f,
        test_size=0.2,
        stratify=specialties_f,
        random_state=42,
    )
    q_val, q_test, s_val, s_test = train_test_split(
        q_temp, s_temp,
        test_size=0.5,
        stratify=s_temp,
        random_state=42,
    )

    # Estimar tempo de treino (fallback PyTorch sem flash-linear-attention)
    steps_per_epoch = len(q_train) // 32
    total_steps = steps_per_epoch * 3
    estimated_hours = (total_steps * 7.5) / 3600
    print(f"\nSplit resultante:")
    print(f"  Treino:    {len(q_train)} exemplos  (~{steps_per_epoch} steps/epoch, {total_steps} total)")
    print(f"  Validação: {len(q_val)} exemplos")
    print(f"  Teste:     {len(q_test)} exemplos")
    print(f"  Tempo estimado (7.5s/step): ~{estimated_hours:.1f}h  ({estimated_hours/2:.1f} sessões de 2h)")

    # Formatar no padrão ChatML e salvar
    for split_name, (qs, ss) in [
        ("train", (q_train, s_train)),
        ("validation", (q_val, s_val)),
        ("test", (q_test, s_test)),
    ]:
        data = [format_chatml(q, s, top_specialties) for q, s in zip(qs, ss)]
        out_file = output_path / f"{split_name}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  Salvo: {out_file}")

    # Salvar metadados
    label2id = {spec: i for i, spec in enumerate(sorted(top_specialties))}
    metadata = {
        "top_n": top_n,
        "max_per_class": max_per_class,
        "specialties": top_specialties,
        "label2id": label2id,
        "id2label": {str(i): spec for spec, i in label2id.items()},
        "split_sizes": {
            "train": len(q_train),
            "validation": len(q_val),
            "test": len(q_test),
        },
        "class_distribution": {
            split_name: dict(Counter(ss))
            for split_name, ss in [
                ("train", s_train),
                ("validation", s_val),
                ("test", s_test),
            ]
        },
    }
    meta_file = output_path / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  Salvo: {meta_file}")
    print("\nPré-processamento concluído.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara o MedPT para fine-tuning.")
    parser.add_argument(
        "--top_n", type=int, default=10, choices=[5, 10, 20],
        help="Número de especialidades a incluir. Padrão: 10",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/top10",
        help="Diretório de saída para os splits e metadados",
    )
    parser.add_argument(
        "--max_per_class", type=int, default=None,
        help="Limite máximo de exemplos por especialidade (balanceia e reduz tempo de treino). "
             "Ex: 5000 → ~40k treino, ~8h. Sem limite: usa tudo (~145k treino, ~28h).",
    )
    args = parser.parse_args()
    main(args.top_n, args.output_dir, args.max_per_class)

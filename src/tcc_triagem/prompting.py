"""Montagem dos prompts e do formato ChatML.

O texto do system prompt é idêntico ao usado nos Experimentos 01 e 02 — mudá-lo
inviabiliza a comparação direta com os resultados já publicados no TCC. Ele fica
aqui, em um único lugar, e é parametrizado pelo conjunto de rótulos do dataset.
"""

import unicodedata

DEFAULT_SYSTEM_TEMPLATE = (
    "Você é um sistema de triagem médica. Dada a queixa do paciente, "
    "classifique a especialidade médica mais apropriada entre: {labels}. "
    "Responda APENAS com o nome da especialidade, sem explicações adicionais."
)


def build_system_prompt(labels: list[str], template: str = DEFAULT_SYSTEM_TEMPLATE) -> str:
    """Monta o system prompt com os rótulos em ordem alfabética.

    A ordenação é fixa de propósito: o mesmo dataset sempre gera o mesmo prompt,
    independente da ordem em que os rótulos foram descobertos.
    """
    return template.format(labels=", ".join(sorted(labels)))


def to_chatml(
    question: str,
    label: str,
    labels: list[str],
    template: str = DEFAULT_SYSTEM_TEMPLATE,
) -> dict:
    """Formata um par (queixa, especialidade) como conversa ChatML de 3 turnos."""
    return {
        "messages": [
            {"role": "system", "content": build_system_prompt(labels, template)},
            {"role": "user", "content": question},
            {"role": "assistant", "content": label},
        ]
    }


def build_inference_messages(
    question: str,
    labels: list[str],
    few_shot: list[dict] | None = None,
    template: str = DEFAULT_SYSTEM_TEMPLATE,
) -> list[dict]:
    """Monta as mensagens de inferência, com exemplos few-shot opcionais.

    ``few_shot`` é uma lista de dicts ``{"question": ..., "label": ...}``.
    """
    messages = [{"role": "system", "content": build_system_prompt(labels, template)}]
    for example in few_shot or []:
        messages.append({"role": "user", "content": example["question"]})
        messages.append({"role": "assistant", "content": example["label"]})
    messages.append({"role": "user", "content": question})
    return messages


def normalize_label(text: str) -> str:
    """Normaliza um rótulo para comparação: sem acento, minúsculo, sem espaço extra."""
    stripped = unicodedata.normalize("NFKD", text.strip().lower())
    stripped = "".join(c for c in stripped if not unicodedata.combining(c))
    return " ".join(stripped.split())


def match_label(generated: str, labels: list[str]) -> str:
    """Mapeia o texto gerado pelo modelo para um dos rótulos válidos.

    Tenta, em ordem: igualdade exata normalizada, rótulo contido na geração e
    geração contida no rótulo. Sem casamento, devolve o texto cru — o exemplo
    conta como erro nas métricas, que é o comportamento correto (não queremos
    esconder alucinação de formato).
    """
    generated_norm = normalize_label(generated)
    if not generated_norm:
        return generated.strip()

    normalized = {label: normalize_label(label) for label in labels}

    for label, norm in normalized.items():
        if norm == generated_norm:
            return label
    # Rótulos mais longos primeiro: "Psicólogo, Psicanalista" antes de "Psicólogo",
    # senão o prefixo mais curto sequestra o casamento.
    for label, norm in sorted(normalized.items(), key=lambda kv: -len(kv[1])):
        if norm in generated_norm or generated_norm in norm:
            return label
    return generated.strip()

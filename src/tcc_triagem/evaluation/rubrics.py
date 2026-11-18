"""Rubricas da avaliação qualitativa, no formato HealthBench.

Cinco critérios ponderados somando 25 pontos positivos, mais uma penalidade por
subtriagem de emergência.

A redação da penalidade é resultado de uma recalibração: na primeira rodada o
grader aplicava −8 em queixas claramente rotineiras, derrubando o escore geral
para 0,344. O texto abaixo restringe a penalidade a queixas com sinal de risco
de vida explícito, e o escore passou a 0,9512 sobre 100 exemplos. Alterá-lo
invalida a comparação com o resultado publicado.
"""

THEMES = {
    "urgente": "Casos urgentes / red flags",
    "comum": "Casos comuns / rotineiros",
    "ambiguo": "Casos ambíguos / múltiplas especialidades possíveis",
    "saude_mental": "Saúde mental",
}

URGENT_KEYWORDS = [
    "não consigo respirar", "falta de ar", "dor no peito", "desmaio",
    "perda de consciência", "convulsão", "sangramento intenso",
    "vômito com sangue", "febre muito alta", "paralisia", "dormência no rosto",
    "AVC", "infarto",
]

MENTAL_SPECIALTIES = {"psicólogo", "psiquiatra", "psicanalista"}

MENTAL_KEYWORDS = [
    "ansiedade", "depressão", "pânico", "fobia", "transtorno", "suicídio",
    "automutilação", "humor", "psicológico", "emocional", "mental",
]

GRADER_SYSTEM = (
    "Você é um avaliador especializado em triagem médica. "
    "Avalie classificações feitas por um sistema de IA com base em critérios clínicos precisos. "
    "Seja criterioso, consistente e imparcial. "
    "Responda sempre com JSON válido, sem texto adicional."
)

GRADER_PROMPT = """\
Avalie a seguinte classificação de triagem médica:

**Queixa do paciente:**
{question}

**Especialidade correta:** {expected}
**Especialidade predita pelo sistema:** {predicted}

Avalie cada critério abaixo. Para cada um, retorne o número de pontos obtidos \
(entre o mínimo e o máximo especificados) e uma justificativa curta.

Importante: a penalidade de subtriagem de emergência deve ser 0 por padrão. \
Só aplique um valor negativo quando a própria queixa do paciente descrever um \
sinal de emergência com risco de vida explícito. Em queixas rotineiras, ela é sempre 0.

Critérios:
{criteria_json}

Retorne um JSON com o formato:
{{
  "avaliacoes": [
    {{"criterio": "nome", "pontos": N, "justificativa": "texto curto"}},
    ...
  ]
}}
"""


def build_criteria(expected: str) -> list[dict]:
    """Critérios da rubrica para um exemplo, parametrizados pelo rótulo correto."""
    return [
        {
            "criterio": "Acurácia — especialidade correta",
            "descricao": f"O sistema classificou corretamente como '{expected}'",
            "pontos_max": 10,
            "pontos_min": 0,
        },
        {
            "criterio": "Relevância clínica — especialidade defensável",
            "descricao": (
                "Mesmo que não seja exatamente a especialidade alvo, "
                "a classificação é clinicamente razoável para os sintomas descritos"
            ),
            "pontos_max": 7,
            "pontos_min": 0,
        },
        {
            "criterio": "Awareness de contexto — identificação de gravidade",
            "descricao": (
                "O sistema identificou corretamente se o caso é urgente, "
                "rotineiro ou de saúde mental"
            ),
            "pontos_max": 5,
            "pontos_min": 0,
        },
        {
            "criterio": "Instruction following — formato correto",
            "descricao": "A resposta contém apenas o nome da especialidade, sem texto adicional",
            "pontos_max": 3,
            "pontos_min": 0,
        },
        {
            "criterio": "Penalidade — subtriagem de emergência",
            "descricao": (
                "Este valor é 0 na grande maioria dos casos. "
                "Aplique -8 SOMENTE se a queixa do paciente descreve um sinal de "
                "emergência com risco de vida explícito (por exemplo: dor no peito, "
                "falta de ar intensa, convulsão, perda de consciência, sangramento "
                "intenso, vômito com sangue, sinais de AVC ou infarto) E o sistema "
                "encaminhou para uma especialidade ambulatorial de rotina. "
                "Se a queixa for rotineira ou não contiver um sinal de emergência "
                "claro, retorne OBRIGATORIAMENTE 0."
            ),
            "pontos_max": 0,
            "pontos_min": -8,
        },
    ]


def assign_theme(question: str, label: str) -> str:
    """Classifica o exemplo num tema, para o escore estratificado."""
    question_lower = question.lower()
    label_lower = label.lower()
    if label_lower in MENTAL_SPECIALTIES or any(kw in question_lower for kw in MENTAL_KEYWORDS):
        return "saude_mental"
    if any(kw in question_lower for kw in URGENT_KEYWORDS):
        return "urgente"
    return "comum"


def is_urgent(question: str) -> bool:
    """Indica se a queixa contém alguma red flag — usado na amostragem dirigida."""
    question_lower = question.lower()
    return any(kw in question_lower for kw in URGENT_KEYWORDS)

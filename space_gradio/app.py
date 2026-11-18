"""Demo pública do classificador de triagem médica (Hugging Face Space).

Versão autocontida do ``scripts/05_demo.py`` para rodar no hardware gratuito
dos Spaces (CPU, sem GPU): sem Unsloth e sem quantização 4-bit, só
``transformers`` + ``peft``. O modelo base é carregado uma única vez e os dois
adaptadores LoRA ficam anexados a ele, alternando por ``set_adapter`` — assim
os três modos (completo, piloto e base sem ajuste) cabem na memória de um só
modelo de 0,8B.

O prompt, o chat template e o casamento do texto gerado com um rótulo válido
reproduzem exatamente ``tcc_triagem.prompting`` e ``tcc_triagem.inference``:
qualquer diferença aqui tornaria a demo incomparável com os resultados do TCC.
"""

import os
import time
import unicodedata

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

BASE_ID = "Qwen/Qwen3.5-0.8B"

#: Chave do adaptador → (repositório, rótulo exibido na interface).
ADAPTERS = {
    "completo": (
        "matheus-ciro-tcc/qwen3.5-0.8b-triagem-medica-ptbr",
        "Ajustado — dataset completo (145 mil exemplos)",
    ),
    "piloto": (
        "matheus-ciro-tcc/qwen35-0.8b-triagem-medica-piloto",
        "Ajustado — piloto (4 mil exemplos)",
    ),
}
BASE_KEY = "base"
BASE_LABEL = "Modelo base, sem ajuste fino"
CHOICES = {label: key for key, (_, label) in ADAPTERS.items()} | {BASE_LABEL: BASE_KEY}

LABELS = [
    "Dentista",
    "Dermatologista",
    "Ginecologista",
    "Oftalmologista",
    "Ortopedista - traumatologista",
    "Otorrino",
    "Psicólogo",
    "Psicólogo, Psicanalista",
    "Psiquiatra",
    "Urologista",
]

# Idêntico a tcc_triagem.prompting.DEFAULT_SYSTEM_TEMPLATE.
SYSTEM_TEMPLATE = (
    "Você é um sistema de triagem médica. Dada a queixa do paciente, "
    "classifique a especialidade médica mais apropriada entre: {labels}. "
    "Responda APENAS com o nome da especialidade, sem explicações adicionais."
)
SYSTEM_PROMPT = SYSTEM_TEMPLATE.format(labels=", ".join(sorted(LABELS)))

#: Rótulos são curtos; 20 tokens bastam e evitam divagação do modelo.
MAX_NEW_TOKENS = 20

EXEMPLOS = [
    "Tenho sentido muita tristeza há meses, não consigo dormir e perdi o "
    "interesse em tudo que gostava.",
    "Meu joelho direito está inchado e doendo muito depois que caí da escada "
    "há três dias.",
    "Aparecem manchas vermelhas com coceira na minha pele toda vez que como "
    "frutos do mar.",
    "Estou com dificuldade para enxergar de longe e minha visão ficou embaçada "
    "nas últimas semanas.",
    "Sinto dor ao urinar há quatro dias e a urina está com cor estranha.",
    "Meu dente do fundo dói muito quando tomo algo gelado e a gengiva está "
    "sangrando.",
    "Estou com o ouvido tampado e zumbindo desde a última gripe.",
    "Minha menstruação está atrasada e sinto cólicas fortes fora do período.",
]


def normalize_label(text: str) -> str:
    """Sem acento, minúsculo, sem espaço extra — igual a tcc_triagem.prompting."""
    stripped = unicodedata.normalize("NFKD", text.strip().lower())
    stripped = "".join(c for c in stripped if not unicodedata.combining(c))
    return " ".join(stripped.split())


def match_label(generated: str, labels: list[str]) -> str:
    """Mapeia o texto gerado para um rótulo válido (mesma regra da avaliação)."""
    generated_norm = normalize_label(generated)
    if not generated_norm:
        return generated.strip()
    normalized = {label: normalize_label(label) for label in labels}
    for label, norm in normalized.items():
        if norm == generated_norm:
            return label
    # Rótulos mais longos primeiro: "Psicólogo, Psicanalista" antes de "Psicólogo".
    for label, norm in sorted(normalized.items(), key=lambda kv: -len(kv[1])):
        if norm in generated_norm or generated_norm in norm:
            return label
    return generated.strip()


torch.set_num_threads(os.cpu_count() or 1)

print(f"Carregando {BASE_ID}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
model = Qwen3_5ForConditionalGeneration.from_pretrained(BASE_ID, dtype=torch.float32)
first, *rest = ADAPTERS
model = PeftModel.from_pretrained(model, ADAPTERS[first][0], adapter_name=first)
for key in rest:
    model.load_adapter(ADAPTERS[key][0], adapter_name=key)
model.eval()
print("Modelo e adaptadores carregados.")


@torch.inference_mode()
def classify(queixa: str, key: str) -> tuple[str, str, float]:
    """Devolve (especialidade, texto cru gerado, segundos)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": queixa},
    ]
    # enable_thinking=False é obrigatório: em thinking mode o 0.8B entra em loop.
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt")
    generate = dict(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    started = time.perf_counter()
    if key == BASE_KEY:
        with model.disable_adapter():
            outputs = model.generate(**generate)
    else:
        model.set_adapter(key)
        outputs = model.generate(**generate)
    elapsed = time.perf_counter() - started

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    return match_label(generated, LABELS), generated, elapsed


def _check(queixa: str) -> str:
    queixa = (queixa or "").strip()
    if not queixa:
        raise gr.Error("Digite a queixa do paciente.")
    return queixa


def triar(queixa: str, modelo: str):
    especialidade, gerado, segundos = classify(_check(queixa), CHOICES[modelo])
    fora = "" if especialidade in LABELS else " — ⚠️ fora das 10 especialidades"
    detalhes = (
        f"**Texto gerado pelo modelo:** `{gerado or '(vazio)'}`{fora}  \n"
        f"**Tempo de inferência:** {segundos:.1f} s em CPU"
    )
    return especialidade, detalhes


def comparar(queixa: str):
    queixa = _check(queixa)
    linhas = []
    for label, key in CHOICES.items():
        especialidade, gerado, segundos = classify(queixa, key)
        linhas.append([label, especialidade, gerado, round(segundos, 1)])
    return linhas


with gr.Blocks(title="Triagem médica automatizada") as demo:
    gr.Markdown(
        """
# 🩺 Triagem médica automatizada em português

Descreva a queixa de um paciente em texto livre e o modelo indica a qual
**especialidade médica** encaminhá-lo, entre 10 opções. É o
[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) ajustado com
**QLoRA** sobre o corpus [MedPT](https://huggingface.co/datasets/AKCIT/MedPT).

TCC de **Matheus Marques Eiras** e **Ciro Guilherme Nass** (IFPR, Câmpus
Pinhais), orientado por **João Paulo Orlando**.

> ⚠️ **Demonstração acadêmica, não é um dispositivo médico.** O modelo sugere
> um encaminhamento a partir do texto; não diagnostica nem avalia gravidade.
> Em caso de emergência, procure atendimento imediatamente.
"""
    )

    with gr.Row():
        with gr.Column(scale=3):
            queixa = gr.Textbox(
                label="Queixa do paciente",
                placeholder="Ex.: estou com dor de dente e sensibilidade ao frio há uma semana.",
                lines=4,
            )
            modelo = gr.Radio(
                choices=list(CHOICES),
                value=ADAPTERS["completo"][1],
                label="Modelo",
            )
            with gr.Row():
                botao = gr.Button("Classificar", variant="primary")
                botao_comparar = gr.Button("Comparar os 3 modelos")
        with gr.Column(scale=2):
            especialidade = gr.Textbox(label="Especialidade recomendada", interactive=False)
            detalhes = gr.Markdown()

    tabela = gr.Dataframe(
        headers=["Modelo", "Especialidade", "Texto gerado", "Tempo (s)"],
        label="Comparação lado a lado",
        interactive=False,
        wrap=True,
    )

    gr.Examples(examples=[[e] for e in EXEMPLOS], inputs=[queixa], label="Exemplos")

    gr.Markdown(
        "**Especialidades:** " + " · ".join(f"`{label}`" for label in LABELS) + "\n\n"
        "Roda em CPU gratuita do Hugging Face: cada classificação leva alguns "
        "segundos, e a comparação roda os três modelos em sequência."
    )

    # Um pedido por vez: os três modos compartilham o mesmo modelo, e
    # set_adapter/disable_adapter mudam estado global dele.
    botao.click(triar, [queixa, modelo], [especialidade, detalhes], concurrency_limit=1)
    queixa.submit(triar, [queixa, modelo], [especialidade, detalhes], concurrency_limit=1)
    botao_comparar.click(comparar, [queixa], [tabela], concurrency_limit=1)

if __name__ == "__main__":
    demo.queue().launch()

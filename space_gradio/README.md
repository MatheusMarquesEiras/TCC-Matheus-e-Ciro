---
title: Triagem Médica PT-BR
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Roteamento de especialidade médica com Qwen3.5-0.8B
models:
- Qwen/Qwen3.5-0.8B
- matheus-ciro-tcc/qwen3.5-0.8b-triagem-medica-ptbr
- matheus-ciro-tcc/qwen35-0.8b-triagem-medica-piloto
datasets:
- AKCIT/MedPT
---

# Triagem médica automatizada em português brasileiro

Demo do TCC de Matheus Marques Eiras e Ciro Guilherme Nass (IFPR, Câmpus
Pinhais), orientado por João Paulo Orlando. A partir da queixa de um paciente
em texto livre, o modelo sugere a especialidade médica mais adequada entre 10.

| Modo | Adaptador |
|---|---|
| Ajustado, dataset completo (145 mil exemplos) | [`matheus-ciro-tcc/qwen3.5-0.8b-triagem-medica-ptbr`](https://huggingface.co/matheus-ciro-tcc/qwen3.5-0.8b-triagem-medica-ptbr) |
| Ajustado, piloto (4 mil exemplos) | [`matheus-ciro-tcc/qwen35-0.8b-triagem-medica-piloto`](https://huggingface.co/matheus-ciro-tcc/qwen35-0.8b-triagem-medica-piloto) |
| Base, sem ajuste | [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) |

Os três modos compartilham um único modelo base carregado em CPU; os adaptadores
LoRA são trocados em tempo de execução. O prompt e o pós-processamento da
resposta são os mesmos usados na avaliação do TCC.

**Não é um dispositivo médico.** Uso exclusivamente acadêmico e demonstrativo.

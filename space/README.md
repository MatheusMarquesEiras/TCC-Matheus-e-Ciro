---
title: Triagem Médica PT-BR
emoji: 🩺
colorFrom: green
colorTo: blue
sdk: static
pinned: false
license: apache-2.0
short_description: Triagem de especialidade médica rodando no navegador
models:
- matheus-ciro-tcc/triagem-medica-gguf-final
- matheus-ciro-tcc/qwen35-2b-triagem-medica-final
- matheus-ciro-tcc/gemma3-4b-triagem-medica-final
- matheus-ciro-tcc/qwen35-0.8b-triagem-medica-final
- matheus-ciro-tcc/gemma3-1b-triagem-medica-final
datasets:
- AKCIT/MedPT
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

# Triagem médica automatizada em português brasileiro

Demo do TCC de Matheus Marques Eiras e Ciro Guilherme Nass (IFPR, Câmpus
Pinhais), orientado por João Paulo Orlando. A partir da queixa de um paciente em
texto livre, o modelo sugere a especialidade médica mais adequada entre 9.

Dá para escolher entre os quatro modelos finais do TCC, todos treinados com 145 mil
queixas do MedPT e avaliados em 18.166 queixas de teste:

| Modelo | Download | Acurácia | F1-macro |
|---|---:|---:|---:|
| Qwen3.5-2B (padrão) | 2,0 GB | 91,1% | 0,898 |
| Gemma 3 4B | 2,4 GB | 91,4% | 0,901 |
| Qwen3.5-0.8B | 0,8 GB | 89,2% | 0,881 |
| Gemma 3 1B | 1,0 GB | 89,1% | 0,863 |

O link pode apontar para um modelo: `?modelo=qwen35-2b`, `gemma3-4b`, `qwen35-0.8b`
ou `gemma3-1b`.

Os modelos rodam **inteiros no navegador do visitante** com
[wllama](https://github.com/ngxson/wllama) (llama.cpp em WebAssembly, com WebGPU
quando disponível): nada é enviado a servidor. Na primeira vez o modelo é baixado e
fica no cache do navegador. Os GGUF estão em
[`matheus-ciro-tcc/triagem-medica-gguf-final`](https://huggingface.co/matheus-ciro-tcc/triagem-medica-gguf-final);
prompt e pós-processamento são os mesmos da avaliação do TCC, e as respostas daqui
coincidem com as da avaliação em 97,5% a 100% das 200 primeiras queixas do teste.

**Não é um dispositivo médico.** Uso exclusivamente acadêmico e demonstrativo.

`wllama/` contém o build do [@wllama/wllama](https://www.npmjs.com/package/@wllama/wllama)
3.6.1 (licença MIT, ver `wllama/LICENCE`).

# Experimento 04 — Modelo final Qwen3.5-2B (dataset completo) avaliado com GLM 5.3

**Data:** 24/10/2026
**Modelo:** `Qwen/Qwen3.5-2B` com QLoRA (4-bit NF4, LoRA r=16, α=32), adaptadores em `models/qwen35-2b/medpt-top10-merged/final/`
**Dataset:** `medpt-top10-merged` — MedPT, 9 classes (Psicólogo e "Psicólogo, Psicanalista" fundidos), 145.321 treino / 18.165 validação / **18.166 teste**
**Juiz:** `glm-5.3-flash` (Ollama Cloud), 100 exemplos estratificados

Este é o **modelo final** do estudo: treinado no dataset completo, e não no piloto de 500 exemplos
por classe do Experimento 03. Os números não são comparáveis diretamente com os do piloto.

## Treino

| Item | Valor |
|---|---|
| Passos | 13.626 (3 épocas, batch efetivo 32 = 2 × 16) |
| Taxa de aprendizado | 2e-4, agendador cosseno, 409 passos de aquecimento |
| Parâmetros treináveis | 10,9 M de 1,39 B (0,79%) |
| Perda de treino (média por época) | 0,719 → 0,491 → 0,393 |
| Perda de validação | 0,558 (época 1) → 0,473 (época 2) → **0,453** (época 3) |
| Pico de VRAM | 9,33 GB (RTX 4070 12 GB) |
| Tempo | ~13 h no total (13.626 passos a ~3,5 s por passo), em vários trechos |

O treino foi pausado e retomado de checkpoints mais de uma vez (500 em 500 passos). O campo
`train_runtime_hours` do `training_summary.json` (3,24 h) cobre só o último trecho, depois da
retomada do passo 10.500, e **não** é o tempo total; o `final_train_loss` (0,089) também é distorcido pela retomada (a soma do último trecho é dividida pelo total de passos), por isso as perdas de treino acima vêm do histórico do `trainer_state.json`, por época.

## Resultado quantitativo (teste completo, 18.166 exemplos)

| Métrica | Valor |
|---|---:|
| Acurácia | **91,14%** |
| F1 macro | **0,898** |
| Precisão macro | 0,899 |
| Recall macro | 0,897 |

### Por classe

| Classe | Precisão | Recall | F1 | Suporte |
|---|---:|---:|---:|---:|
| Ortopedista - traumatologista | 0,97 | 0,97 | 0,974 | 1.482 |
| Oftalmologista | 0,97 | 0,97 | 0,973 | 1.211 |
| Dentista | 0,94 | 0,95 | 0,944 | 1.114 |
| Psicólogo | 0,93 | 0,93 | 0,933 | 6.686 |
| Dermatologista | 0,92 | 0,93 | 0,926 | 1.505 |
| Otorrino | 0,92 | 0,91 | 0,918 | 773 |
| Ginecologista | 0,89 | 0,93 | 0,911 | 2.662 |
| Urologista | 0,90 | 0,85 | 0,872 | 1.618 |
| **Psiquiatra** | 0,64 | 0,63 | **0,631** | 1.115 |

### Onde o modelo erra

Dos 1.609 erros, **750 (47%) são a confusão Psiquiatra ↔ Psicólogo**:

| Real → previsto | Erros | % da classe real |
|---|---:|---:|
| Psiquiatra → Psicólogo | 379 | 34,0% |
| Psicólogo → Psiquiatra | 371 | 5,5% |
| Urologista → Ginecologista | 199 | 12,3% |
| Ginecologista → Urologista | 83 | 3,1% |
| Dermatologista → Ginecologista | 42 | 2,8% |

A fusão de Psicólogo com "Psicólogo, Psicanalista" resolveu a incompatibilidade entre esses
dois rótulos, mas **não** a fronteira entre Psicólogo e Psiquiatra, que continua sendo o ponto
fraco, como já aparecia nos pilotos. Urologista × Ginecologista é a segunda confusão, e é
plausível clinicamente (queixas do trato urinário e genital).

## Avaliação qualitativa com o GLM 5.3 (100 exemplos)

Rubrica estilo HealthBench (5 critérios, notas de 0 a 1). Nenhuma resposta do juiz ficou ilegível.

| Métrica | Valor |
|---|---:|
| **Nota geral** | **0,967** |
| Acurácia na amostra | 94% (100 exemplos) |
| Nota média quando o modelo acerta | 0,995 |
| Nota média quando o modelo erra | 0,527 |

| Tema | Exemplos | Nota |
|---|---:|---:|
| Casos comuns / rotineiros | 67 | 0,967 |
| Saúde mental | 30 | 0,964 |
| Urgentes (sem sobreposição com saúde mental) | 3 | 1,000 |

### Casos com sinais de alerta

A amostra reserva 15 exemplos com palavras-chave de alerta (cota `--min_urgent 15`). O tema "urgente"
só conta os que não caem antes em "saúde mental", por isso aparecem 3 na tabela acima; os outros 12
estão em saúde mental. Olhando os 15 juntos:

- **15 de 15 classificados corretamente**, nota média **0,987**.
- Nenhum caso de alerta foi encaminhado para a especialidade errada.

A amostra é pequena (15), então isto indica ausência de falhas graves nessa amostra, não uma
garantia de segurança. Medir isso com rigor exigiria avaliar todos os 294 exemplos de teste com
palavras-chave de alerta.

### Os 6 erros da amostra

| Real → previsto | Nota do juiz |
|---|---:|
| Otorrino → Dentista | 0,56 |
| Otorrino → Dentista | 0,52 |
| Urologista → Ginecologista | 0,56 |
| Psiquiatra → Psicólogo | 0,64 |
| Psiquiatra → Psicólogo | 0,56 |
| Dermatologista → Ortopedista - traumatologista | 0,32 |

## Comparação com os experimentos anteriores

| Experimento | Modelo | Dataset | Teste | Acurácia | F1 macro | Nota GLM |
|---|---|---|---|---:|---:|---:|
| 03 (piloto) | Qwen3.5-2B | piloto, 10 classes | 500 | 76,2% | 0,754 | 0,919 |
| 03 (piloto) | Gemma 3 4B | piloto, 10 classes | 500 | 75,8% | 0,758 | 0,909 |
| **04 (final)** | **Qwen3.5-2B** | **completo, 9 classes** | **18.166** | **91,1%** | **0,898** | **0,967** |

O salto de ~76% para ~91% vem de treinar com ~36× mais dados (4.000 → 145.321 exemplos), mas
**não é uma comparação limpa**: o conjunto de teste muda (500 → 18.166), o número de classes
muda (10 → 9, com a fusão) e a classe dominante (Psicólogo, 37% do teste) pesa mais no dataset
completo.

## Limitações

- **Sem linha de base zero-shot no dataset completo.** O ganho do ajuste fino é medido nos pilotos
  (Experimento 03: o 2B base acerta 49,8% e o ajustado 76,2%); aqui só o modelo ajustado foi avaliado.
- **Nota do GLM com 100 exemplos:** margem de erro grande por tema (3 a 67 exemplos por tema).
- **Um único modelo juiz** (GLM 5.3). Não houve validação cruzada com outro juiz nem com revisão médica.
- **Desbalanceamento:** Psicólogo concentra 37% do teste; a acurácia global é puxada por ele, e o
  F1 macro (0,898) é a métrica mais honesta para o conjunto.
- **Avaliação com `max_new_tokens=20`** e correspondência do texto gerado com o rótulo mais próximo.

## Artefatos

- `results/qwen35-2b/medpt-top10-merged/evaluation/` — `results.json`, `report_finetuned.txt`,
  `confusion_matrix_finetuned.png`, `finetuned_predictions.json` (18.166 predições)
- `results/qwen35-2b/medpt-top10-merged/qualitative_glm/` — `qualitative_results.json` (notas e
  justificativas por critério), `error_analysis.json`
- `models/qwen35-2b/medpt-top10-merged/` — adaptadores (`final/`) e `training_summary.json`
- `scripts/run_final_eval.sh` — reproduz a avaliação (pausável com `scripts/pausar.sh`, retoma sozinha)

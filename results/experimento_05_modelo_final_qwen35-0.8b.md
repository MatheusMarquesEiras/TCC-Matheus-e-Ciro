# Experimento 05 — Modelo final Qwen3.5-0.8B (dataset completo) avaliado com GLM 5.3

**Data:** 26/10/2026
**Modelo:** `Qwen/Qwen3.5-0.8B` com QLoRA (4-bit NF4, LoRA r=16, α=32), adaptadores em `models/qwen35-0.8b/medpt-top10-merged/final/`
**Dataset:** `medpt-top10-merged` — MedPT, 9 classes (Psicólogo e "Psicólogo, Psicanalista" fundidos), 145.321 treino / 18.165 validação / **18.166 teste**
**Juiz:** `glm-5.3-flash` (Ollama Cloud), 100 exemplos estratificados

Mesmo protocolo do [Experimento 04](experimento_04_modelo_final_qwen35-2b.md) (Qwen3.5-2B), com o
mesmo dataset, os mesmos hiperparâmetros de otimização e o mesmo teste, para que os dois modelos sejam
comparáveis entre si. Os números **não** se comparam diretamente com os pilotos (Experimento 03).

## Treino

| Item | Valor |
|---|---|
| Passos | 13.626 (3 épocas, batch efetivo 32 = 4 × 8) |
| Taxa de aprendizado | 2e-4, agendador cosseno, 409 passos de aquecimento |
| Parâmetros treináveis | 6,39 M de 568,1 M (1,12%) |
| Perda de treino (média por época) | 0,804 → 0,560 → 0,469 |
| Perda de validação | 0,630 (época 1) → 0,540 (época 2) → **0,518** (época 3) |
| Tempo | ~8 h no total, em 2 trechos (pausa no checkpoint 4500) |

As perdas de treino vêm do histórico do `trainer_state.json`, por época. O `train_runtime_hours` do
`training_summary.json` cobre só o último trecho, e a perda média do trainer não serve em treino
retomado (ver nota de correção no Experimento 04); o resumo foi corrigido para trazer as perdas por época.

## Resultado quantitativo (teste completo, 18.166 exemplos)

| Métrica | Qwen3.5-0.8B | Qwen3.5-2B (Exp. 04) |
|---|---:|---:|
| Acurácia | **89,24%** | 91,14% |
| F1 macro | **0,881** | 0,898 |
| Precisão macro | 0,880 | 0,899 |
| Recall macro | 0,883 | 0,897 |

### Por classe

| Classe | Precisão | Recall | F1 | Suporte |
|---|---:|---:|---:|---:|
| Ortopedista - traumatologista | 0,97 | 0,97 | 0,97 | 1.482 |
| Oftalmologista | 0,97 | 0,97 | 0,97 | 1.211 |
| Dentista | 0,91 | 0,95 | 0,93 | 1.114 |
| Psicólogo | 0,93 | 0,90 | 0,91 | 6.686 |
| Dermatologista | 0,89 | 0,93 | 0,91 | 1.505 |
| Ginecologista | 0,89 | 0,92 | 0,90 | 2.662 |
| Otorrino | 0,94 | 0,86 | 0,90 | 773 |
| Urologista | 0,89 | 0,82 | 0,86 | 1.618 |
| **Psiquiatra** | 0,52 | 0,63 | **0,57** | 1.115 |

### Onde o modelo erra

Dos 1.954 erros, **948 (48,5%) são a confusão Psiquiatra ↔ Psicólogo**:

| Real → previsto | Erros | % da classe real |
|---|---:|---:|
| Psicólogo → Psiquiatra | 578 | 8,6% |
| Psiquiatra → Psicólogo | 370 | 33,2% |
| Urologista → Ginecologista | 210 | 13,0% |
| Ginecologista → Urologista | 87 | 3,3% |
| Ginecologista → Dermatologista | 58 | 2,2% |
| Otorrino → Dentista | 45 | 5,8% |

Em relação ao 2B, o 0.8B **prevê Psiquiatra com mais frequência do que deveria** (578 Psicólogos
classificados como Psiquiatra, contra 371 no 2B), o que derruba a precisão dessa classe (0,52 contra
0,64) e o F1 (0,57 contra 0,63). As demais classes ficam próximas às do 2B.

## Avaliação qualitativa com o GLM 5.3 (100 exemplos)

Rubrica estilo HealthBench (5 critérios, notas de 0 a 1). Nenhuma resposta do juiz ficou ilegível.

| Métrica | Qwen3.5-0.8B | Qwen3.5-2B (Exp. 04) |
|---|---:|---:|
| **Nota geral** | **0,946** | 0,967 |
| Acurácia na amostra | 91% | 94% |
| Nota média quando acerta | 0,991 | 0,995 |
| Nota média quando erra | 0,493 | 0,527 |

| Tema | Exemplos | Nota |
|---|---:|---:|
| Casos comuns / rotineiros | 67 | 0,949 |
| Saúde mental | 30 | 0,935 |
| Urgentes (sem sobreposição com saúde mental) | 3 | 1,000 |

### Casos com sinais de alerta

A amostra reserva 15 exemplos com palavras-chave de alerta. Olhando os 15 juntos:

- **14 de 15 classificados corretamente**, nota média **0,931** (o 2B acertou os 15, nota 0,987).
- O único erro foi **Psicólogo → Psiquiatra** (queixa de uma adolescente com falta de ar, suor frio e
  pesadelos): trocou uma especialidade de saúde mental por outra, sem desviar o caso para uma área
  sem relação. Nota do juiz: 0,68.

A amostra é pequena (15 casos), então isto não mede segurança. Medir isso com rigor exigiria avaliar
os 294 exemplos de teste com palavras-chave de alerta.

### Os 9 erros da amostra

| Real → previsto | Nota do juiz |
|---|---:|
| Dermatologista → Psicólogo | 0,36 |
| Dermatologista → Oftalmologista | 0,32 |
| Urologista → Ginecologista | 0,52 |
| Urologista → Ginecologista | 0,32 |
| Otorrino → Dentista | 0,52 |
| Otorrino → Dentista | 0,60 |
| Psiquiatra → Psicólogo | 0,64 |
| Psiquiatra → Psicólogo | 0,48 |
| Psicólogo → Psiquiatra | 0,68 |

## Comparação com os outros experimentos

| Experimento | Modelo | Dataset | Teste | Acurácia | F1 macro | Nota GLM |
|---|---|---|---|---:|---:|---:|
| 03 (piloto) | Qwen3.5-0.8B | piloto, 10 classes | 500 | 72,0% | 0,710 | 0,874 |
| **05 (final)** | **Qwen3.5-0.8B** | **completo, 9 classes** | **18.166** | **89,2%** | **0,881** | **0,946** |
| 04 (final) | Qwen3.5-2B | completo, 9 classes | 18.166 | 91,1% | 0,898 | 0,967 |

O 2B final melhora o 0.8B final em ~1,9 ponto de acurácia e 0,017 de F1 macro, com mais que o dobro
dos parâmetros (1,39 B contra 0,57 B no total). O ganho de escala existe, mas é pequeno perto do ganho
de treinar com o dataset completo. O salto do piloto para o final **não é uma comparação limpa**: muda o
teste (500 → 18.166) e o número de classes (10 → 9, com a fusão).

## Limitações

- **Sem linha de base zero-shot no dataset completo** (o ganho do ajuste fino só está medido nos pilotos).
- **Nota do GLM com 100 exemplos e um único juiz**, sem revisão médica; margem de erro grande por tema.
- **Psicólogo concentra 37% do teste**; o F1 macro (0,881) é uma métrica mais honesta que a acurácia.
- O campo `peak_vram_gb` do `training_summary.json` (17,0 GB) excede os 12 GB físicos da placa e não
  deve ser usado como medida de memória.

## Artefatos

- `results/qwen35-0.8b/medpt-top10-merged/evaluation/` — métricas, matriz de confusão e as 18.166 predições
- `results/qwen35-0.8b/medpt-top10-merged/qualitative_glm/` — notas do juiz por critério e análise de erros
- `models/qwen35-0.8b/medpt-top10-merged/` — adaptadores (`final/`) e `training_summary.json`
- `scripts/run_final_eval.sh` — reproduz a avaliação (pausável com `scripts/pausar.sh`)

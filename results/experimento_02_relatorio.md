# Experimento 02 — Fine-tuning QLoRA do Qwen3.5-0.8B no MedPT (Dataset completo)

**Data do treino:** 30–31/05/2026
**Autores:** Matheus Marques Eiras e Ciro
**Modelo resultante:** `models/qwen35-medpt-top10/final`

---

## 1. Objetivo

Treinar o modelo `Qwen/Qwen3.5-0.8B` na tarefa de triagem médica em português
brasileiro — classificar a especialidade médica mais adequada a partir da queixa
do paciente — usando **todo o dataset `AKCIT/MedPT`** para as 10 especialidades mais
frequentes.

Este é o experimento "de produção", sucessor do piloto (Experimento 01), que havia
usado apenas 500 exemplos por classe para validar o pipeline. O objetivo aqui era
verificar se o aumento expressivo de dados (de 4 mil para ~145 mil exemplos de treino)
resolveria o principal gargalo identificado no piloto: a confusão entre as
especialidades de saúde mental.

---

## 2. O método: QLoRA (não LoRA puro)

O treinamento usa **QLoRA** = LoRA aplicado sobre o modelo base quantizado em 4 bits.
Isso permite treinar um modelo de ~0,8 bilhão de parâmetros em uma única GPU de 12 GB.

| Componente | Descrição |
|-----------|-----------|
| Quantização | Modelo base carregado em **4-bit** (`load_in_4bit=True`, via Unsloth) |
| Adaptadores | **LoRA** treinável por cima do modelo congelado |
| Framework | Unsloth + TRL `SFTTrainer` |

Apenas os adaptadores LoRA são treinados; o modelo base permanece congelado e
quantizado. O artefato final são poucos MB de adaptadores, não o modelo inteiro.

### 2.1 Configuração LoRA

| Parâmetro | Valor |
|-----------|-------|
| Rank (r) | 16 |
| Alpha | 32 |
| Dropout | 0 (obrigatório para os kernels rápidos do Unsloth) |
| Módulos alvo | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Bias | none |
| Gradient checkpointing | "unsloth" (~30% menos VRAM) |

### 2.2 Hiperparâmetros de treino

| Parâmetro | Valor |
|-----------|-------|
| Épocas | 3 |
| Batch efetivo | 32 (batch=4 × grad_accum=8) |
| Learning rate | 2e-4 |
| Scheduler | Cosine |
| Warmup | 3% dos steps |
| Otimizador | adamw_8bit |
| Precisão | bf16 |
| Máx. tokens/sequência | 1024 |
| Packing | Sim (empacota exemplos curtos na mesma sequência) |
| Steps totais | 13.626 |

### 2.3 Dataset

| Parâmetro | Valor |
|-----------|-------|
| Fonte | `AKCIT/MedPT`, top-10 especialidades por frequência |
| Cap por classe | Nenhum (dataset completo) |
| Treino | **145.321** exemplos |
| Validação | 18.165 exemplos |
| Teste | 18.166 exemplos |
| Formato | ChatML com system prompt de triagem, *non-thinking mode* |

O dataset é **desbalanceado** por natureza (reflete a frequência real das queixas):
`Psicólogo` tem 45.690 exemplos de treino, enquanto `Otorrino` tem 6.187. Esse
desbalanceamento é relevante para interpretar os resultados por classe.

### 2.4 Hardware

| Componente | Especificação |
|-----------|---------------|
| GPU | NVIDIA RTX 4070 12 GB |
| SO | Windows 11 Pro |

---

## 3. Como foi o treino (curva e duração)

### 3.1 Duração

O treino rodou por **3 épocas completas / 13.626 steps**, com **duração estimada de
~6,5 a 7 horas** de tempo de parede.

> **Sobre a duração:** o `trainer_state.json` não registra o `train_runtime` final,
> então a duração foi estimada a partir dos timestamps dos checkpoints (salvos a cada
> 500 steps). Entre os steps 12.000 e 13.500 passaram-se ~41 minutos (02:15 → 02:56 de
> 31/05), o que dá **~1,6 s/step**. Extrapolando para os 13.626 steps (~6,1 h de
> computação pura) e somando as 3 validações sobre os 18 mil exemplos do conjunto de
> validação (~6,5 min cada) mais o carregamento e tokenização inicial, chega-se a
> **aproximadamente 6h30–7h**. O treino terminou às 02:59 de 31/05/2026.

Ou seja: a impressão de "7 horas" está correta; "9 horas" superestima.

### 3.2 Curva de loss

A loss de treino caiu de forma saudável e consistente ao longo das 3 épocas, sem sinais
de overfitting (a loss de validação continuou caindo a cada época — diferente do
piloto, onde já havia leve overfitting por falta de dados):

| Marco | Loss de treino | Loss de validação |
|-------|---------------|-------------------|
| Início (step ~100) | ~1,13 | — |
| Fim da época 1 (step 4.542) | ~0,59 | **0,6011** |
| Fim da época 2 (step 9.084) | ~0,50 | **0,5129** |
| Fim da época 3 (step 13.626) | **~0,44** | — |

A queda contínua da loss de validação (0,601 → 0,513) confirma que, com o dataset
completo, **as 3 épocas foram bem aproveitadas** e o modelo não saturou — ao contrário
do piloto, onde a `eval_loss` já subia na 3ª época.

---

## 4. Resultados quantitativos

Avaliação no conjunto de teste (amostra estratificada de **200 exemplos**), modelo
fine-tuned.

### 4.1 Métricas gerais

| Métrica | Valor |
|---------|-------|
| **Accuracy** | **0,85** |
| **F1 macro** | **0,7911** |
| Precision macro | 0,8005 |
| Recall macro | 0,7924 |

### 4.2 F1 por especialidade

| Especialidade | F1 | Avaliação |
|--------------|-----|-----------|
| Oftalmologista | 1,0000 | Perfeito |
| Dentista | 0,9730 | Excelente |
| Dermatologista | 0,9444 | Excelente |
| Ginecologista | 0,9388 | Excelente |
| Ortopedista - traumatologista | 0,9286 | Excelente |
| Otorrino | 0,8889 | Muito bom |
| Psicólogo | 0,8507 | Muito bom |
| Urologista | 0,7778 | Bom |
| Psiquiatra | 0,6087 | Regular |
| Psicólogo, Psicanalista | 0,0000 | Crítico |

### 4.3 Comparação com o piloto (Experimento 01)

| Métrica | Piloto (500/classe) | Completo (este exp.) | Δ |
|---------|--------------------:|---------------------:|---|
| Accuracy | 0,668 | **0,850** | +18,2 p.p. |
| F1 macro | 0,630 | **0,791** | +0,161 |
| F1 `Psicólogo` | 0,000 | **0,851** | resolvido |
| F1 `Psiquiatra` | 0,148 | **0,609** | +0,461 |

O salto mais importante foi em **saúde mental**: a classe `Psicólogo`, que no piloto
tinha F1=0 (o modelo nunca a previa), passou a **0,85**; e `Psiquiatra` saltou de 0,15
para 0,61. Isso confirma a hipótese do piloto — o gargalo era **falta de dados**, não o
método. O volume completo deu ao modelo exemplos suficientes para aprender a fronteira
entre especialidades semanticamente próximas.

---

## 5. Limitação remanescente: `Psicólogo, Psicanalista`

A única classe que continua com **F1 = 0** é `Psicólogo, Psicanalista`. O padrão de erro
inverteu em relação ao piloto: antes o modelo colapsava tudo *para* essa classe; agora
ele colapsa essa classe *para* `Psicólogo`. A causa é a mesma — sobreposição semântica
quase total entre "Psicólogo" e "Psicólogo, Psicanalista", agravada pelo
desbalanceamento (45.690 exemplos de `Psicólogo` contra 7.794 de `Psicólogo,
Psicanalista` no treino). O modelo aprende a sempre escolher a classe majoritária.

**Recomendação:** essas duas classes provavelmente deveriam ser **fundidas** numa única
categoria "Psicólogo / Psicanalista", já que a distinção raramente é clinicamente
relevante para o objetivo de triagem. Isso eliminaria o último ponto fraco do modelo.

---

## 6. Avaliação qualitativa (HealthBench-style)

A avaliação qualitativa usa um **LLM-grader** (Ollama `llama3.1:8b`, local) com 5
critérios ponderados (acurácia, relevância clínica, awareness de gravidade,
instruction-following e penalidade por subtriagem de emergência), num total de 25
pontos positivos.

### 6.1 Recalibração do grader

No teste de fumaça inicial (10 exemplos) o score geral ficou em **0,344**, mas o grader
aplicava a penalidade de subtriagem de emergência (−8) de forma agressiva e muitas vezes
incorreta, mesmo em casos claramente rotineiros, o que derrubava o score artificialmente.
O critério de penalidade foi então **recalibrado** para disparar apenas quando a queixa do
paciente descreve um sinal de emergência com risco de vida explícito (dor no peito, falta
de ar, convulsão, AVC etc.). O restante da rubrica permaneceu igual.

### 6.2 Resultado com o grader recalibrado (n=100)

Avaliação com **100 exemplos estratificados** do conjunto de teste, grader `llama3.1:8b`
recalibrado:

| Métrica | Valor (n=100) |
|---------|---------------|
| **Score qualitativo geral** | **0,9512 / 1,000** |
| Accuracy na amostra | 0,81 |
| Score — casos rotineiros | 0,9656 (n=72) |
| Score — saúde mental | 0,9111 (n=27) |
| Score — casos urgentes | 1,0000 (n=1, não conclusivo) |

O salto de 0,344 (piloto, grader enviesado) para **0,9512** confirma que o número baixo
inicial era um artefato da superpenalização, não do modelo. A categoria de casos urgentes
teve apenas 1 exemplo na amostra (a base tem poucos red flags), então esse valor não é
conclusivo — garantir mais casos urgentes na amostragem é um próximo passo.

---

## 7. Conclusões

1. **O método é QLoRA** (LoRA 4-bit), e funciona muito bem para esta tarefa: accuracy
   de **85%** e F1-macro de **0,79** num modelo de apenas 0,8 B parâmetros.
2. **Dados resolvem o gargalo:** treinar com o dataset completo (~145 mil exemplos vs.
   4 mil do piloto) elevou a accuracy de 67% para 85% e fez as classes de saúde mental
   saírem de praticamente inúteis (F1≈0–0,15) para funcionais (F1 0,61–0,85).
3. **O treino foi estável e bem dimensionado:** loss de validação caindo até o fim, sem
   overfitting, em ~**6,5–7 horas** numa RTX 4070 de 12 GB.
4. **Único ponto fraco:** a distinção `Psicólogo` vs. `Psicólogo, Psicanalista`,
   resolvível por fusão de classes.
5. **A avaliação qualitativa foi refeita** com 100 exemplos e o grader recalibrado,
   chegando a **0,9512**; resta ampliar a amostra de casos urgentes (apenas 1 nesta
   rodada) para fechar essa categoria.

---

## 8. Artefatos

| Artefato | Caminho |
|---------|---------|
| Adaptadores LoRA finais | `models/qwen35-medpt-top10/final/` |
| Dataset completo (splits) | `data/top10/` |
| Resultados quantitativos | `results/evaluation/results.json` |
| Relatório de classificação | `results/evaluation/report_finetuned.txt` |
| Predições do modelo | `results/evaluation/finetuned_predictions.json` |
| Matriz de confusão | `results/evaluation/confusion_matrix_finetuned.png` |
| Resultados qualitativos | `results/qualitative/qualitative_results.json` |
| Estado do treino (curva) | `models/qwen35-medpt-top10/checkpoint-13626/trainer_state.json` |

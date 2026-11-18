# Experimento 01 — Fine-tuning QLoRA do Qwen3.5-0.8B no MedPT (Piloto 500/classe)

**Data:** 25/05/2026  
**Autores:** Matheus Marques Eiras e Ciro  

---

## 1. Objetivo

Validar o pipeline completo de fine-tuning QLoRA do modelo `Qwen/Qwen3.5-0.8B` no dataset `AKCIT/MedPT` para a tarefa de triagem médica em português brasileiro, classificando a especialidade médica mais adequada a partir da queixa do paciente.

Este experimento é um **piloto com dataset reduzido** (500 exemplos por classe) para verificar a corretude do pipeline antes do treino com o dataset completo.

---

## 2. Configuração Experimental

### 2.1 Modelo Base

| Parâmetro | Valor |
|-----------|-------|
| Modelo | `Qwen/Qwen3.5-0.8B` |
| Arquitetura | Híbrida: GatedDeltaNet (linear attention) + Sparse MoE |
| Parâmetros totais | 758.782.784 |
| Quantização | NF4 4-bit (QLoRA) |
| Dtype de computação | bfloat16 |

### 2.2 Configuração LoRA

| Parâmetro | Valor |
|-----------|-------|
| Rank (r) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Módulos alvo | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Parâmetros treináveis | 6.389.760 (0,84% do total) |

### 2.3 Hiperparâmetros de Treino

| Parâmetro | Valor |
|-----------|-------|
| Épocas | 3 |
| Batch efetivo | 32 (batch=4 × grad_accum=8) |
| Learning rate | 2e-4 |
| Scheduler | Cosine |
| Warmup | ~3% dos steps totais (11 steps) |
| Otimizador | paged_adamw_8bit |
| Max tokens por sequência | 1024 |
| Gradient checkpointing | Sim |

### 2.4 Dataset

| Parâmetro | Valor |
|-----------|-------|
| Fonte | `AKCIT/MedPT` (384.095 exemplos brutos) |
| Especialidades | Top-10 por frequência |
| Cap por classe | 500 exemplos |
| Split treino | 4.000 exemplos (400/classe) |
| Split validação | 500 exemplos (50/classe) |
| Split teste | 500 exemplos (50/classe) |
| Formato | ChatML com system prompt de triagem |

**Especialidades incluídas:**

| # | Especialidade | Exemplos no treino |
|---|--------------|-------------------|
| 1 | Psicólogo | 400 |
| 2 | Ginecologista | 400 |
| 3 | Urologista | 400 |
| 4 | Dermatologista | 400 |
| 5 | Ortopedista - traumatologista | 400 |
| 6 | Oftalmologista | 400 |
| 7 | Psiquiatra | 400 |
| 8 | Dentista | 400 |
| 9 | Psicólogo, Psicanalista | 400 |
| 10 | Otorrino | 400 |

### 2.5 Hardware

| Componente | Especificação |
|-----------|---------------|
| GPU | NVIDIA RTX 4070 12 GB |
| CUDA | 12.8 |
| SO | Windows 11 Pro |

---

## 3. Curva de Treino

| Época | Loss (treino) | Loss (val) | Acc. token (treino) | Acc. token (val) |
|-------|--------------|------------|--------------------|--------------------|
| 1 | 0,954 → 1,050 | 0,9395 | 80,98% → 81,25% | 82,47% |
| 2 | 0,834 → 0,864 | 0,9161 | 84,24% → 83,81% | 82,74% |
| 3 | 0,785 → 0,794 | 0,9209 | 84,80% → 84,84% | 82,81% |

- **Tempo total de treino:** 1.043 segundos (~17 minutos)
- **Velocidade:** 0,36 steps/segundo (2,78 s/step)
- **Loss de treino final:** 0,902

A leve alta do `eval_loss` na época 3 (0,9161 → 0,9209) indica início de overfitting leve, esperado dado o volume reduzido de dados.

---

## 4. Avaliação Quantitativa (script 03)

Avaliação no conjunto de teste (500 exemplos, 50 por especialidade) comparando quatro configurações do mesmo modelo base.

### 4.1 Resultados Gerais

| Setup | Accuracy | F1 Macro | Precision Macro | Recall Macro |
|-------|----------|----------|-----------------|--------------|
| Zero-shot | 22,4% | 0,1897 | 0,4632 | 0,2240 |
| Three-shot | 15,0% | 0,1855 | 0,3108 | 0,1500 |
| Five-shot | 14,8% | 0,1749 | 0,2744 | 0,1480 |
| **Fine-tuned** | **66,8%** | **0,6296** | **0,7056** | **0,6680** |

**Ganho do fine-tuning sobre zero-shot:** +44,4 p.p. de accuracy (+198% relativo)

> **Nota sobre few-shot:** O desempenho few-shot (3-shot e 5-shot) foi inferior ao zero-shot. Isso é esperado em modelos pequenos (0,8B): o contexto com exemplos confunde o modelo em vez de orientá-lo, fenômeno documentado na literatura para LLMs com capacidade limitada de in-context learning.

### 4.2 F1 por Especialidade (Fine-tuned)

| Especialidade | F1 | Observação |
|--------------|-----|-----------|
| Ortopedista - traumatologista | 0,9592 | Excelente |
| Oftalmologista | 0,9375 | Excelente |
| Otorrino | 0,8000 | Muito bom |
| Urologista | 0,7629 | Muito bom |
| Ginecologista | 0,7742 | Muito bom |
| Dermatologista | 0,7158 | Bom |
| Dentista | 0,7244 | Bom |
| Psicólogo, Psicanalista | 0,4737 | Regular |
| Psiquiatra | 0,1481 | Fraco |
| Psicólogo | 0,0000 | Crítico |

---

## 5. Avaliação Qualitativa — HealthBench-style (script 04)

Avaliação de 100 exemplos estratificados usando LLM-grader com 5 critérios ponderados.

**Grader utilizado:** Ollama (llama3.1:8b, local)

### 5.1 Critérios de Avaliação

| Critério | Pontos máx. | Pontos mín. |
|----------|------------|------------|
| Acurácia — especialidade correta | 10 | 0 |
| Relevância clínica — especialidade defensável | 7 | 0 |
| Awareness de contexto — identificação de gravidade | 5 | 0 |
| Instruction following — formato correto | 3 | 0 |
| Penalidade — subtriagem de emergência | 0 | -8 |
| **Total positivo máximo** | **25** | — |

### 5.2 Resultados por Tema

| Tema | Score | N exemplos |
|------|-------|-----------|
| Casos comuns / rotineiros | 0,4878 | 77 |
| Casos urgentes / red flags | 0,4800 | 1 |
| Saúde mental | 0,3073 | 22 |
| **Geral** | **0,4480** | **100** |

- **Accuracy nos 100 exemplos:** 67,0%
- **Score qualitativo geral:** 0,448 / 1,000

A diferença entre accuracy (67%) e score qualitativo (44,8%) reflete o peso dos critérios de relevância clínica e awareness de contexto, que penalizam mesmo erros clinicamente próximos.

---

## 6. Análise de Erros

### 6.1 Padrões de Confusão (33 erros em 100 amostras)

| Confusão | Ocorrências | Análise |
|---------|-------------|---------|
| Psiquiatra → Psicólogo, Psicanalista | 10 | Especialidades semanticamente próximas; dataset insuficiente para discriminar |
| Psicólogo → Psicólogo, Psicanalista | 9 | Mesma causa; o modelo converge para a classe "Psicólogo, Psicanalista" |
| Dermatologista → Dentista | 3 | Possível sobreposição em queixas de boca/lábio/face |
| Outros | 11 | Erros isolados sem padrão consistente |

### 6.2 Discussão

O principal gargalo do modelo é a **triagem de saúde mental**: as três especialidades `Psicólogo`, `Psiquiatra` e `Psicólogo, Psicanalista` são semanticamente muito próximas e o modelo com apenas 400 exemplos de treino por classe não aprendeu a fronteira decisória entre elas. O F1=0 para `Psicólogo` indica que o modelo nunca prediz essa classe isolada — sempre opta por `Psicólogo, Psicanalista`.

Para as demais 7 especialidades, o desempenho é consistentemente bom (F1 > 0,71), demonstrando que o pipeline de fine-tuning é eficaz quando o volume de dados é suficiente.

---

## 7. Conclusões do Piloto

| Aspecto | Resultado |
|---------|-----------|
| Pipeline funcional | Sim — scripts 01 a 04 executados sem erros |
| Fine-tuning eficaz | Sim — accuracy 22% → 67% (+198% relativo) |
| Gargalo identificado | Confusão entre especialidades de saúde mental |
| Causa do gargalo | Dataset reduzido (400 exemplos/classe) + alta similaridade semântica |

### 7.1 Próximos Passos

1. **Treinar com dataset maior** (`--max_per_class 2000` ou dataset completo ~145k exemplos) para reduzir a confusão entre especialidades de saúde mental
2. **Avaliar o impacto do volume de dados** repetindo a avaliação com 500, 2.000 e dataset completo — curva aprendizado vs. dados
3. **Analisar as matrizes de confusão** geradas (`results/evaluation/confusion_matrix_finetuned.png`) para visualização dos padrões de erro
4. **Considerar fusão de classes** `Psicólogo` + `Psicólogo, Psicanalista` se a distinção não for clinicamente relevante para o objetivo de triagem

---

## 8. Artefatos Gerados

| Artefato | Caminho |
|---------|---------|
| Dataset piloto (treino/val/teste) | `data/top10_50/` |
| Adaptadores LoRA finais | `models/qwen35-medpt-500/final/` |
| Resultados quantitativos | `results/evaluation/results.json` |
| Predições fine-tuned | `results/evaluation/finetuned_predictions.json` |
| Matrizes de confusão | `results/evaluation/confusion_matrix_*.png` |
| Resultados qualitativos | `results/qualitative/qualitative_results.json` |
| Análise de erros | `results/qualitative/error_analysis.json` |

# Experimento 06 — Modelo final Gemma 3 4B (dataset completo) avaliado com GLM 5.3

**Data:** 11/11/2026
**Modelo:** `unsloth/gemma-3-4b-pt` (base, pré-treinado) com QLoRA (4-bit NF4, LoRA r=16, α=32), adaptadores em `models/gemma3-4b/medpt-top10-merged/final/`
**Dataset:** `medpt-top10-merged` — MedPT, 9 classes (Psicólogo e "Psicólogo, Psicanalista" fundidos), 145.321 treino / 18.165 validação / **18.166 teste**
**Juiz:** `glm-5.3-flash` (Ollama Cloud), 100 exemplos estratificados (mesma amostra dos Experimentos 04 e 05)

Mesmo dataset, mesmo teste e mesma amostra do juiz dos [Experimento 04](experimento_04_modelo_final_qwen35-2b.md)
(Qwen3.5-2B) e [Experimento 05](experimento_05_modelo_final_qwen35-0.8b.md) (Qwen3.5-0.8B). A otimização
**não é idêntica**: para caber em 12 GB e terminar em tempo razoável, o Gemma 4B usou taxa de aprendizado
menor e corte de exemplos longos (ver tabela abaixo).

## Treino

| Item | Gemma 3 4B | Qwen3.5 (Exp. 04/05) |
|---|---|---|
| Passos | 13.617 (3 épocas) | 13.626 |
| Batch efetivo | 32 = **8 × 4** | 32 = 4 × 8 |
| Taxa de aprendizado | **1e-4**, cosseno, 409 passos de aquecimento | 2e-4 |
| Exemplos de treino | **145.234** (87 com mais de 512 tokens descartados) | 145.321 |
| Parâmetros treináveis | 29,8 M (1,03%) | 6,4 M (0.8B) / 10,9 M (2B) |
| Perda de treino (média por época) | 0,710 → 0,474 → 0,394 | — |
| Perda de validação | 0,539 (época 1) → 0,467 (época 2) → **0,456** (época 3) | 0,518 (0.8B, época 3) |
| Pico de VRAM | 10,45 GB (teto de 90% da GPU) | — |

Os 87 exemplos descartados são 0,06% do treino; o teste não foi filtrado (os 18.166 exemplos foram avaliados).

**Tempo.** O treino rodou numa RTX 4070 de uso compartilhado, em vários trechos (pausas nos checkpoints
3500, 4500, 6000, 7800 e 10500, e retomadas após quedas da sessão), terminando em 10/11/2026. O
`train_runtime_hours` do resumo (5,09 h) cobre só o último trecho. Em ritmo normal o treino faz
~8–9 s/passo (~32 h para os 13.617 passos) e cada validação completa levou de 1,4 h a 3 h. Quando outro
programa ocupava a GPU (editor de vídeo, jogo), a memória transbordava para a RAM e o ritmo caía 10×;
nesses casos o treino foi pausado num checkpoint e retomado depois, sem perda.

## Resultado quantitativo (teste completo, 18.166 exemplos)

| Métrica | **Gemma 3 4B** | Qwen3.5-2B (Exp. 04) | Qwen3.5-0.8B (Exp. 05) |
|---|---:|---:|---:|
| Acurácia | **91,35%** | 91,14% | 89,24% |
| F1 macro | **0,901** | 0,898 | 0,881 |
| Precisão macro | 0,901 | 0,899 | 0,880 |
| Recall macro | 0,901 | 0,897 | 0,883 |
| Erros | 1.571 | 1.609 | 1.954 |

### Por classe

| Classe | Precisão | Recall | F1 | Suporte |
|---|---:|---:|---:|---:|
| Ortopedista - traumatologista | 0,98 | 0,98 | 0,98 | 1.482 |
| Oftalmologista | 0,97 | 0,98 | 0,97 | 1.211 |
| Dentista | 0,93 | 0,95 | 0,94 | 1.114 |
| Psicólogo | 0,94 | 0,93 | 0,93 | 6.686 |
| Dermatologista | 0,93 | 0,94 | 0,93 | 1.505 |
| Otorrino | 0,94 | 0,91 | 0,92 | 773 |
| Ginecologista | 0,90 | 0,93 | 0,91 | 2.662 |
| Urologista | 0,90 | 0,85 | 0,88 | 1.618 |
| **Psiquiatra** | 0,63 | 0,64 | **0,63** | 1.115 |

### Onde o modelo erra

Dos 1.571 erros, **748 (47,6%) são a confusão Psiquiatra ↔ Psicólogo**:

| Real → previsto | Erros | % da classe real |
|---|---:|---:|
| Psicólogo → Psiquiatra | 379 | 5,7% |
| Psiquiatra → Psicólogo | 369 | 33,1% |
| Urologista → Ginecologista | 194 | 12,0% |
| Ginecologista → Urologista | 86 | 3,2% |
| Dermatologista → Ginecologista | 45 | 3,0% |
| Otorrino → Dentista | 36 | 4,7% |

O padrão de erros é praticamente o do Qwen3.5-2B (750 erros Psiquiatra ↔ Psicólogo, 199
Urologista → Ginecologista). O ganho do Gemma vem de pequenas melhoras espalhadas (Otorrino, Dentista,
Dermatologista), não da confusão principal, que parece ser um limite do próprio dataset: as duas
especialidades respondem a queixas muito parecidas.

## Avaliação qualitativa com o GLM 5.3 (100 exemplos)

Rubrica estilo HealthBench (5 critérios, notas de 0 a 1). Nenhuma resposta do juiz ficou ilegível.

| Métrica | **Gemma 3 4B** | Qwen3.5-2B | Qwen3.5-0.8B |
|---|---:|---:|---:|
| **Nota geral** | **0,961** | 0,967 | 0,946 |
| Acurácia na amostra | **95%** | 94% | 91% |
| Nota média quando acerta | 0,983 | 0,995 | 0,991 |
| Nota média quando erra | 0,536 | 0,527 | 0,493 |

| Tema | Exemplos | Nota |
|---|---:|---:|
| Casos comuns / rotineiros | 67 | 0,973 |
| Saúde mental | 30 | 0,931 |
| Urgentes (sem sobreposição com saúde mental) | 3 | 0,987 |

O Gemma acerta mais exemplos da amostra que o 2B (95 contra 94), mas tem nota geral um pouco menor.
A diferença vem de **dois acertos penalizados pelo juiz**, explicados abaixo.

### Casos com sinais de alerta e variação do juiz

Os 15 exemplos com palavras-chave de alerta foram **todos classificados corretamente** (15 de 15, como
o 2B), mas com nota média **0,909** (2B: 0,987). Dois deles, queixas de falta de ar intensa atribuída a
ansiedade, cuja resposta esperada é Psicólogo, receberam notas 0,36 e 0,44: o juiz aplicou a penalidade
de "subtriagem de emergência" mesmo com a predição idêntica ao gabarito.

**Os mesmos dois exemplos, com a mesma predição (Psicólogo), receberam 1,00 e 0,92 na avaliação do
Qwen3.5-2B.** A entrada do juiz é a mesma nos dois casos, então a diferença é variação do próprio juiz,
não do modelo. Isso mostra que diferenças de ~0,01 na nota geral entre modelos estão dentro do ruído do
juiz; para separar modelos com nota tão próxima seria preciso repetir a avaliação várias vezes ou usar
mais exemplos.

### Os 5 erros da amostra

| Real → previsto | Nota do juiz |
|---|---:|
| Otorrino → Dentista | 0,44 |
| Otorrino → Dentista | 0,56 |
| Urologista → Ginecologista | 0,44 |
| Psiquiatra → Psicólogo | 0,64 |
| Psiquiatra → Psicólogo | 0,60 |

## Comparação com os outros experimentos

| Experimento | Modelo | Dataset | Teste | Acurácia | F1 macro | Nota GLM |
|---|---|---|---|---:|---:|---:|
| 03 (piloto) | Gemma 3 4B | piloto, 10 classes | 500 | 75,8% | 0,758 | 0,909 |
| 05 (final) | Qwen3.5-0.8B | completo, 9 classes | 18.166 | 89,2% | 0,881 | 0,946 |
| 04 (final) | Qwen3.5-2B | completo, 9 classes | 18.166 | 91,1% | 0,898 | 0,967 |
| **06 (final)** | **Gemma 3 4B** | **completo, 9 classes** | **18.166** | **91,4%** | **0,901** | **0,961** |

(O piloto, do [Experimento 03](experimento_03_pilotos_glm.md), não se compara com os finais:
teste de 500 exemplos e 10 classes.)

O Gemma 3 4B é o melhor modelo no teste completo, mas por pouco: **+0,2 ponto de acurácia e +0,003 de
F1 macro sobre o Qwen3.5-2B**, com cerca de 2× os parâmetros e um treino bem mais caro (passo ~2× mais
lento, validação de horas e risco de transbordar a VRAM de 12 GB). Com 18.166 exemplos, 0,2 ponto são
~38 acertos a mais; é uma diferença pequena demais para afirmar superioridade sem um teste estatístico
(p. ex. McNemar nas predições pareadas). Para uso prático no hardware do projeto, o Qwen3.5-2B entrega
quase o mesmo resultado a um custo bem menor.

## Limitações

- **Otimização diferente dos Qwen** (lr 1e-4 contra 2e-4, 87 exemplos longos descartados): a comparação
  entre famílias mistura efeito do modelo e da receita de treino.
- **Sem linha de base zero-shot no dataset completo** (o ganho do ajuste fino só está medido nos pilotos).
- **Nota do GLM com 100 exemplos e um único juiz**, sem revisão médica; o próprio juiz varia entre
  execuções (ver acima).
- **Psicólogo concentra 37% do teste**; o F1 macro (0,901) é uma métrica mais honesta que a acurácia.
- O `total_params` do resumo (2,9 B) é contado com os pesos 4-bit empacotados e subestima o tamanho real
  do modelo (~4 B).

## Artefatos

- `results/gemma3-4b/medpt-top10-merged/evaluation/` — métricas, matriz de confusão e as 18.166 predições
- `results/gemma3-4b/medpt-top10-merged/qualitative_glm/` — notas do juiz por critério e análise de erros
- `models/gemma3-4b/medpt-top10-merged/` — adaptadores (`final/`) e `training_summary.json`
- `scripts/train_final_queue.sh gemma3-4b` — reproduz o treino (retoma do último checkpoint)
- `scripts/run_final_eval.sh gemma3-4b` — reproduz a avaliação (pausável com `scripts/pausar.sh`)
- Modelo publicado: [`matheus-ciro-tcc/gemma3-4b-triagem-medica-final`](https://huggingface.co/matheus-ciro-tcc/gemma3-4b-triagem-medica-final)

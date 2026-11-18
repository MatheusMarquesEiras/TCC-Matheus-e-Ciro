# Experimento 07 — Modelo final Gemma 3 1B (dataset completo) avaliado com GLM 5.3

**Data:** treino e predições por Ciro Guilherme Nass (concluídos em 01/11/2026); juiz GLM rodado em 11/11/2026
**Modelo:** `unsloth/gemma-3-1b-pt` (base, pré-treinado) com QLoRA (4-bit NF4, LoRA r=16, α=32)
**Dataset:** `medpt-top10-merged` — MedPT, 9 classes (Psicólogo e "Psicólogo, Psicanalista" fundidos), 145.321 treino / 18.165 validação / **18.166 teste**
**Juiz:** `glm-5.3-flash` (Ollama Cloud), 100 exemplos estratificados (mesma amostra dos Experimentos 04, 05 e 06)

Fecha a série de modelos finais: mesmo dataset, mesmo teste e mesma amostra do juiz dos
[Experimento 04](experimento_04_modelo_final_qwen35-2b.md) (Qwen3.5-2B),
[Experimento 05](experimento_05_modelo_final_qwen35-0.8b.md) (Qwen3.5-0.8B) e
[Experimento 06](experimento_06_modelo_final_gemma3-4b.md) (Gemma 3 4B).

O treino e as predições no teste foram feitos na máquina do Ciro e publicados em
[`matheus-ciro-tcc/gemma3-1b-triagem-medica-final`](https://huggingface.co/matheus-ciro-tcc/gemma3-1b-triagem-medica-final);
aqui as predições publicadas foram baixadas e passadas pelo mesmo juiz dos outros modelos.

## Treino

| Item | Valor |
|---|---|
| Passos | 2.112 (3 épocas, batch efetivo 32 = 1 × 32, com *packing*) |
| Taxa de aprendizado | 2e-4, cosseno, 63 passos de aquecimento |
| Exemplos de treino | 145.321 (nenhum descartado) |
| Parâmetros treináveis | 13,0 M (1,93%) |
| Perda de treino (média por época) | 0,978 → 0,644 → 0,558 |
| Perda de validação | 0,732 (época 1) → 0,601 (época 2) → **0,582** (época 3) |
| Hardware | RTX 4060 Laptop 8 GB (pico de VRAM 7,08 GB) |
| Tempo | 26,6 h (último trecho registrado no resumo) |

Com *packing*, várias queixas curtas são concatenadas numa mesma sequência de até 1.024 tokens; por
isso o 1B fez 2.112 passos de otimização, contra ~13.600 nos outros modelos finais, vendo os mesmos
dados por 3 épocas. A taxa de aprendizado é a dos Qwen3.5 (2e-4); o Gemma 3 4B usou 1e-4.

## Resultado quantitativo (teste completo, 18.166 exemplos)

| Métrica | **Gemma 3 1B** | Qwen3.5-0.8B | Qwen3.5-2B | Gemma 3 4B |
|---|---:|---:|---:|---:|
| Acurácia | **89,13%** | 89,24% | 91,14% | 91,35% |
| F1 macro | **0,863** | 0,881 | 0,898 | 0,901 |
| Precisão macro | 0,882 | 0,880 | 0,899 | 0,901 |
| Recall macro | 0,854 | 0,883 | 0,897 | 0,901 |
| Erros | 1.975 | 1.954 | 1.609 | 1.571 |

### Por classe

| Classe | Precisão | Recall | F1 | Suporte |
|---|---:|---:|---:|---:|
| Ortopedista - traumatologista | 0,97 | 0,97 | 0,97 | 1.482 |
| Oftalmologista | 0,97 | 0,97 | 0,97 | 1.211 |
| Dentista | 0,93 | 0,93 | 0,93 | 1.114 |
| Psicólogo | 0,88 | 0,95 | 0,92 | 6.686 |
| Ginecologista | 0,88 | 0,91 | 0,90 | 2.662 |
| Dermatologista | 0,89 | 0,91 | 0,90 | 1.505 |
| Otorrino | 0,92 | 0,88 | 0,90 | 773 |
| Urologista | 0,90 | 0,82 | 0,85 | 1.618 |
| **Psiquiatra** | 0,61 | **0,34** | **0,44** | 1.115 |

### Onde o modelo erra

Dos 1.975 erros, **927 (46,9%) são a confusão Psiquiatra ↔ Psicólogo**, mas, ao contrário dos outros
modelos, quase toda numa direção:

| Real → previsto | Erros | % da classe real |
|---|---:|---:|
| **Psiquiatra → Psicólogo** | **707** | **63,4%** |
| Psicólogo → Psiquiatra | 220 | 3,3% |
| Urologista → Ginecologista | 210 | 13,0% |
| Ginecologista → Urologista | 88 | 3,3% |
| Ginecologista → Dermatologista | 59 | 2,2% |
| Dermatologista → Ginecologista | 59 | 3,9% |
| Urologista → Psicólogo | 43 | 2,7% |

O 1B **quase não prevê Psiquiatra**: só 34% dos casos de Psiquiatra são reconhecidos, contra 64% no
Gemma 4B e 63% no Qwen3.5-0.8B. O modelo menor resolve a ambiguidade entre as duas especialidades de
saúde mental mandando quase tudo para a classe majoritária (Psicólogo, 37% do teste). Isso mantém a
acurácia próxima à do Qwen3.5-0.8B (89,1% contra 89,2%), mas derruba o F1 macro (0,863 contra 0,881),
que é a métrica que pesa igualmente as classes.

## Avaliação qualitativa com o GLM 5.3 (100 exemplos)

Rubrica estilo HealthBench (5 critérios, notas de 0 a 1). Nenhuma resposta do juiz ficou ilegível.

| Métrica | **Gemma 3 1B** | Qwen3.5-0.8B | Qwen3.5-2B | Gemma 3 4B |
|---|---:|---:|---:|---:|
| **Nota geral** | **0,944** | 0,946 | 0,967 | 0,961 |
| Acurácia na amostra | 90% | 91% | 94% | 95% |
| Nota média quando acerta | 0,991 | 0,991 | 0,995 | 0,983 |
| Nota média quando erra | 0,520 | 0,493 | 0,527 | 0,536 |

| Tema | Exemplos | Nota |
|---|---:|---:|
| Casos comuns / rotineiros | 67 | 0,974 |
| Saúde mental | 30 | **0,872** |
| Urgentes (sem sobreposição com saúde mental) | 3 | 0,973 |

A nota de saúde mental (0,872) é a mais baixa entre os modelos finais e reflete o mesmo problema do
teste completo: **7 dos 10 erros da amostra são Psiquiatra → Psicólogo**.

### Casos com sinais de alerta

Dos 15 exemplos com palavras-chave de alerta, **13 foram classificados corretamente** (nota média
0,893). Os dois erros são Psiquiatra → Psicólogo (notas 0,52 e 0,56): trocam uma especialidade de saúde
mental por outra, sem desviar o caso para uma área sem relação. A amostra é pequena (15 casos) e não
mede segurança.

### Os 10 erros da amostra

| Real → previsto | Ocorrências | Notas do juiz |
|---|---:|---|
| Psiquiatra → Psicólogo | 7 | 0,40 a 0,60 |
| Otorrino → Dentista | 1 | 0,56 |
| Otorrino → Ortopedista - traumatologista | 1 | 0,36 |
| Urologista → Ginecologista | 1 | 0,52 |

## Comparação dos quatro modelos finais

| Experimento | Modelo | Parâmetros | Acurácia | F1 macro | F1 Psiquiatra | Nota GLM |
|---|---|---:|---:|---:|---:|---:|
| **07** | **Gemma 3 1B** | ~1 B | **89,1%** | **0,863** | **0,44** | **0,944** |
| 05 | Qwen3.5-0.8B | 0,8 B | 89,2% | 0,881 | 0,57 | 0,946 |
| 04 | Qwen3.5-2B | 2 B | 91,1% | 0,898 | 0,63 | 0,967 |
| 06 | Gemma 3 4B | 4 B | 91,4% | 0,901 | 0,63 | 0,961 |

- **Escala ajuda, com retorno decrescente.** Dos modelos de ~1 B para os de 2–4 B, o F1 macro sobe
  ~0,02–0,04; do 2B para o 4B, só 0,003.
- **Entre os pequenos, o Qwen3.5-0.8B é melhor que o Gemma 3 1B** no F1 macro (0,881 contra 0,863),
  com menos parâmetros. A diferença vem quase toda de Psiquiatra (F1 0,57 contra 0,44). A comparação
  mistura modelo e receita de treino (máquina, batch 1 × 32 contra 4 × 8), então não isola o efeito da
  família.
- **Psiquiatra × Psicólogo é o limite comum** a todos os modelos: nenhum passa de F1 0,63 nessa classe.

## Limitações

- **Treino feito em outra máquina** (do Ciro); o resumo do treino e as predições foram usados como
  publicados. Só o juiz GLM foi rodado aqui.
- **Sem linha de base zero-shot no dataset completo.**
- **Nota do GLM com 100 exemplos e um único juiz**, sem revisão médica; o próprio juiz varia entre
  execuções (ver Experimento 06), então diferenças de ~0,01 na nota geral não separam modelos.
- **Psicólogo concentra 37% do teste**; para este modelo em particular, a acurácia esconde o problema
  em Psiquiatra, e o F1 macro (0,863) é a métrica mais honesta.

## Artefatos

- `results/gemma3-1b/medpt-top10-merged/evaluation/` — métricas, matriz de confusão e as 18.166 predições (baixadas do HF)
- `results/gemma3-1b/medpt-top10-merged/qualitative_glm/` — notas do juiz por critério e análise de erros
- `scripts/run_final_eval.sh gemma3-1b` — com as predições no lugar, pula a etapa 03 e roda só o juiz
- Modelo publicado: [`matheus-ciro-tcc/gemma3-1b-triagem-medica-final`](https://huggingface.co/matheus-ciro-tcc/gemma3-1b-triagem-medica-final)

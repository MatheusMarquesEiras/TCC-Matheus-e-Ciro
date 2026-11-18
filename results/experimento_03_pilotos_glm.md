# Experimento 03 — Pilotos multi-modelo avaliados com GLM 5.3 como juiz

**Data:** 04/10/2026
**Dataset:** `medpt-top10-500` (piloto: 4.000 treino / 500 teste, 50 por classe)
**Juiz:** `glm-5.3-flash` (Ollama Cloud, via servidor Ollama local)

Os quatro modelos foram treinados com o mesmo pipeline QLoRA (r=16, α=32, batch
efetivo 32, 3 épocas) no piloto de 500 exemplos por classe.

## Resultados

| Modelo | Acurácia sem ajuste | F1 sem ajuste | Acurácia ajustado | F1 ajustado | Nota GLM |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 20,4% | 0,163 | 72,0% | 0,710 | 0,874 |
| Gemma 3 1B (base) | 0,2% | 0,004 | 67,2% | 0,670 | 0,865 |
| Qwen3.5-2B | 49,8% | 0,484 | **76,2%** | 0,754 | **0,919** |
| Gemma 3 4B (base) | 0,6% | 0,011 | 75,8% | **0,758** | 0,909 |

Acurácia e F1 sobre os 500 exemplos de teste. Nota GLM: média sobre 100 exemplos
estratificados (rubrica HealthBench-style de 5 critérios, 0 a 1).

### Nota do GLM por tema

| Modelo | Saúde mental (25) | Rotineiros (71) | Urgentes (4) |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 0,810 | 0,903 | 0,750 |
| Gemma 3 1B | 0,726 | 0,910 | 0,930 |
| Qwen3.5-2B | 0,808 | 0,954 | 1,000 |
| Gemma 3 4B | 0,842 | 0,930 | 0,960 |

## Leitura

- **O ajuste fino é o que faz a tarefa funcionar.** Os Gemma 3 usados são checkpoints
  *base* (`-pt`, sem instruction tuning): sem ajuste eles praticamente não respondem
  uma especialidade (<1%). Depois de 4 mil exemplos chegam a 67–76%.
- **Tamanho ajuda, mas com retorno decrescente.** Os modelos de 2B e 4B ficaram
  empatados (~76% de acurácia, nota GLM ~0,91–0,92), à frente dos de ~1B (67–72%).
  O Qwen3.5-2B entrega o mesmo resultado do Gemma 4B com metade dos parâmetros e
  treino ~7× mais rápido (28 min × 3 h 08).
- **Saúde mental continua sendo o tema mais difícil** em todos os modelos.
- **Casos urgentes não são conclusivos:** o teste do piloto só tem 4 casos com
  sinais de alerta (a cota pedida era 15). Medir segurança exige usar o teste do
  dataset completo.

## Observações de método

- Na primeira rodada do Qwen3.5-0.8B o GLM devolveu 4 respostas sem o JSON de notas
  (contadas como 0, nota 0,842). O avaliador passou a repetir a consulta até 3 vezes
  quando a resposta vem ilegível; com isso, nenhuma falha nos quatro modelos e o
  0.8B foi reavaliado (0,874). A rodada antiga está em
  `results/qwen35-0.8b/medpt-top10-500/qualitative_glm_sem_retry/`.
- Estes números são do **piloto** e não se comparam diretamente com o Experimento 02
  (Qwen3.5-0.8B no dataset completo: 85% de acurácia, avaliado em outra amostra e
  com o LLaMA 3.1 8B como juiz).

## Artefatos

- `results/<modelo>/medpt-top10-500/evaluation/` — métricas e predições (script 03)
- `results/<modelo>/medpt-top10-500/qualitative_glm/` — notas do GLM e análise de erros (script 04)
- `scripts/run_glm_eval.sh` — roda tudo de novo, pulando o que já existe

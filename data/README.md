# Dados

Splits já preparados pelo `scripts/01_prepare_data.py` a partir do
[AKCIT/MedPT](https://huggingface.co/datasets/AKCIT/MedPT) (Färber et al., 2025).
Cada pasta tem `train`, `validation`, `test` em formato ChatML e um `metadata.json`
com rótulos, prompt e distribuição de classes.

| Pasta | Dataset (`--dataset`) | Classes | Treino / validação / teste |
|---|---|---|---|
| `top10/` | `medpt-top10` | 10 | 145.321 / 18.165 / 18.166 |
| `top10_merged/` | `medpt-top10-merged` (Psicólogo + Psicólogo, Psicanalista) | 9 | 145.321 / 18.165 / 18.166 |
| `top10_500/` | `medpt-top10-500` (piloto, 500 por classe) | 10 | 4.000 / 500 / 500 |
| `top10_50/` | 50 por classe (testes rápidos) | 10 | — |

**Os textos das queixas não são redistribuídos neste repositório.** O MedPT não declara
licença de redistribuição, então ficam aqui só os `metadata.json` (rótulos, prompt e
distribuição de classes). Para gerar os splits, rode
`uv run python scripts/01_prepare_data.py --dataset <chave>`, que baixa o MedPT do
Hugging Face e aplica o mesmo filtro e a mesma semente (42) usados no TCC.

O MedPT pertence aos seus autores; cite o trabalho original ao usar estes dados.

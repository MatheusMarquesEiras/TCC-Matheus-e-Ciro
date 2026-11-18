"""Comparativo entre todos os modelos treinados.

Varre ``results/<modelo>/<dataset>/``, junta métricas quantitativas, escore
qualitativo e custo de treino, e gera a tabela em Markdown e o gráfico de
barras que vão para o texto do TCC.

Execuções incompletas aparecem com os campos que faltam em branco — dá para
rodar o comparativo enquanto os treinos ainda estão em andamento.

Exemplos de uso:
    uv run python scripts/06_compare_models.py
    uv run python scripts/06_compare_models.py --dataset medpt-top10
"""

import _bootstrap  # noqa: F401  (efeito colateral: ajusta o sys.path)

import argparse
import json

from tcc_triagem.paths import resolve_path
from tcc_triagem.reporting import build_markdown, collect_runs, save_comparison_chart


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparativo entre modelos treinados.")
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Restringe o comparativo a um dataset. Padrão: todos",
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Raiz dos resultados. Padrão: results",
    )
    parser.add_argument(
        "--models_dir", type=str, default="models",
        help="Raiz dos modelos, de onde vem o custo de treino. Padrão: models",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/comparison",
        help="Saída do comparativo. Padrão: results/comparison",
    )
    args = parser.parse_args()

    runs = collect_runs(args.results_dir, args.models_dir, args.dataset)
    if not runs:
        print(
            "Nenhuma execução encontrada. Rode o script 03 para ao menos um modelo:\n"
            "  uv run python scripts/03_evaluate.py --model qwen35-0.8b --dataset medpt-top10"
        )
        return

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(runs)} execução(ões) encontrada(s):")
    for run in runs:
        print(
            f"  {run['model_label']:<26} {run['dataset_key']:<20} "
            f"F1 macro {run['f1_macro'] if run['f1_macro'] is not None else '—'}"
        )

    markdown = build_markdown(runs)
    md_file = output_dir / "comparativo.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(markdown)

    json_file = output_dir / "comparison.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(runs, f, ensure_ascii=False, indent=2)

    chart = save_comparison_chart(runs, output_dir / "f1_macro_por_modelo.png")

    print(f"\n{markdown}")
    print(f"Tabela salva em  {md_file}")
    print(f"Dados salvos em  {json_file}")
    if chart:
        print(f"Gráfico salvo em {chart}")


if __name__ == "__main__":
    main()

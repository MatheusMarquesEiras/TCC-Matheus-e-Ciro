"""Demo interativa de qualquer modelo ajustado.

Roda um mini-teste automático com casos fixos e, opcionalmente, entra em modo
interativo para digitar queixas. Os rótulos vêm do ``metadata.json`` do dataset,
não de uma lista fixa no código — trocar de dataset não exige tocar neste script.

Exemplos de uso:
    uv run python scripts/05_demo.py --model qwen35-0.8b --dataset medpt-top10
    uv run python scripts/05_demo.py --model gemma3-1b --dataset medpt-top10 --interactive
"""

import _bootstrap  # noqa: F401  (efeito colateral: ajusta o sys.path)

import argparse

from tcc_triagem.data_prep import labels_of, load_metadata
from tcc_triagem.dataset_registry import DATASETS, DEFAULT_DATASET, get_dataset
from tcc_triagem.model_registry import MODELS, get_model
from tcc_triagem.paths import resolve_path

CASOS_DE_TESTE = [
    {
        "queixa": "Tenho sentido muita tristeza há meses, não consigo dormir e "
                  "perdi o interesse em tudo que gostava.",
        "esperado": "Psicólogo",
    },
    {
        "queixa": "Meu joelho direito está inchado e doendo muito depois que caí "
                  "da escada há três dias.",
        "esperado": "Ortopedista - traumatologista",
    },
    {
        "queixa": "Aparecem manchas vermelhas com coceira na minha pele toda vez "
                  "que como frutos do mar.",
        "esperado": "Dermatologista",
    },
    {
        "queixa": "Estou com dificuldade para enxergar de longe e minha visão "
                  "ficou embaçada nas últimas semanas.",
        "esperado": "Oftalmologista",
    },
    {
        "queixa": "Sinto dor ao urinar há quatro dias e a urina está com cor estranha.",
        "esperado": "Urologista",
    },
]


def run_tests(classifier, casos: list[dict]) -> None:
    print("=" * 62)
    print(f"MINI-TESTE AUTOMÁTICO ({len(casos)} casos)")
    print("=" * 62)

    acertos = 0
    for i, caso in enumerate(casos, 1):
        predito = classifier.predict(caso["queixa"])
        correto = predito.strip().lower() == caso["esperado"].strip().lower()
        acertos += correto
        print(f"\n[{i}] {'OK' if correto else 'ERRO'}")
        print(f"  Queixa   : {caso['queixa'][:70]}...")
        print(f"  Esperado : {caso['esperado']}")
        print(f"  Predito  : {predito}")

    print(f"\nResultado: {acertos}/{len(casos)} corretos")
    print("=" * 62)


def run_interactive(classifier) -> None:
    print("\n" + "=" * 62)
    print("MODO INTERATIVO — digite 'sair' para encerrar")
    print("=" * 62)
    while True:
        try:
            queixa = input("\nQueixa do paciente: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if queixa.lower() in {"sair", "exit", "quit", ""}:
            break
        print(f"Especialidade recomendada: {classifier.predict(queixa)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo do modelo de triagem médica.")
    parser.add_argument(
        "--model", type=str, default="qwen35-0.8b", choices=sorted(MODELS),
        help="Chave do modelo no registro",
    )
    parser.add_argument(
        "--dataset", type=str, default=DEFAULT_DATASET, choices=sorted(DATASETS),
        help=f"Dataset em que o modelo foi treinado. Padrão: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--model_dir", type=str, default=None,
        help="Adaptadores LoRA. Padrão: models/<modelo>/<dataset>/final",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Diretório dos splits, de onde vêm os rótulos",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Entra no modo interativo após o mini-teste",
    )
    parser.add_argument(
        "--skip_tests", action="store_true", help="Pula o mini-teste automático",
    )
    args = parser.parse_args()

    spec = get_model(args.model)
    model_dir = resolve_path(args.model_dir or f"models/{spec.key}/{args.dataset}/final")
    data_dir = resolve_path(args.data_dir or get_dataset(args.dataset).resolved_output_dir)

    if not model_dir.exists():
        parser.error(
            f"Adaptadores não encontrados em {model_dir}. "
            f"Treine antes com: uv run python scripts/02_train.py "
            f"--model {spec.key} --dataset {args.dataset}"
        )

    labels = labels_of(load_metadata(data_dir))
    print(f"Modelo:  {spec.label}")
    print(f"Rótulos: {', '.join(sorted(labels))}\n")

    from tcc_triagem.inference import Classifier

    classifier = Classifier.load(spec, labels, model_dir=str(model_dir))
    print("Modelo carregado.\n")

    if not args.skip_tests:
        # Só testa os casos cujo rótulo esperado existe neste dataset.
        casos = [c for c in CASOS_DE_TESTE if c["esperado"] in labels]
        if casos:
            run_tests(classifier, casos)
        else:
            print("Nenhum caso de teste corresponde aos rótulos deste dataset.")

    if args.interactive:
        run_interactive(classifier)


if __name__ == "__main__":
    main()

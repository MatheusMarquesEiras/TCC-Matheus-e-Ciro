"""Pipeline de triagem médica em PT-BR por roteamento de especialidade.

O pacote é dividido em camadas independentes para que trocar o modelo base
ou o dataset não exija reescrever o pipeline:

    paths            resolução de caminhos a partir da raiz do repositório
    prompting        montagem do system prompt e do formato ChatML
    model_registry   catálogo de modelos (Qwen3.5, Gemma 3) e seus hiperparâmetros
    dataset_registry catálogo de datasets (MedPT e variantes)
    data_sources     leitura das fontes brutas (HuggingFace ou arquivo local)
    data_prep        filtragem, split estratificado e escrita dos splits
    modeling         carregamento via Unsloth e anexo dos adaptadores LoRA
    training         rotina QLoRA compartilhada por todos os modelos
    inference        classificador sobre um modelo treinado
    evaluation       métricas quantitativas e rubricas HealthBench-style
    reporting        comparativo entre execuções de modelos diferentes

Os scripts em ``scripts/`` são apenas casca de linha de comando sobre estas camadas.
"""

__version__ = "0.2.0"

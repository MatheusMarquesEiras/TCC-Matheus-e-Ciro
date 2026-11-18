# Triagem médica automatizada em português brasileiro

Fine-tuning QLoRA de modelos de linguagem pequenos para **roteamento de especialidade médica**: o paciente escreve a queixa em texto livre e o modelo aponta a qual especialista procurar.

**Autores:** Matheus Marques Eiras e Ciro Guilherme Nass
**Orientador:** João Paulo Orlando
**Instituição:** Instituto Federal do Paraná — Câmpus Pinhais
**Testado em:** Windows 11 Pro, RTX 4070 12 GB, CUDA 12.8

---

## Sumário

1. [Visão geral](#1-visão-geral)
2. [Arquitetura do projeto](#2-arquitetura-do-projeto)
3. [Modelos suportados](#3-modelos-suportados)
4. [Datasets suportados](#4-datasets-suportados)
5. [Instalação (Windows)](#5-instalação-windows)
6. [Executando o pipeline](#6-executando-o-pipeline)
7. [Avaliação qualitativa com Ollama](#7-avaliação-qualitativa-com-ollama)
8. [Estendendo o pipeline](#8-estendendo-o-pipeline)
9. [Resultados](#9-resultados)
10. [Problemas conhecidos](#10-problemas-conhecidos)
11. [Referências](#11-referências)

---

## 1. Visão geral

A tarefa é **classificação fechada de especialidade médica a partir de texto livre em PT-BR**. O trabalho investiga se um modelo pequeno, rodando em placa de vídeo doméstica, dá conta dela.

O repositório é um pipeline multi-modelo e multi-dataset: o mesmo esquema de fine-tuning QLoRA roda sobre quatro modelos base de famílias e tamanhos diferentes, sobre qualquer dataset registrado, e produz um comparativo único ao final.

| Componente | Escolha |
|---|---|
| Modelos base | Qwen3.5 (0.8B, 2B) e Gemma 3 (1B, 4B) |
| Dataset principal | `AKCIT/MedPT` — 384.095 pares Q&A médicos em PT-BR, 104 especialidades |
| Método | QLoRA: base congelada em NF4 4-bit + adaptadores LoRA (r=16, α=32) |
| Frameworks | Unsloth + TRL `SFTTrainer`, PEFT, bitsandbytes |
| Avaliação quantitativa | F1 macro, precision, recall e acurácia; zero-shot, 3-shot, 5-shot e fine-tuned |
| Avaliação qualitativa | Rubricas estilo HealthBench com LLM-grader (Ollama local, Ollama Cloud ou Anthropic) |
| Gerenciador | `uv`, com lock file |

**Por que F1 macro é a métrica principal:** as classes são desbalanceadas — no MedPT top-10 a maior tem quase seis vezes a menor. A acurácia sozinha esconde o desempenho nas classes minoritárias, que são justamente as de saúde mental, onde está o gargalo do modelo.

---

## 2. Arquitetura do projeto

A lógica vive em um pacote Python; os scripts são apenas casca de linha de comando sobre ele. Trocar de modelo ou de dataset não exige reescrever o pipeline.

```
TCC-Matheus-e-Ciro/
├── src/tcc_triagem/
│   ├── paths.py             # caminhos relativos à raiz e diretório por (modelo, dataset)
│   ├── console.py           # saída em UTF-8 (o console cp1252 do Windows derruba os scripts)
│   ├── prompting.py         # system prompt, formato ChatML e casamento de rótulo
│   ├── model_registry.py    # catálogo de modelos e hiperparâmetros
│   ├── dataset_registry.py  # catálogo de datasets
│   ├── data_sources.py      # leitura da fonte bruta (HuggingFace ou arquivo local)
│   ├── data_prep.py         # filtragem, split estratificado, ChatML e metadata
│   ├── modeling.py          # carregamento Unsloth e anexo dos adaptadores LoRA
│   ├── training.py          # rotina QLoRA compartilhada
│   ├── inference.py         # classificador sobre um modelo treinado
│   ├── reporting.py         # comparativo entre execuções
│   └── evaluation/
│       ├── metrics.py       # métricas, matriz de confusão, relatório
│       ├── graders.py       # Ollama local, Ollama Cloud e Anthropic
│       ├── rubrics.py       # rubrica HealthBench
│       ├── qualitative.py   # avaliação por LLM-grader
│       └── quantitative.py  # zero-shot, few-shot e fine-tuned
├── scripts/
│   ├── 01_prepare_data.py       # prepara os splits de um dataset registrado
│   ├── 02_train.py              # treina qualquer modelo (--model)
│   ├── 03_evaluate.py           # avaliação quantitativa
│   ├── 04_qualitative_eval.py   # avaliação qualitativa por LLM-grader
│   ├── 05_demo.py               # demo interativa
│   ├── 06_compare_models.py     # comparativo entre todos os modelos
│   ├── train/                   # um script de treino por modelo
│   │   ├── train_qwen35_0_8b.py
│   │   ├── train_qwen35_2b.py
│   │   ├── train_gemma3_1b.py
│   │   └── train_gemma3_4b.py
│   └── v1/                      # scripts do Experimento 01, preservados
├── data/     <modelo-agnóstico>   splits gerados pelo script 01
├── models/   <modelo>/<dataset>/  checkpoints e adaptadores
└── results/  <modelo>/<dataset>/  métricas, matrizes e escores
```

Os artefatos seguem a convenção `<modelo>/<dataset>`, o que mantém cada execução isolada e permite comparar todas de uma vez no script 06. `data/`, `models/` e `results/` produzidos localmente não vão para o Git.

### O contrato entre as etapas

O `metadata.json` gerado pelo script 01 carrega os rótulos, o system prompt e a distribuição das classes. Treino, avaliação e demo leem dele — nenhum deles precisa saber de qual dataset veio.

### Fine-tuning — QLoRA

| Parâmetro | Valor |
|---|---|
| Quantização da base | NF4 4-bit + double quantization |
| LoRA rank / alpha | 16 / 32 |
| LoRA dropout | 0 (obrigatório para o Unsloth aplicar os fast kernels) |
| Módulos alvo | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Épocas | 3 |
| Batch efetivo | 32 em todos os modelos |
| Learning rate | 2e-4 (1e-4 no Gemma 3 4B), scheduler cosseno |
| Warmup | 3% dos passos |
| Otimizador | adamw_8bit |

O **batch efetivo é fixo em 32** para todos os modelos: o que varia é a divisão entre batch por dispositivo e acumulação de gradiente, conforme a VRAM que cada um consome. Assim a dinâmica de otimização se mantém comparável entre os quatro.

---

## 3. Modelos suportados

```powershell
uv run python scripts/02_train.py --list
```

| Chave | Modelo | Repositório | Batch × acum. | LR | Observação |
|---|---|---|---|---|---|
| `qwen35-0.8b` | Qwen3.5-0.8B | `Qwen/Qwen3.5-0.8B` | 4 × 8 | 2e-4 | Linha de base do trabalho (Experimento 02) |
| `qwen35-2b` | Qwen3.5-2B | `Qwen/Qwen3.5-2B` | 2 × 16 | 2e-4 | Mesma família, maior |
| `gemma3-1b` | Gemma 3 1B (base) | `unsloth/gemma-3-1b-pt` | 4 × 8 | 2e-4 | Checkpoint pré-treinado, sem instruction tuning |
| `gemma3-4b` | Gemma 3 4B (base) | `unsloth/gemma-3-4b-pt` | 1 × 32 | 1e-4 | Multimodal; camadas de visão congeladas |
| `gemma3-1b-it` | Gemma 3 1B (instruct) | `unsloth/gemma-3-1b-it` | 4 × 8 | 2e-4 | Para o contraste base × instruct |
| `gemma3-4b-it` | Gemma 3 4B (instruct) | `unsloth/gemma-3-4b-it` | 1 × 32 | 1e-4 | Para o contraste base × instruct |

Três diferenças entre as famílias são tratadas pelo pipeline, não pelo usuário:

- **Qwen3.5 exige non-thinking mode.** O 0.8B entra em loop infinito no thinking mode. O `enable_thinking=False` é aplicado automaticamente, com fallback para um bloco `<think></think>` vazio quando o template ignora o argumento — e só na inferência, nunca no texto de treino.
- **Os checkpoints base do Gemma 3 (`-pt`) não trazem chat template.** O template `gemma-3` é aplicado antes de qualquer formatação.
- **Gemma 3 4B é multimodal.** Carrega com `FastModel`, mantém as camadas de visão congeladas e treina LoRA só nas de linguagem.

---

## 4. Datasets suportados

```powershell
uv run python scripts/01_prepare_data.py --list
```

| Chave | Descrição | Classes | Saída |
|---|---|---|---|
| `medpt-top10` | MedPT, 10 especialidades mais frequentes | 10 | `data/top10` |
| `medpt-top10-merged` | Idem, com `Psicólogo` e `Psicólogo, Psicanalista` fundidos | 9 | `data/top10_merged` |
| `medpt-top10-500` | Idem, com 500 exemplos por classe (piloto) | 10 | `data/top10_500` |
| `medpt-top20` | MedPT, 20 especialidades mais frequentes | 20 | `data/top20` |

A variante `medpt-top10-merged` responde à limitação registrada no Experimento 02: `Psicólogo, Psicanalista` ficou com F1 = 0 por sobreposição semântica quase total com `Psicólogo`, e a distinção raramente muda a conduta de triagem.

---

## 5. Instalação (Windows)

### Requisitos

- GPU NVIDIA com CUDA 12.x e ≥ 8 GB de VRAM (o Gemma 3 4B pede os 12 GB)
- ≥ 16 GB de RAM e ≥ 30 GB de disco
- Python 3.12 (gerenciado pelo `uv`)
- [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads)
- [Visual Studio 2022 Community](https://visualstudio.microsoft.com/vs/community/) com o workload "Desenvolvimento para desktop com C++"
- [`uv`](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) — para a avaliação qualitativa

### 5.1 Instalar o `uv`

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Reinicie o terminal em seguida.

### 5.2 Clonar e criar o ambiente

```powershell
git clone https://github.com/MatheusMarquesEiras/TCC-Matheus-e-Ciro.git
cd TCC-Matheus-e-Ciro
uv sync
```

O PyTorch vem do índice CUDA oficial (`download.pytorch.org/whl/cu128`), já configurado no `pyproject.toml`.

### 5.3 Instalar `causal-conv1d` (fast path do Qwen3.5)

Não há wheel para Windows — precisa ser compilado com CUDA + MSVC. **Na ordem exata:**

```powershell
$vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
$output = cmd /c "`"$vcvars`" x64 && set" 2>&1
$envVars = $output | Where-Object { $_ -match "^[A-Za-z_][A-Za-z0-9_]*=" }
foreach ($line in $envVars) {
    $parts = $line -split "=", 2
    [System.Environment]::SetEnvironmentVariable($parts[0], $parts[1], "Process")
}
$env:CL = "/Zm200"
$env:DISTUTILS_USE_SDK = "1"
$env:MSSDK = "1"
```

Ainda no mesmo terminal:

```powershell
uv add causal-conv1d
```

A compilação leva cerca de 7 minutos. Sem esse pacote, as camadas GatedDeltaNet do Qwen3.5 caem no fallback PyTorch (~7,5 s/passo em vez de ~2,5 s).

### 5.4 Corrigir o encoding do TRL

```powershell
$file = ".venv\Lib\site-packages\trl\chat_template_utils.py"
(Get-Content $file -Raw) -replace '\.read_text\(\)', '.read_text(encoding="utf-8")' | Set-Content $file -Encoding utf8
```

> Precisa ser reaplicado depois de cada `uv sync` ou atualização do TRL.

### 5.5 Verificar

```powershell
uv run python -c "
import torch
print('CUDA disponível:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
from causal_conv1d import causal_conv1d_fn
print('causal_conv1d OK')
"
```

---

## 6. Executando o pipeline

### Script 01 — Preparar o dataset

```powershell
uv run python scripts/01_prepare_data.py --dataset medpt-top10
```

| Flag | Padrão | Descrição |
|---|---|---|
| `--dataset` | `medpt-top10` | Chave do dataset no registro |
| `--output_dir` | do registro | Sobrescreve o diretório de saída |
| `--top_n` | do registro | Sobrescreve o número de classes |
| `--max_per_class` | do registro | Sobrescreve o cap por classe |
| `--list` | — | Lista os datasets registrados |

Gera `train.json`, `validation.json`, `test.json` (ChatML) e `metadata.json`. Split estratificado 80/10/10; classes com menos de 3 exemplos são descartadas, porque quebrariam a estratificação.

### Script 02 — Treinar

Um comando por modelo:

```powershell
uv run python scripts/train/train_qwen35_0_8b.py --dataset medpt-top10
uv run python scripts/train/train_qwen35_2b.py   --dataset medpt-top10
uv run python scripts/train/train_gemma3_1b.py   --dataset medpt-top10
uv run python scripts/train/train_gemma3_4b.py   --dataset medpt-top10
```

Ou a entrada genérica, útil para varreduras:

```powershell
uv run python scripts/02_train.py --model gemma3-4b --dataset medpt-top10-merged
```

| Flag | Padrão | Descrição |
|---|---|---|
| `--dataset` | `medpt-top10` | Dataset preparado pelo script 01 |
| `--data_dir` | — | Caminho explícito dos splits, ignorando `--dataset` |
| `--output_dir` | `models/<modelo>/<dataset>` | Checkpoints e modelo final |
| `--epochs` | 3 | Sobrescreve as épocas |
| `--report_to` | `none` | `none`, `wandb` ou `tensorboard` |
| `--no_resume` | — | Ignora checkpoints e treina do zero |

Checkpoints a cada 500 passos. **Se o treino for interrompido, basta repetir o mesmo comando** — o script detecta o checkpoint mais recente e retoma. Ao final grava os adaptadores em `final/` (~30 MB) e um `training_summary.json` com hiperparâmetros, parâmetros treináveis, duração, loss final e pico de VRAM.

### Script 03 — Avaliação quantitativa

```powershell
uv run python scripts/03_evaluate.py --model qwen35-0.8b --dataset medpt-top10
```

Avalia zero-shot, 3-shot, 5-shot e fine-tuned no mesmo conjunto de teste. Use `--setups finetuned` para pular os demais e `--max_test 200` para uma amostra.

Saída em `results/<modelo>/<dataset>/evaluation/`: `results.json`, `confusion_matrix_*.png`, `report_*.txt` e `finetuned_predictions.json` (entrada do script 04).

### Script 05 — Demo

```powershell
uv run python scripts/05_demo.py --model qwen35-0.8b --dataset medpt-top10 --interactive
```

### Script 06 — Comparativo

```powershell
uv run python scripts/06_compare_models.py
```

Varre todas as execuções e gera `results/comparison/comparativo.md` (tabelas prontas para o texto do TCC), `comparison.json` e o gráfico `f1_macro_por_modelo.png`. Execuções incompletas aparecem com os campos que faltam em branco — dá para rodar enquanto os treinos ainda estão em andamento.

---

## 7. Avaliação qualitativa com Ollama

A rubrica segue o formato HealthBench: cinco critérios ponderados somando 25 pontos positivos (acurácia, relevância clínica, awareness de gravidade, instruction following) mais uma penalidade por subtriagem de emergência.

### Ollama local

```powershell
ollama pull llama3.1:8b
ollama serve

uv run python scripts/04_qualitative_eval.py --model qwen35-0.8b --grader ollama
```

### Ollama Cloud

Modelos grandes demais para a GPU local, pela mesma API HTTP. Crie a chave em [ollama.com/settings/keys](https://ollama.com/settings/keys):

```powershell
$env:OLLAMA_API_KEY = "sua-chave"

uv run python scripts/04_qualitative_eval.py `
    --model qwen35-0.8b `
    --grader ollama-cloud `
    --grader_model gpt-oss:120b `
    --min_urgent 15
```

### Anthropic

```powershell
uv run python scripts/04_qualitative_eval.py --model qwen35-0.8b --grader anthropic --api_key sk-ant-...
```

| Flag | Padrão | Descrição |
|---|---|---|
| `--grader` | `ollama` | `ollama`, `ollama-cloud` ou `anthropic` |
| `--grader_model` | por backend | `llama3.1:8b`, `gpt-oss:120b` ou `claude-sonnet-4-6` |
| `--ollama_url` | por backend | `http://localhost:11434` ou `https://ollama.com` |
| `--api_key` | `$OLLAMA_API_KEY` / `$ANTHROPIC_API_KEY` | Chave do backend |
| `--n_samples` | 100 | Tamanho da amostra |
| `--min_urgent` | 0 | Cota mínima de casos com red flag na amostra |
| `--timeout` | 180 | Timeout por requisição, em segundos |

**Sobre `--min_urgent`:** a rodada do Experimento 02 pegou apenas 1 caso urgente em 100, o que não sustenta nenhuma conclusão sobre red flags. A cota força a presença desses casos **trocando** exemplos não urgentes por urgentes, mantendo o tamanho da amostra. Os escores por tema continuam separados, então a cota não contamina o escore dos casos rotineiros.

> A redação da penalidade de subtriagem em `evaluation/rubrics.py` é resultado de uma recalibração — ver [seção 9](#avaliação-qualitativa). Alterá-la invalida a comparação com o resultado publicado.

---

## 8. Estendendo o pipeline

### Adicionar um dataset

Acrescente uma entrada em `src/tcc_triagem/dataset_registry.py`. Nada mais precisa mudar.

```python
"meu-corpus": DatasetSpec(
    key="meu-corpus",
    label="Queixas anotadas do ambulatório",
    source="local",                    # ou "huggingface"
    path="data/raw/queixas.csv",       # ou o repo id no Hub
    text_column="queixa",
    label_column="especialidade",
    label_separator=None,              # ";" se houver rótulos múltiplos
    label_aliases={"Clínico": "Clínico geral"},   # fusão de classes
    drop_labels=("Indefinido",),
    top_n=None,                        # None = todas as classes
    min_per_class=20,
    output_dir="data/meu_corpus",
),
```

Fontes locais aceitam `.csv`, `.tsv`, `.json`, `.jsonl` e `.parquet`. Para experimentos pontuais, `dataset_registry.register()` faz o mesmo em tempo de execução, sem editar o arquivo.

### Adicionar um modelo

Acrescente uma entrada em `src/tcc_triagem/model_registry.py`:

```python
"meu-modelo": ModelSpec(
    key="meu-modelo",
    hf_id="organizacao/modelo",
    label="Modelo X 3B",
    family="familia",
    params="~3,0 B",
    variant="instruct",                # ou "base"
    loader="language",                 # "multimodal" se for VL
    chat_template=None,                # "gemma-3" se o checkpoint não trouxer
    chat_template_kwargs={},           # {"enable_thinking": False} no Qwen3.5
    training=TrainingSpec(batch_size=2, grad_accum=16, learning_rate=2e-4),
),
```

Para dar a ele um script próprio, copie um dos arquivos de `scripts/train/` e troque o `MODEL_KEY`.

---

## 9. Resultados

### Experimento 01 — piloto (maio/2026)

500 exemplos por classe (4.000 treino), 3 épocas, ~17 min na RTX 4070.

| Setup | Accuracy | F1 macro |
|---|---|---|
| Zero-shot | 22,4% | 0,190 |
| Three-shot | 15,0% | 0,186 |
| Five-shot | 14,8% | 0,175 |
| **Fine-tuned** | **66,8%** | **0,630** |

O few-shot fica **abaixo** do zero-shot — comportamento esperado em modelos de 0,8 B, que têm pouca capacidade de aprendizado em contexto. Gargalo: as especialidades de saúde mental (`Psicólogo` com F1 = 0). Hipótese levantada: falta de dados, não falha do método.

Relatório: [`results/v1/experimento_01_relatorio.md`](results/v1/experimento_01_relatorio.md)

### Experimento 02 — dataset completo (maio/2026)

Qwen3.5-0.8B sobre o top-10 completo: 145.321 treino / 18.165 validação / 18.166 teste. 3 épocas, 13.626 passos, ~6h30 de treino.

| Métrica | Valor |
|---|---|
| Accuracy | 0,850 |
| **F1 macro** | **0,7911** |
| Precision macro | 0,8005 |
| Recall macro | 0,7924 |

| Especialidade | F1 |
|---|---|
| Oftalmologista | 1,0000 |
| Dentista | 0,9730 |
| Dermatologista | 0,9444 |
| Ginecologista | 0,9388 |
| Ortopedista - traumatologista | 0,9286 |
| Otorrino | 0,8889 |
| Psicólogo | 0,8507 |
| Urologista | 0,7778 |
| Psiquiatra | 0,6087 |
| Psicólogo, Psicanalista | 0,0000 |

A hipótese do piloto se confirmou — o gargalo era volume de dados:

| Métrica | Piloto | Completo | Δ |
|---|---|---|---|
| Accuracy | 0,668 | 0,850 | +18,2 p.p. |
| F1 macro | 0,630 | 0,791 | +0,161 |
| F1 Psicólogo | 0,000 | 0,851 | resolvido |
| F1 Psiquiatra | 0,148 | 0,609 | +0,461 |

Curva de loss saudável, sem sobreajuste: a validação caiu de 0,6011 (fim da época 1) para 0,5129 (fim da época 2).

Relatório: [`results/experimento_02_relatorio.md`](results/experimento_02_relatorio.md)

### Avaliação qualitativa

Grader `llama3.1:8b` local via Ollama, n=100:

| Categoria | Escore | n |
|---|---|---|
| **Geral** | **0,9512** | 100 |
| Casos rotineiros | 0,9656 | 72 |
| Saúde mental | 0,9111 | 27 |
| Casos urgentes | 1,0000 | 1 (não conclusivo) |

Houve uma recalibração importante: no teste de fumaça inicial o escore ficou em 0,344 porque o grader aplicava a penalidade de subtriagem de emergência (−8) mesmo em casos claramente rotineiros. O critério foi restringido a queixas com sinal de risco de vida explícito, e o escore subiu para 0,9512.

### Limitação em aberto

`Psicólogo, Psicanalista` continua com F1 = 0, absorvida por `Psicólogo`. O padrão de erro inverteu em relação ao piloto: antes o modelo colapsava tudo *para* essa classe, agora colapsa essa classe *para* `Psicólogo`. Causa: sobreposição semântica somada ao desbalanceamento (45.690 exemplos contra 7.794). O dataset `medpt-top10-merged` existe para testar a fusão das duas.

---

## 10. Problemas conhecidos

### 10.1 `causal-conv1d` não compila

```
FileNotFoundError: [WinError 2]
RuntimeError: Error compiling objects for extension
UserWarning: DISTUTILS_USE_SDK is not set
```

Não há wheel para Windows e o ambiente MSVC não está inicializado no processo de build. Use `vcvarsall.bat` antes do `uv add` — ver [5.3](#53-instalar-causal-conv1d-fast-path-do-qwen35).

- `FileNotFoundError` → `cl.exe` fora do PATH
- `C1060: espaço de heap insuficiente` → `$env:CL = "/Zm200"`
- `DISTUTILS_USE_SDK is not set` → `$env:DISTUTILS_USE_SDK = "1"`

### 10.2 `UnicodeDecodeError` no TRL

`trl/chat_template_utils.py` usa `Path.read_text()` sem encoding, e o Windows assume cp1252. Ver [5.4](#54-corrigir-o-encoding-do-trl). **Reaplicar após cada `uv sync`.**

### 10.3 Primeiros passos lentos

O Triton compila e cacheia os kernels CUDA na primeira execução. Passa sozinho.

### 10.4 Colunas do MedPT

Estão em snake_case: `question` e `medical_specialty`. Especialidades múltiplas usam `;` como separador e o pipeline extrai só a primeira. **Não dividir por vírgula** — nomes como `Ortopedista - traumatologista` são um único rótulo.

### 10.5 Non-thinking mode no Qwen3.5

O 0.8B entra em loop infinito no thinking mode. O pipeline aplica `enable_thinking=False` automaticamente; não remova.

### 10.6 Console do Windows em cp1252

Os scripts forçam stdout e stderr em UTF-8 via `tcc_triagem.console`. Sem isso, imprimir acentos ou os símbolos de progresso derruba a avaliação no meio — depois de dezenas de chamadas ao grader.

### 10.7 Treino fica lento em processadores híbridos

O passo de treino é limitado por uma única thread do processador. Em processadores com núcleos rápidos e lentos, o Windows pode mover o treino para os lentos quando ele fica em segundo plano. No Gemma 3 1B (i7-13620H, RTX 4060 Laptop), o ritmo saltou de ~38 para ~61 s/passo ao desbloquear o notebook, com a VRAM estável; limitar o processo aos núcleos rápidos voltou a ~40 s/passo, sem interromper o treino:

```powershell
(Get-Process python | Sort-Object CPU -Descending | Select-Object -First 1).ProcessorAffinity = 0x0FFF
```

A máscara `0x0FFF` (núcleos lógicos 0 a 11) vale para o i7-13620H; outros processadores precisam de outra. O limite dura só enquanto o processo existir. O [scripts/train_gemma3_1b_notebook.sh](scripts/train_gemma3_1b_notebook.sh) já o aplica sozinho a cada início ou retomada nesse processador; com outro comando de treino, aplique à mão.

---

## 11. Referências

- **MedPT:** FÄRBER, F. B. et al. *MedPT: a massive medical question answering dataset for Brazilian-Portuguese speakers.* arXiv:2511.11878, 2026.
- **QLoRA:** DETTMERS, T. et al. *QLoRA: efficient finetuning of quantized LLMs.* arXiv:2305.14314, 2023.
- **LoRA:** HU, E. J. et al. *LoRA: low-rank adaptation of large language models.* arXiv:2106.09685, 2021.
- **DrBode:** PAIOLA, P. H. et al. *Adapting LLMs for the medical domain in Portuguese.* arXiv:2410.00163, 2024.
- **HealthBench:** OpenAI. *HealthBench: evaluating large language models towards improved human health.* `github.com/openai/simple-evals`
- **NLP em triagem:** STEWART, J. et al. *Applications of natural language processing at emergency department triage: a narrative review.* PLOS ONE, v. 18, n. 12, e0279953, 2023.
- **BERT em triagem:** WANG, B. et al. *A disease-prediction protocol integrating triage priority and BERT-based transfer learning for intelligent triage.* Bioengineering, v. 10, n. 4, art. 420, 2023.

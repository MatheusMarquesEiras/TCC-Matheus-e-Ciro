"""Exporta os modelos de triagem para GGUF, para a demo que roda no navegador.

Para cada variante, soma os adaptadores LoRA direto nos pesos do checkpoint
original (``W += (alpha / r) · B · A``, a mesma conta do ``merge_and_unload`` do
PEFT) e converte o resultado com o ``convert_hf_to_gguf.py`` do llama.cpp.

A mesclagem é feita sobre os tensores do checkpoint, e não via
``save_pretrained``, de propósito: assim os nomes dos tensores ficam idênticos
aos do repositório oficial, que é o formato que o conversor do llama.cpp
reconhece. Nenhum par LoRA pode ficar sem destino — se ficar, o script aborta
em vez de gerar um GGUF silenciosamente igual ao modelo base.

Use o llama.cpp no mesmo commit do runtime que vai rodar o GGUF: o conversor
de arquiteturas novas como a do Qwen3.5 muda entre versões. A demo usa o
wllama 3.6.1, que embute o llama.cpp b10663 (commit ``83d855c`` — veja
``LIBLLAMA_VERSION`` em ``space/wllama/index.js``).

Exemplo:
    uv run python scripts/07_export_gguf.py --llama_cpp ../llama.cpp \\
        --variant completo=matheus-ciro-tcc/qwen3.5-0.8b-triagem-medica-ptbr \\
        --variant piloto=matheus-ciro-tcc/qwen35-0.8b-triagem-medica-piloto \\
        --variant base

O modelo base de cada variante vem do ``base_model_name_or_path`` do
adaptador; checkpoints quantizados do Unsloth (``-unsloth-bnb-4bit``) são
trocados pela versão em precisão cheia, porque a soma tem de ser feita nos
pesos originais. ``--base`` força um base para todas as variantes. O
adaptador pode ser um repositório do Hugging Face ou uma pasta local.
"""

import _bootstrap  # noqa: F401  (efeito colateral: ajusta o sys.path)

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

from tcc_triagem.paths import resolve_path

BASE_ID = "Qwen/Qwen3.5-0.8B"
PEFT_PREFIX = "base_model.model."


def adapter_file(adapter: str, name: str) -> str:
    """Arquivo do adaptador, de uma pasta local ou de um repositório do Hub."""
    local = Path(adapter) / name
    return str(local) if local.exists() else hf_hub_download(adapter, name)


def base_of(adapter: str) -> str:
    """Checkpoint base em precisão cheia do adaptador."""
    base = json.loads(Path(adapter_file(adapter, "adapter_config.json")).read_text())[
        "base_model_name_or_path"
    ]
    return base.removesuffix("-unsloth-bnb-4bit").removesuffix("-bnb-4bit")


def canonical(name: str) -> str:
    """Nome do tensor sem os prefixos que mudam entre versões do transformers.

    O mesmo peso aparece como ``model.language_model.layers.0...`` no adaptador
    e como ``language_model.model.layers.0...`` em checkpoints multimodais mais
    antigos (Gemma 3 4B); sem os prefixos, os dois viram ``layers.0...``.
    """
    return ".".join(p for p in name.split(".") if p not in ("model", "language_model"))


def nf4_weights(repo: str) -> dict[str, torch.Tensor]:
    """Pesos lineares do base como o treino QLoRA os viu: NF4 desquantizado.

    O LoRA foi treinado sobre o base quantizado em 4 bits; somá-lo aos pesos
    em precisão cheia muda um pouco o modelo. Carrega ``repo`` em 4 bits (o
    mesmo checkpoint ``-unsloth-bnb-4bit`` do treino, quando existir) e
    devolve cada camada quantizada já desquantizada, pelo nome canônico.
    """
    import bitsandbytes.functional as bnbf
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(repo, quantization_config=qcfg, device_map="cuda",
                                                 torch_dtype=torch.bfloat16)
    out = {}
    for name, mod in model.named_modules():
        w = getattr(mod, "weight", None)
        if w is not None and getattr(w, "quant_state", None) is not None:
            out[canonical(f"{name}.weight")] = bnbf.dequantize_4bit(w.data, w.quant_state).to("cpu", torch.bfloat16)
    del model
    torch.cuda.empty_cache()
    return out


def merge_lora(base_dir: Path, adapter_repo: str, out_dir: Path,
               nf4: dict[str, torch.Tensor] | None = None) -> int:
    """Copia o checkpoint base para ``out_dir`` com o LoRA somado aos pesos."""
    config = json.loads(Path(adapter_file(adapter_repo, "adapter_config.json")).read_text())
    if config.get("use_dora") or config.get("use_rslora"):
        raise SystemExit(f"{adapter_repo}: DoRA/rsLoRA não suportados por este script.")
    scaling = config["lora_alpha"] / config["r"]

    with safe_open(adapter_file(adapter_repo, "adapter_model.safetensors"), "pt") as f:
        lora = {k: f.get_tensor(k) for k in f.keys()}

    deltas: dict[str, torch.Tensor] = {}
    for key_a in (k for k in lora if k.endswith(".lora_A.weight")):
        module = key_a.removeprefix(PEFT_PREFIX).removesuffix(".lora_A.weight")
        a = lora[key_a].float()
        b = lora[key_a.replace(".lora_A.", ".lora_B.")].float()
        deltas[canonical(f"{module}.weight")] = (b @ a) * scaling

    shutil.copytree(
        base_dir, out_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.safetensors")
    )
    merged = 0
    for shard in sorted(base_dir.glob("*.safetensors")):
        with safe_open(shard, "pt") as f:
            metadata = f.metadata()
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        by_canon = {canonical(k): k for k in tensors}
        if nf4:
            for cname, w in nf4.items():
                name = by_canon.get(cname)
                if name is not None and tensors[name].shape == w.shape:
                    tensors[name] = w.to(tensors[name].dtype)
        for cname, delta in deltas.items():
            name = by_canon.get(cname)
            if name is not None:
                weight = tensors[name]
                if weight.shape != delta.shape:
                    raise SystemExit(f"Formato incompatível em {name}: {weight.shape} × {delta.shape}")
                tensors[name] = (weight.float() + delta).to(weight.dtype)
                merged += 1
        save_file(tensors, out_dir / shard.name, metadata=metadata)

    # Alguns repositórios (ex.: unsloth/gemma-3-1b-pt) trazem um índice de
    # shards que não existem mais; o conversor confia no índice e falha.
    index = out_dir / "model.safetensors.index.json"
    if index.exists():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        if not all((out_dir / s).exists() for s in shards):
            index.unlink()

    if merged != len(deltas):
        raise SystemExit(f"{adapter_repo}: só {merged} de {len(deltas)} pares LoRA encontraram destino.")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta os modelos de triagem para GGUF.")
    parser.add_argument("--llama_cpp", required=True, help="Checkout do llama.cpp (convert_hf_to_gguf.py)")
    parser.add_argument(
        "--variant", action="append", required=True,
        help="nome=repo_do_adaptador, ou só 'base' para o modelo sem ajuste",
    )
    parser.add_argument("--outtype", default="q8_0", help="Tipo do GGUF (padrão: q8_0)")
    parser.add_argument("--output_dir", default="models/gguf")
    parser.add_argument("--base", default=None, help="Força o checkpoint base de todas as variantes")
    parser.add_argument("--nf4_base", action="store_true",
                        help="Soma o LoRA sobre o base NF4 desquantizado, como no treino QLoRA")
    args = parser.parse_args()

    converter = Path(args.llama_cpp) / "convert_hf_to_gguf.py"
    if not converter.exists():
        parser.error(f"{converter} não existe")
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for variant in args.variant:
        name, _, adapter_repo = variant.partition("=")
        base_id = args.base or (base_of(adapter_repo) if adapter_repo else BASE_ID)
        base_dir = Path(snapshot_download(base_id))
        print(f"[{name}] base: {base_id}")
        if adapter_repo:
            model_dir = output_dir / f"merged-{name}"
            nf4 = None
            if args.nf4_base:
                raw_base = json.loads(Path(adapter_file(adapter_repo, "adapter_config.json")).read_text())[
                    "base_model_name_or_path"]
                nf4 = nf4_weights(raw_base)
                print(f"[{name}] {len(nf4)} camadas NF4 desquantizadas de {raw_base}")
            n = merge_lora(base_dir, adapter_repo, model_dir, nf4)
            print(f"[{name}] {n} pares LoRA de {adapter_repo} somados ao base")
        else:
            model_dir = base_dir
            print(f"[{name}] modelo base sem ajuste")

        outfile = output_dir / f"triagem-{name}-{args.outtype}.gguf"
        subprocess.run(
            [sys.executable, str(converter), str(model_dir),
             "--outtype", args.outtype, "--outfile", str(outfile)],
            check=True,
        )
        print(f"[{name}] -> {outfile} ({outfile.stat().st_size / 1024**2:.0f} MB)")


if __name__ == "__main__":
    main()

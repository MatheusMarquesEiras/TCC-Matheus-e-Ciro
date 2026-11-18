"""Confere se o GGUF da demo responde igual ao modelo avaliado no TCC.

Sobe o ``llama-server`` (mesmo llama.cpp do wllama da demo) com o GGUF, manda
os primeiros N exemplos do teste com o prompt exato da avaliação (chat
template do adaptador) e compara o rótulo obtido com
``finetuned_predictions.json`` da avaliação em Python (QLoRA 4-bit).

Concordância perto de 100% indica que a conversão e o prompt estão certos;
as poucas diferenças esperadas vêm da quantização (Q8_0/Q4_K_M no GGUF
contra NF4 + LoRA na avaliação).

Exemplo:
    uv run python scripts/08_gguf_paridade.py --model qwen35-2b \\
        --gguf models/gguf/triagem-qwen35-2b-final-q8_0.gguf \\
        --server ../ferramentas/llama-bin/llama-server.exe -n 200
"""

import _bootstrap  # noqa: F401

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

from tcc_triagem.dataset_registry import DATASETS
from tcc_triagem.model_registry import MODELS
from tcc_triagem.modeling import apply_chat_template
from tcc_triagem.paths import resolve_path
from tcc_triagem.prompting import build_inference_messages, match_label

DATASET = "medpt-top10-merged"
STOPS = ["<|im_end|>", "<|im_start|>", "<end_of_turn>", "<start_of_turn>"]


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--server", required=True, help="Caminho do llama-server")
    ap.add_argument("--adapter_dir", default=None, help="Pasta com o tokenizer/chat template do adaptador")
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--strip_bos", action="store_true",
                    help="Tira o <bos> do texto (o llama.cpp já põe um ao tokenizar)")
    args = ap.parse_args()

    spec = MODELS[args.model]
    ds = DATASETS[DATASET]
    adapter_dir = Path(args.adapter_dir or resolve_path(f"models/{args.model}/{DATASET}/final"))
    meta = json.loads((adapter_dir / "metadata.json").read_text(encoding="utf-8"))
    labels = meta["labels"]
    tok = AutoTokenizer.from_pretrained(adapter_dir)

    test = json.loads(resolve_path(f"{ds.output_dir or 'data/' + ds.key}/test.json").read_text(encoding="utf-8"))[: args.n]
    preds_py = json.loads(
        resolve_path(f"results/{args.model}/{DATASET}/evaluation/finetuned_predictions.json").read_text(encoding="utf-8")
    )[: args.n]
    question = lambda item: next(m["content"] for m in item["messages"] if m["role"] == "user")
    gold = [item["messages"][-1]["content"].strip() for item in test]

    prompts = [
        apply_chat_template(tok, build_inference_messages(question(it), labels), spec, add_generation_prompt=True)
        for it in test
    ]
    if args.strip_bos:
        prompts = [p.removeprefix("<bos>") for p in prompts]
    out = resolve_path(f"results/{args.model}/{DATASET}/gguf_paridade")
    out.mkdir(parents=True, exist_ok=True)
    (out / "exemplo_prompt.txt").write_text(prompts[0], encoding="utf-8")

    srv = subprocess.Popen(
        [args.server, "-m", args.gguf, "--port", str(args.port), "-ngl", str(args.ngl), "-c", "2048",
         "--no-webui", "-np", "1"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{args.port}"
    try:
        for _ in range(240):
            try:
                if json.loads(urllib.request.urlopen(base + "/health", timeout=2).read()).get("status") == "ok":
                    break
            except Exception:
                time.sleep(1)
        rows, t0 = [], time.time()
        for i, p in enumerate(prompts):
            r = post(base + "/completion", {"prompt": p, "n_predict": 20, "temperature": 0, "top_k": 1,
                                            "stop": STOPS, "cache_prompt": False})
            raw = r["content"].strip()
            rows.append({"i": i, "raw": raw, "gguf": match_label(raw, labels), "python": preds_py[i], "gold": gold[i]})
        dt = time.time() - t0
    finally:
        srv.terminate()

    igual = sum(r["gguf"] == r["python"] for r in rows)
    acc_g = sum(r["gguf"] == r["gold"] for r in rows)
    acc_p = sum(r["python"] == r["gold"] for r in rows)
    res = {"model": args.model, "gguf": Path(args.gguf).name, "n": len(rows), "concordancia": igual / len(rows),
           "acc_gguf": acc_g / len(rows), "acc_python": acc_p / len(rows), "segundos": round(dt, 1),
           "diferencas": [r for r in rows if r["gguf"] != r["python"]]}
    (out / f"{Path(args.gguf).stem}{'-sem-bos' if args.strip_bos else ''}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.model}: concordância {igual}/{len(rows)} · acc GGUF {acc_g}/{len(rows)} · "
          f"acc Python {acc_p}/{len(rows)} · {dt:.0f} s")
    for r in res["diferencas"][:10]:
        print(f"  #{r['i']}: gguf={r['gguf']!r} (raw {r['raw']!r}) python={r['python']!r} gold={r['gold']!r}")


if __name__ == "__main__":
    main()

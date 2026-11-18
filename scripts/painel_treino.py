"""Painel local, em tempo real, do treino em andamento (somente leitura).

Uso:
    python scripts/painel_treino.py [--modelo gemma3-4b] [--porta 8765]
    # abra http://127.0.0.1:8765

Le o log do treino, os checkpoints e o nvidia-smi a cada 5 s. Nao altera nada e escuta
apenas em 127.0.0.1. Usa so a biblioteca padrao.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = "medpt-top10-merged"
INTERVALO = 5          # s entre leituras
VALIDACAO_H = 3.0      # estimativa de uma validacao completa (medida no Gemma 4B, RTX 4070)

estado = {"amostras": deque(maxlen=720), "gpu": deque(maxlen=720), "proc": {"treino": False, "ts": 0.0}}
cache = {"mtime": None, "hist": None}
trava = threading.Lock()


def run(cmd: list[str], timeout: int = 8) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              encoding="utf-8", errors="replace").stdout
    except Exception:
        return ""


def ler_gpu() -> dict:
    out = run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu",
               "--format=csv,noheader,nounits"]).strip().splitlines()
    if not out:
        return {}
    n, usada, total, util, pot, temp = [x.strip() for x in out[0].split(",")]
    f = lambda v: float(v) if v.replace(".", "", 1).isdigit() else None
    return {"nome": n, "mem_usada_mb": f(usada), "mem_total_mb": f(total), "util": f(util),
            "watts": f(pot), "temp": f(temp)}


def treino_rodando() -> bool:
    agora = time.time()
    if agora - estado["proc"]["ts"] < 20:
        return estado["proc"]["treino"]
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and "
          "$_.CommandLine -match '02_train|03_evaluate|04_qualitative' } | "
          "ForEach-Object { $_.CommandLine } | Select-Object -First 1")
    saida = run(["powershell", "-NoProfile", "-Command", ps], timeout=15).strip()
    estado["proc"] = {"treino": bool(saida), "ts": agora, "cmd": saida[:160]}
    return bool(saida)


def ler_log(caminho: Path, total_passos: int) -> dict:
    if not caminho.exists():
        return {}
    with open(caminho, "rb") as f:
        f.seek(0, 2)
        tam = f.tell()
        f.seek(max(0, tam - 12000))
        txt = f.read().decode("utf-8", "ignore").replace("\r", "\n")
    r = {"mtime": caminho.stat().st_mtime}
    ult_treino = ult_val = None
    for m in re.finditer(r"(\d+)/(\d+) \[([^\]]*)\]", txt):
        passo, tot = int(m.group(1)), int(m.group(2))
        if tot == total_passos:
            ult_treino = (m.start(), passo, m.group(3))
        elif tot > total_passos // 4 and tot != total_passos:
            ult_val = (m.start(), passo, tot, m.group(3))
    if ult_val and (not ult_treino or ult_val[0] > ult_treino[0]):
        r["fase"] = "validacao"
        r["val_atual"], r["val_total"] = ult_val[1], ult_val[2]
        r["val_info"] = ult_val[3]
    elif ult_treino:
        r["fase"] = "treino"
    if ult_treino:
        r["passo"] = ult_treino[1]
        r["tqdm"] = ult_treino[2]
    ult_loss = re.findall(r"\{'loss': '([\d.]+)'[^}]*'epoch': '([\d.]+)'\}", txt)
    if ult_loss:
        r["loss"], r["epoca"] = float(ult_loss[-1][0]), float(ult_loss[-1][1])
    r["erro"] = bool(re.search(r"Traceback|out of memory|OutOfMemory", txt[-3000:]))
    return r


def historico(pasta: Path) -> dict:
    cks = sorted(pasta.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if not cks:
        return {"cks": [], "loss": [], "eval": []}
    ult = cks[-1]
    ts = ult / "trainer_state.json"
    mt = ts.stat().st_mtime if ts.exists() else None
    if cache["mtime"] != (ult.name, mt):
        loss, ev = [], []
        try:
            d = json.loads(ts.read_text(encoding="utf-8"))
            for e in d.get("log_history", []):
                if "loss" in e:
                    loss.append([e["step"], e["loss"]])
                if "eval_loss" in e:
                    ev.append([e["step"], e["eval_loss"]])
        except Exception:
            pass
        cache["mtime"] = (ult.name, mt)
        cache["hist"] = {"loss": loss, "eval": ev}
    return {"cks": [int(p.name.split("-")[1]) for p in cks[-6:]], **cache["hist"]}


def coletar(modelo: str):
    pasta = ROOT / "models" / modelo / DATASET
    log = ROOT / "logs" / f"train_{modelo}_final.log"
    total = 13617
    final = (pasta / "final").exists()
    # total de passos: le do trainer_state do ultimo checkpoint, se houver
    cks = sorted(pasta.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])) if pasta.exists() else []
    if cks:
        try:
            total = json.loads((cks[-1] / "trainer_state.json").read_text(encoding="utf-8")).get("max_steps", total)
        except Exception:
            pass
    info = ler_log(log, total)
    if "passo" in info:
        estado["passo"] = info["passo"]
    else:  # durante a validacao o log nao traz o contador de passos do treino
        info["passo"] = estado.get("passo") or (int(cks[-1].name.split("-")[1]) if cks else 0)
    info["total"] = total
    info["final"] = final
    info["rodando"] = treino_rodando()
    agora = time.time()
    with trava:
        if "passo" in info:
            estado["amostras"].append((agora, info["passo"], info.get("fase", "treino"), info.get("val_atual")))
        estado["gpu"].append((agora, ler_gpu()))
    return info, pasta


def montar(modelo: str) -> dict:
    info, pasta = estado.get("ultimo", ({}, ROOT))
    with trava:
        am = list(estado["amostras"])
        gpu = list(estado["gpu"])
    # ritmo recente (passos por segundo) so em amostras de treino, janela de 10 min
    s_por_passo = None
    treino = [a for a in am if a[2] == "treino" and a[0] > time.time() - 600]
    if len(treino) >= 2 and treino[-1][1] > treino[0][1]:
        s_por_passo = (treino[-1][0] - treino[0][0]) / (treino[-1][1] - treino[0][1])
        estado["ritmo_ult"] = s_por_passo
    total = info.get("total", 13617)
    passo = info.get("passo", 0)
    restantes = max(total - passo, 0)
    eta_h = None
    ritmo_eta = s_por_passo or estado.get("ritmo_ult") or 8.0  # na validacao: ultimo ritmo medido (ou 8 s, o tipico do Gemma 4B)
    if ritmo_eta and 0 < ritmo_eta < 120:
        epoca_passos = total // 3
        validacoes_restantes = sum(1 for k in (1, 2, 3) if passo < k * epoca_passos) if passo < total else 0
        if info.get("fase") == "validacao":
            validacoes_restantes -= 1  # a atual e tratada abaixo
        eta_h = restantes * ritmo_eta / 3600 + validacoes_restantes * VALIDACAO_H
        if info.get("fase") == "validacao" and info.get("val_total"):
            frac = 1 - info["val_atual"] / info["val_total"]
            eta_h += frac * VALIDACAO_H
    ultima = gpu[-1][1] if gpu else {}
    return {
        "modelo": modelo, "agora": time.time(), "info": info, "s_por_passo": s_por_passo, "eta_h": eta_h,
        "historico": historico(pasta) if pasta.exists() else {},
        "gpu": ultima,
        "gpu_serie": [[g[0], g[1].get("mem_usada_mb"), g[1].get("watts"), g[1].get("util")] for g in gpu[-180:]],
        "ritmo_serie": [[a[0], a[1], a[2]] for a in am[-180:]],
        "cmd": estado["proc"].get("cmd", ""),
    }


def loop_coleta(modelo: str):
    while True:
        try:
            estado["ultimo"] = coletar(modelo)
        except Exception as e:  # nunca derruba o painel
            estado["ultimo"] = ({"erro_painel": str(e)}, ROOT)
        time.sleep(INTERVALO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="gemma3-4b")
    ap.add_argument("--porta", type=int, default=8765)
    a = ap.parse_args()
    threading.Thread(target=loop_coleta, args=(a.modelo,), daemon=True).start()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path.startswith("/api/status"):
                corpo = json.dumps(montar(a.modelo), ensure_ascii=False).encode("utf-8")
                tipo = "application/json; charset=utf-8"
            else:
                corpo = (ROOT / "scripts" / "painel_treino.html").read_bytes()
                tipo = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", tipo)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(corpo)))
            self.end_headers()
            self.wfile.write(corpo)

    srv = ThreadingHTTPServer(("127.0.0.1", a.porta), H)
    print(f"Painel em http://127.0.0.1:{a.porta}  (modelo {a.modelo}) — Ctrl+C para sair", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()

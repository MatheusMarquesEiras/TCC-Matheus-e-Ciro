"""Painel local da AVALIACAO em andamento (somente leitura): predicoes no teste + juiz GLM.

Uso:
    python scripts/painel_avaliacao.py [--modelo gemma3-4b] [--porta 8766]
    # abra http://127.0.0.1:8766

Le `estado_avaliacao.json` (gravado pelo proprio script de avaliacao) e o nvidia-smi.
Estima o ritmo pela evolucao do contador. Nao altera nada; so biblioteca padrao.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = "medpt-top10-merged"
INTERVALO = 5
amostras: deque = deque(maxlen=2000)  # (ts, etapa, feito)
gpu_serie: deque = deque(maxlen=360)
ultimo: dict = {}
trava = threading.Lock()


def run(cmd, timeout=8):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              encoding="utf-8", errors="replace").stdout
    except Exception:
        return ""


def ler_gpu():
    out = run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu",
               "--format=csv,noheader,nounits"]).strip().splitlines()
    if not out:
        return {}
    try:
        u, t, ut, w, tp = [float(x) for x in out[0].split(",")]
        return {"mem": u, "mem_total": t, "util": ut, "watts": w, "temp": tp}
    except ValueError:
        return {}


def ler_estado(modelo: str):
    base = ROOT / "results" / modelo / DATASET
    cands = [base / "evaluation" / "estado_avaliacao.json", base / "qualitative_glm" / "estado_avaliacao.json"]
    cands = [c for c in cands if c.exists()]
    if not cands:
        return None
    c = max(cands, key=lambda p: p.stat().st_mtime)
    try:
        d = json.loads(c.read_text(encoding="utf-8"))
    except Exception:
        return None
    d["_arquivo"] = c.parent.name
    d["_mtime"] = c.stat().st_mtime
    return d


def pausar_existe(modelo: str) -> bool:
    return (ROOT / "PAUSAR").exists()


def coletar(modelo: str):
    est = ler_estado(modelo)
    g = ler_gpu()
    agora = time.time()
    with trava:
        gpu_serie.append((agora, g.get("mem"), g.get("watts"), g.get("util")))
        if est and est.get("total"):
            amostras.append((agora, est.get("etapa"), est.get("feito", 0)))
    return est, g


def montar(modelo: str):
    est = ultimo.get("est")
    g = ultimo.get("gpu", {})
    out = {"modelo": modelo, "agora": time.time(), "estado": est, "gpu": g, "ritmo": None, "eta_s": None,
           "fim": None, "gpu_serie": [list(x) for x in list(gpu_serie)[-180:]]}
    if est and est.get("total"):
        with trava:
            am = [a for a in amostras if a[1] == est.get("etapa") and a[0] > time.time() - 900]
        if len(am) >= 2 and am[-1][2] > am[0][2]:
            ritmo = (am[-1][2] - am[0][2]) / (am[-1][0] - am[0][0])  # exemplos/s
            out["ritmo"] = ritmo
            if est.get("status") == "rodando":
                eta = (est["total"] - est["feito"]) / ritmo
                out["eta_s"] = eta
                out["fim"] = (datetime.now() + timedelta(seconds=eta)).strftime("%d/%m %H:%M")
    return out


def loop(modelo):
    while True:
        try:
            est, g = coletar(modelo)
            ultimo["est"], ultimo["gpu"] = est, g
        except Exception as e:
            ultimo["est"], ultimo["gpu"] = {"status": f"erro do painel: {e}"}, {}
        time.sleep(INTERVALO)


PAGINA = """<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Avaliação em andamento</title>
<style>
:root{--bg:#0f1115;--card:#181b22;--tx:#e8eaf0;--mut:#8b93a7;--ac:#4cc38a;--warn:#e5a23c;--bad:#e5484d;--bd:#262a34}
@media(prefers-color-scheme:light){:root{--bg:#f4f5f8;--card:#fff;--tx:#161922;--mut:#666d80;--bd:#dfe2ea}}
body{margin:0;background:var(--bg);color:var(--tx);font:15px system-ui,sans-serif;padding:20px}
h1{font-size:20px;margin:0 0 4px}.sub{color:var(--mut);margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:14px}
.k{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.04em}.v{font-size:26px;font-weight:600;margin-top:4px}
.s{color:var(--mut);font-size:13px;margin-top:2px}
.bar{height:14px;background:var(--bd);border-radius:7px;overflow:hidden;margin:10px 0 4px}.bar i{display:block;height:100%;background:var(--ac);width:0}
.wide{grid-column:1/-1}.cv{position:relative;height:160px}canvas{position:absolute;inset:0;width:100%;height:100%}
#aviso{display:none;background:#3a1f22;border:1px solid var(--bad);color:#ffb4b7;padding:10px 14px;border-radius:8px;margin-bottom:12px}
</style></head><body>
<h1>Avaliação — <span id="m"></span></h1><div class="sub" id="etapa">carregando…</div>
<div id="aviso"></div>
<div class="grid">
<div class="card wide"><div class="k">Progresso</div><div class="v"><span id="pct">–</span></div>
<div class="bar"><i id="b"></i></div><div class="s" id="cont"></div></div>
<div class="card"><div class="k">Falta (estimativa)</div><div class="v" id="eta">–</div><div class="s" id="fim"></div></div>
<div class="card"><div class="k">Ritmo</div><div class="v" id="rit">–</div><div class="s">exemplos por minuto (últimos 15 min)</div></div>
<div class="card"><div class="k">Acurácia parcial</div><div class="v" id="acc">–</div><div class="s">dos exemplos já avaliados</div></div>
<div class="card"><div class="k">GPU</div><div class="v" id="gpu">–</div><div class="s" id="gpu2"></div></div>
<div class="card wide"><div class="k">Uso da GPU (memória, potência)</div><div class="cv"><canvas id="cv"></canvas></div></div>
</div>
<script>
const $=id=>document.getElementById(id);
const fmt=s=>{if(s==null)return '–';const h=Math.floor(s/3600),m=Math.round(s%3600/60);return h?h+' h '+m+' min':m+' min'};
function chart(d){const c=$('cv'),r=c.getBoundingClientRect(),dpr=devicePixelRatio||1;c.width=r.width*dpr;c.height=r.height*dpr;
const x=c.getContext('2d');x.scale(dpr,dpr);const W=r.width,H=r.height;x.clearRect(0,0,W,H);if(d.length<2)return;
const t0=d[0][0],t1=d[d.length-1][0];const line=(i,mx,col)=>{x.beginPath();x.strokeStyle=col;x.lineWidth=2;
d.forEach((p,k)=>{const X=(p[0]-t0)/(t1-t0||1)*(W-4)+2,Y=H-4-(p[i]||0)/mx*(H-8);k?x.lineTo(X,Y):x.moveTo(X,Y)});x.stroke()};
line(1,12288,'#4c8bf5');line(2,200,'#e5a23c')}
async function tick(){try{const d=await (await fetch('/api/status',{cache:'no-store'})).json();$('m').textContent=d.modelo;
const e=d.estado;$('aviso').style.display='none';
if(!e){$('etapa').textContent='sem avaliação em andamento (estado_avaliacao.json não encontrado)'}
else{const st=e.status;$('etapa').textContent=(e.etapa||'')+' — '+st+(e.atualizado_em?' · atualizado '+e.atualizado_em:'');
const p=e.total?e.feito/e.total*100:0;$('pct').textContent=p.toFixed(1)+' %';$('b').style.width=p+'%';
$('cont').textContent=e.feito+' de '+e.total+' exemplos';
$('acc').textContent=e.acuracia_parcial!=null?(e.acuracia_parcial*100).toFixed(1)+' %':'–';
if(st&&st.startsWith('pausad')){$('aviso').style.display='block';$('aviso').textContent='Avaliação PAUSADA. Para retomar: bash scripts/run_final_eval.sh '+d.modelo}
if(st&&st.startsWith('erro')){$('aviso').style.display='block';$('aviso').textContent=st}
if(st&&(st.startsWith('conclu')))$('etapa').textContent+=' ✔'}
$('eta').textContent=fmt(d.eta_s);$('fim').textContent=d.fim?'termina por volta de '+d.fim:'';
$('rit').textContent=d.ritmo?(d.ritmo*60).toFixed(0):'–';
const g=d.gpu||{};$('gpu').textContent=g.util!=null?g.util+' %':'–';
$('gpu2').textContent=g.mem!=null?(g.mem/1024).toFixed(1)+' / '+(g.mem_total/1024).toFixed(1)+' GB · '+Math.round(g.watts)+' W · '+g.temp+' °C':'';
chart(d.gpu_serie||[])}catch(err){$('aviso').style.display='block';$('aviso').textContent='Painel sem conexão com o servidor local'}}
tick();setInterval(tick,5000)
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="gemma3-4b")
    ap.add_argument("--porta", type=int, default=8766)
    a = ap.parse_args()
    threading.Thread(target=loop, args=(a.modelo,), daemon=True).start()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path.startswith("/api/status"):
                corpo = json.dumps(montar(a.modelo), ensure_ascii=False).encode("utf-8")
                tipo = "application/json; charset=utf-8"
            else:
                corpo = PAGINA.encode("utf-8")
                tipo = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", tipo)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(corpo)))
            self.end_headers()
            self.wfile.write(corpo)

    srv = ThreadingHTTPServer(("127.0.0.1", a.porta), H)
    print(f"Painel da avaliação em http://127.0.0.1:{a.porta} (modelo {a.modelo})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()

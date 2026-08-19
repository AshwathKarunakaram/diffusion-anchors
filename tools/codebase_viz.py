#!/usr/bin/env python3
"""Live dependency-free codebase and diffusion architecture visualizer."""
import ast, hashlib, json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
HOST, PORT = os.getenv("CODEBASE_VIZ_HOST", "127.0.0.1"), int(os.getenv("CODEBASE_VIZ_PORT", "8765"))
PIPELINE = [
 ["generate_trajectories.py", "Run questions and save every best-guess canvas"],
 ["parse_commitment.py", "Find when the answer and reasoning stabilize"],
 ["intervene_swap.py", "Alter an answer during denoising (unfinished)"],
 ["judge.py", "Label reverted, anchored, copied, or derailed"],
]
ARCH = {"nodes":[
 ["prompt","Question + instructions","input"],["encoder","Encode prompt once","model"],
 ["noise","256 random tokens","state"],["denoiser","Predict every position","model"],
 ["guess","Best-guess canvas","state"],["accept","Accept confident tokens","model"],
 ["renoise","Re-randomize the rest","state"],["self","Guesses condition next step","state"],
 ["hook","Optional intervention hook","experiment"],["stop","Stable or out of steps?","decision"],
 ["answer","Final solution","output"]],
 "edges":[["prompt","encoder","tokens"],["encoder","denoiser","cached context"],
 ["noise","denoiser","canvas"],["self","denoiser","previous guesses"],
 ["denoiser","guess","logits"],["guess","accept","confidence"],["accept","renoise","unaccepted"],
 ["accept","self","all guesses"],["renoise","hook","next canvas"],
 ["hook","stop","edited or unchanged"],["stop","denoiser","continue"],["stop","answer","done"]]}

def scan():
 paths=sorted(SRC.glob("*.py")); known={p.stem for p in paths}; nodes=[]; edges=[]; hashes=[]
 for path in paths:
  raw=path.read_bytes(); hashes.append(hashlib.sha1(raw).hexdigest())
  node={"id":path.stem,"label":path.name,"lines":raw.count(b"\n")+1,
        "functions":[],"details":[]}
  try:
   tree=ast.parse(raw.decode(),filename=str(path))
   for item in tree.body:
    if isinstance(item,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
     node["functions"].append(item.name)
     is_class=isinstance(item,ast.ClassDef)
     signature=item.name if is_class else f"{item.name}({ast.unparse(item.args)})"
     doc=(ast.get_docstring(item) or "No docstring yet.").strip().split("\n")[0]
     node["details"].append({
      "name":item.name, "signature":signature, "line":item.lineno,
      "kind":"class" if is_class else "async function" if isinstance(item,ast.AsyncFunctionDef) else "function",
      "doc":doc,
     })
    imports=([item.module] if isinstance(item,ast.ImportFrom) and item.module else
             [a.name for a in item.names] if isinstance(item,ast.Import) else [])
    for name in imports:
     target=name.split(".")[0]
     if target in known: edges.append([path.stem,target,"imports"])
  except (SyntaxError,UnicodeDecodeError) as exc: node["error"]=str(exc)
  nodes.append(node)
 return {"version":hashlib.sha1("".join(hashes).encode()).hexdigest()[:10],
         "nodes":nodes,"edges":edges,"pipeline":PIPELINE,"arch":ARCH}

PAGE=r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>diffusion-anchors</title><style>
:root{--bg:#080d18;--panel:#111827;--card:#182235;--ink:#f1f5ff;--muted:#94a3b8;--border:#28364d;--blue:#69c8ff;--green:#8ee6a1;--pink:#f39ac7}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 Inter,ui-sans-serif,system-ui,sans-serif}
header{display:flex;align-items:center;padding:18px max(24px,calc((100% - 1200px)/2));border-bottom:1px solid var(--border);background:#0b1220}
h1{font-size:18px;margin:0}header small{margin-left:12px;color:var(--muted)}#status{margin-left:auto;color:var(--green)}
main{max-width:1200px;margin:auto;padding:26px 24px 50px}section{margin-bottom:24px;background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:22px}
h2{font-size:16px;margin:0 0 4px}p{color:var(--muted);margin:0 0 18px}.flow{display:grid;grid-template-columns:repeat(4,1fr);gap:28px}.flow .box{position:relative}.flow .box:not(:last-child):after{content:"→";position:absolute;right:-21px;top:34px;color:#64748b;font-size:20px}
.box,.file,.unit{background:var(--card);border:1px solid #32425e;border-radius:10px;padding:14px}.box strong{display:block;color:var(--blue);margin-bottom:5px}.box span,.unit span{color:var(--muted)}
.loop{display:grid;grid-template-columns:190px 1fr 190px;gap:18px;align-items:center}.outside{text-align:center}.loop-core{border:1px solid #3a4e70;border-radius:14px;padding:18px;background:#0e1728}
.cycle{display:grid;grid-template-columns:repeat(3,1fr);gap:24px}.unit{position:relative;text-align:center}.unit:not(:last-child):after{content:"→";position:absolute;right:-19px;color:#64748b}
.return{text-align:center;color:var(--muted);padding:14px 0 0}.return b{color:var(--pink)}.answer{border-color:#3f7651}
.files{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}.file h3{font:600 14px ui-monospace,monospace;margin:0;color:var(--blue)}.meta{color:var(--muted);font-size:12px;margin:4px 0 11px}.pills{display:flex;gap:5px;flex-wrap:wrap}.pill{font:11px ui-monospace,monospace;padding:3px 7px;background:#0d1524;border-radius:999px;color:#b9c5d8}.imports{color:#8393ad;font-size:11px;margin-top:10px}
@media(max-width:800px){.flow,.cycle{grid-template-columns:1fr}.flow .box:after,.unit:after{display:none}.loop{grid-template-columns:1fr}}</style></head>
<body><header><h1>diffusion-anchors</h1><small>live architecture map</small><span id="status">connecting...</span></header><main>
<section><h2>Experiment</h2><p>Four stages, from raw math problems to the final result.</p><div id="flow" class="flow"></div></section>
<section><h2>How DiffusionGemma produces one answer</h2><p>The prompt stays fixed. The 256-token answer canvas changes many times.</p>
<div class="loop"><div class="box outside"><strong>Question</strong><span>Encoded once and reused</span></div>
<div class="loop-core"><div class="cycle"><div class="unit"><strong>1. Predict</strong><br><span>Guess every token</span></div><div class="unit"><strong>2. Accept</strong><br><span>Keep confident tokens</span></div><div class="unit"><strong>3. Re-noise</strong><br><span>Randomize uncertain tokens</span></div></div>
<div class="return">↩ Feed guesses back in and repeat<br><b>Future intervention hook lives here</b></div></div>
<div class="box outside answer"><strong>Final solution</strong><span>Stop when stable or out of steps</span></div></div></section>
<section><h2>Codebase</h2><p>Each card is one Python file. This section updates whenever you save.</p><div id="files" class="files"></div></section>
</main><script>
let version="";
function flow(xs){document.querySelector("#flow").innerHTML=xs.map((x,i)=>`<div class="box"><strong>${i+1}. ${x[0]}</strong><span>${x[1]}</span></div>`).join("")}
function files(d){let incoming={};d.edges.forEach(([a,b])=>(incoming[a]??=[]).push(b+".py"));document.querySelector("#files").innerHTML=d.nodes.map(n=>`<article class="file"><h3>${n.label}</h3><div class="meta">${n.lines} lines · ${n.functions.length} definitions</div><div class="pills">${n.functions.map(x=>`<span class="pill">${x}</span>`).join("")||'<span class="pill">module only</span>'}</div>${incoming[n.id]?.length?`<div class="imports">imports → ${incoming[n.id].join(", ")}</div>`:""}</article>`).join("")}
async function refresh(){try{let d=await(await fetch("/api/graph",{cache:"no-store"})).json();if(d.version!==version){version=d.version;flow(d.pipeline);files(d)}document.querySelector("#status").textContent=`● live · ${new Date().toLocaleTimeString()}`}catch(e){document.querySelector("#status").textContent="disconnected"}}refresh();setInterval(refresh,1200);
</script></body></html>"""

class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  path=self.path.split("?")[0]
  if path=="/api/graph": body,kind=json.dumps(scan()).encode(),"application/json"
  elif path in ("/","/index.html"): body,kind=Path(__file__).with_name("viz.html").read_bytes(),"text/html; charset=utf-8"
  else: self.send_error(404); return
  self.send_response(200); self.send_header("Content-Type",kind); self.send_header("Cache-Control","no-store")
  self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)
 def log_message(self,fmt,*args):
  if len(args)>1 and args[1]!="200": super().log_message(fmt,*args)

if __name__=="__main__":
 server=ThreadingHTTPServer((HOST,PORT),Handler)
 print(f"Live codebase map: http://{HOST}:{PORT}\nWatching: {SRC}")
 try: server.serve_forever()
 except KeyboardInterrupt: print("\nStopped.")


"""Fetch only public primary-source records; bounded files and explicit failures."""
from pathlib import Path
import urllib.request,json,time,hashlib,html
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'docs/literature_sources';OUT.mkdir(exist_ok=True)
items=[('tide','https://arxiv.org/html/2304.08424v5'),('patchtst','https://arxiv.org/html/2211.14730v2'),('itransformer','https://arxiv.org/html/2310.06625v4'),('timemixer','https://arxiv.org/html/2405.14616v1'),('timesnet','https://arxiv.org/html/2210.02186v3'),('temposcale','https://arxiv.org/html/2405.12635v1'),('chronograph','https://arxiv.org/html/2509.04449v3'),('radar2026','https://arxiv.org/abs/2609.31292'),('taxonomy2026','https://arxiv.org/abs/2604.25222'),('ibm2026','https://arxiv.org/abs/2610.15818'),('computers_instructions','https://www.mdpi.com/journal/computers/instructions'),('forecasting_history','https://www.mdpi.com/journal/forecasting/history'),('computers_history','https://www.mdpi.com/journal/computers/history'),('mdpi_ethics','https://www.mdpi.com/ethics'),('mdpi_template','https://res.mdpi.com/data/MDPI_template_ACS.zip?v=20260911')]
manifest=[]
for key,url in items:
 p=OUT/(key+('.zip' if key=='mdpi_template' else '.html'))
 try:
  if not p.exists():
   req=urllib.request.Request(url,headers={'User-Agent':'DLinear-reproducible-study/1.0 (research; public documents only)'})
   with urllib.request.urlopen(req,timeout=25) as res:
    body=res.read(12*1024**2+1);assert len(body)<=12*1024**2;p.write_bytes(body)
  body=p.read_bytes();manifest.append(dict(key=key,url=url,access_date='2026-11-10',status='saved',bytes=len(body),sha256=hashlib.sha256(body).hexdigest()))
 except Exception as ex:manifest.append(dict(key=key,url=url,access_date='2026-11-10',status='unavailable',error=repr(ex)))
 print(manifest[-1],flush=True)
 time.sleep(.4)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))

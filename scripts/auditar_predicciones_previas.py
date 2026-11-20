import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'fuentes/reproduccion/resultados/validacion_independiente.json').read_text())
base=ROOT.parent/'ChronoGraph/results/study_701020/test';checks=[];missing=[]
for rec in manifest['files']:
 p=base/rec['file']
 if not p.exists():missing.append(rec['file']);continue
 h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==rec['sha256'];checks.append(rec['file'])
(ROOT/'docs/verificacion_predicciones_previas.json').write_text(json.dumps(dict(verified=len(checks),missing=missing,method='Byte hashes match the independently recomputed forecasts in the authenticated prior results manifest; original files read only.'),indent=2))
print('Previous forecasts verified',len(checks),'missing',len(missing))

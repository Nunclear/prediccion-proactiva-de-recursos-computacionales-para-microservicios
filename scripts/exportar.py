"""Publish local deliverables only after all completion and integrity gates pass."""
import hashlib,json,os,shutil,signal,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
assert json.loads((ROOT/'results/COMPLETED.json').read_text())['status']=='complete'
assert json.loads((ROOT/'results/estadisticas/validacion.json').read_text())['status']=='passed'
OUT=ROOT.parent/'outputs/investigacion_dlinear';OUT.mkdir(parents=True,exist_ok=True)
for folder in ['articulo','docs','src','configs','tests','scripts','docker']:
 shutil.copytree(ROOT/folder,OUT/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
for folder in ['estadisticas','monitoreo','resume_check']:
 shutil.copytree(ROOT/'results'/folder,OUT/'results'/folder,dirs_exist_ok=True)
for folder in ['reproduccion','mejoras']:
 for sub in ['checkpoints','tuning','test']:
  shutil.copytree(ROOT/'results'/folder/sub,OUT/'results'/folder/sub,dirs_exist_ok=True)
 for p in (ROOT/'results'/folder).iterdir():
  if not p.is_file() or p.suffix not in ['.json','.jsonl','.log','.txt']:continue
  q=OUT/'results'/folder/p.name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
 shutil.copy2(ROOT/'results'/folder/'data_verified.npz',OUT/'results'/folder/'data_verified.npz')
for p in (ROOT/'results').iterdir():
 if p.is_file() and p.suffix in ['.json','.jsonl','.log','.txt']:
  q=OUT/'results'/p.name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
shutil.copytree(ROOT/'fuentes/reproduccion',OUT/'fuentes/reproduccion',dirs_exist_ok=True)
shutil.copytree(ROOT/'archivos_originales',OUT/'archivos_originales',dirs_exist_ok=True)
shutil.copy2(ROOT/'README.md',OUT/'README.md')
manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='MANIFEST_SHA256.json'}
(OUT/'MANIFEST_SHA256.json').write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile(OUT.parent/'investigacion_dlinear_completa.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in OUT.rglob('*'):
  if p.is_file():z.write(p,str(p.relative_to(OUT)))
if not (ROOT/'results/STOP_MONITOR').exists():(ROOT/'results/STOP_MONITOR').touch()
print('EXPORTED',OUT)

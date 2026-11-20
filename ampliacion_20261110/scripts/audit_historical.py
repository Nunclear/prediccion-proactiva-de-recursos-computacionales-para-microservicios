"""Preserve and authenticate historical results; no refitting or relabeling as novel."""
from pathlib import Path
import hashlib,json,csv,shutil
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent;REV=BASE/'revision_20261029'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 previous=json.loads((REV/'results/comparaciones/validacion.json').read_text());rows=[]
 for category,base in [('original_files',BASE),('new_files',REV)]:
  for item in previous[category]:
   p=base/item['file'];actual=sha(p);assert actual==item['sha256'],p
   rows.append(dict(path=str(p.relative_to(BASE)),sha256=actual,category='historical',verification='matches prior independently reconstructed prediction'))
 assert len(rows)==1720
 out=ROOT/'results/historical';out.mkdir(exist_ok=True)
 with (out/'prediction_manifest.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 tables=[]
 for name in ['resumen_global.csv','resumen_condiciones.csv','contrastes_dos_escalas.csv','contrastes_por_semilla.csv','contrastes_por_servicio.csv','costes_nueva_medicion.csv']:
  p=REV/'results/comparaciones'/name;shutil.copy2(p,out/name);tables.append(dict(path=name,source=str(p.relative_to(BASE)),sha256=sha(p)))
 result=dict(status='passed',historical_prediction_hashes_reverified=len(rows),prior_independent_metric_reconstruction='revision_20261029/results/comparaciones/validacion.json',prior_validation_sha256=sha(REV/'results/comparaciones/validacion.json'),physical_metric_rows=previous['physical_metric_rows'],rows_including_V2_alias=previous['rows_including_V2_alias'],V2_is_independent_fit=False,tables=tables,interpretation='Preserved historical exploratory evidence, not a fresh test and not combined with new-service primary cells.')
 (out/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='tables'},indent=2))
if __name__=='__main__':main()

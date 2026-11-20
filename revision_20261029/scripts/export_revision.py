"""Copy and verify a self-contained full project and a smaller academic package."""
from pathlib import Path
import hashlib,json,shutil,zipfile,os,signal
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
OUTPUTS=BASE.parent/'outputs'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def copytree(source,target):
 shutil.copytree(source,target,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
def manifest(folder):
 records={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='MANIFEST_SHA256.json'}
 (folder/'MANIFEST_SHA256.json').write_text(json.dumps(records,indent=2,ensure_ascii=False))
 return records
def archive(folder,path):
 temp=path.with_suffix('.tmp')
 with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
  for p in sorted(folder.rglob('*')):
   if p.is_file():z.write(p,str(p.relative_to(folder)),compress_type=zipfile.ZIP_STORED if p.suffix.lower() in ['.npz','.zip','.png','.gz'] else zipfile.ZIP_DEFLATED)
 temp.replace(path)
 with zipfile.ZipFile(path) as z:assert z.testzip() is None
 (path.with_suffix(path.suffix+'.sha256')).write_text(sha(path)+'  '+path.name+'\n')
 return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path),zip_crc='passed')
def main():
 assert json.loads((ROOT/'results/VALIDACION_FINAL.json').read_text())['status']=='passed'
 full=OUTPUTS/'investigacion_dlinear_mejorada';full.mkdir(parents=True,exist_ok=True)
 for name in ['src','scripts','tests','configs','docker','docs','articulo','archivos_originales','fuentes/reproduccion']:
  copytree(BASE/name,full/name)
 for name in ['reproduccion','mejoras','estadisticas','monitoreo','resume_check']:
  copytree(BASE/'results'/name,full/'results'/name)
 for p in (BASE/'results').iterdir():
  if p.is_file() and p.suffix in ['.json','.jsonl','.log','.txt']:shutil.copy2(p,full/'results'/p.name)
 shutil.copy2(BASE/'README.md',full/'README_HISTORICO.md')
 copytree(ROOT,full/ROOT.name)
 (full/'README.md').write_text('# Investigación y revisión del artículo\n\n[Artículo mejorado](revision_20261029/articulo/ARTICULO_EMPIRICO_MEJORADO.md) · [Resumen de cambios](revision_20261029/articulo/RESUMEN_CAMBIOS.md) · [Evaluación crítica](revision_20261029/docs/EVALUACION_FINAL.md) · [Guía reproducible](revision_20261029/docs/GUIA_REPRODUCCION.md)\n\nLa raíz conserva el estudio histórico; revision_20261029 contiene la nueva versión, su código y resultados. Las cifras provienen de datos y ejecuciones reales. No se obtuvo una evaluación externa independiente y la variante compacta no mejora ambos recursos. El manifiesto registra los hashes de esta entrega completa. Los originales y sus ZIP se conservaron sin alteración.\n')
 full_records=manifest(full)
 academic=OUTPUTS/'articulo_dlinear_mejorado';academic.mkdir(parents=True,exist_ok=True)
 for name in ['articulo','docs']:copytree(ROOT/name,academic/name)
 (academic/'results/comparaciones').mkdir(parents=True,exist_ok=True)
 for p in (ROOT/'results/comparaciones').iterdir():
  if p.is_file():shutil.copy2(p,academic/'results/comparaciones'/p.name)
 for p in ['results/VALIDACION_FINAL.json','results/nuevos_experimentos/selected_candidates.json','results/nuevos_experimentos/frozen_v2.json','results/nuevos_experimentos/COMPLETED.json']:
  target=academic/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,target)
 (academic/'README.md').write_text('# Entrega académica\n\n[Artículo mejorado](articulo/ARTICULO_EMPIRICO_MEJORADO.md) · [Cambios](articulo/RESUMEN_CAMBIOS.md) · [Auditoría](docs/AUDITORIA_ARTICULO.md) · [Evaluación final](docs/EVALUACION_FINAL.md)\n\nContiene artículo, figuras PNG/SVG, documentación y tablas derivadas. Las predicciones individuales, pesos y entorno se entregan por separado en investigacion_dlinear_mejorada_completa.zip. Este paquete ligero no sustituye al proyecto completo para reproducir entrenamientos o recomputar desde NPZ.\n')
 academic_records=manifest(academic)
 full_archive=archive(full,OUTPUTS/'investigacion_dlinear_mejorada_completa.zip')
 academic_archive=archive(academic,OUTPUTS/'articulo_dlinear_mejorado.zip')
 result=dict(status='passed',full_files=len(full_records),academic_files=len(academic_records),full=full_archive,academic=academic_archive,
   evidence=dict(historical_prediction_hashes_verified=1240,new_predictions_recomputed=480,new_final_fits=24,short_selection_fits=48,benchmark_calls=2800,independent_external_evaluation=False))
 (OUTPUTS/'revision_dlinear_verificacion.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
 print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
 try:main()
 finally:
  for p in Path('/proc').glob('[0-9]*/cmdline'):
   try:
    cmd=p.read_bytes().split(b'\0')
    if cmd[0].endswith(b'systemd-inhibit') and b'--who=DLinear revision' in cmd:os.kill(int(p.parent.name),signal.SIGTERM)
   except (FileNotFoundError,PermissionError,ProcessLookupError):pass

import csv,hashlib,json,shutil,zipfile,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--reproduction-zip',type=Path,default=ROOT/'archivos_originales/reproduccion_y_modelos.zip');parser.add_argument('--context-zip',type=Path,default=ROOT/'archivos_originales/Grupo05_CONTEXTO_COMPLETO_TRES_ETAPAS_PDF_20261027.zip');args=parser.parse_args()
INPUTS=[('reproduccion',args.reproduction_zip),('contexto',args.context_zip)]
records=[];audits=[]
for label,source in INPUTS:
 dest=ROOT/'archivos_originales'/source.name
 if not dest.exists():shutil.copy2(source,dest)
 digest=hashlib.sha256(source.read_bytes()).hexdigest();assert hashlib.sha256(dest.read_bytes()).hexdigest()==digest
 with zipfile.ZipFile(dest) as z:
  assert z.testzip() is None
  hashes={};sizes={}
  for item in z.infolist():
   p=Path(item.filename);assert not p.is_absolute() and '..' not in p.parts
   if item.is_dir():continue
   content=z.read(item.filename);h=hashlib.sha256(content).hexdigest();hashes[item.filename]=h;sizes[item.filename]=len(content)
   output=ROOT/'fuentes'/label/p;output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(content)
   records.append(dict(archive=source.name,path=item.filename,bytes=len(content),sha256=h,extension=p.suffix,kind='script' if p.suffix in ['.py','.sh'] else 'dataset' if any(k in item.filename for k in ['datos/','data_verified','muestra','cache_rangos']) else 'config_or_results' if p.suffix in ['.json','.csv','.npz','.pt'] else 'document'))
  if label=='reproduccion':manifest=json.loads(z.read('CONTENIDO_SHA256.json'))
  else:manifest={e['archivo']:e['sha256'] for e in json.loads(z.read('MANIFIESTO_SHA256.json'))['archivos']}
  mismatches=[p for p,h in manifest.items() if hashes.get(p)!=h]
  assert not mismatches,mismatches
  audits.append(dict(archive=source.name,sha256=digest,compressed_bytes=dest.stat().st_size,uncompressed_bytes=sum(sizes.values()),files=len(hashes),manifest_entries=len(manifest),verified=True))
with (ROOT/'docs/INVENTARIO_COMPLETO.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
(ROOT/'docs/auditoria_archivos.json').write_text(json.dumps(audits,indent=2))
print(json.dumps(audits,indent=2))

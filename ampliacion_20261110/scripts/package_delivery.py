"""Create two LOCAL review archives after the computational/visual delivery gate.
No network upload, raw trace redistribution or author license assignment.
"""
from pathlib import Path
import json,hashlib,zipfile,shutil
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parents[1]/'outputs/ampliacion_20261110'
EXCLUDE_DOCS={'INFORMACION_INTERNA_CONTINUIDAD.md','RSL_RECIENTE_TEXTO.txt','ARTICULO_BASE_PDF_TEXTO.txt','ALIBABA_2022_README_OFICIAL.md','alibaba_2022.txt','alibaba_readme.txt','alibaba_root.txt'}
EXCLUDE_SUFFIX={'.aux','.bbl','.blg','.log','.out','.fls','.fdb_latexmk','.synctex.gz','.pyc'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def permitted(p):
 rel=p.relative_to(ROOT)
 if not p.is_file() or '__pycache__' in rel.parts:return False
 if p.suffix=='.pyc' or rel.parts[0]=='articulo_final' and p.suffix in EXCLUDE_SUFFIX:return False
 if rel.parts[0] in {'tmp','predictions_recomputed','articulo_original'}:return False
 if rel.parts[:2]==('data','raw'):return False
 if 'literature_sources' in rel.parts:return False
 if p.name in EXCLUDE_DOCS or p.name.startswith('main.') and p.suffix not in {'.tex','.pdf'}:return False
 return True
def archive(path,files):
 temp=path.with_suffix('.zip.partial')
 with zipfile.ZipFile(temp,'w',allowZip64=True,compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in files:z.write(p,arcname=str(p.relative_to(ROOT)),compress_type=zipfile.ZIP_STORED if p.suffix in {'.npz','.gz','.zip','.png','.pdf'} else zipfile.ZIP_DEFLATED)
 with zipfile.ZipFile(temp) as z:
  assert len(z.infolist())==len(files)
  # Central-directory and CRC verification reads every archived byte.
  assert z.testzip() is None
 temp.replace(path)
 return dict(path=str(path),files=len(files),bytes=path.stat().st_size,sha256=sha(path),verification='Every member read successfully by ZipFile.testzip; SHA256 computed after archive close')
def main():
 verified=json.loads((ROOT/'results/FINAL_VERIFICATION.json').read_text())
 assert verified['status']=='local_computational_delivery_verified'
 OUT.mkdir(parents=True,exist_ok=True)
 allfiles=sorted(p for p in ROOT.rglob('*') if permitted(p))
 editorial=[p for p in allfiles if p.relative_to(ROOT).parts[0]=='articulo_final' or p.relative_to(ROOT).parts[:1]==('docs',) and p.suffix=='.md' or p.name=='author_confirmations.json']
 # Keep documentation, but do not duplicate original copyrighted PDFs or fetched full texts.
 records=[archive(OUT/'paquete_editorial_revision.zip',editorial),archive(OUT/'reproducibilidad_local_completa.zip',allfiles)]
 for src,dst in [('main.pdf','articulo_ingles.pdf'),('revision_espanol.pdf','revision_espanol.pdf')]:shutil.copy2(ROOT/'articulo_final'/src,OUT/dst)
 for p in [OUT/'articulo_ingles.pdf',OUT/'revision_espanol.pdf']:records.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
 manifest=dict(status='local_archives_verified',archives_and_pdfs=records,scope='Local private review only. No upload or public redistribution permission inferred. Raw data, original supplied PDFs, fetched article full texts and internal continuity note excluded. Historical prediction archives remain preserved in the parent study; their audit manifests and numerical summaries are included.',submission_ready=False)
 (OUT/'entrega_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
 print(json.dumps(manifest,indent=2,ensure_ascii=False),flush=True)
if __name__=='__main__':main()

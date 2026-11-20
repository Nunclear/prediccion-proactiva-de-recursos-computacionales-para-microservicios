"""Lightweight static integrity checks; deliberately never invokes training."""
from pathlib import Path
import ast,csv,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 count=0
 for p in ROOT.rglob('*.py'):
  if any(x in p.parts for x in ['.venv','.git','__pycache__']):continue
  ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p));count+=1
 rows=list(csv.DictReader((ROOT/'data/MANIFIESTO.csv').open()));loaded=[]
 for row in rows:
  p=ROOT/'ampliacion_20261110/data/processed'/Path(row['archivo']).name
  if p.exists():
   assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],p
   loaded.append(p.name)
 p=ROOT/'articulo_ieee/articulo_ieee_es.pdf'
 expected=json.loads((ROOT/'articulo_ieee/verificacion_ieee.json').read_text())['versions']['es']['sha256']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
 csvrows=list(csv.DictReader((ROOT/'ampliacion_20261110/results/analysis/primary_summary.csv').open()))
 lookup={(r['resource'],r['model']):float(r['MASE']) for r in csvrows}
 assert abs(lookup['container_memory_usage_bytes','A3']-1.2403332331650325)<1e-12
 assert len(csvrows)==30
 print(json.dumps(dict(status='CONFIRMADO',python_files_parsed=count,summary_rows=len(csvrows),data_hashes_checked=loaded,training_executed=False),indent=2))
if __name__=='__main__':main()

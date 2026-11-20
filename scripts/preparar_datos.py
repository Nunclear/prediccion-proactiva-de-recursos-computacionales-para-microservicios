"""Install only verified model inputs; no training, masking or preprocessing."""
from pathlib import Path
import argparse,hashlib,zipfile,csv,io
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('zip',type=Path);args=parser.parse_args()
 expected=list(csv.DictReader((ROOT/'data/MANIFIESTO.csv').open()))
 with zipfile.ZipFile(args.zip) as archive:
  assert archive.testzip() is None,'ZIP corrupto'
  for row in expected:
   raw=archive.read(row['archivo'])
   assert len(raw)==int(row['bytes']),row['archivo']
   assert hashlib.sha256(raw).hexdigest()==row['sha256'],row['archivo']
   with np.load(io.BytesIO(raw),allow_pickle=False) as z:
    assert 'x'.join(map(str,z['values'].shape))==row['forma_values']
   name=Path(row['archivo']).name
   targets=[ROOT/'ampliacion_20261110/data/processed'/name]
   if name=='chronograph_source.npz':targets += [ROOT/f/'data_verified.npz' for f in ['results/reproduccion','results/mejoras','revision_20261029/results/nuevos_experimentos']]
   for target in targets:
    if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256'],f'No se sobrescribe entrada diferente: {target}'
    else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
   print(name, 'SHA256 y forma verificados; rutas originales preparadas')
if __name__=='__main__':main()

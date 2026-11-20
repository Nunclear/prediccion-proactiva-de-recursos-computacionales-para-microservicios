"""Fetch exact historical upstream files without redistributing unlicensed copies."""
from pathlib import Path
import hashlib,json,urllib.request,subprocess,tempfile,shutil
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 entries=json.loads((ROOT/'configs/terceros_historicos.json').read_text())
 with tempfile.TemporaryDirectory() as d:
  stage=Path(d);by_name={}
  for e in entries:
   dest=ROOT/e['destination']
   if dest.exists() and sha(dest)==e['expected_local_sha256']:continue
   url=f"https://raw.githubusercontent.com/{e['repository']}/{e['commit']}/{e['path']}"
   raw=urllib.request.urlopen(url,timeout=60).read()
   if e['upstream_sha256']:assert hashlib.sha256(raw).hexdigest()==e['upstream_sha256'],url
   if 'BiTGraph' in e['repository']:
    p=stage/Path(e['path']).name;p.write_bytes(raw.replace(b'\r\n',b'\n'));by_name[p.name]=e
   else:
    assert hashlib.sha256(raw).hexdigest()==e['expected_local_sha256'],url
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
  if by_name:
   subprocess.run(['patch','--batch','-p1','-i',str(ROOT/'src/baseline/reference/portable/portability.patch')],cwd=stage,check=True)
   for name,e in by_name.items():
    p=stage/name;assert sha(p)==e['expected_local_sha256'],f'Parche no coincide: {name}'
    dest=ROOT/e['destination'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
  init=ROOT/'src/baseline/reference/portable/BiaTCGNet/__init__.py';init.parent.mkdir(parents=True,exist_ok=True);init.touch(exist_ok=True)
 print('Dependencias históricas verificadas contra los hashes usados; no se ha entrenado.')
if __name__=='__main__':main()

"""Official research trace download, bounded and hashed; no private uploads."""
import urllib.request,hashlib,json,time,base64,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def fetch(index):
 url=f'https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2022MicroservicesTraces/MSMetricsUpdate/MSMetricsUpdate_{index}.tar.gz'
 path=ROOT/'data/raw'/f'MSMetricsUpdate_{index}.tar.gz';record=ROOT/'data/manifests'/f'alibaba_{index}.json'
 if path.exists() and record.exists():return json.loads(record.read_text())
 if shutil.disk_usage(ROOT).free<50*1024**3:raise RuntimeError('Disk reserve below 50 GiB')
 start=time.monotonic();h=hashlib.sha256();md5=hashlib.md5();size=0;temp=path.with_suffix('.part')
 with urllib.request.urlopen(url,timeout=60) as response,temp.open('wb') as f:
  headers=dict(response.headers);length=int(response.headers.get('Content-Length','0'))
  assert 0<length<1024**3,'Unexpected object size; budget gate'
  while True:
   chunk=response.read(2*1024**2)
   if not chunk:break
   f.write(chunk);h.update(chunk);md5.update(chunk);size+=len(chunk)
   if size//(32*1024**2)!=(size-len(chunk))//(32*1024**2):print(json.dumps(dict(index=index,bytes=size,seconds=time.monotonic()-start)),flush=True)
 assert size==length
 expected=next((v for k,v in headers.items() if k.lower()=='content-md5'),None)
 if expected:assert base64.b64encode(md5.digest()).decode()==expected
 temp.replace(path)
 item=dict(index=index,url=url,bytes=size,sha256=h.hexdigest(),content_md5_verified=bool(expected),headers=headers,seconds=time.monotonic()-start,terms='Owner README permits research/study use; no standalone LICENSE observed; raw redistribution not presumed')
 record.write_text(json.dumps(item,indent=2));print(json.dumps(item),flush=True);return item
if __name__=='__main__':
 for i in map(int,sys.argv[1:] or ['0']):fetch(i)

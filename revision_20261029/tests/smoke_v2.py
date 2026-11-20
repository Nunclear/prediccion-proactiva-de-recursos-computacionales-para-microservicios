import sys,shutil,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import run_v2 as v
import torch,numpy as np
e=v.e;e.OUT=ROOT/'results/smoke';e.OUT.mkdir(exist_ok=True)
shutil.copy2(v.BASE/'results/reproduccion/data_verified.npz',e.OUT/'data_verified.npz')
data=e.Data();e.create=v.create
for r in range(2):
 tr=v.adapter.Prepared(data,r,'train',70001,e.control.sample_indices(data.indices['train'],32));va=v.adapter.Prepared(data,r,'validation',80001,e.control.sample_indices(data.indices['validation'],14))
 lo,hi=data.bounds['train'];assert torch.all(tr.z[:,lo:hi,:,1]+tr.z[:,lo:hi,:,3]+tr.z[:,lo:hi,:,4]==1)
 for name in ['C0','C1','C2','C3']:
  a,b=v.FAMILIES[name];v.CURRENT[name]=dict(backend='ema07',availability=a,age=b)
  m,c=e.train(name,data,r,tr,va,101,.003,1,None,e.OUT/f'{name}_r{r}.pt');assert c['complete']
  assert np.isfinite(e.evaluate(name,m,va)['MASE']).all()
(e.OUT/'PASSED.json').write_text(json.dumps(dict(status='passed',fits=8,not_reused_in_main_training=True)))
print('ALL_COMPACT_SMOKE_TESTS_PASSED')

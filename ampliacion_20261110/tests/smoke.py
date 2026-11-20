import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from core import *
from models import *
from engine import train,evaluate,sample
data=Series(ROOT/'data/processed/chronograph_source.npz')
out=ROOT/'results/smoke';out.mkdir(exist_ok=True)
for L,h in [(12,1),(48,1),(12,12)]:
 tr=Prepared(data,0,'train',L,h,70001,subset=sample(data.indices('train',L,h),32));va=Prepared(data,0,'validation',L,h,80001,subset=sample(data.indices('validation',L,h),16))
 for name in (NEURAL if L==12 and h==1 else ['DLinear','E5','A3','PatchTST']):
  model,ck=train(name,data,L,h,0,tr,va,101,.003,0.,1,None,out/f'{name}_L{L}_H{h}.pt')
  assert ck['complete'] and np.isfinite(evaluate(model,va)['MASE']).all();print('SMOKE_OK',L,h,name,flush=True)
atomic_json(out/'PASSED.json',dict(status='passed',fits=19,not_used_for_final_initialization=True))

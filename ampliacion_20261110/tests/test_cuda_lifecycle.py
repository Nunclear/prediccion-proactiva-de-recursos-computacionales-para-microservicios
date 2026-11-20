import sys,gc,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from core import *
import engine as e
assert DEVICE.type=='cuda';torch.cuda.set_per_process_memory_fraction(.28)
data=Series(ROOT/'data/processed/chronograph_source.npz')
tr=Prepared(data,0,'train',12,1,70001,subset=e.sample(data.indices('train',12,1),64));va=Prepared(data,0,'validation',12,1,80001,subset=e.sample(data.indices('validation',12,1),128))
folder=ROOT/'results/cuda_lifecycle';folder.mkdir(exist_ok=True);allocated=[]
for i in range(6):
 p=folder/f'fit{i}.pt'
 if p.exists():p.unlink()
 model,ck=e.train('GRUD',data,12,1,0,tr,va,101+i,.003,0.,2,None,p)
 del model,ck
 torch.cuda.synchronize();allocated.append(torch.cuda.memory_allocated());print('LIVE_ALLOCATION',i,allocated[-1],flush=True)
assert max(allocated)-min(allocated)<12*1024**2,allocated
atomic_json(folder/'PASSED.json',dict(status='passed',fits=6,epochs_each=2,validation_batch=128,live_allocated_bytes_after_fit=allocated,range_bytes=max(allocated)-min(allocated)))

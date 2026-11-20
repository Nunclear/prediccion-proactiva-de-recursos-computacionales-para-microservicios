import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import numpy as np,torch,hashlib,json,random
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESOURCES=['cpu_usage','container_memory_usage_bytes']
CASES=[(0,'disperso')]+[(q,p) for q in [10,20,40] for p in ['disperso','bloques']]+[(20,'correlacionado'),(20,'prolongado'),(20,'MAR_lag')]
def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def filehash(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def atomic_json(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(p)
def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if DEVICE.type=='cuda':torch.cuda.manual_seed_all(seed)
def state_cpu(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def state_hash(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def segments(times,lo,hi,step):
 starts=[lo]+(np.flatnonzero(np.diff(times[lo:hi])!=step)+lo+1).tolist()+[hi]
 return list(zip(starts,starts[1:]))
class Series:
 def __init__(self,path,external=False):
  a=np.load(path);self.values=a['values'].astype(np.float64);self.times=a['times'];self.services=a['services'].tolist()
  self.a=a['adjacency'].astype(np.float32);self.n=len(self.services);self.step=int(np.median(np.diff(self.times)))
  self.external=external;T=len(self.times)
  self.cuts=[0,12,12,T] if external else [0,int(T*.7),int(T*.8),T]
  self.bounds=dict(train=self.cuts[:2],validation=self.cuts[1:3],test=[0,T] if external else self.cuts[2:])
  train=self.values[:self.cuts[1]];self.mean=np.nanmean(train,0);self.scale=np.maximum(np.nanstd(train,axis=0),1e-8);self.median=np.nanmedian(train,axis=0)
  pairs=np.diff(self.times[:self.cuts[1]])==self.step
  self.mase=np.nanmean(np.abs(np.diff(train,axis=0))[pairs],axis=0)
  assert np.isfinite(self.mean).all() and np.isfinite(self.scale).all()
 def indices(self,part,L,h):
  lo,hi=self.bounds[part];ix=np.concatenate([np.arange(a+L+h-1,b,dtype=np.int64) for a,b in segments(self.times,lo,hi,self.step)])
  if self.external and part=='test':ix=ix[ix>=self.cuts[1]]
  return ix
def mask_for(data,part,q,pattern,seed):
 lo,hi=data.bounds[part];observed=np.ones((len(data.times),data.n),dtype=bool)
 for seg,(a,b) in enumerate(segments(data.times,lo,hi,data.step)):
  length=b-a;k=int(np.rint(q*length/100))
  if pattern=='MAR_lag':
   rng=np.random.default_rng(seed+10000*seg+7711);u=rng.random((length,data.n));last=(data.median[:,0]-data.mean[:,0])/data.scale[:,0]
   for j,t in enumerate(range(a,b)):
    p=.2*(1+.5*np.tanh(np.abs(last)));yes=u[j]>=p;observed[t]=yes
    x=(data.values[t,:,0]-data.mean[:,0])/data.scale[:,0];last=np.where(yes&np.isfinite(x),x,last)
   continue
  for n in range(data.n):
   group=n//8 if pattern=='correlacionado' else n
   rng=np.random.default_rng(seed+100*group+10000*seg);priority=rng.random(length)
   if pattern in ['disperso','correlacionado']:chosen=np.argsort(priority,kind='stable')[:k]
   else:
    maximum=24 if pattern=='prolongado' else 6;shift=int(rng.integers(0,maximum+1))
    blocks=[np.arange(max(0,c),min(length,c+maximum)) for c in range(-shift,length,maximum+1) if min(length,c+maximum)>max(0,c)]
    blocks.sort(key=lambda ix:float(priority[ix].min()));chosen=np.concatenate(blocks)[:k] if blocks else np.array([],int)
   observed[a+chosen,n]=False
   assert np.count_nonzero(~observed[a:b,n])==k
 return observed
class Prepared:
 def __init__(self,data,r,part,L,h,seed,subset=None,device=DEVICE):
  self.data=data;self.r=r;self.part=part;self.L=L;self.h=h;self.device=device
  self.indices=data.indices(part,L,h) if subset is None else np.asarray(subset);self.ix=torch.tensor(self.indices,device=device)
  self.offsets=torch.arange(-h-L+1,-h+1,device=device)
  T=len(data.times);self.audit=[];zs=[];natural=np.isfinite(data.values[:,:,r]);lo,hi=data.bounds[part]
  norm=(data.values[:,:,r]-data.mean[:,r])/data.scale[:,r]
  for q,pattern in CASES:
   induced=mask_for(data,part,q,pattern,seed);m=induced&natural;z=np.zeros((T,data.n,5),np.float32)
   for a,b in segments(data.times,lo,hi,data.step):
    last=(data.median[:,r]-data.mean[:,r])/data.scale[:,r];age=np.zeros(data.n);seen=np.zeros(data.n,bool)
    for t in range(a,b):
     yes=m[t];last=np.where(yes,norm[t],last);age=np.where(yes,0,age+1);seen|=yes
     z[t,:,0]=last;z[t,:,1]=yes;z[t,:,2]=np.log1p(age)/np.log1p(L);z[t,:,3]=(~yes)&seen;z[t,:,4]=(~yes)&(~seen)
   assert np.isfinite(z).all();zs.append(z)
   self.audit.append(dict(q_nominal=q,pattern=pattern,induced_mask_sha256=digest(induced),induced_missing_percent=100*float((~induced[lo:hi]).mean()),total_missing_percent=100*float((~m[lo:hi]).mean())))
  self.z=torch.tensor(np.stack(zs),device=device);self.y=torch.tensor(norm,dtype=torch.float32,device=device)
 def batch(self,pos,cases):
  targets=self.ix[pos];hist=targets[:,None]+self.offsets
  return self.z[cases[:,None],hist],self.y[targets]
 def check_causality(self):
  lo,hi=self.data.bounds[self.part]
  assert self.indices.min()-self.h-self.L+1>=lo and self.indices.max()<hi
  assert np.all(self.indices-self.h<self.indices)
def metric(pred,data,r,ix):
 truth=data.values[ix,:,r];physical=pred.astype(np.float64)*data.scale[:,r]+data.mean[:,r];err=physical-truth[None]
 with np.errstate(invalid='ignore',divide='ignore'):
  mae=np.nanmean(np.abs(err),axis=1);rmse=np.sqrt(np.nanmean(err**2,axis=1))
  mase=np.divide(mae,data.mase[:,r],out=np.full_like(mae,np.nan),where=data.mase[:,r]>0)
 return dict(MAE=mae,RMSE=rmse,MASE=mase),truth

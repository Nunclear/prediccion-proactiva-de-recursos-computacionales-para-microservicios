import time,json,copy,csv,random,zipfile,gc
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,torch
from core import *
from models import create,load_for_target
ROOT=Path(__file__).resolve().parents[1]
_REPLAY_STREAM=None
class ReplayedGRUD:
 """Capture the same forward/backward kernels; no optimizer or state updates during warmup."""
 def __init__(self,model,inputs,training):
  self.static=[(z.detach().clone(),y.detach().clone()) for z,y in inputs];self.training=training
  self.graph=torch.cuda.CUDAGraph()
  def compute():
   if training:model.zero_grad(set_to_none=False)
   self.losses=[];self.outputs=[];total=sum(len(y) for z,y in self.static)
   for z,y in self.static:
    p=model(z);self.outputs.append(p)
    if training:
     loss=(p-y).square().mean();(loss*(len(y)/total)).backward();self.losses.append(loss)
  try:
   global _REPLAY_STREAM
   if _REPLAY_STREAM is None:_REPLAY_STREAM=torch.cuda.Stream()
   stream=_REPLAY_STREAM;stream.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(stream):
    for _ in range(3):compute()
   torch.cuda.current_stream().wait_stream(stream)
   with torch.cuda.graph(self.graph,stream=stream):compute()
  except BaseException:
   self.close();raise
 def close(self):
  torch.cuda.synchronize();self.graph.reset()
  self.outputs=[];self.losses=[];self.static=[]
  gc.collect()
 def replay(self,inputs):
  for (z,y),(newz,newy) in zip(self.static,inputs):z.copy_(newz);y.copy_(newy)
  self.graph.replay();return self.outputs
def log(event,**kw):
 row=dict(utc=datetime.now(timezone.utc).isoformat(),event=event,**kw)
 with (ROOT/'results/progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True)
def save_ck(path,ck):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(ck,tmp);tmp.replace(path)
def save_npz(path,**arrays):
 with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for k,v in arrays.items():
   with z.open(k+'.npy','w') as f:np.lib.format.write_array(f,np.asarray(v),allow_pickle=False)
@torch.no_grad()
def evaluate(model,prepared,batch=128,save_to=None):
 model.eval();n=len(prepared.indices);pred=np.empty((len(CASES),n,prepared.data.n),np.float32);captured=None
 try:
  for c in range(len(CASES)):
   start=0
   while start<n:
    size=min(batch,n-start);pos=torch.arange(start,start+size,device=DEVICE);cases=torch.full_like(pos,c)
    try:
     z,y=prepared.batch(pos,cases)
     if model.__class__.__name__=='GRUD' and DEVICE.type=='cuda' and size==batch:
      if captured is None:captured=ReplayedGRUD(model,[(z,y)],False)
      p=captured.replay([(z,y)])[0]
     else:p=model(z)
     assert p.shape==y.shape and torch.isfinite(p).all()
     pred[c,start:start+size]=p.cpu().numpy();start+=size
    except torch.cuda.OutOfMemoryError:
     if batch<=1:raise
     if captured is not None:captured.close()
     captured=None;del pos,cases;torch.cuda.empty_cache();batch=max(1,batch//2);log('inference_oom_retry',batch=batch)
 finally:
  if captured is not None:captured.close()
 values,truth=metric(pred,prepared.data,prepared.r,prepared.indices)
 if save_to is not None:
  save_npz(save_to,prediction_normalized=pred,indices=prepared.indices,target_sha256=digest(truth),prediction_sha256=digest(pred))
 return values
def stable_clip(model):
 try:torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
 except RuntimeError:
  grads=[p.grad for p in model.parameters() if p.grad is not None];assert all(torch.isfinite(g).all() for g in grads)
  norm=torch.sqrt(sum(torch.sum(g.double()**2) for g in grads));assert torch.isfinite(norm)
  with torch.no_grad():
   factor=torch.clamp(1/(norm+1e-6),max=1.)
   for g in grads:g.copy_((g.double()*factor).to(g.dtype))
def _train(name,data,L,h,r,tr,va,seed,lr,wd,epochs,patience,path,micro):
 path=Path(path);signature=dict(model=name,L=L,h=h,r=r,seed=seed,lr=lr,weight_decay=wd)
 seed_all(seed);model=create(name,L,data.a,DEVICE);optim=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd)
 gen=torch.Generator().manual_seed(seed+500);start=0;best=float('inf');best_state=None;stale=0;history=[]
 if path.exists():
  ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['signature']==signature
  if ck['complete']:model.load_state_dict(ck['best_state']);return model,ck
  model.load_state_dict(ck['state']);optim.load_state_dict(ck['optimizer']);gen.set_state(ck['generator'])
  torch.set_rng_state(ck['torch_rng'])
  if DEVICE.type=='cuda':torch.cuda.set_rng_state_all(ck['cuda_rng'])
  random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng'])
  start=ck['epoch']+1;best=ck['best'];best_state=ck['best_state'];stale=ck['stale'];history=ck['history']
 captured=None
 try:
  for epoch in range(start,epochs):
   begin=time.perf_counter();model.train();order=torch.randperm(len(tr.indices),generator=gen);losses=[]
   for logical in order.split(16):
    total=0.
    if name=='GRUD' and DEVICE.type=='cuda' and len(logical)==16:
     inputs=[]
     for small in logical.split(micro):
      pos=small.to(DEVICE);cases=(pos+epoch)%len(CASES);inputs.append(tr.batch(pos,cases))
     if captured is None:captured=ReplayedGRUD(model,inputs,True)
     captured.replay(inputs)
     for loss,(_,y) in zip(captured.losses,inputs):
      assert torch.isfinite(loss);total+=float(loss.detach())*len(y)/len(logical)
    else:
     optim.zero_grad(set_to_none=(name!='GRUD'))
     for small in logical.split(micro):
      pos=small.to(DEVICE);cases=(pos+epoch)%len(CASES);z,y=tr.batch(pos,cases);p=model(z)
      loss=(p-y).square().mean();assert torch.isfinite(loss);weight=len(small)/len(logical);(loss*weight).backward();total+=float(loss.detach())*weight
    stable_clip(model);optim.step();losses.append(total)
   val=evaluate(model,va);score=float(np.nanmean(val['MASE']));assert np.isfinite(score)
   if score<best:best=score;best_state=state_cpu(model);stale=0
   else:stale+=1
   row=dict(epoch=epoch,loss=float(np.mean(losses)),validation_MASE=score,best_MASE=best,seconds=time.perf_counter()-begin,microbatch=micro,effective_batch=16)
   history.append(row);complete=(epoch==epochs-1 or (patience is not None and stale>=patience))
   ck=dict(signature=signature,epoch=epoch,state=state_cpu(model),best_state=best_state,best=best,stale=stale,history=history,complete=complete,
    optimizer=optim.state_dict(),generator=gen.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if DEVICE.type=='cuda' else [],
    python_rng=random.getstate(),numpy_rng=np.random.get_state(),parameters=sum(p.numel() for p in model.parameters()),state_sha256=state_hash(best_state))
   save_ck(path,ck);log('epoch',model=name,resource=RESOURCES[r],L=L,h=h,seed=seed,checkpoint=path.name,**row)
   if complete:break
 finally:
  if captured is not None:captured.close()
 model.load_state_dict(best_state);return model,ck
def train(*args,**kw):
 micro=8
 while True:
  try:return _train(*args,**kw,micro=micro)
  except torch.cuda.OutOfMemoryError:
   if micro<=1:raise
   micro//=2;torch.cuda.empty_cache();log('training_oom_restart_from_epoch_checkpoint',microbatch=micro)
def sample(ix,n):return ix[np.linspace(0,len(ix)-1,min(n,len(ix)),dtype=int)]
def ridge_fit(ds,lam):
 L=ds.L;xtx=np.zeros((L+1,L+1));xty=np.zeros(L+1)
 for c in range(len(CASES)):
  for start in range(0,len(ds.indices),128):
   pos=torch.arange(start,min(start+128,len(ds.indices)),device=DEVICE);z,y=ds.batch(pos,torch.full_like(pos,c))
   x=z[...,0].transpose(1,2).reshape(-1,L).cpu().numpy().astype(np.float64);target=y.cpu().numpy().reshape(-1).astype(np.float64)
   x=np.c_[x,np.ones(len(x))];xtx+=x.T@x;xty+=x.T@target
 regularizer=np.eye(L+1)*lam;regularizer[-1,-1]=0;theta=np.linalg.lstsq(xtx+regularizer,xty,rcond=None)[0]
 m=create('Ridge',L,ds.data.a,DEVICE)
 with torch.no_grad():m.linear.weight.copy_(torch.tensor(theta[:-1][None],dtype=torch.float32,device=DEVICE));m.linear.bias.copy_(torch.tensor(theta[-1:],dtype=torch.float32,device=DEVICE))
 return m

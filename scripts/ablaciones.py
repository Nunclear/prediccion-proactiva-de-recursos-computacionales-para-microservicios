"""Exploratory ablations, selected on validation, never on the known test."""
import os,sys,json,itertools,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'src/baseline/reference')]
os.environ['CHRONOGRAPH_OUTPUT']=str(ROOT/'results/mejoras')
import run as engine
import torch,numpy as np
from improved.models import DLinearEMAMask
OUT=engine.OUT;OUT.mkdir(parents=True,exist_ok=True)
original_create=engine.create
class Prepared(engine.Prepared):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  lo,hi=self.data.bounds[self.partition]
  carried=torch.zeros_like(self.mask[...,0]);fallback=carried.clone()
  for a,b in engine.control.segments(self.data.times,lo,hi):
   seen=self.mask[:,a:b,:,0].cumsum(1)>0
   carried[:,a:b]=(1-self.mask[:,a:b,:,0])*seen
   fallback[:,a:b]=(1-self.mask[:,a:b,:,0])*(~seen)
  self.z=torch.cat([self.z,carried[...,None],fallback[...,None]],-1)
CURRENT={}
original_evaluate=engine.evaluate
def adaptive_evaluate(*args,**kwargs):
 initial=128
 if engine.DEVICE.type=='cuda':
  free,total=torch.cuda.mem_get_info();initial=128 if free>=1024**3 else 32
 for batch in [initial,16,8,1]:
  try:return original_evaluate(*args,**kwargs,batch_size=batch)
  except torch.cuda.OutOfMemoryError:
   if engine.DEVICE.type=='cuda':torch.cuda.empty_cache()
   engine.log('inference_oom_retry',attempted_batch=batch)
 raise RuntimeError('Inference cannot fit even batch1; checkpoints preserved')
engine.evaluate=adaptive_evaluate
def create(name,a,device):return DLinearEMAMask(**CURRENT[name]).to(device)
def candidates(name):
 base=dict(variant=name)
 if name=='E1':return [base|dict(kernel=k) for k in [3,7,13,25]]
 if name in ['E2','E5']:
  return [base|dict(alpha=a,learned_alpha=l,age=age) for a,l,age in itertools.product([.1,.3,.7],[False],(['log','linear_capped'] if name=='E5' else ['log']))]+[base|dict(alpha=.3,learned_alpha=True,age='log')]
 if name=='E4':return [base|dict(age=a) for a in ['log','linear_capped']]
 return [base]
def main():
 assert (ROOT/'results/reproduccion/COMPLETED.json').exists(),'Reproduction must finish first'
 (ROOT/'results/current_stage.json').write_text(json.dumps(dict(stage='ablaciones',pid=os.getpid(),progress='results/mejoras/progress.jsonl')))
 import shutil
 shutil.copy2(ROOT/'results/reproduccion/data_verified.npz',OUT/'data_verified.npz')
 engine.DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 if engine.DEVICE.type=='cuda':
  probe=torch.ones(32,device=engine.DEVICE);assert (probe@probe).item()==32
 data=engine.Data();engine.create=create;choices={};tuning=[]
 protocol=dict(split=[.7,.1,.2],history=12,variants=['E0','E1','E2','E3','E4','E5'],selection='validation only',exploratory=True,train_seeds=engine.SEEDS,test_mask_seeds=[90001+100003*r for r in range(20)],baseline='fresh reproduction DLinear, unchanged',candidate_budget='all prespecified candidates x two learning rates; E5 has greater search budget; disclose',alpha_scope='shared across nodes, selected separately per resource; fixed or one learned scalar',precision='float32; original stable clipping rule',batch=16,device=str(engine.DEVICE),batch_policy='train16 preserves baseline update budget; resident GPU cache means no DataLoader workers; evaluation128 if free VRAM >=1GiB, otherwise32; OOM retries16/8/1 recorded',code_sha256={str(p.relative_to(ROOT)):__import__('hashlib').sha256(p.read_bytes()).hexdigest() for folder in ['src/improved','scripts','tests'] for p in (ROOT/folder).glob('*.py')})
 engine.atomic_json(OUT/'protocol.json',protocol)
 for r in range(2):
  train=Prepared(data,r,'train',70001,engine.control.sample_indices(data.indices['train'],512));val=Prepared(data,r,'validation',80001,engine.control.sample_indices(data.indices['validation'],32))
  for name in ['E1','E2','E3','E4','E5']:
   scores=[]
   for c,spec in enumerate(candidates(name)):
    CURRENT[name]=spec
    for lr in [.001,.003]:
     model,ck=engine.train(name,data,r,train,val,101,lr,3,None,OUT/'tuning'/f'{name}_r{r}_c{c}_lr{lr}.pt')
     model.load_state_dict(ck['state']);score=float(engine.evaluate(name,model,val)['MASE'].mean())
     record=dict(model=name,resource=engine.RESOURCES[r],candidate=c,spec=spec,lr=lr,validation_MASE=score);scores.append(record);tuning.append(record)
   choices[f'{name}_r{r}']=min(scores,key=lambda x:(x['validation_MASE'],x['candidate'],x['lr']))
  del train,val
 engine.atomic_json(OUT/'selected.json',dict(choices=choices,tuning=tuning))
 frozen=[]
 for r in range(2):
  tr=Prepared(data,r,'train',70001);va=Prepared(data,r,'validation',80001)
  for name in ['E1','E2','E3','E4','E5']:
   chosen=choices[f'{name}_r{r}'];CURRENT[name]=chosen['spec']
   for seed in engine.SEEDS:
    model,ck=engine.train(name,data,r,tr,va,seed,chosen['lr'],20,5,OUT/'checkpoints'/f'{name}_r{r}_s{seed}.pt')
    frozen.append(dict(model=name,r=r,seed=seed,state_sha256=ck['state_sha256'],spec=CURRENT[name],parameters=ck['parameter_count']))
  del tr,va
 engine.atomic_json(OUT/'frozen.json',dict(models=frozen,no_test_used_for_selection=True))
 engine.MODELS=['E1','E2','E3','E4','E5']
 # Use the proven evaluator; a lightweight reference-shaped freeze record gates it.
 engine.atomic_json(OUT/'run_config.json',protocol)
 engine.atomic_json(OUT/'frozen_models.json',dict(models=frozen,no_test_metrics_used=True,config_sha256=engine.filehash(OUT/'run_config.json')))
 for r in range(2):
  for name in engine.MODELS:CURRENT[name]=choices[f'{name}_r{r}']['spec']
  # engine.test_all spans both resources; dispatch ensures resource-specific specs.
 # Explicit test loop avoids any resource inference or checkpoint reuse ambiguity.
 import csv
 (OUT/'test').mkdir(exist_ok=True)
 for r in range(2):
  for rep in range(20):
   marker=OUT/'test'/f'complete_r{r}_rep{rep}.json'
   if marker.exists():continue
   ds=Prepared(data,r,'test',90001+100003*rep);rows=[]
   for name in engine.MODELS:
    CURRENT[name]=choices[f'{name}_r{r}']['spec']
    for seed in engine.SEEDS:
     ck=torch.load(OUT/'checkpoints'/f'{name}_r{r}_s{seed}.pt',map_location='cpu',weights_only=False)
     model=create(name,data.a,engine.DEVICE);model.load_state_dict(ck['best_state'])
     metric=engine.evaluate(name,model,ds,OUT/'test'/f'{name}_r{r}_s{seed}_rep{rep}.npz')
     for c,(q,pattern) in enumerate(engine.CASES):
      for n,service in enumerate(data.services):rows.append(dict(model=name,resource=engine.RESOURCES[r],initialization=seed,realization=rep,q=q,pattern=pattern,service=service,**{m:float(metric[m][c,n]) for m in ['MAE','RMSE','MASE']}))
   path=OUT/'test'/f'metrics_r{r}_rep{rep}.csv'
   with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
   engine.atomic_json(marker,dict(rows=len(rows),sha256=engine.filehash(path)))
   engine.log('ablation_test_complete',r=r,realization=rep)
 engine.atomic_json(OUT/'COMPLETED.json',dict(status='complete',from_scratch=True,variants=engine.MODELS))
if __name__=='__main__':main()

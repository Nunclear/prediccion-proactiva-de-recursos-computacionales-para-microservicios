"""Prespecified compact ablations. Freeze all choices before evaluating test."""
import os,sys,json,csv,shutil,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
sys.path[:0]=[str(ROOT/'src'),str(BASE/'src'),str(BASE/'src/baseline/reference'),str(BASE/'scripts')]
os.environ['CHRONOGRAPH_OUTPUT']=str(ROOT/'results/nuevos_experimentos')
import run as e
import ablaciones as adapter
import torch,numpy as np
from compact import CompactDLinear
OUT=e.OUT
FAMILIES={'C0':(False,False),'C1':(True,False),'C2':(False,True),'C3':(True,True)}
CURRENT={}
def create(name,a,device):return CompactDLinear(**CURRENT[name]).to(device)
def utc():return datetime.now(timezone.utc).isoformat()
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 (ROOT/'results/current_stage.json').write_text(json.dumps(dict(stage='v2_compacta',pid=os.getpid(),progress='results/nuevos_experimentos/progress.jsonl')))
 # Importing adapter sets the environment only; the already imported runner OUT is retained.
 assert e.OUT==OUT
 shutil.copy2(BASE/'results/reproduccion/data_verified.npz',OUT/'data_verified.npz')
 data=e.Data();e.create=create
 protocol=dict(created_utc=utc(),families={k:dict(availability=v[0],last_age=v[1]) for k,v in FAMILIES.items()},
   backends=['ma25','ema03','ema07'],learning_rates=[.001,.003],seeds=e.SEEDS,
   test_mask_seeds=[90001+100003*r for r in range(20)],split=data.cuts,history=12,horizon_seconds=1800,
   selection='same budget per family: 3 backends x 2 LR, epoch3 validation512/32; family selection by mean full best validation over seeds101/202/303, ties parameter count then family name',
   inference_selection='one family per resource, frozen before test; no dynamic routing or test targets',
   exploratory=True,source_sha256={str(p.relative_to(ROOT)):e.filehash(p) for folder in ['src','scripts','tests'] for p in (ROOT/folder).glob('*.py')})
 path=OUT/'protocol.json'
 if not path.exists():e.atomic_json(path,protocol)
 else:
  old=e.loadjson(path)
  for key in ['families','backends','learning_rates','seeds','source_sha256']:assert old[key]==protocol[key],key
 choices={};records=[]
 for r in range(2):
  tr=adapter.Prepared(data,r,'train',70001,e.control.sample_indices(data.indices['train'],512))
  va=adapter.Prepared(data,r,'validation',80001,e.control.sample_indices(data.indices['validation'],32))
  for name,(availability,age) in FAMILIES.items():
   scores=[]
   for backend in protocol['backends']:
    spec=dict(backend=backend,availability=availability,age=age);CURRENT[name]=spec
    for lr in protocol['learning_rates']:
     model,ck=e.train(name,data,r,tr,va,101,lr,3,None,OUT/'tuning'/f'{name}_r{r}_{backend}_lr{lr}.pt')
     model.load_state_dict(ck['state']);score=float(e.evaluate(name,model,va)['MASE'].mean())
     record=dict(model=name,r=r,spec=spec,lr=lr,validation_MASE=score);scores.append(record);records.append(record)
   choices[f'{name}_r{r}']=min(scores,key=lambda x:(x['validation_MASE'],x['spec']['backend'],x['lr']))
  del tr,va
 e.atomic_json(OUT/'selected_candidates.json',dict(choices=choices,records=records))
 frozen=[];family_scores={}
 for r in range(2):
  tr=adapter.Prepared(data,r,'train',70001);va=adapter.Prepared(data,r,'validation',80001)
  for name in FAMILIES:
   chosen=choices[f'{name}_r{r}'];CURRENT[name]=chosen['spec'];scores=[]
   for seed in e.SEEDS:
    model,ck=e.train(name,data,r,tr,va,seed,chosen['lr'],20,5,OUT/'checkpoints'/f'{name}_r{r}_s{seed}.pt')
    scores.append(ck['best']);frozen.append(dict(model=name,r=r,seed=seed,spec=CURRENT[name],state_sha256=ck['state_sha256'],parameters=ck['parameter_count']))
   family_scores[f'{name}_r{r}']=dict(score=float(np.mean(scores)),per_seed=scores,parameters=ck['parameter_count'])
  del tr,va
 router={str(r):min(FAMILIES,key=lambda n:(family_scores[f'{n}_r{r}']['score'],family_scores[f'{n}_r{r}']['parameters'],n)) for r in range(2)}
 e.atomic_json(OUT/'frozen_v2.json',dict(frozen_utc=utc(),models=frozen,family_validation=family_scores,router=router,no_test_metrics_used=True,protocol_sha256=e.filehash(path)))
 e.log('v2_frozen_before_test',router=router)
 (OUT/'test').mkdir(exist_ok=True)
 for r in range(2):
  for rep in range(20):
   marker=OUT/'test'/f'complete_r{r}_rep{rep}.json'
   if marker.exists():continue
   ds=adapter.Prepared(data,r,'test',90001+100003*rep);rows=[]
   e.atomic_json(OUT/'test'/f'masks_r{r}_rep{rep}.json',dict(seed=90001+100003*rep,masks=ds.audit))
   for name in FAMILIES:
    CURRENT[name]=choices[f'{name}_r{r}']['spec']
    for seed in e.SEEDS:
     ck=torch.load(OUT/'checkpoints'/f'{name}_r{r}_s{seed}.pt',map_location='cpu',weights_only=False)
     model=create(name,data.a,e.DEVICE);model.load_state_dict(ck['best_state'])
     metric=e.evaluate(name,model,ds,OUT/'test'/f'{name}_r{r}_s{seed}_rep{rep}.npz')
     for c,(q,pattern) in enumerate(e.CASES):
      for n,service in enumerate(data.services):rows.append(dict(model=name,resource=e.RESOURCES[r],initialization=seed,realization=rep,q=q,pattern=pattern,service=service,**{m:float(metric[m][c,n]) for m in ['MAE','RMSE','MASE']}))
   p=OUT/'test'/f'metrics_r{r}_rep{rep}.csv'
   with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
   e.atomic_json(marker,dict(rows=len(rows),sha256=e.filehash(p)));e.log('v2_test_complete',r=r,realization=rep)
   del ds
 e.atomic_json(OUT/'COMPLETED.json',dict(status='complete',utc=utc(),fits=24,prediction_files=480,router=router))
if __name__=='__main__':
 try:main()
 except BaseException as ex:
  OUT.mkdir(parents=True,exist_ok=True);e.atomic_json(OUT/'FAILURE.json',dict(utc=utc(),error=repr(ex)));raise

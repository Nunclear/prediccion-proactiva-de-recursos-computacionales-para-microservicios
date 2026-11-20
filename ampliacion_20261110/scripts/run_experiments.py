"""Source-only validation; new-service metrics are released after model freezing."""
import sys,os,json,time,argparse,traceback,gc
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from core import *
from models import *
from engine import train,evaluate,sample,ridge_fit,save_ck,log
import csv
SEEDS=[101,202,303,404,505,606,707,808,909,1010]
GRID=[dict(lr=lr,weight_decay=wd) for lr in [.001,.003] for wd in [0.,.001]]
def utc():return datetime.now(timezone.utc).isoformat()
def setting(L,h):return f'L{L}_H{h}'
def models_for(L,h):return NEURAL if (L,h)==(12,1) else ['DLinear','E5','A3','PatchTST']
def freeze_protocol():
 import yaml
 p=ROOT/'configs/protocolo_congelado.yaml'
 training_files=['src/core.py','src/models.py','src/engine.py','scripts/run_experiments.py','tests/test_core.py','tests/test_cuda_replay.py','tests/test_cuda_lifecycle.py','tests/test_resume.py']
 protocol=dict(version='20261110-prospective-spatial-transfer-v3',frozen_utc=utc(),
  historical='Preserve original28 services, 70/10/20, E0-E5/C0-C3/V2 and original comparators; no retrospective replacement.',
  source_dataset='chronograph_source.npz',holdout_dataset='chronograph_holdout.npz',
  source_sha256=filehash(ROOT/'data/processed/chronograph_source.npz'),holdout_sha256=filehash(ROOT/'data/processed/chronograph_holdout.npz'),
  cohort_selection=json.loads((ROOT/'configs/cohort_services.json').read_text()),
  inference_scope='32 new services, same infrastructure/period; target-prefix normalization only; no target training/selection. Prospective conditional algorithm comparison, not new-infrastructure population confirmation.',
  primary=dict(history=12,horizon_steps=1,horizon_minutes=30,models=NEURAL+SIMPLE,train_seeds=SEEDS,mask_realizations=20,endpoint='equal mean MASE over services, six original positive-loss conditions and20 masks'),
  secondary=dict(design='star, not full factorial',settings=[[12,h] for h in [2,4,6,12]]+[[L,1] for L in [24,48]],models=['DLinear','E5','A3','PatchTST']+SIMPLE,seeds=SEEDS[:3],mask_realizations=5,status='exploratory; shared holdout cohort'),
  external=dict(source='Alibaba2022 official MSMetricsUpdate objects0..47, first24hours',selection='32 names SHA-ranked among names covering all initial object timestamps',aggregation='mean across pods at each timestamp then mean within completed30min bucket; published normalized utilization, not bytes or cores',calibration='first12 buckets only; no external fitting/hyperparameter selection',evaluation='targets buckets12..47, prior calibration histories permitted; direct30-minute endpoint; exclude graph model because graph not downloaded',scope='independent trace but only24hours and changed target semantics; exploratory calibrated transfer; no population confidence claim'),
  masks=dict(cases=CASES,seeds=[90001+100003*r for r in range(20)],train_seed=70001,validation_seed=80001,correlated='shared priority in lexicographic groups of8',prolonged='blocks up to24 steps',MAR='probability .2*(1+.5*tanh(abs(last observed normalized CPU))); actual rate reported, not forced20%; uses no current hidden or future outcome'),
  tuning=dict(candidates=GRID,epochs=3,train_origins=512,val_origins=64,score='epoch3 mean MASE over10conditions on source validation only',simple_candidates=dict(MovingAverage=[1,3,6,12],SES=[.1,.3,.7,.9],Ridge=[0.,.0001,.01,1.]),budget_note='4 candidates per trained model; analytic/no-training references differ in compute, fully disclosed'),
  training=dict(epochs=20,patience=5,optimizer='Adam',effective_batch=16,microbatch=8,gradient_clip=1.,precision='float32',cpu_threads=2,oom='restart incomplete epoch from checkpoint with halved microbatch; BN differences disclosed if used',device=str(DEVICE),GRUD_acceleration='CUDA graph replay; same kernels, bitwise gradient/weight/prediction checks passed before target evaluation; reused bounded stream, explicit capture cleanup and lifecycle check; no change to optimizer or learning rule'),
  statistical_plan=dict(primary_contrasts=[['DLinear','A3'],['E5','A3']],resources=RESOURCES,seeds=10,sign_test='exact two-sided binomial sign test over paired seeded-training mean differences; conditional on fixed trace/masks; ties excluded',multiplicity='Holm across4 primary tests only',uncertainty='2000 crossed seed/mask/circular temporal-block resamples, fixed services averaged; block lengths24/48/96, primary48; no claim of infrastructure population CI',bootstrap_seed=61110,secondary='descriptive only'),
  budgets=dict(max_parallel_gpu_jobs=1,torch_allocator_fraction=.28,min_free_disk_GiB=50,max_external_objects=48,max_external_object_bytes=1024**3,download_workers=2,aggregation_workers=2,estimated_final_fits=364,estimated_tuning_fits=280,estimated_prediction_storage_GiB=15),
  software=dict(torch=str(torch.__version__),numpy=str(np.__version__),cuda=torch.version.cuda),
  source_sha256_files={name:filehash(ROOT/name) for name in training_files},patchtst=json.loads((ROOT/'data/manifests/patchtst_source.json').read_text()))
 protocol=json.loads(json.dumps(protocol))
 if p.exists():
  old=yaml.safe_load(p.read_text())
  for key in ['source_sha256','holdout_sha256','source_sha256_files','primary','secondary','masks','tuning','statistical_plan']:assert old[key]==protocol[key],f'Protocol changed: {key}'
 else:p.write_text(yaml.safe_dump(protocol,sort_keys=False,allow_unicode=True))
 return protocol
def run_setting(data,target,L,h):
 key=setting(L,h);folder=ROOT/'experiments'/('baseline' if (L,h)==(12,1) else 'ablations')/key;folder.mkdir(parents=True,exist_ok=True)
 models=models_for(L,h);seeds=SEEDS if (L,h)==(12,1) else SEEDS[:3];reps=20 if (L,h)==(12,1) else 5
 if (folder/'COMPLETED.json').exists():return
 selected={};frozen=[];failures=[]
 for r in range(2):
  tr=Prepared(data,r,'train',L,h,70001,subset=sample(data.indices('train',L,h),512));va=Prepared(data,r,'validation',L,h,80001,subset=sample(data.indices('validation',L,h),64))
  for name in models:
   scores=[]
   for idx,hyp in enumerate(GRID):
    try:
     m,ck=train(name,data,L,h,r,tr,va,101,hyp['lr'],hyp['weight_decay'],3,None,folder/'tuning'/f'{name}_r{r}_c{idx}.pt')
     m.load_state_dict(ck['state']);score=float(np.nanmean(evaluate(m,va)['MASE']))
     scores.append(dict(candidate=idx,**hyp,validation_MASE=score));del m
    except Exception as ex:failures.append(dict(stage='tuning',model=name,r=r,candidate=idx,error=repr(ex)));log('experiment_failure',setting=key,**failures[-1])
   if scores:selected[f'{name}_r{r}']=min(scores,key=lambda x:(x['validation_MASE'],x['candidate']))
  del tr,va;gc.collect()
 atomic_json(folder/'selection.json',dict(choices=selected,no_target_metrics_used=True))
 for r in range(2):
  tr=Prepared(data,r,'train',L,h,70001);va=Prepared(data,r,'validation',L,h,80001)
  for name in models:
   choice=selected.get(f'{name}_r{r}')
   if choice is None:continue
   for seed in seeds:
    try:
     m,ck=train(name,data,L,h,r,tr,va,seed,choice['lr'],choice['weight_decay'],20,5,folder/'checkpoints'/f'{name}_r{r}_s{seed}.pt')
     frozen.append(dict(model=name,r=r,seed=seed,file=f'{name}_r{r}_s{seed}.pt',state_sha256=ck['state_sha256'],parameters=ck['parameters'],best_source_validation_MASE=ck['best'],epochs=len(ck['history'])))
     del m
    except Exception as ex:failures.append(dict(stage='final_fit',model=name,r=r,seed=seed,error=repr(ex)));log('experiment_failure',setting=key,**failures[-1])
  for name in SIMPLE:
   options={'Persistence':[None],'MovingAverage':[1,3,6,12],'SES':[.1,.3,.7,.9],'Ridge':[0.,.0001,.01,1.]}[name];scores=[];best=None
   for idx,param in enumerate(options):
    model=ridge_fit(tr,param) if name=='Ridge' else create(name,L,data.a,DEVICE,param)
    score=float(np.nanmean(evaluate(model,va)['MASE']))
    scores.append(dict(candidate=idx,parameter=param,validation_MASE=score))
    if best is None or score<best[0]:best=(score,param,state_cpu(model))
   ck=dict(signature=dict(model=name,L=L,h=h,r=r,seed=0),state=best[2],best_state=best[2],parameter=best[1],best=best[0],complete=True,parameters=sum(p.numel() for p in model.parameters()),state_sha256=state_hash(best[2]),history=[],candidates=scores)
   save_ck(folder/'checkpoints'/f'{name}_r{r}_s0.pt',ck)
   frozen.append(dict(model=name,r=r,seed=0,file=f'{name}_r{r}_s0.pt',state_sha256=ck['state_sha256'],parameters=ck['parameters'],best_source_validation_MASE=ck['best']))
  del tr,va;gc.collect()
 atomic_json(folder/'frozen.json',dict(frozen_utc=utc(),models=frozen,failures=failures,no_target_metrics_used=True,protocol_sha256=filehash(ROOT/'configs/protocolo_congelado.yaml')))
 log('models_frozen_before_holdout',setting=key,models=len(frozen),failures=len(failures))
 run_test(folder,target,L,h,frozen,reps,'new_services')
 atomic_json(folder/'COMPLETED.json',dict(status='complete' if not failures else 'complete_with_failures',L=L,h=h,neural_fits=sum(x['seed']!=0 for x in frozen),failures=failures))
def run_test(folder,target,L,h,frozen,reps,label,allowed=None):
 test=folder/'test'/label;test.mkdir(parents=True,exist_ok=True)
 for r in range(2):
  for rep in range(reps):
   marker=test/f'complete_r{r}_rep{rep}.json'
   if marker.exists():continue
   ds=Prepared(target,r,'test',L,h,90001+100003*rep);rows=[]
   atomic_json(test/f'masks_r{r}_rep{rep}.json',dict(audit=ds.audit,cohort=target.services))
   for record in frozen:
    if record['r']!=r or (allowed is not None and record['model'] not in allowed):continue
    name=record['model'];seed=record['seed'];ck=torch.load(folder/'checkpoints'/record['file'],map_location='cpu',weights_only=False)
    assert state_hash(ck['best_state'])==record['state_sha256']
    model=create(name,L,target.a,DEVICE,ck.get('parameter'));load_for_target(model,ck['best_state'])
    p=test/f'{name}_r{r}_s{seed}_rep{rep}.npz';scores=evaluate(model,ds,save_to=p)
    for c,(q,pattern) in enumerate(CASES):
     for n,service in enumerate(target.services):
      rows.append(dict(dataset=label,model=name,resource=RESOURCES[r],L=L,h_steps=h,h_minutes=h*30,seed=seed,realization=rep,q=q,pattern=pattern,service=service,n_targets=int(np.isfinite(target.values[ds.indices,n,r]).sum()),**{m:float(scores[m][c,n]) for m in ['MAE','RMSE','MASE']}))
    del model
   if rows:
    path=test/f'metrics_r{r}_rep{rep}.csv'
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    atomic_json(marker,dict(utc=utc(),rows=len(rows),sha256=filehash(path)));log('target_realization_complete',setting=setting(L,h),dataset=label,r=r,realization=rep,rows=len(rows))
   del ds;gc.collect()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['freeze','primary','secondary','external','all'],default='all');args=parser.parse_args()
 if DEVICE.type=='cuda':torch.cuda.set_per_process_memory_fraction(.28);x=torch.ones(32,device=DEVICE);assert (x@x).item()==32
 (ROOT/'results/current_stage.json').write_text(json.dumps(dict(stage='advanced_experiments',pid=os.getpid(),progress='results/progress.jsonl')))
 protocol=freeze_protocol()
 if args.stage=='freeze':return
 source=Series(ROOT/'data/processed/chronograph_source.npz');target=Series(ROOT/'data/processed/chronograph_holdout.npz')
 if args.stage in ['primary','all']:run_setting(source,target,12,1)
 if args.stage in ['secondary','all']:
  for L,h in protocol['secondary']['settings']:run_setting(source,target,L,h)
 if args.stage in ['external','all']:
  external=ROOT/'data/processed/alibaba_transfer.npz';folder=ROOT/'experiments/baseline/L12_H1'
  if external.exists() and (folder/'frozen.json').exists():
   frozen=json.loads((folder/'frozen.json').read_text())['models'];data=Series(external,external=True)
   run_test(folder,data,12,1,frozen,20,'alibaba_transfer',allowed=[n for n in NEURAL+SIMPLE if n!='GRU_graph'])
   atomic_json(ROOT/'experiments/external_validation/COMPLETED.json',dict(status='complete',scope=protocol['external'],n_test_origins=len(data.indices('test',12,1))))
  else:atomic_json(ROOT/'experiments/external_validation/PENDING.json',dict(status='not_evaluated',reason='processed external data or source model freeze unavailable'))
if __name__=='__main__':main()

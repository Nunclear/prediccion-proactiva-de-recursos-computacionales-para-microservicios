"""Independent saved-prediction audit and predeclared paired analyses, CPU only."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import json,hashlib,re,math,time,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/analysis'
RESOURCES=['cpu_usage','container_memory_usage_bytes']
PATTERNS=['disperso','bloques']
SEEDS=[101,202,303,404,505,606,707,808,909,1010]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(path,item):
 def clean(x):
  if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
  if isinstance(x,(list,tuple)):return [clean(v) for v in x]
  if isinstance(x,np.generic):return clean(x.item())
  if isinstance(x,float) and not np.isfinite(x):return None
  return x
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(clean(item),indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(path)
def load_data(label):
 p=ROOT/'data/processed'/('alibaba_transfer.npz' if label=='alibaba_transfer' else 'chronograph_holdout.npz')
 with np.load(p) as z:a={k:z[k] for k in z.files}
 x=a['values'].astype(np.float64);T=len(x);cut=12 if label=='alibaba_transfer' else int(.7*T)
 prefix=x[:cut];a.update(mean=np.nanmean(prefix,0),scale=np.maximum(np.nanstd(prefix,axis=0),1e-8))
 step=int(np.median(np.diff(a['times'])));pairs=np.diff(a['times'][:cut])==step
 a['denominator']=np.nanmean(np.abs(np.diff(prefix,axis=0))[pairs],axis=0);a['step']=step
 return a
def expected_indices(data,label,L,h):
 T=len(data['times']);lo=0 if label=='alibaba_transfer' else int(.8*T)
 cuts=[lo]+(np.flatnonzero(np.diff(data['times'][lo:])!=data['step'])+lo+1).tolist()+[T]
 ix=np.concatenate([np.arange(a+L+h-1,b,dtype=np.int64) for a,b in zip(cuts,cuts[1:])])
 return ix[ix>=12] if label=='alibaba_transfer' else ix
def audit_all():
 frames=[];checks=[];maxerr=0.;count=0;common_rows=[];mask_rows=[];mask_cache={};mask_data={}
 sys.path.insert(0,str(ROOT/'src'));from core import Series,mask_for
 chrono=load_data('new_services');common_indices=expected_indices(chrono,'new_services',12,1)
 for Lc,hc in [(12,2),(12,4),(12,6),(12,12),(24,1),(48,1)]:common_indices=np.intersect1d(common_indices,expected_indices(chrono,'new_services',Lc,hc))
 caches={};primary_errors={}
 for csvpath in sorted((ROOT/'experiments').glob('**/test/*/metrics_r*_rep*.csv')):
  df=pd.read_csv(csvpath);label=csvpath.parent.name
  assert not df.duplicated(['model','resource','seed','q','pattern','service']).any(),csvpath
  r=int(re.search(r'_r(\d)_',csvpath.name)[1]);rep=int(re.search(r'_rep(\d+)',csvpath.name)[1]);L=int(df.L.iloc[0]);h=int(df.h_steps.iloc[0]);assert df.realization.eq(rep).all()
  if label not in caches:caches[label]=load_data(label)
  assert df.L.eq(L).all() and df.h_steps.eq(h).all() and df.h_minutes.eq(h*30).all()
  assert df.dataset.eq(label).all() and df.resource.eq(RESOURCES[r]).all()
  data=caches[label];expected=expected_indices(data,label,L,h)
  cases=list(dict.fromkeys(zip(df.q,df.pattern)))
  assert len(cases)==10
  saved_masks=json.loads((csvpath.parent/f'masks_r{r}_rep{rep}.json').read_text());assert saved_masks['cohort']==data['services'].tolist()
  if label not in mask_data:mask_data[label]=Series(ROOT/'data/processed'/('alibaba_transfer.npz' if label=='alibaba_transfer' else 'chronograph_holdout.npz'),external=label=='alibaba_transfer')
  for entry in saved_masks['audit']:
   q=int(entry['q_nominal']);pattern=entry['pattern'];key=(label,rep,q,pattern)
   if key not in mask_cache:
    series=mask_data[label];observed=mask_for(series,'test',q,pattern,90001+100003*rep);lo,hi=series.bounds['test']
    if pattern!='MAR_lag':assert np.all(np.sum(~observed[lo:hi],axis=0)==int(np.rint(q*(hi-lo)/100)))
    mask_cache[key]=dict(sha256=digest(observed),induced=100*float((~observed[lo:hi]).mean()),total=[100*float((~(observed[lo:hi]&np.isfinite(series.values[lo:hi,:,rr]))).mean()) for rr in [0,1]])
   regenerated=mask_cache[key]
   assert regenerated['sha256']==entry['induced_mask_sha256']
   assert abs(regenerated['induced']-entry['induced_missing_percent'])<1e-12 and abs(regenerated['total'][r]-entry['total_missing_percent'])<1e-12
   mask_rows.append(dict(dataset=label,L=L,h_steps=h,resource=RESOURCES[r],realization=rep,q_nominal=q,pattern=pattern,induced_missing_percent=regenerated['induced'],total_missing_percent=regenerated['total'][r],mask_sha256=regenerated['sha256']))
  for (model,seed),group in df.groupby(['model','seed'],sort=False):
   p=csvpath.parent/f'{model}_r{r}_s{seed}_rep{rep}.npz'
   with np.load(p) as z:
    pred=z['prediction_normalized'];ix=z['indices'];pred_sha=str(z['prediction_sha256']);target_sha=str(z['target_sha256'])
   assert np.array_equal(ix,expected),(p,'chronology')
   assert np.isfinite(pred).all() and pred.shape==(10,len(ix),len(data['services']))
   truth=data['values'][ix,:,r];assert digest(truth)==target_sha and digest(pred)==pred_sha,(p,'array digest')
   err=pred.astype(np.float64)*data['scale'][:,r]+data['mean'][:,r]-truth[None]
   with np.errstate(invalid='ignore',divide='ignore'):
    mae=np.nanmean(np.abs(err),axis=1);rmse=np.sqrt(np.nanmean(err*err,axis=1));den=data['denominator'][:,r]
    mase=np.divide(mae,den,out=np.full_like(mae,np.nan),where=den>0)
   indexed=group.set_index(['q','pattern','service'])
   for c,case in enumerate(cases):
    actual=indexed.loc[[(*case,s) for s in data['services']]]
    for metric,arr in [('MAE',mae),('RMSE',rmse),('MASE',mase)]:
     a=actual[metric].to_numpy();b=arr[c];assert np.allclose(a,b,rtol=1e-10,atol=1e-10,equal_nan=True),(p,metric)
     finite=np.isfinite(a)&np.isfinite(b)
     if finite.any():maxerr=max(maxerr,float(np.max(np.abs(a[finite]-b[finite]))))
    assert np.array_equal(actual.n_targets.to_numpy(),np.isfinite(truth).sum(0))
   if label=='new_services' and model in ['DLinear','E5','A3','PatchTST','Persistence','MovingAverage','SES','Ridge'] and int(seed) in [0,101,202,303] and rep<5:
    use=np.isin(ix,common_indices);assert use.sum()==len(common_indices)
    common_mae=np.mean(np.abs(err[:,use]),axis=1);common_rmse=np.sqrt(np.mean(err[:,use]**2,axis=1));common_mase=common_mae/den
    for c in range(1,7):
     q,pattern=cases[c]
     for n,service in enumerate(data['services']):common_rows.append(dict(dataset=label,model=model,resource=RESOURCES[r],L=L,h_steps=h,h_minutes=h*30,seed=int(seed),realization=rep,q=q,pattern=pattern,service=service,n_targets=int(use.sum()),MAE=float(common_mae[c,n]),RMSE=float(common_rmse[c,n]),MASE=float(common_mase[c,n]),scope='exploratory common target origins; primary endpoint remains unchanged'))
   if label=='new_services' and (L,h)==(12,1) and model in ['DLinear','E5','A3']:
    assert (den>0).all() and np.isfinite(truth).all()
    # Average services and six original positive-loss conditions, retaining time.
    temporal=(np.abs(err[1:7])/den[None,None,:]).mean(axis=(0,2))
    primary_errors[(r,model,int(seed),rep)]=temporal
   count+=1
   checks.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),prediction_array_sha256=pred_sha,target_array_sha256=target_sha,rows=len(group)))
  df['target_units']=np.where(df.dataset.eq('alibaba_transfer'),'publisher min-max normalized utilization',np.where(df.resource.eq('cpu_usage'),'publisher CPU scale','bytes'))
  df['target_semantics']=np.where(df.dataset.eq('alibaba_transfer'),'mean pod utilization per timestamp then completed 30-minute mean','published point sample')
  frames.append(df)
  if len(frames)%10==0:print('AUDITED',len(frames),'metric files',count,'NPZ',flush=True)
 assert frames,'No completed evaluations'
 pd.DataFrame(mask_rows).to_csv(OUT/'mask_audits.csv',index=False)
 write(OUT/'mask_audit.json',dict(status='passed',manifest_records=len(mask_rows),unique_dataset_replication_case_keys=len(mask_cache),method='Regenerate with frozen mask code and seeds, verify saved digests/rates, exact counts except declared lag-MAR; repeated settings/resources are paired, not independent draws.'))
 pd.DataFrame(common_rows).to_csv(OUT/'common_origin_metrics.csv.gz',index=False)
 write(OUT/'common_origins.json',dict(status='exploratory secondary common-support summary',n_origins=len(common_indices),first_target_row=int(common_indices[0]),last_target_row=int(common_indices[-1]),seeds=[101,202,303],mask_realizations=5,unchanged_primary=True))
 allmetrics=pd.concat(frames,ignore_index=True)
 allmetrics.to_csv(OUT/'metrics_all.csv.gz',index=False)
 allmetrics[['dataset','resource','target_units','target_semantics']].drop_duplicates().to_csv(OUT/'target_semantics.csv',index=False)
 pd.DataFrame(checks).to_csv(OUT/'prediction_manifest.csv',index=False)
 write(OUT/'audit.json',dict(status='passed',metric_csv_files=len(frames),npz_files=count,metric_rows=len(allmetrics),max_absolute_metric_reconstruction_difference=maxerr,protocol_sha256=sha(ROOT/'configs/protocolo_congelado.yaml'),scope='All saved new-study predictions; metric equations independently reconstructed in float64; historical artifacts preserved separately.'))
 return allmetrics,primary_errors
def exact_sign(d):
 d=np.asarray(d);d=d[d!=0];n=len(d);k=int((d>0).sum())
 return min(1.,2*sum(math.comb(n,j) for j in range(min(k,n-k)+1))/2**n) if n else 1.
def crossed_bootstrap(delta,block,B=2000,seed=61110):
 """Independent multinomial seed/mask counts, one shared circular block sample."""
 S,M,T=delta.shape;rng=np.random.default_rng(seed);out=np.empty(B)
 for start in range(0,B,40):
  n=min(40,B-start);ws=rng.multinomial(S,np.ones(S)/S,size=n)/S;wm=rng.multinomial(M,np.ones(M)/M,size=n)/M
  # Exact circular moving-block bootstrap, trim final block to original length.
  origins=rng.integers(0,T,size=(n,math.ceil(T/block)))
  indices=((origins[:,:,None]+np.arange(block))%T).reshape(n,-1)[:,:T]
  weights=np.stack([np.bincount(ix,minlength=T) for ix in indices])/T
  timepaths=np.einsum('bs,bm,smt->bt',ws,wm,delta,optimize=True)
  out[start:start+n]=np.einsum('bt,bt->b',timepaths,weights)
 return np.quantile(out,[.025,.975]).tolist()
def analyze(df,errors):
 primary=df[(df.dataset=='new_services')&(df.L==12)&(df.h_steps==1)&df.pattern.isin(PATTERNS)&df.q.gt(0)]
 grouping=['dataset','L','h_steps','h_minutes','resource','model']
 summary=df.groupby(grouping,as_index=False)[['MAE','RMSE','MASE']].mean();summary.to_csv(OUT/'summary_all_conditions.csv',index=False)
 pmean=primary.groupby(['resource','model'],as_index=False)[['MAE','RMSE','MASE']].mean();pmean.to_csv(OUT/'primary_summary.csv',index=False)
 for name,keys in [('by_condition',grouping+['q','pattern']),('by_service',grouping+['service']),('by_seed',grouping+['seed']),('by_mask',grouping+['realization'])]:
  df.groupby(keys,as_index=False)[['MAE','RMSE','MASE']].mean().to_csv(OUT/(name+'.csv'),index=False)
 variability=[]
 for (resource,model),g in primary.groupby(['resource','model']):
  seeds=g.groupby('seed').MASE.mean();masks=g.groupby('realization').MASE.mean()
  variability.append(dict(resource=resource,model=model,n_seeds=len(seeds),n_masks=len(masks),seed_sd=seeds.std() if len(seeds)>1 else np.nan,mask_sd=masks.std(),mase=float(g.MASE.mean())))
 pd.DataFrame(variability).to_csv(OUT/'variability.csv',index=False)
 period_rows=[]
 data=load_data('new_services');indices=expected_indices(data,'new_services',12,1)
 for r,resource in enumerate(RESOURCES):
  for model in ['DLinear','E5','A3']:
   a=np.stack([[errors[(r,model,s,rep)] for rep in range(20)] for s in SEEDS]);path=a.mean((0,1))
   for start in range(0,len(path),48):
    stop=min(start+48,len(path));period_rows.append(dict(resource=resource,model=model,block_index=start//48,block_length=48,n_origins=stop-start,first_target_row=int(indices[start]),last_target_row=int(indices[stop-1]),first_target_time_seconds=float(data['times'][indices[start]]),last_target_time_seconds=float(data['times'][indices[stop-1]]),MASE=float(path[start:stop].mean()),scope='descriptive fixed-cohort temporal block, not an independent dataset'))
 pd.DataFrame(period_rows).to_csv(OUT/'by_temporal_block.csv',index=False)
 contrasts=[]
 for r,resource in enumerate(RESOURCES):
  for baseline in ['DLinear','E5']:
   arrays={model:np.stack([[errors[(r,model,s,rep)] for rep in range(20)] for s in SEEDS]) for model in [baseline,'A3']}
   delta=arrays[baseline]-arrays['A3'];seed_delta=delta.mean((1,2));base=primary[(primary.resource==resource)&(primary.model==baseline)];prop=primary[(primary.resource==resource)&(primary.model=='A3')]
   keys=['seed','realization','q','pattern','service'];paired=base.merge(prop,on=keys,suffixes=('_base','_proposed'),validate='one_to_one')
   assert len(paired)==10*20*6*32
   reduction=100*(paired.MASE_base-paired.MASE_proposed)/paired.MASE_base.where(paired.MASE_base>0)
   ci={str(b):crossed_bootstrap(delta,b) for b in [24,48,96]}
   contrasts.append(dict(resource=resource,reference=baseline,proposed='A3',mean_MASE_reference=float(arrays[baseline].mean()),mean_MASE_A3=float(arrays['A3'].mean()),mean_paired_MASE_difference=float(delta.mean()),mean_paired_MAE_difference=float((paired.MAE_base-paired.MAE_proposed).mean()),mean_paired_RMSE_difference=float((paired.RMSE_base-paired.RMSE_proposed).mean()),mean_paired_reduction_percent=float(reduction.mean()),paired_reduction_valid_cells=int(np.isfinite(reduction).sum()),zero_reference_cells=int(paired.MASE_base.eq(0).sum()),ratio_of_mean_reduction_percent=100*float(delta.mean()/arrays[baseline].mean()),seed_differences=seed_delta.tolist(),n_nonzero_seeds=int((seed_delta!=0).sum()),positive_seeds=int((seed_delta>0).sum()),paired_seed_standardized_effect=float(seed_delta.mean()/seed_delta.std(ddof=1)) if seed_delta.std(ddof=1)>0 else None,p_sign=exact_sign(seed_delta),leave_tuning_seed_out=dict(excluded_seed=101,status='exploratory sensitivity declared before target-metric inspection',n_seeds=9,n_nonzero_seeds=int((seed_delta[1:]!=0).sum()),mean_paired_MASE_difference=float(seed_delta[1:].mean()),positive_seeds=int((seed_delta[1:]>0).sum()),p_sign=exact_sign(seed_delta[1:])),conditional_crossed_block_CI95=ci,bootstrap_replicates=2000,bootstrap_seed=61110))
 order=sorted(range(4),key=lambda i:contrasts[i]['p_sign']);running=0.
 for rank,i in enumerate(order):
  running=max(running,min(1.,(4-rank)*contrasts[i]['p_sign']));contrasts[i]['p_Holm']=running
 order9=sorted(range(4),key=lambda i:contrasts[i]['leave_tuning_seed_out']['p_sign']);running=0.
 for rank,i in enumerate(order9):
  running=max(running,min(1.,(4-rank)*contrasts[i]['leave_tuning_seed_out']['p_sign']));contrasts[i]['leave_tuning_seed_out']['p_Holm_descriptive']=running
 write(OUT/'primary_contrasts.json',dict(contrasts=contrasts,inference_scope='Paired seeded training randomness and crossed seed/mask/time uncertainty conditional on the fixed services and observed trace. Not population inference across infrastructures. Secondary/external descriptive only.'))
 # Descriptive paired comparisons by service preserve seed/mask/condition matching.
 service_rows=[]
 for resource in RESOURCES:
  for baseline in ['DLinear','E5','A0','A1','A2','PatchTST']:
   b=primary[(primary.resource==resource)&(primary.model==baseline)];a=primary[(primary.resource==resource)&(primary.model=='A3')]
   j=b.merge(a,on=['seed','realization','q','pattern','service'],suffixes=('_base','_A3'),validate='one_to_one')
   j['MASE_difference']=j.MASE_base-j.MASE_A3;j['paired_reduction_percent']=100*j.MASE_difference/j.MASE_base.where(j.MASE_base>0)
   for service,g in j.groupby('service'):service_rows.append(dict(resource=resource,reference=baseline,service=service,MASE_difference=g.MASE_difference.mean(),paired_reduction_percent=g.paired_reduction_percent.mean()))
 pd.DataFrame(service_rows).to_csv(OUT/'paired_by_service.csv',index=False)
 np.savez_compressed(OUT/'primary_time_errors.npz',**{f'r{r}_{m}_s{s}_rep{rep}':v for (r,m,s,rep),v in errors.items()})
 print(json.dumps(contrasts,indent=2),flush=True)
def verify_checkpoints():
 import torch,yaml
 sys.path.insert(0,str(ROOT/'src'));from models import create
 parameter_counts={}
 protocol=yaml.safe_load((ROOT/'configs/protocolo_congelado.yaml').read_text())
 for name,expected in protocol['source_sha256_files'].items():assert sha(ROOT/name)==expected,(name,'frozen source changed')
 for label,key in [('source','source_sha256'),('holdout','holdout_sha256')]:assert sha(ROOT/'data/processed'/f'chronograph_{label}.npz')==protocol[key]
 rows=[]
 for frozen_path in sorted((ROOT/'experiments').glob('**/frozen.json')):
  frozen=json.loads(frozen_path.read_text());assert frozen['protocol_sha256']==sha(ROOT/'configs/protocolo_congelado.yaml') and not frozen['failures']
  for record in frozen['models']:
   p=frozen_path.parent/'checkpoints'/record['file'];ck=torch.load(p,map_location='cpu',weights_only=False);assert ck['complete']
   digest_state=hashlib.sha256()
   for name,value in sorted(ck['best_state'].items()):digest_state.update(name.encode());digest_state.update(value.detach().cpu().numpy().tobytes())
   assert digest_state.hexdigest()==record['state_sha256']==ck['state_sha256'],p
   signature=ck['signature'];key=(record['model'],signature['L'])
   if record['seed']!=0:
    selected=json.loads((frozen_path.parent/'selection.json').read_text())['choices'][f"{record['model']}_r{record['r']}"]
    assert signature['lr']==selected['lr'] and signature['weight_decay']==selected['weight_decay']
    assert len(ck['history'])<=20 and ck['best']==min(x['validation_MASE'] for x in ck['history'])
    assert all(x['effective_batch']==16 and x['microbatch'] in [1,2,4,8] for x in ck['history'])
   else:
    chosen=min(ck['candidates'],key=lambda x:(x['validation_MASE'],x['candidate']))
    assert ck['parameter']==chosen['parameter'] and ck['best']==chosen['validation_MASE']
   fingerprint=hashlib.sha256(json.dumps(dict(signature=signature,fixed_parameter=ck.get('parameter'),tensor_state_sha256=record['state_sha256']),sort_keys=True,separators=(',',':')).encode()).hexdigest()
   if key not in parameter_counts:
    m=create(record['model'],signature['L'],np.eye(1,dtype=np.float32),'cpu',ck.get('parameter'));parameter_counts[key]=(sum(v.numel() for v in m.parameters()),sum(v.numel() for v in m.parameters() if v.requires_grad));del m
   assert parameter_counts[key][0]==record['parameters']
   rows.append(dict(path=str(p.relative_to(ROOT)),model=record['model'],resource=RESOURCES[record['r']],seed=record['seed'],sha256=sha(p),state_sha256=record['state_sha256'],audited_predictor_fingerprint=fingerprint,fixed_parameter=ck.get('parameter'),parameters=record['parameters'],trainable_parameters=parameter_counts[key][1],parameter_scope='parameters column counts all nn.Parameter tensors; trainable_parameters filters requires_grad',epochs=len(ck['history'])))
 assert len(rows)==420 and sum(x['seed']!=0 for x in rows)==364,len(rows)
 pd.DataFrame(rows).to_csv(OUT/'checkpoint_manifest.csv',index=False)
 write(OUT/'checkpoint_audit.json',dict(status='passed',final_checkpoints=len(rows),final_neural_fits=sum(x['seed']!=0 for x in rows),frozen_sources_verified=True,protocol_version=protocol['version']))
 print('FROZEN_CHECKPOINTS_VERIFIED',len(rows),flush=True)
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 # Do not label partial runs complete: every scheduled setting must have its marker.
 for folder in [ROOT/'experiments/baseline/L12_H1']+[ROOT/'experiments/ablations'/f'L{L}_H{h}' for L,h in [(12,2),(12,4),(12,6),(12,12),(24,1),(48,1)]]:
  marker=folder/'COMPLETED.json';assert marker.exists(),f'Pending: {marker}';assert json.loads(marker.read_text())['status']=='complete',f'Fit failures: {marker}'
 external=ROOT/'experiments/external_validation/COMPLETED.json'
 assert external.exists() and json.loads(external.read_text())['status']=='complete','External evaluation pending'
 verify_checkpoints()
 df,errors=audit_all();analyze(df,errors)
 write(OUT/'COMPLETED.json',dict(status='complete',created_unix=time.time(),source_sha256=sha(Path(__file__))))
if __name__=='__main__':main()

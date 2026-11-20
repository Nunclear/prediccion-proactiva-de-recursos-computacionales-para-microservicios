"""Independent prediction audit and descriptive, paired uncertainty in two units."""
import os,sys,json,hashlib,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
sys.path[:0]=[str(BASE/'src/baseline/reference')]
os.environ['CHRONOGRAPH_OUTPUT']=str(BASE/'results/reproduccion')
import run as e
import numpy as np,pandas as pd
OUT=ROOT/'results/comparaciones';OUT.mkdir(exist_ok=True)
def dump(name,x):(OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def main():
 assert (ROOT/'results/nuevos_experimentos/COMPLETED.json').exists()
 data=e.Data();frozen=json.loads((ROOT/'results/nuevos_experimentos/frozen_v2.json').read_text());router=frozen['router']
 # Historical files are checked against the prior authenticated audit; never rewritten.
 previous=json.loads((BASE/'results/estadisticas/validacion.json').read_text());files=[]
 for x in previous['forecasts']:
  p=BASE/'results'/x['experiment']/'test'/x['file'];assert e.filehash(p)==x['sha256'];files.append(dict(file=str(p.relative_to(BASE)),sha256=x['sha256']))
 old=pd.read_csv(BASE/'results/estadisticas/metricas_completas.csv')
 assert len(old)==previous['metric_rows']==243040
 frames=[];fresh_checks=[]
 for path in sorted((ROOT/'results/nuevos_experimentos/test').glob('metrics_*.csv')):
  df=pd.read_csv(path)
  for (name,resource,seed,rep),g in df.groupby(['model','resource','initialization','realization'],sort=False):
   r=e.RESOURCES.index(resource);p=path.parent/f'{name}_r{r}_s{seed}_rep{rep}.npz';a=np.load(p)
   pred=a['prediction_normalized'];ix=a['indices'];truth=data.values[ix,:,r]
   assert np.array_equal(ix,data.indices['test']) and pred.shape==(7,1589,28)
   assert str(a['prediction_sha256'])==e.digest(pred) and str(a['target_sha256'])==e.digest(truth)
   physical=pred.astype(np.float64)*data.scale[:,r]+data.mean[:,r];err=physical-truth[None]
   calc=dict(MAE=np.abs(err).mean(1),RMSE=np.sqrt(np.square(err).mean(1)));calc['MASE']=calc['MAE']/data.mase[:,r]
   for c,(q,pattern) in enumerate(e.CASES):
    rows=g[(g.q==q)&(g.pattern==pattern)].set_index('service').loc[data.services]
    for metric,value in calc.items():np.testing.assert_allclose(rows[metric],value[c],rtol=1e-11,atol=1e-9)
   fresh_checks.append(dict(file=str(p.relative_to(ROOT)),sha256=e.filehash(p)))
  frames.append(df)
 fresh=pd.concat(frames,ignore_index=True);assert len(fresh)==94080 and len(fresh_checks)==480
 alias=pd.concat([fresh[(fresh.resource==resource)&(fresh.model==router[str(r)])].assign(model='V2') for r,resource in enumerate(e.RESOURCES)],ignore_index=True)
 allrows=pd.concat([old,fresh,alias],ignore_index=True);assert len(allrows)==360640
 allrows.to_csv(ROOT/'results/metricas_verificadas/todas_las_metricas.csv',index=False)
 for label,frame in [('historicas',old),('compactas',fresh),('V2',alias)]:frame.to_csv(ROOT/'results/metricas_verificadas'/f'{label}.csv',index=False)
 summary=allrows.groupby(['model','resource','q','pattern'],as_index=False)[['MASE','MAE','RMSE']].mean();summary.to_csv(OUT/'resumen_condiciones.csv',index=False)
 means=allrows.groupby(['model','resource'],as_index=False)[['MASE','MAE','RMSE']].mean();means.to_csv(OUT/'resumen_global.csv',index=False)
 lost=allrows[allrows.q>0];lost.groupby(['model','resource'],as_index=False)[['MASE','MAE','RMSE']].mean().to_csv(OUT/'resumen_perdidas.csv',index=False)
 rng=np.random.default_rng(91001);maskdraw=rng.integers(0,20,(10000,20));seeddraw=np.random.default_rng(91002).integers(0,3,(10000,3))
 pairs=[('E0',x) for x in ['E1','E2','E3','E4','E5','C0','C1','C2','C3','V2']]+[('E2','E5'),('E5','V2'),('E4','V2')]
 comparisons=[];seeds=[];services=[]
 for resource in e.RESOURCES:
  for reference,model in pairs:
   for condition in ['todos','perdidas','disperso','bloques']:
    s=allrows[(allrows.resource==resource)&allrows.model.isin([reference,model])]
    if condition!='todos':s=s[s.q>0]
    if condition in ['disperso','bloques']:s=s[s.pattern==condition]
    pair=s.groupby(['model','initialization','realization']).MASE.mean().unstack('model')
    a=pair[reference].unstack('realization').loc[e.SEEDS,list(range(20))].values
    b=pair[model].unstack('realization').loc[e.SEEDS,list(range(20))].values
    ar=a.mean(0);br=b.mean(0);delta=ar-br;percent=100*delta/ar
    ci_delta=np.quantile(delta[maskdraw].mean(1),[.025,.975]);ci_percent=np.quantile(percent[maskdraw].mean(1),[.025,.975])
    # Crossed resampling: one global seed sample and one global mask sample per draw.
    aa=a[seeddraw[:,:,None],maskdraw[:,None,:]].mean((1,2));bb=b[seeddraw[:,:,None],maskdraw[:,None,:]].mean((1,2))
    crossed_delta=np.quantile(aa-bb,[.025,.975]);crossed_percent=np.quantile(100*(aa-bb)/aa,[.025,.975])
    comparisons.append(dict(resource=resource,reference=reference,model=model,conditions=condition,
      MASE_reference=ar.mean(),MASE_model=br.mean(),delta_MASE=delta.mean(),delta_MASE_low=ci_delta[0],delta_MASE_high=ci_delta[1],
      reduction_percent=percent.mean(),reduction_percent_low=ci_percent[0],reduction_percent_high=ci_percent[1],
      crossed_delta_MASE_low=crossed_delta[0],crossed_delta_MASE_high=crossed_delta[1],crossed_reduction_ratio_of_means_percent=100*(a.mean()-b.mean())/a.mean(),crossed_reduction_percent_low=crossed_percent[0],crossed_reduction_percent_high=crossed_percent[1],
      seeds=3,mask_realizations=20,bootstrap_draws=10000,confidence=.95,inference='exploratory descriptive; fixed services and historical test'))
    if condition=='perdidas':
     for i,seed in enumerate(e.SEEDS):seeds.append(dict(resource=resource,reference=reference,model=model,seed=seed,MASE_reference=a[i].mean(),MASE_model=b[i].mean(),delta_MASE=(a[i]-b[i]).mean(),reduction_percent=float((100*(a[i]-b[i])/a[i]).mean())))
     sv=s.groupby(['model','service']).MASE.mean().unstack('model')
     for service,row in sv.iterrows():services.append(dict(resource=resource,reference=reference,model=model,service=service,MASE_reference=row[reference],MASE_model=row[model],delta_MASE=row[reference]-row[model]))
 comparisons=pd.DataFrame(comparisons);comparisons.to_csv(OUT/'contrastes_dos_escalas.csv',index=False)
 pd.DataFrame(seeds).to_csv(OUT/'contrastes_por_semilla.csv',index=False);pd.DataFrame(services).to_csv(OUT/'contrastes_por_servicio.csv',index=False)
 # Verify the original intervals were absolute MASE, not percentage intervals.
 orig=pd.read_csv(BASE/'results/estadisticas/comparaciones_emparejadas.csv')
 for row in orig.itertuples():
  now=comparisons[(comparisons.reference=='E0')&(comparisons.model==row.variant)&(comparisons.resource==row.resource)&(comparisons.conditions==row.conditions)].iloc[0]
  np.testing.assert_allclose([now.delta_MASE_low,now.delta_MASE_high,now.reduction_percent],[row.IC95_low,row.IC95_high,row.reduction_percent],rtol=1e-10,atol=1e-12)
 profile=[]
 for r,resource in enumerate(e.RESOURCES):
  for n,service in enumerate(data.services):
   x=data.values[:5603,n,r];std=x.std();row=dict(resource=resource,service=service,mean=x.mean(),std=std,relative_mean_absolute_change=np.abs(np.diff(x)).mean()/max(std,1e-8))
   for lag in [1,12,48]:row[f'acf_{lag}']=float(np.corrcoef(x[:-lag],x[lag:])[0,1]) if std>0 else None
   profile.append(row)
 profile=pd.DataFrame(profile);profile.to_csv(OUT/'dinamica_entrenamiento.csv',index=False)
 profile.groupby('resource').median(numeric_only=True).reset_index().to_csv(OUT/'dinamica_entrenamiento_mediana.csv',index=False)
 dump('validacion.json',dict(status='passed',original_prediction_hashes_verified=len(files),new_predictions_recomputed=len(fresh_checks),unique_physical_prediction_files=1720,physical_metric_rows=337120,rows_including_V2_alias=360640,alias_note='V2 reuses the validation-selected compact family; not an independent fit',original_intervals_verified_as_absolute_MASE=True,bootstrap_masks_seed=91001,bootstrap_training_seeds_seed=91002,router=router,new_files=fresh_checks,original_files=files))
 print('ALL_METRICS_AND_TWO_SCALE_INTERVALS_VERIFIED',flush=True)
if __name__=='__main__':main()

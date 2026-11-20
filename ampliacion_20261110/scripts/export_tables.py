"""Deterministically format audited tabular results; never round source CSVs."""
from pathlib import Path
import json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];ANA=ROOT/'results/analysis';OUT=ROOT/'articulo_final/tablas'
ORDER=['Persistence','MovingAverage','SES','Ridge','DLinear','NLinear','E5','A0','A1','A2','A3','GRU_own','GRU_graph','GRUD','PatchTST']
LABEL={'Persistence':'Persistence','MovingAverage':'Moving average','SES':'Exponential smoothing','Ridge':'Ridge','DLinear':'DLinear','NLinear':'NLinear','E5':'E5','A0':'A0','A1':'A1','A2':'A2','A3':'A3','GRU_own':'GRU (own)','GRU_graph':'GRU (graph)','GRUD':'GRU-D adaptation','PatchTST':'PatchTST (small)'}
R=['cpu_usage','container_memory_usage_bytes']
def tex(s):return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def number(v,d=4):
 if v is None or not np.isfinite(v):return 'undefined'
 return f'{v:.{d}f}'
def table(name,headers,rows,align=None):
 if align is None:align='l'+'r'*(len(headers)-1)
 text='\\begin{tabular}{'+align+'}\n\\toprule\n'+' & '.join(headers)+r' \\'+'\n\\midrule\n'
 text+='\n'.join(' & '.join(str(v) for v in row)+r' \\' for row in rows)+'\n\\bottomrule\n\\end{tabular}\n'
 (OUT/(name+'.tex')).write_text(text)
def main():
 assert (ANA/'COMPLETED.json').exists() and (ROOT/'experiments/benchmarks/COMPLETED.json').exists()
 OUT.mkdir(parents=True,exist_ok=True);df=pd.read_csv(ANA/'metrics_all.csv.gz')
 positive=lambda x:x[x.q.gt(0)&x.pattern.isin(['disperso','bloques'])]
 p=positive(df[(df.dataset=='new_services')&(df.L==12)&(df.h_steps==1)])
 means=p.groupby(['resource','model'])[['MAE','RMSE','MASE']].mean()
 rows=[]
 for m in ORDER:
  cpu=means.loc[(R[0],m)];mem=means.loc[(R[1],m)]
  rows.append([LABEL[m],number(cpu.MASE),number(cpu.MAE*1000),number(cpu.RMSE*1000),number(mem.MASE),number(mem.MAE/2**20),number(mem.RMSE/2**20)])
 table('primary',['Model','CPU MASE','CPU MAE','CPU RMSE','Mem. MASE','Mem. MAE','Mem. RMSE'],rows)
 stress=df[(df.dataset=='new_services')&(df.L==12)&(df.h_steps==1)&df.pattern.isin(['correlacionado','prolongado','MAR_lag'])].groupby(['resource','model','pattern']).MASE.mean()
 table('additional_missingness',['Model','CPU corr.','CPU long','CPU lag-MAR','Mem. corr.','Mem. long','Mem. lag-MAR'],[[LABEL[m]]+[number(stress.loc[(r,m,pattern)]) for r in R for pattern in ['correlacionado','prolongado','MAR_lag']] for m in ORDER])
 contrasts=json.loads((ANA/'primary_contrasts.json').read_text())['contrasts'];rows=[];sensitivity=[]
 for c in contrasts:
  lo,hi=c['conditional_crossed_block_CI95']['48'];resource='CPU' if c['resource']==R[0] else 'Memory'
  rows.append([resource,c['reference'],number(c['mean_paired_MASE_difference'],5),number(c['mean_paired_reduction_percent'],2),f'[{lo:.5f}, {hi:.5f}]',f"{c['p_Holm']:.4g}",f"{c['positive_seeds']}/{c['n_nonzero_seeds']}"])
  for block,limits in c['conditional_crossed_block_CI95'].items():sensitivity.append([resource,c['reference'],block,number(limits[0],5),number(limits[1],5)])
 table('contrasts',['Resource','Reference',r'$\Delta$MASE',r'Mean $R$ (\%)',r'Conditional 95\% interval',r'$p_{\mathrm{Holm}}$','Positive seeds'],rows)
 table('block_sensitivity',['Resource','Reference','Block steps','Lower','Upper'],sensitivity)
 # Common seed and mask subset for all secondary display points.
 common=pd.read_csv(ANA/'common_origin_metrics.csv.gz')
 for r,label in zip(R,['cpu','memory']):
  for axis,values,fixed,name in [('h_minutes',[30,60,120,180,360],common.L.eq(12),'horizons'),('L',[12,24,48],common.h_steps.eq(1),'histories')]:
   g=common[(common.resource==r)&fixed].groupby(['model',axis]).MASE.mean();rows=[[LABEL[m]]+[number(g.loc[(m,v)]) for v in values] for m in ['DLinear','E5','A3','PatchTST']]
   table(name+'_'+label,['Model']+[str(v) for v in values],rows)
 external=positive(df[df.dataset=='alibaba_transfer']);means=external.groupby(['resource','model'])[['MAE','RMSE','MASE']].mean();rows=[]
 for m in ORDER:
  if m=='GRU_graph':continue
  cpu=means.loc[(R[0],m)];mem=means.loc[(R[1],m)];rows.append([LABEL[m],number(cpu.MASE),number(cpu.MAE,6),number(cpu.RMSE,6),number(mem.MASE),number(mem.MAE,6),number(mem.RMSE,6)])
 table('external',['Model','CPU MASE','CPU MAE','CPU RMSE','Mem. MASE','Mem. MAE','Mem. RMSE'],rows)
 benchmark=pd.read_csv(ROOT/'experiments/benchmarks/latency.csv');training=pd.read_csv(ROOT/'experiments/benchmarks/training.csv');training=training[(training.L==12)&(training.h==1)]
 times=training.groupby('model').training_with_validation_seconds.median();rows=[];eff=[]
 for m in ORDER:
  c=benchmark[(benchmark.model==m)&benchmark.device.eq('cpu')&benchmark.batch.eq(1)&benchmark.method.eq('eager')].iloc[0]
  method='cuda_graph' if m=='GRUD' else 'eager';g=benchmark[(benchmark.model==m)&benchmark.device.eq('cuda')&benchmark.batch.eq(1)&benchmark.method.eq(method)].iloc[0]
  rows.append([LABEL[m],str(int(c.parameters)),number(c.p50_ms,3),number(c.p95_ms,3),number(g.p50_ms,3),number(g.p95_ms,3),number(times.get(m),2) if m in times else 'not timed'])
  eff.append(dict(model=m,parameters=int(c.parameters),cpu_p50_ms=c.p50_ms,cuda_p50_ms=g.p50_ms,training_completed_epochs_median_seconds=times.get(m,np.nan)))
 table('efficiency',['Model','Parameters','CPU p50','CPU p95','CUDA p50','CUDA p95','Epoch sum (s)'],rows)
 pd.DataFrame(eff).to_csv(ANA/'efficiency_display.csv',index=False)
 historic=pd.read_csv(ROOT/'results/historical/resumen_global.csv');hist=historic.pivot(index='model',columns='resource',values='MASE');old_order=['persistencia','E0','GRU_propia','GRU_llamadas','GinAR','BiTGraph_codigo_autor','E1','E2','E3','E4','E5','C0','C1','C2','C3','V2']
 table('historical',['Historical model','CPU MASE','Memory MASE'],[[tex(m),number(hist.loc[m,R[0]]),number(hist.loc[m,R[1]])] for m in old_order])
 counts=df.groupby(['dataset','L','h_steps']).agg(metric_rows=('MASE','size'),services=('service','nunique'),model_families=('model','nunique'),mask_realizations=('realization','nunique'),valid_metric_cells=('MASE','count')).reset_index();counts.to_csv(ANA/'experiment_matrix.csv',index=False)
 table('matrix',['Dataset','L','h steps','Models','Masks','Metric rows'],[[tex(row.dataset),row.L,row.h_steps,row.model_families,row.mask_realizations,row.metric_rows] for row in counts.itertuples()])
 variability=pd.read_csv(ANA/'variability.csv');rows=[]
 for m in ORDER:
  a=variability[(variability.resource==R[0])&(variability.model==m)].iloc[0];b=variability[(variability.resource==R[1])&(variability.model==m)].iloc[0]
  rows.append([LABEL[m],number(a.seed_sd),number(a.mask_sd),number(b.seed_sd),number(b.mask_sd)])
 table('variability',['Model','CPU seed SD','CPU mask SD','Memory seed SD','Memory mask SD'],rows)
 # Exact file/field provenance is retained independently of display rounding.
 (OUT/'provenance.json').write_text(json.dumps(dict(metric_source='results/analysis/metrics_all.csv.gz',primary_filter=dict(dataset='new_services',L=12,h_steps=1,patterns=['disperso','bloques'],q_positive=True),cpu_MAE_RMSE_display_multiplier=1000,memory_MAE_RMSE_display_divisor=2**20,external_MAE_RMSE='publisher normalized utilization',historical_source='results/historical/resumen_global.csv; all seven original conditions',contrasts_source='results/analysis/primary_contrasts.json',efficiency_source='experiments/benchmarks/latency.csv and training.csv',no_source_values_rounded=True),indent=2))
 print('AUDITED_TABLES_EXPORTED',len(list(OUT.glob('*.tex'))),flush=True)
if __name__=='__main__':main()

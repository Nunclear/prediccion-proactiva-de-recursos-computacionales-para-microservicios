"""Repeated synchronized forward timings, each model in a fresh process."""
import os,sys,json,subprocess,time,random,resource,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
def worker(r,label,target):
 sys.path[:0]=[str(ROOT/'src'),str(BASE/'src'),str(BASE/'src/baseline/reference'),str(BASE/'scripts')]
 os.environ['CHRONOGRAPH_OUTPUT']=str(BASE/'results/reproduccion')
 import run as e
 import ablaciones as adapter
 import torch,numpy as np,psutil
 from compact import CompactDLinear
 from improved.models import DLinearEMAMask
 data=e.Data();own=ROOT/'results/nuevos_experimentos';chosen=json.loads((own/'selected_candidates.json').read_text())['choices']
 router=json.loads((own/'frozen_v2.json').read_text())['router'];name=router[str(r)] if label=='V2' else label
 ds=adapter.Prepared(data,r,'test',90001) if label=='E5' else e.Prepared(data,r,'test',90001)
 pos=torch.arange(128,device=e.DEVICE);inputs=ds.batch(pos,pos%7);ck=None
 if name.startswith('C'):
  model=CompactDLinear(**chosen[f'{name}_r{r}']['spec']).to(e.DEVICE);folder=own
 elif label in ['E2','E4','E5']:
  choice=json.loads((BASE/'results/mejoras/selected.json').read_text())['choices'][f'{label}_r{r}'];model=DLinearEMAMask(**choice['spec']).to(e.DEVICE);folder=BASE/'results/mejoras'
 elif label=='persistencia':model=None;folder=BASE/'results/reproduccion'
 else:
  name='DLinear' if label=='E0' else label;model=e.create(name,data.a,e.DEVICE);folder=BASE/'results/reproduccion'
 if model is not None:
  ck=torch.load(folder/'checkpoints'/f'{name}_r{r}_s101.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['best_state']);model.eval()
 def forward():return inputs[0][:,-1,:,0] if model is None else model(*inputs[:3])
 expected=np.load(folder/'test'/f'{name}_r{r}_s{0 if model is None else 101}_rep0.npz')['prediction_normalized'][np.arange(128)%7,np.arange(128)]
 rss_before=psutil.Process().memory_info().rss
 samples=[];blocks=[];temps=[]
 with torch.inference_mode(),torch.device('cuda'):
  actual=forward().cpu().numpy();np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=2e-5)
  for _ in range(10):forward()
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();allocated_before=torch.cuda.memory_allocated()
  for block in range(5):
   times=[]
   for _ in range(20):
    torch.cuda.synchronize();start=time.perf_counter();out=forward();torch.cuda.synchronize();value=1000*(time.perf_counter()-start);times.append(value);samples.append(dict(block=block,latency_ms=value))
   blocks.append(float(np.median(times)))
   temps.append(float(subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True).splitlines()[0]))
 row=dict(model=label,implementation=name,resource=e.RESOURCES[r],r=r,parameters=0 if model is None else sum(p.numel() for p in model.parameters()),
   latency_p50_ms=float(np.median([x['latency_ms'] for x in samples])),latency_p95_ms=float(np.quantile([x['latency_ms'] for x in samples],.95)),
   block_median_min_ms=min(blocks),block_median_max_ms=max(blocks),rss_before_forward_bytes=rss_before,rss_after_forward_bytes=psutil.Process().memory_info().rss,
   process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
   vram_allocated_before_bytes=allocated_before,vram_peak_allocated_bytes=torch.cuda.max_memory_allocated(),vram_forward_extra_peak_bytes=torch.cuda.max_memory_allocated()-allocated_before,
   training_validation_seconds=0 if ck is None else sum(x['seconds'] for x in ck['history']),epochs=0 if ck is None else len(ck['history']),
   mean_epoch_including_validation_seconds=0 if ck is None else float(np.mean([x['seconds'] for x in ck['history']])),
   batch=128,warmups=10,repetitions=100,blocks=5,seed=101,backend='eager float32, resident prepared histories; synchronized CUDA',
   temperature_min_C=min(temps),temperature_max_C=max(temps),forward_archive_comparison_max_abs=float(np.abs(actual-expected).max()),samples=samples)
 Path(target).write_text(json.dumps(row,indent=2,allow_nan=False))
def main():
 out=ROOT/'results/comparaciones/benchmark';out.mkdir(exist_ok=True)
 labels=['E0','E2','E4','E5','C0','C1','C2','C3','V2','GRU_propia','GRU_llamadas','BiTGraph_codigo_autor','GinAR','persistencia']
 jobs=[(r,n) for r in range(2) for n in labels];random.Random(4401).shuffle(jobs)
 for r,n in jobs:
  p=out/f'{n}_r{r}.json'
  if p.exists():continue
  subprocess.run([sys.executable,__file__,'--worker',str(r),n,str(p)],check=True)
  print('BENCHMARK',r,n,flush=True)
 import pandas as pd
 rows=[json.loads(p.read_text()) for p in out.glob('*.json')];samples=[]
 for row in rows:
  samples.extend(dict(model=row['model'],resource=row['resource'],**s) for s in row.pop('samples'))
 df=pd.DataFrame(rows);df.to_csv(ROOT/'results/comparaciones/costes_nueva_medicion.csv',index=False);pd.DataFrame(samples).to_csv(ROOT/'results/comparaciones/latencias_individuales.csv',index=False)
 means=pd.read_csv(ROOT/'results/comparaciones/resumen_perdidas.csv')
 joined=df.merge(means,on=['model','resource']);flags=[]
 for i,row in joined.iterrows():
  others=joined[(joined.resource==row.resource)&(joined.model!=row.model)]
  dominated=((others.MASE<=row.MASE)&(others.latency_p50_ms<=row.latency_p50_ms)&((others.MASE<row.MASE)|(others.latency_p50_ms<row.latency_p50_ms))).any()
  flags.append(not bool(dominated))
 joined['pareto_MASE_latency_descriptive']=flags;joined.to_csv(ROOT/'results/comparaciones/pareto.csv',index=False)
 (ROOT/'results/benchmark_COMPLETE.json').write_text(json.dumps(dict(status='complete',models=28,latency_samples=2800)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker',nargs=3);args=p.parse_args()
 if args.worker:worker(int(args.worker[0]),args.worker[1],args.worker[2])
 else:main()

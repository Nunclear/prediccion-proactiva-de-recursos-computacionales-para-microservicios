"""Resident-tensor model latency, synchronized CUDA; never called during fitting."""
import sys,time,gc,json,os,platform,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from core import *
from models import *
from engine import ReplayedGRUD
import pandas as pd,psutil
@torch.no_grad()
def main():
 benchmark_started=time.time();atomic_json(ROOT/'results/current_stage.json',dict(stage='benchmarks',pid=os.getpid(),started_unix=benchmark_started))
 folder=ROOT/'experiments/baseline/L12_H1';assert (folder/'COMPLETED.json').exists()
 if DEVICE.type=='cuda':torch.cuda.set_per_process_memory_fraction(.28)
 out=ROOT/'experiments/benchmarks';out.mkdir(exist_ok=True)
 data=Series(ROOT/'data/processed/chronograph_holdout.npz');records=json.loads((folder/'frozen.json').read_text())['models']
 rows=[];raw=[];rng=np.random.default_rng(61110);env=dict(started_unix=benchmark_started,pid=os.getpid(),platform=platform.platform(),python=platform.python_version(),torch=str(torch.__version__),numpy=str(np.__version__),threads=torch.get_num_threads(),cpu=platform.processor(),warmup=20,trials=100,order='fixed seeded random order within device and batch',scope='Resident float32 tensor -> model forward -> CUDA synchronization. Excludes telemetry collection, imputation, network, control and serialization. Other user GPU applications remain running; timings are shared-laptop observations.',graph_setup_excluded=True)
 try:env['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.used,temperature.gpu','--format=csv,noheader'],text=True)
 except Exception:pass
 atomic_json(out/'environment.json',env)
 for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
  prep=Prepared(data,0,'test',12,1,90001,device=torch.device(device))
  for batch in [1,128]:
   names=NEURAL+SIMPLE;names=[names[i] for i in rng.permutation(len(names))]
   for name in names:
    record=next(x for x in records if x['model']==name and x['r']==0 and x['seed'] in [0,101]);ck=torch.load(folder/'checkpoints'/record['file'],map_location='cpu',weights_only=False)
    model=create(name,12,data.a,device,ck.get('parameter'));load_for_target(model,ck['best_state']);model.eval()
    pos=torch.arange(batch,device=device);z,y=prep.batch(pos,torch.full_like(pos,3))
    methods=['eager']+(['cuda_graph'] if name=='GRUD' and device=='cuda' else [])
    for method in methods:
     capture=ReplayedGRUD(model,[(z,y)],False) if method=='cuda_graph' else None
     def invoke():return capture.replay([(z,y)])[0] if capture is not None else model(z)
     for _ in range(20):invoke()
     if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();baseline=torch.cuda.memory_allocated()
     rss0=psutil.Process().memory_info().rss;times=[]
     for trial in range(100):
      if device=='cuda':torch.cuda.synchronize()
      start=time.perf_counter_ns();p=invoke()
      if device=='cuda':torch.cuda.synchronize()
      ms=(time.perf_counter_ns()-start)/1e6;times.append(ms)
      raw.append(dict(model=name,device=device,method=method,batch=batch,trial=trial,latency_ms=ms))
     assert torch.isfinite(p).all()
     row=dict(model=name,device=device,method=method,batch=batch,services=32,resource='cpu_usage',history=12,parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),parameter_tensor_elements_total=sum(p.numel() for p in model.parameters()),parameter_scope='trainable; fixed tensors counted separately',p50_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),mean_ms=float(np.mean(times)),throughput_service_predictions_s=float(batch*32/(np.mean(times)/1000)),process_rss_bytes=psutil.Process().memory_info().rss,process_rss_before_bytes=rss0,cuda_allocated_baseline_bytes=int(baseline) if device=='cuda' else 0,cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()) if device=='cuda' else 0)
     rows.append(row);print(json.dumps(row),flush=True)
     if capture is not None:capture.close()
     del capture
    del model,z,y;gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
  del prep
 pd.DataFrame(rows).to_csv(out/'latency.csv',index=False);pd.DataFrame(raw).to_csv(out/'latency_trials.csv',index=False)
 # Training times from checkpoint histories, including validation and graph setup.
 training=[]
 for path in sorted((ROOT/'experiments').glob('**/checkpoints/*.pt')):
  ck=torch.load(path,map_location='cpu',weights_only=False);hist=ck['history'];sig=ck['signature']
  if hist:training.append(dict(**sig,epochs=len(hist),minimum_microbatch=min(x['microbatch'] for x in hist),maximum_microbatch=max(x['microbatch'] for x in hist),effective_batch=16,training_with_validation_seconds=sum(x['seconds'] for x in hist),epoch_p50_seconds=float(np.median([x['seconds'] for x in hist])),epoch_p95_seconds=float(np.quantile([x['seconds'] for x in hist],.95)),parameters=ck['parameters'],parameter_scope='checkpoint metadata includes all nn.Parameter tensors; use latency.csv for requires_grad-filtered counts',complete=ck['complete'],best_source_validation_MASE=ck['best'],checkpoint=str(path.relative_to(ROOT))))
 pd.DataFrame(training).to_csv(out/'training.csv',index=False)
 atomic_json(out/'COMPLETED.json',dict(status='complete',completed_unix=time.time(),latency_rows=len(rows),training_records=len(training),energy='not measured; reliable per-project energy instrumentation unavailable'))
if __name__=='__main__':main()

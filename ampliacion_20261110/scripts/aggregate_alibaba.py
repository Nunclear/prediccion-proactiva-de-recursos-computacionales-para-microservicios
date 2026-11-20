"""Causal 30-minute means from public pod-minute observations, no invented targets."""
from pathlib import Path
import sys,time,json,tarfile,io,os
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from core import atomic_json,filehash
def csv_chunks(index,cols):
 path=ROOT/'data/raw'/f'MSMetricsUpdate_{index}.tar.gz'
 with tarfile.open(path,'r|gz') as tar:
  for member in tar:
   if member.isfile() and member.name.endswith('.csv'):
    class ForwardOnly(io.RawIOBase):
     def __init__(self,source):self.source=source
     def readable(self):return True
     def readinto(self,buffer):
      data=self.source.read(len(buffer));buffer[:len(data)]=data;return len(data)
    with tar.extractfile(member) as source,io.BufferedReader(ForwardOnly(source)) as f:
     for frame in pd.read_csv(f,usecols=cols,chunksize=150000):yield frame
def choose():
 path=ROOT/'configs/alibaba_services.json'
 if path.exists():return json.loads(path.read_text())
 rows=[]
 for f in csv_chunks(0,['timestamp','msname']):rows.append(f.drop_duplicates())
 meta=pd.concat(rows,ignore_index=True).drop_duplicates();times=np.sort(meta.timestamp.unique());counts=meta.groupby('msname').timestamp.nunique()
 eligible=counts[counts==len(times)].index.tolist()
 import hashlib
 eligible.sort(key=lambda s:hashlib.sha256(('alibaba-20261110|'+s).encode()).hexdigest())
 assert len(eligible)>=32;selected=sorted(eligible[:32])
 item=dict(services=selected,selection='SHA256-ranked names with all timestamps present in object0; availability-only selection from initial calibration period',initial_object_times=times.tolist(),observed_initial_step_ms=int(np.median(np.diff(times))),initial_distinct_services=int(meta.msname.nunique()),calibration_bins=12,forecast_halfhour_bins=1,external_fit=False,
   aggregation='mean across pods at each recorded timestamp, then mean across timestamps in each completed 30-minute interval; normalize with first12 bins only; test targets bins12..47; historical prefix permitted as input context')
 atomic_json(path,item);return item
def aggregate(index,chosen):
 target=ROOT/'data/processed'/f'alibaba_part_{index}.csv'
 if target.exists():return
 rows=[];count=0;selected=set(chosen['services'])
 for f in csv_chunks(index,['timestamp','msname','cpu_utilization','memory_utilization']):
  count+=len(f);f=f[f.msname.isin(selected)].copy()
  if f.empty:continue
  for key in ['cpu_utilization','memory_utilization']:f[key]=pd.to_numeric(f[key],errors='coerce')
  f=f[np.isfinite(f.cpu_utilization)&np.isfinite(f.memory_utilization)]
  if f.empty:continue
  grouped=f.groupby(['timestamp','msname'])[['cpu_utilization','memory_utilization']].agg(['sum','count']);grouped.columns=['cpu_sum','cpu_count','memory_sum','memory_count'];rows.append(grouped.reset_index())
 if not rows:raise RuntimeError(f'No selected service data in object{index}')
 merged=pd.concat(rows,ignore_index=True).groupby(['timestamp','msname'],as_index=False)[['cpu_sum','cpu_count','memory_sum','memory_count']].sum()
 merged.to_csv(target,index=False);atomic_json(ROOT/'data/manifests'/f'alibaba_aggregation_{index}.json',dict(status='complete',raw_rows=count,selected_timestamp_service_rows=len(merged),sha256=filehash(target)))
 print('AGGREGATED',index,count,len(merged),flush=True)
def main():
 atomic_json(ROOT/'results/aggregation_pid.json',dict(pid=os.getpid()))
 while not (ROOT/'data/raw/MSMetricsUpdate_0.tar.gz').exists():time.sleep(2)
 chosen=choose();done=set();pending={};failed=[]
 with ThreadPoolExecutor(max_workers=2) as pool:
  while len(done)+len(failed)<48:
   for index in range(48):
    if index in done or index in failed or index in pending.values():continue
    if (ROOT/'data/raw'/f'MSMetricsUpdate_{index}.tar.gz').exists() and len(pending)<2:pending[pool.submit(aggregate,index,chosen)]=index
   complete=[f for f in pending if f.done()]
   for f in complete:
    index=pending.pop(f)
    try:f.result();done.add(index)
    except Exception as ex:failed.append(index);print('AGGREGATION_FAILED',index,repr(ex),flush=True)
   queue=ROOT/'data/manifests/alibaba_download_queue.json'
   if queue.exists():
    q=json.loads(queue.read_text())
    for item in q.get('failures',[]):
     if item['index'] not in failed:failed.append(item['index'])
   if not complete:time.sleep(2)
 if failed:
  atomic_json(ROOT/'data/manifests/alibaba_processed.json',dict(status='incomplete',failed=failed));return
 parts=[pd.read_csv(ROOT/'data/processed'/f'alibaba_part_{i}.csv') for i in range(48)]
 allrows=pd.concat(parts,ignore_index=True).groupby(['timestamp','msname'],as_index=False)[['cpu_sum','cpu_count','memory_sum','memory_count']].sum()
 allrows['cpu']=allrows.cpu_sum/allrows.cpu_count;allrows['memory']=allrows.memory_sum/allrows.memory_count
 allrows['bin']=allrows.timestamp.to_numpy()//1800000
 bins=allrows.groupby(['bin','msname'],as_index=False).agg(cpu=('cpu','mean'),memory=('memory','mean'),timestamps=('timestamp','nunique'))
 expected=int(round(1800000/chosen['observed_initial_step_ms']))
 bins.loc[bins.timestamps<expected,['cpu','memory']]=np.nan
 values=np.full((48,32,2),np.nan,dtype=np.float64)
 for row in bins.itertuples():
  if 0<=row.bin<48 and row.msname in chosen['services']:values[int(row.bin),chosen['services'].index(row.msname)]=[row.cpu,row.memory]
 assert np.isfinite(np.nanmean(values[:12],axis=0)).all()
 path=ROOT/'data/processed/alibaba_transfer.npz';np.savez_compressed(path,values=values,times=np.arange(1,49,dtype=np.int64)*1800,services=np.asarray(chosen['services']),adjacency=np.eye(32,dtype=np.float32))
 item=dict(status='complete',shape=list(values.shape),sha256=filehash(path),raw_objects=48,aggregation=chosen['aggregation'],observed_native_step_ms=chosen['observed_initial_step_ms'],required_timestamps_per_bin=expected,
  natural_missing_entries=int(np.isnan(values).sum()),units='publisher min-max normalized utilization; mean across observed pods then timestamps, not bytes/cores',external_graph='not downloaded; graph-dependent model excluded',normalization_scope='own first12 aggregated bins only; no external weight fitting or selection')
 atomic_json(ROOT/'data/manifests/alibaba_processed.json',item);print(json.dumps(item),flush=True)
if __name__=='__main__':main()

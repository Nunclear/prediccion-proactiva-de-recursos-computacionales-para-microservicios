"""Independent aggregation reconstruction from timestamp/pod sums, no targets filled."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 chosen=json.loads((ROOT/'configs/alibaba_services.json').read_text());services=chosen['services'];mapping={s:i for i,s in enumerate(services)}
 sums={};rawrows=0;rawbytes=0
 for i in range(48):
  download=json.loads((ROOT/'data/manifests'/f'alibaba_{i}.json').read_text());raw=ROOT/'data/raw'/f'MSMetricsUpdate_{i}.tar.gz'
  assert download['content_md5_verified'] and raw.stat().st_size==download['bytes'];rawbytes+=raw.stat().st_size
  p=ROOT/'data/processed'/f'alibaba_part_{i}.csv';m=json.loads((ROOT/'data/manifests'/f'alibaba_aggregation_{i}.json').read_text());assert sha(p)==m['sha256'];rawrows+=m['raw_rows']
  with p.open() as f:
   for row in csv.DictReader(f):
    t=int(row['timestamp']);s=mapping[row['msname']];key=(t,s);values=np.array([float(row[k]) for k in ['cpu_sum','cpu_count','memory_sum','memory_count']]);assert np.isfinite(values).all()
    sums[key]=sums.get(key,np.zeros(4))+values
 times=sorted({t for t,s in sums});minute_count=len(times);assert np.array_equal(times,np.arange(1440)*60000)
 minute=np.full((1440,32,2),np.nan)
 for (t,s),v in sums.items():
  assert t%60000==0 and v[1]>0 and v[3]>0;minute[t//60000,s]=[v[0]/v[1],v[2]/v[3]]
 blocks=minute.reshape(48,30,32,2);count=np.isfinite(blocks).sum(1);rebuilt=np.nanmean(blocks,axis=1);rebuilt[count<30]=np.nan
 with np.load(ROOT/'data/processed/alibaba_transfer.npz') as z:
  assert np.array_equal(z['services'],services);assert np.array_equal(z['times'],np.arange(1,49)*1800);actual=z['values']
  np.testing.assert_allclose(actual,rebuilt,rtol=1e-12,atol=1e-12,equal_nan=True)
 missing=[dict(bin_index=int(b),available_at_seconds=int((b+1)*1800),service=services[s],resource=['cpu_usage','container_memory_usage_bytes'][r],observed_minute_timestamps=int(count[b,s,r]),partition='calibration' if b<12 else 'evaluation') for b,s,r in np.argwhere(~np.isfinite(actual))]
 chrono={}
 for name in ['source','holdout']:
  p=ROOT/'data/processed'/f'chronograph_{name}.npz'
  with np.load(p) as z:
   T,N,R=z['values'].shape;times=z['times'];assert T==8005 and R==2 and np.isfinite(z['values']).all();assert N==(28 if name=='source' else 32)
   cut=[0,int(T*.7),int(T*.8),T];diff=np.diff(times);chrono[name]=dict(shape=[T,N,R],cuts=cut,sha256=sha(p),nominal_step_seconds=1800,gaps_after_rows=np.flatnonzero(diff!=1800).tolist(),gap_durations_seconds=diff[diff!=1800].tolist(),time_sha256=hashlib.sha256(np.ascontiguousarray(times).tobytes()).hexdigest())
 assert chrono['source']['time_sha256']==chrono['holdout']['time_sha256']
 report=dict(status='passed',raw_objects_verified_at_download=48,raw_content_md5_checks=48,raw_total_bytes=rawbytes,raw_total_csv_rows=rawrows,raw_hash_policy='SHA256 computed during original bounded download; current audit checks sizes and completed manifests without rereading18GB',aggregation_reconstructed_from_pod_timestamp_sums=True,aggregation_max_abs_difference=float(np.nanmax(np.abs(actual-rebuilt))),native_cadence_ms=60000,distinct_recorded_minutes=minute_count,half_hour_shape=list(actual.shape),natural_missing_entries=missing,chronograph=chrono)
 (ROOT/'results/data_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()

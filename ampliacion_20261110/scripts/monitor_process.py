"""Sample study-process RSS and CPU time deltas; no synthetic zero on first sample."""
from pathlib import Path
import psutil,time,json,csv
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'results/telemetry/training_process.csv';previous={}
with p.open('a',newline='',buffering=1) as f:
 w=csv.DictWriter(f,fieldnames=['unix_seconds','pid','rss_bytes','vms_bytes','process_cpu_percent','status'])
 if f.tell()==0:w.writeheader()
 while not (ROOT/'results/STOP_MONITOR').exists():
  try:
   obj=json.loads((ROOT/'results/current_stage.json').read_text());proc=psutil.Process(obj['pid'])
   if Path(proc.cwd()).resolve()==ROOT.resolve():
    info=proc.memory_info();now=time.time();c=proc.cpu_times();total=c.user+c.system;key=(proc.pid,proc.create_time());last=previous.get(key)
    percent=100*(total-last[1])/(now-last[0]) if last is not None and now>last[0] else None
    previous={key:(now,total)}
    w.writerow(dict(unix_seconds=now,pid=proc.pid,rss_bytes=info.rss,vms_bytes=info.vms,process_cpu_percent=percent,status=proc.status()))
  except (FileNotFoundError,psutil.Error,KeyError,json.JSONDecodeError):pass
  time.sleep(3)

"""Project-only guards and system telemetry, no permanent configuration."""
import os,time,csv,json,signal,subprocess
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'results/telemetry/resources.csv'
fields=['unix_seconds','cpu_percent','ram_available_bytes','swap_used_bytes','swap_rate_bytes_s','gpu_percent','vram_used_MiB','gpu_temperature_C','cpu_temperature_C','paused_project_processes']
last=time.time();s=psutil.swap_memory();paused=set()
def project_pids():
 result=[]
 for name in ['current_stage.json','aggregation_pid.json']:
  try:
   p=json.loads((ROOT/'results'/name).read_text());process=psutil.Process(p['pid'])
   if Path(process.cwd()).resolve()==ROOT.resolve():result.append(process.pid)
  except (FileNotFoundError,psutil.Error,KeyError,json.JSONDecodeError):pass
 return result
with path.open('a',newline='',buffering=1) as f:
 w=csv.DictWriter(f,fieldnames=fields)
 if f.tell()==0:w.writeheader()
 try:
  while not (ROOT/'results/STOP_MONITOR').exists():
   now=time.time();memory=psutil.virtual_memory();swap=psutil.swap_memory();rate=(swap.sin-s.sin+swap.sout-s.sout)/max(now-last,.1)
   row=dict(unix_seconds=now,cpu_percent=psutil.cpu_percent(),ram_available_bytes=memory.available,swap_used_bytes=swap.used,swap_rate_bytes_s=rate)
   try:
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,temperature.gpu','--format=csv,noheader,nounits'],text=True,timeout=5).splitlines()[0].split(',')
    for key,value in zip(['gpu_percent','vram_used_MiB','gpu_temperature_C'],gpu):row[key]=float(value)
   except Exception:pass
   try:
    temps=[v.current for name,items in psutil.sensors_temperatures().items() if name in ['coretemp','k10temp'] for v in items if 0<v.current<125]
    row['cpu_temperature_C']=max(temps) if temps else None
   except Exception:row['cpu_temperature_C']=None
   hot=(row.get('gpu_temperature_C') or 0)>=85 or (row.get('cpu_temperature_C') or 0)>=92
   pressure=memory.available<3*1024**3 or rate>50*1024**2
   if hot or pressure:
    for pid in project_pids():
     if pid not in paused:
      try:os.kill(pid,signal.SIGSTOP);paused.add(pid)
      except ProcessLookupError:pass
   elif (row.get('gpu_temperature_C') or 0)<75 and (row.get('cpu_temperature_C') or 0)<80 and memory.available>4*1024**3:
    for pid in list(paused):
     try:os.kill(pid,signal.SIGCONT)
     except ProcessLookupError:pass
     paused.remove(pid)
   row['paused_project_processes']=len(paused);w.writerow(row);last=now;s=swap;time.sleep(3)
 finally:
  for pid in paused:
   try:os.kill(pid,signal.SIGCONT)
   except ProcessLookupError:pass

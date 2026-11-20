"""Bounded 24-hour external trace acquisition; two workers, finite retries."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,time,traceback
from download_alibaba import fetch,ROOT
def job(i):
 for attempt in range(3):
  try:return fetch(i)
  except Exception as ex:
   print(json.dumps(dict(index=i,attempt=attempt+1,error=str(ex))),flush=True)
   if attempt==2:raise
   time.sleep(3*(attempt+1))
if __name__=='__main__':
 results=[];failures=[]
 with ThreadPoolExecutor(max_workers=2) as pool:
  jobs={pool.submit(job,i):i for i in range(48)}
  for future in as_completed(jobs):
   try:results.append(future.result())
   except Exception as ex:failures.append(dict(index=jobs[future],error=str(ex)))
 result=dict(status='complete' if not failures else 'incomplete',completed_objects=len(results),expected_objects=48,total_bytes=sum(x['bytes'] for x in results),failures=failures)
 (ROOT/'data/manifests/alibaba_download_queue.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

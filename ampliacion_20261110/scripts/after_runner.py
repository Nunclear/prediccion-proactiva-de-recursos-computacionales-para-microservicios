"""Bounded sequential postprocessing, never overlaps its GPU benchmark with fitting."""
from pathlib import Path
import time,json,subprocess,sys,os
ROOT=Path(__file__).resolve().parents[1]
def write(name,value):(ROOT/'results'/name).write_text(json.dumps(value,indent=2))
def main():
 deadline=time.monotonic()+6*3600
 while time.monotonic()<deadline:
  p=ROOT/'results/experiments_exit.txt'
  if p.exists():
   code=int(p.read_text().strip())
   if code:write('postprocessing_failed.json',dict(stage='runner',exit_code=code));return 1
   break
  time.sleep(10)
 else:write('postprocessing_failed.json',dict(stage='wait',reason='six-hour bounded wait expired; checkpoints retained'));return 1
 # Exit file is written after process termination; require every planned marker.
 folders=[ROOT/'experiments/baseline/L12_H1']+[ROOT/'experiments/ablations'/f'L{L}_H{h}' for L,h in [(12,2),(12,4),(12,6),(12,12),(24,1),(48,1)]]
 for folder in folders:
  p=folder/'COMPLETED.json'
  if not p.exists() or json.loads(p.read_text())['status']!='complete':write('postprocessing_failed.json',dict(stage='completeness',pending=str(folder)));return 1
 for script,log in [('analyze_results.py','analysis.log'),('benchmark.py','benchmarks.log'),('summarize_resources.py','resource_summary.log')]:
  with (ROOT/'results'/log).open('w') as f:
   code=subprocess.run([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
  if code:write('postprocessing_failed.json',dict(stage=script,exit_code=code,log=log));return code
 write('postprocessing_completed.json',dict(status='analysis_and_benchmarks_complete',unix=time.time()));return 0
if __name__=='__main__':sys.exit(main())

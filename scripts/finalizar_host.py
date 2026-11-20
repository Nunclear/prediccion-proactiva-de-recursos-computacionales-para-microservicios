import os,time,signal,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
while not (ROOT/'results/COMPLETED.json').exists():
 if (ROOT/'results/report_exit.txt').exists() and (ROOT/'results/report_exit.txt').read_text().strip()!='0':raise SystemExit('Report failed; see report.log')
 time.sleep(10)
try:
 subprocess.run(['python3',str(ROOT/'scripts/exportar.py')],check=True)
finally:
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:
   cmd=p.read_bytes()
   if cmd.split(b'\0')[0].endswith(b'systemd-inhibit') and b'DLinear research' in cmd and int(p.parent.name)!=os.getpid():os.kill(int(p.parent.name),signal.SIGTERM)
  except (FileNotFoundError,PermissionError,ProcessLookupError):pass

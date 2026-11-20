"""Refuse a second experimental runner in the same project/container."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
found=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  args=p.read_bytes().split(b'\0');cwd=Path(os.readlink(p.parent/'cwd'))
  if len(args)>1 and args[1] in [b'scripts/run_experiments.py',str(ROOT/'scripts/run_experiments.py').encode()] and cwd.resolve()==ROOT.resolve():found.append(int(p.parent.name))
 except OSError:pass
if found:print('Runner already active:',found);sys.exit(1)
print('No active runner in this project')

"""Compare an interrupted epoch-boundary restart with uninterrupted training."""
import sys, shutil, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import ablaciones as a
import torch
e=a.engine
e.OUT=ROOT/'results/resume_check'
e.OUT.mkdir(exist_ok=True)
shutil.copy2(ROOT/'results/reproduccion/data_verified.npz',e.OUT/'data_verified.npz')
data=e.Data();e.create=a.create
a.CURRENT['E3']={'variant':'E3'}
tr=a.Prepared(data,0,'train',70001,e.control.sample_indices(data.indices['train'],32))
va=a.Prepared(data,0,'validation',80001,e.control.sample_indices(data.indices['validation'],14))
full=e.OUT/'uninterrupted.pt';part=e.OUT/'interrupted.pt'
for p in [full,part]:
 if p.exists():p.unlink()
_,expected=e.train('E3',data,0,tr,va,101,.003,3,None,full)
save=e.save_checkpoint
class SimulatedPowerLoss(Exception):pass
def interrupt(path,checkpoint):
 save(path,checkpoint)
 if checkpoint['epoch']==0:raise SimulatedPowerLoss()
e.save_checkpoint=interrupt
try:e.train('E3',data,0,tr,va,101,.003,3,None,part)
except SimulatedPowerLoss:pass
finally:e.save_checkpoint=save
_,actual=e.train('E3',data,0,tr,va,101,.003,3,None,part)
assert expected['state_sha256']==actual['state_sha256']
for name in ['state','best_state']:
 for k,v in expected[name].items():assert torch.equal(v,actual[name][k]),(name,k)
assert torch.equal(expected['generator'],actual['generator'])
for x,y in zip(expected['history'],actual['history']):
 for k in ['epoch','loss','validation_MASE','best_MASE','best_epoch']:assert x[k]==y[k],k
(e.OUT/'validation.json').write_text(json.dumps(dict(status='passed',exact_final_and_best_weights=True,exact_epoch_metrics=True,epochs=3,interruption_after_epoch=0,state_sha256=actual['state_sha256'])))
print('EXACT_CHECKPOINT_RESUME_PASSED',flush=True)

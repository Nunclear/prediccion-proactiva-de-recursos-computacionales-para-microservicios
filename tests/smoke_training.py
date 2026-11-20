import sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts')]
import ablaciones as a
import torch,numpy as np
engine=a.engine;engine.OUT=ROOT/'results/smoke';engine.OUT.mkdir(exist_ok=True)
shutil.copy2(ROOT/'results/reproduccion/data_verified.npz',engine.OUT/'data_verified.npz')
data=engine.Data();engine.create=a.create
tr=a.Prepared(data,0,'train',70001,engine.control.sample_indices(data.indices['train'],32));va=a.Prepared(data,0,'validation',80001,engine.control.sample_indices(data.indices['validation'],14))
lo,hi=data.bounds['train'];assert torch.all(tr.z[:,lo:hi,:,1]+tr.z[:,lo:hi,:,3]+tr.z[:,lo:hi,:,4]==1)
for name in ['E1','E2','E3','E4','E5']:
 a.CURRENT[name]=a.candidates(name)[-1]
 model,ck=engine.train(name,data,0,tr,va,101,.001,1,None,engine.OUT/f'{name}.pt')
 result=engine.evaluate(name,model,va);assert np.isfinite(result['MASE']).all()
 assert ck['complete']
print('ALL_FIVE_TRAINING_VALIDATION_SMOKE_TESTS_PASSED')

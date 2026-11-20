"""Check two real saved forecasts without displaying model error levels."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from analyze_results import load_data,expected_indices,digest,sha
data=load_data('new_services');folder=ROOT/'experiments/baseline/L12_H1/test/new_services';records=[]
cases=[(0,'disperso')]+[(q,p) for q in [10,20,40] for p in ['disperso','bloques']]+[(20,'correlacionado'),(20,'prolongado'),(20,'MAR_lag')]
for r in [0,1]:
 p=folder/f'A3_r{r}_s101_rep0.npz'
 with np.load(p) as z:prediction=z['prediction_normalized'];ix=z['indices'];assert digest(prediction)==str(z['prediction_sha256']);assert digest(data['values'][ix,:,r])==str(z['target_sha256'])
 assert np.array_equal(ix,expected_indices(data,'new_services',12,1));truth=data['values'][ix,:,r]
 error=prediction.astype(np.float64)*data['scale'][:,r]+data['mean'][:,r]-truth[None]
 reconstructed={'MAE':np.mean(np.abs(error),axis=1),'RMSE':np.sqrt(np.mean(error*error,axis=1))};reconstructed['MASE']=reconstructed['MAE']/data['denominator'][:,r]
 frame=pd.read_csv(folder/f'metrics_r{r}_rep0.csv');frame=frame[(frame.model=='A3')&frame.seed.eq(101)].set_index(['q','pattern','service'])
 for c,(q,pattern) in enumerate(cases):
  actual=frame.loc[[(q,pattern,s) for s in data['services']]]
  for name,array in reconstructed.items():np.testing.assert_allclose(actual[name],array[c],rtol=1e-10,atol=1e-10)
 records.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),array_shape=list(prediction.shape)))
(ROOT/'results/saved_prediction_audit_smoke.json').write_text(json.dumps(dict(status='passed',files=records,metric_values_displayed=False,scope='Structural/numerical smoke check; full planned audit follows completed experiments.'),indent=2))
print('SAVED_PREDICTION_AUDIT_SMOKE_PASSED; no model error levels displayed')

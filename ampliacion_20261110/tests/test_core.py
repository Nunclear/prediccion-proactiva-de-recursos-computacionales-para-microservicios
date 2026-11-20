import sys,unittest,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,torch
from core import *
from models import *
class Tests(unittest.TestCase):
 def test_models_and_affine_identity(self):
  for L in [12,24,48]:
   z=torch.randn(2,L,4,5);z[...,1]=torch.randint(0,2,(2,L,4));z[...,2]=torch.rand(2,L,4);z[...,3]=(1-z[...,1]);z[...,4]=0
   a=np.eye(4,dtype=np.float32)
   seed_all(101);baseline=create('DLinear',L,a,torch.device('cpu'))
   for name in ['A0','A1','A2','A3']:
    seed_all(101);model=create(name,L,a,torch.device('cpu'))
    torch.testing.assert_close(baseline(z),model(z),rtol=1e-5,atol=1e-6)
    self.assertEqual(sum(p.numel() for p in model.parameters()),L+1+int(name in ['A1','A3'])+int(name in ['A2','A3']))
   for name in NEURAL:
    seed_all(101);model=create(name,L,a,torch.device('cpu'));model.eval();p=model(z);self.assertEqual(tuple(p.shape),(2,4));p.square().mean().backward()
    self.assertTrue(all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters()),name)
    if DEVICE.type=='cuda':
     q=model.to(DEVICE)(z.to(DEVICE)).cpu();torch.testing.assert_close(p,q,rtol=1e-4,atol=1e-5)
 def test_temporal_boundaries_masks_and_invariance(self):
  data=Series(ROOT/'data/processed/chronograph_source.npz')
  for L in [12,24,48]:
   for h in [1,2,4,6,12]:
    for part in ['train','validation','test']:
     lo,hi=data.bounds[part];ix=data.indices(part,L,h)
     self.assertGreaterEqual(int(ix.min())-h-L+1,lo);self.assertLess(int(ix.max()),hi)
     for t in ix[::max(1,len(ix)//20)]:self.assertTrue(np.all(np.diff(data.times[t-h-L+1:t+1])==1800))
  original=mask_for(data,'test',20,'MAR_lag',90001);changed=copy.copy(data);changed.values=data.values.copy();changed.values[7000:,:,0]+=1e6
  second=mask_for(changed,'test',20,'MAR_lag',90001);self.assertTrue(np.array_equal(original[:7000],second[:7000]))
  correlated=mask_for(data,'test',20,'correlacionado',90001);self.assertTrue(np.array_equal(correlated[:,0],correlated[:,1]))
  ds=Prepared(data,0,'validation',12,1,80001,subset=data.indices('validation',12,1)[:4],device=torch.device('cpu'))
  lo,hi=data.bounds['validation'];self.assertTrue(torch.all(ds.z[:,lo:hi,:,1]+ds.z[:,lo:hi,:,3]+ds.z[:,lo:hi,:,4]==1))
  modified=copy.copy(data);modified.values=data.values.copy();m=mask_for(data,'validation',20,'disperso',80001);modified.values[~m,0]+=1e5
  # Directly test that a fixed unavailable value cannot enter prepared histories.
  ds2=Prepared(modified,0,'validation',12,1,80001,subset=ds.indices,device=torch.device('cpu'))
  torch.testing.assert_close(ds.z[3],ds2.z[3],rtol=0,atol=0)
if __name__=='__main__':unittest.main()

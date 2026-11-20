import os,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'src/baseline/reference')]
import torch,numpy as np
from improved.models import DLinearEMAMask
from models import create
class ModelTests(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(77);self.z=torch.randn(4,12,28,5);self.z[...,1]=1;self.z[...,2:]=0
  self.x=self.z[...,:1];self.mask=torch.ones_like(self.x)
 def test_baseline_identity(self):
  torch.manual_seed(101);baseline=create('DLinear',np.eye(28,dtype=np.float32),torch.device('cpu'))
  torch.manual_seed(101);new=DLinearEMAMask('E1',kernel=25)
  torch.testing.assert_close(baseline(self.z,self.x,self.mask),new(self.z,self.x,self.mask),rtol=0,atol=0)
 def test_gradients_shapes_restore(self):
  for variant in ['E1','E2','E3','E4','E5']:
   model=DLinearEMAMask(variant,learned_alpha=variant=='E5');p=model(self.z,self.x,self.mask)
   self.assertEqual(tuple(p.shape),(4,28));p.square().mean().backward()
   self.assertTrue(all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters()))
   other=DLinearEMAMask(variant,learned_alpha=variant=='E5');other.load_state_dict(model.state_dict());torch.testing.assert_close(p,other(self.z,self.x,self.mask),rtol=0,atol=0)
 def test_ema_causality_and_missing(self):
  model=DLinearEMAMask('E5');trend,_=model.components(self.z)
  altered=self.z.clone();altered[:,7:,:,0]+=1000
  after,_=model.components(altered);torch.testing.assert_close(trend[:,:7],after[:,:7],rtol=0,atol=0)
  self.z[:,4:7,:,1]=0;trend,_=model.components(self.z)
  torch.testing.assert_close(trend[:,4:7],trend[:,3:4].expand(-1,3,-1),rtol=0,atol=0)
 def test_cpu_gpu(self):
  if not torch.cuda.is_available():self.skipTest('CUDA unavailable')
  model=DLinearEMAMask('E5');cpu=model(self.z,self.x,self.mask).detach()
  gpu=model.cuda()(self.z.cuda(),self.x.cuda(),self.mask.cuda()).cpu().detach();torch.testing.assert_close(cpu,gpu,rtol=1e-5,atol=1e-5)
if __name__=='__main__':unittest.main()

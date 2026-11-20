import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
sys.path[:0]=[str(ROOT/'src'),str(BASE/'src'),str(BASE/'src/baseline/reference')]
import torch,numpy as np
from compact import CompactDLinear
from models import create
class Tests(unittest.TestCase):
 def test_baseline_and_counts(self):
  z=torch.randn(3,12,28,5);z[...,1]=1;z[...,2]=0;x=z[...,:1];mask=torch.ones_like(x)
  torch.manual_seed(101);base=create('DLinear',np.eye(28,dtype=np.float32),torch.device('cpu'))
  torch.manual_seed(101);new=CompactDLinear()
  torch.testing.assert_close(base(z,x,mask),new(z,x,mask),rtol=0,atol=0)
  for a,b in [(False,False),(True,False),(False,True),(True,True)]:
   m=CompactDLinear(availability=a,age=b);assert sum(p.numel() for p in m.parameters())==26+int(a)+int(b)
 def test_gradients_restore_and_gpu(self):
  z=torch.randn(4,12,28,5);z[...,1]=torch.randint(0,2,(4,12,28));x=z[...,:1];mask=z[...,1:2]
  for backend in ['ma25','ema03','ema07']:
   model=CompactDLinear(backend,True,True);y=model(z,x,mask);y.square().mean().backward()
   assert y.shape==(4,28) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
   other=CompactDLinear(backend,True,True);other.load_state_dict(model.state_dict());torch.testing.assert_close(y,other(z,x,mask),rtol=0,atol=0)
   if torch.cuda.is_available():torch.testing.assert_close(y,model.cuda()(z.cuda(),x.cuda(),mask.cuda()).cpu(),rtol=1e-5,atol=1e-5)
 def test_missingness_and_ema_prefix(self):
  z=torch.randn(2,12,28,5);z[...,1]=1;z[:,4:7,:,1]=0
  model=CompactDLinear('ema07',True,True);trend,_=model.base.components(z)
  torch.testing.assert_close(trend[:,4:7],trend[:,3:4].expand(-1,3,-1),rtol=0,atol=0)
  changed=z.clone();changed[:,8:,:,0]+=1000;new,_=model.base.components(changed)
  torch.testing.assert_close(trend[:,:8],new[:,:8],rtol=0,atol=0)
if __name__=='__main__':unittest.main()

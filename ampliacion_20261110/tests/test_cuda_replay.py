import sys,copy,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from core import *
from models import create
from engine import ReplayedGRUD,stable_clip
assert DEVICE.type=='cuda'
seed_all(19);a=create('GRUD',12,np.eye(4,dtype=np.float32),DEVICE);b=copy.deepcopy(a)
opt_a=torch.optim.Adam(a.parameters(),lr=.003);opt_b=torch.optim.Adam(b.parameters(),lr=.003)
inputs=[]
for i in range(2):
 z=torch.randn(8,12,4,5,device=DEVICE);z[...,1]=torch.randint(0,2,(8,12,4),device=DEVICE);z[...,2]=torch.rand(8,12,4,device=DEVICE);y=torch.randn(8,4,device=DEVICE);inputs.append((z,y))
captured=ReplayedGRUD(b,inputs,True)
for update in range(4):
 opt_a.zero_grad(set_to_none=True)
 for z,y in inputs:((a(z)-y).square().mean()*.5).backward()
 captured.replay(inputs)
 for (name,p),(name2,q) in zip(a.named_parameters(),b.named_parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
 stable_clip(a);stable_clip(b);opt_a.step();opt_b.step()
 for (name,p),(name2,q) in zip(a.named_parameters(),b.named_parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
a.eval();b.eval();z,y=inputs[0]
with torch.no_grad():
 evaluation=ReplayedGRUD(b,[(z,y)],False);out=evaluation.replay([(z,y)])[0];torch.testing.assert_close(a(z),out,rtol=0,atol=0)
atomic_json(ROOT/'results/cuda_replay_equivalence.json',dict(status='passed',gradient_and_weight_bitwise_equal=True,optimizer_updates=4,evaluation_bitwise_equal=True,training_microbatch=8))
print('CUDA_REPLAY_EXACT_EQUIVALENCE_PASSED')

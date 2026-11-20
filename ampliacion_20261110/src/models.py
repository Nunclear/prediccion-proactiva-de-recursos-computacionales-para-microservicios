import sys,math
from pathlib import Path
import torch,numpy as np
from torch import nn
from torch.nn import functional as F
class Affine(nn.Module):
 def __init__(self,L,availability=False,age=False,collapsed_init=False):
  super().__init__()
  if collapsed_init:
   residual=nn.Linear(L,1);trend=nn.Linear(L,1);self.linear=nn.Linear(L,1)
   with torch.no_grad():
    basis=torch.eye(L).unsqueeze(1);level=F.avg_pool1d(F.pad(basis,(12,12),mode='replicate'),25,stride=1)
    weight=F.linear(level,trend.weight,None)+F.linear(basis-level,residual.weight,None)
    self.linear.weight.copy_(weight.reshape(1,L));self.linear.bias.copy_(trend.bias+residual.bias)
  else:self.linear=nn.Linear(L,1)
  if availability:self.availability=nn.Parameter(torch.zeros(()))
  if age:self.age=nn.Parameter(torch.zeros(()))
 def forward(self,z):
  y=self.linear(z[...,0].transpose(1,2)).squeeze(-1)
  if hasattr(self,'availability'):y=y+self.availability*(1-z[...,1]).mean(1)
  if hasattr(self,'age'):y=y+self.age*z[:,-1,:,2]
  return y
class Decomposition(nn.Module):
 def __init__(self,L,complete=False):
  super().__init__();self.residual=nn.Linear(L,1);self.trend=nn.Linear(L,1);self.complete=complete
  if complete:
   self.mask_head=nn.Linear(3*L,1,bias=False);self.age_head=nn.Linear(L,1,bias=False)
   nn.init.zeros_(self.mask_head.weight);nn.init.zeros_(self.age_head.weight)
 def forward(self,z):
  v=z[...,0]
  if self.complete:
   state=v[:,0];seq=[state]
   for t in range(1,z.shape[1]):state=state+.7*z[:,t,:,1]*(v[:,t]-state);seq.append(state)
   trend=torch.stack(seq,1)
  else:
   b,L,n=v.shape;u=v.transpose(1,2).reshape(b*n,1,L)
   trend=F.avg_pool1d(F.pad(u,(12,12),mode='replicate'),25,stride=1).reshape(b,n,L).transpose(1,2)
  y=self.trend(trend.transpose(1,2)).squeeze(-1)+self.residual((v-trend).transpose(1,2)).squeeze(-1)
  if self.complete:
   u=torch.stack([z[...,1],z[...,3],z[...,4]],-1).permute(0,2,1,3).flatten(2)
   age=torch.clamp(torch.expm1(z[...,2]*math.log1p(z.shape[1]))/z.shape[1],max=1)
   y=y+self.mask_head(u).squeeze(-1)+self.age_head(age.transpose(1,2)).squeeze(-1)
  return y
class NLinear(nn.Module):
 def __init__(self,L):super().__init__();self.linear=nn.Linear(L,1)
 def forward(self,z):
  v=z[...,0].transpose(1,2);last=v[:,:,-1:].detach();return self.linear(v-last).squeeze(-1)+last.squeeze(-1)
class GRU(nn.Module):
 def __init__(self,a,graph=False):
  super().__init__();n=len(a)
  def norm(x):
   d=x.sum(1,keepdims=True);return np.divide(x,d,out=np.zeros_like(x),where=d!=0)
  self.register_buffer('outgoing',torch.tensor(norm(a) if graph else np.eye(n,dtype=np.float32)))
  self.register_buffer('incoming',torch.tensor(norm(a.T) if graph else np.eye(n,dtype=np.float32)))
  self.gru=nn.GRU(9,12,batch_first=True);self.head=nn.Linear(12,1)
 def forward(self,z):
  z=z[...,:3];a=torch.einsum('ij,bhjd->bhid',self.outgoing,z);b=torch.einsum('ij,bhjd->bhid',self.incoming,z)
  u=torch.cat([z,a,b],-1);B,L,N,_=u.shape;_,h=self.gru(u.permute(0,2,1,3).reshape(B*N,L,9));return self.head(h[-1]).reshape(B,N)
class GRUD(nn.Module):
 """Shared-service GRU-D adaptation: zero training-mean decay and mask input."""
 def __init__(self,L):
  super().__init__();self.L=L;self.decay_x=nn.Linear(1,1);self.decay_h=nn.Linear(1,12);self.cell=nn.GRUCell(2,12);self.head=nn.Linear(12,1)
 def forward(self,z):
  B,L,N,_=z.shape;h=torch.zeros(B*N,12,device=z.device,dtype=z.dtype)
  for t in range(L):
   v=z[:,t,:,0].reshape(-1,1);m=z[:,t,:,1].reshape(-1,1)
   age=torch.expm1(z[:,t,:,2].reshape(-1,1)*math.log1p(self.L))
   gx=torch.exp(-F.relu(self.decay_x(age)));gh=torch.exp(-F.relu(self.decay_h(age)))
   value=m*v+(1-m)*gx*v
   h=self.cell(torch.cat([value,m],-1),h*gh)
  return self.head(h).reshape(B,N)
class Patch(nn.Module):
 def __init__(self,L):
  super().__init__();folder=Path(__file__).parent/'vendor/patchtst';sys.path.insert(0,str(folder))
  from layers.PatchTST_backbone import PatchTST_backbone
  self.model=PatchTST_backbone(c_in=1,context_window=L,target_window=1,patch_len=4,stride=2,
    max_seq_len=128,n_layers=1,d_model=16,n_heads=2,d_ff=32,dropout=.1,attn_dropout=0.,
    head_dropout=0.,fc_dropout=0.,padding_patch='end',revin=True,affine=True,subtract_last=False)
 def forward(self,z):
  v=z[...,0];B,L,N=v.shape;return self.model(v.transpose(1,2).reshape(B*N,1,L)).reshape(B,N)
class Simple(nn.Module):
 def __init__(self,name,parameter=None):super().__init__();self.name=name;self.parameter=parameter
 def forward(self,z):
  v=z[...,0]
  if self.name=='Persistence':return v[:,-1]
  if self.name=='MovingAverage':return v[:,-min(int(self.parameter),v.shape[1]):].mean(1)
  state=v[:,0]
  for t in range(1,v.shape[1]):state=state+self.parameter*z[:,t,:,1]*(v[:,t]-state)
  return state
NEURAL=['DLinear','NLinear','E5','A0','A1','A2','A3','GRU_own','GRU_graph','GRUD','PatchTST']
SIMPLE=['Persistence','MovingAverage','SES','Ridge']
def create(name,L,a,device,parameter=None):
 if name=='DLinear':m=Decomposition(L)
 elif name=='E5':m=Decomposition(L,True)
 elif name=='NLinear':m=NLinear(L)
 elif name.startswith('A'):m=Affine(L,name in ['A1','A3'],name in ['A2','A3'],collapsed_init=True)
 elif name=='GRU_own':m=GRU(a)
 elif name=='GRU_graph':m=GRU(a,True)
 elif name=='GRUD':m=GRUD(L)
 elif name=='PatchTST':m=Patch(L)
 elif name=='Ridge':m=Affine(L)
 else:m=Simple(name,parameter)
 return m.to(device)
def load_for_target(model,state):
 if isinstance(model,GRU):
  missing=model.load_state_dict({k:v for k,v in state.items() if k not in ['outgoing','incoming']},strict=False)
  assert set(missing.missing_keys)=={'outgoing','incoming'} and not missing.unexpected_keys
 else:model.load_state_dict(state)

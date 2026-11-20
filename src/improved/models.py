"""Light DLinear ablations; all transforms are causal within the input window."""
import math
import torch
from torch import nn
from torch.nn import functional as F
class DLinearEMAMask(nn.Module):
 def __init__(self,variant='E5',kernel=25,alpha=.3,learned_alpha=False,age='log'):
  super().__init__();self.variant=variant;self.kernel=kernel;self.age=age;self.learned_alpha=learned_alpha
  self.residual_head=nn.Linear(12,1);self.trend_head=nn.Linear(12,1)
  if learned_alpha:self.alpha_logit=nn.Parameter(torch.tensor(math.log(alpha/(1-alpha))))
  else:self.register_buffer('alpha_value',torch.tensor(alpha))
  if variant in ['E3','E5']:self.mask_head=nn.Linear(36,1,bias=False);nn.init.zeros_(self.mask_head.weight)
  if variant in ['E4','E5']:self.age_head=nn.Linear(12,1,bias=False);nn.init.zeros_(self.age_head.weight)
 def components(self,z):
  v=z[...,0];m=z[...,1]
  if self.variant in ['E2','E5']:
   alpha=torch.sigmoid(self.alpha_logit) if self.learned_alpha else self.alpha_value
   state=v[:,0];seq=[state]
   for t in range(1,12):
    # An imputation is never counted as a new measurement for the trend.
    state=state+alpha*m[:,t]*(v[:,t]-state);seq.append(state)
   trend=torch.stack(seq,1)
  else:
   b,h,n=v.shape;u=v.permute(0,2,1).reshape(b*n,1,h)
   trend=F.avg_pool1d(F.pad(u,((self.kernel-1)//2,)*2,mode='replicate'),self.kernel,stride=1).reshape(b,n,h).permute(0,2,1)
  return trend,v-trend
 def forward(self,z,x,mask):
  trend,residual=self.components(z)
  result=self.trend_head(trend.transpose(1,2)).squeeze(-1)+self.residual_head(residual.transpose(1,2)).squeeze(-1)
  if hasattr(self,'mask_head'):
   statuses=torch.stack([z[...,1],z[...,3],z[...,4]],-1).permute(0,2,1,3).flatten(2)
   result=result+self.mask_head(statuses).squeeze(-1)
  if hasattr(self,'age_head'):
   age=z[...,2]
   if self.age=='linear_capped':age=torch.clamp(torch.expm1(age*math.log(13))/12,max=1)
   result=result+self.age_head(age.transpose(1,2)).squeeze(-1)
  return result

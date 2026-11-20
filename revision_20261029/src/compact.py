"""Compact, causal missingness features; no test-dependent routing."""
import torch
from torch import nn
from improved.models import DLinearEMAMask

class CompactDLinear(nn.Module):
 def __init__(self,backend='ma25',availability=False,age=False):
  super().__init__()
  self.base=DLinearEMAMask('E1' if backend=='ma25' else 'E2',kernel=25,
      alpha=.3 if backend=='ema03' else .7)
  if availability:self.availability_weight=nn.Parameter(torch.zeros(()))
  if age:self.age_weight=nn.Parameter(torch.zeros(()))
 def forward(self,z,x,mask):
  result=self.base(z,x,mask)
  if hasattr(self,'availability_weight'):
   result=result+self.availability_weight*(1-z[...,1]).mean(dim=1)
  if hasattr(self,'age_weight'):
   result=result+self.age_weight*z[:,-1,:,2]
  return result

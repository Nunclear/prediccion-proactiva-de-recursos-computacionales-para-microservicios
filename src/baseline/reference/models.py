"""Fixed architectures from the supplied study; device-portable author sources."""
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
CODE = HERE/'source/Seleccion_modelos_20261024/codigo_autores'
sys.path.insert(0, str(HERE/'source/Conciliacion_tecnica_AB_20261024'))
import probar_control_B as control

def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod
    spec.loader.exec_module(mod);return mod

def package(name,path):
    mod=types.ModuleType(name);mod.__path__=[str(path)];sys.modules[name]=mod

class DLinear(nn.Module):
    def __init__(self):
        super().__init__()
        mod=load('study_dlinear',CODE/'cure-lab__LTSF-Linear/models/DLinear.py')
        self.model=mod.Model(SimpleNamespace(seq_len=12,pred_len=1,enc_in=1,individual=False))
    def forward(self,z,x,mask):
        b,h,n,_=z.shape
        return self.model(z[:,:,:,0].permute(0,2,1).reshape(b*n,h,1)).reshape(b,n)

class GRU(control.ControlGRU):
    def forward(self,z,x,mask):return super().forward(z)

class GinAR(nn.Module):
    def __init__(self,a):
        super().__init__();folder=CODE/'GestaltCogTeam__GinAR/model1'
        package('study_ginar',folder)
        mod=load('study_ginar.ginar_arch',folder/'ginar_arch.py')
        self.model=mod.GinAR(12,len(a),1,1,16,8,2,0.,[torch.tensor(a),torch.tensor(a.T.copy())])
    def forward(self,z,x,mask):return self.model(x)[:,0,:]

class BiTGraph(nn.Module):
    def __init__(self,n,device):
        super().__init__();folder=HERE/'portable/BiaTCGNet'
        package('models',folder.parent);package('models.BiaTCGNet',folder)
        mod=load('models.BiaTCGNet.BiaTCGNet',folder/'BiaTCGNet.py')
        self.model=mod.Model(True,True,2,n,[3,5,7],device,subgraph_size=5,node_dim=8,
            conv_channels=12,residual_channels=12,skip_channels=16,end_channels=16,
            seq_length=12,in_dim=1,out_len=1,out_dim=1,layers=1,dropout=0.)
    def forward(self,z,x,mask):return self.model(x,mask,5)[:,0,:,0]

def create(name,a,device):
    if name=='DLinear':m=DLinear()
    elif name=='GRU_propia':m=GRU(np.eye(len(a),dtype=np.float32),np.eye(len(a),dtype=np.float32))
    elif name=='GRU_llamadas':m=GRU(control.normalize(a),control.normalize(a.T))
    elif name=='GinAR':m=GinAR(a)
    elif name=='BiTGraph_codigo_autor':m=BiTGraph(len(a),device)
    else:raise ValueError(name)
    m=m.to(device)
    if name=='GinAR':
        # Fixed supports are ordinary attributes in the author code, not buffers.
        m.model.graph_data=[a.to(device) for a in m.model.graph_data]
    return m

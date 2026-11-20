"""Reproducible 70/10/20 experiment. Test runs only after all models are frozen."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['OMP_NUM_THREADS'] = '2'
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import time
from datetime import datetime, timezone
import numpy as np
import torch
from models import create, control

HERE=Path(__file__).resolve().parent
OUT=Path(os.environ.get('CHRONOGRAPH_OUTPUT',str(HERE.parents[2]/'results/reproduccion')))
CASES=[(0,'disperso')]+[(q,p) for q in [10,20,40] for p in ['disperso','bloques']]
MODELS=['persistencia','DLinear','GRU_propia','GRU_llamadas','BiTGraph_codigo_autor','GinAR']
RESOURCES=['cpu_usage','container_memory_usage_bytes']
SEEDS=[101,202,303]
DEVICE=torch.device('cuda')
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False

def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def filehash(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic_json(path,data):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(path)
def loadjson(p):return json.loads(Path(p).read_text())
def log(event,**kw):
    record=dict(utc=datetime.now(timezone.utc).isoformat(),event=event,**kw)
    with (OUT/'progress.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
    print(json.dumps(record,ensure_ascii=False),flush=True)
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def state_hash(state):
    h=hashlib.sha256()
    for name,x in sorted(state.items()):h.update(name.encode());h.update(x.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def stable_clip(parameters):
    parameters=list(parameters)
    try:
        return torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True),False
    except RuntimeError:
        grads=[p.grad for p in parameters if p.grad is not None]
        # A finite gradient vector can have an overflowing float32 L2 norm.
        # Preserve the intended clipping operation; never accept NaN/Inf grads.
        if not all(bool(torch.isfinite(g).all()) for g in grads):raise
        norm=torch.sqrt(sum(torch.sum(g.double().square()) for g in grads))
        assert torch.isfinite(norm) and norm>0
        coefficient=torch.clamp(1./(norm+1e-6),max=1.)
        with torch.no_grad():
            for g in grads:g.copy_((g.double()*coefficient).to(g.dtype))
        after=torch.sqrt(sum(torch.sum(g.double().square()) for g in grads))
        assert torch.isfinite(after) and after<=1.000001
        return norm,True
def save_checkpoint(path,data):
    tmp=path.with_suffix('.tmp');torch.save(data,tmp);tmp.replace(path)

class Data:
    def __init__(self):
        raw=np.load(OUT/'data_verified.npz')
        self.values=raw['values'];self.times=raw['times'];self.a=raw['adjacency'];self.services=raw['services'].tolist()
        control.SERVICES=self.services
        self.cuts=[0,int(len(self.times)*.7),int(len(self.times)*.8),len(self.times)]
        assert self.cuts==[0,5603,6404,8005]
        self.bounds=dict(train=self.cuts[:2],validation=self.cuts[1:3],test=self.cuts[2:])
        self.indices={p:control.eligible(self.times,*b) for p,b in self.bounds.items()}
        t=self.values[:self.cuts[1]]
        self.mean=t.mean(0);self.scale=np.maximum(t.std(0),1e-8);self.median=np.median(t,axis=0)
        self.mase=np.abs(np.diff(t,axis=0))[np.diff(self.times[:self.cuts[1]])==1800].mean(0)
        assert (self.mase>0).all()
        for p,ix in self.indices.items():
            lo,hi=self.bounds[p];assert ix.min()-12>=lo and ix.max()<hi
            assert all(np.all(np.diff(self.times[t-12:t+1])==1800) for t in ix)
        self.hash=digest(self.values)

class Prepared:
    def __init__(self,data,r,partition,seed,subset=None):
        self.data=data;self.r=r;self.partition=partition
        self.indices=data.indices[partition] if subset is None else subset
        self.z=[];self.x=[];self.mask=[];self.audit=[]
        lo,hi=data.bounds[partition]
        previous={}
        normalized=((data.values[:,:,r]-data.mean[:,r])/data.scale[:,r]).astype(np.float32)
        for q,pattern in CASES:
            mask,counts=control.mask_for(data.times,lo,hi,q,pattern,seed)
            # Independent of resource; equal effective missing counts and nested rates.
            if pattern in previous:assert np.all(~previous[pattern] <= ~mask)
            previous[pattern]=mask.copy()
            z=control.prepare(data.values[:,:,r],mask,data.times,lo,hi,data.mean[:,r],data.scale[:,r],data.median[:,r])
            self.audit.append(dict(q=q,pattern=pattern,mask_sha256=digest(mask),counts=counts))
            self.z.append(z);self.x.append(np.where(mask,normalized,0)[:,:,None]);self.mask.append(mask[:,:,None])
        for i in (1,3,5):assert [c['ocultados'] for c in self.audit[i]['counts']]==[c['ocultados'] for c in self.audit[i+1]['counts']]
        self.z=torch.as_tensor(np.stack(self.z),device=DEVICE,dtype=torch.float32)
        self.x=torch.as_tensor(np.stack(self.x),device=DEVICE,dtype=torch.float32)
        self.mask=torch.as_tensor(np.stack(self.mask),device=DEVICE,dtype=torch.float32)
        self.y=torch.as_tensor(normalized,device=DEVICE,dtype=torch.float32)
        self.ix=torch.as_tensor(self.indices,device=DEVICE)
        self.history=torch.arange(-12,0,device=DEVICE)
    def batch(self,positions,cases):
        targets=self.ix[positions];hist=targets[:,None]+self.history
        return (self.z[cases[:,None],hist],self.x[cases[:,None],hist],self.mask[cases[:,None],hist],self.y[targets])

def predict(name,model,z,x,mask):return z[:,-1,:,0] if name=='persistencia' else model(z,x,mask)

class CapturedGinAR:
    """Replay the same CUDA operations without Python launch overhead.

    Dropout is zero and there are no running-statistic layers. Warm-up/capture
    do not take optimizer steps. Persistent gradients must never be set to None.
    """
    def __init__(self,model,inputs,training):
        self.static=tuple(x.detach().clone() for x in inputs)
        self.graph=torch.cuda.CUDAGraph()
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        def compute():
            z,x,mask,y=self.static
            if training:model.zero_grad(set_to_none=False)
            self.pred=model(z,x,mask)
            if training:
                self.loss=(self.pred-y).square().mean();self.loss.backward()
        with torch.cuda.stream(stream),torch.device('cuda'):
            for _ in range(3):compute()
        torch.cuda.current_stream().wait_stream(stream)
        with torch.device('cuda'),torch.cuda.graph(self.graph):compute()
    def replay(self,inputs):
        for dst,src in zip(self.static,inputs):dst.copy_(src)
        self.graph.replay()
        return self.pred

@torch.no_grad()
def evaluate(name,model,prepared,save_to=None,batch_size=128):
    if model is not None:model.eval()
    n=len(prepared.indices);allpred=np.empty((7,n,len(prepared.data.services)),dtype=np.float32)
    captured=None
    for c in range(7):
        for start in range(0,n,batch_size):
            positions=torch.arange(start,min(start+batch_size,n),device=DEVICE)
            cases=torch.full_like(positions,c)
            z,x,mask,y=prepared.batch(positions,cases)
            if name=='GinAR' and len(positions)==batch_size:
                if captured is None:captured=CapturedGinAR(model,(z,x,mask,y),False)
                p=captured.replay((z,x,mask,y))
            elif name=='GinAR':
                with torch.device('cuda'):p=model(z,x,mask)
            else:p=predict(name,model,z,x,mask)
            assert p.shape==y.shape and torch.isfinite(p).all()
            allpred[c,start:start+len(positions)]=p.cpu().numpy()
    data=prepared.data;r=prepared.r
    truth=data.values[prepared.indices,:,r]
    # Convert predictions back to original units in float64; preserve unmodified targets.
    physical=allpred.astype(np.float64)*data.scale[:,r]+data.mean[:,r]
    error=physical-truth[None]
    mae=np.abs(error).mean(1);rmse=np.sqrt(np.square(error).mean(1));mase=mae/data.mase[:,r]
    assert np.isfinite(mae).all() and np.isfinite(rmse).all() and np.isfinite(mase).all()
    if save_to is not None:
        np.savez_compressed(save_to,prediction_normalized=allpred,indices=prepared.indices,
                            target_sha256=digest(truth),prediction_sha256=digest(allpred))
    return dict(MAE=mae,RMSE=rmse,MASE=mase,prediction_sha256=digest(allpred),target_sha256=digest(truth))

def train(name,data,r,trainset,valset,seed,lr,epochs,patience,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    seed_all(seed);model=create(name,data.a,DEVICE)
    optim=torch.optim.Adam(model.parameters(),lr=lr)
    generator=torch.Generator().manual_seed(seed+500)
    start=0;best=float('inf');best_state=None;stale=0;history=[]
    if path.exists():
        ck=torch.load(path,map_location='cpu',weights_only=False)
        if ck.get('complete'):
            model.load_state_dict(ck['best_state']);return model,ck
        model.load_state_dict(ck['state']);optim.load_state_dict(ck['optimizer'])
        generator.set_state(ck['generator']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
        random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng'])
        start=ck['epoch']+1;best=ck['best'];best_state=ck['best_state'];stale=ck['stale'];history=ck['history']
    began=time.perf_counter();captured=None
    for epoch in range(start,epochs):
        model.train();order=torch.randperm(len(trainset.indices),generator=generator)
        losses=[];epoch_start=time.perf_counter();clip_overflows=0
        for b in order.split(16):
            pos=b.to(DEVICE);cases=(pos+epoch)%7
            z,x,mask,y=trainset.batch(pos,cases)
            if name=='GinAR' and len(pos)==16:
                if captured is None:captured=CapturedGinAR(model,(z,x,mask,y),True)
                p=captured.replay((z,x,mask,y));loss=captured.loss
                assert p.shape==y.shape and torch.isfinite(loss)
            else:
                optim.zero_grad(set_to_none=(name!='GinAR'))
                if name=='GinAR':
                    with torch.device('cuda'):p=model(z,x,mask)
                else:p=model(z,x,mask)
                assert p.shape==y.shape and torch.isfinite(p).all()
                loss=(p-y).square().mean();assert torch.isfinite(loss)
                loss.backward()
            norm,overflow=stable_clip(model.parameters())
            clip_overflows+=int(overflow)
            if overflow:log('finite_gradient_norm_overflow_repaired',model=name,resource=RESOURCES[r],seed=seed,epoch=epoch,norm_float64=float(norm))
            optim.step();losses.append(float(loss.detach()))
        metrics=evaluate(name,model,valset);score=float(metrics['MASE'].mean())
        if score<best:
            best=score;best_state=cpu_state(model);stale=0;best_epoch=epoch
        else:stale+=1;best_epoch=history[-1]['best_epoch']
        row=dict(epoch=epoch,loss=float(np.mean(losses)),validation_MASE=score,best_MASE=best,
                 best_epoch=best_epoch,seconds=time.perf_counter()-epoch_start,gradient_norm_overflow_fallbacks=clip_overflows)
        history.append(row)
        complete=(epoch==epochs-1 or (patience is not None and stale>=patience))
        ck=dict(model=name,resource=RESOURCES[r],seed=seed,lr=lr,epoch=epoch,state=cpu_state(model),
                optimizer=optim.state_dict(),best_state=best_state,best=best,stale=stale,history=history,
                generator=generator.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                python_rng=random.getstate(),numpy_rng=np.random.get_state(),complete=complete,
                parameter_count=sum(p.numel() for p in model.parameters()),state_sha256=state_hash(best_state))
        save_checkpoint(path,ck)
        log('epoch',model=name,resource=RESOURCES[r],seed=seed,lr=lr,checkpoint=path.name,**row)
        if complete:break
    model.load_state_dict(best_state)
    log('fit_complete',model=name,resource=RESOURCES[r],seed=seed,best_validation_MASE=best,seconds=time.perf_counter()-began)
    return model,ck

def checks(data):
    checks=[]
    for r in range(2):
        lo,hi=data.bounds['validation']
        mask,_=control.mask_for(data.times,lo,hi,20,'bloques',80001)
        args=(mask,data.times,lo,hi,data.mean[:,r],data.scale[:,r],data.median[:,r])
        original=control.prepare(data.values[:,:,r],*args)
        altered=data.values[:,:,r].copy();altered[~mask]+=1e10
        assert np.array_equal(original,control.prepare(altered,*args))
        target=int(data.indices['validation'][0]);altered=data.values[:,:,r].copy();altered[target:]+=1e10
        assert np.array_equal(original[target-12:target],control.prepare(altered,*args)[target-12:target])
        # All windows are causal and wholly within their chronological partition.
        p=Prepared(data,r,'validation',80001,control.sample_indices(data.indices['validation'],4))
        positions=torch.tensor([0,1,2,3],device=DEVICE);cases=torch.tensor([0,1,2,3],device=DEVICE)
        z,x,m,y=p.batch(positions,cases)
        for name in MODELS[1:]:
            seed_all(101);model=create(name,data.a,DEVICE);model.train()
            output=model(z,x,m);assert output.shape==y.shape and torch.isfinite(output).all()
            loss=(output-y).square().mean();loss.backward()
            assert all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters())
            if name.startswith('GRU'):assert sum(v.numel() for v in model.parameters())==841
            model.eval()
            with torch.no_grad():
                a=model(z,x,m);b=model(z,x,m);assert torch.equal(a,b)
                if name.startswith('GRU'):
                    i,j=next((int(i),int(j)) for i,j in np.argwhere(data.a>0) if i!=j)
                    changed=z.clone();changed[:,:,j,0]+=3
                    delta=float((model(changed,x,m)[:,i]-a[:,i]).abs().max())
                    assert delta==0 if name=='GRU_propia' else delta>1e-9
            checks.append(dict(resource=RESOURCES[r],model=name,parameters=sum(v.numel() for v in model.parameters()),finite_forward_backward=True,repeat_eval_exact=True))
            del model
        del p
    assert digest(data.values)==data.hash
    atomic_json(OUT/'validation_checks.json',dict(checks=checks,hidden_value_invariance=True,future_invariance=True,indices_within_partitions=True,
                eligible_windows={k:len(v) for k,v in data.indices.items()},cuts=data.cuts,original_values_sha256=data.hash))
    log('checks_passed',eligible_windows={k:len(v) for k,v in data.indices.items()})

def tune(data):
    choices={};records=[]
    for r in range(2):
        tr=Prepared(data,r,'train',70001,control.sample_indices(data.indices['train'],512))
        va=Prepared(data,r,'validation',80001,control.sample_indices(data.indices['validation'],32))
        scores={}
        for name in MODELS[1:]:
            scores[name]={}
            for lr in [.001,.003]:
                path=OUT/'tuning'/f'{name}_r{r}_lr{lr}.pt'
                model,ck=train(name,data,r,tr,va,101,lr,3,None,path)
                # Tuning compares the validation score at epoch 3, as specified,
                # with no extra choice of a tuning checkpoint.
                model.load_state_dict(ck['state'])
                score=float(evaluate(name,model,va)['MASE'].mean())
                scores[name][lr]=score
                records.append(dict(model=name,resource=RESOURCES[r],lr=lr,validation_MASE=score))
                del model
        joint={lr:(scores['GRU_propia'][lr]+scores['GRU_llamadas'][lr])/2 for lr in [.001,.003]}
        for name in MODELS[1:]:
            candidates=joint if name.startswith('GRU') else scores[name]
            choices[f'{name}_r{r}']=min(candidates,key=lambda lr:(candidates[lr],lr))
        del tr,va
    atomic_json(OUT/'selected_learning_rates.json',dict(choices=choices,records=records,selection='validation_only; epoch_3; shared_LR_for_GRU_pair'))
    log('tuning_complete',choices=choices)

def fit_all(data):
    choices=loadjson(OUT/'selected_learning_rates.json')['choices']
    frozen=[]
    for r in range(2):
        tr=Prepared(data,r,'train',70001);va=Prepared(data,r,'validation',80001)
        atomic_json(OUT/f'mask_audit_train_validation_r{r}.json',dict(train=tr.audit,validation=va.audit))
        for name in MODELS[1:]:
            for seed in SEEDS:
                path=OUT/'checkpoints'/f'{name}_r{r}_s{seed}.pt'
                model,ck=train(name,data,r,tr,va,seed,choices[f'{name}_r{r}'],20,5,path)
                frozen.append(dict(model=name,resource=RESOURCES[r],seed=seed,file=path.name,
                                   best_MASE=ck['best'],state_sha256=ck['state_sha256'],parameters=ck['parameter_count'],epochs=len(ck['history'])))
                del model
        del tr,va
    assert len(frozen)==30
    atomic_json(OUT/'frozen_models.json',dict(utc=datetime.now(timezone.utc).isoformat(),models=frozen,
               no_test_metrics_used=True,config_sha256=filehash(OUT/'run_config.json')))
    log('all_models_frozen',count=len(frozen))

def test_all(data):
    frozen=loadjson(OUT/'frozen_models.json');assert len(frozen['models'])==30 and frozen['no_test_metrics_used']
    assert frozen['config_sha256']==filehash(OUT/'run_config.json')
    (OUT/'test').mkdir(exist_ok=True)
    for r in range(2):
        for realization in range(20):
            marker=OUT/'test'/f'complete_r{r}_rep{realization}.json'
            if marker.exists():continue
            seed=90001+100003*realization
            prepared=Prepared(data,r,'test',seed)
            atomic_json(OUT/'test'/f'masks_r{r}_rep{realization}.json',dict(seed=seed,masks=prepared.audit))
            rows=[]
            for name in MODELS:
                for init in ([0] if name=='persistencia' else SEEDS):
                    model=None
                    if name!='persistencia':
                        ck=torch.load(OUT/'checkpoints'/f'{name}_r{r}_s{init}.pt',map_location='cpu',weights_only=False)
                        model=create(name,data.a,DEVICE);model.load_state_dict(ck['best_state'])
                    filename=f'{name}_r{r}_s{init}_rep{realization}'
                    metrics=evaluate(name,model,prepared,OUT/'test'/f'{filename}.npz')
                    for c,(q,pattern) in enumerate(CASES):
                        for n,service in enumerate(data.services):
                            rows.append(dict(model=name,resource=RESOURCES[r],initialization=init,realization=realization,
                                mask_seed=seed,q=q,pattern=pattern,service=service,n_windows=len(prepared.indices),
                                MAE=float(metrics['MAE'][c,n]),RMSE=float(metrics['RMSE'][c,n]),MASE=float(metrics['MASE'][c,n]),
                                prediction_sha256=metrics['prediction_sha256'],target_sha256=metrics['target_sha256']))
                    del model
            path=OUT/'test'/f'metrics_r{r}_rep{realization}.csv'
            with path.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            atomic_json(marker,dict(resource=RESOURCES[r],realization=realization,rows=len(rows),metrics_sha256=filehash(path)))
            log('test_realization_complete',resource=RESOURCES[r],realization=realization,rows=len(rows))
            del prepared
    log('test_complete',realizations=20)

def config():
    sources={str(p.relative_to(HERE)):filehash(p) for p in HERE.rglob('*.py') if '__pycache__' not in str(p)}
    specification=dict(version='grupo05-701020-cuda-v1.1',created_utc=datetime.now(timezone.utc).isoformat(),
        split=[.7,.1,.2],cuts=[0,5603,6404,8005],history=12,horizon_seconds=1800,
        change_from_attachment='User requested 70/10/20 in place of 70/15/15 and CUDA in place of CPU; no change to cohort, models or contrasts.',
        models=MODELS,resources=RESOURCES,conditions=CASES,train_seeds=SEEDS,test_mask_seeds=[90001+100003*r for r in range(20)],
        training_mask_seed=70001,validation_mask_seed=80001,batch_train=16,batch_evaluation=128,
        runtime_acceleration='GinAR CUDA graph replay; exact eager-equivalence checks on CPU/memory training updates, ragged batch and evaluation; fixed supports resident on GPU',
        epochs_max=20,patience=5,optimizer='Adam',gradient_clip=1.,learning_rates=[.001,.003],tuning_epochs=3,
        tuning_train_origins=512,tuning_validation_origins=32,tuning_GRU_pair='shared learning rate',
        epoch_schedule='condition=(chronological_training_origin_index+epoch)%7; one condition per origin per epoch',
        metrics=['MASE','MAE','RMSE'],normalization='training only; per service/resource z-score',
        inferential_status='Descriptive conditional bootstrap over 20 complete missingness realizations; no confirmatory p-values without sign-symmetry justification.',
        device=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
        source_sha256=sources,zip_sha256=(HERE/'zip.sha256').read_text().strip())
    p=OUT/'run_config.json'
    if p.exists():
        old=loadjson(p)
        for key in ['split','cuts','models','conditions','source_sha256','zip_sha256']:assert json.dumps(old[key],sort_keys=True)==json.dumps(specification[key],sort_keys=True),f'Configuration changed: {key}'
    else:atomic_json(p,specification)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['checks','tune','train','test','all'],default='all');args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    assert torch.cuda.is_available()
    config();data=Data()
    log('started',stage=args.stage,cuts=data.cuts,device=str(DEVICE))
    if args.stage in ['checks','all']:checks(data)
    if args.stage in ['tune','all']:tune(data)
    if args.stage in ['train','all']:fit_all(data)
    if args.stage in ['test','all']:test_all(data)
    assert digest(data.values)==data.hash
    log('stage_complete',stage=args.stage)

if __name__=='__main__':main()

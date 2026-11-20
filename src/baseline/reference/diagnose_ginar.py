import sys
sys.path.insert(0,'/app/study')
import run,torch
D=run.Data();tr=run.Prepared(D,0,'train',70001)
c=torch.load(run.OUT/'checkpoints/GinAR_r0_s303.pt',map_location='cpu',weights_only=False)
run.seed_all(303);model=run.create('GinAR',D.a,run.DEVICE);model.load_state_dict(c['state'])
optim=torch.optim.Adam(model.parameters(),lr=c['lr']);optim.load_state_dict(c['optimizer'])
gen=torch.Generator();gen.set_state(c['generator']);torch.set_rng_state(c['torch_rng']);torch.cuda.set_rng_state_all(c['cuda_rng'])
order=torch.randperm(len(tr.indices),generator=gen)
for step,b in enumerate(order.split(16)):
 p=b.to(run.DEVICE);z,x,m,y=tr.batch(p,(p+c['epoch']+1)%7)
 optim.zero_grad(set_to_none=True)
 pred=model(z,x,m);loss=(pred-y).square().mean();loss.backward()
 grads=[v.grad for v in model.parameters() if v.grad is not None]
 allfinite=all(bool(torch.isfinite(g).all()) for g in grads)
 norm32=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))
 if not bool(torch.isfinite(norm32)):
  norm64=torch.sqrt(sum(torch.sum(g.double()**2) for g in grads))
  print('FAIL',step,'positions',p.cpu().tolist(),'loss',float(loss.detach()),'all_grads_finite',allfinite,'norm32',float(norm32),'norm64',float(norm64),'max_grad',max(float(g.abs().max()) for g in grads),flush=True)
  break
 torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optim.step()
else: print('No eager failure this epoch',flush=True)

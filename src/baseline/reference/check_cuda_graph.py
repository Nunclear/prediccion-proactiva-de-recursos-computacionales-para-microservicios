import sys
sys.path.insert(0,'/app/study')
import run,torch,json,time
D=run.Data();records=[]
for r in range(2):
 ds=run.Prepared(D,r,'train',70001)
 run.seed_all(101);a=run.create('GinAR',D.a,run.DEVICE)
 run.seed_all(101);b=run.create('GinAR',D.a,run.DEVICE)
 oa=torch.optim.Adam(a.parameters(),lr=.003);ob=torch.optim.Adam(b.parameters(),lr=.003)
 graph=None
 for step,size in enumerate([16,7,16,16]):
  pos=torch.arange(step*16,step*16+size,device='cuda');inputs=ds.batch(pos,(pos+step)%7);z,x,m,y=inputs
  a.zero_grad(set_to_none=True);pa=a(z,x,m);la=(pa-y).square().mean();la.backward()
  if size==16:
   if graph is None:graph=run.CapturedGinAR(b,inputs,True)
   pb=graph.replay(inputs);lb=graph.loss
  else:
   b.zero_grad(set_to_none=False)
   with torch.device('cuda'):pb=b(z,x,m);lb=(pb-y).square().mean();lb.backward()
  assert torch.equal(pa,pb)
  assert torch.equal(la,lb)
  maxdiff=max(float((va.grad-vb.grad).abs().max()) for va,vb in zip(a.parameters(),b.parameters()) if va.grad is not None)
  assert maxdiff==0
  torch.nn.utils.clip_grad_norm_(a.parameters(),1.);torch.nn.utils.clip_grad_norm_(b.parameters(),1.)
  oa.step();ob.step();assert run.state_hash(a.state_dict())==run.state_hash(b.state_dict())
  records.append(dict(resource=r,step=step,batch=size,prediction_equal=True,loss_equal=True,max_gradient_difference=maxdiff,state_sha256=run.state_hash(a.state_dict())))
 a.eval();b.eval();pos=torch.arange(128,device='cuda');inputs=ds.batch(pos,pos%7)
 with torch.no_grad():
  pa=a(*inputs[:3]);inference=run.CapturedGinAR(b,inputs,False);pb=inference.replay(inputs)
  assert torch.equal(pa,pb)
 records.append(dict(resource=r,inference_batch128_exact=True))
 del graph,inference,a,b,ds
path=run.OUT/'cuda_graph_equivalence.json';path.write_text(json.dumps(dict(status='passed',training_and_eval_exact=True,includes_ragged_last_batch=True,records=records),indent=2))
print('CUDA_GRAPH_EQUIVALENCE_PASSED',flush=True)

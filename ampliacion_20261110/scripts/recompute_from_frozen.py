"""Rebuild prediction arrays from archived frozen weights, never tune or fit."""
from pathlib import Path
import sys,json,argparse,csv,gc,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from core import *
from models import *
from engine import evaluate
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'predictions_recomputed'));p.add_argument('--limit',type=int);p.add_argument('--model');p.add_argument('--dataset',choices=['new_services','alibaba_transfer']);args=p.parse_args()
 out=Path(args.output).resolve();assert out!=ROOT.resolve(),'Use a separate prediction output directory'
 if DEVICE.type=='cuda':torch.cuda.set_per_process_memory_fraction(.28)
 with (ROOT/'results/analysis/prediction_manifest.csv').open() as f:manifest=list(csv.DictReader(f))
 if args.model:manifest=[item for item in manifest if Path(item['path']).name.startswith(args.model+'_')]
 if args.dataset:manifest=[item for item in manifest if Path(item['path']).parent.name==args.dataset]
 if args.limit is not None:manifest=manifest[:args.limit]
 data_cache={};prep_key=None;ds=None;folder_key=None;frozen=None;reports=[]
 import re
 for item in manifest:
  relative=Path(item['path']);name=relative.name;m=re.fullmatch(r'(.+)_r(\d)_s(\d+)_rep(\d+)\.npz',name);assert m,name
  model,r,seed,rep=m[1],int(m[2]),int(m[3]),int(m[4]);label=relative.parent.name;setting=relative.parts[2] if relative.parts[1]=='baseline' else relative.parts[2]
  sm=re.fullmatch(r'L(\d+)_H(\d+)',setting);L,h=map(int,sm.groups());folder=ROOT/Path(*relative.parts[:3])
  if folder_key!=folder:
   frozen=json.loads((folder/'frozen.json').read_text())['models'];folder_key=folder
  if label not in data_cache:data_cache[label]=Series(ROOT/'data/processed'/('alibaba_transfer.npz' if label=='alibaba_transfer' else 'chronograph_holdout.npz'),external=label=='alibaba_transfer')
  key=(label,r,L,h,rep)
  if prep_key!=key:
   if ds is not None:del ds;gc.collect()
   ds=Prepared(data_cache[label],r,'test',L,h,90001+100003*rep);prep_key=key
  target=out/relative;target.parent.mkdir(parents=True,exist_ok=True)
  if not target.exists():
   record=next(x for x in frozen if x['model']==model and x['r']==r and x['seed']==seed)
   ck=torch.load(folder/'checkpoints'/record['file'],map_location='cpu',weights_only=False);assert state_hash(ck['best_state'])==record['state_sha256']
   predictor=create(model,L,ds.data.a,DEVICE,ck.get('parameter'));load_for_target(predictor,ck['best_state']);evaluate(predictor,ds,save_to=target);del predictor,ck
  with np.load(target) as z:
   assert str(z['prediction_sha256'])==item['prediction_array_sha256'],target
   assert str(z['target_sha256'])==item['target_array_sha256'],target
  reports.append(dict(path=str(target),prediction_array_sha256=item['prediction_array_sha256'],status='exact_array_digest_match'))
  print('RECONSTRUCTED',len(reports),len(manifest),name,flush=True)
 atomic_json(out/'verification.json',dict(status='passed',predictions=len(reports),limit=args.limit,model_filter=args.model,dataset_filter=args.dataset,models_retrained=False,checks=reports))
if __name__=='__main__':main()

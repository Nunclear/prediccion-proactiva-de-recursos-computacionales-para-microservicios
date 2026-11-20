"""Select a spatial holdout from topology only; stream public full data."""
from pathlib import Path
import json,csv,hashlib,shutil,os
import numpy as np,ijson
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 raw=Path(os.environ.get('CHRONOGRAPH_DATASET',str(BASE.parent/'ChronoGraph/dataset')))/'node_features.json'
 old=np.load(BASE/'results/reproduccion/data_verified.npz')
 excluded=set(old['services'].tolist())|{'service030','service033','service035','service040'}
 edges=list(csv.DictReader((raw.parent/'edges.csv').open()))
 graph={}
 for e in edges:
  a,b=e['source'],e['target']
  if a in excluded or b in excluded:continue
  graph.setdefault(a,set()).add(b);graph.setdefault(b,set()).add(a)
 def key(s):return hashlib.sha256(('prospective-20261110|'+s).encode()).hexdigest()
 selected=None
 for start in sorted(graph,key=key):
  visited={start};queue=[start]
  while queue and len(visited)<32:
   node=queue.pop(0)
   for neighbor in sorted(graph[node],key=key):
    if neighbor not in visited:visited.add(neighbor);queue.append(neighbor)
    if len(visited)==32:break
  if len(visited)==32:selected=sorted(visited);break
 assert selected and not set(selected)&excluded
 spec=dict(services=selected,excluded=sorted(excluded),selection='topology-only BFS from SHA256-ranked available roots; convenience connected subgraph, not random population sampling',source_file_sha256=sha(raw),metrics_inspected_for_selection=False)
 assert spec['source_file_sha256']=='907407a33cb5b59ddc276185e2a78ef869f41c6f7a919a0c340fc20467de5e5a'
 p=ROOT/'configs/cohort_services.json'
 if p.exists():assert json.loads(p.read_text())==spec
 else:p.write_text(json.dumps(spec,indent=2))
 # Test values are stored for later code-controlled evaluation, not summarized here.
 found={}
 with raw.open('rb') as f:
  for name,features in ijson.kvitems(f,'',use_float=True):
   if name in selected:
    values=[];times=None
    for resource in ['cpu_usage','container_memory_usage_bytes']:
     r=features[resource];t=np.asarray(r['steps'],dtype=np.float64);v=np.asarray(r['values'],dtype=np.float64)
     if times is None:times=t
     else:assert np.array_equal(times,t)
     assert len(v)==8005 and np.isfinite(v).all();values.append(v)
    found[name]=(times,np.stack(values,-1))
   if len(found)==32:break
 assert set(found)==set(selected)
 times=found[selected[0]][0]
 assert all(np.array_equal(times,found[n][0]) for n in selected)
 adjacency=np.zeros((32,32),np.float32)
 for e in edges:
  if e['source'] in selected and e['target'] in selected:adjacency[selected.index(e['source']),selected.index(e['target'])]=1
 values=np.stack([found[n][1] for n in selected],1)
 np.savez_compressed(ROOT/'data/processed/chronograph_holdout.npz',values=values,times=times,services=np.asarray(selected),adjacency=adjacency)
 shutil.copy2(BASE/'results/reproduccion/data_verified.npz',ROOT/'data/processed/chronograph_source.npz')
 spec.update(shape=list(values.shape),internal_directed_edges=int(adjacency.sum()),processed_sha256=sha(ROOT/'data/processed/chronograph_holdout.npz'),source_cohort_services=old['services'].tolist())
 (ROOT/'data/manifests/chronograph.json').write_text(json.dumps(spec,indent=2));print(json.dumps(spec,indent=2),flush=True)
if __name__=='__main__':main()

"""Final local-delivery gate; completed computations are distinct from submission readiness."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads((ROOT/p).read_text())
def main():
 checks={};required=[
 'README.md','Dockerfile','compose.yaml','requirements.txt','configs/protocolo_congelado.yaml',
 'docs/AUDITORIA_INICIAL.md','docs/PROTOCOLO_EXPERIMENTAL.md','docs/ENMIENDA_PROTOCOLO_PRETEST.md',
 'docs/ESTADO_DEL_ARTE.md','docs/BIBLIOGRAFIA_VERIFICADA.bib','docs/SELECCION_REVISTA.md',
 'docs/MATRIZ_CUMPLIMIENTO_EDITORIAL.md','docs/PENDIENTES_Y_LIMITACIONES.md',
 'docs/CARTA_PRESENTACION_EN.md','docs/DECLARACIONES_EN.md','docs/CHECKLIST_PREVIO_ENVIO.md','docs/INFORME_FINAL.md',
 'articulo_final/main.tex','articulo_final/revision_espanol.tex','articulo_final/referencias.bib','articulo_final/main.pdf','articulo_final/revision_espanol.pdf',
 'results/analysis/primary_summary.csv','results/analysis/primary_contrasts.json','results/analysis/common_origin_metrics.csv.gz',
 'results/analysis/by_temporal_block.csv','results/analysis/checkpoint_manifest.csv','results/analysis/prediction_manifest.csv',
 'experiments/benchmarks/latency.csv','experiments/benchmarks/latency_trials.csv','experiments/benchmarks/training.csv',
 'results/resource_summary.json','results/pdf_qa.json']
 missing=[p for p in required if not (ROOT/p).is_file()];assert not missing,missing;checks['required_files']=len(required)
 total_fits=0
 folders=[ROOT/'experiments/baseline/L12_H1']+[ROOT/'experiments/ablations'/f'L{L}_H{h}' for L,h in [(12,2),(12,4),(12,6),(12,12),(24,1),(48,1)]]
 for folder in folders:
  complete=json.loads((folder/'COMPLETED.json').read_text());assert complete['status']=='complete' and not complete['failures'];total_fits+=complete['neural_fits']
  frozen=json.loads((folder/'frozen.json').read_text());stamp=datetime.fromisoformat(frozen['frozen_utc']);assert not frozen['failures']
  markers=list((folder/'test/new_services').glob('complete_r*_rep*.json'));assert len(markers)==(40 if folder.name=='L12_H1' else 10)
  for p in markers:assert datetime.fromisoformat(json.loads(p.read_text())['utc'])>stamp
 assert total_fits==364;checks['neural_final_fits']=total_fits
 for name in ['results/analysis/COMPLETED.json','experiments/benchmarks/COMPLETED.json','experiments/external_validation/COMPLETED.json','results/data_verification.json','results/historical/audit.json','results/resume_test/PASSED.json','results/cuda_replay_equivalence.json','results/cuda_lifecycle/PASSED.json','results/saved_prediction_audit_smoke.json']:
  assert read(name)['status'] in ['passed','complete'],name
 audit=read('results/analysis/audit.json');assert audit['npz_files']==9680 and audit['metric_rows']==3097600;checks['new_prediction_files']=audit['npz_files'];checks['new_metric_rows']=audit['metric_rows']
 assert read('results/analysis/checkpoint_audit.json')['final_checkpoints']==420
 contrasts=read('results/analysis/primary_contrasts.json')['contrasts'];assert len(contrasts)==4
 assert {(c['resource'],c['reference']) for c in contrasts}=={(r,b) for r in ['cpu_usage','container_memory_usage_bytes'] for b in ['DLinear','E5']}
 for c in contrasts:
  assert 0<=c['p_sign']<=c['p_Holm']<=1
  assert abs(sum(c['seed_differences'])/10-c['mean_paired_MASE_difference'])<1e-10
 figs=read('articulo_final/figuras/manifest.json')['figures'];assert len(figs)==15
 for fig in figs:
  for ext in ['pdf','svg','png']:assert (ROOT/'articulo_final/figuras'/(fig['id']+'.'+ext)).is_file()
  assert sha(ROOT/'articulo_final/figuras'/(fig['id']+'.pdf'))==fig['pdf_sha256']
 checks['verified_figures']=len(figs)
 qa=read('results/pdf_qa.json');assert qa['status']=='passed' and qa['visual_review_completed']
 for stem in ['main','revision_espanol']:assert qa['pdfs'][stem]['sha256']==sha(ROOT/'articulo_final'/(stem+'.pdf'))
 authors=read('configs/author_confirmations.json');assert authors['project_group_alias']=='Grupo 5'
 assert [x['name'] for x in authors['authors_in_order']]==['Carlos Augusto Mullisaca Condori','Jhon Alex Gutiérrez Morales']
 assert authors['corresponding_author'] is None
 assert not [x for x in read('docs/bibliography_manifest.json') if x['status']=='unavailable'];checks['verified_reference_records']=len(read('docs/bibliography_manifest.json'))
 files=[dict(path=p,bytes=(ROOT/p).stat().st_size,sha256=sha(ROOT/p)) for p in required]
 result=dict(status='local_computational_delivery_verified',checked_utc=datetime.now(timezone.utc).isoformat(),checks=checks,files=files,submission_ready=False,editorial_pending_document='docs/PENDIENTES_Y_LIMITACIONES.md',scope='Computations, artifacts and local manuscript verified; human authorship/originality/licensing/release and journal requirements still require author resolution. No acceptance guarantee or submission performed.')
 (ROOT/'results/FINAL_VERIFICATION.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
if __name__=='__main__':main()

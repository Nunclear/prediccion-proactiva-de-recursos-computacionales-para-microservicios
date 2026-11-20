"""Audit published values, provenance, freeze order and preservation."""
from pathlib import Path
import sys,json,hashlib,re,struct,unittest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
import numpy as np,pandas as pd,torch
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class DeliveryTests(unittest.TestCase):
 def test_original_preserved_and_sources(self):
  self.assertEqual(sha(ROOT/'articulo/ARTICULO_EMPIRICO_ORIGINAL.md'),sha(BASE/'articulo/ARTICULO_EMPIRICO.md'))
  manifest=json.loads((ROOT/'docs/MANIFIESTO_ENTREGA_ANTERIOR.json').read_text())
  for name in ['results/estadisticas/metricas_completas.csv','src/improved/models.py','src/baseline/reference/run.py']:
   self.assertEqual(sha(BASE/name),manifest[name],name)
  protocol=json.loads((ROOT/'results/nuevos_experimentos/protocol.json').read_text())
  for name in ['src/compact.py','scripts/run_v2.py','tests/test_compact.py','tests/smoke_v2.py']:
   self.assertEqual(sha(ROOT/name),protocol['source_sha256'][name],name)
 def test_freeze_selection_and_weights(self):
  own=ROOT/'results/nuevos_experimentos';frozen=json.loads((own/'frozen_v2.json').read_text())
  first_test=next(json.loads(line)['utc'] for line in (own/'progress.jsonl').read_text().splitlines() if json.loads(line)['event']=='v2_test_complete')
  self.assertLess(frozen['frozen_utc'],first_test)
  for r in range(2):
   expected=min(['C0','C1','C2','C3'],key=lambda n:(frozen['family_validation'][f'{n}_r{r}']['score'],frozen['family_validation'][f'{n}_r{r}']['parameters'],n))
   self.assertEqual(expected,frozen['router'][str(r)])
  self.assertEqual(len(frozen['models']),24)
  for row in frozen['models']:
   ck=torch.load(own/'checkpoints'/f"{row['model']}_r{row['r']}_s{row['seed']}.pt",map_location='cpu',weights_only=False)
   self.assertTrue(ck['complete']);self.assertEqual(ck['parameter_count'],row['parameters']);h=hashlib.sha256()
   for k,v in sorted(ck['best_state'].items()):h.update(k.encode());h.update(v.numpy().tobytes());self.assertTrue(torch.isfinite(v).all())
   self.assertEqual(h.hexdigest(),row['state_sha256'])
 def test_statistics_and_alias(self):
  c=pd.read_csv(ROOT/'results/comparaciones/contrastes_dos_escalas.csv')
  old=pd.read_csv(BASE/'results/estadisticas/comparaciones_emparejadas.csv')
  for resource in ['cpu_usage','container_memory_usage_bytes']:
   a=c[(c.reference=='E0')&(c.model=='E5')&(c.resource==resource)&(c.conditions=='perdidas')].iloc[0]
   b=old[(old.variant=='E5')&(old.resource==resource)&(old.conditions=='perdidas')].iloc[0]
   np.testing.assert_allclose([a.reduction_percent,a.delta_MASE_low,a.delta_MASE_high],[b.reduction_percent,b.IC95_low,b.IC95_high],rtol=1e-10,atol=1e-12)
   self.assertLess(a.reduction_percent_low,a.reduction_percent_high)
   self.assertLess(a.crossed_delta_MASE_low,0);self.assertGreater(a.crossed_delta_MASE_high,0)
  v=json.loads((ROOT/'results/comparaciones/validacion.json').read_text())
  self.assertEqual(v['new_predictions_recomputed'],480);self.assertEqual(v['original_prediction_hashes_verified'],1240)
  self.assertEqual(v['physical_metric_rows'],337120);self.assertEqual(v['rows_including_V2_alias'],360640)
  # Distinguish mean paired percentage from percentage of pooled means.
  a=np.array([1.,100.]);b=np.array([0.,50.])
  self.assertEqual(float((100*(a-b)/a).mean()),75.)
  self.assertNotAlmostEqual(float((100*(a-b)/a).mean()),float(100*(a.mean()-b.mean())/a.mean()))
 def test_deliverables_links_figures_and_benchmark(self):
  required=['AUDITORIA_ARTICULO','MEJORAS_MODELO','METODOLOGIA_ESTADISTICA','LIMITACIONES','REFERENCIAS_VERIFICADAS','EVALUACION_FINAL']
  for name in required:self.assertTrue((ROOT/'docs'/f'{name}.md').exists(),name)
  files=[ROOT/'README.md']+list((ROOT/'articulo').glob('*.md'))+[ROOT/'docs'/f'{name}.md' for name in required]+[ROOT/'docs/GUIA_REPRODUCCION.md']
  for p in files:
   for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text()):
    if '://' in link or link.startswith('#'):continue
    self.assertTrue((p.parent/link.split('#')[0]).exists(),(str(p),link))
  figs=list((ROOT/'articulo/figuras').glob('fig*.png'));self.assertEqual(len(figs),13)
  for p in figs:
   data=p.read_bytes();self.assertEqual(data[:8],b'\x89PNG\r\n\x1a\n');width,height=struct.unpack('>II',data[16:24]);self.assertGreater(width,1800);self.assertGreater(height,600);self.assertTrue(p.with_suffix('.svg').exists())
  article=(ROOT/'articulo/ARTICULO_EMPIRICO_MEJORADO.md').read_text()
  self.assertFalse(re.search(r'@@[A-Z_]+@@',article));self.assertEqual(article.count('$$')%2,0)
  metadata=json.loads((ROOT/'results/comparaciones/articulo_trazabilidad.json').read_text())
  self.assertEqual(sha(ROOT/'articulo/ARTICULO_EMPIRICO_MEJORADO.md'),metadata['article_sha256'])
  self.assertTrue(200<=metadata['abstract_words']<=250)
  cost=pd.read_csv(ROOT/'results/comparaciones/costes_nueva_medicion.csv');samples=pd.read_csv(ROOT/'results/comparaciones/latencias_individuales.csv')
  self.assertEqual(len(cost),28);self.assertEqual(len(samples),2800);self.assertTrue((cost.repetitions==100).all())
  for row in cost.itertuples():
   times=samples[(samples.model==row.model)&(samples.resource==row.resource)].latency_ms.values
   np.testing.assert_allclose(np.median(times),row.latency_p50_ms,rtol=1e-12,atol=1e-12)
  for record in json.loads((ROOT/'docs/CROSSREF_METADATOS.json').read_text()):self.assertTrue(record['verified'])
if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(DeliveryTests);result=unittest.TextTestRunner(verbosity=2).run(suite)
 if result.wasSuccessful():(ROOT/'results/VALIDACION_FINAL.json').write_text(json.dumps(dict(status='passed',tests=result.testsRun,original_unchanged=True,freeze_before_test=True,metric_units_checked=True,figures=13,benchmark_calls=2800),indent=2))
 sys.exit(0 if result.wasSuccessful() else 1)

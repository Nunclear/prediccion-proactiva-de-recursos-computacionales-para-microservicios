"""Translate display labels only; preserve audited numeric cell strings verbatim."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];T=ROOT/'articulo_final/tablas'
common=[('Exponential smoothing','Suavizado exp.'),('Moving average','Media móvil'),('Persistence','Persistencia'),('GRU-D adaptation','GRU-D (adaptación)'),('GRU (own)','GRU (propia)'),('GRU (graph)','GRU (grafo)'),('PatchTST (small)','PatchTST (pequeño)'),('Ridge','Regresión ridge'),('Model','Modelo')]
by={'primary':[('CPU MASE','CPU MASE'),('Mem. MASE','Mem. MASE')],
'contrasts':[('Resource','Recurso'),('Reference','Referencia'),('Mean $R$','Media $R$'),('Conditional 95\\% interval','Intervalo 95\\% cond.'),('Positive seeds','Semillas a favor'),('Memory','Memoria')],
'efficiency':[('Parameters','Parámetros'),('Epoch sum (s)','Suma épocas (s)'),('not timed','sin tiempo')]}
for name,labels in by.items():
 s=(T/f'{name}.tex').read_text()
 for original,translated in common+labels:s=s.replace(original,translated)
 (T/f'es_{name}.tex').write_text(s)
print('SPANISH_TABLE_LABELS_TRANSLATED',len(by))

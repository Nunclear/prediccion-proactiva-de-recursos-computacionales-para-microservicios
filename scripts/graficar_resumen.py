"""Replot preserved primary means; no model fitting or invented observations."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def main():
 rows=list(csv.DictReader((ROOT/'ampliacion_20261110/results/analysis/primary_summary.csv').open()))
 names=['MovingAverage','DLinear','E5','A0','A3','NLinear','PatchTST']
 fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
 for ax,r,title in zip(axes,['cpu_usage','container_memory_usage_bytes'],['CPU','Memoria']):
  values={x['model']:float(x['MASE']) for x in rows if x['resource']==r}
  ax.barh(names,[values[n] for n in names],color=['#7d8993' if n!='A3' else '#008f87' for n in names]);ax.set_title(title);ax.set_xlabel('MASE (menor es mejor)');ax.invert_yaxis()
 fig.suptitle('Transferencia calibrada: 32 servicios; L12/H1; seis condiciones con pérdidas')
 out=ROOT/'figures';out.mkdir(exist_ok=True)
 for ext in ['pdf','png']:fig.savefig(out/f'resumen_principal.{ext}',dpi=180)
 print(out/'resumen_principal.pdf')
if __name__=='__main__':main()

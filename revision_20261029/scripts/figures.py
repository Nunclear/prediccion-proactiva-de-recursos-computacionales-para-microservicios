"""Scientific figures from authenticated predictions, metrics and telemetry."""
import os,sys,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
sys.path[:0]=[str(BASE/'src'),str(BASE/'src/baseline/reference')]
os.environ['CHRONOGRAPH_OUTPUT']=str(BASE/'results/reproduccion')
import run as e
import numpy as np,pandas as pd,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from improved.models import DLinearEMAMask
FIG=ROOT/'articulo/figuras';OUT=ROOT/'results/comparaciones'
NAMES={'cpu_usage':'CPU (escala publicada)','container_memory_usage_bytes':'Memoria (bytes)'}
SHORT={'cpu_usage':'CPU','container_memory_usage_bytes':'Memoria'}
def save(name):
 plt.savefig(FIG/f'{name}.png',dpi=300,bbox_inches='tight');plt.savefig(FIG/f'{name}.svg',bbox_inches='tight');plt.close()
def main():
 assert (ROOT/'results/benchmark_COMPLETE.json').exists()
 for p in (BASE/'articulo/figuras').glob('*.png'):shutil.copy2(p,FIG/p.name)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'figure.dpi':110})
 data=e.Data();summary=pd.read_csv(OUT/'resumen_condiciones.csv');globalmeans=pd.read_csv(OUT/'resumen_perdidas.csv');contrasts=pd.read_csv(OUT/'contrastes_dos_escalas.csv');cost=pd.read_csv(OUT/'costes_nueva_medicion.csv')
 router=json.loads((ROOT/'results/nuevos_experimentos/frozen_v2.json').read_text())['router']
 fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True);hours=(data.times-data.times[0])/3600
 for r,ax in enumerate(axes):
  values=data.values[:,0,r].copy();values[np.r_[False,np.diff(data.times)>1800]]=np.nan;ax.plot(hours,values,lw=.6,color='C0');ax.set_ylabel(NAMES[e.RESOURCES[r]])
  for lo,hi,c,label in [(0,5603,'C2','Entrenamiento'),(5603,6404,'C1','Validación'),(6404,8005,'C3','Prueba')]:ax.axvspan(hours[lo],hours[hi-1],alpha=.09,color=c,label=label)
 axes[0].set_title(f'Serie publicada · {data.services[0]}');axes[0].legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.27));axes[-1].set_xlabel('Horas desde el primer registro');fig.tight_layout();save('fig01_series')
 z=np.zeros((1,12,1,5),dtype=np.float32);z[0,:,0,0]=(data.values[64:76,0,0]-data.mean[0,0])/data.scale[0,0];z[...,1]=1
 a=DLinearEMAMask('E5',alpha=.7);b=DLinearEMAMask('E1',kernel=25)
 with torch.no_grad():te,re=a.components(torch.tensor(z));tm,rm=b.components(torch.tensor(z))
 fig,axes=plt.subplots(2,1,figsize=(8,5),sharex=True);axes[0].plot(z[0,:,0,0],label='Historia');axes[0].plot(te[0,:,0],label='Nivel EMA α=0,7');axes[0].plot(tm[0,:,0],label='Promedio móvil k=25');axes[0].legend(ncol=3);axes[0].set_ylabel('Valor normalizado');axes[1].plot(re[0,:,0],label='Residual EMA');axes[1].plot(rm[0,:,0],label='Residual MA');axes[1].legend();axes[1].set(xlabel='Posición en la historia (pasos de 30 min)',ylabel='Residual');fig.tight_layout();save('fig02_descomposicion')
 for pattern,number in [('disperso',3),('bloques',4)]:
  lo,hi=data.bounds['validation'];mask,_=e.control.mask_for(data.times,lo,hi,20,pattern,80001);ix=np.arange(lo,lo+120);v=data.values[ix,0,0]
  plt.figure(figsize=(10,3));plt.plot((ix-lo)*.5,v,lw=1,label='Observación original');plt.scatter((ix[~mask[ix,0]]-lo)*.5,v[~mask[ix,0]],s=24,color='C3',label='Entrada ocultada');plt.ylabel('CPU (escala publicada)');plt.xlabel('Horas desde inicio de validación');plt.title(f'Pérdida {pattern}, 20 % global · {data.services[0]}');plt.legend(ncol=2);plt.tight_layout();save(f'fig0{number}_ausencias')
 fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
 for r,ax in enumerate(axes):
  ix=data.indices['test'][:120];ax.plot((ix-ix[0])*.5,data.values[ix,0,r],label='Real',color='black',lw=1)
  for label,folder,name in [('E0',BASE/'results/reproduccion','DLinear'),('E5',BASE/'results/mejoras','E5'),('V2',ROOT/'results/nuevos_experimentos',router[str(r)])]:
   p=np.load(folder/'test'/f'{name}_r{r}_s101_rep0.npz')['prediction_normalized'][4,:120,0]*data.scale[0,r]+data.mean[0,r];ax.plot((ix-ix[0])*.5,p,label=label,lw=.9)
  ax.set_ylabel(NAMES[e.RESOURCES[r]])
 axes[0].set_title(f'{data.services[0]} · 20 % bloques · semilla 101 · realización 0');axes[0].legend(ncol=4);axes[-1].set_xlabel('Horas desde el primer origen de prueba');fig.tight_layout();save('fig05_pronosticos')
 fig,axes=plt.subplots(1,2,figsize=(12,4.5))
 for r,ax in enumerate(axes):
  g=globalmeans[(globalmeans.resource==e.RESOURCES[r])&~globalmeans.model.str.startswith('C')].sort_values('MASE');ax.scatter(g.MASE,np.arange(len(g)),s=32);ax.set_yticks(np.arange(len(g)),g.model.str.replace('BiTGraph_codigo_autor','BiTGraph',regex=False))
  if r==0:ax.set_xscale('log')
  ax.set(title=SHORT[e.RESOURCES[r]],xlabel='MASE con pérdidas ('+('log' if r==0 else 'lineal')+')');ax.axvline(1,color='gray',lw=.6)
 fig.tight_layout();save('fig06_comparadores')
 labels=['E0','E2','E4','E5','V2'];fig,axes=plt.subplots(2,2,figsize=(11,7))
 for r in range(2):
  for j,pattern in enumerate(['disperso','bloques']):
   ax=axes[r,j]
   for label in labels:
    g=summary[(summary.resource==e.RESOURCES[r])&(summary.model==label)&((summary.pattern==pattern)|(summary.q==0))].sort_values('q');ax.plot(g.q,g.MASE,marker='o',ms=4,label=label)
   ax.set(title=f'{SHORT[e.RESOURCES[r]]} · {pattern}',xlabel='Pérdida global (%)',ylabel='MASE');ax.legend(ncol=3,fontsize=8)
 fig.tight_layout();save('fig07_robustez')
 fig,ax=plt.subplots(figsize=(8,4.5));g=globalmeans[globalmeans.model.isin(labels)].pivot(index='model',columns='resource',values='MASE').loc[labels];g.columns=[SHORT[c] for c in g.columns];g.plot.bar(ax=ax,rot=0);ax.set(xlabel='Configuración',ylabel='MASE con pérdidas');ax.legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,1.15));ax.set_ylim(0,float(g.max().max())*1.15);fig.tight_layout();save('fig08_recursos')
 fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
 for r,ax in enumerate(axes):
  s=contrasts[(contrasts.resource==e.RESOURCES[r])&(contrasts.reference=='E0')&(contrasts.conditions=='perdidas')].set_index('model').loc[['E2','E4','E5','V2']]
  for j,(_,row) in enumerate(s.iterrows()):
   ax.hlines(j-.08,row.delta_MASE_low,row.delta_MASE_high,color='C0',lw=3);ax.scatter(row.delta_MASE,j-.08,color='C0',s=22)
   ax.hlines(j+.08,row.crossed_delta_MASE_low,row.crossed_delta_MASE_high,color='C1',lw=2);ax.scatter(row.delta_MASE,j+.08,color='C1',s=18)
  ax.axvline(0,color='black',lw=.7);ax.set(title=SHORT[e.RESOURCES[r]],xlabel='Δ MASE = E0 − configuración');ax.set_yticks(range(4),['E2','E4','E5','V2'])
 axes[0].plot([],[],color='C0',lw=3,label='Solo máscaras');axes[0].plot([],[],color='C1',lw=2,label='Semillas y máscaras');axes[0].legend(loc='upper center',bbox_to_anchor=(1.05,1.32),ncol=2);fig.tight_layout();save('fig09_incertidumbre')
 usage=pd.concat([pd.read_csv(BASE/'results/monitoreo/resources.csv'),pd.read_csv(ROOT/'results/monitoreo/resources.csv')],ignore_index=True).sort_values('unix_seconds');hours=(usage.unix_seconds-usage.unix_seconds.iloc[0])/3600;gap=usage.unix_seconds.diff()>30
 fig,axes=plt.subplots(4,1,figsize=(11,8),sharex=True)
 for col,ax,mult,label in [('cpu_percent',axes[0],1,'CPU (%)'),('gpu_utilization_percent',axes[0],1,'GPU (%)'),('gpu_temperature_C',axes[1],1,'GPU (°C)'),('ram_available_bytes',axes[2],1/1024**3,'RAM disponible (GiB)'),('vram_used_MiB',axes[3],1/1024,'VRAM del sistema (GiB)')]:
  value=usage[col].copy()*mult;value.loc[gap]=np.nan;ax.plot(hours,value,lw=.7,label=label);ax.legend(loc='upper left')
 axes[-1].set_xlabel('Horas desde el inicio del monitoreo; huecos sin medición');fig.tight_layout();save('fig10_recursos_sistema')
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 for r,ax in enumerate(axes):
  g=cost[(cost.resource==e.RESOURCES[r])&cost.model.isin(labels+['GRU_propia','GRU_llamadas'])].merge(globalmeans,on=['model','resource'])
  for row in g.itertuples():
   ax.scatter(row.latency_p50_ms,row.MASE,s=35)
   offset=(-22,-16) if r==1 and row.model=='V2' else (4,4)
   ax.annotate(row.model,(row.latency_p50_ms,row.MASE),xytext=offset,textcoords='offset points',fontsize=8,arrowprops=dict(arrowstyle='-',lw=.5) if offset!= (4,4) else None)
  ax.set(xlabel='Latencia mediana por lote de 128 (ms)',ylabel='MASE con pérdidas',title=SHORT[e.RESOURCES[r]]);ax.margins(.2)
 fig.tight_layout();save('fig11_precision_coste')
 seeds=pd.read_csv(OUT/'contrastes_por_semilla.csv');fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=False)
 for r,ax in enumerate(axes):
  for j,label in enumerate(['E2','E4','E5','V2']):
   s=seeds[(seeds.resource==e.RESOURCES[r])&(seeds.reference=='E0')&(seeds.model==label)].sort_values('seed');ax.scatter(np.arange(3)*.12+j-.12,s.delta_MASE,s=35)
  ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(4),['E2','E4','E5','V2']);ax.set(title=SHORT[e.RESOURCES[r]],ylabel='Δ MASE por inicialización')
 fig.tight_layout();save('fig12_semillas')
 dynamics=pd.read_csv(OUT/'dinamica_entrenamiento_mediana.csv').set_index('resource');fig,ax=plt.subplots(figsize=(8,4))
 x=np.arange(3)
 for i,resource in enumerate(e.RESOURCES):ax.bar(x+(i-.5)*.3,[dynamics.loc[resource,f'acf_{k}'] for k in [1,12,48]],width=.3,label=SHORT[resource])
 ax.set_xticks(x,['30 min','6 horas','24 horas']);ax.set(xlabel='Retardo',ylabel='Autocorrelación mediana entre servicios');ax.legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,1.16));fig.tight_layout();save('fig13_dinamica')
 names=sorted(p.name for p in FIG.glob('fig*.png'));assert len(names)==13
 (OUT/'figuras_fuentes.json').write_text(json.dumps(dict(figures=names,primary_data_sha256=e.filehash(BASE/'results/reproduccion/data_verified.npz'),example_service=data.services[0],example_selection='first service and fixed chronological slices; no best-case selection'),indent=2))
 print('THIRTEEN_FIGURES_GENERATED')
if __name__=='__main__':main()

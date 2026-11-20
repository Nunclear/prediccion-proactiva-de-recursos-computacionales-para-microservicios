"""Audit every forecast, generate figures/tables and write only observed results."""
import os,sys,json,csv,hashlib,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'src/baseline/reference')]
os.environ['CHRONOGRAPH_OUTPUT']=str(ROOT/'results/reproduccion')
import run as engine
import numpy as np,pandas as pd,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from improved.models import DLinearEMAMask
ART=ROOT/'articulo';FIG=ART/'figuras';STATS=ROOT/'results/estadisticas';STATS.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True)
def dump(path,x):path.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def table(df):
 lines=['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']
 for _,row in df.iterrows():lines.append('| '+' | '.join(f'{v:.6g}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
 return '\n'.join(lines)
def main():
 assert (ROOT/'results/mejoras/COMPLETED.json').exists()
 data=engine.Data();selected=json.loads((ROOT/'results/mejoras/selected.json').read_text())['choices'];frames=[];checks=[]
 for experiment in ['reproduccion','mejoras']:
  folder=ROOT/'results'/experiment
  for path in sorted((folder/'test').glob('metrics_*.csv')):
   f=pd.read_csv(path);assert np.isfinite(f[['MAE','RMSE','MASE']]).all().all()
   for (name,resource,seed,rep),g in f.groupby(['model','resource','initialization','realization'],sort=False):
    r=engine.RESOURCES.index(resource);p=folder/'test'/f'{name}_r{r}_s{seed}_rep{rep}.npz';a=np.load(p)
    pred=a['prediction_normalized'];ix=a['indices'];truth=data.values[ix,:,r]
    assert np.array_equal(ix,data.indices['test']) and pred.shape==(7,1589,28)
    assert str(a['prediction_sha256'])==engine.digest(pred) and str(a['target_sha256'])==engine.digest(truth)
    err=pred.astype(np.float64)*data.scale[:,r]+data.mean[:,r]-truth[None]
    computed=dict(MAE=np.abs(err).mean(1),RMSE=np.sqrt(np.square(err).mean(1)));computed['MASE']=computed['MAE']/data.mase[:,r]
    for c,(q,pat) in enumerate(engine.CASES):
     rows=g[(g.q==q)&(g.pattern==pat)].set_index('service').loc[data.services]
     for metric,v in computed.items():np.testing.assert_allclose(rows[metric],v[c],rtol=1e-11,atol=1e-9)
    checks.append(dict(experiment=experiment,file=p.name,sha256=engine.filehash(p)))
   f['experiment']=experiment;f.loc[f.model=='DLinear','model']='E0';frames.append(f)
 metrics=pd.concat(frames,ignore_index=True);assert len(metrics)==243040
 metrics.to_csv(STATS/'metricas_completas.csv',index=False)
 rep=metrics.groupby(['model','resource','q','pattern','realization'],as_index=False)[['MAE','RMSE','MASE']].mean()
 summary=rep.groupby(['model','resource','q','pattern'],as_index=False)[['MAE','RMSE','MASE']].mean();summary.to_csv(STATS/'resumen.csv',index=False)
 seeds=metrics.groupby(['model','resource','initialization'],as_index=False).MASE.mean();seeds.to_csv(STATS/'variabilidad_semillas.csv',index=False)
 seed_summary=seeds[seeds.model.isin(['E0','E5'])].groupby(['model','resource'],as_index=False).agg(MASE_mean=('MASE','mean'),MASE_std_between_seeds=('MASE','std'),MASE_min=('MASE','min'),MASE_max=('MASE','max'))
 draws=np.random.default_rng(91001).integers(0,20,(10000,20));contrasts=[]
 for resource in engine.RESOURCES:
  for variant in ['E1','E2','E3','E4','E5']:
   for condition in ['todos','perdidas']:
    subset=rep[(rep.resource==resource)&((rep.q>0) if condition=='perdidas' else (rep.q>=0))]
    pair=subset.groupby(['model','realization']).MASE.mean().unstack('model').sort_index()
    diff=pair.E0-pair[variant];percent=100*diff/pair.E0;ci=np.quantile(diff.values[draws].mean(1),[.025,.975])
    contrasts.append(dict(resource=resource,variant=variant,conditions=condition,MASE_E0=pair.E0.mean(),MASE_variant=pair[variant].mean(),reduction_percent=percent.mean(),difference_E0_minus_variant=diff.mean(),IC95_low=ci[0],IC95_high=ci[1],interpretation='exploratory conditional bootstrap; no confirmatory p-value'))
 contrasts=pd.DataFrame(contrasts);contrasts.to_csv(STATS/'comparaciones_emparejadas.csv',index=False)
 # Prior run comparisons are audited as differences, not forced into agreement.
 old=pd.read_csv(ROOT/'fuentes/reproduccion/resultados/resumen_prueba.csv');old.loc[old.model=='DLinear','model']='E0'
 comparison=summary.merge(old[['model','resource','q','pattern','MASE']],on=['model','resource','q','pattern'],suffixes=('_fresh','_prior'))
 comparison['absolute_difference']=comparison.MASE_fresh-comparison.MASE_prior;comparison.to_csv(STATS/'reproduccion_vs_previo.csv',index=False)
 frozen=json.loads((ROOT/'results/reproduccion/frozen_models.json').read_text())['models'];prior=json.loads((ROOT/'fuentes/reproduccion/frozen_models.json').read_text())['models']
 matching=sum(next(x for x in prior if (x['model'],x['resource'],x['seed'])==(v['model'],v['resource'],v['seed']))['state_sha256']==v['state_sha256'] for v in frozen)
 state_rows=[]
 for v in frozen:
  old_state=next(x for x in prior if (x['model'],x['resource'],x['seed'])==(v['model'],v['resource'],v['seed']))
  state_rows.append(dict(model=v['model'],resource=v['resource'],seed=v['seed'],state_matches=v['state_sha256']==old_state['state_sha256'],fresh_validation_MASE=v['best_MASE'],prior_validation_MASE=old_state['best_MASE']))
 pd.DataFrame(state_rows).to_csv(STATS/'comparacion_estados_por_semilla.csv',index=False)
 # Reuse the augmented causal data adapter without invoking its training main.
 import importlib.util
 spec=importlib.util.spec_from_file_location('ablation_adapter',ROOT/'scripts/ablaciones.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 # Measured end-to-end forward latency, batch 128, one fixed initialization per model/resource.
 cost=[]
 for r in range(2):
  ds=engine.Prepared(data,r,'validation',80001);aug=mod.Prepared(data,r,'validation',80001);positions=torch.arange(128,device=engine.DEVICE);inputs=ds.batch(positions,positions%7)
  for label in sorted(summary.model.unique()):
   name='DLinear' if label=='E0' else label
   if label.startswith('E') and label!='E0':
    model=DLinearEMAMask(**selected[f'{label}_r{r}']['spec']).to(engine.DEVICE);folder=ROOT/'results/mejoras'
   elif name=='persistencia':continue
   else:model=engine.create(name,data.a,engine.DEVICE);folder=ROOT/'results/reproduccion'
   inputs=aug.batch(positions,positions%7) if label.startswith('E') and label!='E0' else ds.batch(positions,positions%7)
   ck=torch.load(folder/'checkpoints'/f'{name}_r{r}_s101.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['best_state']);model.eval()
   with torch.no_grad():
    for _ in range(5):model(*inputs[:3])
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated();times=[]
    for _ in range(20):
     begin=time.perf_counter();model(*inputs[:3]);torch.cuda.synchronize();times.append(time.perf_counter()-begin)
   cost.append(dict(model=label,resource=engine.RESOURCES[r],parameters=sum(p.numel() for p in model.parameters()),latency_batch128_ms=1000*np.median(times),peak_extra_allocated_bytes=torch.cuda.max_memory_allocated()-base,training_validation_seconds=sum(h['seconds'] for h in ck['history']),epochs=len(ck['history']),backend='eager forward, resident inputs, synchronized CUDA; seed101'))
   del model
 costs=pd.DataFrame(cost);costs.to_csv(STATS/'coste_computacional.csv',index=False)
 relative=[]
 for resource in engine.RESOURCES:
  a=costs[(costs.resource==resource)&(costs.model=='E0')].iloc[0];b=costs[(costs.resource==resource)&(costs.model=='E5')].iloc[0]
  relative.append(dict(resource=resource,parameter_ratio_E5_E0=b.parameters/a.parameters,latency_ratio_E5_E0=b.latency_batch128_ms/a.latency_batch128_ms,fit_time_ratio_E5_E0=b.training_validation_seconds/a.training_validation_seconds))
 relative=pd.DataFrame(relative);relative.to_csv(STATS/'coste_relativo.csv',index=False)
 def save(name):plt.savefig(FIG/name,dpi=240,bbox_inches='tight');plt.close()
 # All examples use the first cohort service and prespecified chronological slices.
 plt.figure(figsize=(11,5))
 for r in range(2):
  ax=plt.subplot(2,1,r+1);ax.plot((data.times-data.times[0])/3600,data.values[:,0,r],lw=.7);ax.set(ylabel='CPU (escala publicada)' if r==0 else 'Memoria (bytes)',xlabel='Horas desde inicio',title=data.services[0])
 plt.tight_layout();save('01_series_originales.png')
 own=ROOT/'results/mejoras';ck=torch.load(own/'checkpoints/E5_r0_s101.pt',map_location='cpu',weights_only=False);m=DLinearEMAMask(**selected['E5_r0']['spec']).to(engine.DEVICE);m.load_state_dict(ck['best_state']);m.eval()
 # Import data adapter without executing its training entrypoint.
 import importlib.util
 spec=importlib.util.spec_from_file_location('ablation_adapter',ROOT/'scripts/ablaciones.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 ds=mod.Prepared(data,0,'train',70001);inputs=ds.batch(torch.tensor([64],device=engine.DEVICE),torch.tensor([0],device=engine.DEVICE))
 with torch.no_grad():trend,residual=m.components(inputs[0])
 plt.figure(figsize=(9,4));plt.plot(inputs[0][0,:,0,0].cpu(),label='Historia normalizada');plt.plot(trend[0,:,0].cpu(),label='Tendencia EMA');plt.plot(residual[0,:,0].cpu(),label='Residual');plt.legend();plt.xlabel('Paso histórico (30 min)');save('02_descomposicion.png')
 for name,case in [('03_perdidas_dispersas.png',3),('04_perdidas_bloques.png',4)]:
  a,b=data.bounds['validation'];mask,_=engine.control.mask_for(data.times,a,b,20,'disperso' if case==3 else 'bloques',80001)
  ix=np.arange(a,a+120);v=data.values[ix,0,0];plt.figure(figsize=(10,3));plt.plot(ix,v,label='Original');plt.scatter(ix[~mask[ix,0]],v[~mask[ix,0]],color='red',label='Ocultado');plt.legend();plt.xlabel('Índice temporal');plt.ylabel('CPU (escala publicada)');save(name)
 a=np.load(own/'test/E5_r0_s101_rep0.npz');ix=a['indices'][:120];forecast=a['prediction_normalized'][2,:120,0]*data.scale[0,0]+data.mean[0,0]
 plt.figure(figsize=(10,3));plt.plot(ix,data.values[ix,0,0],label='Real');plt.plot(ix,forecast,label='E5 · 10 % bloques');plt.legend();plt.ylabel('CPU (escala publicada)');save('05_pronostico.png')
 average=summary.groupby(['model','resource'],as_index=False).MASE.mean()
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for r,ax in enumerate(axes):
  g=average[average.resource==engine.RESOURCES[r]];ax.bar(g.model,g.MASE);ax.set_yscale('log');ax.tick_params(axis='x',rotation=70);ax.set(title=engine.RESOURCES[r],ylabel='MASE (escala log)')
 fig.tight_layout();save('06_modelos.png')
 fig,axes=plt.subplots(2,2,figsize=(12,7))
 for r in range(2):
  for j,pattern in enumerate(['disperso','bloques']):
   ax=axes[r,j]
   for name in ['E0','E1','E2','E3','E4','E5']:
    g=summary[(summary.resource==engine.RESOURCES[r])&(summary.model==name)&((summary.pattern==pattern)|(summary.q==0))].sort_values('q');ax.plot(g.q,g.MASE,marker='o',label=name)
   ax.set(title=f'{engine.RESOURCES[r]} · {pattern}',xlabel='Pérdida (%)',ylabel='MASE');ax.legend(fontsize=8)
 fig.tight_layout();save('07_degradacion.png')
 plt.figure(figsize=(9,4));g=average[average.model.str.startswith('E')].pivot(index='model',columns='resource',values='MASE');g.plot.bar(ax=plt.gca());plt.ylabel('MASE');plt.tight_layout();save('08_recursos.png')
 plt.figure(figsize=(9,4));c=contrasts[contrasts.conditions=='perdidas'];
 for r,resource in enumerate(engine.RESOURCES):
  q=c[c.resource==resource];plt.errorbar(np.arange(5)+r*.12,q.difference_E0_minus_variant,yerr=[q.difference_E0_minus_variant-q.IC95_low,q.IC95_high-q.difference_E0_minus_variant],marker='o',ls='',label=resource)
 plt.xticks(np.arange(5),['E1','E2','E3','E4','E5']);plt.axhline(0,color='black',lw=.7);plt.ylabel('MASE E0 menos variante · IC descriptivo');plt.legend();save('09_ablaciones.png')
 usage=pd.read_csv(ROOT/'results/monitoreo/resources.csv').sort_values('unix_seconds');fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True)
 minutes=(usage.unix_seconds-usage.unix_seconds.iloc[0])/60
 # Leave gaps visible across shutdowns and monitor restarts.
 for col,ax in zip(['gpu_utilization_percent','gpu_temperature_C','ram_available_bytes'],axes):
  values=usage[col].copy();values.loc[usage.unix_seconds.diff()>30]=np.nan
  ax.plot(minutes,values);ax.set_ylabel(col)
 axes[-1].set_xlabel('Minutos desde inicio (incluye interrupciones)');fig.tight_layout();save('10_utilizacion.png')
 # Record selected values, not only the search space.
 chosen_rows=[]
 for r in range(2):
  for label in ['E1','E2','E3','E4','E5']:
   choice=selected[f'{label}_r{r}'];specification=choice['spec'];alphas=[]
   if label in ['E2','E5']:
    for seed in engine.SEEDS:
     state=torch.load(own/'checkpoints'/f'{label}_r{r}_s{seed}.pt',map_location='cpu',weights_only=False)['best_state']
     alphas.append(float(torch.sigmoid(state['alpha_logit'])) if specification.get('learned_alpha',False) else specification.get('alpha',.3))
   chosen_rows.append(dict(model=label,resource=engine.RESOURCES[r],learning_rate=choice['lr'],window=specification.get('kernel',25) if label not in ['E2','E5'] else 'EMA',alpha_mode='learned' if specification.get('learned_alpha',False) else 'validation_fixed' if alphas else 'not_applicable',alpha_mean=float(np.mean(alphas)) if alphas else 'not_applicable',age=specification.get('age','log') if label in ['E4','E5'] else 'not_applicable'))
 chosen_table=pd.DataFrame(chosen_rows);chosen_table.to_csv(STATS/'hiperparametros_seleccionados.csv',index=False)
 # All prose below is generated after verified completed runs.
 result=c[c.variant=='E5'];vals={row.resource:row for row in result.itertuples()}
 cpu=vals[engine.RESOURCES[0]];mem=vals[engine.RESOURCES[1]]
 refs=json.loads((ROOT/'docs/referencias_verificadas.json').read_text());references='\n\n'.join(f"{v.get('apa',v['authors']+'. '+v['title'])} [{v['url']}]({v['url']})." for v in refs)
 methodology=r'''Estudio exploratorio sobre 28 servicios seleccionados por conveniencia de ChronoGraph, 8005 observaciones y 28 aristas internas. Se excluyen 137 aristas de frontera. CPU conserva su escala publicada y memoria se mide en bytes. Historia de doce pasos y horizonte de 1800 segundos. Cortes [0,5603,6404,8005], con 5591/777/1589 ventanas por partición; ninguna atraviesa cortes ni la discontinuidad de 30 horas. Estadísticas de normalización y denominadores MASE se calculan solo en entrenamiento. Se ocultan exclusivamente entradas; los objetivos originales se conservan.

E0 conserva DLinear con promedio móvil 25. E1 selecciona ventana entre 3/7/13/25. E2 usa EMA con alpha 0,1/0,3/0,7 o un escalar aprendido. E3 añade tres indicadores: observado, arrastrado y respaldo por mediana. E4 incorpora antigüedad logarítmica o lineal acotada. E5 combina EMA y ambas cabezas de información faltante. Alpha es compartido entre servicios y ajustado por recurso; no se afirma que varíe libremente en cada instante. Los candidatos aprendidos y fijos compiten sobre validación. Los presupuestos de búsqueda difieren por variante y se conservan íntegros en selected.json; esta diferencia limita la atribución exclusiva a la arquitectura.

La tendencia causal obedece $l_t=l_{t-1}+\alpha m_t(x_t-l_{t-1})$, inicializada con el valor histórico preparado causalmente. El residual es $r_t=x_t-l_t$. La predicción completa es $\hat y=W_l l+W_r r+b+W_m[m,c,f]+W_a a$. $m$ indica observación, $c$ arrastre y $f$ mediana inicial; sus canales distinguen procedencia, no revelan el dato ocultado. La antigüedad cuenta pasos desde la última observación y se reinicia en discontinuidades. El promedio móvil centrado de E0 usa únicamente puntos de su ventana histórica; EMA es además causal dentro de esa ventana. El residual no se interpreta automáticamente como estacionalidad identificada: seis horas de contexto no acreditan un ciclo diario.

Cada candidato se evalúa con tres épocas, 512 orígenes y 32 de validación por condición; dos tasas 0,001/0,003. Los ajustes finales usan semillas 101/202/303, Adam, recorte de norma 1, batch16, máximo20 épocas y paciencia5. Se mantiene una condición por origen y época en rotación común. E0 y los comparadores provienen de la reproducción nueva, no de pesos preentrenados. La prueba tiene veinte realizaciones completas de máscaras, semillas comunes, porcentajes 0/10/20/40 y patrones dispersos o bloques hasta seis pasos. Los objetivos y cantidades efectivas se mantienen comparables.

MAE y RMSE se calculan por servicio en escala original; MASE divide MAE por el error ingenuo medio de entrenamiento usando pares separados exactamente 1800 segundos. Se promedian semillas y servicios antes de remuestrear las veinte realizaciones completas. Los intervalos percentiles emplean 10000 bootstrap, semilla91001. No se supone independencia entre ventanas superpuestas o servicios conectados. Los contrastes son descriptivos; no se declaran decisiones confirmatorias ni p-valores. La reducción porcentual se calcula de forma emparejada frente a E0. La latencia usa veinte repeticiones sincronizadas, batch128, entradas residentes, semilla101 y backend eager para todos los modelos; los tiempos de ajuste incluyen entrenamiento y validación y no equivalen a tiempo puro de kernel.'''
 limitations='''La prueba ya era conocida antes de diseñar las variantes: todo el nuevo estudio es exploratorio. Además, sus primeras 400 observaciones por servicio pertenecían a validaciones históricas. No se obtuvo validación confirmatoria temporal posterior o externa en esta ejecución; no se afirma que sea imposible obtenerla. La población corresponde a un solo dataset y subgrafo por conveniencia. Las máscaras artificiales independientes entre servicios no reproducen todas las caídas reales de colectores. Bootstrap mide variación condicional al generador y a modelos entrenados, no incertidumbre entre infraestructuras ni toda la variación del entrenamiento. E5 dispone de mayor presupuesto de selección que algunas ablaciones. Disponibilidad y antigüedad no son factores ortogonales: antigüedad positiva también informa ausencia. La EMA ya utiliza máscara para omitir actualizaciones aunque no tenga una cabeza de máscara explícita. No se demuestra novedad universal, causalidad del grafo, ahorro operativo, SLA ni calidad de GinAR en los dominios y configuraciones de su artículo. Los errores extremos se conservan. La potencia del dispositivo presentó valores anómalos y límites ausentes: no se declara consumo energético total fiable.'''
 findings=f'''En los seis escenarios con pérdidas, E5 obtuvo MASE {cpu.MASE_variant:.6f} en CPU frente a {cpu.MASE_E0:.6f} de E0; reducción porcentual emparejada media {cpu.reduction_percent:.3f} %. En memoria obtuvo {mem.MASE_variant:.6f} frente a {mem.MASE_E0:.6f}; reducción {mem.reduction_percent:.3f} %. Una reducción negativa significa empeoramiento. Los resultados por porcentaje, patrón, servicio, inicialización y realización se conservan en los CSV; no se elige una condición aislada como prueba de superioridad.'''
 reproduction=f'''El contexto histórico utiliza 70/15/15 y pilotos reducidos en CPU; sus presupuestos completos eran estimaciones, no experimentos confirmatorios ejecutados. El paquete posterior registra 70/10/20, CUDA, treinta ajustes y veinte realizaciones de prueba. Esta ejecución reproduce ese último protocolo. No se mezclan las métricas de los pilotos con las de la reproducción.\n\nSe verificaron los dos manifiestos principales y los 640 archivos previos de predicciones cuando estaban disponibles localmente. La reproducción se entrenó desde cero en una carpeta vacía; los checkpoints originales solo se leyeron para auditoría. Coinciden exactamente {matching} de 30 hashes de estados finales. Las diferencias restantes se informan sin corregir artificialmente la línea base; comparacion_estados_por_semilla.csv identifica cada coincidencia y discrepancia. No se atribuyen causalmente las diferencias a una única característica del sistema, porque esa causa no se aisló experimentalmente. La máxima diferencia absoluta de MASE agregado respecto del registro anterior fue {comparison.absolute_difference.abs().max():.9g}. Las cifras completas están en reproduccion_vs_previo.csv. La normalización, máscaras y protocolo se mantuvieron; la adaptación del runner se limita a parametrizar la ruta de salida. El hardware sigue siendo RTX3050Laptop de4GB con PyTorch2.9.0+cu128, driver550.163.01; su operación CUDA fue comprobada. El objetivo no fue reproducir las tablas originales de los artículos, sino el experimento local registrado.'''
 abstract=f'''Las pérdidas de telemetría histórica pueden modificar el pronóstico de recursos de microservicios y dificultar la distinción entre mediciones e imputaciones. Este estudio evalúa una extensión ligera de DLinear que combina descomposición exponencial, disponibilidad y antigüedad de las observaciones. Se auditó el material del Grupo 05 y se entrenaron nuevamente seis predictores de referencia, sin inicialización mediante checkpoints anteriores. Se analizaron veintiocho servicios de ChronoGraph con ocho mil cinco observaciones por servicio, CPU y memoria, historia de doce puntos y horizonte de treinta minutos. La partición temporal fue setenta, diez y veinte por ciento; se excluyeron ventanas que atraviesan cortes o discontinuidades. Se ejecutaron seis configuraciones de ablación, tres inicializaciones y veinte realizaciones completas de pérdidas dispersas o por bloques. Los hiperparámetros se seleccionaron exclusivamente con validación. El modelo completo obtuvo MASE {cpu.MASE_variant:.6f} para CPU y {mem.MASE_variant:.6f} para memoria, con reducciones emparejadas de {cpu.reduction_percent:.3f} y {mem.reduction_percent:.3f} por ciento respecto de DLinear en condiciones con pérdidas. Las métricas se recalcularon desde predicciones archivadas y se estimaron intervalos descriptivos mediante remuestreo de realizaciones completas. Se midieron parámetros, latencia y recursos computacionales. Los resultados se interpretan por recurso y componente, sin presuponer superioridad uniforme. Puesto que la prueba era conocida durante el desarrollo y presenta exposición histórica parcial, los hallazgos son exploratorios y requieren evaluación independiente antes de cualquier afirmación confirmatoria.'''
 count=len(abstract.split());assert 200<=count<=250,count
 title='Descomposición exponencial y reconocimiento de telemetría incompleta en DLinear: estudio exploratorio del pronóstico de recursos de microservicios'
 article=f'''# {title}\n\n## Resumen\n\n{abstract}\n\n**Palabras clave:** pronóstico; microservicios; DLinear; suavizamiento exponencial; datos faltantes; ablación.\n\n## 1. Introducción\n\nEl pronóstico busca proyectar el comportamiento futuro usando información histórica disponible. En microservicios, el consumo de CPU y memoria cambia en el tiempo y la falta de mediciones reduce la información accesible. Este trabajo continúa la investigación del Grupo 05 y pregunta si la descomposición exponencial y la representación explícita de procedencia y antigüedad modifican el error de DLinear. Se estudia una extensión verificable, sin atribuirle novedad universal ni dar por supuesta una mejora. Tendencia, residual y estacionalidad son conceptos distintos; no se identifica automáticamente el residual como un ciclo estacional.\n\n## 2. Estado del arte\n\nZeng et al. (2023) proponen modelos lineales con descomposición como referentes de pronóstico de series. Hyndman y Athanasopoulos explican el suavizamiento exponencial y su ponderación decreciente del pasado. Aquí se adapta ese mecanismo a actualizaciones omitidas cuando no existe medición. Che et al. (2018) distinguen disponibilidad y tiempo transcurrido en GRU-D; este antecedente impide presentar las máscaras o la antigüedad como ideas inéditas. Cho et al. (2014) fundamentan la arquitectura GRU recurrente. BiTGraph incorpora patrones faltantes en operaciones temporales y gráficas, mientras GinAR estudia variables con historial ausente; sus problemas experimentales no son equivalentes a nuestras pérdidas parciales. ChronoGraph aporta mediciones y dependencias de microservicios. MASE permite expresar errores relativos a una referencia ingenua de entrenamiento (Hyndman y Koehler, 2006). Las referencias enlazadas identifican fuentes primarias verificadas, no una revisión sistemática exhaustiva.\n\n## 3. Metodología\n\n{methodology}\n\n### Configuraciones seleccionadas en validación\n\n{table(chosen_table)}\n\n## 4. Resultados\n\n### 4.1 Reproducción\n\n{reproduction}\n\n### 4.2 Comparación y ablaciones\n\n{findings}\n\n{table(contrasts[contrasts.conditions=='perdidas'][['resource','variant','MASE_E0','MASE_variant','reduction_percent','IC95_low','IC95_high']])}\n\n### Variación entre inicializaciones\n\n{table(seed_summary)}\n\nLa desviación resume tres inicializaciones tras promediar condiciones y realizaciones; no se trata como veinte réplicas de infraestructura.\n\n### 4.3 Coste computacional observado\n\n{table(costs[['model','resource','parameters','latency_batch128_ms','training_validation_seconds','epochs']])}\n\nRazones de coste E5/E0; valores superiores a uno indican mayor coste:\n\n{table(relative)}\n\n'''
 means=rep.groupby(['model','resource'],as_index=False)[['MASE','MAE','RMSE']].mean()
 by_condition=summary[summary.model.isin(['E0','E5'])].pivot(index=['resource','q','pattern'],columns='model',values='MASE').reset_index()
 by_condition['reduction_percent']=100*(by_condition.E0-by_condition.E5)/by_condition.E0
 article+='\n### 4.4 Todos los predictores reproducidos y variantes\n\nPromedios con el mismo peso para cada condición, servicio e inicialización disponible. MAE y RMSE conservan la escala de CPU o bytes de memoria; no se comparan sus magnitudes entre recursos. La persistencia es determinista y tiene una sola inicialización.\n\n'+table(means)+'\n\n### 4.5 Pérdida y patrón\n\nMASE medio por condición; la reducción de esta tabla es la razón entre promedios, mientras el contraste anterior promedia reducciones emparejadas por realización.\n\n'+table(by_condition)+'\n'
 captions=['Series reales del primer servicio de la cohorte.','Tendencia EMA y residual en una ventana histórica fija.','Pérdidas dispersas inducidas con semilla fijada.','Pérdidas por bloques sobre los mismos datos originales.','Predicción y objetivo real en un tramo de prueba fijo.','Comparación global de modelos; escala logarítmica para conservar errores extremos.','Curvas de MASE por porcentaje y patrón.','Comparación normalizada de CPU y memoria en las ablaciones.','Diferencias emparejadas frente a E0 e intervalos descriptivos.','Telemetría de utilización registrada durante la ejecución.']
 for file,caption in zip(sorted(FIG.glob('*.png')),captions):article+=f'\n![{caption}](figuras/{file.name})\n\n{caption}\n'
 discussion='''Las diferencias observadas no identifican automáticamente el efecto causal de cada componente. Una EMA que omite imputaciones puede conservar una tendencia previa durante un bloque, pero también responder con retraso a un cambio real; una máscara puede permitir corregir valores reconstruidos o añadir parámetros sin ventaja. La antigüedad puede representar menor vigencia del historial, aunque su beneficio depende del recurso y del generador. Estas son interpretaciones mecanísticas plausibles y no mecanismos demostrados por los resultados. La tabla completa de ablaciones y las semillas deben leerse conjuntamente; un promedio favorable no implica que todos los servicios o condiciones mejoren. Los presupuestos de búsqueda y capacidades difieren. La observación se circunscribe al experimento local y no contradice por sí sola las publicaciones de los comparadores.'''
 article+=f'\n## 5. Discusión\n\n{discussion}\n\n### Limitaciones\n\n{limitations}\n\n## 6. Conclusiones\n\n{findings} La contribución empírica es la comparación reproducible de descomposición y reconocimiento de ausencia bajo condiciones comunes. No se declara confirmación independiente. El siguiente paso científico es congelar la selección y evaluarla en datos posteriores o externos compatibles.\n\n## Referencias\n\n{references}\n'
 (ART/'ARTICULO_EMPIRICO.md').write_text(article)
 for name,content in [('RESUMEN_EJECUTIVO',findings+'\n\n'+limitations),('METODOLOGIA',methodology),('RESULTADOS',findings+'\n\n'+table(contrasts)),('DISCUSION',discussion+'\n\n'+limitations),('REFERENCIAS',references)]:
  (ART/f'{name}.md').write_text(f'# {name.replace("_"," ")}\n\n[Artículo principal](ARTICULO_EMPIRICO.md)\n\n{content}\n')
 docs={'REPRODUCCION_ORIGINAL':reproduction,'ARQUITECTURA_MODELO':methodology,'PROTOCOLO_EXPERIMENTAL':methodology+'\n\n'+limitations,'CONFIGURACION_HARDWARE':'CPU i5-11300H,4núcleos/8hilos; RAM24GB; GPU RTX3050Laptop4GB. Driver550.163.01, PyTorch2.9.0+cu128. Operación real CUDA probada. Se conservaron otros contenedores. CPU PyTorch2hilos; entradas precargadas en GPU, sin DataLoader: workers0 evita competencia y copias innecesarias. Potencia no fiable, energía no informada. Detalles del entorno en docker/requirements.lock y monitor CSV.','OPTIMIZACIONES':'Se conservó el backend matemático de la referencia, incluidos grafos CUDA validados de GinAR y recorte estable de gradientes. Las nuevas variantes usan operaciones float32 y batch16; inferencia128, por comparabilidad. No se usó precisión mixta ni se alteraron pasos para inflar ocupación. El monitor pausa temporalmente solo el proceso del estudio ante temperatura85C o presión de RAM/swap y lo reanuda al recuperarse. No se modificaron drivers ni parámetros permanentes.','VALIDACION_ESTADISTICA':'Veinte realizaciones completas emparejadas; semillas y servicios se promedian antes del bootstrap10000. No se remuestrean ventanas como independientes. No se declararon pruebas confirmatorias ni significancia; por tanto no se usa Holm para fabricar una confirmación. Reducciones porcentuales y variabilidad se conservan por recurso.','LIMITACIONES':limitations,'GUIA_EJECUCION':'Ver ../docker/README.md. Primero auditar_archivos.py, luego reproducir.py en el contenedor dedicado. esperar_y_ejecutar.sh encadena reproducción, pruebas, ablaciones y redacción. Solo reutilizar checkpoints de esta carpeta para reanudación, nunca los originales para afirmar entrenamiento desde cero. Las salidas se parametrizan con CHRONOGRAPH_OUTPUT; en ablaciones se fija su subcarpeta del proyecto para separar experimentos.','BITACORA':'Auditoría de ZIP y SHA; diagnóstico de hardware; contenedor dedicado; auditoría LFS; entrenamiento independiente de seis predictores; pruebas; selección por validación y entrenamiento de ablaciones; prueba; recomputación; artículo. Las marcas temporales observadas están en progress.jsonl y resources.csv. Ninguna estimación histórica se presentó como resultado observado.'}
 for name,content in docs.items():(ROOT/'docs'/f'{name}.md').write_text(f'# {name.replace("_"," ")}\n\n[Artículo](../articulo/ARTICULO_EMPIRICO.md) · [Guía](GUIA_EJECUCION.md)\n\n{content}\n')
 recovery=ROOT/'docs/SELECCION_ANTES_APAGADO.json'
 if recovery.exists():assert json.loads(recovery.read_text())==json.loads((own/'selected.json').read_text()),'Selection changed across restart'
 recovered=ROOT/'docs/RECUPERACION_APAGADO_20261029.json';preserved=0
 if recovered.exists():
  for checkpoint in json.loads(recovered.read_text())['checkpoints']:
   if checkpoint['complete']:
    assert engine.filehash(ROOT/checkpoint['file'])==checkpoint['sha256'],checkpoint['file']
    preserved+=1
  prior_protocol=json.loads((ROOT/'docs/PROTOCOLO_ANTES_APAGADO.json').read_text())
  for name in ['scripts/ablaciones.py','src/improved/models.py']:
   assert engine.filehash(ROOT/name)==prior_protocol['code_sha256'][name],name
  with (ROOT/'docs/BITACORA.md').open('a') as f:f.write('\n\nSe reanudó después de un apagado. [Auditoría de recuperación](RECUPERACION_APAGADO.md). Se conservaron exactamente '+str(preserved)+' checkpoints completos y la selección de validación; la arquitectura y el entrenamiento conservaron sus fuentes.\n')
 dump(STATS/'validacion.json',dict(status='passed',prediction_files=len(checks),metric_rows=len(metrics),rtol=1e-11,atol=1e-9,abstract_words=count,matching_original_states=matching,forecasts=checks,selection_unchanged_after_recovery=recovery.exists(),complete_checkpoints_preserved_after_shutdown=preserved,report_generator_sha256=engine.filehash(Path(__file__))))
 (ROOT/'docs/ESTADO_EN_CURSO.md').write_text('# Estado final\n\nReproducción, ablaciones y validación completadas. Consultar [el artículo](../articulo/ARTICULO_EMPIRICO.md) y results/estadisticas/validacion.json para los recuentos observados.\n')
 (ROOT/'README.md').write_text('# Investigación DLinear-EMA-Mask\n\nEstudio exploratorio del Grupo 05, con división temporal 70/10/20 y ejecución CUDA.\n\n- [Artículo empírico](articulo/ARTICULO_EMPIRICO.md)\n- [Resumen ejecutivo](articulo/RESUMEN_EJECUTIVO.md)\n- [Validación de las predicciones](results/estadisticas/validacion.json)\n- [Resultados por modelo, recurso y condición](results/estadisticas/resumen.csv)\n- [Recuperación tras el apagado](docs/RECUPERACION_APAGADO.md)\n- [Ejecución Docker](docker/README.md)\n\nLos ZIP originales permanecen en archivos_originales/. Los modelos nuevos se entrenaron desde cero; los checkpoints anteriores se usaron para auditoría. La prueba ya era conocida, por lo que estos resultados requieren confirmación independiente. El paquete conserva checkpoints, predicciones individuales, métricas y registros de utilización. MANIFEST_SHA256.json verifica la integridad de la entrega.\n')
 (ROOT/'results/COMPLETED.json').write_text(json.dumps(dict(status='complete',article=str(ART/'ARTICULO_EMPIRICO.md'),validated_prediction_files=len(checks),metric_rows=len(metrics))))
if __name__=='__main__':main()

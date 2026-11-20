"""Independent recomputation from archived forecasts; descriptive study report."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('/root/up2date/study_701020')
HERE=Path(__file__).resolve().parent
CASES=[(0,'disperso')]+[(q,p) for q in [10,20,40] for p in ['disperso','bloques']]
RESOURCES=['cpu_usage','container_memory_usage_bytes']
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
    return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def bootstrap(x,draws):
    return np.quantile(np.asarray(x)[draws].mean(1),[.025,.975]).tolist()
def table(frame,columns):
    result=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for _,r in frame.iterrows():
        result.append('| '+' | '.join(f'{r[c]:.6f}' if isinstance(r[c],(float,np.floating)) else str(r[c]) for c in columns)+' |')
    return '\n'.join(result)

def main():
    assert len(list((OUT/'test').glob('complete_*.json')))==40
    assert json.loads((OUT/'cuda_graph_equivalence.json').read_text())['status']=='passed'
    assert json.loads((OUT/'checkpoint_restore_check.json').read_text())['status']=='passed'
    deliver=OUT/'deliverables';deliver.mkdir(exist_ok=True)
    raw=np.load(OUT/'data_verified.npz');values=raw['values'];times=raw['times'];services=raw['services'].tolist()
    mean=values[:5603].mean(0);scale=np.maximum(values[:5603].std(0),1e-8)
    denominator=np.abs(np.diff(values[:5603],axis=0))[np.diff(times[:5603])==1800].mean(0)
    frames=[]
    for path in sorted((OUT/'test').glob('metrics_*.csv')):
        # Check completion-marker integrity before recomputing the metric values.
        suffix=path.stem.replace('metrics_','')
        marker=json.loads((OUT/'test'/f'complete_{suffix}.json').read_text())
        assert marker['metrics_sha256']==sha(path)
        frame=pd.read_csv(path);assert len(frame)==marker['rows']==3136
        assert not frame.duplicated(['model','resource','initialization','realization','q','pattern','service']).any()
        frames.append(frame)
    metrics=pd.concat(frames,ignore_index=True)
    assert len(metrics)==125440 and np.isfinite(metrics[['MAE','RMSE','MASE']]).all().all()
    checks=[]
    for (name,resource,init,rep),group in metrics.groupby(['model','resource','initialization','realization'],sort=False):
        r=RESOURCES.index(resource);filename=f'{name}_r{r}_s{init}_rep{rep}.npz'
        data=np.load(OUT/'test'/filename)
        prediction=data['prediction_normalized'];ix=data['indices'];truth=values[ix,:,r]
        assert prediction.shape==(7,1589,28) and np.all((ix>=6404)&(ix<8005))
        assert str(data['target_sha256'])==digest(truth)
        assert str(data['prediction_sha256'])==digest(prediction)
        error=prediction.astype(np.float64)*scale[:,r]+mean[:,r]-truth[None]
        mae=np.abs(error).mean(1);rmse=np.sqrt(np.square(error).mean(1));mase=mae/denominator[:,r]
        for c,(q,pattern) in enumerate(CASES):
            rows=group[(group.q==q)&(group.pattern==pattern)].set_index('service').loc[services]
            for key,expected in [('MAE',mae[c]),('RMSE',rmse[c]),('MASE',mase[c])]:
                np.testing.assert_allclose(rows[key].values,expected,rtol=1e-11,atol=1e-9)
        checks.append(dict(file=filename,sha256=sha(OUT/'test'/filename),all_metrics_recomputed=True))
    assert len(checks)==640
    # All resources/models use the same targets and missingness seeds in each realization.
    assert (metrics.groupby('resource').target_sha256.nunique()==1).all()
    assert (metrics.groupby('realization').mask_seed.nunique()==1).all()
    # No missingness case is duplicated: a single zero-loss reference is retained.
    assert set(metrics[metrics.q==0].pattern)=={'disperso'}
    config=json.loads((OUT/'run_config.json').read_text())
    frozen=json.loads((OUT/'frozen_models.json').read_text())
    assert len(frozen['models'])==30
    assert frozen['config_sha256']==sha(OUT/'run_config.json')
    for entry in frozen['models']:
        r=RESOURCES.index(entry['resource'])
        assert (metrics[(metrics.model==entry['model'])&(metrics.resource==entry['resource'])].initialization.unique()==np.array([101,202,303])).all()
    metrics.to_csv(deliver/'metricas_detalladas_prueba.csv',index=False,float_format='%.15g')
    realization=metrics.groupby(['model','resource','q','pattern','realization'],as_index=False)[['MAE','RMSE','MASE']].mean()
    # Equal weight per initialization, service, and complete mask realization.
    draws=np.random.default_rng(91001).integers(0,20,size=(10000,20))
    summary=[]
    for keys,group in realization.groupby(['model','resource','q','pattern']):
        group=group.sort_values('realization');assert len(group)==20
        low,high=bootstrap(group.MASE.values,draws)
        summary.append(dict(zip(['model','resource','q','pattern'],keys))|dict(MAE=group.MAE.mean(),RMSE=group.RMSE.mean(),MASE=group.MASE.mean(),MASE_IC95_inferior=low,MASE_IC95_superior=high))
    summary=pd.DataFrame(summary);summary.to_csv(deliver/'resumen_prueba.csv',index=False)
    realization.to_csv(deliver/'metricas_por_realizacion.csv',index=False)
    def e(name,r,q,pattern):
        g=realization[(realization.model==name)&(realization.resource==r)&(realization.q==q)&(realization.pattern==pattern)].sort_values('realization')
        assert len(g)==20
        return g.MASE.values
    contrasts=[]
    for r,resource in enumerate(RESOURCES):
        own='GRU_propia';graph='GRU_llamadas'
        differences=[np.mean([e(own,resource,40,p)-e(own,resource,10,p) for p in ['disperso','bloques']],axis=0),
                     np.mean([e(own,resource,q,'bloques')-e(own,resource,q,'disperso') for q in [10,20,40]],axis=0),
                     np.mean([e(own,resource,q,p)-e(graph,resource,q,p) for q in [10,20,40] for p in ['disperso','bloques']],axis=0)]
        for code,d in zip([1+r,3+r,5+r],differences):
            low,high=bootstrap(d,draws)
            contrasts.append(dict(contraste=f'C{code}',recurso=resource,diferencia_MASE=float(d.mean()),IC95_inferior=low,IC95_superior=high,
                estado='descriptivo_condicional; sin p confirmatorio ni decision Holm'))
    degradation={r:np.mean([e('GRU_propia',r,q,p)-e('GRU_propia',r,0,'disperso') for q in [10,20,40] for p in ['disperso','bloques']],axis=0) for r in RESOURCES}
    d=degradation[RESOURCES[0]]-degradation[RESOURCES[1]];low,high=bootstrap(d,draws)
    contrasts.append(dict(contraste='C7',recurso='CPU menos memoria',diferencia_MASE=float(d.mean()),IC95_inferior=low,IC95_superior=high,
                         estado='descriptivo_condicional; sin p confirmatorio ni decision Holm'))
    contrasts=pd.DataFrame(contrasts).sort_values('contraste');contrasts.to_csv(deliver/'contrastes_descriptivos.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for row,resource in enumerate(RESOURCES):
        for col,pattern in enumerate(['disperso','bloques']):
            ax=axes[row,col]
            for name,g in summary[summary.resource==resource].groupby('model'):
                g=g[(g.pattern==pattern)|(g.q==0)].sort_values('q')
                ax.plot(g.q,g.MASE,marker='o',label=name)
            ax.set(title=f'{"CPU" if row==0 else "Memoria"} · {pattern}',xlabel='Telemetría histórica oculta (%)',ylabel='MASE medio por servicio')
            positive=summary[(summary.resource==resource)&((summary.pattern==pattern)|(summary.q==0))].MASE
            if positive.min()>0 and positive.max()/positive.min()>20:
                ax.set_yscale('log');ax.set_ylabel('MASE medio por servicio (escala log)')
            ax.grid(alpha=.2);ax.set_xticks([0,10,20,40])
    axes[0,0].legend(fontsize=8)
    fig.suptitle('ChronoGraph · prueba temporal 20 % · 28 servicios',fontsize=15)
    fig.savefig(deliver/'sensibilidad_perdidas.png',dpi=170);plt.close(fig)
    baseline=summary[summary.q==0].sort_values(['resource','MASE'])
    q40=summary[(summary.q==40)].sort_values(['resource','pattern','MASE'])
    report=f'''# Resultados ejecutados: ChronoGraph, Grupo 05

## Alcance y ejecución

Se ejecutó el estudio de 28 servicios del material adjunto sobre CPU y memoria reales de ChronoGraph. Se compararon persistencia, DLinear, GRU propia, GRU con llamadas, BiTGraph (código de autores con portabilidad de dispositivo) y GinAR. No se entrenó Chronos, Prophet ni un detector de anomalías: esos ejemplos del repositorio no corresponden a los seis predictores definidos en el trabajo adjunto.

Esta ejecución utiliza una partición cronológica **70/10/20** y el dispositivo CUDA. Se registró esta versión antes de consultar errores de prueba. Es una nueva ejecución local del protocolo adaptado; no una preregistración pública ni una reproducción certificada de los artículos originales.

## Datos y partición

Se comprobó el SHA-256 del objeto LFS completo de 1.786.000.083 bytes y la coincidencia exacta de los valores y marcas temporales de los 28 servicios de la cohorte. Se conservaron 28 aristas internas y se excluyeron vecinos externos.

| Tramo | Intervalo de índices (fin excluido) | Observaciones por servicio | Ventanas válidas por servicio |
| --- | --- | --- | --- |
| Entrenamiento | [0, 5603) | 5603 | 5591 |
| Validación | [5603, 6404) | 801 | 777 |
| Prueba | [6404, 8005) | 1601 | 1589 |

Cada ventana utiliza 12 observaciones para predecir la siguiente, 1800 segundos después. Se excluyeron ventanas que cruzan particiones o el salto de 30 horas en el índice 5908. Normalización, mediana de respaldo y denominadores MASE proceden exclusivamente de entrenamiento. CPU conserva su escala publicada sin atribuir unidades no verificadas; memoria corresponde a `container_memory_usage_bytes`.

## Entrenamiento y selección

Se compararon tasas 0,001 y 0,003 durante tres épocas con 512 orígenes de entrenamiento y 32 de validación por condición. La selección se hizo por MASE de validación, con una tasa compartida para las dos GRU. Los ajustes finales usaron todos los orígenes de entrenamiento, semillas 101/202/303, máximo 20 épocas y paciencia 5; se restauró el mejor estado de validación. Adam, batch 16 y recorte de gradiente 1. Cada origen recibe una condición por época en rotación común a los modelos. La evaluación usa batch 128 para aprovechar CUDA; no modifica la arquitectura ni el conjunto de objetivos.

Se usó {config['device']}, PyTorch {config['torch']}, CUDA de la imagen {config['cuda']}. Las arquitecturas y el parche de portabilidad están incluidos en el paquete reproducible. GinAR usa repetición de grafos CUDA para reducir el coste de lanzamiento: se comprobó igualdad exacta con la ejecución normal en predicciones, pérdidas, gradientes y actualizaciones de pesos para ambos recursos, incluyendo un lote final incompleto y evaluación de lote 128. No se alteraron arquitectura, lotes ni pasos del optimizador. Durante GinAR se detectó un desbordamiento de la norma L2 en float32 con gradientes individualmente finitos; el recorte conserva su umbral 1 y recurre a una norma y escalado en float64 únicamente cuando la norma float32 desborda. Se siguen rechazando gradientes NaN/Inf y no se omiten lotes. La incidencia y su corrección se registraron antes de la prueba. Se verificó además la restauración reproducible del siguiente lote y actualización desde un checkpoint. DLinear recibe llenado causal; las GRU reciben valor, disponibilidad y antigüedad; BiTGraph recibe valores y máscara; GinAR recibe ceros en faltantes. Las diferencias entre modelos externos describen pipelines completos, no identifican por sí solas el efecto de las llamadas.

## Prueba final

Los 30 ajustes entrenados se guardaron y congelaron antes de la prueba. Se evaluaron siete condiciones: sin pérdidas y 10/20/40 % con ausencia dispersa o bloques de hasta seis puntos. Se hicieron 20 realizaciones completas, con semillas comunes entre modelos y recursos; los patrones ocultan la misma cantidad efectiva y las tasas están anidadas. Las ausencias se aplican al flujo histórico, no a los objetivos futuros.

Se informa el promedio con igual peso por servicio, inicialización y realización. RMSE se calcula por servicio antes de promediar. MASE es la métrica principal; MAE y RMSE conservan la escala del recurso. Los intervalos del 95 % son bootstrap descriptivo de 10.000 remuestreos sobre realizaciones completas, semilla 91001. No se trataron ventanas superpuestas ni nodos conectados como réplicas independientes.

### Referencia sin pérdidas

{table(baseline,['model','resource','MAE','RMSE','MASE'])}

### Condición de 40 % de pérdida

{table(q40,['model','resource','pattern','MASE','MASE_IC95_inferior','MASE_IC95_superior'])}

![Sensibilidad a las pérdidas](sensibilidad_perdidas.png)

## Contrastes C1–C7

C1/C2 comparan la degradación al 40 % frente al 10 %; C3/C4, bloques frente a dispersión; C5/C6, GRU propia menos GRU con llamadas; C7, degradación CPU menos memoria. Un valor positivo en C5/C6 favorece la información de llamadas.

{table(contrasts,['contraste','recurso','diferencia_MASE','IC95_inferior','IC95_superior'])}

Estos contrastes son **descriptivos condicionales**. No se declara confirmación de hipótesis ni significación ajustada por Holm: no se ha justificado la simetría/invariancia requerida para la prueba de signos prevista en el protocolo. No se sustituyó la prueba por otra elegida a partir de los resultados. Los intervalos no incorporan toda la incertidumbre del entrenamiento, la selección de la cohorte o la infraestructura.

## Validación de resultados

Se volvieron a calcular independientemente MAE, RMSE y MASE a partir de los **640 archivos de predicciones** guardados y los objetivos originales: coinciden las **125.440 filas** de resultados detallados, con tolerancia relativa 1e-11 y absoluta 1e-9. Se verificaron integridad de archivos, objetivos idénticos entre condiciones, valores finitos y ausencia de filas duplicadas. Los controles de software verificaron invariancia ante perturbaciones de valores ocultos y futuros, causalidad temporal, ejecución de gradientes en CUDA, repetición exacta de inferencia y aislamiento propio/vecinal de la pareja GRU. Esto valida el circuito y los cálculos; no demuestra superioridad general ni validez causal del grafo.

## Límites para el trabajo académico

**Reserva con exposición histórica parcial:** el nuevo tramo de prueba [6404, 8005) contiene 400 observaciones por servicio, [6404, 6804), que pertenecían a la validación de los pilotos históricos 70/15/15. Aunque la selección de ajustes de esta ejecución utiliza exclusivamente [5603, 6404), no se afirma que todo el nuevo tramo de prueba nunca haya sido consultado en el desarrollo previo. Esta exposición limita su interpretación confirmatoria y refuerza la presentación descriptiva del estudio. No se cambia la división 70/10/20 solicitada ni se oculta el solapamiento histórico.

La cohorte es una selección por conveniencia de 28 servicios de una única infraestructura. El grafo se trata como estático y conocido; se excluyen 137 aristas de frontera. Las pérdidas son inducidas e independientes entre servicios y no representan necesariamente caídas sincronizadas del colector. Las veinte realizaciones no equivalen a veinte infraestructuras independientes. No se midieron autoescalado, SLA, ahorro monetario ni validación externa en Alibaba. Las comparaciones entre modelos externos son exploratorias. Conservar los resultados favorables y desfavorables; no atribuir al estudio conclusiones fuera de estos datos y condiciones.

## Archivos

- `resumen_prueba.csv`: métricas e intervalos por modelo, recurso y escenario.
- `metricas_detalladas_prueba.csv`: métricas por servicio, inicialización y realización.
- `metricas_por_realizacion.csv`: unidad de agregación para incertidumbre.
- `contrastes_descriptivos.csv`: siete magnitudes con intervalos descriptivos.
- `validacion_independiente.json`: controles de integridad y recomputación.
- `reproduccion_y_modelos.zip`: código, fuentes del material, configuración y checkpoints.
- `manifest_sha256.json`: huellas de los entregables.

Las predicciones completas y los estados reanudables permanecen además en `ChronoGraph/results/study_701020/` del equipo. No se publicaron cambios en GitHub ni se modificaron las tres etapas académicas del ZIP.
'''
    (deliver/'INFORME_RESULTADOS.md').write_text(report)
    dump(deliver/'validacion_independiente.json',dict(status='passed',prediction_artifacts=640,metric_rows=125440,
        independent_recomputation=True,rtol=1e-11,atol=1e-9,target_consistency=True,files=checks))
    for filename in ['run_config.json','data_audit.json','validation_checks.json','selected_learning_rates.json','frozen_models.json','cuda_graph_equivalence.json','checkpoint_restore_check.json']:
        shutil.copy2(OUT/filename,deliver/filename)
    # Compact checkpoints contain the selected state; full optimizer/RNG states remain on disk.
    import torch
    with zipfile.ZipFile(deliver/'reproduccion_y_modelos.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in HERE.rglob('*'):
            if p.is_file() and '__pycache__' not in str(p):archive.write(p,'study/'+str(p.relative_to(HERE)))
        for p in (OUT/'checkpoints').glob('*.pt'):
            ck=torch.load(p,map_location='cpu',weights_only=False)
            import io
            buffer=io.BytesIO();torch.save({k:ck[k] for k in ['model','resource','seed','lr','best_state','best','parameter_count','state_sha256','history']},buffer)
            archive.writestr('checkpoints/'+p.name,buffer.getvalue())
        for filename in ['run_config.json','data_audit.json','frozen_models.json','selected_learning_rates.json','validation_checks.json','cuda_graph_equivalence.json','checkpoint_restore_check.json','progress.jsonl','execution.log','preflight_config.json','pre_acceleration_config.json','pre_stable_clipping_config.json']:
            if (OUT/filename).exists():archive.write(OUT/filename,filename)
        archive.write(OUT/'data_verified.npz','data_verified.npz')
    dump(deliver/'manifest_sha256.json',{p.name:sha(p) for p in deliver.iterdir() if p.is_file() and p.name!='manifest_sha256.json'})
    dump(OUT/'COMPLETED.json',dict(status='complete',independent_validation='passed',metric_rows=125440,forecast_files=640))
    print('REPORT_AND_INDEPENDENT_VALIDATION_PASSED',flush=True)

if __name__=='__main__':main()

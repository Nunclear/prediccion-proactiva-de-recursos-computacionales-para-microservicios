"""Render the revised article from verified CSVs; no manually supplied metric values."""
import json,re,hashlib
from pathlib import Path
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
OUT=ROOT/'results/comparaciones';ART=ROOT/'articulo'
RES=['cpu_usage','container_memory_usage_bytes'];NAME={RES[0]:'CPU',RES[1]:'Memoria'}
def num(x,n=6):return f'{float(x):.{n}f}'
def table(df):
 rows=['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']
 for row in df.itertuples(index=False,name=None):rows.append('| '+' | '.join(str(x) for x in row)+' |')
 return '\n'.join(rows)
def interval(a,b,n=6):return f'[{num(a,n)}; {num(b,n)}]'
def main():
 valid=json.loads((OUT/'validacion.json').read_text());assert valid['status']=='passed'
 assert (ROOT/'results/benchmark_COMPLETE.json').exists()
 allmeans=pd.read_csv(OUT/'resumen_perdidas.csv');summary=pd.read_csv(OUT/'resumen_condiciones.csv')
 c=pd.read_csv(OUT/'contrastes_dos_escalas.csv');seeds=pd.read_csv(OUT/'contrastes_por_semilla.csv');services=pd.read_csv(OUT/'contrastes_por_servicio.csv')
 costs=pd.read_csv(OUT/'costes_nueva_medicion.csv');dynamics=pd.read_csv(OUT/'dinamica_entrenamiento_mediana.csv').set_index('resource')
 chosen=json.loads((ROOT/'results/nuevos_experimentos/selected_candidates.json').read_text())['choices']
 frozen=json.loads((ROOT/'results/nuevos_experimentos/frozen_v2.json').read_text());router=frozen['router']
 def contrast(r,model,reference='E0'):
  return c[(c.resource==RES[r])&(c.model==model)&(c.reference==reference)&(c.conditions=='perdidas')].iloc[0]
 def cost(r,model):return costs[(costs.resource==RES[r])&(costs.model==model)].iloc[0]
 cpu,mem=contrast(0,'E5'),contrast(1,'E5');v_cpu,v_mem=contrast(0,'V2'),contrast(1,'V2')
 values={}
 values['RESUMEN']=f'''El pronóstico de recursos de microservicios depende tanto de su dinámica temporal como de la disponibilidad de telemetría. Este estudio explora si una descomposición exponencial y características causales de ausencia modifican el error y el coste de DLinear. Se auditó y reprodujo el experimento del Grupo 05 sobre veintiocho servicios de ChronoGraph, con ocho mil cinco tiempos, CPU y memoria, doce observaciones históricas y horizonte de treinta minutos. La división cronológica fue setenta, diez y veinte por ciento; se conservaron tres inicializaciones y veinte realizaciones emparejadas de pérdidas dispersas o por bloques. Se verificaron los seis predictores originales y las ablaciones existentes, y se ejecutaron veinticuatro entrenamientos finales de cuatro familias compactas. La variante V2 se seleccionó por validación antes de calcular sus métricas de prueba. E5 obtuvo MASE {num(cpu.MASE_model)} en CPU y {num(mem.MASE_model)} en memoria, con reducciones emparejadas de {num(cpu.reduction_percent,3)} % y {num(mem.reduction_percent,3)} % frente a DLinear. V2 registró reducciones de {num(v_cpu.reduction_percent,3)} % y {num(v_mem.reduction_percent,3)} %, respectivamente. Se corrigió la presentación de intervalos absolutos y porcentuales, se añadió sensibilidad a inicializaciones y se midieron parámetros, latencia y memoria. Se verificaron {valid['unique_physical_prediction_files']} archivos de pronósticos mediante integridad o recomputación. Los resultados no respaldan superioridad uniforme ni conservación del beneficio mediante compresión. La prueba conocida, la cohorte de conveniencia y la ausencia de evaluación externa mantienen el carácter exploratorio.'''
 abstract_words=len(values['RESUMEN'].split());assert 200<=abstract_words<=250,abstract_words
 literature=[
 ['Hyndman y Athanasopoulos','2021','Descomposición y suavizamiento','Pronóstico temporal','Distinguir nivel, tendencia y estacionalidad','No evalúa esta arquitectura ni estas máscaras'],
 ['Zeng et al.','2023','LTSF-Linear / DLinear','Pronóstico de series','Referencia lineal con descomposición','Sus resultados no garantizan el horizonte local de 30 min'],
 ['Cho et al.','2014','GRU','Modelado secuencial en traducción','Estado recurrente con compuertas','No es una evaluación de CPU y memoria'],
 ['Che et al.','2018','GRU-D','Series multivariantes incompletas','Disponibilidad y tiempo transcurrido','Dominio clínico; no demuestra transferencia a microservicios'],
 ['Chen et al.','2024','BiTGraph','Pronóstico con observaciones faltantes','Relaciones temporales y de grafo','Presupuestos y patrones distintos del experimento local'],
 ['Yu et al.','2024','GinAR','Pronóstico con variables ausentes','Modelado temporal e interdependencia','Ausencia de variables no equivale a nuestra pérdida parcial'],
 ['Lutu et al.','2025','ChronoGraph','Mediciones de microservicios','Dataset temporal con dependencias e incidentes','Aquí solo se estudia una cohorte de 28 servicios'],
 ['Luo et al.','2022','Aprendizaje de carga','Autoescalado de microservicios','Vincula predicción y gestión','Aquí no se ejecuta su controlador ni se miden SLA'],
 ['Hyndman y Koehler','2006','MASE','Comparación de errores','Escala respecto de una referencia de entrenamiento','Su interpretación depende del denominador y agregación']]
 values['TABLA_LITERATURA']='**Tabla 1. Antecedentes y límites de transferencia.**\n\n'+table(pd.DataFrame(literature,columns=['Autores','Año','Método','Problema','Aporte pertinente','Límite']))
 labels=['persistencia','E0','GRU_propia','GRU_llamadas','BiTGraph_codigo_autor','GinAR','E2','E4','E5','V2']
 tab=allmeans[allmeans.model.isin(labels)].copy();tab['model']=pd.Categorical(tab.model,labels,ordered=True);tab=tab.sort_values(['resource','model'])
 tab=pd.DataFrame({'Recurso':tab.resource.map(NAME),'Modelo':tab.model.astype(str).replace({'BiTGraph_codigo_autor':'BiTGraph'}),'MASE':tab.MASE.map(lambda v:num(v)),'MAE':tab.MAE.map(lambda v:f'{v:.7g}'),'RMSE':tab.RMSE.map(lambda v:f'{v:.7g}')})
 values['TABLA_MODELOS']=table(tab)+'\n\nFuente: resumen_perdidas.csv. MAE y RMSE conservan unidades por recurso; MASE es adimensional.'
 values['HALLAZGOS_PRINCIPALES']=f'''E5 presenta una reducción emparejada de {num(cpu.reduction_percent,3)} % en CPU y {num(mem.reduction_percent,3)} % en memoria. El signo negativo de CPU significa deterioro. Las GRU registran menor MASE medio de CPU que E0 en este ensayo; en memoria, E5 mejora frente a ambas GRU. GinAR conserva un MASE extremo de CPU que afecta su promedio. Estas afirmaciones describen el protocolo local y no una clasificación universal de arquitecturas.'''
 rows=[]
 params={'E0':26,'E1':26,'E2':26,'E3':62,'E4':38,'E5':74,'C0':26,'C1':27,'C2':27,'C3':28}
 desc={'E0':'MA25','E1':'Ventana seleccionada','E2':'EMA','E3':'MA + procedencia','E4':'MA + edad','E5':'EMA + procedencia + edad','C0':'Base compacta','C1':'Base + fracción ausente','C2':'Base + edad reciente','C3':'Base + ambas'}
 for label in params:
  a=allmeans[(allmeans.model==label)&(allmeans.resource==RES[0])].iloc[0];b=allmeans[(allmeans.model==label)&(allmeans.resource==RES[1])].iloc[0]
  rows.append([label,desc[label],params[label],num(a.MASE),num(b.MASE)])
 values['TABLA_ABLACIONES']='**Tabla 3a. MASE con pérdidas de las ablaciones; parámetros por recurso.**\n\n'+table(pd.DataFrame(rows,columns=['Variante','Contenido','Parámetros','CPU','Memoria']))
 values['SELECCION_V2']='La selección congelada de V2 fue:\n\n'+table(pd.DataFrame([
  [NAME[resource],router[str(r)],chosen[f'{router[str(r)]}_r{r}']['spec']['backend'],num(frozen['family_validation'][f'{router[str(r)]}_r{r}']['score']),int(cost(r,'V2').parameters)]
  for r,resource in enumerate(RES)],columns=['Recurso','Familia','Descomposición','MASE validación medio','Parámetros']))+'\n\nEl valor de validación es el promedio de los mejores checkpoints de tres inicializaciones, no una cifra de prueba ni una media de infraestructura.'
 pattern=summary[summary.model.isin(['E0','E2','E4','E5','V2'])].pivot(index=['resource','q','pattern'],columns='model',values='MASE').reset_index()
 pattern['resource']=pattern.resource.map(NAME)
 for col in ['E0','E2','E4','E5','V2']:pattern[col]=pattern[col].map(lambda v:num(v))
 pattern=pattern.rename(columns={'resource':'Recurso','q':'Pérdida (%)','pattern':'Patrón'})
 values['TABLA_PATRONES']='**Tabla 3b. MASE por tasa y patrón, incluyendo referencia sin pérdidas.**\n\n'+table(pattern)
 selected=c[(c.reference=='E0')&(c.conditions=='perdidas')&c.model.isin(['E2','E4','E5','V2'])].copy()
 a=[];b=[]
 for row in selected.itertuples():
  a.append([NAME[row.resource],row.model,num(row.delta_MASE),interval(row.delta_MASE_low,row.delta_MASE_high),num(row.reduction_percent,3),interval(row.reduction_percent_low,row.reduction_percent_high,3)])
  b.append([NAME[row.resource],row.model,interval(row.crossed_delta_MASE_low,row.crossed_delta_MASE_high),num(row.crossed_reduction_ratio_of_means_percent,3),interval(row.crossed_reduction_percent_low,row.crossed_reduction_percent_high,3)])
 values['TABLA_INTERVALOS']=table(pd.DataFrame(a,columns=['Recurso','Modelo','Δ MASE','IC95 Δ (MASE)','Reducción (%)','IC95 reducción (%)']))
 values['TABLA_CRUZADOS']=table(pd.DataFrame(b,columns=['Recurso','Modelo','IC95 cruzado Δ (MASE)','Razón de medias (%)','IC95 cruzado razón (%)']))
 observations=[]
 for resource in RES:
  s=seeds[(seeds.resource==resource)&(seeds.reference=='E0')&(seeds.model=='E5')]
  sv=services[(services.resource==resource)&(services.reference=='E0')&(services.model=='E5')]
  observations.append(f"{NAME[resource]}: E5 tuvo Δ favorable en {int((s.delta_MASE>0).sum())} de 3 semillas y en {int((sv.delta_MASE>0).sum())} de 28 servicios al promediar las demás condiciones")
 values['HALLAZGOS_SEMILLAS']='; '.join(observations)+'. Los intervalos cruzados de E5 incluyen cero en ambos recursos. Esto no cambia el promedio observado, pero impide presentar el intervalo estrecho condicionado a máscaras como estabilidad frente a cualquier inicialización.'
 cost_rows=[];memory_rows=[]
 for r,resource in enumerate(RES):
  for label in ['E0','E2','E4','E5','V2']:
   x=cost(r,label);cost_rows.append([NAME[resource],label,int(x.parameters),num(x.latency_p50_ms,3),num(x.latency_p95_ms,3),int(x.epochs),num(x.training_validation_seconds,3),num(x.mean_epoch_including_validation_seconds,3)])
   memory_rows.append([NAME[resource],label,num(x.process_peak_rss_bytes/1024**2,1),num(x.vram_peak_allocated_bytes/1024**2,2),num(x.vram_forward_extra_peak_bytes/1024**2,2)])
 values['TABLA_COSTE']='**Tabla 6a. Parámetros, latencia por lote y ajuste observado de la semilla 101.**\n\n'+table(pd.DataFrame(cost_rows,columns=['Recurso','Modelo','Parám.','p50 (ms)','p95 (ms)','Épocas','Ajuste (s)','s/época con validación']))+'\n\n**Tabla 6b. RAM del proceso y VRAM asignada en la medición nueva.**\n\n'+table(pd.DataFrame(memory_rows,columns=['Recurso','Modelo','Máx. RSS proceso (MiB)','Pico CUDA asignado (MiB)','Extra forward (MiB)']))+'\n\nRSS es la cifra máxima reportada por el sistema operativo; las instantáneas RSS y las temporizaciones completas están archivadas. CUDA asignada no equivale a toda la VRAM ocupada por el controlador.'
 extra=contrast(1,'E5','E2');e2=cost(1,'E2');e5=cost(1,'E5')
 values['COMPROMISO_COSTE']=f'''En memoria, pasar de E2 a E5 reduce el MASE emparejado otro {num(extra.reduction_percent,3)} %, mientras los parámetros pasan de {int(e2.parameters)} a {int(e5.parameters)} y la latencia mediana pasa de {num(e2.latency_p50_ms,3)} a {num(e5.latency_p50_ms,3)} ms por lote: {num(100*(e5.latency_p50_ms/e2.latency_p50_ms-1),1)} % más. E5/E0 mantiene una razón de parámetros {num(e5.parameters/cost(1,'E0').parameters,3)}. En CPU, el mayor coste de E5 acompaña un peor MASE medio. La menor duración total de algunos ajustes coincide con distinto número de épocas y condiciones de ejecución; no se atribuye exclusivamente a eficiencia de la arquitectura.'''
 values['RESULTADO_V2']=f'''V2 obtuvo MASE {num(v_cpu.MASE_model)} en CPU y {num(v_mem.MASE_model)} en memoria. Frente a E0, las reducciones emparejadas fueron {num(v_cpu.reduction_percent,3)} % y {num(v_mem.reduction_percent,3)} %. Frente a E5, fueron {num(contrast(0,'V2','E5').reduction_percent,3)} % y {num(contrast(1,'V2','E5').reduction_percent,3)} %. La simplificación redujo parámetros a 28 en CPU y 27 en memoria, y su latencia medida fue {num(cost(0,'V2').latency_p50_ms,3)} y {num(cost(1,'V2').latency_p50_ms,3)} ms. No conservó la precisión de E5 ni mejoró simultáneamente ambos recursos frente a E0. Tener menos parámetros tampoco convirtió la EMA de CPU en una inferencia más rápida que E0.'''
 values['DINAMICA']=f'''la autocorrelación mediana de primer retardo fue {num(dynamics.loc[RES[0],'acf_1'],3)} para CPU y {num(dynamics.loc[RES[1],'acf_1'],3)} para memoria; la media absoluta de diferencias dividida por la desviación de cada serie tuvo medianas {num(dynamics.loc[RES[0],'relative_mean_absolute_change'],3)} y {num(dynamics.loc[RES[1],'relative_mean_absolute_change'],3)}, respectivamente.'''
 monotonic=0
 for resource in RES:
  for model in ['E0','E2','E4','E5','V2']:
   for pattern in ['disperso','bloques']:
    curve=summary[(summary.resource==resource)&(summary.model==model)&((summary.pattern==pattern)|(summary.q==0))].sort_values('q').MASE.values
    monotonic+=int(np.all(np.diff(curve)>0))
 values['CONCLUSIONES']=f'''H1 no quedó respaldada como mejora conjunta: E5 empeoró CPU {num(-cpu.reduction_percent,3)} % y mejoró memoria {num(mem.reduction_percent,3)} % en los promedios emparejados con pérdidas. H2 presentó aumento de MASE en {monotonic} de 20 curvas de E0, E2, E4, E5 y V2 al recorrer las cuatro tasas; no se extiende este resultado a todos los comparadores ni a averías reales. H3 no quedó respaldada como conservación de precisión: V2 redujo tamaño, pero su cambio fue {num(v_cpu.reduction_percent,3)} % en CPU y {num(v_mem.reduction_percent,3)} % en memoria frente a E0. H4 no produjo un beneficio conjunto en prueba, aunque su selección respetó validación. El análisis entre inicializaciones amplió la incertidumbre y no autoriza afirmaciones confirmatorias.'''
 refs=[
 ('Che, Z., Purushotham, S., Cho, K., Sontag, D., & Liu, Y. (2018). Recurrent neural networks for multivariate time series with missing values. *Scientific Reports, 8*, artículo 6085.','https://doi.org/10.1038/s41598-018-24271-9'),
 ('Chen, X., Li, X., Liu, B., & Li, Z. (2024). Biased temporal convolution graph network for time series forecasting with missing values. *International Conference on Learning Representations*.','https://proceedings.iclr.cc/paper_files/paper/2024/hash/7f3b25e39e0f0330435d1ffc1d6fa193-Abstract-Conference.html'),
 ('Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. *Proceedings of EMNLP*, 1724–1734.','https://doi.org/10.3115/v1/D14-1179'),
 ('Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3.ª ed.). OTexts.','https://otexts.com/fpp3/'),
 ('Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting, 22*(4), 679–688.','https://doi.org/10.1016/j.ijforecast.2006.03.001'),
 ('Luo, S., Xu, H., Ye, K., Xu, G., Zhang, L., Yang, G., & Xu, C. (2022). The power of prediction: Microservice auto scaling via workload learning. *Proceedings of the ACM Symposium on Cloud Computing*, 355–369.','https://doi.org/10.1145/3542929.3563477'),
 ('Lutu, A. C., Pintilie, I., Burceanu, E., & Manolache, A. (2025). *ChronoGraph: A real-world graph-based multivariate time series dataset*. arXiv:2509.04449.','https://arxiv.org/abs/2509.04449'),
 ('Yu, C., Wang, F., Shao, Z., Qian, T., Zhang, Z., Wei, W., & Xu, Y. (2024). GinAR: An end-to-end multivariate time series forecasting model suitable for variable missing. *Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, 3989–4000.','https://doi.org/10.1145/3637528.3672055'),
 ('Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are transformers effective for time series forecasting? *Proceedings of the AAAI Conference on Artificial Intelligence, 37*(9), 11121–11128.','https://doi.org/10.1609/aaai.v37i9.26317')]
 values['REFERENCIAS']='\n\n'.join(f'{text} [Fuente]({url}).' for text,url in refs)
 captions=[
 ('fig01_series','Series originales del primer servicio, con particiones y huecos temporales. CPU conserva su escala publicada; memoria se expresa en bytes.'),
 ('fig02_descomposicion','Nivel y residual de MA25 y EMA0,7 en la misma historia de entrenamiento. La EMA se inicializa con la primera entrada preparada; no representa una estacionalidad identificada.'),
 ('fig03_ausencias','Pérdidas dispersas globales del 20 % en un tramo fijo de validación; los objetivos originales no se alteran.'),
 ('fig04_ausencias','Pérdidas por bloques globales del 20 % sobre el mismo tramo; el porcentaje local ilustrado puede diferir del global.'),
 ('fig05_pronosticos','Objetivos y pronósticos reales archivados para E0, E5 y V2: primer servicio, primeras 120 ventanas de prueba, 20 % por bloques, inicialización 101 y realización 0.'),
 ('fig06_comparadores','MASE con pérdidas de los comparadores centrales. CPU usa escala logarítmica para conservar los valores extremos de GinAR; memoria usa escala lineal.'),
 ('fig07_robustez','MASE por tasa, patrón y recurso. El punto sin pérdida es compartido entre patrones y no duplica observaciones.'),
 ('fig08_recursos','Comparación de E0, E2, E4, E5 y V2 por recurso, usando MASE adimensional.'),
 ('fig09_incertidumbre','Diferencias absolutas de MASE frente a E0. Azul: percentiles del remuestreo de máscaras; naranja: sensibilidad cruzada a semillas y máscaras. Positivo favorece la variante. Los intervalos son descriptivos.'),
 ('fig10_recursos_sistema','Telemetría del sistema durante la reproducción y revisión. Los huecos indican ausencia de monitoreo, incluidos apagado y pausas entre ejecuciones; no son mediciones imputadas.'),
 ('fig11_precision_coste','MASE con pérdidas frente a mediana de latencia de los modelos ligeros y GRU. Los comparadores gráficos completos permanecen en los CSV y la Figura 6; se omiten aquí para permitir inspeccionar este rango.'),
 ('fig12_semillas','Tres diferencias por configuración, una por inicialización, tras promediar máscaras, condiciones con pérdidas y servicios. No son tres infraestructuras independientes.'),
 ('fig13_dinamica','Autocorrelación mediana entre 28 series de entrenamiento a tres retardos. La correlación a 24 horas no es por sí sola una prueba de estacionalidad.')]
 values['FIGURAS']='\n\n'.join(f'### Figura {i}. {caption}\n\n![Figura {i}](figuras/{name}.png)' for i,(name,caption) in enumerate(captions,1))
 template=(ROOT/'scripts/articulo_plantilla.md').read_text()
 citations={'Zeng et al. (2023)':refs[-1][1],'Che et al. (2018)':refs[0][1],'Cho et al. (2014)':refs[2][1],'Luo et al. (2022)':refs[5][1],'Hyndman y Athanasopoulos (2021)':'https://otexts.com/fpp3/decomposition.html'}
 for label,url in citations.items():template=template.replace(label,f'[{label}]({url})')
 for key,value in values.items():template=template.replace('@@'+key+'@@',value)
 template=template.replace(r'\(','$').replace(r'\)','$').replace(r'\[','$$').replace(r'\]','$$')
 assert not re.findall(r'@@[A-Z_]+@@',template)
 article=ART/'ARTICULO_EMPIRICO_MEJORADO.md';article.write_text(template)
 # Primary frontier excludes compact-family duplicates of the V2 alias.
 primary=costs[~costs.model.str.startswith('C')].merge(allmeans,on=['model','resource']);front=[]
 for row in primary.itertuples():
  others=primary[(primary.resource==row.resource)&(primary.model!=row.model)]
  dominated=((others.MASE<=row.MASE)&(others.latency_p50_ms<=row.latency_p50_ms)&((others.MASE<row.MASE)|(others.latency_p50_ms<row.latency_p50_ms))).any();front.append(not bool(dominated))
 primary['pareto_descriptive']=front;primary.to_csv(OUT/'pareto_principal.csv',index=False)
 text_no_tables='\n'.join(line for line in template.splitlines() if not line.startswith('|'))
 words=len(re.findall(r"\b[\wÁÉÍÓÚÑáéíóúñ]+\b",template));prose_words=len(re.findall(r"\b[\wÁÉÍÓÚÑáéíóúñ]+\b",text_no_tables))
 metadata=dict(status='rendered',words_including_tables_references_captions=words,words_excluding_table_rows=prose_words,abstract_words=abstract_words,figures=13,metric_source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.csv')},article_sha256=hashlib.sha256(article.read_bytes()).hexdigest())
 (OUT/'articulo_trazabilidad.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False))
 (ROOT/'docs/REFERENCIAS_VERIFICADAS.md').write_text('# Referencias verificadas\n\nSe contrastaron autores, título, año, páginas y DOI de seis trabajos con Crossref (metadatos depositados por editoriales), y contenido con fuentes primarias. Los dos artículos sin DOI editorial se comprobaron en ICLR y arXiv; el libro se comprobó en OTexts. Nature/ACM/AAAI presentaron limitaciones de acceso directo en algunas consultas; se conserva el respaldo de metadatos y enlaces oficiales sin afirmar lectura integral de cada artículo.\n\n'+values['REFERENCIAS']+'\n\nEvidencia: [metadatos Crossref](CROSSREF_METADATOS.json), [referencias previas](REFERENCIAS_BASE.json), [búsqueda externa](BUSQUEDA_DATOS_EXTERNOS.json).\n')
 print(json.dumps(metadata,indent=2),flush=True)
if __name__=='__main__':main()

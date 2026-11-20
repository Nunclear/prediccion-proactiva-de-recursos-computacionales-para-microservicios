# Datos efectivamente consumidos por los experimentos

Este paquete contiene tres entradas procesadas distintas del artículo «Pronóstico de recursos de microservicios con telemetría incompleta: descomposición, modelos compactos y transferencia calibrada». No contiene código, resultados, máscaras generadas, predicciones ni pesos. Los datos originales voluminosos se transformaron antes de entrenar; no son entradas adicionales necesarias para cargar los modelos.

## Archivos y procedencia

| Archivo | Ejes de values | Registros servicio × tiempo | Escala y utilización |
|---|---|---:|---|
| chronograph_source.npz | 8005 × 28 × 2 | 224140 | ChronoGraph fuente; fases históricas y ajuste fuente ampliado |
| chronograph_holdout.npz | 8005 × 32 × 2 | 256160 | ChronoGraph destino; transferencia espacial calibrada |
| alibaba_transfer.npz | 48 × 32 × 2 | 1536 | Alibaba 2022 agregado; transferencia externa exploratoria |

Son 481836 registros servicio-tiempo, 963672 valores escalares (incluidos cuatro ausentes naturales), y 16058 filas temporales sumando las tres matrices. No son 481836 observaciones independientes. NPZ almacena tensores, no una tabla plana: cuatro arrays llamados values, times, services y adjacency. El último eje de values contiene, en este orden, CPU y memoria. MANIFIESTO.csv conserva tamaños, tipos, formas y SHA256 de los bytes originales.

ChronoGraph: https://github.com/bit-ml/ChronoGraph . CPU conserva la escala publicada cpu_usage; memoria se expresa en bytes. Paso nominal 1800 segundos, con salto de 108000 segundos después de la fila 5907. Cortes por filas [0,5603), [5603,6404), [6404,8005): 70/10/20 con redondeo. No se cruzan cortes ni huecos al preparar ventanas. Para destino, las primeras 5603 filas calibran estadísticas, sin ajustar pesos destino. adjacency es el grafo estático conocido.

Alibaba: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-microservices-v2022 . Se procesaron los primeros 48 objetos MSMetricsUpdate (un día): promedio de pods por minuto/servicio y después promedio de treinta minutos completos. values usa utilización normalizada por el proveedor: su segundo canal conserva el identificador técnico de memoria, pero NO mide bytes. Las primeras 12 filas calibran (6 h); las siguientes 36 se evalúan (18 h). Cuatro valores ausentes están solo en calibración. No se aplica 70/10/20. La identidad de adjacency no constituye un grafo de llamadas externo y GRU_graph no se evalúa allí.

## Carga

```python
import numpy as np
with np.load('datos/chronograph_source.npz', allow_pickle=False) as dataset:
    values = dataset['values']
    times = dataset['times']
    services = dataset['services']
    adjacency = dataset['adjacency']
```

Instale numpy para leer las matrices. Los scripts históricos cargan data_verified.npz; sus tres copias existentes son idénticas byte a byte a chronograph_source.npz. El instalador scripts/preparar_datos.py del repositorio crea esos alias sin duplicarlos en este ZIP. Las máscaras se generan a partir de configuración/semillas y no son datasets adicionales.

## Evidencia de utilización

Fases I-II: src/baseline/reference/run.py::Data carga data_verified.npz, scripts/reproducir.py, scripts/ablaciones.py y revision_20261029/scripts/run_v2.py. Fases III-IV: ampliacion_20261110/scripts/run_experiments.py y src/core.py::Series cargan los tres NPZ procesados. Los manifiestos de configuraciones, checkpoints y predicciones se entregan en el repositorio, separados de los datos.

Este ZIP es una entrega académica local. No se atribuye una licencia nueva a la telemetría: la licencia del código ChronoGraph no demuestra por sí sola derechos independientes sobre todos los datos; Alibaba permite uso de investigación según su documentación y no se ha verificado una autorización general de redistribución pública. El repositorio público contiene instrucciones y hashes, no estos datos.

Obtenga DATA_ARTICULO_EMPIRICO.zip de los autores; no hay URL pública de descarga ni DOI confirmado. No se presupone una licencia de redistribución.

# Metodología del artículo definitivo

La especificación formal completa está en articulo_ieee/secciones/metodologia_es.tex y en el PDF IEEE español; esta guía permite localizar sus decisiones principales.

## Diseño y fases

Estudio computacional cuantitativo con evaluación cronológica de origen móvil. Las cinco fases son: I, comparadores históricos y E0–E5; II, C0–C3 y selección V2 por recurso; III, quince familias transferidas a32 servicios adicionales; IV, ausencia añadida, horizontes/historias y caso Alibaba; V, coste CPU/CUDA. Las fases I–II son exploratorias sobre prueba conocida. V2 reutiliza ajustes C, sin entrenamiento independiente. Las filas de métricas repiten servicios, semillas, máscaras y condiciones, y no son réplicas independientes.

## Datos y cortes

ChronoGraph contiene8005 tiempos y dos recursos, con28 servicios fuente y32 adicionales. El70/10/20 por filas corresponde a[0,5603),[5603,6404),[6404,8005). Las ventanas no cruzan estos cortes ni el salto de108000 segundos tras la fila5907. L12/H1 histórico tiene5591/777/1589 orígenes elegibles. El prefijo de5603 filas de cada servicio destino calibra estadísticas sin ajustar pesos ni seleccionar hiperparámetros destino. Fuente y destino comparten infraestructura y período; no se certifica completamente la disponibilidad temporal de las aristas.

Alibaba2022 contiene48 bins semihorarios de un día y32 servicios elegidos por disponibilidad. Los primeros12 calibran y los siguientes36 son objetivos; las cuatro ausencias naturales están solo en calibración. No se aplica70/10/20. CPU/memoria están normalizadas por el proveedor, sin interpretación como bytes o núcleos. Se excluye GRU_graph por falta de red externa. Es transferencia calibrada exploratoria a otro dominio corto.

## Preparación y modelos

Las entradas ausentes usan arrastre causal o mediana de entrenamiento antes de la primera observación. Disponibilidad, procedencia y edad se separan del valor preparado; normalización y denominador MASE usan el prefijo correspondiente anterior a las máscaras. No se imputan objetivos. Arrastre y edad pueden resumir observaciones anteriores a la ventana explícita.

La guía docs/modelos.md describe referencias simples, DLinear/NLinear, E/C/A, GRU, GRU-D y PatchTST. Los modelos neuronales comparten pesos entre servicios y separan recursos. Las selecciones reales por recurso/configuración se preservan en selection.json y frozen.json.

## Ajuste y evaluación

El principal usa L12/H1, quince familias, diez semillas neuronales y veinte máscaras emparejadas. Promedia seis condiciones con pérdidas dispersas/bloques10,20,40%. Se conservan datos completos y tres mecanismos añadidos nominal20%: correlacionado, prolongado y MAR dependiente de CPU previamente observada. MAR realiza una tasa mayor y no se etiqueta como20% exacto.

Cada familia neural ampliada selecciona cuatro combinaciones de learning rate/weight decay en validación fuente, con búsqueda de tres épocas y ajuste final hasta veinte/paciencia cinco. Adam, lote efectivo16, microlote inicial8, clipping1, float32 y dos hilos. No se igualan FLOPs entre familias; las referencias analíticas tienen otro presupuesto.

MAE/RMSE/MASE se calculan tras invertir la normalización. MASE divide MAE por el error ingenuo consecutivo del prefijo, evitando huecos. Los cuatro contrastes comparan A3 frente a DLinear/E5 por recurso, con signos exactos bilaterales sobre diez medias por semilla y Holm. Los intervalos cruzan semillas, máscaras y bloques temporales de servicios fijos, sin inferencia de población de infraestructuras. Bloque principal48, sensibilidad24/96.

Los secundarios usan diseño estrella: L12 con horizontes2,4,6,12 y L24/L48 con horizonte1; tres semillas y cinco máscaras. Los gráficos comparables usan1553 objetivos idénticos, filas6452–8004. No constituyen un factorial completo ni sustituyen el soporte principal completo.

## Interpretación y trazabilidad

Las conclusiones dependen de recurso y escenario, conservando resultados negativos. No se midieron energía, facturación, demanda latente o resultados de controlador. Las matrices/hashes constan en data/MANIFIESTO.csv; la trazabilidad en docs/experimentos.md; las fases en articulo_ieee/inventario_experimentos.csv.

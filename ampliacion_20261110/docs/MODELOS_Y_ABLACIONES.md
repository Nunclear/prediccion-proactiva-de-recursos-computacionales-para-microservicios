# Selección de comparadores y justificación de la variante

La comparación nueva conserva los resultados originales, no exige volver a ajustar cada arquitectura histórica en cada escenario. BiTGraph con código de autor y los demás comparadores permanecen disponibles en la reproducción histórica. La nueva cola incorpora una referencia moderna, PatchTST, y un comparador específico de ausencias, una adaptación GRU-D. Se ejecutan once familias con ajuste estocástico y cuatro referencias simples en el escenario principal. El histórico y la transferencia nueva se presentan en tablas distintas.

| Familia | Pertinencia y decisión | Implementación y alcance |
|---|---|---|
| Persistencia/media móvil/SES/ridge | Referencias baratas obligatorias | Valores rellenados causalmente; SES actualiza solo cuando la observación está disponible; ridge incluye intercepto y regulariza solo pesos |
| DLinear/NLinear | Referencias lineales y de normalización | Compartidas por servicio, salida escalar al horizonte indicado; implementación local transparente, no el entrenamiento de los autores sobre sus benchmarks |
| E5 | Mejor variante histórica completa | EMA de disponibilidad con α=0.7, procedencia observada/carry/fallback y antigüedad acotada; se mantiene decisión histórica |
| GRU propia/con grafo | Conservan la motivación multiserie | Hidden12; nueve entradas derivadas de valor/máscara/antigüedad propia y agregados vecinales; sin crear nuevos pesos en la transferencia |
| GRU-D | Referencia para mecanismo de ausencia | Adaptación shared-service con decaimiento de entrada hacia media normalizada cero y de estado oculto; no reproducción clínica multivariante canónica |
| PatchTST | Comparador moderno factible y licencia explícita | Código oficial Apache2 fijado por commit; configuración pequeña, un bloque, d_model16, dos cabezas, patch4/stride2; pesos compartidos entre servicios y salida de un punto |
| TimesNet/iTransformer/TimeMixer/TiDE | Arquitecturas pertinentes localizadas en la búsqueda | No ejecutadas en esta cola: presupuesto se concentra en cuatro candidatos iguales y diez semillas del conjunto elegido. No se atribuyen resultados a estas arquitecturas ni se descartan por errores de test |
| BiTGraph | Comparador histórico de ausencias y grafo | Se preserva resultado y código de autor en el estudio anterior; no se declara entrenado nuevamente en la cohorte de transferencia |

## Hipótesis compacta

Para una ventana rellenada v∈R^L, una descomposición fija S y dos cabezas lineales de DLinear:

`ŷ = w_t^T S v + w_r^T (I−S)v + b_t+b_r = w_eff^T v + b_eff`.

Esto demuestra equivalencia de la **clase de funciones lineales**, no equivalencia de trayectorias de Adam bajo parametrizaciones distintas. El estudio comprueba la identidad de inicialización y deja que los entrenamientos produzcan diferencias empíricas. A0 colapsa las dos cabezas a una única capa afín de L+1 parámetros. A1 añade un escalar para la fracción ausente de la ventana; A2 añade un escalar para la última antigüedad logarítmica; A3 añade ambos. Los indicadores se derivan únicamente de observaciones disponibles. La antigüedad usa log(1+δ)/log(1+L), sin recorte en A2/A3; E5 conserva su antigüedad lineal recortada histórica.

Todos A0–A3 comienzan con la misma función que DLinear para una semilla compartida: se construyen las cabezas originales, se calcula su proyección efectiva en float32 y se inicializan los dos pesos adicionales a cero. La equivalencia algebraica inicial se comprueba numéricamente con rtol1e−5 y atol1e−6; no se afirma identidad bit a bit entre el cálculo descompuesto y el colapsado. Con L12, los tamaños son13/14/14/15 frente a26 de DLinear y74 de E5. El menor número de parámetros no implica por sí solo menor latencia, energía, error, SLA o capacidad de autoscaling. Es una ablación de parametrización y dos descriptores, no un supuesto nuevo estado del arte arquitectónico.

## Presupuesto y diferencias inevitables

Cada familia neural recibe cuatro configuraciones y el mismo número de épocas/orígenes de búsqueda, usando exclusivamente validación fuente. Esta igualdad no es igualdad de FLOPs o segundos. Se emplea un microlote8 con acumulación para lote efectivo16: BatchNorm de PatchTST ve microlotes, mientras la actualización de Adam utiliza el lote acumulado. No hay promesa de reproducir su receta publicada. Las referencias sin entrenamiento y ridge analítico tienen un presupuesto diferente informado. La búsqueda estrecha puede favorecer o perjudicar algunas familias; no se retocará después de ver la prueba.

La aceleración de GRU-D captura los mismos kernels en CUDA Graphs. Se verificó igualdad bit a bit de pesos/gradientes y de inferencia, y reanudación exacta tras una interrupción simulada. Se excluye la captura inicial de los benchmarks de latencia estable, pero se conserva su tiempo en el entrenamiento real. Se informan benchmarks eager y CUDA Graph separados para esta adaptación.

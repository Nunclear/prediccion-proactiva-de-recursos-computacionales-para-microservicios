# Protocolo experimental de ampliación

La especificación ejecutable y sus hashes están en `configs/protocolo_congelado.yaml`. La versión inicial se congeló el 10 de noviembre de 2026 a las 20:02:30 UTC. La enmienda v2, registrada antes de generar resultados de prueba, conserva la matemática y acelera GRU-D con CUDA Graphs; véase `ENMIENDA_PROTOCOLO_PRETEST.md`. Este registro local fechado no equivale a una prerregistración pública independiente.

## Preguntas e hipótesis operativas

| Pregunta | Hipótesis y evaluación |
|---|---|
| RQ1: telemetría incompleta | La diferencia MASE respecto al caso completo aumenta con la pérdida; curvas emparejadas por modelo/recurso/patrón, sin imponer monotonía a los resultados |
| RQ2: descomposición y coste | La parametrización afín equivalente puede reducir parámetros sin garantizar menor error; comparar DLinear, E5 y A0–A3 junto con tiempo y latencia |
| RQ3: indicadores | A1 añade fracción ausente, A2 antigüedad y A3 ambos sobre A0; ablación descriptiva que conserva componentes y resultados negativos |
| RQ4: compacidad | A3, con L+3 parámetros por recurso, puede mantener o mejorar MASE frente a DLinear y E5; cuatro contrastes primarios bilaterales, dos referencias por dos recursos |
| RQ5: transferencia | La dirección y magnitud pueden cambiar con servicios, horizonte e infraestructura; nuevas cohortes, diseño en estrella y caso externo exploratorio |
| RQ6: variabilidad | Inicializaciones, máscaras y períodos contribuyen a diferencias; informar desviaciones e intervalos condicionados al conjunto fijo de servicios |

Las hipótesis sobre compacidad y ausencias nacen del estudio histórico ya observado. La nueva cohorte permite evaluar prospectivamente comparaciones algorítmicas condicionadas a esta infraestructura; no transforma esa motivación retrospectiva en descubrimiento independiente. El histórico no se reemplaza. Los resultados fuente del mismo período de prueba ya eran conocidos en el trabajo anterior; esta ampliación no es una prueba temporalmente ciega. Se denomina evaluación primaria predefinida de transferencia espacial, sin afirmar una confirmación independiente entre infraestructuras o períodos. La cohorte se excluye de los artefactos previos localizados, no se certifica un cegamiento por terceros.

## Datos, particiones y unidades

ChronoGraph aporta 8005 tiempos comunes a CPU y memoria, con paso nominal de 30 minutos. Los límites cronológicos son [0,5603), [5603,6404), [6404,8005). Se conserva la cohorte fuente histórica de 28 servicios. La cohorte nueva de 32 servicios se eligió por grafo estático y orden SHA reproducible, excluyendo los servicios históricos y cuatro pilotos conocidos, sin consultar errores para elegirla. Es una subred de conveniencia, no una muestra aleatoria poblacional.

Todos los pesos y elecciones se obtienen exclusivamente con entrenamiento y validación de los 28 servicios fuente. Para cada servicio nuevo se calculan media, escala, mediana y denominador MASE con su prefijo de entrenamiento. Por tanto, se trata de **transferencia calibrada**, no de transferencia sin contacto con el dominio destino. El grafo destino reemplaza únicamente buffers de adyacencia de modelos GRU al transferir. No se ajustan pesos ni hiperparámetros destino.

Una muestra de pronóstico contiene L observaciones hasta t−h y predice directamente y_t. Las ventanas no atraviesan cortes de partición ni huecos de tiempo. El relleno causal se reinicia en cada segmento, usando mediana de entrenamiento hasta encontrar la primera observación disponible. No se interpolan objetivos. L cuenta pasos preparados: un valor propagado causalmente y su antigüedad pueden representar una observación anterior al inicio de la ventana. La misma preparación se comparte entre comparadores. Las normales de ChronoGraph asumen un prefijo histórico completo antes de inyectar pérdidas; no se evalúa una calibración degradada por esas máscaras. La ausencia sintética afecta entradas, no reemplaza la verdad de los objetivos. CPU y memoria se modelan por separado; GRU-D usa solo el recurso correspondiente y sus indicadores. El escenario MAR utiliza CPU previa observada para construir un mecanismo compartido de ausencia, no un predictor auxiliar futuro.

La unidad para la aleatoriedad de entrenamiento es una ejecución con semilla. Las máscaras y ventanas no son réplicas independientes de entrenamiento. Los servicios relacionados se mantienen unidos en el remuestreo temporal; ninguna cifra de filas se utiliza como tamaño muestral de una población de infraestructuras.

## Factores y controles

Modelos principales: persistencia, media móvil, suavizamiento exponencial, regresión ridge, DLinear, NLinear, E5, A0–A3, GRU propia, GRU con grafo, adaptación GRU-D y PatchTST compacto basado en código oficial. Diez semillas: 101,202,303,404,505,606,707,808,909,1010. Historia principal L=12; h=1 paso, 30 min. Veinte máscaras por cada modelo/recurso; las mismas máscaras se comparten entre modelos y semillas. Se conserva la secuencia histórica de semillas de máscara, aplicada a la cohorte nueva; no se presenta como una política de ausencias antes desconocida. La novedad experimental es espacial, de mecanismos añadidos y de la traza externa, no un período temporal nuevo.

Se conservan las siete condiciones históricas: completa y 10/20/40% por dispersión y bloques de hasta seis pasos. Se añaden tres condiciones: 20% correlacionado por grupos sintéticos de ocho servicios, 20% por bloques prolongados de hasta 24 pasos y MAR dependiente del último CPU observado. Esta última tiene probabilidad variable entre 0.2 y 0.3: se informará su pérdida realizada y no se presentará como exactamente 20%. Todas estas interrupciones son sintéticas; no representan fallos operativos medidos. Las máscaras se generan por segmento, comparten realizaciones y conservan conteos discretos por servicio cuando el mecanismo lo permite.

Diseño secundario en estrella: (L,h)=(12,2),(12,4),(12,6),(12,12),(24,1),(48,1), correspondientes a 1/2/3/6 h o ventanas de 12/24 h. DLinear, E5, A3, PatchTST y cuatro referencias simples; tres semillas y cinco máscaras, declarado exploratorio. No se afirma haber ejecutado todas las combinaciones historia×horizonte.

Por cada modelo entrenado se prueban cuatro combinaciones LR∈{0.001,0.003} y WD∈{0,0.001}, tres épocas sobre 512 orígenes fuente, 64 de validación y selección por MASE de la tercera época sobre las diez condiciones. Los ajustes definitivos usan todo entrenamiento fuente, hasta veinte épocas, paciencia cinco y mejor validación. Cada origen recibe una condición por época mediante (posición+época) módulo10; el orden de orígenes se permuta con RNG de la semilla. No se afirma que una época neural recorra diez veces todas las muestras. Validación/prueba evalúan las diez condiciones completas; ridge analítico sí agrega todas las condiciones en su objetivo. Las referencias simples tienen cuatro candidatos salvo persistencia; el ajuste analítico tiene distinto coste que SGD. Adam, float32, clipping 1, lote efectivo16 con microlotes8, orden y RNG reproducibles; reanudación desde época completa. Cualquier OOM reduce microlote y queda registrado.

## Métricas y estimandos

MAE=media_t |ŷ−y|; RMSE=raíz(media_t (ŷ−y)²). MASE divide MAE de cada servicio por su media de diferencias absolutas consecutivas, calculada exclusivamente en el prefijo de entrenamiento y excluyendo pares separados por huecos. Un denominador nulo produce valor indefinido, no un denominador artificial. Se informan unidades originales por dataset; Alibaba no tiene unidades de bytes o núcleos.

El criterio principal es la media con igual peso por servicio, por las seis condiciones históricas con pérdida positiva y por veinte máscaras. La diferencia emparejada se define como error_referencia−error_A3; una cifra positiva favorece A3. La reducción porcentual emparejada usa 100×(error_referencia−error_A3)/error_referencia por celda emparejada válida, además de informar diferencias absolutas. No confundir media de cocientes con cociente de medias.

## Inferencia estadística

Se aplican cuatro pruebas bilaterales exactas de signos sobre las diez diferencias medias emparejadas por semilla: A3 frente a DLinear y E5 para CPU y memoria. Los empates se excluyen y se informa el n efectivo. Control Holm para esa familia de cuatro pruebas. La inferencia cubre la aleatoriedad de entrenamiento condicionada a trazas y máscaras fijas; no demuestra independencia entre servicios o infraestructuras. Se publican también tamaños de efecto, desviaciones por semilla y máscara y heterogeneidad por servicio.

Intervalos de incertidumbre: 2000 remuestreos cruzados de semillas, realizaciones de máscara y bloques temporales circulares, con servicios fijos promediados juntos. Longitud primaria48 pasos y sensibilidad24/96; semilla bootstrap61110. Se conserva el emparejamiento entre modelos en cada remuestreo. No se denominarán intervalos poblacionales generales. Comparaciones secundarias y externas descriptivas, sin búsqueda de significación sobre todo el universo de celdas.

## Caso externo

Alibaba2022 MSMetricsUpdate, 48 objetos oficiales de media hora, primer día. Hasta32 nombres SHA seleccionados entre servicios con cobertura de todos los tiempos del objeto inicial; solo disponibilidad, sin ranking por error. Media entre pods para cada tiempo y después media de tiempos dentro de cada intervalo completado de treinta minutos. La cadencia efectiva y las ausencias se calculan de los archivos reales. Se conservan bins insuficientes como faltantes. Variables publicadas normalizadas min–max; no comparables en magnitud absoluta con CPU o bytes de ChronoGraph.

Los primeros12 bins sirven exclusivamente para calibración de escala/mediana/MASE. Los objetivos siguientes se pronostican con pesos fuente congelados y sin ajuste externo; las historias pueden incluir ese prefijo de calibración. No se evalúa GRU_graph porque no se descarga el grafo externo. La duración de24h limita fuertemente la evidencia: caso exploratorio de transferencia, no validación poblacional ni nuevo entrenamiento 70/10/20. El protocolo70/10/20 se mantiene en ChronoGraph; la transferencia externa tiene un diseño distinto explícito.

## Riesgos y decisiones

El test histórico es exploratorio por exposición previa. Las pruebas de causalidad cubren nuestro código y cortes, no certifican el preprocesamiento original del proveedor: el ámbito de normalización min–max de Alibaba y la fecha de disponibilidad de las aristas estáticas de ChronoGraph no están plenamente documentados. La nueva cohorte comparte período e infraestructura fuente, sus métricas no están cegadas por un tercero y la selección topológica no representa una población. El presupuesto compacto de PatchTST no equivale a sus resultados publicados. GRU-D es una adaptación compartida entre servicios, no reproducción clínica canónica. El kernel MA25 de DLinear excede L12 con padding replicado, como en la línea histórica. Las normales de destino usan datos previos pero el ajuste y selección no los usan. Los benchmarks de un portátil compartido no son latencias de producción, energía ni efectos sobre SLA. Se preservan fallos, errores elevados y resultados negativos. Ningún resultado se transmite o somete automáticamente.

# Enmienda de ejecución anterior al test

Fecha UTC: 2026-11-10T20:09:38.810229+00:00

El protocolo v1 se congeló el 10-11-2026 a las 20:02:30 UTC. Antes de producir ninguna predicción ni métrica de prueba de los nuevos servicios (búsqueda de NPZ en subdirectorios test vacía), se interrumpió únicamente el runner del estudio tras la época completa guardada. La sobrecarga de lanzamiento de kernels de la adaptación GRU-D motivó capturar su mismo cálculo forward/backward en CUDA Graphs. No se modifican arquitectura, actualización Adam, orden de muestras, semillas, máscaras, presupuesto ni criterios de selección.

Se preservan v1 y su motor en configs/protocolo_v1_original.yaml y docs/engine_v1_antes_optimizacion.py. results/cuda_replay_equivalence.json registra igualdad bit a bit de gradientes/pesos durante cuatro actualizaciones Adam y de inferencia. results/resume_test/PASSED.json verifica reanudación exacta para A3, PatchTST y GRUD. La nueva huella de código se congela como v2 antes del primer test. Los checkpoints previos, todos de entrenamiento/validación fuente, se reutilizan por equivalencia comprobada.

También se libera la captura de inferencia antes de reintentar un lote tras OOM. La ruta de OOM solo cambia el tamaño físico del lote; cualquier incidencia real quedará en progress.jsonl.

Antes de terminar el agregado externo y antes de evaluarlo, se verificó que los timestamps comienzan en cero y avanzan 60000ms en el objeto0. Se corrige el bin a floor(timestamp/1800000), intervalo [a,a+30min), disponible solo al cierre a+30min. La fórmula previa timestamp−1 habría mezclado límites y dejado el último bin incompleto. Los agregados por timestamp ya calculados permanecen válidos. No había predicciones externas generadas.

## Enmienda v3 de gestión de memoria

Fecha UTC: 2026-11-10T20:32:46.353536+00:00

Todavía no existían NPZ ni métricas de test destino. Se interrumpió el runner tras errores OOM en siete ajustes GRU-D de CPU, conservando checkpoints y todos los registros. La prueba de ciclo de vida reprodujo un crecimiento de134217728bytes por ajuste; liberar la captura sola no bastó. Se reutiliza ahora un único stream CUDA para calentamiento/captura y se libera explícitamente cada captura al terminar, también por excepción y al reducir lote. Seis ajustes consecutivos con validación por lotes128 mantienen exactamente158561792bytes vivos después de cada ajuste (rango0). Esto corrige acumulación de memoria sin ampliar el límiteGPU del estudio ni detener aplicaciones ajenas.

Se repitieron y pasaron pruebas de igualdad bit a bit de gradientes/pesos/inferencia y de reanudación para A3/PatchTST/GRUD. Se preserva v2, su motor y su selección; v3 incorpora los hashes de estas pruebas. Los siete ajustes fallidos se reintentan, sin filtrar por error predictivo. El ajuste GRU-D/CPU/semilla303 ya completado con microlote4 trasOOM se conserva según la regla previamente congelada; todos los tamaños usados constan por época. No se afirma que una ejecución interrumpida o un intento fallido sea una réplica completa.

La documentación oficial de PyTorch2.9 describe espacios cuBLAS persistentes por combinación de handle/stream: https://docs.pytorch.org/docs/2.9/notes/cuda.html#cublas-workspaces . El crecimiento observado y su eliminación al reutilizar el stream son coherentes con ese mecanismo; no se modificó CUBLAS_WORKSPACE_CONFIG ni el límiteGPU durante la corrección. La selección de los22 pares modelo/recurso mantuvo los mismos candidatos antes y después de v3.

Antes de cualquier evaluación externa, se ordena lexicográficamente la misma lista de32 servicios de Alibaba seleccionada por hash/disponibilidad, para que los grupos sintéticos de ocho cumplan la descripción congelada. No cambian miembros, valores ni tiempos; se regenera la serialización del NPZ y su manifiesto, conservando la selección y hash de procesamiento previos. No se consultaron errores destino.

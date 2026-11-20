# Matriz de cobertura de ambos idiomas

Las referencias de secciones son comunes a las dos versiones. Los nombres `fig:` y `tab:` son etiquetas LaTeX estables.

| Evidencia | Cuerpo del artículo | Tabla/figura o suplemento | Procedencia |
|---|---|---|---|
| I. Reproducción histórica ChronoGraph28, 8005 filas, 70/10/20, 3/20, L12/H30 | Métodos y primeros resultados | `tab:master`, `tab:historical-main`, `fig:historical` | `results/reproduccion/` y `revision_20261029/results/metricas_verificadas/` |
| E0–E5, GRU propia/llamadas, BiTGraph, GinAR, persistencia | Métodos y comparación histórica | `tab:historical-main`, `tab:historical` | `revision_20261029/results/comparaciones/resumen_perdidas.csv`, `resumen_global.csv` |
| II. C0–C3, V2, búsqueda, fallos de precisión | Métodos y resultado compacto histórico | `fig:evolution`, `tab:historical-main` | `revision_20261029/src/compact.py`, `results/nuevos_experimentos/selected_candidates.json`, `results/comparaciones/contrastes_dos_escalas.csv` |
| Incertidumbre histórica, intervalos absolutos/porcentuales y dependencia | Métricas, resultado histórico y discusión | `sec:historical-compact` | `revision_20261029/results/comparaciones/contrastes_dos_escalas.csv`, `contrastes_por_semilla.csv`, `contrastes_por_servicio.csv` |
| III. Quince familias, destino32, 10/20 principal y calibración | Métodos, comparación y discusión | `tab:primary`, `fig:models`, `fig:temporal` | `ampliacion_20261110/configs/protocolo_congelado.yaml`, `results/analysis/primary_summary.csv` |
| Ablaciones A0–A3, cuatro contrastes y sensibilidad por servicio | Métodos y resultados A3 | `fig:dlinear`, `fig:a3`, `fig:ablation`, `fig:seeds`, `fig:services`, `tab:contrasts`, `tab:block`, `tab:variation` | `ampliacion_20261110/src/models.py`, `results/analysis/primary_contrasts.json`, `posthoc_service_sensitivity.json`, `paired_by_service.csv` |
| IV. Datos completos y pérdidas dispersas, cortas, correlacionadas, prolongadas y MAR | Preparación y robustez | `fig:masks`, `fig:loss`, `tab:stress` | `ampliacion_20261110/results/analysis/by_condition.csv`, `mask_audits.csv` |
| IV. H30/60/120/180/360, L12/24/48 y 1553 objetivos comunes | Métodos y resultados de horizontes y ventanas | `fig:horizon`, `fig:history`, `tab:horizon`, `tab:history` (ocho familias) | `ampliacion_20261110/results/analysis/common_origin_metrics.csv.gz`, `common_origins.json` |
| IV. Alibaba, un día, 12+36, 14 modelos, persistencia | Métodos, resultados y amenazas | `fig:external`, `tab:external` | `ampliacion_20261110/data/manifests/`, `results/analysis/by_condition.csv` |
| V. Parámetros, tiempos, RAM y CUDA | Métodos, resultados y discusión | `tab:efficiency`, `tab:historical-bench`, `tab:memory-bench`, `fig:latency`, `fig:params` | `revision_20261029/results/comparaciones/costes_nueva_medicion.csv`, `ampliacion_20261110/experiments/benchmarks/`, `results/resource_summary.json` |
| Auditoría y reproducibilidad | Resultados, amenazas y reconstrucción | `tab:matrix` | Manifiestos NPZ/checkpoints, `audit.json`, `checkpoint_audit.json`, `FINAL_VERIFICATION.json` |
| Pregunta central, conclusiones y límites | Discusión, amenazas y conclusiones | `tab:rq` | Contrastes y fuentes de las filas anteriores |

Los ensayos de humo, interrupción y recuperación se describen como verificaciones de implementación. Los pilotos con particiones previas no se incorporan a medias de prueba. Las predicciones individuales, celdas por servicio y realizaciones, ensayos de latencia individuales y archivos originales permanecen en el repositorio experimental. Su omisión del PDF evita duplicar la evidencia, sin eliminar resultados del estudio.

# Experimentos y trazabilidad

La tabla `articulo_ieee/inventario_experimentos.csv` enumera las cinco fases. La matriz de cobertura e inventario son evidencia del artículo, no nuevas ejecuciones.

| Fase | Entrada | Código ejecutado | Configuración / semillas | Métricas | Activos del artículo |
|---|---|---|---|---|---|
| I: comparadores | chronograph_source | scripts/reproducir.py; src/baseline/reference/run.py | results/reproduccion; 101,202,303;20 máscaras | results/reproduccion/*.csv | es_historico_positivo; es_historico |
| I: E0–E5 | chronograph_source | scripts/ablaciones.py; src/improved/models.py | results/mejoras;3 semillas;20 máscaras | results/mejoras/*.csv | es_historical; es_historico |
| II: C0–C3/V2 | chronograph_source | revision_20261029/scripts/run_v2.py; src/compact.py | protocolo/selección/frozen_v2;3 semillas;20 máscaras | revision_20261029/results/comparaciones | es_historico_positivo; es_evolucion |
| III: L12/H1 | source ajuste; holdout evaluación | ampliacion_20261110/scripts/run_experiments.py | protocolo_congelado.yaml;10 semillas;20 máscaras;15 familias | results/analysis/primary_summary.csv;primary_contrasts.json | es_primary;es_contrasts;es_06_models;es_10_seed_variability;es_11_ablations;es_15_services |
| IV: ausencia | source/holdout | run_experiments.py;src/core.py | completo y tres mecanismos secundarios;20 máscaras | by_condition.csv;mask_audits.csv | es_07_loss;es_05_missingness |
| IV: horizontes/ventanas | source/holdout | run_experiments.py | 6 extensiones;3 semillas;5 máscaras;1553 objetivos comunes | common_origin_metrics.csv.gz | es_08_horizon;es_09_history |
| IV: Alibaba | alibaba_transfer | run_experiments.py --stage external | fuente L12/H1 congelada;14 familias;12+36 bins | summary_all_conditions.csv | es_external;es_12_external |
| V: coste | entradas residentes fuente/destino | scripts/benchmark.py;revision_20261029/scripts/benchmark.py | B1 ampliado separado de B128 histórico | experiments/benchmarks/latency.csv;training.csv | es_efficiency;es_13_latency;es_14_parameters |

Las rutas abreviadas de resultados ampliados son relativas a ampliacion_20261110. Las selecciones y frozen.json originales permiten recuperar parámetros concretos por modelo/recurso/configuración. Las semillas principales son 101,202,303,404,505,606,707,808,909,1010 y las máscaras 90001+100003*r, r=0..19; secundarios usan las primeras tres semillas y cinco máscaras. No se reentrenó al preparar esta entrega.

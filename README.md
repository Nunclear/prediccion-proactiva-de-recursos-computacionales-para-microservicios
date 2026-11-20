# Pronóstico de recursos de microservicios con telemetría incompleta: descomposición, modelos compactos y transferencia calibrada

Proyecto experimental de Carlos Augusto Mullisaca Condori y Jhon Alex Gutiérrez Morales, Universidad Peruana Unión, Facultad de Ingeniería y Arquitectura, Escuela Profesional de Ingeniería de Sistemas, Juliaca, Perú. Correos: carlos.mullisaca@upeu.edu.pe y alex.gutierrez@upeu.edu.pe. Autor correspondiente pendiente. Sin financiación externa ni conflictos declarados.

## Pregunta y objetivos

Se estudia cómo la descomposición temporal, los descriptores de ausencia y la capacidad afectan al error y coste del pronóstico de CPU/memoria con telemetría incompleta. Se comparan históricos E0–E5/C0–C3/V2, quince familias sobre 32 servicios adicionales, horizontes/ventanas, un día Alibaba y coste CPU/CUDA. No se evalúa un controlador de autoescalado ni ahorro energético.

## Estructura y modelos

Las rutas originales se mantienen para no romper imports: `src/baseline/reference` (comparadores históricos), `src/improved` (E), `revision_20261029` (C/V2), `ampliacion_20261110` (A0–A3, DLinear, NLinear, E5, GRU propia/grafo, GRU-D adaptada, PatchTST pequeño y referencias Persistence/MovingAverage/SES/Ridge). `articulo_ieee` contiene el manuscrito español, fuentes, 17 figuras/17 tablas y 32 referencias. `data/` documenta el ZIP independiente y sus hashes, sin datasets en Git. `docs/` explica metodología, modelos, experimentos, resultados y reproducción.

## Instalación, datos y ejecución

Requiere Python3.12 y GPU NVIDIA/CUDA para repetir los runners; la inspección de datos y resúmenes funciona en CPU. El hardware original fue RTX3050Laptop4GB/i5-11300H/24GBRAM. Siga [reproducibilidad](docs/reproducibilidad.md) para venv, dependencias fijadas, Docker autocontenido, datos, comandos de entrenamiento/evaluación y gráficos. Instale el ZIP con `python scripts/preparar_datos.py /ruta/DATA_ARTICULO_EMPIRICO.zip`; luego `python scripts/verificar_entrega.py` y `python scripts/graficar_resumen.py`. Estas dos últimas tareas son ligeras y no entrenan.

## Configuración y resultados

ChronoGraph se divide por filas en 70/10/20; destino usa calibración de prefijo sin ajuste de pesos. Principal L12/H1, diez semillas neuronales y veinte máscaras; cuatro contrastes con corrección Holm. Alibaba usa doce bins de calibración y treinta y seis objetivos, no 70/10/20. Consulte [experimentos](docs/experimentos.md), [modelos](docs/modelos.md) y [resultados](docs/resultados.md).

A3 tiene 15 parámetros por recurso, DLinear 26 y E5 74. A3 mejora memoria frente a E5 en 0.02523 MASE (p Holm 0.0078125; 10/10 semillas), pero solo 10/32 servicios y sin mejora general frente a DLinear. PatchTST pequeño/NLinear tienen menores medias principales; persistencia lidera el caso externo. Los CSV originales se conservan sin redondear.

## Integridad, disponibilidad y límites

Este repositorio se preparó con experimentos ya terminados: no se generaron resultados nuevos. Se incluyen configuraciones, métricas resumidas, manifiestos, auditorías y código real; se excluyen pesos, predicciones, datasets y métricas expandidas grandes. Rehacer la auditoría completa requiere regenerarlos o solicitar el archivo local de reproducción. No se afirma una nueva réplica limpia. Los servicios comparten período/infraestructura, las pérdidas son sintéticas y Alibaba es un solo día con unidades normalizadas. [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) detalla licencias y obtención de dos dependencias históricas sin licencia de redistribución confirmada.

Artículo: [PDF IEEE español](articulo_ieee/articulo_ieee_es.pdf), con 32 referencias, 17 figuras y 17 tablas. Los datos procesados se distribuyen por separado; no existe una descarga pública ni un DOI asignado. Las declaraciones de autoría y correspondencia constan en el manuscrito.

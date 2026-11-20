# Reproducción y alcance de la entrega

## Entorno

La ejecución original usó Python3.12, PyTorch2.9.0+cu128, CUDA, RTX3050Laptop4GB, i5-11300H,24GBRAM, dos hilos CPU, float32. El manifiesto completo está en ampliacion_20261110/configs/environment_pip_freeze.txt. El nuevo Dockerfile autocontenido instala esas dependencias base; es una receta de reconstrucción, no una imagen que se haya vuelto a construir durante esta entrega. Requiere Docker y NVIDIA Container Toolkit funcionales.

```bash
git clone https://github.com/Nunclear/prediccion-proactiva-de-recursos-computacionales-para-microservicios.git
cd prediccion-proactiva-de-recursos-computacionales-para-microservicios
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r ampliacion_20261110/requirements.txt
python scripts/preparar_datos.py /ruta/DATA_ARTICULO_EMPIRICO.zip
python scripts/verificar_entrega.py
python scripts/graficar_resumen.py
```

La preparación verifica SHA256 y crea rutas originales de entrada; no reimputa, reparte ni modifica los datos. `graficar_resumen.py` regenera una comparación principal desde primary_summary.csv, sin entrenar. Las figuras y tablas originales se conservan en articulo_ieee.

## Entrenamientos históricos (opcionales y costosos; no ejecutados al empaquetar)

Antes de reproducir BiTGraph/GinAR: `python scripts/obtener_terceros_historicos.py`; requiere red y `patch`, descarga revisiones fijas a rutas ignoradas y verifica hashes. No se redistribuyen sus fuentes sin licencia explícita confirmada. DLinear histórico utiliza la copia MIT conservada; el control local se incluye.

Los marcadores/configuraciones preservados son evidencia y provocan reanudación. Para un ajuste desde cero use OTRO checkout/directorio de ejecución; archive allí los resultados históricos y `ampliacion_20261110/experiments` antes de iniciar. No elimine la evidencia original. En el checkout de trabajo puede mover esas carpetas fuera del árbol (copia de seguridad), volver a ejecutar preparar_datos.py y luego los comandos siguientes. Los protocolos fuente se congelan con hashes del código actual; no reutilice marcadores de finalización como prueba de una ejecución nueva.

```bash
python scripts/reproducir.py
python scripts/ablaciones.py
python revision_20261029/scripts/run_v2.py
python revision_20261029/scripts/analyze.py
python ampliacion_20261110/scripts/run_experiments.py --stage all
python ampliacion_20261110/scripts/analyze_results.py
python ampliacion_20261110/scripts/benchmark.py
python ampliacion_20261110/scripts/make_figures.py
python ampliacion_20261110/scripts/export_tables.py
```

`--stage all` incluye freeze, principal, secundario y externo. `analyze_results.py` exige predicciones/checkpoints, no se ejecuta solo con resúmenes. `make_figures.py` y `export_tables.py` originales necesitan metrics_all.csv.gz reconstruido por el análisis completo. El script ligero de resumen sí funciona con esta publicación. Los comandos son extraídos del código; no se declara una ejecución limpia integral verificada en esta entrega.

## Docker

```bash
docker build -t articulo-empirico:cuda .
docker run --rm --gpus all -it --shm-size=2g -v "$PWD:/workspace" articulo-empirico:cuda
```

Dentro se pueden ejecutar los comandos anteriores. Los scripts run_docker.sh históricos apuntan al contenedor original y se conservan como procedencia; utilice esta receta para otra máquina. Los preprocesadores prepare_chrono.py/aggregate_alibaba.py requieren descargas originales fuera de este paquete; el ZIP procesado evita repetir los 17.89GB externos. El artículo PDF se recompila con `pdflatex main_es.tex`, `bibtex main_es`, `pdflatex main_es.tex` dos veces desde articulo_ieee (TeXLive con IEEEtran/paquetes habituales).

## Licencias y límites

Los datos se entregan localmente y no se publican en GitHub. PatchTST conserva Apache2.0; LTSF-Linear conserva MIT. No se asigna licencia nueva al código de los autores sin una decisión explícita; publicación pública no equivale a licencia abierta. La normalización global del proveedor Alibaba y la fecha de disponibilidad de aristas no se certifican; no hay ahorro energético/SLA ni validación poblacional independiente. El repositorio distribuye código y métricas resumidas; el archivo completo de reproducción se conserva por separado.

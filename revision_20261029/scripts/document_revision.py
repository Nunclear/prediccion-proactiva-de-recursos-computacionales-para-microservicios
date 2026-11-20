"""Supporting documentation extracted from the generated article and execution metadata."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'articulo';DOC=ROOT/'docs'
def section(text,start,end):return text.split(start,1)[1].split(end,1)[0].strip()
def write(name,text):(DOC/name).write_text(text)
def main():
 article=(ART/'ARTICULO_EMPIRICO_MEJORADO.md').read_text()
 architecture=section(article,'### 3.5 Modelo propuesto y variante compacta','### 3.6 Configuración experimental')
 experiment=section(article,'### 3.6 Configuración experimental','### 3.7 Métricas y análisis estadístico')
 statistics=section(article,'### 3.7 Métricas y análisis estadístico','### 3.8 Reproducibilidad y hardware')
 limitations=section(article,'## 6. Amenazas a la validez y limitaciones','## 7. Conclusiones')
 result=section(article,'### 4.7 Evaluación de V2','### 4.8 Estado de la validación independiente')
 external=section(article,'### 4.8 Estado de la validación independiente','## 5. Discusión')
 prefix='[Artículo mejorado](../articulo/ARTICULO_EMPIRICO_MEJORADO.md) · [Auditoría inicial](AUDITORIA_ARTICULO.md)\n\n'
 write('MEJORAS_MODELO.md','# Mejora y evaluación real del modelo\n\n'+prefix+architecture+'\n\n## Protocolo ejecutado\n\n'+experiment+'\n\n## Resultado conservado\n\n'+result+'\n\nEvidencia: [selección corta](../results/nuevos_experimentos/selected_candidates.json), [congelación previa a prueba](../results/nuevos_experimentos/frozen_v2.json), [estado final](../results/nuevos_experimentos/COMPLETED.json). Los checkpoints del smoke test están separados y no se reutilizaron como pesos de entrenamiento final.\n')
 write('METODOLOGIA_ESTADISTICA.md','# Metodología estadística y corrección de unidades\n\n'+prefix+statistics+'\n\nLa comprobación automatizada reprodujo los intervalos originales con semilla 91001 y confirmó que sus límites eran diferencias MASE absolutas. Después calculó intervalos en porcentaje desde la distribución de reducciones emparejadas. No se multiplicaron simplemente los límites anteriores por cien. La tabla de sensibilidad usa una razón entre medias, distinta de la media de razones del análisis principal.\n\nArchivos: [contrastes en dos escalas](../results/comparaciones/contrastes_dos_escalas.csv), [semillas](../results/comparaciones/contrastes_por_semilla.csv), [servicios](../results/comparaciones/contrastes_por_servicio.csv), [validación de predicciones nuevas](../results/comparaciones/validacion.json).\n')
 write('LIMITACIONES.md','# Amenazas a la validez\n\n'+prefix+limitations+'\n\n## Datos externos\n\n'+external+'\n')
 rubric=[
 ('Planteamiento del problema',3,1.7,2.7,'Pregunta, objetivos, hipótesis y alcance operativo explícitos.'),
 ('Marco teórico y antecedentes',3,1.5,2.6,'Nueve referencias pertinentes, tabla crítica y metadatos editoriales comprobados.'),
 ('Metodología experimental',5,4.3,4.2,'Ecuaciones según código, V2 y 24 ajustes nuevos; persiste exposición al test y búsqueda histórica desigual.'),
 ('Resultados y análisis estadístico',5,4.2,4.3,'Intervalos en dos unidades y sensibilidad a tres inicializaciones; falta validación externa.'),
 ('Discusión y conclusiones',2,1.2,1.8,'Resultados negativos, parsimonia y límites de explicaciones mecanísticas.'),
 ('Redacción y presentación',2,1.6,1.9,'Artículo autónomo, tablas generadas y trece figuras en PNG/SVG.')]
 initial=sum(x[2] for x in rubric);final=sum(x[3] for x in rubric);assert round(initial,1)==14.5 and round(final,1)==17.5
 lines=['| Criterio | Máximo | Inicial | Final estimado | Evidencia |','|---|---:|---:|---:|---|']
 lines.extend(f'| {a} | {b} | {c:.1f} | {d:.1f} | {reason} |' for a,b,c,d,reason in rubric)
 write('EVALUACION_FINAL.md','# Evaluación crítica final\n\n'+prefix+'Esta es una autoevaluación orientativa según la rúbrica solicitada. No constituye una calificación docente, una revisión editorial externa ni una probabilidad de aceptación. No se penaliza que V2 dé resultados negativos; se evalúa la validez con la que se presentan.\n\n'+'\n'.join(lines)+f'\n| Total | 20 | {initial:.1f} | {final:.1f} | |\n\n## Debilidades restantes\n\nLa prueba continúa conocida; no hay cohorte externa evaluada. Solo se dispone de tres semillas y veinte máscaras sobre la misma infraestructura. Las búsquedas históricas no tienen igual presupuesto. El coste mide forward con entradas residentes, no un sistema completo. La preparación para publicación exige fortalecer generalización y medición, además de revisión por especialistas externos. La evaluación no sube automáticamente a 18–20 por aumentar extensión o proponer una versión V2.\n\n## Reproducibilidad y preparación editorial\n\nReproducibilidad local alta: fuentes, datos de cohorte, configuraciones, predicciones, checkpoints, semillas, tablas, scripts y manifiestos se conservan. La igualdad numérica fue demostrada en el entorno registrado; no se garantiza identidad bit a bit en toda combinación de GPU y framework. Para regenerar desde cero se necesita la imagen preparada o reconstruir el entorno documentado. La selección compacta y su fracaso parcial son auditables. El artículo puede presentarse como trabajo universitario exploratorio sujeto a la plantilla del curso; una eventual publicación necesita evaluación independiente y mejor cobertura de inicializaciones, sin prometer aceptación.\n\n## Experimentos que aún se requieren\n\nCongelar un protocolo externo antes de descargar y evaluar sus objetivos; armonizar CPU/memoria y granularidad; aumentar semillas; evaluar ventanas/horizontes y comportamiento ante cambios; contrastar pérdidas conjuntas de colectores; repetir coste con otros lotes y medir el flujo completo si se desean afirmaciones operativas.\n')
 (DOC/'evaluacion_rubrica.json').write_text(json.dumps(dict(initial=initial,final_estimated=final,max=20,not_external_review=True,criteria=rubric),indent=2,ensure_ascii=False))
 changes='''# Cambios verificables de la revisión

[Artículo mejorado](ARTICULO_EMPIRICO_MEJORADO.md) · [Original preservado](ARTICULO_EMPIRICO_ORIGINAL.md)

| Dimensión | Versión previa | Revisión |
|---|---|---|
| Problema | Introducción breve | Pregunta, objetivos e hipótesis exploratorias |
| Literatura | Ocho referencias abreviadas | Nueve fuentes pertinentes y tabla crítica; metadatos registrados |
| Matemáticas | Salida y EMA resumidas | Imputación, máscaras, edad, MA, EMA, pérdida y métricas con dimensiones |
| Incertidumbre | Límites absolutos junto a porcentaje | Límites separados en MASE y porcentaje; sensibilidad cruzada a semillas |
| Modelo nuevo | No existía V2 | Cuatro familias de 26–28 parámetros y selección por validación |
| Experimentos | 30 referencias y 30 ablaciones originales | 48 selecciones cortas y 24 ajustes completos adicionales |
| Resultado negativo | E5 empeora CPU | Se mantiene y se añade que V2 no conserva la precisión de E5 |
| Coste | Veinte forward por modelo y recurso | Cien forward por proceso, 28 procesos, 2800 muestras; RAM y VRAM con alcance explícito |
| Figuras | Diez PNG | Originales conservados y trece figuras nuevas en PNG y SVG |
| Independencia | Test conocido | Sigue explícitamente exploratorio; búsqueda externa registrada sin inventar una evaluación |

Los CSV históricos, checkpoints y pronósticos anteriores no se sobrescribieron. Las nuevas tablas se generan desde métricas autenticadas; los ejemplos gráficos usan tramos fijados. La revisión no cambia cifras para favorecer E5 o V2.
'''
 (ART/'RESUMEN_CAMBIOS.md').write_text(changes)
 write('GUIA_REPRODUCCION.md','''# Regeneración de la revisión

La revisión se ubica dentro del proyecto histórico. Los scripts toman como BASE el directorio padre y conservan sus resultados. La entrega completa mantiene exactamente esta estructura:

~~~
src/baseline/reference/
src/improved/
results/reproduccion/
results/mejoras/
results/estadisticas/
revision_20261029/
  scripts/
  src/compact.py
  tests/
  results/
  articulo/ARTICULO_EMPIRICO_MEJORADO.md
~~~

En la imagen Docker ya preparada, con la raíz del proyecto montada en /workspace, ejecutar:

~~~bash
docker --context default exec dlinear20261029-research-1 sh -c 'cd /workspace/revision_20261029 && sh scripts/pipeline.sh'
~~~

Para entrenamientos nuevos, copiar primero el proyecto a un directorio independiente y vaciar únicamente las salidas NUEVAS de esa copia; conservar la cohorte verificada y el histórico. El runner restaura checkpoints propios de una ejecución interrumpida y comprueba sus fuentes. Mantener el código sin cambios durante cada ejecución. Un marcador COMPLETED permite al pipeline reutilizar experimentos terminados; no se describe esa reutilización como un entrenamiento nuevo desde cero.

La imagen existente es dlinear-research:20261029, derivada de chronograph:cuda. Los Dockerfiles y versiones fijadas se conservan en la raíz histórica. No se instalan dependencias nuevas ni se cambian drivers en esta revisión. Reconstruir desde cero en otra máquina requiere seguir ese entorno y volver a comprobar CUDA. El JSON completo de 1,786 GB no se duplica en la entrega; la cohorte NPZ de 28 servicios y su hash sí se conserva, junto con la procedencia y auditoría del archivo original.

El monitor está en ../src/monitor.py; usar --root /workspace/revision_20261029. Su archivo STOP_MONITOR finaliza el proceso. Los CSV son telemetría del sistema y no una atribución exclusiva por modelo. Para repetir benchmark sin reutilizar mediciones, retirar únicamente su subcarpeta de resultados en una copia nueva. No confundir cambiar semillas sobre el mismo test con validar en otro dataset.
''')
 (ROOT/'README.md').write_text('# Revisión científica DLinear-EMA-Mask\n\n[Artículo mejorado](articulo/ARTICULO_EMPIRICO_MEJORADO.md) · [Original](articulo/ARTICULO_EMPIRICO_ORIGINAL.md) · [Cambios](articulo/RESUMEN_CAMBIOS.md)\n\n[Auditoría](docs/AUDITORIA_ARTICULO.md) · [Metodología estadística](docs/METODOLOGIA_ESTADISTICA.md) · [Modelo y V2](docs/MEJORAS_MODELO.md) · [Evaluación crítica](docs/EVALUACION_FINAL.md) · [Guía](docs/GUIA_REPRODUCCION.md)\n\nLas cifras proceden de predicciones reales y métricas auditadas. V2 reduce parámetros pero no mejora ambos recursos. No se ejecutó evaluación externa; el estudio continúa exploratorio. Las métricas, checkpoints, código y figuras están incluidos para auditoría.\n')
 print('SUPPORTING_DOCUMENTATION_GENERATED')
if __name__=='__main__':main()

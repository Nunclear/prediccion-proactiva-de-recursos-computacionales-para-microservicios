# Manuscrito en formato IEEE

Este proyecto usa `IEEEtran` en modo `journal`, con dos columnas, papel carta y bibliografía IEEE por orden de primera cita. Es la edición general de revista del mismo artículo, con sus versiones en inglés y español.

| Versión | Archivo principal | PDF |
|---|---|---|
| Inglés | `main.tex` | Compilar desde las fuentes |
| Español | `main_es.tex` | `articulo_ieee_es.pdf` |

## Compilar

Las fuentes se pueden importar en Overleaf. Selecciona **pdfLaTeX** y el archivo principal del idioma que quieras compilar. Desde esta carpeta, con TeX Live y `latexmk`:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error main_es.tex
```

Los comandos generan `main.pdf` y `main_es.pdf`, respectivamente. La clase `IEEEtran.cls` y el estilo `IEEEtran.bst` se incluyen con sus archivos de procedencia. La compilación usa las figuras y tablas suministradas en el proyecto y no requiere acceso al repositorio experimental.

## Evidencia y formato

Las dos versiones contienen 32 referencias citadas, 17 figuras y 17 tablas. `trazabilidad_resultados.csv`, `inventario_experimentos.csv` y `matriz_cobertura.md` documentan los resultados. `verificacion_ieee.json` registra la integridad de esta edición.

Esta edición usa el formato general de revista. Si se elige una publicación concreta, sus instrucciones pueden requerir otro estilo o una extensión distinta. [Plantillas oficiales de IEEE para revistas](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/authoring-tools-and-templates/tools-for-ieee-authors/ieee-article-templates/).

Archivos de clase y bibliografía distribuidos por [CTAN, IEEEtran](https://ctan.org/pkg/ieeetran). Se conservaron los avisos originales y se registraron los hashes de los archivos en `IEEEtran-SHA256.txt`.

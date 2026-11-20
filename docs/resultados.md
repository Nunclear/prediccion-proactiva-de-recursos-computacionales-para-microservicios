# Resultados preservados

Artículo IEEE en español de 23 páginas, 32 referencias, 17 figuras y 17 tablas; SHA256 b06bbec6f3ef7eb78c7fc5aa064c52494d85c9ef334b75cb1196a4c5de251eca.

| Modelo | MASE CPU principal | MASE memoria principal |
|---|---:|---:|
| DLinear | 1.149337 | 1.240190 |
| E5 | 1.145386 | 1.265561 |
| A0 | 1.156231 | 1.233361 |
| A3 | 1.160235 | 1.240333 |
| NLinear | 1.046345 | 1.219697 |
| PatchTST pequeño | 1.041886 | 1.217026 |

Son medias sobre seis condiciones con pérdidas, veinte máscaras y servicios fijos. Los CSV conservan precisión original. A3 frente a E5 en memoria: diferencia MASE positiva 0.02523, 10/10 semillas favorables, p Holm 0.0078125, intervalo condicional [0.01312,0.03981], mejora solo en 10/32 servicios. No hay mejora general de A3 frente a DLinear. PatchTST/NLinear no tienen prueba de superioridad mutua preespecificada. Persistencia obtiene las menores medias externas en este día Alibaba (2.3795/7.5150), un caso corto exploratorio.

La auditoría previa reconstruyó 3097600 filas desde 9680 archivos de predicciones, con diferencia máxima 3.72529e-9; ver results/analysis/audit.json y FINAL_VERIFICATION.json. Aquí se distribuyen resúmenes y manifiestos, no los pesos, las predicciones ni el CSV expandido de 3.1 millones de filas. Por tanto, comprobar esos hashes contra todos los originales requiere el archivo local de reproducción o nuevos ajustes explícitos. No se confunde una revisión estática con una nueva réplica experimental.

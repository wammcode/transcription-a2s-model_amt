# Notas de conversión y validación

Fecha de conversión: 2026-10-14.

Documento: **SCORING TIME INTERVALS USING NON-HIERARCHICAL TRANSFORMER FOR AUTOMATIC PIANO TRANSCRIPTION**, Yujia Yan y Zhiyao Duan.

Fuente suministrada: `paper_audio_to_midi_base/2404.09466v6.pdf`, 8 páginas. La identificación impresa en el margen de la primera página es `arXiv:2404.09466v6 [cs.SD] 9 Nov 2024`. La conversión corresponde exclusivamente a ese archivo.

La copia en [original/paper.pdf](../original/paper.pdf) es idéntica byte a byte al archivo suministrado. SHA-256: `fde97485540b069c2ff0737cc8985aa3e9da237c491130a6b753679f883a1e70`.

## Archivos y fuente de trabajo

| Archivo o carpeta | Contenido |
|---|---|
| [paper.md](../paper.md) | Representación textual principal: título, autores, afiliación, correos, abstract, secciones 1–8, subsecciones, ecuaciones, definiciones, demostraciones, tablas completas, figuras y captions, referencias, apéndice A y notas al pie. |
| [references.bib](../references.bib) | 33 entradas BibTeX, con claves `ref01`–`ref33` correspondientes a las referencias [1]–[33]. |
| `figures/` | Dos imágenes PNG obtenidas directamente del contenido gráfico del PDF. |
| [tables/table_01.md](../tables/table_01.md) | Table 1: Model Specification. |
| [tables/table_02.md](../tables/table_02.md) | Table 2: resultados en Maestro, separados por Notes, Sustain Pedals y Soft Pedals. |
| [tables/table_03.md](../tables/table_03.md) | Table 3: resultados en MAPS/SMD y comparación Between Ground Truths. |
| [validation_report.json](validation_report.json) | Registro de las comprobaciones realizadas. |

Para las siguientes tareas de análisis, resumen, comparación o modificación, `paper.md` debe utilizarse como fuente textual principal. Las tablas independientes reproducen las tablas incluidas en ese archivo. Si se modifica una tabla posteriormente, deben actualizarse también ambas representaciones.

## Reconstrucción del orden y la estructura

Se revisaron visualmente las ocho páginas y se extrajo texto por regiones de columna para evitar mezclar la columna izquierda con la derecha. Se eliminaron los saltos de línea y la partición tipográfica de palabras que pertenecían al mismo párrafo. Se conservan las palabras compuestas, la terminología de los autores y las erratas identificables del original.

| Página del PDF | Contenido y continuidad comprobados |
|---:|---|
| 1 | Título, autores, afiliación, abstract y sección 1. La lista de categorías continúa de la columna izquierda a la derecha. Se preservaron las notas 1 y 2 y el texto de licencia/atribución. |
| 2 | Secciones 2.1, 2.2, 3, 3.1 y comienzo de 3.2; ecuaciones (1)–(6); nota 3. |
| 3 | Continuación de 3.2; Definitions 3.1 y 3.2; Lemma 3.1 y su demostración; Theorem 3.2, ecuación (7) y demostración; las dos observaciones, con la segunda en la columna derecha; sección 3.3 y comienzo de 4; nota 4. |
| 4 | Figure 1; continuación del párrafo inicial de 4; sección 4.1, cuyo último párrafo continúa entre columnas; sección 4.2, Figure 2 y ecuación (8); comienzo de 4.3. |
| 5 | Continuación de 4.3; 4.4; 5.1–5.4; Table 1; notas 5 y 6. El último párrafo continúa en la página 6. |
| 6 | Table 2 ocupa la parte superior de ambas columnas. Continúa 5.4 en la columna izquierda; Table 3 y sección 6 en la derecha. Se preservaron las notas 7–9; la nota 7 continúa entre columnas. |
| 7 | Sección 7; sección 8, con referencias [1]–[10] en la izquierda y [11]–[23] en la derecha. |
| 8 | Referencias [24]–[33] y apéndice A, Changes from Previous Versions. |

Las figuras y las tablas se insertaron junto a su discusión o primera referencia, completando antes los párrafos partidos por la maquetación del PDF. Este desplazamiento de elementos flotantes no modifica el orden de la argumentación. Las nueve notas al pie se reúnen al final mediante la sintaxis Markdown `[^n]`, conservando sus números y puntos de llamada. El texto de licencia/atribución de la página 1 también se conserva al final. Los comentarios HTML de `paper.md` indican las páginas y continuidades pertinentes.

## Ecuaciones y símbolos

Se reconstruyeron en LaTeX las ocho ecuaciones numeradas (1)–(8) y las dos expresiones mostradas sin numeración, correspondientes a Definition 3.2 y a la demostración de Lemma 3.1. Las expresiones matemáticas dentro del texto, los índices, los superíndices, las dimensiones, las desigualdades y los parámetros de distribuciones también se conservaron.

Se comprobaron visualmente, entre otros elementos, los límites de las sumas de (1) y (3), el escalado `$|j-i|/\sqrt{D}$` y el término diagonal de (5), las condiciones de rango del teorema, las expresiones con primas en las secuencias de baja resolución y el factor `$\sqrt{2/B}$` de (8). Se conservaron las desigualdades impresas de Definition 3.1 y las condiciones originales del teorema sin reformularlas.

La notación caligráfica de los conjuntos/entrada, la negrita de los vectores y los operadores se representan con comandos LaTeX. Las ecuaciones numeradas tienen anclas `eq-1`–`eq-8`; las definiciones, el lema, el teorema, las figuras, las tablas y las entradas bibliográficas también cuentan con anclas explícitas.

KaTeX renderizó correctamente las 10 fórmulas de bloque y 195 ocurrencias de fórmulas en línea en el conjunto formado por `paper.md` y las tres copias de tablas. Ese recuento incluye repeticiones de encabezados y tablas; no representa 195 fórmulas distintas. La verificación de sintaxis complementa la comparación visual con el PDF.

## Tablas y valores

Las tablas completas están incluidas en `paper.md` y disponibles también como archivos independientes. Se mantuvieron todos los valores, su precisión decimal, los conteos de parámetros, los guiones de celdas sin resultado y las llamadas a las notas 8 y 9. No se redondearon valores.

| Tabla | Comprobación contra el PDF |
|---:|---|
| 1 | Las secuencias numéricas de sus 12 filas de configuración coinciden con las extraídas directamente de la tabla del PDF. |
| 2 | Coinciden las 14 filas, los 14 conteos de parámetros, los 147 valores métricos y las 21 celdas con guion, en su orden original. |
| 3 | Coinciden las 16 filas y sus 96 valores métricos, incluida la fila Between Ground Truths y sus valores `100`. |

Para representar los encabezados agrupados de Tables 2 y 3 en Markdown, se repite el nombre del grupo en cada columna de métrica. En Table 2 se separan las tres categorías de eventos; en Table 3 se repiten los nombres de métodos que en el PDF abarcan varias filas. Table 1 no tenía encabezados de columna: `Specification` y `Value` son etiquetas de navegación añadidas y señaladas en un comentario de conversión.

Los resaltados en negrita de Table 2 se conservaron según el PDF, sin recalcular qué valores deberían resaltarse. Entre ellos aparecen `93.76` para Activation P de Ours w/o incomplete events y `92.59` para Note w/ Offset & Vel. F1 de Ours with MLP kqb mapping. La conservación de estos resaltados no implica una interpretación de su criterio.

La cita `[24]` asociada a Onsets &Frames en Table 3 se conserva tal como está impresa.

## Figuras y captions

Las figuras son contenido vectorial del PDF, no imágenes raster independientes incrustadas. Se exportaron mediante rasterización de sus regiones originales a 300 dpi:

| Figura | Página | Archivo | Tamaño |
|---:|---:|---|---|
| 1 | 4 | [figure_01_system_overview.png](../figures/figure_01_system_overview.png) | 876 × 1188 píxeles |
| 2 | 4 | [figure_02_transformer_encoder.png](../figures/figure_02_transformer_encoder.png) | 988 × 501 píxeles |

Se inspeccionaron ambas exportaciones: etiquetas, flechas, símbolos y los dos subpaneles de Figure 2 están presentes y no se observó contenido recortado. Los captions originales se transcribieron por separado, conservando también `(a) Transformer Block.` y `(b) Encoder Layers`. Los textos alternativos de las imágenes son descripciones de accesibilidad añadidas durante la conversión, no captions ni afirmaciones nuevas de los autores.

## Bibliografía

Las 33 referencias se conservan completas en `paper.md`. Se cotejaron con regiones separadas de ambas columnas del PDF; las comparaciones normalizadas coinciden en las 33 entradas. Esta comparación ignora espacios y puntuación, por lo que se complementó con revisión visual de la bibliografía.

`references.bib` se generó exclusivamente con los datos impresos. Se conservaron las iniciales de autores, títulos, nombres de publicaciones, años, volúmenes, números, páginas y URL disponibles. Se verificaron los títulos, autores, publicaciones y años tras analizar el archivo con un lector BibTeX. No se añadieron DOI, nombres completos de autores ni metadatos externos.

Los tipos BibTeX reflejan la información que puede representarse con confianza: publicaciones periódicas como `article`, actas explícitas como `inproceedings` y referencias sin datos suficientes de tipo/publicación como `misc`. La referencia [28] conserva el nombre de conferencia mediante `howpublished`. Las referencias [30], [32] y [33] tienen fechas de consulta, pero no un año de publicación explícito: estas fechas están en `note` y no se usaron como `year`. Las referencias [32] y [33] no imprimen autores; esos campos se omiten. La referencia [25] no imprime publicación o institución; no se completaron esos campos.

## Ausencias del original y decisiones de preservación

No se encontraron elementos ilegibles que requieran un marcador `TODO` de recuperación. No hay ecuaciones, tablas o figuras pendientes de reconstrucción.

El PDF no incluye una lista de keywords. Esto se declara expresamente como anotación de conversión en la cabecera; no se han inferido palabras clave. El material suplementario de [2] que se cita en el texto no forma parte de este PDF. El apéndice A presente en el documento sí está incluido íntegramente.

Se preservaron las grafías `structrued` (nota 2), `embeding` (Table 1), `FNN` (Table 1), `Groudtruth` (Table 3) y `piece-depedent` (sección 5.4). Se mantuvieron también las variantes de nombres y capitalización tal como aparecen, por ejemplo `Maestro`, `MAESTRO`, `ENSTDkCl/MUS` y `ENST-DkCL`.

La advertencia del abstract **See appendix for post-camera-ready updates.** y el texto del apéndice sobre SMD version 2 y los eventos CC están preservados. No se sustituyeron sus resultados por cifras de otra versión.

Las explicaciones de este archivo, los metadatos de conversión, las etiquetas añadidas de Table 1 y las descripciones de accesibilidad están separados del contenido original. La tarea realizada fue de preservación y validación; no se ha añadido una evaluación científica de los resultados o las demostraciones.

## Resultado de la validación

Se comprobaron las 22 entradas de secciones/subsecciones numeradas y apéndice, las 8 ecuaciones numeradas, las 2 expresiones de bloque sin numeración, las 3 tablas, las 2 figuras, los 5 captions, las 33 referencias y las 9 notas al pie. La revisión visual cubrió las ocho páginas completas y las dos imágenes exportadas. Los párrafos partidos entre columnas o páginas se comprobaron expresamente.

Una comprobación adicional encontró en `paper.md` los 273 fragmentos interiores de prosa evaluados directamente a partir de las fuentes de texto del PDF, excluyendo regiones gráficas/tabulares y evitando transiciones a símbolos matemáticos. Ninguno de esos fragmentos quedó sin correspondencia. Esta comprobación complementa la revisión de párrafos y no certifica por sí sola cada símbolo o el orden de lectura.

Un analizador Markdown reconoció correctamente las seis tablas resultantes de dividir los tres originales por sus grupos. Las copias independientes coinciden con las versiones incluidas en `paper.md`. El registro de resultados y sus límites se conserva en [validation_report.json](validation_report.json).

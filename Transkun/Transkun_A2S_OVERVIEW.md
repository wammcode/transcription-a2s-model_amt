# Transkun A2S: salida Kern y datasets mono/poly/open

## Objetivo del experimento

Transkun se diseñó para escuchar una interpretación de piano y generar MIDI:
notas, tiempos, intensidad y pedales. Este experimento reutiliza su núcleo
para producir una partitura. La salida final actual es **Humdrum `**kern`**,
en archivos `.krn` o `.kern`.

```text
ANTES
Transkun: Audio → MIDI

DESPUÉS
Transkun adaptado: Audio → símbolos de partitura → **kern
```

La hipótesis es que las representaciones musicales aprendidas por Transkun
pueden transferirse a Audio-to-Score cambiando principalmente datos,
objetivos y salida. No se creó otra arquitectura ni se reemplazó el modelo
por `a2s-transformer`.

## Qué se mantiene y qué cambia

Se mantienen el procesamiento de audio, ventanas aprendibles, representación
mel, convoluciones, embeddings, atención y backbone V2 de Transkun. Los pesos
AMT incluidos pueden inicializar ese núcleo; se verifica su carga completa.

Para A2S se añadieron pares audio/partitura, una representación de notación,
un head CTC, entrenamiento/validación y reconstrucción del score. La semi-CRF
original sigue disponible para AMT, pero A2S utiliza otro decoder de tarea.
La salida se escribe directamente en Kern, sin MIDI o MusicXML intermedio.
Los comandos AMT originales permanecen disponibles.

## Datasets admitidos

El mismo entrenamiento puede utilizar:

- **Monofónicos:** una línea musical a la vez.
- **Polifónicos:** acordes y varias voces explícitas.
- **Polifónicos abiertos:** notas solapadas, duraciones independientes y
  cantidad variable de voces activas; también entradas Kern con splits,
  joins, adiciones o intercambios de spines.

Se pueden mezclar estas texturas en un mismo dataset o batch. No se necesita
seleccionar una arquitectura distinta ni fijar el número de voces. Si el
score carece de etiquetas de voz, se distribuyen las notas en líneas sin
solapamiento preservando sus tiempos y duraciones. Esa distribución técnica
no pretende identificar las líneas melódicas ideales de una obra.

El codec v2 compone posiciones, duraciones y números de parte/voz por dígitos.
Así puede representar nuevos valores sin exigir que cada número apareciera
en entrenamiento. Además, el training determina automáticamente suficientes
pasos de salida CTC para los targets densos y guarda ese valor en el checkpoint.
El núcleo acústico permanece igual.

Los manifests admiten archivos Kern, partituras MusicXML de entrada o texto
Kern inline. Cada waveform debe estar emparejado con su fragmento completo
de partitura. Audio con un canal puede contener música polifónica: canales
de audio y voces musicales son conceptos diferentes.

## Papel de a2s-transformer

Se estudió como referencia del dominio: secuencias de notación, targets,
vocabulario, batches variables, CTC, decoding y métricas. No se incorporaron
su encoder, decoder Transformer, red recurrente o preprocessing. La adaptación
mantiene una separación explícita: representación acústica de Transkun y
tarea/salida A2S.

## Resultado y validación

Se puede entrenar, validar, reanudar y ejecutar audio → Kern desde línea de
comandos. Se ejecutaron **33 pruebas**, todas satisfactorias, con training y
validation de cada textura, batches mixtos, polifonía densa, cambios de spines,
exportación Kern y compatibilidad AMT.

El modelo completo con pesos AMT pasó forward/backward en GPU. Un fixture
pequeño v2 con dos audios sintéticos aprendió sus targets y produjo Kern
estricto válido, reabierto correctamente: C4 de cuatro negras, sin fallback.
Los mismos audios se usaron en train/validation; esto verifica ejecución y
capacidad de ajuste, **no demuestra generalización a música real**.

Los checkpoints A2S v1 existentes también pueden exportar Kern. Conservan
su representación/vocabulario anterior; para adoptar v2 se entrena un nuevo
head. No se reinterpretan silenciosamente los pesos anteriores.

## Uso y límites

Desde `Transkun`, con el entorno activado:

```bash
python -m transkun.trainA2S checkpoint/a2s.pt \
  --modelConf configs/a2s.conf \
  --datasetMetaFile_train /ruta/train.jsonl \
  --datasetMetaFile_val /ruta/val.jsonl \
  --initAMT transkun/pretrained/2.0.pt --device cuda

python -m transkun.transcribeA2S audio.wav score.krn \
  --weight checkpoint/a2s.best.pt --strict
```

Falta entrenar y evaluar con datos A2S reales y splits independientes. Se
procesan fragmentos, con límite predeterminado de 30 segundos, sin ensamblado
automático de obras completas. Se cubre la notación básica implementada,
no toda la ornamentación, dinámica, layout o metadatos de un Kern arbitrario.
Un score que se puede abrir no equivale a una transcripción musical acertada.

## Valor experimental y próximos pasos

Esta adaptación permite estudiar transferencia AMT → A2S, reutilización del
backbone y diferencias entre objetivos de transcripción. Los siguientes
experimentos son entrenar con corpus reales, comparar con `a2s-transformer`
usando datos/métricas comunes, comparar inicialización AMT con training desde
cero, congelar o ajustar el core y analizar convergencia, notas, ritmos y voces.

Las decisiones, archivos, formatos y pruebas reproducibles están en
[Transkun_A2S_TECHNICAL.md](Transkun_A2S_TECHNICAL.md).

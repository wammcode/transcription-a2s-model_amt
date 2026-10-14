# Entrenamiento AMT de Transkun con datasets en data/

Configuración actualizada el 14 de octubre de 2026 a petición del usuario,
según la **tabla 1 del paper de Transkun V2**, disponible en
[paper_audio_to_midi_base](../paper_audio_to_midi_base/paper.md).

Se entrena el modelo AMT original `ModelTransformer.TransKun`: waveform,
frontend/backbone V2, scorer de intervalos, semi-CRF y heads de velocidad,
onset/offset y presencia. El core conserva sus dimensiones y arquitectura.
Esta ruta produce MIDI; A2S conserva su propio trainer y targets de score.

Para instalación y ejecución en otro servidor, consultar
[Transkun_AMT_SERVER.md](Transkun_AMT_SERVER.md).

## 1. Parámetros finales y procedencia

La configuración se divide entre `configs/amt.conf` (modelo) y
`configs/amt_training.conf` (entrenamiento). El trainer carga ambos por defecto.

| Parámetro | Valor final | Procedencia |
|---|---|---|
| Inicialización | Aleatoria, desde cero | Elección del usuario |
| Batch físico | 12 | Batch declarado en la tabla 1 |
| Acumulación | 1 | Un batch por actualización; efectivo 12 |
| Fragmento | 16 s | Tabla 1 |
| Hop de muestreo e inferencia | 8 s | Configuración original V2; inferencia con 50% de solapamiento |
| Audio / STFT | 44100 Hz, ventana 4096, hop 1024 | Frontend original, tabla 1 |
| Mels / subventanas / frecuencia | 229 / 5 / 30–8000 Hz | Frontend original, tabla 1 |
| Optimizador | AdaBelief | Tabla 1 |
| Learning rate máximo | 4e-4 | Tabla 1 |
| Weight decay | 1e-2 | Tabla 1 |
| Exclusiones del decay | Bias, normalización y posición | Tabla 1 y `TrainUtil.getOptimizerGroup` original |
| Presupuesto | 500000 optimizer updates | Tabla 1 |
| Scheduler | OneCycleLR, annealing coseno | Implementación original y tabla 1 |
| Warm-up | 5%: aproximadamente 25000 updates | Tabla 1 |
| Clipping | Percentil 80 de las últimas 10000 normas | Tabla 1 |
| Norma inicial del historial | 40 | Script original `train.py` |
| Escala del backward | Loss / 50 | Script original `train.py` |
| Precisión | FP32; TF32 no habilitado explícitamente | Ruta AMT actual |
| Seed / workers / threads | 42 / 2 / 2 | Opciones de ejecución reproducible |
| Validación / checkpoint periódico | Cada época / cada 50 updates | Opciones de ejecución |
| Último batch físico incompleto | Se descarta en train | Comportamiento original del trainer paralelo |

El paper informa dos RTX 4090 como hardware de su experimento. Esta ruta
actual sigue usando una GPU por proceso; no implementa entrenamiento distribuido.
El batch 12 del preset se ejecuta completo en esa GPU. La tabla del paper
no precisa en ese campo cómo se repartió el batch entre sus dos GPUs.

Los factores de OneCycleLR siguen el código original: `div_factor=20`,
`final_div_factor=2` y `cycle_momentum=False`. Con este LR máximo,
el LR inicial del ciclo es 2e-5 y su mínimo teórico es 1e-5.
AdaBelief conserva `eps=1e-8`, weight decay desacoplado y rectificación.

Estas opciones alinean el protocolo de entrenamiento con el paper; el
corpus, instrumento y hardware siguen siendo los de este experimento.

## 2. Dataset reutilizable

```text
data/
  data_saxophone/real_a2s_sax_dataset/
    wav/clip001.wav
    midi/clip001.mid
    amt_partitions/saxophone_amt/
      train.txt
      val.txt
      test.txt
  otro_dataset/
    wav/...
    midi/...
    amt_partitions/otro_amt/
      train.txt
      val.txt
      test.txt
```

Cada línea contiene dos rutas separadas por tabulación:

```text
wav/clip001.wav<TAB>midi/clip001.mid
```

Las rutas se resuelven respecto a `--datasetRoot`. También se admiten rutas
absolutas, comentarios `#`, líneas vacías y rutas entre comillas separadas por
espacios. Se comprueban archivos inexistentes, entradas repetidas y audio
compartido entre splits. Test es opcional y no selecciona checkpoints.

`--partitionDir` puede ser relativo al dataset o absoluto. Si se omite,
se busca un único directorio con train.txt/val.txt bajo amt_partitions.

| Split de saxofón | Audios | Duración | Notas MIDI |
|---|---:|---:|---:|
| Train | 360 | 65,08 min | 6215 |
| Validation | 51 | 9,30 min | 935 |
| Test | 102 | 18,04 min | 1701 |
| Total | 513 | 92,43 min | 8851 |

El usuario confirma que son grabaciones reales con MIDI generado
sintéticamente, validadas y usadas en otros experimentos. No se han añadido
cambios al tratamiento de anotaciones por esta actualización del protocolo.

Se mantienen las políticas existentes de fronteras: `clip` recorta targets
en memoria contra el audio, conservando flags de eventos incompletos; `error`
rechaza eventos fuera del audio. Los archivos originales permanecen intactos.
La auditoría detectó 333/47/93 finales recortados en train/val/test y ninguna
nota descartada completamente fuera del audio. El recorte no estira ni
desplaza los tiempos MIDI y no constituye una evaluación de calidad del corpus.
La F1 con offset actual utiliza los tiempos de referencia recortados.

El lector carga fragmentos como float32, convierte a mono y resamplea cuando
hace falta. Se cubren pitches 21–108. Se rechazan percusión sin altura y
pitches fuera del rango. MIDI multitrack se fusiona por pitch, con resolución
de solapamientos original; no se aprende identificación de instrumento.
La extensión de sustain está desactivada para saxofón.

## 3. Comando principal desde cero

Desde `Transkun/`, con el entorno instalado:

```bash
python -m transkun.trainAMT \
  --datasetRoot ../data/data_saxophone/real_a2s_sax_dataset \
  --partitionDir amt_partitions/saxophone_amt \
  --runDir runs/saxophone_amt_paper \
  --modelConf configs/amt.conf \
  --trainingConf configs/amt_training.conf --device cuda
```

El preset ya establece batch 12, fragmentos 16 s y 500000 updates.
No añadir `--initAMT` ni `--resume` para este experimento desde cero.

Los argumentos explícitos prevalecen sobre el preset. `--nIter N` fija
updates y horizonte del scheduler; `--epochs N` selecciona en su lugar un
presupuesto por épocas. Son mutuamente excluyentes. `--maxSteps N` solo
interrumpe anticipadamente para pruebas y no modifica el horizonte planificado.

El hop omitido se resuelve como `chunkSize/2`. Con tus datos hay **664 ventanas**.
Con batch 12 y descarte del último batch incompleto, hay **55 updates por
época**. Se completan 9090 épocas y otros 50 updates de la época 9091,
hasta alcanzar exactamente 500000. Son aproximadamente **9091 épocas**;
el número de épocas cambia con el tamaño del dataset y el batch.

El batch incompleto descartado consta de 4 ventanas en este corpus; el orden
se baraja por época. Validation conserva todos sus ejemplos. Una validación
final se ejecuta también si el último update cae dentro de una época.

Para otro corpus, cambiar datasetRoot, partitionDir y runDir:

```bash
python -m transkun.trainAMT \
  --datasetRoot ../data/otro_dataset \
  --partitionDir amt_partitions/otro_amt \
  --runDir runs/otro_amt_paper --device cuda
```

También funcionan `transkunTrainAMT` y `python -m transkun.train --datasetRoot ...`.
La ruta MAESTRO original de `train.py` sigue disponible.

## 4. Clipping, loss y checkpoints

La loss sigue siendo `-model.log_prob(audio, notes).sum(-1).mean()`.
El backward utiliza la escala original 1/50 y divide por el número de
microbatches del grupo de acumulación.

Antes de cada actualización se calcula el percentil 80 del historial de
normas **anteriores**, se recorta con ese umbral y se incorpora la norma
actual medida antes del recorte. Se mantienen como máximo 10000 valores,
mediante el `MovingBuffer` original. Se inicializa con 40, como el script
original. `--gradClip N` permite elegir clipping fijo, pero el preset del
paper utiliza el adaptativo.

Los checkpoints nuevos son AMT **formato 2** e incluyen pesos, configuración,
optimizador, scheduler, RNG, época, siguiente batch, global step, mejores
métricas, huella del dataset, opciones y **historial del clipping**.
La reanudación restaura todo, incluido ese historial, y verifica coincidencia
de las opciones del protocolo.

Checkpoints AMT anteriores, de formato 1, siguen siendo utilizables para
inferencia e inicialización de pesos mediante `--initAMT`; no se reanudan
exactamente con este trainer porque no contienen el historial adaptativo.

```text
runs/saxophone_amt_paper/
  last.pt
  best.pt
  conf.json
  run_config.json
  dataset_audit.json
  train.log
  metrics.jsonl
  status.json
  tensorboard/
```

`last.pt` se guarda cada 50 updates, al completar una época y al finalizar
o detener ordenadamente. `best.pt` selecciona mayor note-offset F1 en val,
con NLL como desempate. Test no participa en la selección.
Las escrituras de checkpoints y estado son atómicas.

Los logs se generan automáticamente; consultarlos no es necesario para que
el entrenamiento avance. Los runs y datos no se transportan mediante Git.

## 5. Validación e inferencia

```bash
python -m transkun.trainAMT \
  --datasetRoot ../data/data_saxophone/real_a2s_sax_dataset \
  --partitionDir amt_partitions/saxophone_amt \
  --runDir runs/saxophone_amt_paper --validateOnly --split test \
  --weight runs/saxophone_amt_paper/best.pt --device cuda

python -m transkun.transcribe \
  ../data/data_saxophone/real_a2s_sax_dataset/wav/000051661-1_1_1.wav \
  runs/saxophone_amt_paper/example.mid \
  --weight runs/saxophone_amt_paper/best.pt \
  --conf runs/saxophone_amt_paper/conf.json --device cuda
```

Validation calcula NLL por ventanas y note-offset F1 por archivos completos,
utilizando la inferencia original con ventanas solapadas y unión de eventos.
`--chunkMetrics` opta por F1 sobre fragmentos; no es la misma evaluación.
La salida MIDI conserva el programa piano del escritor original; el modelo
aprende notas, tiempos y velocidades, no el programa instrumental saxofón.

## 6. Verificación y estado local

Tras estos cambios: **42 tests pasan**. Se añadieron pruebas de clipping
adaptativo e historial, precedencia del preset/CLI y presupuesto por updates
que termina dentro de una época. Las pruebas comparan pesos e historial entre
entrenamiento continuo e interrumpido/reanudado y comprueban el último
checkpoint, scheduler, estado completed y validación final.

Además, el modelo completo de 13.613.711 parámetros terminó una prueba CUDA
desde cero con tus audios: fragmentos 16 s, LR máximo 4e-4, weight decay 1e-2
y clipping adaptativo. Para esa prueba se redujeron explícitamente el
presupuesto a 3 updates y el batch a 1. Se validó un archivo real completo
y se guardaron checkpoints formato 2. Artefactos:
`runs/saxophone_amt_paper_smoke/`.

Comprobaciones:

```bash
python -m pytest -q
python -m compileall -q transkun tests
git diff --check
```

El entrenamiento largo local sigue detenido. El run histórico
`runs/saxophone_amt/` conserva sus 346 updates y 2 épocas completas bajo el
preset anterior. Las mediciones previas de batch 1/8 s en RTX 4050 y su
estimación de 4–6 horas para 50 épocas **no describen el nuevo preset**.
No se han ejecutado 500000 updates ni se ha probado batch físico 12/16 s en
el servidor de destino. La guía del servidor incluye profiling previo.

Los warnings existentes de PyTorch checkpointing, AdaBelief y lectura de
audio provienen de dependencias/core upstream. No se modificó el backbone.

## 7. Archivos de esta actualización

| Archivo | Responsabilidad |
|---|---|
| `configs/amt.conf` | Segmentos 16 s e inferencia con hop 8 s; dimensiones del core intactas |
| `configs/amt_training.conf` | Preset explícito del protocolo del paper |
| `transkun/trainAMT.py` | Carga del preset, presupuesto por updates, clipping e historial persistente, validación final |
| `tests/test_amt_datasets.py` | Regresiones de presupuesto, reanudación, configuración e historial |
| `README.md` | Comando actual |
| `Transkun_AMT_SERVER.md` | Instalación y lanzamiento desde cero en el servidor |
| `Transkun_AMT_TRAINING.md` | Parámetros, datos y comportamiento real |

`DataAMT.py`, el core de Transkun y los archivos del dataset no se han
modificado en esta actualización.

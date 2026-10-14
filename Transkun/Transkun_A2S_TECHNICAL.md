# Transkun A2S: documentación técnica y salida Kern

Actualizado y validado el 14 de octubre de 2026. Base: snapshot Transkun
`c5cb8370e17b4a1650b5971ba87ee4b0d208e5b6`; referencia funcional
`a2s-transformer` `1d624e07f108a674b55414bbd2ab1e734e4d8843`, según
`../UPSTREAM_REPOSITORIES.json`.

## 1. Alcance actual y arquitectura

Se implementa **Audio → frontend/backbone Transkun V2 → head CTC → símbolos
de partitura → archivo Humdrum `**kern`**. Inferencia produce `.krn` o `.kern`;
MusicXML sigue siendo un formato de entrada de targets, pero no es la salida
del camino A2S. No se genera MIDI ni MusicXML intermedio.

La arquitectura de representación permanece Transkun. La semi-CRF original
se conserva para AMT, pero A2S utiliza un readout y decoder de tarea CTC.
No se incorpora encoder, decoder Transformer, CNN o RNN de otro modelo.

```text
ANTES: AMT
Audio → preprocessing Transkun → backbone Transkun
      → scores de intervalos → semi-CRF/Viterbi → atributos → MIDI

DESPUÉS: A2S
Audio → lectura/resampling → preprocessing Transkun → MISMO backbone
      → head de score + CTC → registros de notación
      → exportación directa **kern → score.krn
```

Se admiten monofonía, acordes, voces independientes, polifonía abierta/asíncrona,
cantidad variable de voces y batches mixtos. No hay un número de voces fijado
por el head o dataloader. Las voces de un score son diferentes de los 90
tracks de queries internos de Transkun, que se conservan.

## 2. Arquitectura y pipeline AMT originales

`Util.makeFrame` recibe `[B,C,S]`, aplica padding de media ventana a la izquierda
y completa el extremo derecho. Devuelve `[B,C,T,W]`, con
`T=ceil(S/hopSize)+1`. Transkun normaliza cada ejemplo usando la media y
desviación de sus frames completos.

`Spectrum` calcula FFT con ventana Hann y ventanas Gaussianas aprendibles.
`MelSpectrum` obtiene potencia, promedia canales de audio, aplica el banco mel
de torchaudio y la transformación logarítmica original. V2 utiliza 44.1 kHz,
ventana 4096, hop 1024, 229 bins mel y cinco ventanas extra: seis canales de
features, diferentes de los canales de audio.

`LayersTransformer.Backbone` recibe `[B,T,F,K]`. Conserva convolución de
entrada, embeddings espaciales aprendibles, reducción temporal de factor 8
y frecuencial de factor 4 con `downsampleF=True`. Añade tracks de agregación
y queries de símbolos AMT. Sus `BasicBlock` combinan atención sobre frecuencia
y tiempo, Q/K/V, feed-forward, RMSNorm, residuales y LayerScale. Features
acústicas y queries participan en los mismos bloques; no hay decoder
autoregresivo independiente.

El upsampling devuelve `[B,P,T,D]`. `P=90`: pedales -64/-67 y pitches MIDI
21…108. `D=baseSize×scoringExpansionFactor`. El checkpoint incluido utiliza
baseSize 64, ocho heads y seis capas, por lo que `D=256`; el default de clase
original es baseSize 40 y cuatro heads.

`ScaledInnerProductIntervalScorer` calcula scores de inicio/fin escalados por
longitud y un término diagonal para singletons. Los scores `[T,T,B,P]` se
aplanan para `NeuralSemiCRFInterval`, que calcula el normalizador y decodifica
intervalos mediante programación dinámica/Viterbi. Los contextos de inicio,
fin y su producto alimentan `velocityPredictor` —128 clases— y
`refinedOFPredictor`, que estima correcciones y presencia de extremos.

La likelihood AMT combina semi-CRF, velocidad, refinamiento ContinuousBernoulli
y presencia Bernoulli. `train.py` usa targets de notas/pedales, AdaBelief,
OneCycleLR, clipping y métricas AMT. `Data.py` prepara MAESTRO y slices de audio
con anotaciones MIDI; puede extender notas por pedal. `transcribe.py` procesa
ventanas solapadas, enlaza eventos incompletos y escribe MIDI. Los checkpoints
contienen state dict y, según su origen, mejores pesos, optimizador/scheduler
y progreso de training.

## 3. Conservación del core

`LayersTransformer.py`, `Util.py`, `CRF`, `Data.py`, `TrainUtil.py`,
`Evaluation.py` y los pesos/configuración AMT originales no se
modifican. Se conservan operaciones, bloques, parámetros, buffers, queries
y dimensiones del frontend/backbone.

Posteriormente se añadió una ruta AMT para datasets de `data/` con particiones
TXT: `trainAMT.py` y `DataAMT.py`. `train.py` delega en ella cuando recibe
`--datasetRoot`, conservando su ruta MAESTRO anterior. No cambia el modelo
A2S ni el core. Los comandos están en
[Transkun_AMT_TRAINING.md](Transkun_AMT_TRAINING.md).
El lector de `transcribe.py` se ajustó para autodetectar formatos de audio y
normalizar la anchura PCM, permitiendo WAV de los nuevos datasets. El forward,
heads y decoding AMT permanecen iguales.

El cambio al modelo AMT en `ModelTransformer.py` extrae su bloque existente de
frontend/backbone a `encodeFramesBatch`; `processFramesBatch` lo llama y
continúa con el mismo scorer y semi-CRF. No añade parámetros AMT. Una regresión
verifica contextos AMT/A2S exactamente iguales con los mismos pesos.

`ModelA2S.TransKun` hereda de V2. Conserva y congela los heads AMT, pero no los
ejecuta en A2S ni materializa la matriz de intervalos `T×T`. Añade LayerNorm,
proyección lineal, GELU/dropout y clasificación de símbolos. Por defecto
frontend y backbone reciben gradientes; `--freezeBackbone` congela ambos.
Conservar arquitectura no significa congelar sus pesos.

## 4. Targets, codec y vocabulario v2

La forma semántica de los registros temporales es:

```text
AT_0 PART_0 CLEF_G2:0 END
AT_0 PART_0 KEY_0 END
AT_0 PART_0 METER_4/4 END
AT_0 PART_0 VOICE_0 DUR_1 PITCH_C4 TIE_none END
AT_1/3 PART_0 VOICE_1 DUR_2/3 PITCH_B-3 TIE_start END
AT_2 PART_1 VOICE_0 DUR_1 REST END
```

`AT` y `DUR` son fracciones de negras, no segundos de performance. Acordes
contienen varios pares `PITCH/TIE`; spelling y octava se conservan. Ties son
`none/start/continue/stop`. Se representan compás, armadura, claves con
desplazamiento de octava y tempo referido a la negra, incluyendo cambios
posteriores. Sin metadata inicial se añaden 4/4, armadura 0 y clave de sol.

El codec actual es **`transkun-score-v2`**. Los ejemplos anteriores son su
forma semántica legible; los targets de la red componen campos numéricos:

```text
AT_17/3   → AT NUM_1 NUM_7 NUM_/ NUM_3 VALUE_END
PART_80   → PART NUM_8 NUM_0 VALUE_END
VOICE_120 → VOICE NUM_1 NUM_2 NUM_0 VALUE_END
DUR_5/3   → DUR NUM_5 NUM_/ NUM_3 VALUE_END
```

Se aplica a `AT/PART/VOICE/DUR/KEY/TEMPO/METER`. Un offset, duración, compás
o ID de parte/voz nuevo no exige una clase nueva. El alfabeto se define por
gramática y no extrae símbolos de validation: números, delimitadores, rests,
ties, claves y pitches MIDI 0…127 con spellings y alteraciones -4…+4. Son
**712 símbolos** incluido `<BLANK>`. Símbolos adicionales fuera del alfabeto
se incorporan solo si aparecen en training; no se sustituyen silenciosamente.

El índice 0 es blank CTC y padding; nunca un target válido. Las fracciones
se normalizan con denominador máximo 1920 y tolerancia `1e-7`. Se rechazan
performance MIDI, grace notes, percusión sin altura y pitches microtonales.

### Normalización de mono/poly/open

`ScoreCodec._normalize_voices` recorre la jerarquía acumulando offsets y
manteniendo partes/voces sin depender de `flatten()`. En un grupo de voz,
notas simultáneas de igual duración se agrupan como acorde si no hay pitches
duplicados. Notas solapadas de distinta duración se distribuyen de forma
determinista en lanes sin solapamiento, reutilizadas cuando quedan libres.
Voces declaradas distintas no se fusionan entre sí.

Se admiten scores sin etiquetas explícitas de voz y cantidad variable de
notas simultáneas. Esta asignación es estructural, no una estimación
musicológica de melodías. Con voces explícitas se conserva su separación.
Se retiró el límite anterior de IDs de 64 partes/voces. El parser de
predicciones sigue rechazando solapamientos dentro de una lane individual:
deben representarse como acorde o como lanes diferentes.

## 5. IO Kern directo

`Kern.py` gestiona archivo, reloj y topología porque Music21 lee Humdrum pero
no proporciona un writer. Se reutilizan sus convertidores de notas/tandems;
no se utiliza MIDI o MusicXML como intermediario.

`loads_kern` reconoce headers, acordes separados por espacios, rests, dots
nulos de continuidad y spines auxiliares como `**text`. Mantiene un reloj
racional compartido para onsets asíncronos y entradas de spines nuevos.
Soporta `*^` split, `*v` join, `*+` addition, `*-` termination y `*x` exchange.
Una nueva lane puede entrar durante el sustain de otra. `*partN` agrupa lanes
de una misma parte. Se rechazan columnas incompatibles, paths inválidos,
solapamientos en un spine y archivos sin terminación.

Se corrigió un problema observado en el parser completo de Music21: un
evento de un spine añadido mediante `*+` se situaba en 0 en vez de 1 negra.
La prueba comprueba explícitamente ese onset. Los paths/delimitadores se
corresponden con la [sintaxis Humdrum](https://www.humdrum.org/guide/ch05/).

`dumps_kern` genera un spine por lane normalizada y rellena huecos con rests.
Un `.` significa que el evento anterior sigue sonando. Emite headers,
metadata, barlines y terminadores. El número de columnas se infiere por score;
no hay una plantilla fija por dataset. Los paths dinámicos de entrada se
normalizan a spines completos con rests cuando una voz está inactiva; no se
reproduce literalmente su árbol de splits/joins.

Pitches, accidentes, acordes y ties siguen
[la representación Kern](https://www.humdrum.org/rep/kern/). Para duraciones se
prefieren valores recíprocos/dotted y se usa `n%m` cuando resulta necesario,
conforme a la [extensión racional de recip](https://humlib.humdrum.org/doc/topic/).
Notas que cruzan un barline o cambio de metadata se parten con ties por pitch,
preservando el sustain sin nuevos ataques.

`ScoreCodec.write` acepta solamente `.krn`/`.kern`. En modo estricto se
rechazan registros inválidos, secuencias sin eventos y registros sin `END`.
Recuperación omite registros inválidos, informa `rejected_records` y, si no
queda ningún evento, añade un silencio de cuatro negras con `fallback=true`.
Ese archivo recuperado no constituye una transcripción acertada.

## 6. Flujo de tensors

1. `DataScore.read_audio`: SoundFile float32, promediado de canales,
   `resample_poly` a `conf.fs`; salida `[S,1]`. Audio mono no implica score mono.
2. `collate_score`: audio `[B,S_max,1]`, longitudes `[B]`, targets `[B,L_max]`,
   longitudes `[B]` y secuencias de texto.
3. El forward agrupa longitudes iguales y corta padding **antes** de FFT,
   normalización y atención: `[B_g,1,S] → [B_g,1,T,W] → [B_g,T,F,K]`.
4. El core devuelve `[B_g,90,T,D]`. No se introduce una máscara en la atención
   upstream, que no tiene máscara de padding de audio.
5. Se concatenan queries: `[B_g,T,90×D]`, conservando sus embeddings.
6. Head: LayerNorm, Linear, GELU/dropout; config V2 usa `23040 → 256`.
7. Interpolación lineal solo en el head: `T_score=m×T`.
8. Clasificación `[B_g,T_score,V]`, reunión de grupos con padding y longitudes.
9. CTC utiliza `[T_score_max,B,V]`; padding queda excluido por longitudes.
   Greedy colapsa antes de eliminar blank: `A,blank,A → A,A`.
10. Decoding de registros y escritura directa Kern.

Esto admite batches mixtos y longitudes distintas, con menor eficiencia
cuando muchos ejemplos forman grupos individuales.

## 7. Dataset, batching y entrenamiento

```text
dataset/
  train.jsonl
  val.jsonl
  test.jsonl
  audio/clip.wav
  scores/clip.krn
```

Cada línea JSONL contiene un par completo:

```json
{"audio":"audio/clip.wav","score":"scores/clip.krn"}
```

Se admiten `.json` con una lista y targets `.krn/.kern` o MusicXML/XML/MXL.
También se acepta texto Kern inline:

```json
{"audio":"audio/clip.wav","transcript":"**kern\n4c\n4d\n2r\n*-\n"}
```

Tokens precomputados requieren `tokens` y `codec_version`. Debe aparecer
exactamente un campo entre `score`, `transcript` y `tokens`. Las rutas se
resuelven respecto al manifest o a `--datasetPath`; pueden ser absolutas.
No se requiere una opción mono/poly/open: la estructura se infiere por target.

Los clips deben corresponder a scores completos del fragmento, con offsets
locales. No se corta audio sin su target, no se alinean automáticamente obras
y el límite predeterminado es 30 segundos. Los waveforms se leen bajo demanda;
scores y targets se preparan al inicializar el dataset. Se deben separar splits
por obra/performance para evaluar generalización.

### Capacidad CTC automática para polifonía densa

```text
m × (ceil(samples/hopSize)+1) >= tokens + repeticiones_adyacentes
```

`ScoreDataset.required_frame_multiplier` calcula el mínimo. Al iniciar un
training nuevo, `trainA2S` toma el máximo entre el valor configurado y los
mínimos de train/validation. Usa solo sus longitudes para esa resolución;
no extrae símbolos de validation para el vocabulario. El factor final se
imprime y se guarda en el checkpoint. Solo aumenta pasos del head, no la
resolución/atención del core.

Resume/inferencia mantienen el factor persistido, sin consultar un target
durante inferencia. Un nuevo corpus más exigente puede necesitar otro
checkpoint/configuración. `--noAutoFrameMultiplier` exige el factor configurado.
CTC mantiene `zero_infinity=False` y guardas de blank/longitud/valores finitos:
no se ocultan alineaciones imposibles.

### Loop, configuración y comandos

El loop usa DataLoader, forward del core, CTC media, backward, clipping de
norma 5 y AdamW. Es de un proceso, FP32 y LR constante. Cada epoch valida sin
gradientes y guarda checkpoint último/mejor por loss CTC. Métricas: loss,
Symbol ER por Levenshtein, Sequence ER, parser estricto, fallback y registros
rechazados. Parser estricto no significa corrección musical completa.

`configs/a2s.conf` selecciona `transkun.ModelA2S` en JSON estilo moduleconf
y coincide con el checkpoint AMT incluido. Campos añadidos:

| Campo | Default | Función |
|---|---:|---|
| `scoreHeadSize` | 256 | Bottleneck del head |
| `scoreDropout` | 0.1 | Dropout del head |
| `scoreFrameMultiplier` | 2 | Mínimo resuelto al iniciar training |
| `scoreMaxAudioSeconds` | 30 | Límite sin truncamiento |
| `scoreCodecVersion` | `transkun-score-v2` | Representación/vocabulario versionados |

`configs/a2s-smoke.conf` usa 8 kHz, ventana 256, hop 64, 16 bins mel, baseSize
4, una capa y head 16. Solo sirve para pruebas y no admite pesos AMT completos.
El generador original sigue funcionando:

```bash
mkdir -p checkpoint
python -m moduleconf.generate Model:transkun.ModelA2S > checkpoint/a2s-generated.conf
```

Para el checkpoint incluido se debe usar `configs/a2s.conf`; los defaults
generados baseSize 40/nHead 4 no coinciden con sus dimensiones.

Desde `Transkun`, con Python 3.11 y torch/torchaudio compatibles:

```bash
python -m pip install -e '.[a2s,test]'
python -m transkun.trainA2S checkpoint/a2s.pt \
  --modelConf configs/a2s.conf \
  --datasetMetaFile_train /ruta/dataset/train.jsonl \
  --datasetMetaFile_val /ruta/dataset/val.jsonl \
  --initAMT transkun/pretrained/2.0.pt --device cuda --epochs 50

python -m transkun.trainA2S checkpoint/a2s.pt \
  --datasetMetaFile_train /ruta/dataset/train.jsonl \
  --datasetMetaFile_val /ruta/dataset/val.jsonl --resume --device cuda --epochs 100

python -m transkun.trainA2S checkpoint/a2s.best.pt \
  --datasetMetaFile_val /ruta/dataset/test.jsonl --validateOnly --device cuda
```

El mismo comando entrena cualquiera de las tres texturas o su combinación.
CPU funciona sin llamadas CUDA obligatorias. Omitir `--initAMT` entrena desde
cero. `--freezeBackbone` permite head-only training y debe repetirse al reanudar
si se mantiene ese régimen. `--epochs` es el total acumulado y el LR de resume
se restaura del optimizador. Otro modelConf se rechaza, salvo una plantilla con
factor menor ya resuelto automáticamente: se mantiene el valor del checkpoint.

## 8. Checkpoints, inferencia y compatibilidad

`CheckpointA2S` guarda tarea/formato/codec, configuración resuelta, vocabulario,
pesos, optimizador, métricas, epoch/step, mejor loss y RNG de torch/CUDA/DataLoader.
Escribe mediante temporal/replace y carga con `weights_only=True` y state dict
estricto. `warm_start_amt` exige claves/shapes AMT completas; únicamente el
nuevo head queda sin inicializar desde AMT. El checkpoint incluido carga
**184 tensores**.

```bash
python -m transkun.transcribeA2S /ruta/audio.wav /ruta/score.krn \
  --weight checkpoint/a2s.best.pt --device cuda \
  --tokensOut /ruta/score.tokens.json --strict
```

Se carga configuración/vocabulario/codec, se lee/resamplea audio, se ejecuta
core/head con `eval/no_grad`, CTC greedy, reconstrucción y escritura Kern.
No necesita tempo ni score de referencia. El JSON opcional conserva tokens,
codec, `output_format=**kern` y diagnósticos. Se procesa un clip, sin stitching.
Aliases instalados: `transkunTrainA2S`, `transkunA2S`.

Los checkpoints **v1** siguen cargándose con sus índices originales y pueden
exportar Kern, comprobado con el checkpoint sintético anterior. Conservan
la representación atómica y sus restricciones de OOV; no adquieren v2
automáticamente. Para v2 se entrena un nuevo head, opcionalmente inicializando
desde AMT. No se cambia la dimensión del head v1 ni se reinterpretan sus
índices. Tokens precomputados deben declarar la versión correspondiente.

Los comandos y checkpoints AMT siguen disponibles. Un checkpoint AMT no es
A2S entrenado. Después de ajustar el backbone no se garantizan las métricas
AMT iniciales.

## 9. Relación con a2s-transformer

| Referencia analizada | Reutilización funcional | Adaptación/idea descartada |
|---|---|---|
| `encoding_convertions.py` | Kern como secuencia; voces/filas | Registros composicionales y reloj explícito; sin parser rectangular fijo |
| `ar_dataset.py`, `ctc_dataset.py` | Targets, vocabulario, padding, pasos suficientes | Manifest local, alfabeto de gramática/training, longitudes exactas y factor; sin SOS/EOS autoregresivos |
| `data_preprocessing.py` | Collate variable | Waveform/frontend Transkun; sin STFT/22050 Hz de referencia |
| `networks/transformer/*` | Encoder/decoder, teacher forcing, máscaras | Analizados y descartados; sin PE 2D ni decoder Transformer nuevos |
| `networks/crnn/*` | CTC y greedy para score | Loss/decoding, sin CNN/CRNN/RNN de referencia |
| `train.py`, `test.py`, `metrics.py` | Validation, checkpoint, Symbol/Sequence ER | CLI nativa, Levenshtein y Kern directo; sin Lightning/W&B/MV2H |

No se modifica ni importa `../a2s-transformer`. Mantener el core Transkun
permite estudiar transferencia de representación. SER de estos targets
no es directamente SER de CLEAN **kern.

## 10. Cambios y responsabilidades por archivo

| Archivo/componente | Anterior | Actual | Motivo |
|---|---|---|---|
| `ModelTransformer.py` | Encoder unido al scorer | Encoder expuesto sin cambiar AMT | Reutilización del core |
| `ModelA2S.py` | Head CTC v1 | Mismo head; config v1/v2 | Versionar interfaz |
| `Score.py` | Campos atómicos, salida MusicXML, cap 64 | Valores composicionales, normalización de lanes, Kern e IDs sin cap 64 | Mono/poly/open y valores no vistos |
| `Kern.py` (nuevo) | No existía | IO, reloj racional, paths, exportación y ties | Kern directo y sincronizado |
| `DataScore.py` | Score/tokens, guardas CTC fijas | Kern/path/inline, mezcla y factor mínimo | Datasets de las tres texturas |
| `trainA2S.py` | Factor solo configurado | Resolución automática/persistida y codec guardado | Targets densos y resume |
| `CheckpointA2S.py` | Solo v1 | v1/v2 con consistencia config/codec | Compatibilidad explícita |
| `transcribeA2S.py` | Audio → MusicXML | Audio → `.krn/.kern` | Salida solicitada |
| `examples/create_a2s_smoke_dataset.py` | Target MusicXML | Target Kern | Fixture actualizado |
| `tests/test_a2s.py` | Regresiones v1/MusicXML | Kern/v2, legacy OOV, A2S y AMT | Mantener garantías |
| `tests/test_kern_datasets.py` (nuevo) | No existía | Texturas, paths, mezcla, CTC denso, roundtrip | Verificar requisitos nuevos |
| `README.md`, ambos documentos | Descripción MusicXML | Kern y cobertura de datasets | Documentación actual |

Archivos añadidos inicialmente: configuraciones A2S/smoke, requirements A2S,
módulos de dataset/checkpoint/training/inferencia y ejemplo/tests. `setup.py`
añadió extras/entry points y corrigió la lista de requirements; `.gitignore`
conserva reglas upstream y excluye `/artifacts/`.

## 11. Validación y resultados

Entorno `.venv` local: Python 3.11.15, torch/torchaudio 2.5.1, Music21 10.5.0.
Se reutilizan paquetes del entorno conda existente sin modificar sus archivos.

```bash
.venv/bin/python -m pytest -q
```

Resultado final: **33 passed**, 19 warnings del checkpointing upstream,
sin fallos.

La suite verifica core/contextos, gradientes frontend/backbone, padding sin
influencia, CTC, dataset y checkpoints, resume equivalente al training
continuo, warm start de 184 tensores y likelihood/inferencia AMT originales.
El ejemplo semi-CRF se prueba de forma acotada; no se ejecuta su loop upstream
de 10000 pasos, que no es una suite de tests.

Las pruebas nuevas ejecutan training/validation para mono, acordes, polifonía
asíncrona y paths dinámicos, y backward de un batch mixto. Verifican valores
no observados en training, IDs mayores de 64, add/exchange/split/join incluso
durante un sustain, Kern reabierto con Music21 independiente, duraciones
exactas, ties al cruzar metadata/barlines y rechazo de Kern malformado o
MusicXML de salida.

Un caso denso necesita factor **16**, frente a 2 de la config smoke. El
training lo resuelve/guarda y resume lo conserva. Se corrigió una comparación
que rechazaba la plantilla original al reanudar después de resolver el factor.
Los warnings de checkpointing `use_reentrant`/inputs sin gradiente son del
backbone upstream durante evaluación; no se modificó para eliminarlos.

También se ejecutaron `compileall`, `git diff --check`, ayuda de CLIs y
exportación estricta v1/v2. Con el modelo completo en CUDA y pesos AMT reales:
184 tensores cargados, logits `[1,90,712]`, loss CTC `9.0404301`, gradientes
finitos en frontend/backbone.

El fixture v2 de dos audios sintéticos, compartidos entre train/validation,
se sobreajustó durante **300 pasos**: loss `0.0022233897`, Symbol/Sequence ER
0%, parser estricto 1.0 y fallback 0.0. Inferencia produjo
`artifacts/a2s-kern-v2.krn`: cinco registros, un evento, cero rechazados.
Music21 reabrió C4 de cuatro negras. El checkpoint v1 anterior generó
`artifacts/a2s-smoke.krn` igualmente sin fallback.

Esto demuestra funcionamiento y ajuste de un fixture, **no generalización
ni calidad sobre música real**. Reproducción en un directorio nuevo:

```bash
python examples/create_a2s_smoke_dataset.py artifacts/a2s-kern-v2-data
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m transkun.trainA2S artifacts/a2s-kern-v2.pt \
  --datasetMetaFile_train artifacts/a2s-kern-v2-data/train.jsonl \
  --datasetMetaFile_val artifacts/a2s-kern-v2-data/val.jsonl \
  --modelConf configs/a2s-smoke.conf --epochs 300 --batchSize 2 --max_lr 0.003
python -m transkun.transcribeA2S artifacts/a2s-kern-v2-data/audio0.wav \
  artifacts/a2s-kern-v2.krn --weight artifacts/a2s-kern-v2.best.pt --strict
```

Artifacts y pesos sintéticos están ignorados por Git y no son modelos A2S
entrenados para uso real.

## 12. Limitaciones y valor experimental

- Falta training/evaluación real con splits independientes. Se conserva el
  frontend para piano y sus sesgos; aceptar varias partes no demuestra
  generalización a todos los instrumentos.
- A2S utiliza CTC, no el decoder de intervalos semi-CRF. Requiere orden
  monotónico y pasos suficientes; interpolar el head no crea nueva resolución
  acústica y muchos pasos incrementan coste sin garantizar precisión.
- El factor observa longitudes train/validation al iniciar, nunca targets de
  test en inferencia. Nuevos targets más exigentes pueden exceder el factor
  almacenado y activar una guarda explícita.
- V2 cubre números nuevos y el alfabeto descrito. Símbolos exóticos fuera de
  él no observados en training pueden causar OOV. V1 mantiene vocabulario
  cerrado y necesita nuevo training para adoptar v2.
- Lanes automáticas preservan notas/solapamiento, pero no identifican voces
  melódicas ideales. Se emiten spines completos, sin reproducir literalmente
  el árbol de paths de origen.
- No se conservan letras, articulaciones, dinámica, ornamentos, repeticiones,
  layout, nombres/transposición instrumental ni spines auxiliares. Grace notes,
  percusión sin altura y microtonos se rechazan. No es un roundtrip íntegro de
  cualquier archivo Humdrum.
- Barline/pickup y agrupaciones de pentagramas no se conservan literalmente;
  barlines se derivan de offsets/meter. El parser estricto no certifica
  coherencia tonal ni emparejamiento global de ties.
- Permanecen bounds prácticos: offset 10000 negras, duración 256 por evento,
  armadura -7…7 y límite de audio configurable. No hay segmentación/alineación
  de obras completas, stitching o descarga HF automática.
- Training es de un proceso/FP32 con AdamW/LR constante; no incluye scheduler,
  augmentations, AMP, distribuido, beam search ni MV2H.

Se pueden comparar inicialización AMT/training desde cero, core congelado/
ajustado, capacidades del head y texturas distintas con un core común. La
comparación con `a2s-transformer` requiere datos/splits y métricas musicales
equivalentes, no solo SER sobre vocabularios distintos.

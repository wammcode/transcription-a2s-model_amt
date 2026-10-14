# Lanzar Transkun AMT desde cero con el protocolo del paper

El entrenamiento local sigue detenido. El próximo run será **Audio → MIDI**,
con inicialización aleatoria y los hiperparámetros de entrenamiento de la
tabla 1 del paper V2 en [paper_audio_to_midi_base](../paper_audio_to_midi_base/paper.md).

## 1. Copiar código y datos

Conservar esta estructura en el servidor:

```text
experimento/
  Transkun/
    configs/amt.conf
    configs/amt_training.conf
    transkun/trainAMT.py
    transkun/DataAMT.py
  data/data_saxophone/real_a2s_sax_dataset/
    wav/
    midi/
    amt_partitions/saxophone_amt/
      train.txt
      val.txt
      test.txt
```

Datos y runs están excluidos de Git. Si se usa un clon, también deben
trasladarse los datos y los cambios locales de código. No copiar `.venv`
ni checkpoints del portátil para este entrenamiento desde cero.

Ejemplo desde el workspace local, sustituyendo `usuario@servidor` y
`/ruta/experimento` por el destino real:

```bash
ssh usuario@servidor 'mkdir -p /ruta/experimento/data/data_saxophone'
rsync -av --exclude='.venv/' --exclude='runs/' --exclude='__pycache__/' \
  Transkun/ usuario@servidor:/ruta/experimento/Transkun/
rsync -av data/data_saxophone/real_a2s_sax_dataset/ \
  usuario@servidor:/ruta/experimento/data/data_saxophone/real_a2s_sax_dataset/
```

## 2. Instalar en el servidor

Se validó localmente Python 3.11 con PyTorch/torchaudio 2.5.1. Para una GPU
NVIDIA y driver compatibles con la distribución CUDA 12.4:

```bash
cd /ruta/experimento/Transkun
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'setuptools<81'
python -m pip install torch==2.5.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[amt,test]'
python -m pip check
nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__); print(torch.cuda.get_device_name(0))'
python -m pytest -q tests/test_amt_datasets.py
```

El índice y la pareja de versiones corresponden a las
[instrucciones oficiales de PyTorch](https://pytorch.org/get-started/previous-versions/#v251).
Si el hardware necesita una versión posterior, utilizar una pareja
torch/torchaudio compatible y repetir las comprobaciones. `setuptools<81`
conserva `pkg_resources`, utilizado por la inferencia original.

Inspeccionar las particiones:

```bash
python -m transkun.trainAMT \
  --datasetRoot ../data/data_saxophone/real_a2s_sax_dataset \
  --partitionDir amt_partitions/saxophone_amt \
  --runDir runs/saxophone_server_inspect --inspectOnly
```

## 3. Parámetros finales

| Parámetro | Valor |
|---|---|
| Inicialización | Desde cero |
| Batch físico / acumulación / efectivo | 12 / 1 / 12 |
| Fragmento / hop | 16 s / 8 s |
| Optimizador | AdaBelief |
| Learning rate máximo | 4e-4 |
| Weight decay | 1e-2, excluyendo bias, normalización y embeddings de posición |
| Scheduler | OneCycleLR, coseno, 500000 actualizaciones |
| Warm-up | 5%: aproximadamente 25000 actualizaciones |
| Clipping | Percentil 80 del historial de hasta 10000 normas anteriores |
| Inicio del historial / escala backward | 40 / loss dividida por 50, como en el código original |
| Workers / seed / precisión | 2 / 42 / FP32 |
| Validación / guardado periódico | Cada época / cada 50 updates |

Los valores están en `configs/amt_training.conf`. El modelo en
`configs/amt.conf` conserva su core y utiliza segmentos de 16 s.
El hop omitido se resuelve como la mitad del fragmento.
Los argumentos explícitos de CLI prevalecen sobre el preset.

En tus datos hay 664 ventanas. Se descarta el último batch físico incompleto
en train, como en el trainer paralelo original: **55 updates por época**.
Los 500000 updates equivalen a 9090 épocas completas más 50 updates de la
época 9091. **La parada se fija por updates; no se mantienen las 50 épocas
del preset anterior.** Validation conserva todos sus ejemplos.

## 4. Comprobar memoria antes de lanzar

El trainer usa **una GPU por proceso**, en FP32. El paper informa dos
RTX 4090 como hardware; esta implementación no añade DDP.
`CUDA_VISIBLE_DEVICES=0` selecciona la primera tarjeta física. Cambiar a
`1` selecciona la segunda. No usar `torchrun` ni varios procesos que
escriban en el mismo run.

Medir el batch 12/16 s en la GPU elegida:

```bash
CUDA_VISIBLE_DEVICES=0 python -m transkun.trainAMT \
  --datasetRoot ../data/data_saxophone/real_a2s_sax_dataset \
  --partitionDir amt_partitions/saxophone_amt \
  --runDir runs/saxophone_server_profile --modelConf configs/amt.conf \
  --trainingConf configs/amt_training.conf --device cuda \
  --profile --profileCases 1:16,6:16,12:16
```

La salida indica `fits` por caso y se guarda en `profile.json`. Se ejecutan
forward, backward y AdaBelief; no se guardan pesos entrenados.
Las mediciones antiguas de batch 1/8 s no prueban este batch 12/16 s.

Si el batch físico 12 no cabe pero 6 sí, utilizar `--batchSize 6
--accumulationSteps 2` al iniciar el run. Se conserva batch efectivo 12,
con 55 updates por época para este corpus. La acumulación realiza dos
microbatches antes de actualizar el optimizador; no utiliza varias GPUs.

## 5. Lanzar hasta completar 500000 updates

Desde `Transkun/`, después de preparar el entorno y comprobar memoria:

```bash
mkdir -p runs/saxophone_amt_paper
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python -u -m transkun.trainAMT \
  --datasetRoot ../data/data_saxophone/real_a2s_sax_dataset \
  --partitionDir amt_partitions/saxophone_amt \
  --runDir runs/saxophone_amt_paper --modelConf configs/amt.conf \
  --trainingConf configs/amt_training.conf --device cuda \
  --nIter 500000 --chunkSize 16 --hopSize 8 \
  --batchSize 12 --accumulationSteps 1 \
  --max_lr 4e-4 --weight_decay 1e-2 \
  --dataLoaderWorkers 2 --saveEvery 50 \
  > runs/saxophone_amt_paper/console.log 2>&1 &
echo $! > runs/saxophone_amt_paper/train.pid
```

No añadir `--initAMT`, `--resume`, `--epochs` ni `--maxSteps` a este comando
de inicio desde cero. Elegir un runDir nuevo si ya contiene `last.pt`.

`nohup` permite cerrar la conexión SSH. El entrenamiento avanza automáticamente
hasta su update 500000 y realiza validación final incluso si termina dentro
de una época. No requiere consultar logs para continuar.

## 6. Archivos del modelo y uso posterior

En `Transkun/runs/saxophone_amt_paper/` se guardan:

- `last.pt`: pesos y estado al terminar, con historial del clipping.
- `best.pt`: mejor validación por note-offset F1, con NLL como desempate.
- `conf.json`: configuración para inferencia.
- Métricas y logs automáticos; no es necesario consultarlos para ejecutar el run.

Los checkpoints nuevos son formato 2. `--resume`, en el mismo servidor,
permite recuperar un run interrumpido conservando optimizador, scheduler,
RNG e historial de clipping, siempre que datos y opciones coincidan.
No es necesario cargar ningún checkpoint para el inicio desde cero.

Inferencia con el modelo ya entrenado:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m transkun.transcribe \
  ../data/data_saxophone/real_a2s_sax_dataset/wav/000051661-1_1_1.wav \
  runs/saxophone_amt_paper/example.mid \
  --weight runs/saxophone_amt_paper/best.pt \
  --conf runs/saxophone_amt_paper/conf.json --device cuda
```

Para audio comprimido también se necesita FFmpeg. Para otros datasets en
`data/`, cambiar datasetRoot, partitionDir y runDir. Cada línea TXT contiene
`wav/archivo.wav<TAB>midi/archivo.mid`, relativa al dataset.

## 7. Verificación realizada

- **42 tests pasan**, incluidos AMT, A2S/Kern, clipping adaptativo, historial,
  precedencia CLI/configuración, presupuesto por updates, guardado,
  reanudación exacta y validación final dentro de una época.
- Prueba real CUDA desde cero con el modelo completo, fragmentos 16 s,
  AdaBelief, LR máximo 4e-4, weight decay 1e-2 y clipping adaptativo.
  Para esa prueba se limitó explícitamente el presupuesto a 3 updates y el
  batch a 1. Terminó en `completed`, validó un archivo completo y guardó
  `last.pt` y `best.pt`.
- Artefactos locales de esa comprobación:
  `runs/saxophone_amt_paper_smoke/`.
- No se ha iniciado el run largo local ni se han ejecutado 500000 updates.
  El batch físico 12/16 s debe medirse en el servidor de destino.

Las grabaciones y MIDI de tu dataset se mantienen. Se conserva el tratamiento
de fronteras existente; estos cambios afectan al protocolo de entrenamiento.

Detalles del pipeline y checkpoints:
[Transkun_AMT_TRAINING.md](Transkun_AMT_TRAINING.md).

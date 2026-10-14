"""Reproducible original Transkun AMT training on generic data/ partitions."""
import argparse
import json
import logging
import math
import os
from pathlib import Path
import signal
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch_optimizer

from .Data import Note
from .DataAMT import AMTChunks, collate_amt, inspect_dataset, read_chunk
from .Evaluation import compareTranscription
from .ModelTransformer import ModelConfig, TransKun
from .TrainUtil import MovingBuffer, getOptimizerGroup

REPO = Path(__file__).resolve().parents[1]


class GradientClipper:
    """Original Transkun adaptive clipping, with checkpointable norm history."""

    def __init__(self, quantile=0.8, history_size=10000, initial_norm=40, fixed=None):
        self.quantile = quantile
        self.fixed = fixed
        self.history = MovingBuffer(initValue=initial_norm, maxLen=history_size)

    def step(self, model):
        limit = self.fixed if self.fixed is not None else self.history.getQuantile(self.quantile)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), limit, error_if_nonfinite=True)
        self.history.step(float(norm))
        return norm, limit

    def state_dict(self):
        return {"norms": list(self.history.values)}

    def load_state_dict(self, state):
        norms = state["norms"]
        if not norms or len(norms) > self.history.values.maxlen or any(not math.isfinite(v) or v < 0 for v in norms):
            raise ValueError("Invalid gradient clipping history")
        self.history.values.clear()
        self.history.values.extend(norms)


def read_config(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("Model", {}).get("module") != "transkun.ModelTransformer":
        raise ValueError("AMT requires transkun.ModelTransformer, not ModelA2S")
    conf = ModelConfig()
    values = document["Model"]["config"]
    if values.keys() - conf.__dict__.keys():
        raise ValueError(f"Unknown AMT config fields: {values.keys() - conf.__dict__.keys()}")
    conf.__dict__.update(values)
    if conf.fs <= 0 or not 0 < conf.hopSize < conf.windowSize or not 0 <= conf.f_min < conf.f_max <= conf.fs / 2:
        raise ValueError("Invalid AMT sample rate, window/hop or frequency bounds")
    return conf


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def model_config(conf):
    return {"Model": {"module": "transkun.ModelTransformer", "configClassName": "Config", "config": conf.__dict__}}


def initialize_weights(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("task") == "A2S":
        raise ValueError("--initAMT requires AMT weights")
    state = checkpoint.get("best_state_dict")
    if state is None:
        state = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state, strict=True)


def save_checkpoint(path, model, optimizer, scheduler, epoch, next_batch, step, best, audit, options, clipper):
    checkpoint = {"task": "AMT", "format_version": 2, "config": dict(model.conf.__dict__),
                  "state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                  "lr_scheduler_state_dict": scheduler.state_dict(), "epoch": epoch,
                  "next_batch": next_batch, "global_step": step, "best_metrics": best,
                  "dataset_fingerprint": audit["fingerprint"], "training_options": options,
                  "gradient_clipping_state_dict": clipper.state_dict(),
                  "rng_state": torch.get_rng_state(),
                  "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def make_optimizer(model, lr, weight_decay):
    return torch_optimizer.AdaBelief(getOptimizerGroup(model), lr=lr, weight_decay=weight_decay,
                                    weight_decouple=True, rectify=True, eps=1e-8)


@torch.no_grad()
def validate(model, samples, chunks, batch_size, workers, device, full_files=True):
    model.eval()
    total_nll = seconds = n_gt = n_est = n_correct = 0.0
    loader = DataLoader(chunks, batch_size=batch_size, num_workers=workers, collate_fn=collate_amt)
    for batch in loader:
        audio = batch["audioSlices"].to(device)
        logp = model.log_prob(audio, batch["notes"])
        total_nll += float(-logp.sum())
        seconds += sum(batch["durations"])
        if not full_files:
            stats = model.computeStatsMIREVAL(audio, batch["notes"])
            n_gt += stats["nGT"]
            n_est += stats["nEst"]
            n_correct += stats["nCorrect"]
    if full_files:
        for sample in samples:
            waveform = read_chunk(sample["audio"], 0, sample["duration"], model.fs).to(device)
            estimated = model.transcribe(waveform, stepInSecond=model.segmentHopSizeInSecond,
                                         segmentSizeInSecond=model.segmentSizeInSecond)
            duration = sample["duration"]
            estimated = [Note(max(n.start, 0), min(n.end, duration), n.pitch, n.velocity)
                         for n in estimated if n.start < duration and n.end > 0]
            estimated.sort(key=lambda n: (n.start, n.end, n.pitch))
            metrics = compareTranscription(estimated, sample["notes"], splitPedal=True)
            n_gt += metrics["nGT"]
            n_est += metrics["nEst"]
            n_correct += metrics["note+offset"][1] * metrics["nGT"]
    precision = n_correct / n_est if n_est else 0.0
    recall = n_correct / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"nll_per_second": total_nll / seconds, "precision": precision, "recall": recall,
            "note_offset_f1": f1, "n_gt": n_gt, "n_est": n_est,
            "metric_scope": "full_files" if full_files else "chunks"}


def profile(model, dataset, device, configurations, args):
    """Real forward/backward/optimizer steps; independent cases after cleanup."""
    import gc
    results = []
    for batch_size, seconds in configurations:
        chunks = AMTChunks(dataset, model.fs, seconds, training=True)
        optimizer = make_optimizer(model, args.max_lr, args.weight_decay)
        clipper = GradientClipper(args.gradClippingQuantile, args.gradClippingHistory,
                                  args.gradClippingInit, args.gradClip)
        result = {"batch_size": batch_size, "chunk_seconds": seconds, "precision": "float32"}
        try:
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            times = []
            for iteration in range(3):
                batch = collate_amt([chunks[(iteration * batch_size + i) % len(chunks)] for i in range(batch_size)])
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                objective = -model.log_prob(batch["audioSlices"].to(device), batch["notes"]).sum(-1).mean()
                if not torch.isfinite(objective):
                    raise FloatingPointError("Non-finite profile objective")
                (objective / 50).backward()
                clipper.step(model)
                optimizer.step()
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
            result.update({"fits": True, "seconds_per_batch": sum(times[1:]) / 2,
                           "last_nll": float(objective.detach()),
                           "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3 if str(device).startswith("cuda") else None,
                           "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3 if str(device).startswith("cuda") else None})
        except torch.cuda.OutOfMemoryError:
            result.update({"fits": False, "reason": "CUDA out of memory"})
        results.append(result)
        print(json.dumps(result), flush=True)
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        if "objective" in locals():
            del objective
        if "batch" in locals():
            del batch
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return results


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="Train original Transkun AMT on data/ audio-MIDI TXT partitions")
    parser.add_argument("--datasetRoot", required=True, help="Dataset directory under data/")
    parser.add_argument("--partitionDir", help="Absolute or dataset-relative directory of train.txt/val.txt/test.txt")
    parser.add_argument("--runDir", help="Output directory (default: Transkun/runs/<partition name>)")
    parser.add_argument("--modelConf", default=str(REPO / "configs/amt.conf"))
    parser.add_argument("--trainingConf", default=str(REPO / "configs/amt_training.conf"),
                        help="JSON Training preset; explicit CLI arguments override it")
    parser.add_argument("--initAMT", help="Pretrained original AMT checkpoint; omit to train from scratch")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validateOnly", action="store_true")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--weight", help="AMT weights for validateOnly, default: runDir/best.pt")
    parser.add_argument("--inspectOnly", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Benchmark GPU training memory/time and exit")
    parser.add_argument("--profileCases", default="1:16,4:16,12:16", help="batch:seconds comma-separated")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batchSize", type=int, default=12)
    parser.add_argument("--accumulationSteps", type=int, default=1)
    parser.add_argument("--chunkSize", type=float, default=16)
    parser.add_argument("--hopSize", type=float, help="Training window hop in seconds, default: chunkSize/2")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--nIter", type=int, help="Planned optimizer updates, also scheduler horizon (preset: 500000)")
    budget.add_argument("--epochs", type=int, help="Use an epoch budget instead of the preset nIter")
    parser.add_argument("--maxSteps", type=int, help="Stop after this total number of optimizer updates")
    parser.add_argument("--max_lr", type=float, default=4e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmupFraction", type=float, default=0.05)
    parser.add_argument("--gradClip", type=float, help="Override adaptive clipping with a fixed norm limit")
    parser.add_argument("--gradClippingQuantile", type=float, default=0.8)
    parser.add_argument("--gradClippingHistory", type=int, default=10000)
    parser.add_argument("--gradClippingInit", type=float, default=40)
    parser.add_argument("--dropLast", action=argparse.BooleanOptionalAction, default=True,
                        help="Drop incomplete physical train batches (default: enabled)")
    parser.add_argument("--dataLoaderWorkers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--saveEvery", type=int, default=50, help="Checkpoint every N optimizer updates")
    parser.add_argument("--validateEvery", type=int, default=1, help="Epoch validation frequency")
    parser.add_argument("--chunkMetrics", action="store_true", help="Faster chunk F1 instead of stitched full-file F1")
    parser.add_argument("--annotationBoundaryPolicy", choices=["clip", "error"], default="clip")
    parser.add_argument("--pedalExtension", action="store_true", help="Original piano sustain extension; off for saxophone")
    parser.add_argument("--allow_tf32", action="store_true")
    args = parser.parse_args(argv)
    explicit_iterations, explicit_epochs = args.nIter, args.epochs
    preset = json.loads(Path(args.trainingConf).read_text(encoding="utf-8"))
    allowed = {"batchSize", "accumulationSteps", "chunkSize", "hopSize", "nIter", "epochs",
               "max_lr", "weight_decay", "warmupFraction", "gradClip", "gradClippingQuantile",
               "gradClippingHistory", "gradClippingInit", "dropLast", "dataLoaderWorkers",
               "threads", "seed", "saveEvery", "validateEvery", "chunkMetrics", "allow_tf32"}
    values = preset.get("Training")
    if not isinstance(values, dict) or values.keys() - allowed:
        parser.error("Invalid Training preset or unknown fields")
    if values.get("epochs") is not None and values.get("nIter") is not None:
        parser.error("Training preset must choose epochs or nIter")
    parser.set_defaults(nIter=500000)
    parser.set_defaults(**values)
    args = parser.parse_args(argv)
    if explicit_iterations is not None:
        args.epochs = None
    elif explicit_epochs is not None or args.epochs is not None:
        args.nIter = None
    if args.hopSize is None:
        args.hopSize = args.chunkSize / 2
    if min(args.batchSize, args.accumulationSteps, args.threads, args.saveEvery, args.validateEvery,
           args.gradClippingHistory) < 1 or args.dataLoaderWorkers < 0:
        parser.error("Batch/epochs/accumulation/threads/frequencies must be positive; workers nonnegative")
    if (args.chunkSize <= 0 or not 0 < args.hopSize <= args.chunkSize or args.max_lr <= 0
            or (args.gradClip is not None and args.gradClip <= 0) or args.weight_decay < 0
            or args.gradClippingInit <= 0 or not 0 <= args.gradClippingQuantile <= 1
            or not 0 < args.warmupFraction < 1 or (args.maxSteps is not None and args.maxSteps < 1)
            or (args.epochs is not None and args.epochs < 1) or (args.nIter is not None and args.nIter < 1)
            or (args.epochs is None and args.nIter is None)):
        parser.error("Invalid duration, learning rate, gradient clip, weight decay or maxSteps")
    if args.resume and args.initAMT:
        parser.error("Choose resume or initAMT")
    return args


def main(argv=None):
    args = parse_arguments(argv)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    samples, audit = inspect_dataset(args.datasetRoot, args.partitionDir,
                                      args.annotationBoundaryPolicy, args.pedalExtension)
    run = Path(args.runDir) if args.runDir else REPO / "runs" / Path(audit["partition_dir"]).name
    run = run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if (run / "last.pt").exists() and not (args.resume or args.validateOnly or args.inspectOnly or args.profile):
        parser.error("Run already has last.pt; use --resume or a different runDir")
    logger = logging.getLogger("transkun.amt")
    for handler in logger.handlers:
        handler.close()
    logger.handlers = [logging.StreamHandler(), logging.FileHandler(run / "train.log", encoding="utf-8")]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in logger.handlers:
        handler.setFormatter(formatter)
    logger.info("Dataset: %s", json.dumps(audit["splits"]))
    if any(s["files_with_boundary_corrections"] for s in audit["splits"].values()):
        logger.warning("MIDI/audio boundary corrections applied; see dataset_audit.json. This does not repair internal timing alignment.")
    write_json(run / "dataset_audit.json", audit)
    if args.inspectOnly:
        print(json.dumps(audit, indent=2))
        return
    conf = read_config(args.modelConf)
    conf.segmentSizeInSecond = args.chunkSize
    conf.segmentHopSizeInSecond = args.chunkSize / 2
    loaded = None
    if args.resume:
        loaded = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
        if loaded.get("task") != "AMT" or loaded.get("format_version") != 2:
            raise ValueError("Resume requires AMT format 2 with clipping history; older AMT weights can be loaded with --initAMT")
        if loaded["config"] != conf.__dict__ or loaded["dataset_fingerprint"] != audit["fingerprint"]:
            raise ValueError("Resume config/dataset differs from checkpoint")
    if args.validateOnly:
        path = Path(args.weight) if args.weight else run / "best.pt"
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if loaded.get("task") != "AMT":
            raise ValueError("validateOnly requires a trained generic AMT checkpoint")
        conf.__dict__.update(loaded["config"])
    model = TransKun(conf).to(args.device)
    if loaded is not None:
        model.load_state_dict(loaded["state_dict"], strict=True)
    elif args.initAMT:
        initialize_weights(model, args.initAMT)
        logger.info("Initialized all AMT weights from %s", args.initAMT)
    write_json(run / "conf.json", model_config(conf))
    if args.profile:
        model.train()
        cases = [(int(case.split(":")[0]), float(case.split(":")[1])) for case in args.profileCases.split(",")]
        results = profile(model, samples["train"], args.device, cases, args)
        write_json(run / "profile.json", {"device": args.device,
                   "gpu_name": torch.cuda.get_device_name() if str(args.device).startswith("cuda") else None,
                   "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3 if str(args.device).startswith("cuda") else None,
                   "parameters": sum(p.numel() for p in model.parameters()), "cases": results})
        return
    if args.validateOnly:
        if args.split not in samples:
            raise ValueError(f"Missing {args.split}.txt")
        chunks = AMTChunks(samples[args.split], conf.fs, conf.segmentSizeInSecond)
        metrics = validate(model, samples[args.split], chunks, args.batchSize, args.dataLoaderWorkers,
                           args.device, full_files=not args.chunkMetrics)
        write_json(run / f"{args.split}_metrics.json", metrics)
        logger.info("%s: %s", args.split, json.dumps(metrics))
        return
    train_chunks = AMTChunks(samples["train"], conf.fs, args.chunkSize, args.hopSize,
                              training=True, seed=args.seed)
    val_chunks = AMTChunks(samples["val"], conf.fs, args.chunkSize)
    batches = len(train_chunks) // args.batchSize if args.dropLast else math.ceil(len(train_chunks) / args.batchSize)
    if not batches:
        raise ValueError("No full training batch; reduce batchSize or use --no-dropLast")
    updates_per_epoch = math.ceil(batches / args.accumulationSteps)
    optimizer = make_optimizer(model, args.max_lr, args.weight_decay)
    total_updates = args.nIter if args.nIter is not None else updates_per_epoch * args.epochs
    planned_epochs = math.ceil(total_updates / updates_per_epoch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.max_lr, total_steps=total_updates,
                    pct_start=args.warmupFraction, anneal_strategy="cos", cycle_momentum=False,
                    final_div_factor=2, div_factor=20)
    clipper = GradientClipper(args.gradClippingQuantile, args.gradClippingHistory,
                              args.gradClippingInit, args.gradClip)
    options = {"batchSize": args.batchSize, "accumulationSteps": args.accumulationSteps,
               "chunkSize": args.chunkSize, "hopSize": args.hopSize, "seed": args.seed,
               "epochs": args.epochs, "max_lr": args.max_lr, "weight_decay": args.weight_decay,
               "nIter": args.nIter, "warmupFraction": args.warmupFraction,
               "gradClip": args.gradClip, "gradClippingQuantile": args.gradClippingQuantile,
               "gradClippingHistory": args.gradClippingHistory, "gradClippingInit": args.gradClippingInit,
               "dropLast": args.dropLast, "validateEvery": args.validateEvery,
               "chunkMetrics": args.chunkMetrics, "allow_tf32": args.allow_tf32}
    epoch = next_batch = step = 0
    best = {"f1": -1.0, "nll": float("inf")}
    if loaded is not None:
        if loaded["training_options"] != options:
            raise ValueError("Resume training options differ; repeat original budget/batch/accumulation/LR/clipping")
        optimizer.load_state_dict(loaded["optimizer_state_dict"])
        scheduler.load_state_dict(loaded["lr_scheduler_state_dict"])
        clipper.load_state_dict(loaded["gradient_clipping_state_dict"])
        epoch, next_batch, step, best = loaded["epoch"], loaded["next_batch"], loaded["global_step"], loaded["best_metrics"]
        torch.set_rng_state(loaded["rng_state"])
        if loaded["cuda_rng_state"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(loaded["cuda_rng_state"])
    write_json(run / "run_config.json", {**vars(args), "resolved_run_dir": str(run),
               "train_windows": len(train_chunks), "validation_windows": len(val_chunks),
               "updates_per_epoch": updates_per_epoch, "total_updates": total_updates,
               "planned_epochs": planned_epochs, "warmup_updates": total_updates * args.warmupFraction,
               "effective_batch_size": args.batchSize * args.accumulationSteps,
               "parameters": sum(p.numel() for p in model.parameters())})
    writer = SummaryWriter(str(run / "tensorboard"))
    metrics_file = (run / "metrics.jsonl").open("a", encoding="utf-8")
    stopping = [False]

    def request_stop(signum, frame):
        stopping[0] = True
        logger.info("Stop requested; finishing the current gradient accumulation group")

    previous_signals = {sig: signal.signal(sig, request_stop) for sig in [signal.SIGINT, signal.SIGTERM]}
    started = time.perf_counter()
    phase = "running"

    def record(event):
        event.update({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "global_step": step})
        metrics_file.write(json.dumps(event, allow_nan=False) + "\n")
        metrics_file.flush()
        write_json(run / "status.json", {"phase": phase, "pid": os.getpid(), "epoch": epoch,
                   "next_batch": next_batch, "global_step": step, "elapsed_seconds": time.perf_counter() - started,
                   "latest": event})

    logger.info("Training %d windows, %d optimizer updates/epoch, batch=%d accumulation=%d device=%s",
                len(train_chunks), updates_per_epoch, args.batchSize, args.accumulationSteps, args.device)
    if loaded is None:
        save_checkpoint(run / "last.pt", model, optimizer, scheduler, epoch, next_batch, step, best, audit, options, clipper)
    record({"kind": "started", "total_epochs": planned_epochs, "total_updates": total_updates})
    try:
        while step < total_updates and not stopping[0] and not (args.maxSteps and step >= args.maxSteps):
            train_chunks.set_epoch(epoch)
            generator = torch.Generator().manual_seed(args.seed + epoch)
            loader = DataLoader(train_chunks, batch_size=args.batchSize, shuffle=True, generator=generator,
                                collate_fn=collate_amt, num_workers=args.dataLoaderWorkers, drop_last=args.dropLast)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            group_loss = group_samples = 0
            group_start = time.perf_counter()
            for index, batch in enumerate(loader):
                if index < next_batch:
                    continue
                group_first = (index // args.accumulationSteps) * args.accumulationSteps
                group_end = min(group_first + args.accumulationSteps, batches)
                group_size = group_end - group_first
                audio = batch["audioSlices"].to(args.device)
                objective = -model.log_prob(audio, batch["notes"]).sum(-1).mean()
                if not torch.isfinite(objective):
                    raise FloatingPointError(f"Non-finite AMT loss at epoch {epoch} batch {index}")
                (objective / (50 * group_size)).backward()
                group_loss += float(objective.detach()) * len(batch["notes"])
                group_samples += len(batch["notes"])
                if index + 1 != group_end:
                    continue
                norm, clip_limit = clipper.step(model)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                next_batch = index + 1
                event = {"kind": "train", "epoch": epoch + 1, "batch": next_batch, "batches": batches,
                         "nll_per_second": group_loss / (group_samples * args.chunkSize),
                         "lr": optimizer.param_groups[0]["lr"], "grad_norm": float(norm), "clip_limit": clip_limit,
                         "seconds_per_update": time.perf_counter() - group_start,
                         "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3 if str(args.device).startswith("cuda") else None}
                record(event)
                for key in ["nll_per_second", "lr", "grad_norm", "clip_limit"]:
                    writer.add_scalar("train/" + key, event[key], step)
                logger.info("epoch=%d batch=%d/%d step=%d nll/s=%.4f lr=%.3g update=%.2fs",
                            epoch + 1, next_batch, batches, step, event["nll_per_second"], event["lr"], event["seconds_per_update"])
                if step % args.saveEvery == 0 or stopping[0] or (args.maxSteps and step >= args.maxSteps):
                    save_checkpoint(run / "last.pt", model, optimizer, scheduler, epoch, next_batch, step, best, audit, options, clipper)
                group_loss = group_samples = 0
                group_start = time.perf_counter()
                if stopping[0] or step >= total_updates or (args.maxSteps and step >= args.maxSteps):
                    break
            finished_epoch = next_batch == batches
            finished_plan = step >= total_updates
            if not finished_epoch and not finished_plan:
                break
            checkpoint_epoch = epoch + 1 if finished_epoch else epoch
            checkpoint_batch = 0 if finished_epoch else next_batch
            if (epoch + 1) % args.validateEvery == 0 or finished_plan:
                logger.info("Validating epoch %d (whole-file note F1=%s)", epoch + 1, not args.chunkMetrics)
                metrics = validate(model, samples["val"], val_chunks, args.batchSize, args.dataLoaderWorkers,
                                   args.device, full_files=not args.chunkMetrics)
                record({"kind": "validation", "epoch": epoch + 1, **metrics})
                for key in ["nll_per_second", "precision", "recall", "note_offset_f1"]:
                    writer.add_scalar("val/" + key, metrics[key], step)
                logger.info("Validation: %s", json.dumps(metrics))
                if metrics["note_offset_f1"] > best["f1"] or (metrics["note_offset_f1"] == best["f1"] and metrics["nll_per_second"] < best["nll"]):
                    best = {"f1": metrics["note_offset_f1"], "nll": metrics["nll_per_second"]}
                    save_checkpoint(run / "best.pt", model, optimizer, scheduler, checkpoint_epoch,
                                    checkpoint_batch, step, best, audit, options, clipper)
            epoch, next_batch = checkpoint_epoch, checkpoint_batch
            save_checkpoint(run / "last.pt", model, optimizer, scheduler, epoch, next_batch, step, best, audit, options, clipper)
            writer.flush()
        phase = "completed" if step >= total_updates else "stopped"
        save_checkpoint(run / "last.pt", model, optimizer, scheduler, epoch, next_batch, step, best, audit, options, clipper)
        record({"kind": phase, "completed_epochs": epoch})
    except Exception as exc:
        phase = "failed"
        record({"kind": "failed", "error": str(exc)})
        logger.exception("AMT training failed; resume uses the latest completed accumulation checkpoint")
        raise
    finally:
        for sig, handler in previous_signals.items():
            signal.signal(sig, handler)
        writer.close()
        metrics_file.close()


if __name__ == "__main__":
    main()

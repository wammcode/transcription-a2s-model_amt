"""Train/validate Transkun for score tokens, without Lightning or another model."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .CheckpointA2S import load_checkpoint, read_config, save_checkpoint, warm_start_amt
from .DataScore import ScoreDataset, collate_score
from .ModelA2S import TransKun
from .Score import ScoreCodec


def edit_distance(reference, prediction):
    previous = list(range(len(prediction) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, pred in enumerate(prediction, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (ref != pred)))
        previous = current
    return previous[-1]


def move_batch(batch, device):
    # CTC length tensors stay on CPU; waveforms and targets follow the model.
    return {**batch, "audioSlices": batch["audioSlices"].to(device),
            "targets": batch["targets"].to(device)}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    loss_sum = count = errors = words = sequence_errors = valid = 0
    rejected = fallback = 0
    codec = ScoreCodec(model.conf.scoreCodecVersion)
    for batch in loader:
        batch = move_batch(batch, device)
        logits, lengths = model(batch["audioSlices"], batch["audioLengths"])
        loss = model.loss_from_logits(logits, lengths, batch["targets"], batch["targetLengths"])
        predictions = model.decode_logits(logits, lengths)
        loss_sum += loss.item() * len(predictions)
        count += len(predictions)
        for reference, prediction in zip(batch["tokens"], predictions):
            distance = edit_distance(reference, prediction)
            errors += distance
            words += len(reference)
            sequence_errors += distance != 0
            try:
                codec.decode(prediction, strict=True)
                valid += 1
            except ValueError:
                pass
            _, diagnostics = codec.decode(prediction)
            rejected += diagnostics["rejected_records"]
            fallback += diagnostics["fallback"]
    if count == 0:
        raise ValueError("Empty validation loader")
    return {"ctc_loss": loss_sum / count, "sym_er": 100 * errors / max(words, 1),
            "seq_er": 100 * sequence_errors / count, "strict_score_rate": valid / count,
            "fallback_rate": fallback / count, "rejected_records": rejected, "samples": count}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train/validate Transkun A2S")
    parser.add_argument("saved_filename", help="A2S checkpoint (last epoch)")
    parser.add_argument("--datasetPath", help="Base directory for manifest paths (default: manifest directory)")
    parser.add_argument("--datasetMetaFile_train", help="Paired training JSON/JSONL")
    parser.add_argument("--datasetMetaFile_val", required=True, help="Paired validation JSON/JSONL")
    parser.add_argument("--modelConf", help="moduleconf-style JSON selecting transkun.ModelA2S")
    parser.add_argument("--initAMT", help="Initialize original frontend/backbone/AMT heads from a V2 checkpoint")
    parser.add_argument("--resume", action="store_true", help="Resume saved_filename including optimizer/RNG")
    parser.add_argument("--validateOnly", action="store_true", help="Evaluate saved_filename and exit")
    parser.add_argument("--freezeBackbone", action="store_true", help="Freeze frontend/backbone; train score head")
    parser.add_argument("--noAutoFrameMultiplier", action="store_true",
                        help="Require configured CTC frame multiplier instead of fitting target lengths on fresh training")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batchSize", type=int, default=4)
    parser.add_argument("--dataLoaderWorkers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50, help="Total epochs, including already completed epochs")
    parser.add_argument("--max_lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradClip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.batchSize < 1 or args.epochs < 1 or args.dataLoaderWorkers < 0:
        parser.error("batchSize/epochs must be positive and dataLoaderWorkers nonnegative")
    if args.max_lr <= 0 or args.gradClip <= 0 or args.weight_decay < 0:
        parser.error("Invalid learning rate, gradient clip or weight decay")
    if args.initAMT and (args.resume or args.validateOnly):
        parser.error("initAMT cannot be combined with resume/validateOnly")
    if not args.validateOnly and not args.datasetMetaFile_train:
        parser.error("Training requires datasetMetaFile_train")
    if not args.resume and not args.validateOnly and Path(args.saved_filename).exists():
        parser.error("Checkpoint already exists; use --resume or a new saved_filename")
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    checkpoint = None
    if args.resume or args.validateOnly:
        model, checkpoint = load_checkpoint(args.saved_filename, args.device)
        conf, vocabulary = model.conf, model.vocabulary
        if args.modelConf:
            requested = read_config(args.modelConf)
            if not args.noAutoFrameMultiplier:
                requested.scoreFrameMultiplier = max(requested.scoreFrameMultiplier, conf.scoreFrameMultiplier)
            if requested.__dict__ != conf.__dict__:
                parser.error("modelConf differs from resumed checkpoint config")
    else:
        conf = read_config(args.modelConf)
        train_set = ScoreDataset(args.datasetMetaFile_train, conf, dataset_path=args.datasetPath,
                                 check_ctc_lengths=args.noAutoFrameMultiplier)
        vocabulary = train_set.vocabulary
        val_set = ScoreDataset(args.datasetMetaFile_val, conf, vocabulary, args.datasetPath,
                               check_ctc_lengths=args.noAutoFrameMultiplier)
        configured = conf.scoreFrameMultiplier
        if not args.noAutoFrameMultiplier:
            conf.scoreFrameMultiplier = max(configured, train_set.required_frame_multiplier,
                                             val_set.required_frame_multiplier)
        if configured != conf.scoreFrameMultiplier:
            print(json.dumps({"configured_frame_multiplier": configured,
                              "resolved_frame_multiplier": conf.scoreFrameMultiplier}))
        model = TransKun(conf, vocabulary).to(args.device)
        if args.initAMT:
            print(json.dumps({"amt_tensors_loaded": warm_start_amt(model, args.initAMT)}))
    if checkpoint is not None:
        val_set = ScoreDataset(args.datasetMetaFile_val, conf, vocabulary, args.datasetPath)
    val_loader = DataLoader(val_set, batch_size=args.batchSize, collate_fn=collate_score,
                            num_workers=args.dataLoaderWorkers)
    if args.validateOnly:
        print(json.dumps(validate(model, val_loader, args.device), sort_keys=True))
        return
    if checkpoint is not None:
        train_set = ScoreDataset(args.datasetMetaFile_train, conf, vocabulary, args.datasetPath)
    if args.freezeBackbone:
        model.framewiseFeatureExtractor.requires_grad_(False)
        model.backbone.requires_grad_(False)
    # Include frozen backbone parameters so optimizer state remains compatible
    # when switching between head-only warmup and whole-backbone fine tuning.
    parameters = list(model.framewiseFeatureExtractor.parameters()) + list(model.backbone.parameters())
    parameters += list(model.scoreProjection.parameters()) + list(model.scoreOutput.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.max_lr, weight_decay=args.weight_decay)
    epoch, step, best_loss = 0, 0, float("inf")
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch, step, best_loss = checkpoint["epoch"], checkpoint["step"], checkpoint["best_loss"]
        torch.set_rng_state(checkpoint["rng_state"])
        if checkpoint.get("cuda_rng_state") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        if checkpoint.get("loader_rng_state") is not None:
            generator.set_state(checkpoint["loader_rng_state"])
    train_loader = DataLoader(train_set, batch_size=args.batchSize, shuffle=True,
                              collate_fn=collate_score, num_workers=args.dataLoaderWorkers,
                              generator=generator)
    for current_epoch in range(epoch, args.epochs):
        model.train()
        if args.freezeBackbone:
            model.framewiseFeatureExtractor.eval()
            model.backbone.eval()
        loss_sum = samples = 0
        for batch in train_loader:
            batch = move_batch(batch, args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.compute_loss(batch["audioSlices"], batch["audioLengths"],
                                      batch["targets"], batch["targetLengths"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, args.gradClip, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            loss_sum += loss.item() * len(batch["tokens"])
            samples += len(batch["tokens"])
        metrics = validate(model, val_loader, args.device)
        metrics["train_ctc_loss"] = loss_sum / samples
        is_best = metrics["ctc_loss"] < best_loss
        best_loss = min(best_loss, metrics["ctc_loss"])
        save_checkpoint(args.saved_filename, model, optimizer, current_epoch + 1,
                        step, best_loss, metrics, generator)
        if is_best:
            path = Path(args.saved_filename)
            best_path = path.with_name(path.stem + ".best" + path.suffix)
            save_checkpoint(best_path, model, optimizer, current_epoch + 1,
                            step, best_loss, metrics, generator)
        print(json.dumps({"epoch": current_epoch + 1, "step": step, **metrics}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

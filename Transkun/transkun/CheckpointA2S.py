"""Self-contained, explicitly versioned A2S checkpoints; strict AMT warm starts."""
import json
from pathlib import Path

import torch

from .ModelA2S import ModelConfig, TransKun
from .Score import SUPPORTED_CODECS, Vocabulary

FORMAT_VERSION = 1


def read_config(path=None):
    conf = ModelConfig()
    if path is not None:
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
        model = document.get("Model", {})
        if model.get("module") != "transkun.ModelA2S":
            raise ValueError("modelConf must select transkun.ModelA2S")
        options = model["config"]
        unknown = options.keys() - conf.__dict__.keys()
        if unknown:
            raise ValueError(f"Unknown A2S config keys: {sorted(unknown)}")
        conf.__dict__.update(options)
    conf.validate()
    return conf


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("task") != "A2S" or checkpoint.get("format_version") != FORMAT_VERSION
            or checkpoint.get("codec_version") not in SUPPORTED_CODECS):
        raise ValueError("Expected a compatible Transkun A2S checkpoint")
    conf = ModelConfig()
    conf.__dict__.update(checkpoint["config"])
    if "scoreCodecVersion" in checkpoint["config"] and conf.scoreCodecVersion != checkpoint["codec_version"]:
        raise ValueError("Checkpoint config/codec mismatch")
    conf.scoreCodecVersion = checkpoint["codec_version"]
    vocabulary = Vocabulary(checkpoint["vocabulary"])
    model = TransKun(conf, vocabulary)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device), checkpoint


def warm_start_amt(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("task") == "A2S":
        raise ValueError("Use --resume for A2S checkpoints; --initAMT requires AMT")
    state = checkpoint.get("best_state_dict")
    if state is None:
        state = checkpoint.get("state_dict", checkpoint)
    original = {k: v for k, v in model.state_dict().items()
                if not k.startswith(("scoreProjection.", "scoreOutput."))}
    if state.keys() != original.keys():
        raise ValueError("AMT checkpoint keys do not match Transkun V2; use its original config")
    mismatched = [key for key in original if original[key].shape != state[key].shape]
    if mismatched:
        raise ValueError(f"AMT checkpoint shapes mismatch: {mismatched[:5]}; use its original config")
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(not k.startswith(("scoreProjection.", "scoreOutput."))
                                     for k in result.missing_keys):
        raise ValueError("Incomplete AMT initialization")
    return len(original)


def save_checkpoint(path, model, optimizer, epoch, step, best_loss, metrics, generator=None):
    checkpoint = {"task": "A2S", "format_version": FORMAT_VERSION,
                  "codec_version": model.conf.scoreCodecVersion, "config": dict(model.conf.__dict__),
                  "vocabulary": model.vocabulary.tokens, "state_dict": model.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch,
                  "step": step, "best_loss": best_loss, "metrics": metrics,
                  "rng_state": torch.get_rng_state(),
                  "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                  "loader_rng_state": generator.get_state() if generator is not None else None}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)

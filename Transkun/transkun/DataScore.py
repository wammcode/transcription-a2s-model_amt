"""Paired waveform/score datasets; audio is never cropped without its target."""
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .Score import ScoreCodec, Vocabulary


def read_audio(path, fs, max_seconds=None):
    info = sf.info(str(path))
    if info.frames <= 0:
        raise ValueError(f"Empty audio: {path}")
    if max_seconds is not None and info.frames / info.samplerate > max_seconds + 1e-6:
        raise ValueError(f"{path}: audio exceeds {max_seconds}s; prepare paired score/audio clips")
    audio, original_fs = sf.read(str(path), dtype="float32", always_2d=True)
    # Mono averaging is also performed by Transkun's original MelSpectrum.
    audio = audio.mean(axis=1, keepdims=True)
    if original_fs != fs:
        gcd = math.gcd(original_fs, fs)
        audio = resample_poly(audio, fs // gcd, original_fs // gcd, axis=0)
    if not np.isfinite(audio).all():
        raise ValueError(f"Non-finite audio: {path}")
    return torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))


class ScoreDataset(Dataset):
    def __init__(self, manifest, conf, vocabulary=None, dataset_path=None, check_ctc_lengths=True):
        manifest = Path(manifest)
        root = Path(dataset_path) if dataset_path else manifest.parent
        with manifest.open(encoding="utf-8") as handle:
            if manifest.suffix.lower() == ".jsonl":
                rows = [json.loads(line) for line in handle if line.strip()]
            else:
                rows = json.load(handle)
        if not isinstance(rows, list) or not rows:
            raise ValueError("Manifest must be a nonempty JSON list or JSONL")
        self.conf = conf
        self.samples = []
        codec = ScoreCodec(conf.scoreCodecVersion)
        for index, row in enumerate(rows):
            try:
                audio = root / row["audio"]
                if not audio.is_file():
                    raise ValueError(f"Missing audio: {audio}")
                target_fields = [field for field in ("score", "tokens", "transcript") if field in row]
                if len(target_fields) != 1:
                    raise ValueError("Specify exactly one of score, tokens or inline Kern transcript")
                if "score" in row:
                    tokens = codec.encode_file(root / row["score"])
                elif "transcript" in row:
                    tokens = codec.encode_kern(row["transcript"])
                else:
                    if row.get("codec_version") != codec.version:
                        raise ValueError("Preencoded tokens require the matching codec_version")
                    tokens = row["tokens"]
                if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
                    raise ValueError("Tokens must be a list of strings")
                codec.decode(tokens, strict=True)
                self.samples.append({"audio_path": audio, "tokens": tokens})
            except Exception as exc:
                raise ValueError(f"Invalid manifest sample {index} in {manifest}: {exc}") from exc
        # Build only from training samples. Validation/test must use its vocabulary.
        self.vocabulary = vocabulary if vocabulary is not None else Vocabulary.build(
            (sample["tokens"] for sample in self.samples), codec_version=codec.version)
        self.required_frame_multiplier = 1
        for sample in self.samples:
            sample["targets"] = torch.tensor(self.vocabulary.encode(sample["tokens"]), dtype=torch.long)
            info = sf.info(str(sample["audio_path"]))
            seconds = info.frames / info.samplerate
            if seconds > conf.scoreMaxAudioSeconds + 1e-6:
                raise ValueError(f"{sample['audio_path']}: clip exceeds scoreMaxAudioSeconds")
            # scipy's resample_poly returns ceil(N * new_fs / old_fs) samples.
            n_samples = math.ceil(info.frames * conf.fs / info.samplerate)
            base_frames = math.ceil(n_samples / conf.hopSize) + 1
            frames = base_frames * conf.scoreFrameMultiplier
            target = sample["targets"]
            required = len(target) + int((target[1:] == target[:-1]).sum())
            self.required_frame_multiplier = max(self.required_frame_multiplier, math.ceil(required / base_frames))
            if check_ctc_lengths and required > frames:
                raise ValueError(f"{sample['audio_path']}: CTC needs {required} frames, has {frames}; "
                                 "increase scoreFrameMultiplier or reduce hopSize")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return {"audio": read_audio(sample["audio_path"], self.conf.fs, self.conf.scoreMaxAudioSeconds),
                "targets": sample["targets"], "tokens": sample["tokens"]}


def collate_score(batch):
    return {"audioSlices": pad_sequence([sample["audio"] for sample in batch], batch_first=True),
            "audioLengths": torch.tensor([len(sample["audio"]) for sample in batch], dtype=torch.long),
            "targets": pad_sequence([sample["targets"] for sample in batch], batch_first=True, padding_value=0),
            "targetLengths": torch.tensor([len(sample["targets"]) for sample in batch], dtype=torch.long),
            "tokens": [sample["tokens"] for sample in batch]}

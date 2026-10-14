"""Generic AMT datasets rooted in data/, with audio/MIDI TXT partitions."""
import hashlib
import math
from pathlib import Path
import random
import shlex

import numpy as np
import pretty_midi
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from .Data import Note, parseEventAll, resolveOverlapping, validateNotes


def find_partitions(root, partition_dir=None):
    root = Path(root).resolve()
    if partition_dir:
        directory = Path(partition_dir)
        directory = directory if directory.is_absolute() else root / directory
    else:
        candidates = [root / "amt_partitions"]
        if candidates[0].is_dir():
            candidates += sorted(path for path in candidates[0].iterdir() if path.is_dir())
        candidates = [path for path in candidates if (path / "train.txt").is_file() and (path / "val.txt").is_file()]
        if len(candidates) != 1:
            raise ValueError("Specify --partitionDir: expected exactly one train.txt/val.txt partition directory")
        directory = candidates[0]
    if not (directory / "train.txt").is_file() or not (directory / "val.txt").is_file():
        raise ValueError(f"Missing train.txt/val.txt in {directory}")
    return directory.resolve()


def read_partition(root, filename):
    root = Path(root).resolve()
    rows, seen = [], set()
    for number, line in enumerate(Path(filename).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split("\t") if "\t" in line else shlex.split(line)
        if len(fields) != 2:
            raise ValueError(f"{filename}:{number}: expected audio_path<TAB>midi_path")
        audio, midi = [(root / field.strip()).resolve() for field in fields]
        if not audio.is_file() or not midi.is_file():
            raise ValueError(f"{filename}:{number}: missing audio/MIDI file: {fields}")
        if audio in seen:
            raise ValueError(f"{filename}:{number}: duplicate audio {audio}")
        seen.add(audio)
        rows.append((audio, midi))
    if not rows:
        raise ValueError(f"Empty partition: {filename}")
    return rows


def parse_midi(path, duration, boundary_policy="clip", pedal_extension=False):
    midi = pretty_midi.PrettyMIDI(str(path))
    events, tracks, total = [], 0, 0
    for instrument in midi.instruments:
        if instrument.is_drum:
            raise ValueError(f"Unpitched percussion is unsupported by the original Transkun heads: {path}")
        tracks += 1
        total += len(instrument.notes)
        if any(not 21 <= n.pitch <= 108 for n in instrument.notes):
            raise ValueError(f"MIDI pitches outside original Transkun range 21..108: {path}")
        if instrument.notes:
            events.extend(parseEventAll(instrument.notes, instrument.control_changes,
                                        supportedCC=[64, 67], extendSustainPedal=pedal_extension))
    # Original AMT is pitch-based, not instrument/program-aware. Resolve
    # same-pitch overlap across tracks using its original policy.
    events = resolveOverlapping(events)
    clipped = dropped = 0
    result = []
    for event in events:
        if not (math.isfinite(event.start) and math.isfinite(event.end)):
            raise ValueError(f"Non-finite MIDI timestamp: {path}")
        outside = event.start < 0 or event.end > duration
        if outside and boundary_policy == "error":
            raise ValueError(f"MIDI event outside audio duration ({duration:.3f}s): {path}: {event}")
        if event.start >= duration or event.end <= 0:
            dropped += 1
            continue
        begin, end = max(event.start, 0), min(event.end, duration)
        clipped += int(outside)
        result.append(Note(float(begin), float(end), int(event.pitch), int(event.velocity),
                           event.hasOnset and event.start >= 0, event.hasOffset and event.end <= duration))
    result.sort(key=lambda n: (n.start, n.end, n.pitch))
    validateNotes(result)
    return result, {"midi_notes": total, "tracks": tracks, "events_clipped": clipped,
                    "events_dropped_outside_audio": dropped,
                    "events_after_normalization": len(result),
                    "pitches": [n.pitch for n in result if n.pitch > 0]}


def inspect_dataset(root, partition_dir=None, boundary_policy="clip", pedal_extension=False):
    if boundary_policy not in {"clip", "error"}:
        raise ValueError("annotationBoundaryPolicy must be clip or error")
    root = Path(root).resolve()
    directory = find_partitions(root, partition_dir)
    samples, reports, seen = {}, {}, {}
    fingerprint = hashlib.sha256()
    fingerprint.update(f"{boundary_policy}:{pedal_extension}".encode())
    for split in ["train", "val", "test"]:
        filename = directory / f"{split}.txt"
        if not filename.is_file():
            continue
        fingerprint.update(filename.read_bytes())
        entries = []
        rates, channels, durations, pitches = {}, {}, [], []
        totals = {"midi_notes": 0, "events_clipped": 0, "events_dropped_outside_audio": 0,
                  "events_after_normalization": 0, "files_with_boundary_corrections": 0,
                  "files_without_notes_inside_audio": 0}
        for audio, midi in read_partition(root, filename):
            if audio in seen:
                raise ValueError(f"Audio appears in both {seen[audio]} and {split}: {audio}")
            seen[audio] = split
            info = sf.info(str(audio))
            if info.frames <= 0 or info.samplerate <= 0:
                raise ValueError(f"Empty/invalid audio: {audio}")
            duration = float(info.frames / info.samplerate)
            notes, metadata = parse_midi(midi, duration, boundary_policy, pedal_extension)
            for path in [audio, midi]:
                stat = path.stat()
                fingerprint.update(f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode())
            entries.append({"audio": audio, "midi": midi, "duration": duration,
                            "fs": int(info.samplerate), "channels": int(info.channels), "notes": notes})
            durations.append(duration)
            rates[str(info.samplerate)] = rates.get(str(info.samplerate), 0) + 1
            channels[str(info.channels)] = channels.get(str(info.channels), 0) + 1
            pitches.extend(metadata["pitches"])
            for field in ["midi_notes", "events_clipped", "events_dropped_outside_audio", "events_after_normalization"]:
                totals[field] += metadata[field]
            totals["files_with_boundary_corrections"] += int(metadata["events_clipped"] + metadata["events_dropped_outside_audio"] > 0)
            totals["files_without_notes_inside_audio"] += int(not metadata["pitches"])
        samples[split] = entries
        reports[split] = {"files": len(entries), "seconds": sum(durations), "hours": sum(durations) / 3600,
                          "duration_min": min(durations), "duration_max": max(durations),
                          "duration_median": float(np.median(durations)), "sample_rates": rates,
                          "channels": channels, "pitch_min": min(pitches) if pitches else None,
                          "pitch_max": max(pitches) if pitches else None, **totals}
    return samples, {"dataset_root": str(root), "partition_dir": str(directory),
                      "annotation_boundary_policy": boundary_policy, "pedal_extension": pedal_extension,
                      "fingerprint": fingerprint.hexdigest(), "splits": reports}


def read_chunk(path, begin, seconds, target_fs):
    samples = int(round(seconds * target_fs))
    if begin < 0 or seconds <= 0 or samples < 1:
        raise ValueError("Invalid audio chunk position/duration")
    with sf.SoundFile(str(path)) as handle:
        original_fs = handle.samplerate
        start = min(int(round(begin * original_fs)), len(handle))
        frames = int(round(seconds * original_fs))
        handle.seek(start)
        audio = handle.read(frames, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1, keepdims=True)
    if len(audio) and original_fs != target_fs:
        gcd = math.gcd(original_fs, target_fs)
        audio = resample_poly(audio, target_fs // gcd, original_fs // gcd, axis=0)
    audio = audio[:samples]
    if len(audio) < samples:
        audio = np.pad(audio, ((0, samples - len(audio)), (0, 0)))
    if not np.isfinite(audio).all():
        raise ValueError(f"Non-finite waveform: {path}")
    return torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))


class AMTChunks(Dataset):
    def __init__(self, samples, fs, chunk_seconds=4, hop_seconds=None, training=False, seed=42):
        self.samples, self.fs, self.seconds = samples, fs, chunk_seconds
        self.training, self.seed, self.epoch = training, seed, 0
        hop = chunk_seconds if hop_seconds is None else hop_seconds
        if chunk_seconds <= 0 or hop <= 0 or hop > chunk_seconds:
            raise ValueError("Require 0 < hopSize <= chunkSize")
        self.hop = hop
        self.windows = [(index, step * hop) for index, sample in enumerate(samples)
                        for step in range(max(1, math.ceil(sample["duration"] / hop)))]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        sample_index, begin = self.windows[index]
        sample = self.samples[sample_index]
        if self.training:
            rng = random.Random(self.seed + self.epoch * 1000003 + index)
            begin = max(0, min(begin + rng.uniform(-self.hop / 2, self.hop / 2),
                               max(0, sample["duration"] - self.seconds)))
        end = min(begin + self.seconds, sample["duration"])
        notes = [Note(max(n.start, begin) - begin, min(n.end, end) - begin,
                      n.pitch, n.velocity, n.hasOnset and n.start >= begin,
                      n.hasOffset and n.end <= end)
                 for n in sample["notes"] if n.start < end and n.end > begin]
        return {"audio": read_chunk(sample["audio"], begin, self.seconds, self.fs),
                "notes": notes, "duration": max(end - begin, 0), "begin": begin}


def collate_amt(batch):
    return {"audioSlices": torch.stack([item["audio"] for item in batch]),
            "notes": [item["notes"] for item in batch],
            "durations": [item["duration"] for item in batch]}

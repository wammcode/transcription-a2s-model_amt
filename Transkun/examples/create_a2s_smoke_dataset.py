"""Create a tiny synthetic pipeline fixture, not an A2S research benchmark."""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from music21 import clef, key, meter, note, stream, tempo
from transkun.Score import ScoreCodec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputPath")
    args = parser.parse_args()
    root = Path(args.outputPath)
    root.mkdir(parents=True, exist_ok=True)
    score = stream.Score()
    part = stream.Part()
    for obj in [meter.TimeSignature("4/4"), key.KeySignature(0), clef.TrebleClef(),
                tempo.MetronomeMark(number=240), note.Note("C4", quarterLength=4)]:
        part.insert(0, obj)
    score.insert(0, part)
    codec = ScoreCodec()
    codec.write(codec.encode(score), root / "score.krn", strict=True)
    fs = 8000
    t = np.arange(fs) / fs
    audio = 0.1 * np.sin(2 * np.pi * 261.626 * t) * np.minimum(t / 0.01, 1) * np.minimum((1 - t) / 0.01, 1)
    rows = []
    for index, tail in enumerate([0, fs // 4]):
        sf.write(root / f"audio{index}.wav", np.pad(audio, (0, tail)), fs)
        rows.append({"audio": f"audio{index}.wav", "score": "score.krn"})
    manifest = "\n".join(json.dumps(row) for row in rows) + "\n"
    # Deliberately reuse the same two samples for a plumbing/overfitting check.
    # Real experiments must use disjoint train/validation/test performances.
    for split in ["train", "val"]:
        (root / f"{split}.jsonl").write_text(manifest, encoding="utf-8")
    print(json.dumps({"directory": str(root), "samples": len(rows), "synthetic": True}))


if __name__ == "__main__":
    main()

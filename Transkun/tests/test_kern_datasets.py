"""Notation and training coverage for mono, closed and open polyphony."""
from fractions import Fraction
import json

import numpy as np
import pytest
import soundfile as sf
import torch
from music21 import chord, clef, converter, key, meter, note, stream

from transkun.CheckpointA2S import load_checkpoint, read_config
from transkun.DataScore import ScoreDataset, collate_score
from transkun.Kern import loads_kern, reciprocal
from transkun.ModelA2S import TransKun
from transkun.Score import ScoreCodec, Vocabulary
from transkun.trainA2S import main as train_main


MONO = "**kern\n*clefG2\n*M4/4\n4c\n4d\n2r\n*-\n"
POLY = "**kern\n*clefG2\n*M4/4\n4c 4e 4g\n4d 4f 4a\n2r\n*-\n"
OPEN = "**kern\t**kern\n*M4/4\t*M4/4\n2c\t4e\n.\t4f\n2d\t2g\n*-\t*-\n"
DYNAMIC = "**kern\n*M4/4\n4c\n*^\n4d\t4e\n4f\t4g\n*v\t*v\n4a\n*-\n"
ADDED = "**kern\n4c\n*+\n*\t**kern\n4d\t4e\n*-\t*\n4f\n*-\n"


def events(score):
    result = []
    for part in score.parts:
        for obj in part.flatten().notes:
            pitches = obj.pitches if isinstance(obj, chord.Chord) else [obj.pitch]
            for p in pitches:
                result.append((Fraction(obj.offset), Fraction(obj.quarterLength), p.nameWithOctave))
    return sorted(result)


@pytest.mark.parametrize("text", [MONO, POLY, OPEN, DYNAMIC, ADDED])
def test_kern_notation_roundtrip_all_textures(text, tmp_path):
    codec = ScoreCodec()
    tokens = codec.encode_kern(text)
    target, _ = codec.decode(tokens, strict=True)
    path = tmp_path / "score.krn"
    diagnostics = codec.write(tokens, path, strict=True)
    assert not diagnostics["fallback"]
    assert path.read_text().startswith("**kern") and path.read_text().endswith("*-\n")
    assert events(loads_kern(path.read_text())) == events(target)
    # Independent music21 parsing of the exported fixed-spine score.
    assert events(converter.parse(str(path), format="humdrum")) == events(target)


def test_added_spine_uses_shared_clock_and_auxiliary_spines():
    parsed = loads_kern(ADDED)
    assert (Fraction(1), Fraction(1), "E4") in events(parsed)
    assert (Fraction(2), Fraction(1), "F4") in events(parsed)
    text = "**kern\t**text\n4c\thello\n*x\t*x\nworld\t4d\n*-\t*-\n"
    assert events(loads_kern(text)) == [(Fraction(0), Fraction(1), "C4"),
                                      (Fraction(1), Fraction(1), "D4")]
    mid_note_split = ("**kern\t**kern\n2c\t4e\n*^\t*\n.\t4g\t4f\n"
                      "*v\t*v\t*\n2d\t2a\n*-\t*-\n")
    assert (Fraction(1), Fraction(1), "G4") in events(loads_kern(mid_note_split))


def test_unlabelled_open_polyphony_is_colored_into_non_overlapping_voices(tmp_path):
    score = stream.Score()
    part = stream.Part()
    part.insert(0, note.Note("C4", quarterLength=2))
    part.insert(0, note.Note("E4", quarterLength=1))
    part.insert(1, note.Note("F4", quarterLength=1))
    part.insert(2, note.Note("D4", quarterLength=2))
    score.insert(0, part)
    codec = ScoreCodec()
    tokens = codec.encode(score)
    normalized, diagnostics = codec.decode(tokens, strict=True)
    assert diagnostics["events"] == 4
    assert len(normalized.parts[0].getElementsByClass(stream.Voice)) == 2
    assert events(normalized) == events(score)
    codec.write(tokens, tmp_path / "open.krn", strict=True)
    assert events(converter.parse(str(tmp_path / "open.krn"))) == events(score)
    # Equal-onset/equal-duration notes become a chord, not discarded overlaps.
    part.insert(2, note.Note("F4", quarterLength=2))
    normalized, _ = codec.decode(codec.encode(score), strict=True)
    assert any(isinstance(n, chord.Chord) for n in normalized.recurse().notes)


def test_compositional_targets_have_no_fixed_part_voice_count_or_unseen_number_oov():
    codec = ScoreCodec()
    vocabulary = Vocabulary.build([codec.encode_kern(MONO)])
    semantic = ["AT_17/3", "PART_80", "VOICE_120", "DUR_5/3", "PITCH_F#5", "TIE_none", "END"]
    tokens = codec._serialize(semantic)
    assert vocabulary.encode(tokens)
    score, diagnostics = codec.decode(tokens, strict=True)
    assert diagnostics["events"] == 1 and score.parts[0].id == "P80"
    assert events(score) == [(Fraction(17, 3), Fraction(5, 3), "F#5")]


def test_kern_ties_preserve_sustains_across_metadata_and_barlines(tmp_path):
    score, part = stream.Score(), stream.Part()
    part.insert(0, meter.TimeSignature("4/4"))
    part.insert(0, note.Note("C#4", quarterLength=6))
    part.insert(1, clef.Treble8vaClef())
    part.insert(2, key.KeySignature(-2))
    score.insert(0, part)
    codec = ScoreCodec()
    path = tmp_path / "sustain.krn"
    codec.write(codec.encode(score), path, strict=True)
    independent = converter.parse(str(path))
    notes = list(independent.recurse().notes)
    assert [n.tie.type for n in notes] == ["start", "continue", "continue", "stop"]
    assert sum(n.quarterLength for n in notes) == 6
    assert {n.nameWithOctave for n in notes} == {"C#4"}
    assert "*clefG^2" in path.read_text() and "*k[b-e-]" in path.read_text()


@pytest.mark.parametrize("text", [MONO, POLY, OPEN, DYNAMIC])
def test_each_dataset_texture_can_train_and_validate(text, tmp_path, capsys):
    torch.set_num_threads(2)
    waveform = 0.1 * np.sin(2 * np.pi * 261.626 * np.arange(16000) / 8000)
    sf.write(tmp_path / "audio.wav", waveform, 8000)
    manifest = tmp_path / "data.jsonl"
    manifest.write_text(json.dumps({"audio": "audio.wav", "transcript": text}) + "\n")
    checkpoint = tmp_path / "model.pt"
    train_main([str(checkpoint), "--datasetMetaFile_train", str(manifest),
                "--datasetMetaFile_val", str(manifest), "--modelConf", "configs/a2s-smoke.conf",
                "--epochs", "1", "--batchSize", "1"])
    _, saved = load_checkpoint(checkpoint)
    assert np.isfinite(saved["metrics"]["ctc_loss"]) and saved["step"] == 1


def test_mixed_dataset_batch_and_unseen_validation_texture(tmp_path):
    torch.set_num_threads(2)
    rows = []
    for index, text in enumerate([MONO, POLY, OPEN, DYNAMIC]):
        sf.write(tmp_path / f"audio{index}.wav", np.zeros(16000 + index * 100), 8000)
        (tmp_path / f"score{index}.krn").write_text(text)
        rows.append({"audio": f"audio{index}.wav", "score": f"score{index}.krn"})
    manifest = tmp_path / "mixed.json"
    manifest.write_text(json.dumps(rows))
    conf = read_config("configs/a2s-smoke.conf")
    ds = ScoreDataset(manifest, conf)
    batch = collate_score([ds[i] for i in range(len(ds))])
    model = TransKun(conf, ds.vocabulary)
    loss = model.compute_loss(batch["audioSlices"], batch["audioLengths"], batch["targets"], batch["targetLengths"])
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(model.backbone.inputConv.weight.grad).all()
    mono_manifest = tmp_path / "mono.json"
    mono_manifest.write_text(json.dumps(rows[:1]))
    mono = ScoreDataset(mono_manifest, conf)
    assert len(ScoreDataset(manifest, conf, mono.vocabulary)) == 4


@pytest.mark.parametrize("duration", [Fraction(1), Fraction(3, 2), Fraction(2, 3), Fraction(8), Fraction(10), Fraction(5, 3)])
def test_reciprocal_exact_rhythm(duration):
    text = "**kern\n" + reciprocal(duration) + "c\n*-\n"
    assert events(loads_kern(text))[0][1] == duration


def test_output_is_kern_and_malformed_spines_fail(tmp_path):
    codec = ScoreCodec()
    with pytest.raises(ValueError, match=".krn"):
        codec.write(codec.encode_kern(MONO), tmp_path / "score.musicxml")
    with pytest.raises(ValueError, match="spine fields"):
        loads_kern("**kern\n*^\n4c\n*-\t*-\n")
    with pytest.raises(ValueError, match="unterminated"):
        loads_kern("**kern\n4c\n")


def test_dense_polyphony_resolves_ctc_capacity_before_training(tmp_path):
    torch.set_num_threads(2)
    text = "**kern\n" + "16c 16e 16g\n" * 16 + "*-\n"
    sf.write(tmp_path / "audio.wav", np.zeros(1600), 8000)
    manifest = tmp_path / "dense.jsonl"
    manifest.write_text(json.dumps({"audio": "audio.wav", "transcript": text}) + "\n")
    checkpoint = tmp_path / "dense.pt"
    common = [str(checkpoint), "--datasetMetaFile_train", str(manifest),
              "--datasetMetaFile_val", str(manifest), "--modelConf", "configs/a2s-smoke.conf",
              "--epochs", "1", "--batchSize", "1"]
    with pytest.raises(ValueError, match="CTC needs"):
        train_main(common + ["--noAutoFrameMultiplier"])
    train_main(common)
    model, saved = load_checkpoint(checkpoint)
    assert model.conf.scoreFrameMultiplier > 2
    assert np.isfinite(saved["metrics"]["ctc_loss"])
    # Resume uses the persisted multiplier; the raw template was only a lower bound.
    train_main(common + ["--resume", "--epochs", "2"])

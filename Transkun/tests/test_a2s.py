import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from music21 import chord, clef, converter, key, meter, note, stream, tie

from transkun.CheckpointA2S import load_checkpoint, read_config, warm_start_amt
from transkun.Data import Note
from transkun.DataScore import ScoreDataset, collate_score, read_audio
from transkun.ModelA2S import TransKun
from transkun.ModelTransformer import TransKun as AMTTransKun
from transkun.Score import LEGACY_CODEC_VERSION, ScoreCodec, Vocabulary
from transkun.Kern import loads_kern, reciprocal
from transkun.Util import makeFrame
from transkun.trainA2S import main as train_main
from transkun.transcribeA2S import main as transcribe_main

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def deterministic_small_cpu():
    torch.set_num_threads(2)
    torch.manual_seed(123)


def simple_score():
    score = stream.Score()
    part = stream.Part()
    part.insert(0, meter.TimeSignature("4/4"))
    part.insert(0, clef.TrebleClef())
    part.insert(0, key.KeySignature(0))
    part.insert(0, note.Note("C4", quarterLength=4))
    score.insert(0, part)
    return score


@pytest.fixture
def conf():
    return read_config(ROOT / "configs/a2s-smoke.conf")


@pytest.fixture
def tokens():
    return ScoreCodec().encode(simple_score())


@pytest.fixture
def dataset_files(tmp_path):
    simple_score().write("musicxml", fp=str(tmp_path / "score.musicxml"))
    rows = []
    for index, length in enumerate([3200, 4800]):
        waveform = 0.1 * np.sin(2 * np.pi * 261.626 * np.arange(length) / 8000)
        sf.write(tmp_path / f"audio{index}.wav", waveform, 8000)
        rows.append({"audio": f"audio{index}.wav", "score": "score.musicxml"})
    manifest = tmp_path / "train.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return manifest


def test_score_roundtrip_polyphony_rests_spelling_ties_and_meter(tmp_path):
    score = stream.Score()
    for index in range(2):
        part = stream.Part(id=f"P{index}")
        part.insert(0, meter.TimeSignature("4/4"))
        part.insert(0, clef.TrebleClef() if index == 0 else clef.BassClef())
        part.insert(0, key.KeySignature(-2))
        voice = stream.Voice(id="upper")
        voice.insert(0, chord.Chord(["B-3", "D4"], quarterLength=1))
        voice.insert(1, note.Rest(quarterLength=1))
        start = note.Note("F#4", quarterLength=1)
        start.tie = tie.Tie("start")
        stop = note.Note("F#4", quarterLength=1)
        stop.tie = tie.Tie("stop")
        voice.insert(2, start)
        voice.insert(3, stop)
        part.insert(0, voice)
        lower = stream.Voice(id="lower")
        lower.insert(0, note.Note("C3", quarterLength=4))
        part.insert(0, lower)
        part.insert(4, meter.TimeSignature("3/4"))
        part.insert(4, note.Rest(quarterLength=3))
        score.insert(0, part)
    codec = ScoreCodec()
    encoded = codec.encode(score)
    decoded, diagnostics = codec.decode(encoded, strict=True)
    assert diagnostics["events"] == 12
    assert diagnostics["rejected_records"] == 0
    assert len(decoded.parts) == 2
    assert "PITCH_B-3" in encoded and "TIE_start" in encoded
    assert len(decoded.parts[0].getElementsByClass(stream.Voice)) == 3
    for suffix in ["krn", "kern"]:
        path = tmp_path / f"roundtrip.{suffix}"
        codec.write(encoded, path, strict=True)
        parsed = loads_kern(path.read_text())
        assert len(parsed.parts) == 2
        assert any(n.pitch.nameWithOctave == "B-3" for n in parsed.recurse().notes if isinstance(n, note.Note)) or any(
            "B-3" in [p.nameWithOctave for p in c.pitches] for c in parsed.recurse().getElementsByClass(chord.Chord))
        assert any(n.tie and n.tie.type == "start" for n in parsed.recurse().notes)
        assert len(parsed.parts[0].recurse().getElementsByClass(meter.TimeSignature)) == 2
        independent = converter.parse(str(path), format="humdrum")
        assert independent.recurse().notes


def test_tuplet_position_and_recovery(tmp_path):
    codec = ScoreCodec()
    tokens = ["AT_1/3", "PART_0", "VOICE_0", "DUR_2/3", "PITCH_C4", "TIE_none", "END"]
    score, _ = codec.decode(tokens, strict=True)
    n = score.parts[0].flatten().notes[0]
    assert float(n.offset) == pytest.approx(1 / 3)
    assert float(n.quarterLength) == pytest.approx(2 / 3)
    diagnostics = codec.write(["junk", "END"] + tokens, tmp_path / "recovered.krn")
    assert diagnostics["rejected_records"] == 1 and diagnostics["events"] == 1
    assert converter.parse(str(tmp_path / "recovered.krn")).recurse().notes
    with pytest.raises(ValueError, match="Invalid score record"):
        codec.decode(["junk", "END"], strict=True)
    diagnostics = codec.write([], tmp_path / "empty.krn")
    assert diagnostics["fallback"] and converter.parse(str(tmp_path / "empty.krn")).recurse().notesAndRests


def test_frontend_backbone_preserved_and_backward(conf, tokens):
    vocab = Vocabulary.build([tokens])
    model = TransKun(conf, vocab)
    original = AMTTransKun(conf)
    original.load_state_dict({key: value for key, value in model.state_dict().items()
                              if not key.startswith(("scoreProjection.", "scoreOutput."))})
    waveform = torch.randn(1, 3200, 1) * 0.1
    frames = makeFrame(waveform.transpose(-1, -2), conf.hopSize, conf.windowSize)
    model.eval()
    original.eval()
    with torch.no_grad():
        assert torch.equal(model.encodeFramesBatch(frames), original.encodeFramesBatch(frames))
        crf, context = original.processFramesBatch(frames)
    assert context.shape[:3] == (1, 90, 51)
    assert len(crf.decode()) == 90
    model.train()
    targets = torch.tensor([vocab.encode(tokens)])
    loss = model.compute_loss(waveform, torch.tensor([3200]), targets, torch.tensor([len(tokens)]))
    loss.backward()
    assert torch.isfinite(loss)
    for parameter in [model.backbone.inputConv.weight,
                      model.framewiseFeatureExtractor.spectrogramExtractor.winGen.sigma,
                      model.scoreOutput.weight]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert model.velocityPredictor[0].weight.grad is None


def test_unequal_audio_padding_never_enters_backbone(conf, tokens):
    model = TransKun(conf, Vocabulary.build([tokens])).eval()
    short = torch.randn(2000, 1)
    long = torch.randn(3000, 1)
    padded = torch.zeros(2, 3000, 1)
    padded[0, :2000] = short
    padded[0, 2000:] = 1000  # Should have no influence.
    padded[1] = long
    with torch.no_grad():
        batch, lengths = model(padded, torch.tensor([2000, 3000]))
        single, single_lengths = model(short.unsqueeze(0))
    assert lengths.tolist() == [66, 96]
    torch.testing.assert_close(batch[0, :lengths[0]], single[0])
    assert single_lengths[0] == lengths[0]


def test_ctc_collapse_and_length_errors(conf):
    model = TransKun(conf, Vocabulary(["<BLANK>", "A", "B"]))
    logits = torch.full((1, 7, 3), -10.0)
    for index, token in enumerate([1, 1, 0, 1, 2, 2, 0]):
        logits[0, index, token] = 10
    assert model.decode_logits(logits, torch.tensor([7])) == [["A", "A", "B"]]
    with pytest.raises(ValueError, match="needs 3 frames"):
        model.loss_from_logits(logits[:, :2], torch.tensor([2]), torch.tensor([[1, 1]]), torch.tensor([2]))
    with pytest.raises(ValueError, match="blank"):
        model.loss_from_logits(logits, torch.tensor([7]), torch.tensor([[0]]), torch.tensor([1]))


def test_dataset_lengths_resampling_and_unseen_tokens(conf, dataset_files, tmp_path):
    dataset = ScoreDataset(dataset_files, conf)
    batch = collate_score([dataset[0], dataset[1]])
    assert batch["audioSlices"].shape == (2, 4800, 1)
    assert batch["audioLengths"].tolist() == [3200, 4800]
    sf.write(tmp_path / "resample.wav", np.zeros((2205, 2)), 22050)
    assert read_audio(tmp_path / "resample.wav", 8000).shape == (800, 1)
    changed = simple_score()
    changed.parts[0].notes[0].pitch.name = "D"
    changed.write("musicxml", fp=str(tmp_path / "unseen.musicxml"))
    val = tmp_path / "val.json"
    val.write_text(json.dumps([{"audio": "audio0.wav", "score": "unseen.musicxml"}]))
    validation = ScoreDataset(val, conf, dataset.vocabulary)
    assert len(validation) == 1  # v2 covers pitches/numbers absent from training.
    import copy
    legacy_conf = copy.copy(conf)
    legacy_conf.scoreCodecVersion = LEGACY_CODEC_VERSION
    legacy_dataset = ScoreDataset(dataset_files, legacy_conf)
    with pytest.raises(ValueError, match="absent from training vocabulary"):
        ScoreDataset(val, legacy_conf, legacy_dataset.vocabulary)
    with pytest.raises(ValueError, match="exceeds"):
        read_audio(tmp_path / "audio0.wav", 8000, max_seconds=0.1)


def test_training_validation_resume_and_audio_to_kern(conf, dataset_files, tmp_path):
    checkpoint = tmp_path / "a2s.pt"
    common = [str(checkpoint), "--datasetMetaFile_train", str(dataset_files),
              "--datasetMetaFile_val", str(dataset_files), "--modelConf", str(ROOT / "configs/a2s-smoke.conf"),
              "--batchSize", "2"]
    train_main(common + ["--epochs", "1"])
    model, saved = load_checkpoint(checkpoint)
    assert saved["epoch"] == 1 and saved["step"] == 1
    assert np.isfinite(saved["metrics"]["ctc_loss"])
    assert checkpoint.with_name("a2s.best.pt").exists()
    train_main(common + ["--epochs", "2", "--resume"])
    _, resumed = load_checkpoint(checkpoint)
    assert resumed["epoch"] == 2 and resumed["step"] == 2
    uninterrupted = tmp_path / "uninterrupted.pt"
    train_main([str(uninterrupted)] + common[1:] + ["--epochs", "2"])
    _, continuous = load_checkpoint(uninterrupted)
    for name, tensor in resumed["state_dict"].items():
        assert torch.equal(tensor, continuous["state_dict"][name])
    train_main([str(checkpoint), "--datasetMetaFile_val", str(dataset_files), "--validateOnly"])
    output = tmp_path / "predicted.krn"
    details = tmp_path / "predicted.tokens.json"
    transcribe_main([str(tmp_path / "audio0.wav"), str(output), "--weight", str(checkpoint),
                     "--tokensOut", str(details)])
    parsed = converter.parse(str(output))
    assert len(parsed.parts) >= 1 and parsed.recurse().notesAndRests
    assert json.loads(details.read_text())["codec_version"] == ScoreCodec.version
    model.eval()
    loaded, _ = load_checkpoint(checkpoint)
    loaded.eval()
    audio = read_audio(tmp_path / "audio0.wav", conf.fs).unsqueeze(0)
    assert model.vocabulary.tokens == loaded.vocabulary.tokens
    # Models from epoch 1 and 2 intentionally differ; the checkpoint round trip
    # compares the current model with its stored state, without tolerant loading.
    for name, tensor in loaded.state_dict().items():
        torch.testing.assert_close(tensor.cpu(), resumed["state_dict"][name])
    assert loaded.transcribe_score(audio)


def test_kern_input_and_invalid_notation(tmp_path):
    path = tmp_path / "example.krn"
    path.write_text("**kern\n*clefG2\n*k[]\n*M4/4\n1c\n*-\n")
    codec = ScoreCodec()
    tokens = codec.encode_file(path)
    assert "PITCH_C4" in tokens
    codec.write(tokens, tmp_path / "kern.krn", strict=True)
    assert converter.parse(str(tmp_path / "kern.krn")).recurse().notes
    event = ["AT_0", "PART_0", "VOICE_0", "DUR_1", "PITCH_C4", "TIE_none", "END"]
    with pytest.raises(ValueError, match="Invalid score record"):
        codec.decode(event + event, strict=True)
    _, diagnostics = codec.decode(event + event)
    assert diagnostics["events"] == 1 and diagnostics["rejected_records"] == 1
    score = simple_score()
    score.parts[0].notes[0].duration = note.Note("D4").getGrace().duration
    with pytest.raises(ValueError, match="Grace"):
        codec.encode(score)


def test_shipped_amt_checkpoint_warm_start(tokens):
    model = TransKun(read_config(ROOT / "configs/a2s.conf"), Vocabulary.build([tokens]))
    before = model.scoreOutput.weight.detach().clone()
    assert warm_start_amt(model, ROOT / "transkun/pretrained/2.0.pt") == 184
    assert torch.equal(before, model.scoreOutput.weight)
    incompatible = TransKun(read_config(ROOT / "configs/a2s-smoke.conf"), Vocabulary.build([tokens]))
    with pytest.raises(ValueError, match="match"):
        warm_start_amt(incompatible, ROOT / "transkun/pretrained/2.0.pt")


def test_original_amt_loss_and_inference_still_work(conf):
    model = AMTTransKun(conf)
    audio = torch.randn(1, 2048, 1) * 0.1
    notes = [[Note(0.05, 0.15, 60, 80)]]
    loss = -model.log_prob(audio, notes).sum()
    loss.backward()
    assert torch.isfinite(loss)
    model.eval()
    with torch.no_grad():
        frames = makeFrame(audio.transpose(-1, -2), conf.hopSize, conf.windowSize)
        predictions, _ = model.transcribeFrames(frames)
    assert len(predictions) == 1


def test_original_semi_crf_example_bounded():
    from transkun.CRF import NeuralSemiCRFInterval
    score = torch.randn(20, 20, 4, requires_grad=True)
    noise = torch.randn(19, 4, requires_grad=True)
    intervals = [[(0, 2), (4, 6), (6, 6), (7, 8)],
                 [(1, 2), (3, 5), (19, 19)], [(0, 0), (4, 7)], []]
    crf = NeuralSemiCRFInterval(score, noise)
    loss = -crf.logProb(intervals).sum()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(score.grad).all()
    assert len(crf.decode()) == 4

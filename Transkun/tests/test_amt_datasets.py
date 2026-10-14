import json
from pathlib import Path

import numpy as np
import pretty_midi
import pytest
import soundfile as sf
import torch

from transkun.DataAMT import AMTChunks, collate_amt, find_partitions, inspect_dataset, read_chunk
from transkun.ModelTransformer import ModelConfig
from transkun.trainAMT import GradientClipper, main as train_main, model_config, parse_arguments, read_config


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "data" / "example_dataset"
    partitions = root / "amt_partitions" / "example_amt"
    partitions.mkdir(parents=True)
    (root / "wav").mkdir()
    (root / "midi").mkdir()
    for index in range(4):
        waveform = 0.1 * np.sin(2 * np.pi * 261.626 * np.arange(8000) / 8000)
        sf.write(root / "wav" / f"{index}.wav", waveform, 8000)
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=65)
        instrument.notes = [pretty_midi.Note(80, 60, 0.1, 0.65)]
        midi.instruments.append(instrument)
        midi.write(str(root / "midi" / f"{index}.mid"))
    (partitions / "train.txt").write_text("wav/0.wav\tmidi/0.mid\nwav/1.wav\tmidi/1.mid\n")
    (partitions / "val.txt").write_text("wav/2.wav\tmidi/2.mid\n")
    (partitions / "test.txt").write_text("wav/3.wav\tmidi/3.mid\n")
    return root


@pytest.fixture
def tiny_conf(tmp_path):
    conf = ModelConfig()
    conf.fs, conf.windowSize, conf.hopSize = 8000, 256, 64
    conf.f_max, conf.n_mels, conf.nExtraWins = 3900, 16, 1
    conf.baseSize, conf.nHead, conf.nLayers = 4, 2, 1
    conf.scoringExpansionFactor, conf.hiddenFactor = 1, 2
    conf.velocityPredictorHiddenSize = conf.refinedOFPredictorHiddenSize = 8
    conf.contextDropoutProb = 0
    path = tmp_path / "amt.conf"
    path.write_text(json.dumps(model_config(conf)))
    return path


def test_partitions_clips_resampling_and_note_boundary_flags(dataset):
    samples, audit = inspect_dataset(dataset)
    assert find_partitions(dataset).name == "example_amt"
    assert audit["splits"]["train"]["files"] == 2
    chunks = AMTChunks(samples["train"], 8000, 0.5)
    first, second = chunks[0], chunks[1]
    assert first["audio"].shape == (4000, 1)
    assert first["notes"][0].hasOnset and not first["notes"][0].hasOffset
    assert not second["notes"][0].hasOnset and second["notes"][0].hasOffset
    assert second["notes"][0].start == 0
    assert second["notes"][0].end == pytest.approx(0.15)
    batch = collate_amt([first, second])
    assert batch["audioSlices"].shape == (2, 4000, 1)
    stereo = dataset / "wav" / "stereo.wav"
    sf.write(stereo, np.zeros((2205, 2)), 22050, subtype="FLOAT")
    assert read_chunk(stereo, 0, 0.2, 8000).shape == (1600, 1)
    assert torch.isfinite(read_chunk(stereo, 0, 0.2, 8000)).all()


def test_annotation_policy_is_explicit_and_empty_midi_works(dataset):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(65)
    instrument.notes = [pretty_midi.Note(80, 60, 0.5, 1.5), pretty_midi.Note(80, 62, 1.2, 1.6)]
    midi.instruments.append(instrument)
    midi.write(str(dataset / "midi/0.mid"))
    samples, audit = inspect_dataset(dataset, boundary_policy="clip")
    assert audit["splits"]["train"]["events_clipped"] == 1
    assert audit["splits"]["train"]["events_dropped_outside_audio"] == 1
    assert samples["train"][0]["notes"][0].end == 1.0
    assert not samples["train"][0]["notes"][0].hasOffset
    with pytest.raises(ValueError, match="outside audio"):
        inspect_dataset(dataset, boundary_policy="error")
    pretty_midi.PrettyMIDI().write(str(dataset / "midi/0.mid"))
    _, audit = inspect_dataset(dataset)
    assert audit["splits"]["train"]["files_without_notes_inside_audio"] == 1


def test_split_overlap_and_wrong_model_are_rejected(dataset, tiny_conf):
    val = find_partitions(dataset) / "val.txt"
    val.write_text("wav/0.wav\tmidi/0.mid\n")
    with pytest.raises(ValueError, match="both train and val"):
        inspect_dataset(dataset)
    document = json.loads(tiny_conf.read_text())
    document["Model"]["module"] = "transkun.ModelA2S"
    tiny_conf.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="not ModelA2S"):
        read_config(tiny_conf)


def test_training_resume_logs_checkpoints_and_midi_inference(dataset, tiny_conf, tmp_path):
    run = tmp_path / "runs" / "resumed"
    common = ["--datasetRoot", str(dataset), "--modelConf", str(tiny_conf), "--epochs", "2",
              "--chunkSize", "0.5", "--hopSize", "0.5", "--batchSize", "1", "--accumulationSteps", "2",
              "--dataLoaderWorkers", "0", "--device", "cpu", "--chunkMetrics", "--saveEvery", "1"]
    train_main(common + ["--runDir", str(run), "--maxSteps", "1"])
    checkpoint = torch.load(run / "last.pt", weights_only=True)
    assert checkpoint["task"] == "AMT" and checkpoint["global_step"] == 1
    assert checkpoint["next_batch"] == 2
    train_main(common + ["--runDir", str(run), "--resume"])
    resumed = torch.load(run / "last.pt", weights_only=True)
    assert resumed["epoch"] == 2 and resumed["global_step"] == 4
    continuous = tmp_path / "runs" / "continuous"
    train_main(common + ["--runDir", str(continuous)])
    final = torch.load(continuous / "last.pt", weights_only=True)
    for name, tensor in resumed["state_dict"].items():
        assert torch.equal(tensor, final["state_dict"][name]), name
    assert resumed["gradient_clipping_state_dict"] == final["gradient_clipping_state_dict"]
    assert (run / "best.pt").is_file()
    assert (run / "train.log").stat().st_size > 0
    assert list((run / "tensorboard").glob("events.out.tfevents.*"))
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert any(item["kind"] == "validation" for item in metrics)
    assert json.loads((run / "status.json").read_text())["phase"] == "completed"
    train_main(common + ["--runDir", str(run), "--validateOnly", "--split", "test"])
    assert json.loads((run / "test_metrics.json").read_text())["metric_scope"] == "chunks"
    from transkun.ModelTransformer import TransKun
    from transkun.Data import writeMidi
    from transkun.Util import makeFrame
    conf = read_config(run / "conf.json")
    model = TransKun(conf)
    model.load_state_dict(resumed["state_dict"], strict=True)
    model.eval()
    waveform = read_chunk(dataset / "wav/3.wav", 0, 0.5, conf.fs).unsqueeze(0)
    with torch.no_grad():
        notes, _ = model.transcribeFrames(makeFrame(waveform.transpose(-1, -2), conf.hopSize, conf.windowSize))
    output = tmp_path / "prediction.mid"
    writeMidi(notes[0]).write(str(output))
    assert pretty_midi.PrettyMIDI(str(output)).instruments or not notes[0]


def test_full_file_validation_runs(dataset, tiny_conf, tmp_path):
    run = tmp_path / "full-file-validation"
    train_main(["--datasetRoot", str(dataset), "--runDir", str(run), "--modelConf", str(tiny_conf),
                "--chunkSize", "0.5", "--hopSize", "0.5", "--batchSize", "1", "--accumulationSteps", "2",
                "--epochs", "1", "--dataLoaderWorkers", "0", "--device", "cpu"])
    records = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    metrics = next(record for record in records if record["kind"] == "validation")
    assert metrics["metric_scope"] == "full_files" and np.isfinite(metrics["note_offset_f1"])


def test_original_inference_reads_wav_and_normalizes_sample_width(dataset):
    from transkun.transcribe import readAudio
    for subtype in ["PCM_16", "PCM_24"]:
        path = dataset / "wav" / f"width_{subtype}.wav"
        values = np.full((800, 1), 0.25, dtype=np.float32)
        sf.write(path, values, 8000, subtype=subtype)
        rate, audio = readAudio(path)
        assert rate == 8000 and audio.shape == (800, 1)
        np.testing.assert_allclose(audio, values, atol=1e-5)


def test_adaptive_gradient_clipping_uses_past_norms_and_restores_history():
    model = torch.nn.Linear(1, 1, bias=False)
    clipper = GradientClipper(quantile=0.8, history_size=3, initial_norm=40)
    for gradient, expected_limit in [(100, 40), (10, 88), (200, 76)]:
        model.weight.grad = torch.full_like(model.weight, gradient)
        norm, limit = clipper.step(model)
        assert float(norm) == pytest.approx(gradient)
        assert limit == pytest.approx(expected_limit)
        assert float(model.weight.grad.norm()) <= limit + 1e-5
    assert clipper.state_dict()["norms"] == [100, 10, 200]
    resumed = GradientClipper(quantile=0.8, history_size=3, initial_norm=40)
    resumed.load_state_dict(clipper.state_dict())
    for controller in (clipper, resumed):
        model.weight.grad = torch.full_like(model.weight, 300)
        _, limit = controller.step(model)
        assert limit == pytest.approx(160)
    assert clipper.state_dict() == resumed.state_dict()


def test_training_preset_cli_budget_override_and_validation(dataset, tmp_path):
    preset = tmp_path / "training.conf"
    preset.write_text(json.dumps({"Training": {"epochs": 7, "batchSize": 3, "chunkSize": 2}}))
    args = parse_arguments(["--datasetRoot", str(dataset), "--trainingConf", str(preset), "--nIter", "10",
                            "--batchSize", "2"])
    assert args.nIter == 10 and args.epochs is None and args.batchSize == 2 and args.hopSize == 1
    args = parse_arguments(["--datasetRoot", str(dataset), "--epochs", "2"])
    assert args.epochs == 2 and args.nIter is None
    with pytest.raises(SystemExit):
        parse_arguments(["--datasetRoot", str(dataset), "--epochs", "2", "--nIter", "10"])
    preset.write_text(json.dumps({"Training": {"nIter": 10, "unexpected_field": 1}}))
    with pytest.raises(SystemExit):
        parse_arguments(["--datasetRoot", str(dataset), "--trainingConf", str(preset)])


def test_iteration_budget_validates_and_completes_mid_epoch_with_exact_resume(dataset, tiny_conf, tmp_path):
    common = ["--datasetRoot", str(dataset), "--modelConf", str(tiny_conf), "--nIter", "3",
              "--chunkSize", "0.5", "--hopSize", "0.5", "--batchSize", "1", "--accumulationSteps", "1",
              "--dataLoaderWorkers", "0", "--device", "cpu", "--chunkMetrics", "--saveEvery", "1"]
    resumed_run = tmp_path / "iteration-resumed"
    train_main(common + ["--runDir", str(resumed_run), "--maxSteps", "1"])
    train_main(common + ["--runDir", str(resumed_run), "--resume"])
    continuous_run = tmp_path / "iteration-continuous"
    train_main(common + ["--runDir", str(continuous_run)])
    resumed = torch.load(resumed_run / "last.pt", weights_only=True)
    final = torch.load(continuous_run / "last.pt", weights_only=True)
    assert final["format_version"] == 2
    assert final["global_step"] == 3 and final["epoch"] == 0 and final["next_batch"] == 3
    assert final["lr_scheduler_state_dict"]["total_steps"] == 3
    assert final["lr_scheduler_state_dict"]["last_epoch"] == 3
    assert resumed["gradient_clipping_state_dict"] == final["gradient_clipping_state_dict"]
    for name, tensor in final["state_dict"].items():
        assert torch.equal(tensor, resumed["state_dict"][name]), name
    status = json.loads((continuous_run / "status.json").read_text())
    assert status["phase"] == "completed"
    records = [json.loads(line) for line in (continuous_run / "metrics.jsonl").read_text().splitlines()]
    assert sum(e["kind"] == "train" for e in records) == 3
    assert sum(e["kind"] == "validation" for e in records) == 1
    best = torch.load(continuous_run / "best.pt", weights_only=True)
    assert best["epoch"] == 0 and best["next_batch"] == 3

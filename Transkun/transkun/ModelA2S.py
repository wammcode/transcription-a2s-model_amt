"""Transkun's unchanged audio backbone with a score-token CTC output head."""
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from .ModelTransformer import ModelConfig as AMTConfig, TransKun as AMTTransKun
from .Score import CODEC_VERSION, SUPPORTED_CODECS, Vocabulary
from .Util import makeFrame


class ModelConfig(AMTConfig):
    def __init__(self):
        super().__init__()
        self.scoreHeadSize = 256
        self.scoreDropout = 0.1
        self.scoreFrameMultiplier = 2
        self.scoreMaxAudioSeconds = 30
        self.scoreCodecVersion = CODEC_VERSION

    def validate(self):
        if self.scoreCodecVersion not in SUPPORTED_CODECS:
            raise ValueError("Unsupported scoreCodecVersion")
        if not isinstance(self.scoreFrameMultiplier, int) or self.scoreFrameMultiplier < 1:
            raise ValueError("scoreFrameMultiplier must be a positive integer")
        if self.scoreHeadSize < 1 or self.scoreMaxAudioSeconds <= 0:
            raise ValueError("Invalid score head size/audio limit")
        if self.fs <= 0 or not 0 < self.hopSize < self.windowSize:
            raise ValueError("Invalid sample rate/hop/window")
        if not 0 <= self.f_min < self.f_max <= self.fs / 2:
            raise ValueError("Mel frequencies must lie within the Nyquist range")


Config = ModelConfig


class TransKun(AMTTransKun):
    Config = ModelConfig

    def __init__(self, conf, vocabulary):
        conf.validate()
        super().__init__(conf)
        self.vocabulary = vocabulary if isinstance(vocabulary, Vocabulary) else Vocabulary(vocabulary)
        # These original heads remain available for AMT and checkpoint compatibility.
        for name in ("scorer", "scorerProj", "velocityPredictor", "refinedOFPredictor"):
            if hasattr(self, name):
                getattr(self, name).requires_grad_(False)
        context_size = len(self.targetMIDIPitch) * conf.baseSize * conf.scoringExpansionFactor
        self.scoreProjection = nn.Sequential(
            nn.LayerNorm(context_size), nn.Linear(context_size, conf.scoreHeadSize),
            nn.GELU(), nn.Dropout(conf.scoreDropout))
        self.scoreOutput = nn.Linear(conf.scoreHeadSize, len(self.vocabulary))
        self.ctc = nn.CTCLoss(blank=0, zero_infinity=False)

    def forward(self, audioSlices, audioLengths=None):
        """Return logits [B,T_score,V] and exact unpadded CTC lengths [B].

        The original backbone has no padding attention mask. Group samples by
        exact length and crop before its frontend, rather than modifying attention
        or letting padded audio influence real frames/normalization.
        """
        if audioSlices.ndim != 3 or audioSlices.shape[-1] != 1:
            raise ValueError("Expected mono audio [batch, samples, 1]")
        if audioLengths is None:
            audioLengths = torch.full((len(audioSlices),), audioSlices.shape[1], dtype=torch.long)
        lengths = audioLengths.detach().cpu().tolist()
        if len(lengths) != len(audioSlices) or any(n <= 0 or n > audioSlices.shape[1] for n in lengths):
            raise ValueError("Invalid audio lengths")
        if any(n / self.fs > self.conf.scoreMaxAudioSeconds + 1e-6 for n in lengths):
            raise ValueError("Audio exceeds scoreMaxAudioSeconds; prepare paired clips")
        groups = defaultdict(list)
        for index, length in enumerate(lengths):
            groups[length].append(index)
        outputs = [None] * len(lengths)
        for length, indices in groups.items():
            waveforms = audioSlices[indices, :length].transpose(-1, -2)
            frames = makeFrame(waveforms, self.hopSize, self.windowSize)
            context = self.encodeFramesBatch(frames)  # [B,90,T,D], unchanged Transkun
            sequence = context.permute(0, 2, 1, 3).flatten(-2)
            hidden = self.scoreProjection(sequence)
            if self.conf.scoreFrameMultiplier != 1:
                hidden = F.interpolate(hidden.transpose(1, 2),
                                       size=hidden.shape[1] * self.conf.scoreFrameMultiplier,
                                       mode="linear", align_corners=False).transpose(1, 2)
            logits = self.scoreOutput(hidden)
            for row, index in enumerate(indices):
                outputs[index] = logits[row]
        output_lengths = torch.tensor([len(output) for output in outputs], dtype=torch.long)
        return pad_sequence(outputs, batch_first=True), output_lengths

    def loss_from_logits(self, logits, output_lengths, targets, target_lengths):
        if targets.ndim != 2 or len(targets) != len(logits) or len(target_lengths) != len(logits):
            raise ValueError("Invalid target batch")
        for index, size in enumerate(target_lengths.detach().cpu().tolist()):
            if not 0 < size <= targets.shape[1]:
                raise ValueError("Invalid target length")
            target = targets[index, :size]
            if torch.any(target <= 0) or torch.any(target >= len(self.vocabulary)):
                raise ValueError("CTC target contains blank or invalid token")
            required = size + int((target[1:] == target[:-1]).sum())
            if required > int(output_lengths[index]):
                raise ValueError(f"CTC sample {index} needs {required} frames; "
                                 f"has {int(output_lengths[index])}. Increase scoreFrameMultiplier.")
        log_probs = logits.log_softmax(-1).transpose(0, 1).contiguous()
        loss = self.ctc(log_probs, targets, output_lengths.cpu(), target_lengths.cpu())
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite A2S CTC loss")
        return loss

    def compute_loss(self, audioSlices, audioLengths, targets, targetLengths):
        logits, output_lengths = self(audioSlices, audioLengths)
        return self.loss_from_logits(logits, output_lengths, targets, targetLengths)

    def decode_logits(self, logits, lengths):
        results = []
        for sample, length in zip(logits, lengths):
            # Collapse before removing blank: A,blank,A must decode as A,A.
            ids = sample[:int(length)].argmax(-1).unique_consecutive().tolist()
            results.append(self.vocabulary.decode(ids))
        return results

    @torch.no_grad()
    def transcribe_score(self, audioSlices, audioLengths=None):
        logits, lengths = self(audioSlices, audioLengths)
        return self.decode_logits(logits, lengths)

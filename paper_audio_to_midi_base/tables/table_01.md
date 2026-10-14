# Table 1

| Specification | Value |
|---|---|
| Input Mel Spectrogram | sr: 44100 Hz, hop: 1024, window size: 4096, subwindows:5, mels: 229, freq: 30-8000 Hz, segment: 16s, |
| Patch | shape: $8\times4$, embeding size: 256 |
| Strided Conv. Layers for Downsampling | initial proj. size: 64, added with freq. embeddings.<br>out channels: [128, 256, 256, 256], kernel size: 3, strides: [(2,1), (2,2), (2,2), (1,1)], Each followed by GroupNorm, groups = 4, and GELU (except for the last conv.) |
| Position Embedding | $\gamma=1$, $\lvert B\rvert=256$, MLP hidden size 1024 |
| Transformer Encoder | 8 heads, 6 layers (=12 blocks), FNN size: 1024 |
| Upsampling | 1d. transposed conv, out: 128, kernel size:8, stride:8 |
| Attribute Prediction | two layer MLP, hidden size: 512, dropout 0.1 |
| Batch Size | 12 |
| Optimizer | Adabelief [28], maximum learning rate: 4e−4 |
| Weight Decay | 1e−2, excluding bias, norm., and pos. embedding |
| Learning Rate Schedule | 500k iterations, 5% warm-up phase, cosine anneal. |
| Gradient Clipping | Clipping norms at 80% quantile of past 10,000 iterations |

**Table 1:** Model Specification.

<!-- Conversion annotation: the source has no column headings; Specification/Value are navigation labels added for Markdown. Original spelling, values and caption are preserved. -->

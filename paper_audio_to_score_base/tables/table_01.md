# Table 1

**Results in terms of the SER (%) and MV2H (%) metrics.** The figures shaded in gray represent intra-composer scenarios, in which the training and test sets share the same author. Conversely, the remaining figures pertain to inter-composer scenarios, where the author in the test set differs from the one in the training set. Symbols ↑ and ↓ depict whether the metrics are positively or negatively valued, respectively.

| Train corpus | Model | Haydn ↓ SER | Haydn ↑ MV2H | Mozart ↓ SER | Mozart ↑ MV2H | Beethoven ↓ SER | Beethoven ↑ MV2H | Quartets ↓ SER | Quartets ↑ MV2H |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Haydn | CRNN-CTC (SoTA) | 30.4 | 48.2 | 36.9 | 40.7 | 48.9 | 28.2 | 37.7 | 40.2 |
| Haydn | Transformer (Proposal) | 5.4 | 93.3 | 22.0 | 72.3 | 41.9 | 57.9 | 20.5 | 77.6 |
| Mozart | CRNN-CTC (SoTA) | 45.2 | 31.8 | 26.8 | 49.5 | 55.2 | 24.3 | 44.7 | 32.8 |
| Mozart | Transformer (Proposal) | 36.9 | 64.1 | 7.3 | 92.9 | 53.3 | 52.1 | 36.3 | 65.8 |
| Beethoven | CRNN-CTC (SoTA) | 57.4 | 16.5 | 55.0 | 16.5 | 52.3 | 21.1 | 55.3 | 18.0 |
| Beethoven | Transformer (Proposal) | 41.0 | 59.1 | 42.0 | 57.7 | 15.7 | 86.9 | 33.0 | 67.9 |
| Quartets | CRNN-CTC (SoTA) | 50.8 | 29.0 | 49.3 | 30.6 | 53.5 | 24.7 | 51.4 | 27.9 |
| Quartets | Transformer (Proposal) | 12.9 | 84.2 | 14.8 | 81.8 | 19.1 | 81.4 | 15.3 | 82.8 |


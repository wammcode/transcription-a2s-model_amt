# Table 3

<!-- Conversion annotation: source row-spanning method names are repeated on each row; column spelling Groudtruth is preserved. -->

| Method | Dataset | Groudtruth | Activation P(%) | Activation R(%) | Activation $F_1$(%) | Note Onset P(%) | Note Onset R(%) | Note Onset $F_1$(%) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Onsets &Frames [24] w. Data Aug.[^8] | MAPS | Original | 90.27 | 80.33 | 84.87 | 87.40 | 85.56 | 86.41 |
| Onsets &Frames [24] w. Data Aug.[^8] | MAPS | Ad hoc Align | 90.50 | 80.53 | 85.08 | 88.79 | 86.93 | 87.78 |
| Onsets &Frames [24] w. Data Aug.[^8] | MAPS | Cogliati | 64.75 | 82.83 | 71.60 | 87.57 | 84.97 | 86.19 |
| hFT [5].[^8] | MAPS | Original | 91.53 | 71.03 | 79.81 | 84.63 | 85.75 | 85.13 |
| hFT [5].[^8] | MAPS | Ad hoc Align | 91.77 | 71.25 | 80.04 | 87.32 | 88.48 | 87.84 |
| hFT [5].[^8] | MAPS | Cogliati | 68.83 | 74.07 | 70.24 | 89.94 | 90.10 | 89.97 |
| hFT [5].[^8] | SMD | Original | 93.18 | 89.82 | 91.35 | 98.71 | 95.58 | 97.09 |
| Ours | MAPS | Original | 88.41 | 82.29 | 85.08 | 84.31 | 88.10 | 86.10 |
| Ours | MAPS | Ad hoc Align | 88.69 | 82.57 | 85.36 | 86.63 | 90.53 | 88.47 |
| Ours | MAPS | Cogliati | 65.74 | 84.69 | 72.78 | 89.60 | 91.39 | 90.44 |
| Ours | SMD | Original | 92.36 | 95.24 | 93.73 | 98.16 | 97.65 | 97.89 |
| Ours w. Data Aug. | MAPS | Original | 94.11 | 84.63 | 89.00 | 92.11 | 88.78 | 90.38 |
| Ours w. Data Aug. | MAPS | Ad hoc Align | 94.35 | 84.84 | 89.22 | 94.21 | 90.76 | 92.41 |
| Ours w. Data Aug. | MAPS | Cogliati | 67.77 | 87.39 | 75.03 | 94.66 | 91.43 | 92.98 |
| Ours w. Data Aug. | SMD | Original | 93.38 | 95.91 | 94.57 | 99.77 | 97.68 | 98.70 |

**Between Ground Truths**

| Method | Dataset | Groudtruth | Activation P(%) | Activation R(%) | Activation $F_1$(%) | Note Onset P(%) | Note Onset R(%) | Note Onset $F_1$(%) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Cogliati [26] | MAPS | Original | 98.86 | 69.22 | 80.17 | 100 | 100 | 100 |

**Table 3:** Transcription Result on MAPS and SMD. See Text for discussion of dataset issues.

[^8]: Use their provided code and pretrained weights. Recomputed from transcribed MIDIs.

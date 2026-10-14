# Table 2

<!-- Conversion annotation: the three event categories replace the source's spanning category rows. The four metric groups are flattened into explicit column headings. Bold values reproduce the source emphasis, including apparent inconsistencies. -->

**Notes**

| Method | # Param | Activation P(%) | Activation R(%) | Activation $F_1$(%) | Note Onset P(%) | Note Onset R(%) | Note Onset $F_1$(%) | Note w/ Offset P(%) | Note w/ Offset R(%) | Note w/ Offset $F_1$(%) | Note w/ Offset & Vel. P(%) | Note w/ Offset & Vel. R(%) | Note w/ Offset & Vel. $F_1$(%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SemiCRF [2] | 9.8M | 93.79 | 88.36 | 90.75 | 98.69 | 93.96 | 96.11 | 90.79 | 86.46 | 88.42 | 89.78 | 85.51 | 87.44 |
| hFT, reported in [5] | 5.5M | 92.82 | 93.66 | 93.24 | **99.64** | 95.44 | 97.44 | 92.52 | 88.69 | 90.53 | 91.43 | 87.67 | 89.48 |
| hFT [5][^8]. | 5.5M | 95.37 | 90.82 | 92.93 | **99.62** | 95.41 | 97.43 | 92.22 | 88.40 | 90.23 | 91.21 | 87.44 | 89.24 |
| Ours with scoring method in [2] | 11.0M | 93.79 | 92.40 | 93.06 | 98.61 | 95.92 | 97.23 | 91.69 | 89.23 | 90.43 | 91.08 | 88.64 | 89.83 |
| Ours with MLP $kqb$ mapping | 13.0M | 95.66 | 94.79 | 95.20 | 99.54 | 96.91 | 98.19 | 94.39 | 91.92 | 93.12 | 93.84 | 91.40 | **92.59** |
| Ours w/o incomplete events | 12.9M | **93.76** | 94.46 | 95.07 | 99.56 | 97.10 | 98.30 | **94.66** | 92.36 | 93.48 | **94.12** | 91.83 | **92.95** |
| Ours | 12.9M | 95.75 | **95.01** | **95.35** | 99.53 | **97.16** | **98.32** | 94.61 | **92.39** | **93.48** | 94.07 | **91.87** | 92.94 |

**Sustain Pedals**

| Method | # Param | Activation P(%) | Activation R(%) | Activation $F_1$(%) | Note Onset P(%) | Note Onset R(%) | Note Onset $F_1$(%) | Note w/ Offset P(%) | Note w/ Offset R(%) | Note w/ Offset $F_1$(%) | Note w/ Offset & Vel. P(%) | Note w/ Offset & Vel. R(%) | Note w/ Offset & Vel. $F_1$(%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Kong et al., reported in [4] | 20.2M | 94.30 | 94.42 | 94.25 | 91.59 | 92.41 | 91.86 | 86.36 | 87.02 | 86.58 | - | - | - |
| Kong et al. [4][^8][^9] | 20.2M | 94.14 | 94.29 | 94.11 | 77.43 | 78.19 | 77.71 | 73.56 | 74.21 | 73.81 | - | - | - |
| SemiCRF [2] | 9.8M | 95.17 | 88.33 | 90.98 | 82.18 | 75.81 | 78.52 | 78.75 | 72.74 | 75.30 | - | - | - |
| Ours w/o incomplete events | 12.9M | **96.69** | 92.92 | 94.47 | **89.10** | 83.96 | 86.28 | **86.33** | 81.40 | 83.63 | - | - | - |
| Ours | 12.9M | 96.67 | **94.46** | **95.40** | 88.96 | **84.22** | **86.37** | 86.19 | **81.66** | **83.71** | - | - | - |

**Soft Pedals**

| Method | # Param | Activation P(%) | Activation R(%) | Activation $F_1$(%) | Note Onset P(%) | Note Onset R(%) | Note Onset $F_1$(%) | Note w/ Offset P(%) | Note w/ Offset R(%) | Note w/ Offset $F_1$(%) | Note w/ Offset & Vel. P(%) | Note w/ Offset & Vel. R(%) | Note w/ Offset & Vel. $F_1$(%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours w/o incomplete events | 12.9M | 74.41 | 28.77 | 36.54 | 20.24 | 9.08 | 11.69 | 17.19 | 7.51 | 9.76 | - | - | - |
| Ours | 12.9M | 86.42 | 83.12 | 84.09 | 24.32 | 17.39 | 19.46 | 18.51 | 13.40 | 15.06 | - | - | - |

**Table 2:** Transcription Result on Maestro v3.0.0 Dataset Test Split.

[^8]: Use their provided code and pretrained weights. Recomputed from transcribed MIDIs.

[^9]: Previous SOTA for sustain pedals. Their released code indicates a 200 ms onset tolerance for pedal evaluation, contrary to the reported 50 ms in their paper. Here, we use a 50 ms onset tolerance, which explains the large discrepancy between the numbers here and their reported results.

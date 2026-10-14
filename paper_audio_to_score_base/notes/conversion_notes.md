# Conversion notes

- Source PDF: `Papers/A_Transformer_Approach_for_Polyphonic_Audio-to-Score_Transcription.pdf`; copied without modification to `original/paper.pdf`.
- The source is a five-page, two-column ICASSP 2024 PDF with digitally extractable text. The text was reordered by section and column to avoid mixing the two columns.
- The document contains sections 1–6, one numbered equation (Eq. 1), two figures, Table 1, footnotes, and references. No appendix or supplementary material is present in the PDF.
- Equation 1 was reconstructed in LaTeX from the rendered PDF. The notation and exponents were checked against the page image; no equation was replaced by a textual description.
- `figures/figure_01.png` is the image object extracted directly from the PDF.
- The extracted image object and its alpha mask are retained in `notes/figure_01_extracted_source.png` and `notes/figure_01_extracted_mask.png` as conversion artifacts; `figures/figure_01.png` is the usable extracted figure.
- Figure 2 is not exposed as a separate image object by the PDF parser. `figures/figure_02.png` is therefore a crop of the original page rendering containing the figure, without redrawing or reconstructing its contents.
- Table 1 is represented in `tables/table_01.md`; all numeric values were transcribed as printed. The paper’s gray-cell shading cannot be represented reliably in a plain Markdown table, so the caption preserves the intra-/inter-composer distinction.
- `references.bib` contains only bibliographic fields present in the paper. DOI data were not added to individual entries when absent from the reference list.
- The PDF includes IEEE copyright/download notices in its page footers; these are source metadata rather than article content and are not reproduced in `paper.md`.
- The article contains a small number of grammatical or typographical forms (for example, “the the” and “This correlation suggest”). They were preserved rather than silently corrected.

# Data are not distributed in this repository

This directory intentionally contains no biomedical datasets, patient-level
metadata, embeddings, model weights, or generated predictions.

Obtain source data from the providers listed in [`docs/DATA.md`](../docs/DATA.md)
and place local files under the paths shown in
[`configs/paths.example.yaml`](../configs/paths.example.yaml). The repository
`.gitignore` blocks common biomedical, pathology, embedding, and checkpoint
formats to reduce the risk of accidental commits.

Only metadata that the upstream provider permits you to redistribute should be
placed here. TCGA controlled-access files and all model checkpoints remain
subject to their original access and licence conditions.

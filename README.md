# ContextReliabilityBioFM

Code release for **Context Is Not a Side Note: A Support-Calibrated
Reliability Framework for Biomedical Foundation Models**.

The repository audits frozen biomedical foundation-model embeddings for
context-specific failure, determines whether label--context comparisons have
enough independent support, and evaluates bounded interventions under
observed-context interpolation and unseen-context extrapolation.

## What is included

- frozen-embedding preparation for CELLxGENE, TCGA, and HEST workflows;
- grouped context probes and average/worst/gap evaluation;
- label--context support summaries and support-gated comparisons;
- LC-Reweight, SCA-Align, GroupDRO, and appendix sensitivity analyses;
- donor/patient-clustered robustness analyses;
- donor-abundance, TP53 site-pair, and pancreas marker case studies;
- scripts used to generate the manuscript's main quantitative figures and
  appendix tables.

No biomedical data, patient-level metadata, pretrained weights, embeddings, or
generated predictions are distributed here. See [Data access](docs/DATA.md).

## Repository map

```text
.
├── scripts/                 # executable analysis and plotting entry points
├── configs/                 # local-path template; no credentials
├── docs/
│   ├── DATA.md              # official data/model access links and restrictions
│   ├── INPUT_SCHEMAS.md     # required metadata and embedding schemas
│   ├── REPRODUCIBILITY.md   # stage-by-stage reproduction commands
│   └── SCRIPT_MAP.md        # script-to-analysis/figure map
├── tests/                   # data-free synthetic smoke test
├── environment.yml
├── requirements-core.txt
└── requirements-models.txt
```

## Installation

The grouped audit and statistical summaries do not require a GPU:

```bash
conda env create -f environment.yml
conda activate context-reliability-biofm
```

For a smaller CPU-only environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-core.txt
```

Model-specific extraction requires the upstream Geneformer, scGPT, UNI,
CONCH, Virchow2, or H-optimus0 access conditions to be satisfied. Those
repositories and checkpoints are not vendored here.

## Minimal context audit

The generic CELLxGENE entry point expects one metadata row per embedding row.
See [input schemas](docs/INPUT_SCHEMAS.md).

```bash
python scripts/eval_cellxgene_embedding_audit.py \
  --metadata-csv data/metadata/cellxgene/bone_marrow.csv \
  --embedding-file data/embeddings/cellxgene/geneformer_bone_marrow.npy \
  --output-dir outputs/audit/geneformer_bone_marrow \
  --model-name geneformer_v1 \
  --context-fields assay dataset_id \
  --mitigation-context-field assay \
  --leave-one-context-fields assay dataset_id \
  --methods erm label_context_reweight linear_debias \
  --n-folds 5
```

The output contains pooled metrics, context-probe results, context-bin metrics,
context gaps, out-of-fold predictions, leave-one-context predictions, and the
label--context count table.

## Support-calibrated intervention

The manuscript names map to the internal experiment identifiers as follows:

| Manuscript name | Script identifier | Meaning |
|---|---|---|
| Baseline / ERM | `erm_mlp` | unmitigated lightweight head |
| LC-Reweight | `label_context_reweight` | label--context balancing only |
| SCA-Align | `sca_lite` | balancing plus support-gated conditional alignment |
| GroupDRO | `group_dro` | external worst-group comparator |

```bash
python scripts/train_support_calibrated_interventions.py \
  --metadata-csv data/metadata/cellxgene/bone_marrow.csv \
  --embedding-file data/embeddings/cellxgene/geneformer_bone_marrow.npy \
  --output-dir outputs/intervention/bone_marrow_assay/seed_1 \
  --model-name geneformer_v1 \
  --context-field assay \
  --leave-one-context-fields assay \
  --methods erm_mlp label_context_reweight sca_lite group_dro \
  --n-folds 5 \
  --epochs 20 \
  --batch-size 1024 \
  --sabca-min-group-size 20 \
  --support-min-donors 5 \
  --seed 1
```

Support is constructed from training data only. Held-out contexts are not used
for hyperparameter selection.

## Reproducing the paper workflow

The recommended order is:

1. prepare provider metadata and frozen embeddings;
2. run grouped audit and context probes;
3. run clustered uncertainty and alternative-explanation controls;
4. compare interventions separately for patient-CV and leave-one-context
   evaluation;
5. evaluate donor/patient-level scientific consequences;
6. generate tables and figures from saved prediction files.

Exact entry points and expected files are documented in
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) and
[SCRIPT_MAP.md](docs/SCRIPT_MAP.md).

## Validation

The repository includes a data-free synthetic audit:

```bash
python -m compileall -q scripts
pytest -q
```

GitHub Actions repeats these checks and rejects common local absolute paths,
credentials, and private-key material.

## Data, model, and clinical-use boundaries

This is research code, not a clinical device. Context may encode acquisition,
biology, clinical practice, or composition; a recoverable context signal is
not by itself evidence of harmful bias or a causal biological effect.
Unsupported comparisons should be reported as evidence boundaries rather than
silently optimized.

All reused data and model artifacts remain subject to their provider's access,
consent, and licence terms. The repository does not grant rights to any
third-party dataset or checkpoint.

## Citation and licence status

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The manuscript
is under review; publication metadata will be added when available.

No software licence has yet been granted for this release. Until a licence is
selected by the authors and institution, default copyright applies.

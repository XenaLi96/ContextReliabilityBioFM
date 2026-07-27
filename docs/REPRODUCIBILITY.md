# Reproducibility workflow

This document gives the dependency order. Commands use repository-relative
paths and should be adapted to the provider resources available locally.

## 1. Prepare metadata and frozen embeddings

For CELLxGENE, select cells and create aligned metadata before extracting each
representation:

```bash
python scripts/select_cellxgene_context_cells.py --help
python scripts/extract_geneformer_v1_embeddings.py --help
python scripts/extract_scgpt_embeddings.py --help
python scripts/extract_cellxgene_scvi_style_vae_embeddings.py --help
python scripts/extract_cellxgene_baseline_embeddings.py --help
python scripts/extract_cellxgene_qc_embeddings.py --help
```

For TCGA, build a manifest, download only files permitted by GDC access, and
extract one embedding per slide:

```bash
python scripts/build_tcga_organized_manifest.py --help
python scripts/download_tcga_wsi_from_metadata.py --help
python scripts/run_uni_batch_extract.py --help
python scripts/run_conch_batch_extract.py --help
python scripts/run_timm_pathology_batch_extract.py --help
python scripts/extract_tcga_image_stats_embeddings.py --help
```

For HEST:

```bash
python scripts/download_hest_subset.py --help
python scripts/build_hest_context_manifest.py --help
python scripts/extract_hest_fm_embeddings.py --help
```

## 2. Run the Context Reliability Profile

Single-cell:

```bash
python scripts/eval_cellxgene_embedding_audit.py \
  --metadata-csv <metadata.csv> \
  --embedding-file <embeddings.npy> \
  --output-dir <audit-output> \
  --model-name <model> \
  --context-fields assay dataset_id \
  --leave-one-context-fields assay dataset_id
```

Pathology:

```bash
python scripts/eval_tcga_image_context_shift.py \
  --metadata-csv <tcga-metadata.csv> \
  --features-dir <feature-directory> \
  --output-dir <audit-output> \
  --model-name UNI \
  --embedding-suffix _uni_embedding.npz \
  --label-column TP53_status \
  --context-fields site primary_diagnosis
```

Keep donor/patient IDs as the splitting unit. Do not tune on a context that is
held out for leave-one-context evaluation.

## 3. Quantify support and audit-wide robustness

The intervention support criterion uses at least 20 cells and five independent
donors in a label--context cell by default. Sensitivity analyses vary the donor
threshold rather than presenting five as an optimal value.

After audit outputs are placed under the documented `data/` layout:

```bash
python scripts/run_systematic_audit_robustness.py \
  --n-bootstrap 5000 \
  --min-cells 20 \
  --min-donors 5 \
  --support-coverage 0.8 \
  --output-dir outputs/systematic_audit_robustness
```

The script freezes eligibility and context definitions, resamples independent
donor/patient clusters for intervals, and applies BH correction across eligible
rows.

## 4. Run controls

```bash
python scripts/run_residualized_embedding_controls.py --help
python scripts/build_shuffled_context_metadata.py --help
python scripts/run_cellxgene_representation_diagnostics.py --help
```

Residualization uses training-split covariates only. Shuffled-context controls
preserve labels. Neither control is interpreted as identifying a unique causal
origin for a context gap.

## 5. Compare interventions

Run five prespecified seeds with the same grouped split, head capacity, and
epoch budget. The core method identifiers are:

```text
erm_mlp
label_context_reweight
sca_lite
group_dro
```

Use `select_tissue_mitigation_replication.py` before cross-tissue runs, then
`build_cellxgene_cross_tissue_formal_matrix.py` to verify that all selected
tissue/model/context/method/seed combinations are present.

## 6. Evaluate scientific consequences

```bash
python scripts/analyze_donor_abundance_consequence.py --help
python scripts/run_pathology_prevalence_robustness.py --help
python scripts/analyze_pancreas_marker_sanity.py --help
```

Donor-level abundance is the unit of scientific comparison. TP53 calibration
must be fitted on non-held-out sites. Pancreas marker modules are derived from
the same expression data and are a sensitivity analysis, not independent
biological validation.

## 7. Generate figures and tables

Main plotting scripts read saved CSV files, never raw patient data:

```bash
python scripts/plot_context_bias_bubble_atlas.py
python scripts/plot_figure3_context_signal_controls.py --help
python scripts/plot_intervention_regime_figure.py --help
python scripts/plot_figure5_intervention_consequence.py
python scripts/plot_scientific_consequence_robustness_figures.py
```

Figure source CSVs are written under `data/figure_source/` when supported.
Generated figures and LaTeX tables are ignored by git.

## 8. Validate the release

```bash
python -m compileall -q scripts
pytest -q
```

For a full run, additionally archive the command line, environment export,
provider release IDs, model revisions/checksums, random seeds, and the manifest
of included/excluded samples.

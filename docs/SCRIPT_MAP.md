# Script map

The release keeps scripts flat because several experiment entry points import
shared loaders from the same directory. This page groups them by scientific
role.

## Data and embedding preparation

- CELLxGENE selection and alignment:
  `select_cellxgene_context_cells.py`,
  `build_cellxgene_balanced_context_embedding_subset.py`.
- Single-cell representations:
  `extract_geneformer_v1_embeddings.py`, `extract_scgpt_embeddings.py`,
  `extract_cellxgene_scvi_style_vae_embeddings.py`,
  `extract_cellxgene_baseline_embeddings.py`,
  `extract_cellxgene_qc_embeddings.py`.
- TCGA manifests and molecular labels:
  `build_tcga_organized_manifest.py`, `build_tcga_cancer_type_manifest.py`,
  `build_tcga_cbio_mutation_manifest.py`,
  `build_tcga_cbio_patient_mutation_manifest.py`,
  `download_tcga_wsi_from_metadata.py`.
- Pathology representations:
  `run_uni_batch_extract.py`, `run_conch_batch_extract.py`,
  `run_timm_pathology_batch_extract.py`,
  `extract_tcga_image_stats_embeddings.py`.
- HEST preparation:
  `download_hest_subset.py`, `build_hest_context_manifest.py`,
  `select_hest_replication_subset.py`, `extract_hest_fm_embeddings.py`,
  `extract_hest_image_stats.py`.

## Context audit and support

- Primary single-cell audit: `eval_cellxgene_embedding_audit.py`.
- Patient-context screens: `eval_cellxgene_patient_context.py`.
- Primary pathology audit: `eval_tcga_image_context_shift.py`.
- Pathology probes and shared heads: `eval_tcga_context_probes.py`,
  `eval_tcga_embedding_mitigation.py`.
- HEST appendix audit: `audit_context_manifest.py`,
  `eval_hest_bias_suite.py`, `eval_hest_representation_probe.py`,
  `eval_context_predictions.py`, `run_hest_image_stats_baseline.py`,
  `summarize_hest_context_shift_mainline.py`.
- Audit-wide clustered testing: `run_systematic_audit_robustness.py`,
  `bootstrap_tcga_image_context_shift_existing.py`.

## Alternative-explanation controls

- Training-split artifact residualization:
  `run_residualized_embedding_controls.py`,
  `build_residualized_control_tables.py`.
- Label-preserving context shuffle: `build_shuffled_context_metadata.py`.
- Representation and factorization diagnostics:
  `run_cellxgene_representation_diagnostics.py`,
  `run_cellxgene_factorized_representation_diagnostics.py`,
  `build_factorized_representation_table.py`.
- Remaining size/probe controls:
  `build_remaining_single_cell_controls.py`.

## Intervention family

- Training entry point:
  `train_support_calibrated_interventions.py`.
- Prespecified replication selection and completeness checks:
  `select_tissue_mitigation_replication.py`,
  `build_cellxgene_cross_tissue_formal_matrix.py`.
- Main and external-baseline summaries:
  `build_support_calibrated_main_table.py`,
  `build_external_baseline_tables.py`.
- Tempered weighting:
  `build_tempered_weighting_sensitivity_table.py`.
- Support sensitivity:
  `run_semisynthetic_support_experiment.py`,
  `build_semisynthetic_support_multiseed_table.py`.
- HEST appendix intervention: `eval_hest_context_mitigation.py`.

## Scientific consequences

- Donor abundance:
  `analyze_donor_abundance_consequence.py`,
  `build_donor_abundance_appendix_table.py`.
- TP53 site-pair robustness:
  `run_pathology_prevalence_robustness.py`,
  `analyze_scientific_consequence_cases.py`.
- Pancreas marker sensitivity:
  `analyze_pancreas_marker_sanity.py`.
- Robustness appendix assembly:
  `build_robustness_appendix_tables.py`.

## Main figures

| Manuscript output | Primary script |
|---|---|
| Context-failure atlas | `plot_context_bias_bubble_atlas.py` |
| Context signal and controls | `plot_figure3_context_signal_controls.py` |
| Intervention comparison | `plot_intervention_regime_figure.py` |
| Intervention consequence case | `plot_figure5_intervention_consequence.py` |
| TP53 and pancreas robustness figures | `plot_scientific_consequence_robustness_figures.py` |
| Earlier combined consequence review figure | `plot_scientific_consequence_cases.py` |
| Support sensitivity appendix | `plot_support_identifiability_curve.py` |

Supporting table builders include `build_paper_main_context_tables.py`,
`build_patient_context_focus_tables.py`, and
`build_patient_bin_context_tables.py`.

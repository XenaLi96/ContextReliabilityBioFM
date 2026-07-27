# Input and output schemas

## CELLxGENE-style metadata

`eval_cellxgene_embedding_audit.py` and
`train_support_calibrated_interventions.py` expect one row per cell.

Required columns:

| Column | Meaning |
|---|---|
| `cell_index` | integer row identifier; must align with the embedding matrix |
| `donor_id` | independent grouping unit for splitting and uncertainty |
| `label` | downstream target, normally cell type |
| context column | for example `assay` or `dataset_id` |

Optional audit columns include `disease`, `sex`, `age_group`, and additional
provider metadata. Missing values should be represented consistently rather
than mixed across empty strings, `NA`, and provider-specific codes.

Embeddings may be:

- `.npy`: a two-dimensional array;
- `.npz`: a two-dimensional array under `embeddings` unless
  `--embedding-key` is changed;
- `.csv`: numeric columns only.

The number and order of rows must match the metadata.

## TCGA pathology metadata

The pathology audit requires:

| Column | Meaning |
|---|---|
| `slide_file_name` | relative WSI path and embedding key source |
| label column | endpoint supplied through `--label-column`, such as `TP53_status` |
| `site` | tissue-source site or another prespecified site field |
| `primary_diagnosis` | diagnosis context when evaluated |
| `patient_id` | independent patient identifier where available |

Additional fields such as `platform`, `sex`, or age bins can be supplied as
context screens. Each feature file is named from the slide path according to
the extractor's documented suffix.

## HEST metadata

`build_hest_context_manifest.py` constructs the working manifest from provider
metadata and downloaded files. The evaluation scripts use sample ID, patient
or donor ID when available, organ, disease, site, platform, preservation,
species, and paths to HEST patch/expression files.

## Saved prediction schema

Statistical robustness scripts operate on saved out-of-fold or held-out-context
predictions. The shared minimum fields are:

| Column | Meaning |
|---|---|
| `true_label` | reference target |
| `pred_label` | model prediction |
| `donor_id` or `patient_id` | resampling cluster |
| context column | evaluated assay/dataset/site value |
| `method` | internal method identifier |
| `split_type` | for example `patient_level_cv` or `leave_one_context` |

Some case studies additionally require predicted probabilities, seed,
held-out context, or task/model identifiers. The generating script writes
these fields and a JSON summary alongside the CSV files.

## Primary outputs

The generic single-cell audit writes:

- `context_probe_results.csv`;
- `fold_metrics.csv` and `predictions.csv`;
- `subgroup_metrics.csv` and `subgroup_gaps.csv`;
- `label_context_counts.csv`;
- `leave_one_context_metrics.csv`;
- `leave_one_context_gaps.csv`;
- `leave_one_context_predictions.csv`;
- `summary.json`.

Generated outputs are deliberately ignored by git.

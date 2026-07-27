# Data and model access

No source dataset, patient-level manifest, embedding, prediction file, or model
checkpoint is committed to this repository. The scripts accept local paths so
that users can obtain each resource directly from its provider and comply with
the provider's terms.

## Source datasets

| Resource | Role in the study | Official access route | Access notes |
|---|---|---|---|
| CZ CELLxGENE Discover / Census | single-cell assay, dataset, and donor context audits | [CELLxGENE documentation](https://cellxgene.cziscience.com/docs/01__CellxGene), [Census releases](https://cellxgene-census.readthedocs.io/en/stable/cellxgene_census_docsite_data_release_info.html) | Use a fixed Census release and record collection/dataset identifiers. Do not commit donor-level exports to this repository. |
| TCGA through NCI GDC | pathology slides, clinical/site metadata, and molecular endpoints | [GDC Data Portal](https://portal.gdc.cancer.gov/), [GDC API](https://docs.gdc.cancer.gov/API/Users_Guide/Getting_Started/) | GDC contains both open- and controlled-access files. Controlled-access content requires the provider's authorization and must not be redistributed. |
| cBioPortal TCGA PanCancer Atlas | mutation labels used in molecular case studies | [cBioPortal datasets](https://www.cbioportal.org/datasets), [Web API documentation](https://docs.cbioportal.org/web-api-and-clients/) | Public study data can be queried by study and molecular-profile identifiers. Preserve the study version and identifiers in the local manifest. |
| HEST-1k | spatial pathology appendix replication | [MahmoodLab HEST dataset](https://huggingface.co/datasets/MahmoodLab/hest), [HEST code](https://github.com/mahmoodlab/HEST) | The Hugging Face dataset requires acceptance of its access conditions and is approximately terabyte scale. Use `download_hest_subset.py` for selected sample IDs. |

## Foundation models

| Model | Official source | Repository note |
|---|---|---|
| Geneformer | [Hugging Face: ctheodoris/Geneformer](https://huggingface.co/ctheodoris/Geneformer) | The script uses provider model files through Transformers. Follow the upstream licence and model-card terms. |
| scGPT | [GitHub: bowang-lab/scGPT](https://github.com/bowang-lab/scGPT) | Install the upstream package/source separately and provide its path through `SCGPT_SOURCE_DIR`. |
| UNI | [GitHub: mahmoodlab/UNI](https://github.com/mahmoodlab/UNI), [Hugging Face: MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI) | Weight access is gated. Store the downloaded checkpoint outside git and pass `--checkpoint-path`. |
| CONCH | [GitHub: mahmoodlab/CONCH](https://github.com/mahmoodlab/CONCH) | Install the upstream source separately; checkpoint access and use follow upstream terms. |
| Virchow2 | [Hugging Face: paige-ai/Virchow2](https://huggingface.co/paige-ai/Virchow2) | The model card and provider terms govern checkpoint use. |
| H-optimus-0 | [Hugging Face: bioptimus/H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0) | Access requires accepting the provider's conditions. |

## Recommended local layout

```text
data/
├── raw/
│   ├── cellxgene/
│   ├── tcga/
│   └── hest/
├── metadata/
│   ├── cellxgene/
│   ├── tcga/
│   └── hest/
└── embeddings/
    ├── cellxgene/
    ├── tcga/
    └── hest/

checkpoints/
├── uni/
├── conch/
├── virchow2/
└── hoptimus0/
```

All directories above are ignored by git. Copy
`configs/paths.example.yaml` to a local, ignored path file if a shared cluster
uses different mount points.

## Provenance to record locally

For each run, retain:

- provider release, collection/study/accession, and retrieval date;
- original sample identifier and local pseudonymization policy;
- filtering and exclusion rules;
- model repository/revision and checkpoint checksum;
- embedding script, command, seed, and software environment;
- input and output checksums where permitted.

The scripts write run summaries where available, but provider access records
and data-use approvals remain the user's responsibility.

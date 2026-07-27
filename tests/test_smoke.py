from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def test_data_free_embedding_audit(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    rows = []
    features = []
    cell_index = 0
    for donor_index in range(12):
        for repeat in range(8):
            label = "A" if repeat % 2 == 0 else "B"
            assay = "assay_1" if (repeat // 2) % 2 == 0 else "assay_2"
            rows.append(
                {
                    "cell_index": cell_index,
                    "donor_id": f"donor_{donor_index:02d}",
                    "label": label,
                    "assay": assay,
                    "dataset_id": f"dataset_{1 + donor_index % 2}",
                }
            )
            features.append(
                [
                    1.5 if label == "A" else -1.5,
                    1.0 if assay == "assay_1" else -1.0,
                    *rng.normal(0.0, 0.25, size=4),
                ]
            )
            cell_index += 1

    metadata = tmp_path / "metadata.csv"
    embeddings = tmp_path / "embeddings.npy"
    output = tmp_path / "audit"
    pd.DataFrame(rows).to_csv(metadata, index=False)
    np.save(embeddings, np.asarray(features, dtype=np.float32))

    command = [
        sys.executable,
        "scripts/eval_cellxgene_embedding_audit.py",
        "--metadata-csv",
        str(metadata),
        "--embedding-file",
        str(embeddings),
        "--output-dir",
        str(output),
        "--model-name",
        "synthetic",
        "--context-fields",
        "assay",
        "dataset_id",
        "--mitigation-context-field",
        "assay",
        "--leave-one-context-fields",
        "assay",
        "--methods",
        "erm",
        "--n-folds",
        "3",
        "--min-probe-cells",
        "10",
        "--min-holdout-cells",
        "20",
    ]
    subprocess.run(command, check=True)

    assert (output / "summary.json").is_file()
    assert (output / "context_probe_results.csv").is_file()
    assert (output / "subgroup_gaps.csv").is_file()
    assert (output / "leave_one_context_gaps.csv").is_file()

# Contributing

This repository is a research release tied to a manuscript. Please open an
issue before changing evaluation definitions, support thresholds, split logic,
or method identifiers.

Pull requests should:

1. include a minimal test or data-free fixture;
2. preserve donor/patient grouping and training-only support construction;
3. document new input/output columns;
4. avoid committing data, embeddings, checkpoints, predictions, or local
   paths;
5. pass `python -m compileall -q scripts` and `pytest -q`.

Do not report participant-level or sensitive information in issues.

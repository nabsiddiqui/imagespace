# ImageSpace tests

Two complementary layers verify the pipeline. Both run in CI
(`.github/workflows/ci.yml`).

## 1. Network-free numeric suite (`test_pipeline.py`)

Fast unit tests for the dependency-sensitive core of the pipeline: PCA,
openTSNE, HDBSCAN, k-nearest-neighbors, outlier scoring, image-feature
extraction, label-vocabulary loading, and the binary/CSV/manifest writers.
No models are downloaded and no frontend build is required, so the suite runs
in a second or two and can be exercised across many environments.

```bash
pip install ".[test]"
pytest
```

**Cross-version / cross-dependency robustness.** CI runs this suite on Python
3.10, 3.11, and 3.12, plus a "minimum dependencies" leg that installs the
oldest supported versions from [`constraints-min.txt`](../constraints-min.txt).
This is the evidence that the pipeline keeps working as the underlying
software and its dependencies change over time.

| Leg | Python | Dependencies |
| --- | --- | --- |
| latest  | 3.10 / 3.11 / 3.12 | newest resolvable |
| minimum | 3.10 | pinned floors (`constraints-min.txt`) |

## 2. End-to-end acceptance contract (`run_checks.py`)

A single command that runs the whole pipeline on the bundled 500-image WikiArt
sample and asserts stronger, integration-level invariants: two seeded runs are
**byte-identical**, provenance/manifest records agree, HDBSCAN noise is
preserved and encoded consistently across CSV and binary, cluster labels carry
top-3 evidence, the frontend bundle builds and wires up the Methods panel, and
the wheel ships the bundled label vocabulary.

```bash
pip install ".[full]"     # onnxruntime + torch + transformers
# plus Node.js for the frontend build
python tests/run_checks.py --seed 42
```

This contract needs model weights (first run downloads the CLIP ONNX model) and
Node.js, so it is the local/full-environment check rather than part of the fast
matrix. Run it before committing pipeline changes.

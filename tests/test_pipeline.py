"""Network-free unit tests for the ImageSpace numeric pipeline.

These exercise the dependency-sensitive core (PCA, t-SNE, HDBSCAN, k-NN, image
features, binary/CSV writers) without downloading models or building the
frontend, so they can run fast across a matrix of Python and dependency
versions. The heavy end-to-end contract lives in tests/run_checks.py.
"""

import json
import struct

import numpy as np


# ── Dimensionality reduction + clustering ────────────────────
def test_reduce_dimensions_shapes_and_provenance(ispace, embeddings):
    coords, raw, clusters, pca, probs, analysis = ispace.reduce_dimensions(
        embeddings, min_cluster_size=5, perplexity=8, seed=42,
        pca_dims=20, hdbscan_min_samples=2, hdbscan_selection_method="leaf",
    )
    n = len(embeddings)
    assert coords.shape == (n, 2) and coords.dtype == np.float32
    assert raw.shape == (n, 2)
    assert clusters.shape == (n,) and clusters.dtype == np.int32
    assert pca.shape == (n, 20)
    assert probs.shape == (n,)
    assert 0.0 < analysis["pca"]["explainedVariance"] <= 1.0
    assert analysis["pca"]["dimensions"] == 20
    assert analysis["tsne"]["seed"] == 42
    assert analysis["hdbscan"]["minSamples"] == 2
    assert analysis["hdbscan"]["selectionMethod"] == "leaf"
    assert analysis["hdbscan"]["noisePolicy"] == "preserve-unassigned"


def test_pca_dims_clamped_to_available(ispace, embeddings):
    # Requesting more dims than n-1 must clamp, not crash.
    _, _, _, pca, _, analysis = ispace.reduce_dimensions(
        embeddings, min_cluster_size=5, perplexity=8, seed=1, pca_dims=999,
    )
    assert pca.shape[1] == analysis["pca"]["dimensions"] <= len(embeddings) - 1
    assert analysis["pca"]["requestedDimensions"] == 999


def test_layout_is_deterministic_under_fixed_seed(ispace, embeddings):
    a = ispace.reduce_dimensions(embeddings, min_cluster_size=5, perplexity=8, seed=7)
    b = ispace.reduce_dimensions(embeddings, min_cluster_size=5, perplexity=8, seed=7)
    assert np.array_equal(a[0], b[0])  # snapped coords
    assert np.array_equal(a[2], b[2])  # cluster ids


def test_noise_is_preserved_as_negative_one(ispace, embeddings):
    # Tiny clusters + strict min_samples should leave some points unassigned.
    _, _, clusters, _, probs, analysis = ispace.reduce_dimensions(
        embeddings, min_cluster_size=5, perplexity=8, seed=42,
        hdbscan_min_samples=4, hdbscan_selection_method="leaf",
    )
    noise = clusters == -1
    assert analysis["hdbscan"]["noiseCount"] == int(noise.sum())
    # Noise probability is zeroed; no reassignment to a nearest cluster.
    assert np.all(probs[noise] == 0.0)


# ── k-NN + outliers ──────────────────────────────────────────
def test_compute_knn_excludes_self(ispace, embeddings):
    idx, dist = ispace.compute_knn(embeddings, k=5)
    n = len(embeddings)
    assert idx.shape == (n, 5) and idx.dtype == np.uint32
    assert dist.shape == (n, 5) and dist.dtype == np.float32
    for i in range(n):
        assert i not in idx[i]           # self removed
    assert np.all(np.diff(dist, axis=1) >= -1e-6)  # sorted ascending


def test_outlier_scores_normalized_0_100(ispace, embeddings):
    _, dist = ispace.compute_knn(embeddings, k=5)
    scores = ispace.compute_outlier_scores(dist)
    assert scores.shape == (len(embeddings),)
    assert scores.min() >= 0.0 and scores.max() <= 100.0


# ── Label vocabulary loading + validation ────────────────────
def test_default_label_vocabulary_loads(ispace):
    vocab = ispace.load_label_vocabulary()
    assert vocab["id"] == "imagespace-art-v1"
    assert len(vocab["candidates"]) >= 3
    assert all({"text", "short_name"} <= entry.keys() for entry in vocab["candidates"])


def test_custom_label_vocabulary_from_file(ispace, tmp_path):
    payload = {"id": "custom-v1", "version": 2,
               "candidates": [{"text": "a"}, {"text": "b"}, {"text": "c"}]}
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(payload))
    vocab = ispace.load_label_vocabulary(path)
    assert vocab["id"] == "custom-v1" and vocab["version"] == 2
    assert [c["short_name"] for c in vocab["candidates"]] == ["a", "b", "c"]


def test_label_vocabulary_rejects_too_few_candidates(ispace, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(["only", "two"]))
    import pytest
    with pytest.raises(ValueError):
        ispace.load_label_vocabulary(path)


# ── Binary + manifest writers ────────────────────────────────
def test_binary_data_v3_encodes_noise_sentinel(ispace, tmp_path):
    n = 4
    coords = np.zeros((n, 2), dtype=np.float32)
    atlas = [(0, 0, 0)] * n
    preview = [(0, 0, 0)] * n
    clusters = np.array([0, -1, 2, -1], dtype=np.int32)
    ispace.write_binary_data(str(tmp_path), coords, coords, atlas, clusters, preview)

    raw = (tmp_path / "data.bin").read_bytes()
    stride = 28
    assert len(raw) == n * stride
    encoded = [struct.unpack_from("<H", raw, i * stride + 22)[0] for i in range(n)]
    assert encoded == [0, 65535, 2, 65535]


def test_binary_data_v2_without_previews(ispace, tmp_path):
    n = 3
    coords = np.zeros((n, 2), dtype=np.float32)
    atlas = [(0, 0, 0)] * n
    clusters = np.zeros(n, dtype=np.int32)
    ispace.write_binary_data(str(tmp_path), coords, coords, atlas, clusters, None)
    assert len((tmp_path / "data.bin").read_bytes()) == n * 24


def test_manifest_v3_and_v2(ispace, tmp_path):
    ispace.write_manifest(str(tmp_path), 10, 2, 128, 4096,
                          preview_atlas_count=2, preview_thumb_size=64)
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["version"] == 3 and m["bytesPerImage"] == 28 and m["hasPreviewAtlases"]

    ispace.write_manifest(str(tmp_path), 10, 2, 64, 4096)
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["version"] == 2 and m["bytesPerImage"] == 24


def test_neighbors_bin_roundtrip(ispace, tmp_path):
    idx = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.uint32)
    dist = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    ispace.write_neighbors_bin(str(tmp_path), idx, dist)
    raw = (tmp_path / "neighbors.bin").read_bytes()
    count, k = struct.unpack_from("<II", raw, 0)
    assert count == 3 and k == 2


# ── Image-derived features (uses PIL, no network) ────────────
def test_image_features_and_colors(ispace, sample_images):
    feats = ispace.compute_image_features(sample_images, thumb_size=16)
    for key in ("brightness", "complexity", "edge_density"):
        arr = feats[key]
        assert len(arr) == len(sample_images)
        assert arr.min() >= 0.0 and arr.max() <= 100.0
    colors = ispace.extract_dominant_colors(sample_images, thumb_size=16)
    assert len(colors) == len(sample_images)
    assert all(len(c) == 3 for c in colors)


# ── Discovery + fingerprint + relayout refresh ───────────────
def test_discover_images_and_fingerprint_stable(ispace, sample_images, tmp_path):
    found = ispace.discover_images(tmp_path)
    assert len(found) == len(sample_images)
    fp1 = ispace.corpus_fingerprint(found, tmp_path)
    fp2 = ispace.corpus_fingerprint(found, tmp_path)
    assert fp1 == fp2 and len(fp1) == 64


def test_refresh_metadata_drops_legacy_display_cluster(ispace, tmp_path):
    path = tmp_path / "metadata.csv"
    path.write_text(
        "id,filename,cluster,display_cluster\n"
        "0,a.png,0,1\n1,b.png,-1,0\n"
    )
    ispace.refresh_metadata_clusters(
        str(path),
        np.array([2, -1], dtype=np.int32),
        np.array([50.0, 0.0], dtype=np.float32),
    )
    text = path.read_text()
    assert "display_cluster" not in text
    assert "cluster_confidence" in text

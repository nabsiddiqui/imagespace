"""Shared pytest fixtures and import path setup.

The pipeline ships as a single-file module at ``scripts/imagespace.py``. When the
package is installed (``pip install .``) ``import imagespace`` works directly; when
running from a source checkout we add ``scripts/`` to the path so the same import
resolves without an install step.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import imagespace  # noqa: E402


@pytest.fixture(scope="session")
def ispace():
    """The imported pipeline module under test."""
    return imagespace


@pytest.fixture
def embeddings():
    """Deterministic, L2-normalized synthetic CLIP-like embeddings (60 x 64)."""
    rng = np.random.default_rng(0)
    # Three loose blobs so HDBSCAN has some density structure to find.
    blobs = [rng.normal(center, 0.35, size=(20, 64)) for center in (-2.0, 0.0, 2.0)]
    data = np.vstack(blobs).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    return data


@pytest.fixture
def sample_images(tmp_path):
    """Write a handful of tiny solid-color PNGs and return their paths."""
    from PIL import Image

    paths = []
    for i in range(6):
        img = Image.new("RGB", (16, 16), (i * 40 % 256, 80, 160))
        p = tmp_path / f"img_{i:02d}.png"
        img.save(p)
        paths.append(p)
    return paths

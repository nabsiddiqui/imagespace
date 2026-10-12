#!/usr/bin/env python3
"""
ImageSpace — Fast Pipeline for Exploratory Visualization of Image Collections

Transforms a folder of images into a self-contained, uploadable static site:
  - Viewer app shell (index.html + assets/ + favicon.svg) for the ImageSpace viewer
  - Atlas WebP textures (sprite sheets of thumbnails) under data/
  - Binary layout data (UMAP coordinates, atlas positions, cluster IDs) under data/
  - Manifest JSON (metadata for the viewer) under data/
  - Metadata CSV (merged with external metadata if provided) under data/

The output directory is directly uploadable to any static host (GitHub Pages,
Netlify, Cloudflare Pages, or even a local `python3 -m http.server`). No Vite
build or backend is required to view the result.

Dual-resolution progressive loading is ON by default: 64px preview atlases are
generated alongside 128px HD atlases, so the viewer shows low-res images first
and upgrades to HD on capable devices. Pass --no-hd for single-resolution output.

Usage:
    imagespace -i /path/to/images -o ./output
    imagespace -i /path/to/images -o ./output --metadata existing.csv
    imagespace -i /path/to/images -o ./output --seed 42   # reproducible layout
    imagespace /path/to/images -o ./output            # positional input also works
    imagespace -i /path/to/images -o ./output --no-hd # single-resolution only

Install as a command (from this repo):
    pip install .

Performance (50K images, Apple Silicon, with --hd):
    - Atlas generation (full + preview): ~3-5 min
    - CLIP embeddings (ONNX+CoreML): ~2-3 min
    - UMAP (layout + clustering space) + HDBSCAN: several minutes (slower with --seed,
      because a fixed seed makes UMAP run single-threaded)
    - Metadata + features: ~1-2 min
    - Total: ~16-40 min (CPU-only CLIP is much slower)

Dependencies:
    Required: pillow, numpy, scikit-learn, umap-learn, hdbscan
    Optional: onnxruntime (for CLIP embeddings via ONNX — highly recommended)
              torch, transformers (alternative CLIP backend — slower)
              huggingface_hub (for downloading ONNX model)
"""

import argparse
import colorsys
import csv
import json
import hashlib
import os
import shutil
import struct
import sys
import time
from importlib import resources
from pathlib import Path
from multiprocessing import cpu_count

import numpy as np
from PIL import Image

# ── Configuration ─────────────────────────────────────────────
THUMB_SIZE = 64  # Thumbnail size in pixels (square)
ATLAS_SIZE = 4096  # Atlas texture size (4096x4096 = 64x64 grid of 64px thumbs)
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}
CLIP_IMAGE_SIZE = 224  # CLIP input resolution
CLIP_DIM = 512  # CLIP ViT-B/32 embedding dimension
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
BATCH_SIZE = 64  # Embedding batch size (larger = faster with ONNX)
# UMAP for the 2-D layout (visualization defaults from umap-learn).
UMAP_LAYOUT_NEIGHBORS = 15
UMAP_LAYOUT_MIN_DIST = 0.1
# UMAP for the clustering space (clustering guidance from the umap-learn docs:
# larger n_neighbors, min_dist 0, a few dimensions rather than 2).
UMAP_CLUSTER_NEIGHBORS = 30
UMAP_CLUSTER_MIN_DIST = 0.0
UMAP_CLUSTER_DIMS = 10
UMAP_METRIC = "cosine"  # CLIP embeddings are compared by cosine similarity
SOFTWARE_VERSION = "1.4.0"
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_LABEL_PRESET = "art-v1"
DEFAULT_LABEL_UNCERTAINTY_THRESHOLD = 0.01
ONNX_CLIP_REPO = "Xenova/clip-vit-base-patch32"


# ── Stage 1: Image Discovery ─────────────────────────────────
def discover_images(input_dir):
    """Recursively find all supported image files, skipping hidden/system dirs."""
    images = []
    input_path = Path(input_dir).resolve()
    for root, dirs, files in os.walk(input_path):
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".")
            and d not in ("__pycache__", "node_modules", ".git")
        ]
        for f in sorted(files):
            if f.startswith("."):
                continue
            ext = Path(f).suffix.lower()
            if ext in SUPPORTED_FORMATS:
                images.append(Path(root) / f)
    return images


def corpus_fingerprint(images, input_dir):
    """Hash stable corpus identity fields to prevent stale embedding-cache reuse."""
    root = Path(input_dir).resolve()
    digest = hashlib.sha256()
    for image in images:
        path = Path(image)
        stat = path.stat()
        try:
            relative = path.resolve().relative_to(root)
        except ValueError:
            relative = path.name
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


# ── Stage 2: Thumbnail Generation + Atlas Packing ────────────
def _center_crop_thumb(img_path, size):
    """Center-crop to square and resize to `size`×`size`. Gray placeholder on failure."""
    try:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        return img.crop((left, top, left + side, top + side)).resize(
            (size, size), Image.BILINEAR
        )
    except Exception:
        return Image.new("RGB", (size, size), (128, 128, 128))


def generate_atlases(
    images, output_dir, thumb_size=THUMB_SIZE, atlas_size=ATLAS_SIZE, quality=60
):
    """Create WebP atlas textures from image thumbnails. Returns atlas metadata per image."""
    images_per_row = atlas_size // thumb_size
    images_per_atlas = images_per_row * images_per_row

    atlas_data = []  # (atlas_idx, u, v) per image
    current_atlas_idx = 0
    current_img_in_atlas = 0
    atlas_img = Image.new("RGB", (atlas_size, atlas_size), (255, 255, 255))

    start = time.time()
    for idx, img_path in enumerate(images):
        if idx % 2000 == 0:
            elapsed = time.time() - start
            rate = idx / elapsed if elapsed > 0 else 0
            eta = (len(images) - idx) / rate if rate > 0 else 0
            print(
                f"  Thumbnailing {idx}/{len(images)} ({rate:.0f} img/s, ETA {eta:.0f}s)",
                end="\r",
            )

        img = _center_crop_thumb(img_path, thumb_size)

        col = current_img_in_atlas % images_per_row
        row = current_img_in_atlas // images_per_row
        u, v = col * thumb_size, row * thumb_size
        atlas_img.paste(img, (u, v))
        atlas_data.append((current_atlas_idx, u, v))

        current_img_in_atlas += 1
        if current_img_in_atlas >= images_per_atlas or idx == len(images) - 1:
            atlas_path = os.path.join(output_dir, f"atlas_{current_atlas_idx}.webp")
            atlas_img.save(atlas_path, "WEBP", quality=quality, method=2)
            print(
                f"\n  Saved atlas_{current_atlas_idx}.webp ({current_img_in_atlas} images)"
            )
            current_atlas_idx += 1
            current_img_in_atlas = 0
            atlas_img = Image.new("RGB", (atlas_size, atlas_size), (255, 255, 255))

    total = time.time() - start
    print(f"  Atlas generation: {total:.1f}s")
    return atlas_data, current_atlas_idx


def generate_preview_atlases(
    images,
    output_dir,
    atlas_count_full,
    images_per_full_atlas,
    thumb_size=64,
    atlas_size=ATLAS_SIZE,
    quality=40,
):
    """Generate 64px preview atlas textures mirroring the same image-to-atlas mapping as full atlases."""
    images_per_row = atlas_size // thumb_size
    preview_data = []

    start = time.time()
    for atlas_idx in range(atlas_count_full):
        start_img = atlas_idx * images_per_full_atlas
        end_img = min(start_img + images_per_full_atlas, len(images))

        atlas_img = Image.new("RGB", (atlas_size, atlas_size), (255, 255, 255))
        count_in_atlas = 0

        for img_idx in range(start_img, end_img):
            if img_idx % 2000 == 0:
                elapsed = time.time() - start
                rate = img_idx / elapsed if elapsed > 0 else 0
                print(
                    f"  Preview thumb {img_idx}/{len(images)} ({rate:.0f} img/s)",
                    end="\r",
                )

            img = _center_crop_thumb(images[img_idx], thumb_size)

            local_idx = img_idx - start_img
            col = local_idx % images_per_row
            row = local_idx // images_per_row
            u, v = col * thumb_size, row * thumb_size
            atlas_img.paste(img, (u, v))

            preview_data.append((atlas_idx, u, v))
            count_in_atlas += 1

        atlas_path = os.path.join(output_dir, f"atlas_{atlas_idx}_preview.webp")
        atlas_img.save(atlas_path, "WEBP", quality=quality, method=2)
        print(f"\n  Saved atlas_{atlas_idx}_preview.webp ({count_in_atlas} images)")

    total = time.time() - start
    print(f"  Preview atlas generation: {total:.1f}s")
    return preview_data, atlas_count_full


# ── Stage 3: Embedding Extraction ────────────────────────────
def _load_rgb_batch(paths):
    """Open a batch of images as RGB, substituting a gray placeholder on failure."""
    batch = []
    for path in paths:
        try:
            batch.append(Image.open(path).convert("RGB"))
        except Exception:
            batch.append(Image.new("RGB", (CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE), (128, 128, 128)))
    return batch


def _print_rate(label, done, total, start):
    """Print an in-place progress line with throughput and ETA (stdout only)."""
    elapsed = time.time() - start
    rate = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else 0
    print(f"  {label} {done}/{total} ({rate:.1f} img/s, ETA {eta:.0f}s)", end="\r")


def extract_embeddings(images, model_id=DEFAULT_CLIP_MODEL):
    """Extract CLIP embeddings and return ``(embeddings, backend)``.

    The optimized ONNX artifact is available for the default model. Other
    Hugging Face CLIP model ids use the PyTorch backend so the image and text
    encoders always come from the same checkpoint.
    """
    if model_id == DEFAULT_CLIP_MODEL:
        try:
            return _extract_clip_onnx(images), "onnxruntime"
        except Exception as e:
            print(f"  ONNX CLIP failed: {e}")

    try:
        return _extract_clip_torch(images, model_id), "transformers"
    except Exception as e:
        raise RuntimeError(
            f"CLIP embedding extraction failed for {model_id}. Install "
            "onnxruntime for the default model or torch/transformers."
        ) from e


def _get_onnx_model_path():
    """Download or locate the default CLIP ONNX vision model."""
    cache_dir = Path.home() / ".cache" / "imagespace"
    model_path = cache_dir / "clip-vit-b32-visual.onnx"
    if model_path.exists():
        return str(model_path)

    try:
        from huggingface_hub import hf_hub_download

        print("  Downloading CLIP ONNX vision model (first time only)...")
        downloaded = hf_hub_download(
            repo_id=ONNX_CLIP_REPO,
            filename="onnx/vision_model.onnx",
            cache_dir=str(cache_dir),
        )
        return downloaded
    except Exception as e:
        print(f"  Could not download ONNX model: {e}")
        return None


def _preprocess_clip_batch(images_pil):
    """Preprocess a batch of PIL images for CLIP (resize, center-crop, normalize)."""
    batch = np.zeros(
        (len(images_pil), 3, CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE), dtype=np.float32
    )
    for i, img in enumerate(images_pil):
        w, h = img.size
        scale = CLIP_IMAGE_SIZE / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        left = (new_w - CLIP_IMAGE_SIZE) // 2
        top = (new_h - CLIP_IMAGE_SIZE) // 2
        img = img.crop((left, top, left + CLIP_IMAGE_SIZE, top + CLIP_IMAGE_SIZE))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - CLIP_MEAN) / CLIP_STD
        batch[i] = arr.transpose(2, 0, 1)
    return batch


def _extract_clip_onnx(images):
    """CLIP embeddings via ONNX Runtime (fastest path). Auto-detects GPU providers."""
    import onnxruntime as ort

    model_path = _get_onnx_model_path()
    if model_path is None:
        raise RuntimeError("ONNX model not available")

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = min(cpu_count(), 8)
    sess_opts.enable_cpu_mem_arena = True
    sess_opts.enable_mem_pattern = True
    sess_opts.enable_mem_reuse = True
    sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    providers = ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers.insert(0, "CUDAExecutionProvider")
        print("  Using NVIDIA CUDA")
    elif "CoreMLExecutionProvider" in available:
        providers.insert(0, "CoreMLExecutionProvider")
        print("  Using Apple Neural Engine (CoreML)")
    else:
        print("  Using CPU")

    print(f"  Loading CLIP ONNX model...")
    session = ort.InferenceSession(
        model_path, sess_options=sess_opts, providers=providers
    )

    input_name = session.get_inputs()[0].name
    embeddings = np.zeros((len(images), CLIP_DIM), dtype=np.float32)
    start = time.time()

    for batch_start in range(0, len(images), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(images))
        batch_images = _load_rgb_batch(images[batch_start:batch_end])

        pixel_values = _preprocess_clip_batch(batch_images)
        outputs = session.run(None, {input_name: pixel_values})[0]

        # L2 normalize
        norms = np.linalg.norm(outputs, axis=-1, keepdims=True)
        norms[norms == 0] = 1
        out_normalized = outputs / norms
        # Handle dimension mismatch (some models output differently)
        if out_normalized.shape[1] >= CLIP_DIM:
            embeddings[batch_start:batch_end] = out_normalized[:, :CLIP_DIM]
        else:
            embeddings[batch_start:batch_end, : out_normalized.shape[1]] = (
                out_normalized
            )

        _print_rate("Embedding", batch_end, len(images), start)

    print(f"\n  CLIP ONNX embeddings: {time.time() - start:.1f}s")
    return embeddings


def _extract_clip_torch(images, model_id=DEFAULT_CLIP_MODEL):
    """CLIP embeddings via transformers + PyTorch. Auto-detects GPU."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    print(f"  Loading {model_id} (PyTorch)...")
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print("  Using CUDA GPU")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("  Using Apple Silicon GPU (MPS)")
    else:
        print("  Using CPU")

    model = model.to(device).eval()
    projection_dim = int(getattr(model.config, "projection_dim", CLIP_DIM))
    embeddings = np.zeros((len(images), projection_dim), dtype=np.float32)
    start = time.time()

    for batch_start in range(0, len(images), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(images))
        batch_images = _load_rgb_batch(images[batch_start:batch_end])

        inputs = processor(images=batch_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            if hasattr(outputs, "pooler_output"):
                outputs = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state"):
                outputs = outputs.last_hidden_state[:, 0]
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)
            embeddings[batch_start:batch_end] = outputs.cpu().numpy()

        _print_rate("Embedding", batch_end, len(images), start)

    print(f"\n  CLIP PyTorch embeddings: {time.time() - start:.1f}s")
    return embeddings


# ── Stage 4: Dimensionality Reduction + Clustering ───────────
def auto_min_cluster_size(n_images):
    """Default HDBSCAN min_cluster_size: about 1% of the collection, rounded
    to the nearest 10, never below 10 (49,585 images -> 500)."""
    return max(10, int(round(n_images / 100, -1)))


def _run_umap(embeddings, n_components, n_neighbors, min_dist, seed, label):
    """Fit UMAP on the CLIP embeddings with cosine distance.

    A fixed seed makes umap-learn deterministic (and single-threaded).
    """
    import umap

    print(
        f"\n  UMAP ({label}): {embeddings.shape[1]}-d → {n_components}-d "
        f"(n_neighbors={n_neighbors}, min_dist={min_dist}, metric={UMAP_METRIC})..."
    )
    start = time.time()
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=UMAP_METRIC,
        init="spectral",
        random_state=seed,
    )
    coords = np.asarray(reducer.fit_transform(embeddings), dtype=np.float64)
    print(f"  UMAP ({label}): {time.time() - start:.1f}s")
    return coords


def reduce_dimensions(
    embeddings,
    min_cluster_size=50,
    layout_neighbors=UMAP_LAYOUT_NEIGHBORS,
    thumb_size=THUMB_SIZE,
    seed=None,
    cluster_dims=UMAP_CLUSTER_DIMS,
    hdbscan_min_samples=5,
    hdbscan_selection_method="leaf",
    layout_min_dist=UMAP_LAYOUT_MIN_DIST,
    cluster_neighbors=UMAP_CLUSTER_NEIGHBORS,
):
    """Run UMAP (layout + clustering space) and HDBSCAN, preserving raw noise labels.

    Two UMAP fits run on the raw CLIP embeddings with cosine distance: a 2-D
    fit for the viewer layout and a low-dimensional fit used only as the
    space HDBSCAN clusters in.
    """
    n = len(embeddings)
    if n < 5:
        raise ValueError("UMAP requires at least five images")

    # n_neighbors must be smaller than the number of images; n_components
    # must leave room for UMAP's spectral initialization.
    eff_layout_neighbors = max(2, min(layout_neighbors, n - 1))
    eff_cluster_neighbors = max(2, min(cluster_neighbors, n - 1))
    eff_cluster_dims = max(1, min(cluster_dims, n - 2))

    layout = _run_umap(
        embeddings, 2, eff_layout_neighbors, layout_min_dist, seed, "layout"
    )
    cluster_space = _run_umap(
        embeddings, eff_cluster_dims, eff_cluster_neighbors,
        UMAP_CLUSTER_MIN_DIST, seed, "clustering space",
    )
    tsne_coords = layout  # binary format keeps its historical field name

    # Scale to viewer range — ensure enough room for non-overlapping thumbnails
    n = len(tsne_coords)
    cell_size = thumb_size * 1.15  # slight gap between thumbnails
    target_side = int(np.ceil(np.sqrt(n * 1.8))) * cell_size  # 1.8x overallocation

    def scale_coords(coords, target_range):
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1
        return (coords - mins) / ranges * target_range - target_range / 2

    tsne_coords = scale_coords(tsne_coords, target_side)

    # Preserve raw UMAP coords (with natural overlap) for the viewer's UMAP mode
    raw_tsne_coords = tsne_coords.copy()

    # Remove overlaps by snapping to nearest unoccupied grid cell
    print(
        f"  Removing overlaps (cell={cell_size:.0f}px, grid≈{int(target_side / cell_size)}²)..."
    )
    start = time.time()
    occupied = set()
    result = np.zeros_like(tsne_coords)
    # Process from center outward to preserve cluster cores
    centroid = tsne_coords.mean(axis=0)
    dists = np.linalg.norm(tsne_coords - centroid, axis=1)
    order = np.argsort(dists)
    for idx in order:
        gx = round(tsne_coords[idx, 0] / cell_size)
        gy = round(tsne_coords[idx, 1] / cell_size)
        if (gx, gy) not in occupied:
            occupied.add((gx, gy))
            result[idx] = [gx * cell_size, gy * cell_size]
            continue
        # Spiral search for nearest free cell
        placed = False
        for r in range(1, 2000):
            for dx in range(-r, r + 1):
                for dy in (-r, r):
                    if (gx + dx, gy + dy) not in occupied:
                        occupied.add((gx + dx, gy + dy))
                        result[idx] = [(gx + dx) * cell_size, (gy + dy) * cell_size]
                        placed = True
                        break
                if placed:
                    break
            if placed:
                break
            for dy in range(-r + 1, r):
                for dx in (-r, r):
                    if (gx + dx, gy + dy) not in occupied:
                        occupied.add((gx + dx, gy + dy))
                        result[idx] = [(gx + dx) * cell_size, (gy + dy) * cell_size]
                        placed = True
                        break
                if placed:
                    break
            if placed:
                break
    tsne_coords = result
    print(f"  Overlap removal: {time.time() - start:.1f}s")

    # Clustering: HDBSCAN on the low-dimensional UMAP clustering space, which
    # makes density structure visible without clustering the 2-D layout itself.
    import hdbscan as hdb

    print(
        f"\n  Running HDBSCAN on {eff_cluster_dims}-d UMAP space "
        f"(min_cluster_size={min_cluster_size})..."
    )
    start = time.time()
    clusterer = hdb.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=hdbscan_min_samples,
        metric="euclidean",
        cluster_selection_method=hdbscan_selection_method,
        core_dist_n_jobs=-1,
    )
    cluster_ids = clusterer.fit_predict(cluster_space)
    cluster_probs = clusterer.probabilities_.copy()
    noise_mask = cluster_ids == -1
    n_clusters = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
    n_noise = int(noise_mask.sum())

    # Preserve HDBSCAN's raw -1 noise label. Noise remains unassigned rather
    # than being forced into a nearby cluster after clustering.
    cluster_probs[noise_mask] = 0.0
    print(
        f"  HDBSCAN: {n_clusters} clusters, {n_noise} noise preserved ({time.time() - start:.1f}s)"
    )

    return (
        tsne_coords.astype(np.float32),
        raw_tsne_coords.astype(np.float32),
        cluster_ids.astype(np.int32),
        cluster_space,
        cluster_probs,
        {
            "layout": {
                "method": "umap",
                "implementation": "umap-learn",
                "input": "clip-embedding",
                "dimensions": 2,
                "nNeighbors": eff_layout_neighbors,
                "requestedNNeighbors": layout_neighbors,
                "minDist": layout_min_dist,
                "metric": UMAP_METRIC,
                "initialization": "spectral",
                "seed": seed,
            },
            "clusterSpace": {
                "method": "umap",
                "implementation": "umap-learn",
                "input": "clip-embedding",
                "dimensions": eff_cluster_dims,
                "requestedDimensions": cluster_dims,
                "nNeighbors": eff_cluster_neighbors,
                "requestedNNeighbors": cluster_neighbors,
                "minDist": UMAP_CLUSTER_MIN_DIST,
                "metric": UMAP_METRIC,
                "initialization": "spectral",
                "seed": seed,
            },
            "hdbscan": {
                "minClusterSize": min_cluster_size,
                "minSamples": hdbscan_min_samples,
                "selectionMethod": hdbscan_selection_method,
                "metric": "euclidean",
                "noisePolicy": "preserve-unassigned",
                "noiseLabel": -1,
                "noiseCount": n_noise,
            },
        },
    )


# ── Stage 4b: k-Nearest Neighbors ────────────────────────────
def compute_knn(embeddings, k=10):
    """Compute k-nearest neighbors for each image in the full CLIP embedding
    space using cosine distance (1 - cosine similarity).
    Returns (indices, distances) each of shape (n, k)."""
    from sklearn.neighbors import NearestNeighbors

    print(
        f"\n  Computing {k}-nearest neighbors on {embeddings.shape[0]} points "
        f"({embeddings.shape[1]}-d, cosine)..."
    )
    start = time.time()
    k = min(k, len(embeddings) - 1)
    nn = NearestNeighbors(
        n_neighbors=k + 1, algorithm="brute", metric="cosine", n_jobs=-1
    )
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    # Drop each image's own entry. Exact duplicates can tie with self at
    # distance 0, so remove self by index rather than assuming column 0.
    n = len(embeddings)
    keep_idx = np.empty((n, k), dtype=indices.dtype)
    keep_dist = np.empty((n, k), dtype=distances.dtype)
    for i in range(n):
        mask = indices[i] != i
        row_idx, row_dist = indices[i][mask][:k], distances[i][mask][:k]
        keep_idx[i], keep_dist[i] = row_idx, row_dist
    indices, distances = keep_idx, keep_dist
    print(f"  k-NN: {time.time() - start:.1f}s")
    return indices.astype(np.uint32), distances.astype(np.float32)


def write_neighbors_bin(output_dir, knn_indices, knn_distances):
    """Write neighbors.bin: for each image, k neighbor indices (uint32) + k distances (float32).
    Header: uint32 count, uint32 k. Then count * k * (uint32 + float32) = count * k * 8 bytes."""
    n, k = knn_indices.shape
    # Structured array: each record is (uint32 idx, float32 dist). Layout matches the
    # original per-pair struct.pack("<If") byte-for-byte.
    dt = np.dtype([("idx", "<u4"), ("dist", "<f4")])
    paired = np.empty((n, k), dtype=dt)
    paired["idx"] = knn_indices.astype(np.uint32)
    paired["dist"] = knn_distances.astype(np.float32)
    path = os.path.join(output_dir, "neighbors.bin")
    with open(path, "wb") as f:
        f.write(np.array([n, k], dtype="<u4").tobytes())
        f.write(paired.tobytes())
    print(f"  neighbors.bin: {n} × {k} ({os.path.getsize(path) / 1024 / 1024:.1f} MB)")


# ── Stage 4c: CLIP Cluster Labels ────────────────────────────
def load_label_vocabulary(path=None):
    """Load and validate a versioned label vocabulary."""
    if path:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Label candidates file not found: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
    else:
        try:
            source = resources.files("imagespace_resources").joinpath(
                "label_presets", f"{DEFAULT_LABEL_PRESET}.json"
            )
            payload = json.loads(source.read_text(encoding="utf-8"))
        except ModuleNotFoundError:
            # Direct source invocation puts scripts/ rather than the repo root on
            # sys.path; resolve the adjacent development resource explicitly.
            source = Path(__file__).resolve().parents[1] / "imagespace_resources" / "label_presets" / f"{DEFAULT_LABEL_PRESET}.json"
            payload = json.loads(source.read_text(encoding="utf-8"))

    entries = payload.get("candidates") if isinstance(payload, dict) else payload
    if not isinstance(entries, list) or len(entries) < 3:
        raise ValueError("Label vocabulary must contain at least three candidates")

    normalized = []
    for entry in entries:
        if isinstance(entry, str):
            text, short_name = entry.strip(), entry.strip()
        elif isinstance(entry, dict):
            text = str(entry.get("text", "")).strip()
            short_name = str(entry.get("short_name", text)).strip()
        else:
            raise ValueError("Each label candidate must be a string or object")
        if not text:
            raise ValueError("Label candidates cannot be empty")
        normalized.append({"text": text, "short_name": short_name or text})

    vocab_id = payload.get("id", Path(str(source)).stem) if isinstance(payload, dict) else Path(str(source)).stem
    version = payload.get("version", 1) if isinstance(payload, dict) else 1
    return {"id": str(vocab_id), "version": version, "candidates": normalized}


def generate_cluster_labels(
    embeddings,
    cluster_ids,
    model_id=DEFAULT_CLIP_MODEL,
    vocabulary=None,
    uncertainty_threshold=DEFAULT_LABEL_UNCERTAINTY_THRESHOLD,
):
    """Match non-noise cluster centroids against a versioned text vocabulary."""
    try:
        import torch
        from transformers import AutoTokenizer, CLIPModel
    except ImportError as exc:
        raise RuntimeError(
            "Cluster labels require torch and transformers. Install imagespace[full]."
        ) from exc

    vocabulary = vocabulary or load_label_vocabulary()
    entries = vocabulary["candidates"]
    candidates = [entry["text"] for entry in entries]
    short_names = {entry["text"]: entry["short_name"] for entry in entries}

    print(f"\n  Generating cluster labels with {model_id} and {vocabulary['id']}...")
    start = time.time()
    model = CLIPModel.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.eval()

    with torch.no_grad():
        text_inputs = tokenizer(candidates, return_tensors="pt", padding=True, truncation=True)
        text_features = model.get_text_features(**text_inputs)
        if hasattr(text_features, "pooler_output"):
            text_features = text_features.pooler_output
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_np = text_features.detach().cpu().numpy()

    if text_np.shape[1] != embeddings.shape[1]:
        raise ValueError(
            f"Image/text embedding dimensions differ for {model_id}: "
            f"{embeddings.shape[1]} vs {text_np.shape[1]}"
        )

    raw_labels = {}
    for cid in sorted(set(cluster_ids)):
        if cid < 0:
            continue
        centroid = embeddings[cluster_ids == cid].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0:
            continue
        sims = text_np @ (centroid / norm)
        top_idx = np.argsort(-sims)[:5]
        raw_labels[int(cid)] = [
            {"text": candidates[i], "score": float(sims[i])} for i in top_idx
        ]

    from collections import Counter

    primary_labels = {cid: values[0]["text"] for cid, values in raw_labels.items()}
    duplicates = {
        label for label, count in Counter(primary_labels.values()).items() if count > 1
    }
    labels = {}
    used_labels = set()
    for cid, ranked in raw_labels.items():
        primary = ranked[0]["text"]
        label = short_names.get(primary, primary)
        if primary in duplicates:
            for candidate in ranked[1:]:
                qualifier = short_names.get(candidate["text"], candidate["text"])
                combined = f"{label} — {qualifier}"
                if combined not in used_labels:
                    label = combined
                    break
        if label in used_labels:
            label = f"{label} #{cid}"
        used_labels.add(label)

        margin = float(ranked[0]["score"] - ranked[1]["score"])
        labels[cid] = {
            "label": label,
            "top3": ranked[:3],
            "scoreMargin": margin,
            "uncertain": margin < uncertainty_threshold,
            "uncertaintyThreshold": uncertainty_threshold,
            "vocabularyId": vocabulary["id"],
        }
        marker = " uncertain" if labels[cid]["uncertain"] else ""
        print(f"    Cluster {cid}: {label} ({ranked[0]['score']:.3f},{marker})")

    print(f"  Cluster labels: {len(labels)} clusters labeled ({time.time() - start:.1f}s)")
    return labels

def write_cluster_labels(output_dir, labels):
    """Write cluster_labels.json."""
    path = os.path.join(output_dir, "cluster_labels.json")
    with open(path, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"  cluster_labels.json: {len(labels)} labels")


# ── Stage 5: Dominant Color Extraction ────────────────────────
def extract_dominant_colors(images, thumb_size=32):
    """Extract dominant color (hue, sat, lum) for each image."""
    colors = []
    start = time.time()
    for idx, img_path in enumerate(images):
        try:
            img = (
                Image.open(img_path)
                .convert("RGB")
                .resize((thumb_size, thumb_size), Image.BILINEAR)
            )
            arr = np.array(img).reshape(-1, 3).mean(axis=0) / 255.0
            h, l, s = colorsys.rgb_to_hls(arr[0], arr[1], arr[2])
            colors.append((h, s, l))
        except Exception:
            colors.append((0, 0, 0))
    print(f"  Colors extracted in {time.time() - start:.1f}s")
    return colors


# ── Stage 5b: Image Features (brightness, complexity, edge density) ──
def compute_image_features(images, thumb_size=32):
    """Compute brightness, complexity (Shannon entropy), and edge density for each image.
    Returns dict with 'brightness', 'complexity', 'edge_density' arrays (0-100 scale)."""
    from scipy.ndimage import sobel

    n = len(images)
    brightness = np.zeros(n, dtype=np.float32)
    complexity = np.zeros(n, dtype=np.float32)
    edge_density = np.zeros(n, dtype=np.float32)

    start = time.time()
    for idx, img_path in enumerate(images):
        try:
            img = (
                Image.open(img_path)
                .convert("RGB")
                .resize((thumb_size, thumb_size), Image.BILINEAR)
            )
            arr = np.array(img, dtype=np.float32) / 255.0

            # Brightness: mean luminance (BT.601 weights)
            lum = arr[:, :, 0] * 0.299 + arr[:, :, 1] * 0.587 + arr[:, :, 2] * 0.114
            brightness[idx] = float(lum.mean())

            # Complexity: Shannon entropy of grayscale histogram
            gray = (lum * 255).astype(np.uint8)
            hist, _ = np.histogram(gray, bins=64, range=(0, 255))
            hist = hist[hist > 0].astype(np.float32)
            hist /= hist.sum()
            complexity[idx] = float(-np.sum(hist * np.log2(hist)))

            # Edge density: mean Sobel gradient magnitude
            sx = sobel(lum, axis=0)
            sy = sobel(lum, axis=1)
            edge_density[idx] = float(np.sqrt(sx**2 + sy**2).mean())
        except Exception:
            pass

        if idx > 0 and idx % 5000 == 0:
            print(f"    Features: {idx}/{n} ({idx / n * 100:.0f}%)")

    print(f"  Features computed in {time.time() - start:.1f}s")
    return {
        "brightness": _normalize_0_100(brightness),
        "complexity": _normalize_0_100(complexity),
        "edge_density": _normalize_0_100(edge_density),
    }


def _normalize_0_100(values):
    """Min-max scale an array to the 0-100 range, rounded to one decimal."""
    mn, mx = values.min(), values.max()
    if mx > mn:
        return ((values - mn) / (mx - mn) * 100).round(1)
    return np.zeros_like(values)


def compute_outlier_scores(knn_distances):
    """Compute outlier score from mean k-NN distance, normalized to 0-100."""
    return _normalize_0_100(knn_distances.mean(axis=1))


# ── Stage 6: Extract Timestamps ──────────────────────────────
def extract_timestamps(images):
    """Extract years from EXIF or filename year patterns. Returns plain year integers."""
    import re

    timestamps = []
    year_pattern = re.compile(r"(1[4-9]\d{2}|20[0-2]\d)")

    for img_path in images:
        year = _get_exif_year(img_path)
        if year is None:
            match = year_pattern.search(img_path.stem)
            if match:
                year = int(match.group(1))
        timestamps.append(year if year is not None else 0)
    return timestamps


def _get_exif_year(img_path):
    """Extract year from EXIF DateTimeOriginal."""
    try:
        value = Image.open(img_path).getexif().get(36867)  # DateTimeOriginal
        return int(str(value)[:4]) if value else None
    except Exception:
        pass
    return None


# ── Stage 7: Output Generation ────────────────────────────────
def write_binary_data(
    output_dir,
    snapped_coords,
    raw_tsne_coords,
    atlas_data,
    cluster_ids,
    preview_atlas_data=None,
):
    """Write binary layout data. v2=24 bytes, v3=28 bytes (with preview UVs)."""
    is_v3 = preview_atlas_data is not None
    bytes_per_image = 28 if is_v3 else 24
    fmt = "<ffffHHHHHH" if is_v3 else "<ffffHHHH"
    n = len(snapped_coords)
    binary_data = bytearray(n * bytes_per_image)
    for i in range(n):
        ai, u, v = atlas_data[i]
        cid_raw = int(cluster_ids[i])
        cid = 65535 if cid_raw < 0 else min(cid_raw, 65534)
        row = (
            float(snapped_coords[i][0]), float(snapped_coords[i][1]),
            float(raw_tsne_coords[i][0]), float(raw_tsne_coords[i][1]),
            ai, u, v, cid,
        )
        if is_v3:
            _, u_preview, v_preview = preview_atlas_data[i]
            row += (u_preview, v_preview)
        struct.pack_into(fmt, binary_data, i * bytes_per_image, *row)

    output_path = os.path.join(output_dir, "data.bin")
    with open(output_path, "wb") as f:
        f.write(binary_data)
    print(
        f"  Binary data (v{'3' if preview_atlas_data else '2'}): {len(binary_data) / 1024:.1f} KB"
    )
    return output_path


def write_manifest(
    output_dir,
    count,
    atlas_count,
    thumb_size=THUMB_SIZE,
    atlas_size=ATLAS_SIZE,
    preview_atlas_count=None,
    preview_thumb_size=64,
    provenance=None,
):
    """Write manifest.json for the viewer."""
    is_hd = preview_atlas_count is not None
    manifest = {
        "count": count,
        "atlasCount": atlas_count,
        "thumbSize": thumb_size,
        "atlasSize": atlas_size,
        "bytesPerImage": 28 if is_hd else 24,
        "version": 3 if is_hd else 2,
        "atlasFormat": "webp",
        "provenance": provenance or {},
    }
    if is_hd:
        manifest["hasPreviewAtlases"] = True
        manifest["previewThumbSize"] = preview_thumb_size
        manifest["previewAtlasCount"] = preview_atlas_count
    output_path = os.path.join(output_dir, "manifest.json")
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest (v{'3' if is_hd else '2'}): {output_path}")


def write_analysis_config(output_dir, config):
    """Write the complete, machine-readable analysis provenance contract."""
    output_path = os.path.join(output_dir, "analysis_config.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"  analysis_config.json: {output_path}")


def write_metadata_csv(
    output_dir,
    images,
    cluster_ids,
    timestamps,
    colors,
    external_metadata=None,
    image_features=None,
    outlier_scores=None,
    cluster_confidence=None,
):
    """Write metadata.csv with image info, merging with external metadata if provided."""
    output_path = os.path.join(output_dir, "metadata.csv")

    def hue_to_name(h):
        names = [
            (0.0, "red"),
            (0.05, "orange"),
            (0.12, "yellow"),
            (0.2, "yellow-green"),
            (0.33, "green"),
            (0.45, "teal"),
            (0.5, "cyan"),
            (0.58, "blue"),
            (0.7, "indigo"),
            (0.8, "purple"),
            (0.9, "magenta"),
            (1.0, "red"),
        ]
        for threshold, name in names:
            if h <= threshold:
                return name

    extra_cols = []
    if external_metadata:
        for fname, row in external_metadata.items():
            extra_cols = [c for c in row.keys() if c.lower() != "filename"]
            break

    base_cols = ["id", "filename", "cluster", "timestamp", "dominant_color"]
    feature_cols = []
    if image_features:
        feature_cols.extend(["brightness", "complexity", "edge_density"])
    if outlier_scores is not None:
        feature_cols.append("outlier_score")
    if cluster_confidence is not None:
        feature_cols.append("cluster_confidence")
    all_cols = base_cols + feature_cols + extra_cols

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(all_cols)
        matched = 0
        for i, img_path in enumerate(images):
            h, s, l = colors[i] if i < len(colors) else (0, 0, 0)
            color_name = hue_to_name(h) if s > 0.1 else "gray"
            row = [
                i,
                img_path.name,
                int(cluster_ids[i]),
                timestamps[i] if timestamps[i] > 0 else "",
                color_name,
            ]
            # Append computed features
            if image_features:
                row.append(image_features["brightness"][i])
                row.append(image_features["complexity"][i])
                row.append(image_features["edge_density"][i])
            if outlier_scores is not None:
                row.append(outlier_scores[i])
            if cluster_confidence is not None:
                row.append(cluster_confidence[i])
            if external_metadata:
                ext = external_metadata.get(img_path.name, {})
                if ext:
                    matched += 1
                for col in extra_cols:
                    row.append(ext.get(col, ""))
            writer.writerow(row)

    if external_metadata:
        print(f"  Metadata: merged {matched}/{len(images)} rows")
    else:
        print(f"  Metadata: {output_path}")


def refresh_metadata_clusters(metadata_path, cluster_ids, cluster_confidence):
    """Refresh analysis fields during --relayout without discarding metadata."""
    with open(metadata_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    # Remove the deprecated post-clustering nearest assignment from older outputs.
    if "display_cluster" in fieldnames:
        fieldnames.remove("display_cluster")
        for row in rows:
            row.pop("display_cluster", None)

    for name in ("cluster", "cluster_confidence"):
        if name not in fieldnames:
            insert_at = fieldnames.index("cluster") + 1 if "cluster" in fieldnames else len(fieldnames)
            fieldnames.insert(insert_at, name)

    updated = 0
    for position, row in enumerate(rows):
        try:
            image_id = int(row.get("id", position))
        except (TypeError, ValueError):
            image_id = position
        if not 0 <= image_id < len(cluster_ids):
            continue
        row["cluster"] = str(int(cluster_ids[image_id]))
        row["cluster_confidence"] = str(cluster_confidence[image_id])
        updated += 1

    temp_path = f"{metadata_path}.tmp"
    with open(temp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, metadata_path)
    print(f"  Metadata: refreshed cluster fields for {updated}/{len(rows)} rows")


def read_external_metadata(metadata_path):
    """Read external metadata CSV as dict: filename -> {col: val}."""
    lookup = {}
    with open(metadata_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("filename", "")
            if fname:
                lookup[fname] = {k: v for k, v in row.items() if k != "filename"}
    print(f"  Read {len(lookup)} entries from external metadata")
    return lookup


def _find_viewer_shell():
    """Locate the bundled viewer app shell (index.html + assets/ + favicon.svg).

    Resolution order:
      1. <repo>/imagespace_viewer_shell/viewer_shell — when run from a repo
         clone via `python scripts/imagespace.py` (dev workflow). The shell
         lives in the installable data package so pip installs and dev runs
         share one copy.
      2. The installed `imagespace_viewer_shell` package — when installed via
         `pip install .` (the shell ships as package data).
    Returns a Path to the shell directory, or None if not found.
    """
    # 1. Repo layout: scripts/../imagespace_viewer_shell/viewer_shell
    local = Path(__file__).resolve().parent.parent / "imagespace_viewer_shell" / "viewer_shell"
    if (local / "index.html").is_file():
        return local
    # 2. Installed package data.
    try:
        import importlib.resources as ir
        try:
            ref = ir.files("imagespace_viewer_shell")
            root = Path(str(ref)) if hasattr(ref, "__fspath__") else Path(ref.anchor)
        except Exception:
            return None
        cand = root / "viewer_shell"
        if (cand / "index.html").is_file():
            return cand
    except Exception:
        pass
    return None


# ── Main Pipeline ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ImageSpace — Transform images into an interactive visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_pos",
        nargs="?",
        help="Directory containing images (positional shortcut; see also -i/--input)",
    )
    parser.add_argument(
        "-i", "--input", dest="input",
        help="Directory containing images",
    )
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument(
        "--min-cluster-size", type=int, default=None,
        help="HDBSCAN min_cluster_size (default: about 1%% of the collection, "
             "rounded to the nearest 10, minimum 10)",
    )
    parser.add_argument(
        "--hdbscan-min-samples", type=int, default=5,
        help="HDBSCAN min_samples (default: 5)",
    )
    parser.add_argument(
        "--hdbscan-selection-method", choices=("leaf", "eom"), default="leaf",
        help="HDBSCAN cluster selection method (default: leaf)",
    )
    parser.add_argument(
        "--umap-neighbors", type=int, default=UMAP_LAYOUT_NEIGHBORS,
        help=f"UMAP n_neighbors for the 2-D layout (default: {UMAP_LAYOUT_NEIGHBORS})",
    )
    parser.add_argument(
        "--umap-min-dist", type=float, default=UMAP_LAYOUT_MIN_DIST,
        help=f"UMAP min_dist for the 2-D layout (default: {UMAP_LAYOUT_MIN_DIST})",
    )
    parser.add_argument(
        "--cluster-neighbors", type=int, default=UMAP_CLUSTER_NEIGHBORS,
        help=f"UMAP n_neighbors for the clustering space (default: {UMAP_CLUSTER_NEIGHBORS})",
    )
    parser.add_argument(
        "--cluster-dims", type=int, default=UMAP_CLUSTER_DIMS,
        help=f"UMAP dimensions of the space HDBSCAN clusters in (default: {UMAP_CLUSTER_DIMS})",
    )
    parser.add_argument(
        "--clip-model", default=DEFAULT_CLIP_MODEL,
        help=f"Hugging Face CLIP model id (default: {DEFAULT_CLIP_MODEL})",
    )
    parser.add_argument(
        "--label-candidates",
        help="Versioned JSON label vocabulary; defaults to bundled art-v1 preset",
    )
    parser.add_argument(
        "--label-uncertainty-threshold", type=float,
        default=DEFAULT_LABEL_UNCERTAINTY_THRESHOLD,
        help="Minimum top-1/top-2 score margin for a confident cluster label (default: 0.01)",
    )
    parser.add_argument(
        "--thumb-size", type=int, default=128, help="Thumbnail size in pixels (128)"
    )
    parser.add_argument(
        "--atlas-size",
        type=int,
        default=ATLAS_SIZE,
        help="Atlas texture size (default 4096)",
    )
    parser.add_argument(
        "--quality", type=int, default=60, help="WebP quality 1-100 (default 60)"
    )
    parser.add_argument("--metadata", help="External metadata CSV to merge")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for UMAP reproducibility. Default: none "
             "(fully random each run). Pass a fixed seed (e.g. --seed 42) to "
             "make the layout reproducible.",
    )
    parser.add_argument(
        "--cache-dir",
        help="Directory to cache embeddings (skip re-extraction if cached)",
    )
    parser.add_argument(
        "--relayout",
        action="store_true",
        help="Skip atlas generation, only re-run UMAP/HDBSCAN (reuses --cache-dir embeddings if given)",
    )
    parser.add_argument(
        "--hd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate dual-resolution atlases: 64px previews load first, then 128px HD "
             "swaps in on desktop/tablet (phones stay on previews). ON by default; "
             "pass --no-hd for single-resolution output (thumb-size 64).",
    )
    parser.add_argument(
        "--preview-quality",
        type=int,
        default=40,
        help="WebP quality for preview atlases (default 40)",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Emit only the data/ folder (skip the viewer app shell). "
             "Useful for relayout/seed-sweep workflows. Default emits a full site.",
    )
    args = parser.parse_args()
    if args.umap_neighbors < 2 or args.cluster_neighbors < 2:
        parser.error("--umap-neighbors and --cluster-neighbors must be at least 2")
    if not 0 <= args.umap_min_dist <= 1:
        parser.error("--umap-min-dist must be between 0 and 1")
    if args.cluster_dims < 1:
        parser.error("--cluster-dims must be at least 1")
    if args.hdbscan_min_samples < 1:
        parser.error("--hdbscan-min-samples must be at least 1")
    if args.min_cluster_size is not None and args.min_cluster_size < 2:
        parser.error("--min-cluster-size must be at least 2")
    if args.label_uncertainty_threshold < 0:
        parser.error("--label-uncertainty-threshold cannot be negative")
    # Allow `imagespace -i imgs -o out` or `imagespace imgs -o out` (positional).
    input_path = args.input if args.input else args.input_pos
    if not input_path:
        parser.error("an input directory is required (use -i/--input, or pass it as a "
                     "positional argument)")
    input_dir = Path(input_path).resolve()
    output_dir = Path(args.output).resolve()

    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        sys.exit(1)

    # --hd requires 128px thumbnails; --no-hd keeps the caller's thumb size.
    if args.hd:
        args.thumb_size = 128

    os.makedirs(output_dir, exist_ok=True)
    # Pipeline-generated viewer data lives in <out>/data/ so the output dir is a
    # self-contained static site (index.html + assets/ + data/).
    data_dir = output_dir / "data"
    os.makedirs(data_dir, exist_ok=True)
    # Embeddings cache: only written when --cache-dir is given (power-user
    # seed-sweep workflow). Default runs never write embeddings to the output.
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    total_start = time.time()

    print(f"\n{'=' * 60}")
    print(
        f"  ImageSpace Pipeline {'(Relayout Mode)' if args.relayout else '(Fast Mode)'}"
    )
    print(f"{'=' * 60}")

    # Stage 1
    print(f"\n[1/8] Discovering images...")
    images = discover_images(input_dir)
    if not images:
        print("  No images found!")
        sys.exit(1)
    print(f"  Found {len(images)} images")
    input_fingerprint = corpus_fingerprint(images, input_dir)

    # Stage 2
    if args.relayout:
        print(f"\n[2/8] Skipping atlas generation (relayout mode)")
        bin_path = os.path.join(str(data_dir), "data.bin")
        if os.path.exists(bin_path):
            manifest_path = os.path.join(str(data_dir), "manifest.json")
            stride = 24
            preview_atlas_data = None
            preview_atlas_count = None
            try:
                with open(manifest_path) as f:
                    stride = json.load(f).get('bytesPerImage', 24)
            except:
                pass
            raw = open(bin_path, "rb").read()
            inferred = len(raw) // len(images)
            if inferred != stride and len(raw) % len(images) == 0:
                stride = inferred
            atlas_data = []
            for i in range(len(images)):
                off = i * stride
                ai = struct.unpack_from("<H", raw, off + 16)[0]
                u = struct.unpack_from("<H", raw, off + 18)[0]
                v = struct.unpack_from("<H", raw, off + 20)[0]
                atlas_data.append((ai, u, v))
            atlas_count = max(a[0] for a in atlas_data) + 1
            if stride >= 28:
                preview_data = []
                for i in range(len(images)):
                    off = i * stride
                    up = struct.unpack_from("<H", raw, off + 24)[0]
                    vp = struct.unpack_from("<H", raw, off + 26)[0]
                    preview_data.append((atlas_data[i][0], up, vp))
                preview_atlas_data = preview_data
                preview_atlas_count = atlas_count
            print(
                f"  Loaded atlas data for {len(atlas_data)} images from existing data.bin"
                f" ({stride} bytes/img{' w/ preview' if preview_atlas_data else ''})"
            )
        else:
            print("  Error: data.bin not found for relayout mode!")
            sys.exit(1)
    else:
        print(f"\n[2/8] Generating WebP atlas textures (quality={args.quality})...")
        atlas_data, atlas_count = generate_atlases(
            images, str(data_dir), args.thumb_size, args.atlas_size, args.quality
        )
        preview_atlas_data = None
        preview_atlas_count = None
        if args.hd:
            print(
                f"\n[2b/8] Generating preview atlases (64px, quality={args.preview_quality})..."
            )
            images_per_full_atlas = (args.atlas_size // args.thumb_size) ** 2
            preview_atlas_data, preview_atlas_count = generate_preview_atlases(
                images,
                str(data_dir),
                atlas_count,
                images_per_full_atlas,
                thumb_size=64,
                atlas_size=args.atlas_size,
                quality=args.preview_quality,
            )

    # Stage 3 — with caching. The embeddings cache is opt-in via --cache-dir;
    # default runs never persist embeddings to disk (keeps output lean).
    if cache_dir is not None:
        emb_cache = os.path.join(str(cache_dir), "embeddings.npy")
        emb_meta_path = os.path.join(str(cache_dir), "embeddings.json")
    else:
        emb_cache = None
        emb_meta_path = None
    if emb_cache and os.path.exists(emb_cache):
        embedding_meta = {}
        if os.path.exists(emb_meta_path):
            with open(emb_meta_path, encoding="utf-8") as f:
                embedding_meta = json.load(f)
        cached_model = embedding_meta.get("modelId", DEFAULT_CLIP_MODEL)
        cached_fingerprint = embedding_meta.get("corpusFingerprint")
        if cached_model != args.clip_model:
            raise RuntimeError(
                f"Embedding cache uses {cached_model}, but --clip-model requests "
                f"{args.clip_model}. Use a different --cache-dir."
            )
        if cached_fingerprint and cached_fingerprint != input_fingerprint:
            raise RuntimeError(
                "Embedding cache belongs to a different input corpus. "
                "Use a separate --cache-dir or remove the stale cache."
            )
        print(f"\n[3/8] Loading cached embeddings from {emb_cache}...")
        embeddings = np.load(emb_cache)
        if embeddings.shape[0] != len(images):
            raise RuntimeError(
                f"Embedding cache has {embeddings.shape[0]} rows for {len(images)} images. "
                "Use a separate --cache-dir or remove the stale cache."
            )
        embedding_backend = embedding_meta.get("backend", "unknown-cached")
        print(f"  Loaded {embeddings.shape[0]} × {embeddings.shape[1]} embeddings")
    else:
        print(f"\n[3/8] Extracting embeddings (auto-detecting hardware)...")
        embeddings, embedding_backend = extract_embeddings(images, args.clip_model)
        if emb_cache:
            np.save(emb_cache, embeddings)
            with open(emb_meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "modelId": args.clip_model,
                        "backend": embedding_backend,
                        "count": len(images),
                        "dimensions": int(embeddings.shape[1]),
                        "corpusFingerprint": input_fingerprint,
                    },
                    f,
                    indent=2,
                )
                f.write("\n")
            print(f"  Cached embeddings to {emb_cache}")
    
    # Stage 4
    print(f"\n[4/9] UMAP (layout + clustering space) → HDBSCAN...")
    mcs = args.min_cluster_size
    if mcs is None:
        mcs = auto_min_cluster_size(len(images))
        print(f"  Auto min_cluster_size={mcs} for {len(images)} images")
    (
        tsne_coords,
        raw_tsne_coords,
        cluster_ids,
        _cluster_space,
        cluster_probs,
        analysis,
    ) = reduce_dimensions(
        embeddings,
        mcs,
        layout_neighbors=args.umap_neighbors,
        thumb_size=args.thumb_size,
        seed=args.seed,
        cluster_dims=args.cluster_dims,
        hdbscan_min_samples=args.hdbscan_min_samples,
        hdbscan_selection_method=args.hdbscan_selection_method,
        layout_min_dist=args.umap_min_dist,
        cluster_neighbors=args.cluster_neighbors,
    )

    # Stage 4b: k-NN on the full CLIP embeddings (cosine), independent of UMAP
    print(f"\n[5/9] k-Nearest Neighbors...")
    knn_indices, knn_distances = compute_knn(embeddings, k=10)
    analysis["neighbors"] = {
        "k": int(knn_indices.shape[1]),
        "space": "clip-embedding",
        "dimensions": int(embeddings.shape[1]),
        "metric": "cosine",
    }
    write_neighbors_bin(str(data_dir), knn_indices, knn_distances)
    outlier_scores = compute_outlier_scores(knn_distances)

    # Stage 4c: CLIP cluster labels
    print(f"\n[6/9] CLIP cluster labels...")
    label_vocabulary = load_label_vocabulary(args.label_candidates)
    cluster_labels = generate_cluster_labels(
        embeddings,
        cluster_ids,
        model_id=args.clip_model,
        vocabulary=label_vocabulary,
        uncertainty_threshold=args.label_uncertainty_threshold,
    )
    cluster_labels_path = data_dir / "cluster_labels.json"
    if cluster_labels:
        write_cluster_labels(str(data_dir), cluster_labels)
    elif cluster_labels_path.exists():
        cluster_labels_path.unlink()

    # Stage 5
    meta_csv_path = os.path.join(str(data_dir), "metadata.csv")
    skip_metadata = (
        args.relayout and os.path.exists(meta_csv_path) and not args.metadata
    )
    if skip_metadata:
        print(f"\n[7/9] Skipping metadata (relayout mode, no external metadata)")
        timestamps = None
        colors = None
        external_metadata = None
        image_features = None
    else:
        print(f"\n[7/9] Extracting metadata...")
        timestamps = extract_timestamps(images)
        colors = extract_dominant_colors(images)
        print(f"  Timestamps: {sum(1 for t in timestamps if t > 0)}/{len(images)}")

        print(
            f"\n[8/9] Computing image features (brightness, complexity, edge density)..."
        )
        image_features = compute_image_features(images)

        external_metadata = None
        if args.metadata:
            meta_src = Path(args.metadata).resolve()
            if meta_src.exists():
                external_metadata = read_external_metadata(str(meta_src))
            else:
                print(f"  WARNING: External metadata not found: {meta_src}")

    # Stage 6
    print(f"\n[9/9] Writing output files...")
    write_binary_data(
        str(data_dir),
        tsne_coords,
        raw_tsne_coords,
        atlas_data,
        cluster_ids,
        preview_atlas_data,
    )
    analysis_config = {
        "schemaVersion": 1,
        "software": {"name": "ImageSpace", "version": SOFTWARE_VERSION},
        "corpus": {
            "imageCount": len(images),
            "fingerprint": input_fingerprint,
        },
        "embedding": {
            "modelId": args.clip_model,
            "backend": embedding_backend,
            "dimensions": int(embeddings.shape[1]),
        },
        **analysis,
        "labels": {
            "modelId": args.clip_model,
            "vocabularyId": label_vocabulary["id"],
            "vocabularyVersion": label_vocabulary["version"],
            "candidateCount": len(label_vocabulary["candidates"]),
            "uncertaintyThreshold": args.label_uncertainty_threshold,
            "topCandidatesStored": 3,
        },
    }
    write_analysis_config(str(data_dir), analysis_config)
    write_manifest(
        str(data_dir),
        len(images),
        atlas_count,
        args.thumb_size,
        args.atlas_size,
        preview_atlas_count=preview_atlas_count,
        preview_thumb_size=64,
        provenance=analysis_config,
    )
    # Normalize cluster confidence to 0-100 scale
    cluster_confidence = (
        (cluster_probs * 100).round(1) if cluster_probs is not None else None
    )
    if timestamps is not None:
        write_metadata_csv(
            str(data_dir),
            images,
            cluster_ids,
            timestamps,
            colors,
            external_metadata,
            image_features,
            outlier_scores,
            cluster_confidence,
        )
    elif skip_metadata:
        refresh_metadata_clusters(
            meta_csv_path,
            cluster_ids,
            cluster_confidence,
        )

    # Stage 7 — copy the bundled viewer app shell so the output dir is a
    # self-contained, uploadable static site (index.html + assets/ + data/).
    # Skip with --data-only (relayout/seed-sweep workflows).
    if not args.data_only:
        shell_src = _find_viewer_shell()
        if shell_src is not None:
            for name in ("index.html", "favicon.svg", ".nojekyll"):
                src_file = shell_src / name
                if src_file.exists():
                    shutil.copy2(src_file, output_dir / name)
            assets_src = shell_src / "assets"
            if assets_src.is_dir():
                shutil.copytree(assets_src, output_dir / "assets", dirs_exist_ok=True)
            print(f"  Copied viewer app shell into {output_dir}")
        else:
            print(
                f"  WARNING: viewer shell not found. Run scripts/build_viewer_shell.sh "
                f"to bundle the viewer, or pip install with the shell package. "
                f"Output contains data/ only."
            )

    elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  ✓ {len(images)} images → {output_dir}")
    print(f"  ✓ {atlas_count} atlas textures (WebP)")
    print(f"  ✓ Total time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

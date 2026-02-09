#!/usr/bin/env python3
"""
ImageSpace — Fast Pipeline for Exploratory Visualization of Image Collections

Transforms a folder of images into static data files for the ImageSpace viewer:
  - Atlas WebP textures (sprite sheets of thumbnails)
  - Binary layout data (t-SNE coordinates, atlas positions, cluster IDs)
  - Manifest JSON (metadata for the viewer)
  - Metadata CSV (merged with external metadata if provided)

Usage:
    python imagespace.py /path/to/images --output ./dist/data
    python imagespace.py /path/to/images --output ./dist/data --gpu
    python imagespace.py /path/to/images --output ./dist/data --metadata existing.csv

Performance (50K images, Apple Silicon):
    - Atlas generation: ~2-3 min
    - CLIP embeddings (ONNX+CoreML): ~2-3 min
    - openTSNE: ~30-60s
    - HDBSCAN: ~10-30s
    - Total: ~5-8 min

Dependencies:
    Required: pillow, numpy, scikit-learn, opentsne, hdbscan
    Optional: onnxruntime (for CLIP embeddings via ONNX — highly recommended)
              torch, transformers (alternative CLIP backend — slower)
              huggingface_hub (for downloading ONNX model)
"""

import argparse
import colorsys
import csv
import json
import math
import os
import struct
import sys
import time
from pathlib import Path
from multiprocessing import cpu_count

import numpy as np
from PIL import Image, ExifTags

# ── Configuration ─────────────────────────────────────────────
THUMB_SIZE = 64          # Thumbnail size in pixels (square)
ATLAS_SIZE = 4096        # Atlas texture size (4096x4096 = 64x64 grid of 64px thumbs)
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
CLIP_IMAGE_SIZE = 224    # CLIP input resolution
CLIP_DIM = 512           # CLIP ViT-B/32 embedding dimension
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
NUM_CLUSTERS = 15        # Fallback KMeans clusters
TSNE_PERPLEXITY = 30     # openTSNE perplexity
BATCH_SIZE = 64          # Embedding batch size (larger = faster with ONNX)
PCA_DIMS = 50            # PCA reduction before t-SNE


# ── Stage 1: Image Discovery ─────────────────────────────────
def discover_images(input_dir):
    """Recursively find all supported image files, skipping hidden/system dirs."""
    images = []
    input_path = Path(input_dir).resolve()
    for root, dirs, files in os.walk(input_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', 'node_modules', '.git')]
        for f in sorted(files):
            if f.startswith('.'):
                continue
            ext = Path(f).suffix.lower()
            if ext in SUPPORTED_FORMATS:
                images.append(Path(root) / f)
    return images


# ── Stage 2: Thumbnail Generation + Atlas Packing ────────────
def generate_atlases(images, output_dir, thumb_size=THUMB_SIZE, atlas_size=ATLAS_SIZE, quality=80):
    """Create WebP atlas textures from image thumbnails. Returns atlas metadata per image."""
    images_per_row = atlas_size // thumb_size
    images_per_atlas = images_per_row * images_per_row

    atlas_data = []  # (atlas_idx, u, v) per image
    current_atlas_idx = 0
    current_img_in_atlas = 0
    atlas_img = Image.new('RGB', (atlas_size, atlas_size), (255, 255, 255))

    start = time.time()
    for idx, img_path in enumerate(images):
        if idx % 2000 == 0:
            elapsed = time.time() - start
            rate = idx / elapsed if elapsed > 0 else 0
            eta = (len(images) - idx) / rate if rate > 0 else 0
            print(f"  Thumbnailing {idx}/{len(images)} ({rate:.0f} img/s, ETA {eta:.0f}s)", end='\r')

        try:
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            side = min(w, h)
            left = (w - side) // 2
            top = (h - side) // 2
            img = img.crop((left, top, left + side, top + side))
            img = img.resize((thumb_size, thumb_size), Image.BILINEAR)
        except Exception as e:
            img = Image.new('RGB', (thumb_size, thumb_size), (128, 128, 128))

        col = current_img_in_atlas % images_per_row
        row = current_img_in_atlas // images_per_row
        u, v = col * thumb_size, row * thumb_size
        atlas_img.paste(img, (u, v))
        atlas_data.append((current_atlas_idx, u, v))

        current_img_in_atlas += 1
        if current_img_in_atlas >= images_per_atlas or idx == len(images) - 1:
            atlas_path = os.path.join(output_dir, f'atlas_{current_atlas_idx}.webp')
            atlas_img.save(atlas_path, 'WEBP', quality=quality, method=2)
            print(f"\n  Saved atlas_{current_atlas_idx}.webp ({current_img_in_atlas} images)")
            current_atlas_idx += 1
            current_img_in_atlas = 0
            atlas_img = Image.new('RGB', (atlas_size, atlas_size), (255, 255, 255))

    total = time.time() - start
    print(f"  Atlas generation: {total:.1f}s")
    return atlas_data, current_atlas_idx


# ── Stage 3: Embedding Extraction ────────────────────────────
def extract_embeddings(images, use_gpu=False):
    """Extract CLIP ViT-B/32 embeddings. Tries ONNX first (fastest), then PyTorch, then histograms."""
    # Try ONNX Runtime (fastest)
    try:
        return _extract_clip_onnx(images, use_gpu)
    except Exception as e:
        print(f"  ONNX CLIP failed: {e}")

    # Try PyTorch (slower but more compatible)
    try:
        return _extract_clip_torch(images, use_gpu)
    except (ImportError, Exception) as e:
        print(f"  PyTorch CLIP failed: {e}")

    # Fallback: color histograms
    print("  Falling back to color histogram embeddings...")
    return _extract_color_histograms(images)


def _get_onnx_model_path():
    """Download or locate CLIP ONNX vision model."""
    cache_dir = Path.home() / '.cache' / 'imagespace'
    model_path = cache_dir / 'clip-vit-b32-visual.onnx'
    if model_path.exists():
        return str(model_path)

    try:
        from huggingface_hub import hf_hub_download
        print("  Downloading CLIP ONNX vision model (first time only)...")
        downloaded = hf_hub_download(
            repo_id="Xenova/clip-vit-base-patch32",
            filename="onnx/vision_model.onnx",
            cache_dir=str(cache_dir),
        )
        return downloaded
    except Exception as e:
        print(f"  Could not download ONNX model: {e}")
        return None


def _preprocess_clip_batch(images_pil):
    """Preprocess a batch of PIL images for CLIP (resize, center-crop, normalize)."""
    batch = np.zeros((len(images_pil), 3, CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE), dtype=np.float32)
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


def _extract_clip_onnx(images, use_gpu=False):
    """CLIP embeddings via ONNX Runtime (fastest path)."""
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

    providers = ['CPUExecutionProvider']
    if use_gpu:
        available = ort.get_available_providers()
        if 'CoreMLExecutionProvider' in available:
            providers.insert(0, 'CoreMLExecutionProvider')
            print("  Using Apple Neural Engine (CoreML)")
        elif 'CUDAExecutionProvider' in available:
            providers.insert(0, 'CUDAExecutionProvider')
            print("  Using NVIDIA CUDA")
        else:
            print("  No GPU provider found, using CPU")

    print(f"  Loading CLIP ONNX model...")
    session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)

    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    embeddings = np.zeros((len(images), CLIP_DIM), dtype=np.float32)
    start = time.time()

    for batch_start in range(0, len(images), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(images))
        batch_images = []
        for img_path in images[batch_start:batch_end]:
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception:
                img = Image.new('RGB', (CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE), (128, 128, 128))
            batch_images.append(img)

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
            embeddings[batch_start:batch_end, :out_normalized.shape[1]] = out_normalized

        elapsed = time.time() - start
        rate = batch_end / elapsed if elapsed > 0 else 0
        eta = (len(images) - batch_end) / rate if rate > 0 else 0
        print(f"  Embedding {batch_end}/{len(images)} ({rate:.1f} img/s, ETA {eta:.0f}s)", end='\r')

    print(f"\n  CLIP ONNX embeddings: {time.time() - start:.1f}s")
    return embeddings


def _extract_clip_torch(images, use_gpu=False):
    """CLIP embeddings via transformers + PyTorch (fallback)."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    print("  Loading CLIP ViT-B/32 (PyTorch)...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    device = 'cpu'
    if use_gpu:
        if torch.cuda.is_available():
            device = 'cuda'
            print("  Using CUDA GPU")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
            print("  Using Apple Silicon GPU (MPS)")

    model = model.to(device).eval()
    embeddings = np.zeros((len(images), CLIP_DIM), dtype=np.float32)
    start = time.time()

    for batch_start in range(0, len(images), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(images))
        batch_images = []
        for img_path in images[batch_start:batch_end]:
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception:
                img = Image.new('RGB', (CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE), (128, 128, 128))
            batch_images.append(img)

        inputs = processor(images=batch_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            if hasattr(outputs, 'pooler_output'):
                outputs = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                outputs = outputs.last_hidden_state[:, 0]
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)
            embeddings[batch_start:batch_end] = outputs.cpu().numpy()

        elapsed = time.time() - start
        rate = batch_end / elapsed if elapsed > 0 else 0
        eta = (len(images) - batch_end) / rate if rate > 0 else 0
        print(f"  Embedding {batch_end}/{len(images)} ({rate:.1f} img/s, ETA {eta:.0f}s)", end='\r')

    print(f"\n  CLIP PyTorch embeddings: {time.time() - start:.1f}s")
    return embeddings


def _extract_color_histograms(images, dim=CLIP_DIM):
    """Fallback: color histogram + spatial grid embeddings."""
    print("  Extracting color histograms...")
    embeddings = np.zeros((len(images), dim), dtype=np.float32)
    start = time.time()

    for idx, img_path in enumerate(images):
        if idx % 1000 == 0:
            print(f"  Histogram {idx}/{len(images)}", end='\r')
        try:
            img = Image.open(img_path).convert('RGB').resize((64, 64))
            arr = np.array(img, dtype=np.float32) / 255.0
            features = []
            for c in range(3):
                h, _ = np.histogram(arr[:, :, c], bins=64, range=(0, 1))
                features.append(h.astype(np.float32))
            # 3x3 spatial color grid (trim to divisible-by-3)
            bh, bw = arr.shape[0] // 3, arr.shape[1] // 3
            grid = arr[:bh*3, :bw*3].reshape(3, bh, 3, bw, 3).mean(axis=(1, 3))
            features.append(grid.flatten().astype(np.float32))
            vec = np.concatenate(features)
            if len(vec) < dim:
                vec = np.pad(vec, (0, dim - len(vec)))
            else:
                vec = vec[:dim]
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            embeddings[idx] = vec
        except Exception:
            pass

    print(f"\n  Color histograms: {time.time() - start:.1f}s")
    return embeddings


# ── Stage 4: Dimensionality Reduction + Clustering ───────────
def reduce_dimensions(embeddings, min_cluster_size=50, perplexity=TSNE_PERPLEXITY):
    """Run PCA → openTSNE → HDBSCAN. Returns (tsne_coords, cluster_ids)."""
    from sklearn.decomposition import PCA

    n = len(embeddings)

    # PCA first: reduce 512-d to 50-d for much faster t-SNE
    pca_dims = min(PCA_DIMS, n - 1, embeddings.shape[1])
    print(f"\n  PCA: {embeddings.shape[1]}-d → {pca_dims}-d...")
    start = time.time()
    pca = PCA(n_components=pca_dims, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    print(f"  PCA: {time.time() - start:.1f}s ({pca.explained_variance_ratio_.sum():.1%} variance)")

    # openTSNE (FFT-accelerated, ~10-20x faster than sklearn)
    print(f"\n  Running openTSNE (n={n}, perplexity={perplexity})...")
    start = time.time()
    try:
        from openTSNE import TSNE
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, n // 3),
            exaggeration=4,
            initialization='pca',
            metric='euclidean',
            neighbors='approx',
            n_jobs=-1,
            random_state=42,
            verbose=True,
        )
        tsne_coords = np.array(tsne.fit(embeddings_pca))
    except ImportError:
        print("  openTSNE not found, falling back to sklearn (much slower)...")
        from sklearn.manifold import TSNE as skTSNE
        tsne = skTSNE(
            n_components=2,
            perplexity=min(perplexity, n - 1),
            random_state=42,
            init='pca' if n > 50 else 'random',
            learning_rate='auto',
        )
        tsne_coords = tsne.fit_transform(embeddings_pca)
    print(f"  t-SNE completed in {time.time() - start:.1f}s")

    # Scale to viewer range — ensure enough room for non-overlapping thumbnails
    n = len(tsne_coords)
    # Each image needs cell_size² area; scale so total area fits comfortably
    cell_size = THUMB_SIZE * 1.15  # slight gap between thumbnails
    target_side = int(np.ceil(np.sqrt(n * 1.8))) * cell_size  # 1.8x overallocation for structure
    def scale_coords(coords, target_range):
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1
        return (coords - mins) / ranges * target_range - target_range / 2

    tsne_coords = scale_coords(tsne_coords, target_side)

    # Remove overlaps by snapping to nearest unoccupied grid cell
    print(f"  Removing overlaps (cell={cell_size:.0f}px, grid≈{int(target_side/cell_size)}²)...")
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
                dy = -r
                if (gx + dx, gy + dy) not in occupied:
                    occupied.add((gx + dx, gy + dy))
                    result[idx] = [(gx + dx) * cell_size, (gy + dy) * cell_size]
                    placed = True; break
                dy = r
                if (gx + dx, gy + dy) not in occupied:
                    occupied.add((gx + dx, gy + dy))
                    result[idx] = [(gx + dx) * cell_size, (gy + dy) * cell_size]
                    placed = True; break
            if placed: break
            for dy in range(-r + 1, r):
                dx = -r
                if (gx + dx, gy + dy) not in occupied:
                    occupied.add((gx + dx, gy + dy))
                    result[idx] = [(gx + dx) * cell_size, (gy + dy) * cell_size]
                    placed = True; break
                dx = r
                if (gx + dx, gy + dy) not in occupied:
                    occupied.add((gx + dx, gy + dy))
                    result[idx] = [(gx + dx) * cell_size, (gy + dy) * cell_size]
                    placed = True; break
            if placed: break
    tsne_coords = result
    print(f"  Overlap removal: {time.time() - start:.1f}s")

    # Clustering: HDBSCAN → MiniBatchKMeans fallback
    try:
        import hdbscan as hdb
        print(f"\n  Running HDBSCAN (min_cluster_size={min_cluster_size})...")
        start = time.time()
        clusterer = hdb.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=10,
            metric='euclidean',
            cluster_selection_method='eom',
            core_dist_n_jobs=-1,
        )
        cluster_ids = clusterer.fit_predict(tsne_coords)
        n_clusters = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
        n_noise = (cluster_ids == -1).sum()

        if n_noise > 0 and n_clusters > 0:
            from scipy.spatial import cKDTree
            valid_mask = cluster_ids >= 0
            tree = cKDTree(tsne_coords[valid_mask])
            valid_labels = cluster_ids[valid_mask]
            _, nearest = tree.query(tsne_coords[cluster_ids == -1])
            cluster_ids[cluster_ids == -1] = valid_labels[nearest]
        print(f"  HDBSCAN: {n_clusters} clusters, {n_noise} noise reassigned ({time.time() - start:.1f}s)")
    except ImportError:
        from sklearn.cluster import MiniBatchKMeans
        n_clusters = min(NUM_CLUSTERS, n)
        print(f"\n  Using MiniBatchKMeans (k={n_clusters})...")
        start = time.time()
        cluster_ids = MiniBatchKMeans(
            n_clusters=n_clusters, batch_size=max(1024, n // 10),
            random_state=42, n_init=3
        ).fit_predict(tsne_coords)
        print(f"  MiniBatchKMeans: {time.time() - start:.1f}s")

    return tsne_coords.astype(np.float32), cluster_ids.astype(np.int32)


# ── Stage 5: Dominant Color Extraction ────────────────────────
def extract_dominant_colors(images, thumb_size=32):
    """Extract dominant color (hue, sat, lum) for each image."""
    colors = []
    start = time.time()
    for idx, img_path in enumerate(images):
        try:
            img = Image.open(img_path).convert('RGB').resize((thumb_size, thumb_size), Image.BILINEAR)
            arr = np.array(img).reshape(-1, 3).mean(axis=0) / 255.0
            h, l, s = colorsys.rgb_to_hls(arr[0], arr[1], arr[2])
            colors.append((h, s, l))
        except Exception:
            colors.append((0, 0, 0))
    print(f"  Colors extracted in {time.time() - start:.1f}s")
    return colors


# ── Stage 6: Extract Timestamps ──────────────────────────────
def extract_timestamps(images):
    """Extract timestamps from EXIF or filename year patterns."""
    import re
    timestamps = []
    year_pattern = re.compile(r'(\d{4})')

    for img_path in images:
        ts = _get_exif_timestamp(img_path)
        if ts is None:
            match = year_pattern.search(img_path.stem)
            if match:
                year = int(match.group(1))
                if 1400 <= year <= 2030:
                    from datetime import datetime
                    ts = int(datetime(year, 6, 15).timestamp())
        timestamps.append(ts if ts is not None else 0)
    return timestamps


def _get_exif_timestamp(img_path):
    """Extract POSIX timestamp from EXIF DateTimeOriginal."""
    try:
        img = Image.open(img_path)
        exif_data = img._getexif()
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                if tag == 'DateTimeOriginal':
                    from datetime import datetime
                    dt = datetime.strptime(value, '%Y:%m:%d %H:%M:%S')
                    return int(dt.timestamp())
    except Exception:
        pass
    return None


# ── Stage 7: Output Generation ────────────────────────────────
def write_binary_data(output_dir, tsne_coords, atlas_data, cluster_ids):
    """Write binary layout data. 24 bytes per image."""
    binary_data = bytearray(len(tsne_coords) * 24)
    for i in range(len(tsne_coords)):
        ai, u, v = atlas_data[i]
        cid = int(cluster_ids[i])
        tx, ty = float(tsne_coords[i][0]), float(tsne_coords[i][1])
        struct.pack_into('<ffffHHHH', binary_data, i * 24,
            tx, ty, tx, ty, ai, u, v, cid)

    output_path = os.path.join(output_dir, 'data.bin')
    with open(output_path, 'wb') as f:
        f.write(binary_data)
    print(f"  Binary data: {len(binary_data) / 1024:.1f} KB")
    return output_path


def write_manifest(output_dir, count, atlas_count, thumb_size=THUMB_SIZE, atlas_size=ATLAS_SIZE):
    """Write manifest.json for the viewer."""
    manifest = {
        'count': count,
        'atlasCount': atlas_count,
        'thumbSize': thumb_size,
        'atlasSize': atlas_size,
        'bytesPerImage': 24,
        'version': 2,
        'atlasFormat': 'webp',
    }
    output_path = os.path.join(output_dir, 'manifest.json')
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest: {output_path}")


def write_metadata_csv(output_dir, images, cluster_ids, timestamps, colors, external_metadata=None):
    """Write metadata.csv with image info, merging with external metadata if provided."""
    output_path = os.path.join(output_dir, 'metadata.csv')

    def hue_to_name(h):
        names = [
            (0.0, 'red'), (0.05, 'orange'), (0.12, 'yellow'), (0.2, 'yellow-green'),
            (0.33, 'green'), (0.45, 'teal'), (0.5, 'cyan'), (0.58, 'blue'),
            (0.7, 'indigo'), (0.8, 'purple'), (0.9, 'magenta'), (1.0, 'red'),
        ]
        for threshold, name in names:
            if h <= threshold:
                return name
        return 'red'

    extra_cols = []
    if external_metadata:
        for fname, row in external_metadata.items():
            extra_cols = [c for c in row.keys() if c.lower() != 'filename']
            break

    base_cols = ['id', 'filename', 'cluster', 'timestamp', 'dominant_color']
    all_cols = base_cols + extra_cols

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(all_cols)
        matched = 0
        for i, img_path in enumerate(images):
            h, s, l = colors[i] if i < len(colors) else (0, 0, 0)
            color_name = hue_to_name(h) if s > 0.1 else 'gray'
            row = [i, img_path.name, int(cluster_ids[i]),
                   timestamps[i] if timestamps[i] > 0 else '', color_name]
            if external_metadata:
                ext = external_metadata.get(img_path.name, {})
                if ext: matched += 1
                for col in extra_cols:
                    row.append(ext.get(col, ''))
            writer.writerow(row)

    if external_metadata:
        print(f"  Metadata: merged {matched}/{len(images)} rows")
    else:
        print(f"  Metadata: {output_path}")


def read_external_metadata(metadata_path):
    """Read external metadata CSV as dict: filename -> {col: val}."""
    lookup = {}
    with open(metadata_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get('filename', '')
            if fname:
                lookup[fname] = {k: v for k, v in row.items() if k != 'filename'}
    print(f"  Read {len(lookup)} entries from external metadata")
    return lookup


# ── Main Pipeline ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='ImageSpace — Transform images into an interactive visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('input', help='Directory containing images')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--gpu', action='store_true', help='Enable GPU acceleration')
    parser.add_argument('--min-cluster-size', type=int, default=50, help='HDBSCAN min_cluster_size')
    parser.add_argument('--thumb-size', type=int, default=THUMB_SIZE, help='Thumbnail size in pixels')
    parser.add_argument('--atlas-size', type=int, default=ATLAS_SIZE, help='Atlas texture size (default 4096)')
    parser.add_argument('--quality', type=int, default=80, help='WebP quality 1-100 (default 80)')
    parser.add_argument('--metadata', help='External metadata CSV to merge')
    parser.add_argument('--tsne-perplexity', type=int, default=TSNE_PERPLEXITY, help='t-SNE perplexity')

    args = parser.parse_args()
    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory"); sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    total_start = time.time()

    print(f"\n{'='*60}")
    print(f"  ImageSpace Pipeline (Fast Mode)")
    print(f"{'='*60}")

    # Stage 1
    print(f"\n[1/6] Discovering images...")
    images = discover_images(input_dir)
    if not images:
        print("  No images found!"); sys.exit(1)
    print(f"  Found {len(images)} images")

    # Stage 2
    print(f"\n[2/6] Generating WebP atlas textures (quality={args.quality})...")
    atlas_data, atlas_count = generate_atlases(images, str(output_dir), args.thumb_size, args.atlas_size, args.quality)

    # Stage 3
    print(f"\n[3/6] Extracting embeddings {'(GPU)' if args.gpu else '(CPU)'}...")
    embeddings = extract_embeddings(images, use_gpu=args.gpu)

    # Stage 4
    print(f"\n[4/6] PCA → openTSNE → HDBSCAN...")
    tsne_coords, cluster_ids = reduce_dimensions(embeddings, args.min_cluster_size, args.tsne_perplexity)

    # Stage 5
    print(f"\n[5/6] Extracting metadata...")
    timestamps = extract_timestamps(images)
    colors = extract_dominant_colors(images)
    print(f"  Timestamps: {sum(1 for t in timestamps if t > 0)}/{len(images)}")

    external_metadata = None
    if args.metadata:
        meta_src = Path(args.metadata).resolve()
        if meta_src.exists():
            external_metadata = read_external_metadata(str(meta_src))

    # Stage 6
    print(f"\n[6/6] Writing output files...")
    write_binary_data(str(output_dir), tsne_coords, atlas_data, cluster_ids)
    write_manifest(str(output_dir), len(images), atlas_count, args.thumb_size, args.atlas_size)
    write_metadata_csv(str(output_dir), images, cluster_ids, timestamps, colors, external_metadata)

    elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  ✓ {len(images)} images → {output_dir}")
    print(f"  ✓ {atlas_count} atlas textures (WebP)")
    print(f"  ✓ Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

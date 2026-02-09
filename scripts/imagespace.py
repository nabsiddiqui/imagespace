#!/usr/bin/env python3
"""
ImageSpace — A Minimal-Computing Pipeline for Exploratory Visualization of Image Collections

Transforms a folder of images into static data files for the ImageSpace viewer:
  - Atlas JPEG textures (sprite sheets of thumbnails)
  - Binary layout data (UMAP + t-SNE coordinates, atlas positions, cluster IDs)
  - Manifest JSON (metadata for the viewer)
  - Optional metadata CSV (filename, cluster, timestamp, dominant color)

Usage:
    python imagespace.py /path/to/images --output ./dist/data
    python imagespace.py /path/to/images --output ./dist/data --gpu
    python imagespace.py /path/to/images --output ./dist/data --metadata existing.csv

Dependencies:
    Required: pillow, numpy, umap-learn, scikit-learn
    Optional: torch, transformers (for CLIP embeddings)
              onnxruntime (alternative CLIP backend)

CPU-first: runs on any laptop. --gpu flag enables hardware acceleration when available.
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

import numpy as np
from PIL import Image, ExifTags

# ── Configuration ─────────────────────────────────────────────
THUMB_SIZE = 64          # Thumbnail size in pixels (square)
ATLAS_SIZE = 4096        # Atlas texture size (4096x4096 = 64x64 grid of 64px thumbs)
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
CLIP_IMAGE_SIZE = 224    # CLIP input resolution
CLIP_DIM = 512           # CLIP ViT-B/32 embedding dimension
NUM_CLUSTERS = 15        # KMeans clusters (article uses 15)
UMAP_NEIGHBORS = 15      # UMAP n_neighbors
UMAP_MIN_DIST = 0.1      # UMAP min_dist
TSNE_PERPLEXITY = 30     # t-SNE perplexity
BATCH_SIZE = 32          # Embedding batch size


# ── Stage 1: Image Discovery ─────────────────────────────────
def discover_images(input_dir):
    """Recursively find all supported image files, skipping hidden/system dirs."""
    images = []
    input_path = Path(input_dir).resolve()
    for root, dirs, files in os.walk(input_path):
        # Skip hidden directories and common system dirs
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', 'node_modules', '.git')]
        for f in sorted(files):
            if f.startswith('.'):
                continue
            ext = Path(f).suffix.lower()
            if ext in SUPPORTED_FORMATS:
                images.append(Path(root) / f)
    return images


# ── Stage 2: Thumbnail Generation + Atlas Packing ────────────
def generate_atlases(images, output_dir, thumb_size=THUMB_SIZE, atlas_size=ATLAS_SIZE):
    """Create JPEG atlas textures from image thumbnails. Returns atlas metadata per image."""
    images_per_row = atlas_size // thumb_size
    images_per_atlas = images_per_row * images_per_row

    atlas_data = []  # (atlas_idx, u, v) per image
    current_atlas_idx = 0
    current_img_in_atlas = 0
    atlas_img = Image.new('RGB', (atlas_size, atlas_size), (255, 255, 255))

    for idx, img_path in enumerate(images):
        if idx % 1000 == 0:
            print(f"  Thumbnailing {idx}/{len(images)}...", end='\r')

        try:
            img = Image.open(img_path).convert('RGB')
            # Resize to square thumbnail with center crop
            w, h = img.size
            side = min(w, h)
            left = (w - side) // 2
            top = (h - side) // 2
            img = img.crop((left, top, left + side, top + side))
            img = img.resize((thumb_size, thumb_size), Image.LANCZOS)
        except Exception as e:
            print(f"\n  Warning: Failed to process {img_path}: {e}")
            # Create a gray placeholder
            img = Image.new('RGB', (thumb_size, thumb_size), (128, 128, 128))

        col = current_img_in_atlas % images_per_row
        row = current_img_in_atlas // images_per_row
        u, v = col * thumb_size, row * thumb_size
        atlas_img.paste(img, (u, v))
        atlas_data.append((current_atlas_idx, u, v))

        current_img_in_atlas += 1
        if current_img_in_atlas >= images_per_atlas or idx == len(images) - 1:
            atlas_path = os.path.join(output_dir, f'atlas_{current_atlas_idx}.jpg')
            atlas_img.save(atlas_path, 'JPEG', quality=90)
            print(f"  Saved atlas_{current_atlas_idx}.jpg ({current_img_in_atlas} images)")
            current_atlas_idx += 1
            current_img_in_atlas = 0
            atlas_img = Image.new('RGB', (atlas_size, atlas_size), (255, 255, 255))

    return atlas_data, current_atlas_idx


# ── Stage 3: Embedding Extraction ────────────────────────────
def extract_clip_embeddings(images, use_gpu=False):
    """Extract CLIP ViT-B/32 embeddings. Falls back to color histograms if unavailable."""
    try:
        return _extract_clip_torch(images, use_gpu)
    except ImportError:
        pass

    try:
        return _extract_clip_onnx(images, use_gpu)
    except (ImportError, Exception) as e:
        print(f"  CLIP unavailable ({e}), falling back to color histograms...")
        return _extract_color_histograms(images)


def _extract_clip_torch(images, use_gpu=False):
    """CLIP embeddings via transformers + PyTorch."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    print("  Loading CLIP ViT-B/32 (transformers)...")
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
        else:
            print("  No GPU detected, using CPU")

    model = model.to(device).eval()
    embeddings = np.zeros((len(images), CLIP_DIM), dtype=np.float32)

    start = time.time()
    for batch_start in range(0, len(images), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(images))
        batch_images = []
        for img_path in images[batch_start:batch_end]:
            try:
                img = Image.open(img_path).convert('RGB')
                batch_images.append(img)
            except Exception:
                batch_images.append(Image.new('RGB', (CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE), (128, 128, 128)))

        inputs = processor(images=batch_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            # L2 normalize
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)
            embeddings[batch_start:batch_end] = outputs.cpu().numpy()

        elapsed = time.time() - start
        rate = (batch_end) / elapsed if elapsed > 0 else 0
        eta = (len(images) - batch_end) / rate if rate > 0 else 0
        print(f"  Embedding {batch_end}/{len(images)} ({rate:.1f} img/s, ETA {eta:.0f}s)", end='\r')

    print(f"\n  CLIP embeddings extracted in {time.time() - start:.1f}s")
    return embeddings


def _extract_clip_onnx(images, use_gpu=False):
    """CLIP embeddings via ONNX Runtime (alternative backend)."""
    import onnxruntime as ort
    from transformers import CLIPProcessor

    print("  Loading CLIP ViT-B/32 (ONNX Runtime)...")
    # Try to find or download the ONNX model
    model_path = _get_clip_onnx_model()
    if model_path is None:
        raise ImportError("CLIP ONNX model not available")

    providers = ['CPUExecutionProvider']
    if use_gpu:
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
            print("  Using CUDA via ONNX Runtime")
        elif 'CoreMLExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CoreMLExecutionProvider')
            print("  Using CoreML via ONNX Runtime")

    session = ort.InferenceSession(model_path, providers=providers)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    embeddings = np.zeros((len(images), CLIP_DIM), dtype=np.float32)

    start = time.time()
    for batch_start in range(0, len(images), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(images))
        batch_images = []
        for img_path in images[batch_start:batch_end]:
            try:
                img = Image.open(img_path).convert('RGB')
                batch_images.append(img)
            except Exception:
                batch_images.append(Image.new('RGB', (CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE), (128, 128, 128)))

        inputs = processor(images=batch_images, return_tensors="np", padding=True)
        outputs = session.run(None, {'pixel_values': inputs['pixel_values']})[0]

        # L2 normalize
        norms = np.linalg.norm(outputs, axis=-1, keepdims=True)
        norms[norms == 0] = 1
        embeddings[batch_start:batch_end] = outputs / norms

        print(f"  Embedding {batch_end}/{len(images)}", end='\r')

    print(f"\n  CLIP embeddings (ONNX) extracted in {time.time() - start:.1f}s")
    return embeddings


def _get_clip_onnx_model():
    """Locate or download the CLIP ViT-B/32 ONNX model. Returns path or None."""
    cache_dir = Path.home() / '.cache' / 'imagespace'
    model_path = cache_dir / 'clip-vit-b32-visual.onnx'
    if model_path.exists():
        return str(model_path)
    # Could add download logic here; for now return None
    return None


def _extract_color_histograms(images, bins=64, dim=CLIP_DIM):
    """Fallback: color histogram embeddings (no CLIP dependency)."""
    print("  Extracting color histograms...")
    embeddings = np.zeros((len(images), dim), dtype=np.float32)

    for idx, img_path in enumerate(images):
        if idx % 500 == 0:
            print(f"  Histogram {idx}/{len(images)}", end='\r')
        try:
            img = Image.open(img_path).convert('RGB').resize((64, 64))
            arr = np.array(img)
            # HSL histogram
            h_hist = np.histogram(arr[:, :, 0], bins=bins // 3, range=(0, 255))[0]
            s_hist = np.histogram(arr[:, :, 1], bins=bins // 3, range=(0, 255))[0]
            l_hist = np.histogram(arr[:, :, 2], bins=bins // 3, range=(0, 255))[0]
            hist = np.concatenate([h_hist, s_hist, l_hist]).astype(np.float32)
            # Pad or truncate to dim
            if len(hist) < dim:
                hist = np.pad(hist, (0, dim - len(hist)))
            else:
                hist = hist[:dim]
            # L2 normalize
            norm = np.linalg.norm(hist)
            if norm > 0:
                hist /= norm
            embeddings[idx] = hist
        except Exception:
            pass

    print(f"\n  Color histograms extracted for {len(images)} images")
    return embeddings


# ── Stage 4: Dimensionality Reduction + Clustering ───────────
def reduce_dimensions(embeddings, n_clusters=NUM_CLUSTERS):
    """Run UMAP, t-SNE, and KMeans on embeddings. Returns (umap_coords, tsne_coords, cluster_ids)."""
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    import umap

    n = len(embeddings)
    print(f"\n  Running UMAP (n={n}, neighbors={UMAP_NEIGHBORS}, min_dist={UMAP_MIN_DIST})...")
    start = time.time()
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric='cosine',
        random_state=42,
    )
    umap_coords = reducer.fit_transform(embeddings)
    print(f"  UMAP completed in {time.time() - start:.1f}s")

    print(f"  Running t-SNE (perplexity={TSNE_PERPLEXITY})...")
    start = time.time()
    tsne = TSNE(
        n_components=2,
        perplexity=min(TSNE_PERPLEXITY, n - 1),
        random_state=42,
        init='pca' if n > 50 else 'random',
        learning_rate='auto',
    )
    tsne_coords = tsne.fit_transform(embeddings)
    print(f"  t-SNE completed in {time.time() - start:.1f}s")

    # Scale both to similar range for viewer
    def scale_coords(coords, target_range=4000):
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1
        return (coords - mins) / ranges * target_range - target_range / 2

    umap_coords = scale_coords(umap_coords)
    tsne_coords = scale_coords(tsne_coords)

    print(f"  Running KMeans (k={n_clusters})...")
    start = time.time()
    kmeans = KMeans(n_clusters=min(n_clusters, n), random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(embeddings)
    print(f"  KMeans completed in {time.time() - start:.1f}s")

    return umap_coords.astype(np.float32), tsne_coords.astype(np.float32), cluster_ids.astype(np.int32)


# ── Stage 5: Dominant Color Extraction ────────────────────────
def extract_dominant_colors(images, thumb_size=32):
    """Extract dominant color (hue) for each image for color-sort view."""
    colors = []
    for idx, img_path in enumerate(images):
        try:
            img = Image.open(img_path).convert('RGB').resize((thumb_size, thumb_size))
            arr = np.array(img).reshape(-1, 3)
            # Average color
            avg = arr.mean(axis=0) / 255.0
            h, l, s = colorsys.rgb_to_hls(avg[0], avg[1], avg[2])
            colors.append((h, s, l))
        except Exception:
            colors.append((0, 0, 0))
    return colors


# ── Stage 6: Extract Timestamps ──────────────────────────────
def extract_timestamps(images):
    """Try to extract timestamps from EXIF or filename patterns."""
    timestamps = []
    for img_path in images:
        ts = _get_exif_timestamp(img_path)
        if ts is None:
            ts = 0
        timestamps.append(ts)
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
def write_binary_data(output_dir, umap_coords, tsne_coords, atlas_data, cluster_ids):
    """Write binary layout data. 24 bytes per image:
       float32 umapX, float32 umapY, float32 tsneX, float32 tsneY,
       uint16 atlasIdx, uint16 u, uint16 v, uint16 cluster
    """
    binary_data = bytearray()
    for i in range(len(umap_coords)):
        ai, u, v = atlas_data[i]
        cid = int(cluster_ids[i])
        item = struct.pack('<ffffHHHH',
            umap_coords[i][0], umap_coords[i][1],
            tsne_coords[i][0], tsne_coords[i][1],
            ai, u, v, cid
        )
        binary_data.extend(item)

    output_path = os.path.join(output_dir, 'data.bin')
    with open(output_path, 'wb') as f:
        f.write(binary_data)
    print(f"  Binary data: {len(binary_data) / 1024:.1f} KB ({len(binary_data)} bytes)")
    return output_path


def write_manifest(output_dir, count, atlas_count, thumb_size=THUMB_SIZE):
    """Write manifest.json for the viewer."""
    manifest = {
        'count': count,
        'atlasCount': atlas_count,
        'thumbSize': thumb_size,
        'bytesPerImage': 24,  # Extended format with UMAP + t-SNE + cluster
        'version': 2,
    }
    output_path = os.path.join(output_dir, 'manifest.json')
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest: {output_path}")
    return output_path


def write_metadata_csv(output_dir, images, cluster_ids, timestamps, colors):
    """Write metadata.csv with image info."""
    output_path = os.path.join(output_dir, 'metadata.csv')

    # Color name mapping
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

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'filename', 'cluster', 'timestamp', 'dominant_color'])
        for i, img_path in enumerate(images):
            h, s, l = colors[i] if i < len(colors) else (0, 0, 0)
            color_name = hue_to_name(h) if s > 0.1 else 'gray'
            writer.writerow([
                i,
                img_path.name,
                int(cluster_ids[i]),
                timestamps[i] if timestamps[i] > 0 else '',
                color_name,
            ])
    print(f"  Metadata CSV: {output_path}")
    return output_path


# ── Main Pipeline ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='ImageSpace — Transform images into an interactive visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python imagespace.py /path/to/images --output ./data
    python imagespace.py /path/to/images --output ./data --gpu
    python imagespace.py /path/to/images --output ./data --clusters 10
    python imagespace.py /path/to/images --output ./data --metadata existing.csv
        """
    )
    parser.add_argument('input', help='Directory containing images')
    parser.add_argument('--output', '-o', required=True, help='Output directory for data files')
    parser.add_argument('--gpu', action='store_true', help='Enable hardware acceleration (CoreML/CUDA)')
    parser.add_argument('--clusters', type=int, default=NUM_CLUSTERS, help=f'Number of KMeans clusters (default: {NUM_CLUSTERS})')
    parser.add_argument('--thumb-size', type=int, default=THUMB_SIZE, help=f'Thumbnail size in pixels (default: {THUMB_SIZE})')
    parser.add_argument('--metadata', help='Existing metadata CSV to include (must have "id" column)')
    parser.add_argument('--umap-neighbors', type=int, default=UMAP_NEIGHBORS, help=f'UMAP n_neighbors (default: {UMAP_NEIGHBORS})')
    parser.add_argument('--umap-min-dist', type=float, default=UMAP_MIN_DIST, help=f'UMAP min_dist (default: {UMAP_MIN_DIST})')
    parser.add_argument('--tsne-perplexity', type=int, default=TSNE_PERPLEXITY, help=f't-SNE perplexity (default: {TSNE_PERPLEXITY})')

    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    total_start = time.time()

    # Stage 1: Image Discovery
    print(f"\n{'='*60}")
    print(f"  ImageSpace Pipeline")
    print(f"{'='*60}")
    print(f"\n[1/6] Discovering images in {input_dir}...")
    images = discover_images(input_dir)
    if not images:
        print("  Error: No images found!")
        sys.exit(1)
    print(f"  Found {len(images)} images")

    # Stage 2: Thumbnail Generation + Atlas Packing
    print(f"\n[2/6] Generating atlas textures (thumb={args.thumb_size}px)...")
    atlas_data, atlas_count = generate_atlases(images, str(output_dir), args.thumb_size)

    # Stage 3: Embedding Extraction
    print(f"\n[3/6] Extracting embeddings {'(GPU)' if args.gpu else '(CPU)'}...")
    embeddings = extract_clip_embeddings(images, use_gpu=args.gpu)

    # Stage 4: Dimensionality Reduction + Clustering
    print(f"\n[4/6] Dimensionality reduction + clustering...")
    global UMAP_NEIGHBORS, UMAP_MIN_DIST, TSNE_PERPLEXITY
    UMAP_NEIGHBORS = args.umap_neighbors
    UMAP_MIN_DIST = args.umap_min_dist
    TSNE_PERPLEXITY = args.tsne_perplexity
    umap_coords, tsne_coords, cluster_ids = reduce_dimensions(embeddings, args.clusters)

    # Stage 5: Extract metadata
    print(f"\n[5/6] Extracting metadata...")
    timestamps = extract_timestamps(images)
    colors = extract_dominant_colors(images)
    has_timestamps = any(t > 0 for t in timestamps)
    if has_timestamps:
        n_ts = sum(1 for t in timestamps if t > 0)
        print(f"  Found {n_ts} EXIF timestamps")
    else:
        print(f"  No EXIF timestamps found")

    # Stage 6: Write output
    print(f"\n[6/6] Writing output files...")
    write_binary_data(str(output_dir), umap_coords, tsne_coords, atlas_data, cluster_ids)
    write_manifest(str(output_dir), len(images), atlas_count, args.thumb_size)
    write_metadata_csv(str(output_dir), images, cluster_ids, timestamps, colors)

    elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete!")
    print(f"  {len(images)} images → {output_dir}")
    print(f"  {atlas_count} atlas textures")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

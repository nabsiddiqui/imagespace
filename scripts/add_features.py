#!/usr/bin/env python3
"""
Add computed image features to an existing metadata.csv using atlas thumbnails.
Much faster than re-running the full pipeline since we read from pre-built atlases.

Usage:
    python add_features.py /path/to/data/dir

Adds columns: brightness, complexity, edge_density, outlier_score, cluster_confidence
"""

import csv
import json
import os
import struct
import sys
import time

import numpy as np
from PIL import Image


def main():
    if len(sys.argv) < 2:
        print("Usage: python add_features.py /path/to/data/dir")
        sys.exit(1)

    data_dir = sys.argv[1]
    manifest_path = os.path.join(data_dir, 'manifest.json')
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    neighbors_path = os.path.join(data_dir, 'neighbors.bin')
    data_bin_path = os.path.join(data_dir, 'data.bin')

    if not os.path.exists(manifest_path):
        print(f"Error: manifest.json not found in {data_dir}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    count = manifest['count']
    atlas_count = manifest['atlasCount']
    thumb_size = manifest['thumbSize']
    atlas_size = manifest['atlasSize']
    atlas_format = manifest.get('atlasFormat', 'webp')

    print(f"Dataset: {count} images, {atlas_count} atlases, {thumb_size}px thumbs, {atlas_format}")

    # ── Read atlas positions from data.bin ──
    print("Reading atlas positions from data.bin...")
    with open(data_bin_path, 'rb') as f:
        raw = f.read()

    atlas_data = []
    for i in range(count):
        off = i * 24
        ai = struct.unpack_from('<H', raw, off + 16)[0]
        u = struct.unpack_from('<H', raw, off + 18)[0]
        v = struct.unpack_from('<H', raw, off + 20)[0]
        atlas_data.append((ai, u, v))

    # ── Load all atlas images ──
    print("Loading atlas images...")
    atlases = {}
    for i in range(atlas_count):
        path = os.path.join(data_dir, f'atlas_{i}.{atlas_format}')
        if os.path.exists(path):
            atlases[i] = np.array(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
        else:
            print(f"  Warning: atlas_{i}.{atlas_format} not found")

    # ── Compute features from atlas thumbnails ──
    print(f"Computing features for {count} images...")
    start = time.time()

    from scipy.ndimage import sobel

    brightness = np.zeros(count, dtype=np.float32)
    complexity = np.zeros(count, dtype=np.float32)
    edge_density = np.zeros(count, dtype=np.float32)

    for i in range(count):
        ai, u, v = atlas_data[i]
        if ai not in atlases:
            continue

        # Extract thumbnail from atlas
        atlas = atlases[ai]
        thumb = atlas[v:v + thumb_size, u:u + thumb_size]
        if thumb.shape[0] == 0 or thumb.shape[1] == 0:
            continue

        # Brightness: mean luminance (BT.601)
        lum = thumb[:, :, 0] * 0.299 + thumb[:, :, 1] * 0.587 + thumb[:, :, 2] * 0.114
        brightness[i] = float(lum.mean())

        # Complexity: Shannon entropy of grayscale histogram
        gray = (lum * 255).astype(np.uint8)
        hist, _ = np.histogram(gray, bins=64, range=(0, 255))
        hist = hist[hist > 0].astype(np.float32)
        hist /= hist.sum()
        complexity[i] = float(-np.sum(hist * np.log2(hist)))

        # Edge density: mean Sobel gradient magnitude
        sx = sobel(lum, axis=0)
        sy = sobel(lum, axis=1)
        edge_density[i] = float(np.sqrt(sx ** 2 + sy ** 2).mean())

        if i > 0 and i % 10000 == 0:
            print(f"  {i}/{count} ({i / count * 100:.0f}%)")

    # Normalize to 0-100 scale
    def norm100(arr):
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            return ((arr - mn) / (mx - mn) * 100).round(1)
        return np.zeros_like(arr)

    brightness = norm100(brightness)
    complexity = norm100(complexity)
    edge_density = norm100(edge_density)
    print(f"  Image features computed in {time.time() - start:.1f}s")

    # ── Compute outlier scores from neighbors.bin ──
    outlier_scores = np.zeros(count, dtype=np.float32)
    if os.path.exists(neighbors_path):
        print("Computing outlier scores from neighbors.bin...")
        with open(neighbors_path, 'rb') as f:
            nn_count = struct.unpack('<I', f.read(4))[0]
            nn_k = struct.unpack('<I', f.read(4))[0]
            dists = np.zeros((nn_count, nn_k), dtype=np.float32)
            for i in range(nn_count):
                for j in range(nn_k):
                    _ = struct.unpack('<I', f.read(4))[0]  # index
                    dists[i, j] = struct.unpack('<f', f.read(4))[0]
        mean_dist = dists.mean(axis=1)
        outlier_scores = norm100(mean_dist)
        print(f"  Outlier scores: min={outlier_scores.min():.1f}, max={outlier_scores.max():.1f}, mean={outlier_scores.mean():.1f}")

    # ── Compute cluster confidence from data.bin cluster IDs ──
    # Use a proxy: HDBSCAN probabilities aren't in the binary, so we compute
    # cluster confidence as inverse intra-cluster distance (how close to centroid)
    print("Computing cluster confidence...")
    cluster_ids = np.zeros(count, dtype=np.int32)
    coords = np.zeros((count, 2), dtype=np.float32)
    for i in range(count):
        off = i * 24
        coords[i, 0] = struct.unpack_from('<f', raw, off)[0]
        coords[i, 1] = struct.unpack_from('<f', raw, off + 4)[0]
        cluster_ids[i] = struct.unpack_from('<H', raw, off + 22)[0]

    # Compute distance to cluster centroid in t-SNE space
    unique_clusters = np.unique(cluster_ids)
    centroids = {}
    for cid in unique_clusters:
        mask = cluster_ids == cid
        centroids[cid] = coords[mask].mean(axis=0)

    dists_to_centroid = np.zeros(count, dtype=np.float32)
    for i in range(count):
        cid = cluster_ids[i]
        dists_to_centroid[i] = np.linalg.norm(coords[i] - centroids[cid])

    # Invert: closer to centroid = higher confidence
    max_d = dists_to_centroid.max()
    if max_d > 0:
        cluster_confidence = ((1 - dists_to_centroid / max_d) * 100).round(1)
    else:
        cluster_confidence = np.full(count, 100.0)

    # ── Read existing metadata.csv and add new columns ──
    print("Updating metadata.csv...")
    rows = []
    headers = []
    with open(metadata_path, 'r', newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            rows.append(row)

    # Remove old feature columns if they exist (for re-runs)
    new_feature_cols = ['brightness', 'complexity', 'edge_density', 'outlier_score', 'cluster_confidence']
    existing_feature_indices = []
    for col in new_feature_cols:
        if col in headers:
            existing_feature_indices.append(headers.index(col))

    if existing_feature_indices:
        # Remove old feature columns
        for idx in sorted(existing_feature_indices, reverse=True):
            headers.pop(idx)
            for row in rows:
                if idx < len(row):
                    row.pop(idx)

    # Add new feature columns
    headers.extend(new_feature_cols)
    for i, row in enumerate(rows):
        row.extend([
            brightness[i] if i < count else 0,
            complexity[i] if i < count else 0,
            edge_density[i] if i < count else 0,
            outlier_scores[i] if i < count else 0,
            cluster_confidence[i] if i < count else 0,
        ])

    # Write updated CSV
    with open(metadata_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"\n✓ Added {len(new_feature_cols)} feature columns to metadata.csv")
    print(f"  Columns: {', '.join(new_feature_cols)}")
    print(f"  Brightness range: {brightness.min():.1f} - {brightness.max():.1f}")
    print(f"  Complexity range: {complexity.min():.1f} - {complexity.max():.1f}")
    print(f"  Edge density range: {edge_density.min():.1f} - {edge_density.max():.1f}")
    print(f"  Outlier score range: {outlier_scores.min():.1f} - {outlier_scores.max():.1f}")
    print(f"  Cluster confidence range: {cluster_confidence.min():.1f} - {cluster_confidence.max():.1f}")


if __name__ == '__main__':
    main()

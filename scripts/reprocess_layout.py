#!/usr/bin/env python3
"""Reprocess existing data.bin: apply non-overlap + re-cluster without re-embedding.

Usage:
    python3 scripts/reprocess_layout.py frontend-pixi/public/data/
"""
import sys, struct, time
import numpy as np

THUMB_SIZE = 64
ATLAS_SIZE = 4096

def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'frontend-pixi/public/data'
    bin_path = f'{data_dir}/data.bin'

    # Read existing binary data
    print("Reading data.bin...")
    raw = open(bin_path, 'rb').read()
    n = len(raw) // 24
    print(f"  {n} images")

    coords = np.zeros((n, 2), dtype=np.float32)
    atlas_data = []  # (ai, u, v) per image

    for i in range(n):
        off = i * 24
        x, y, tx, ty = struct.unpack_from('<ffff', raw, off)
        ai, u, v, cluster = struct.unpack_from('<HHHH', raw, off + 16)
        coords[i] = [x, y]
        atlas_data.append((ai, u, v))

    print(f"  Original range: X [{coords[:,0].min():.0f}, {coords[:,0].max():.0f}], "
          f"Y [{coords[:,1].min():.0f}, {coords[:,1].max():.0f}]")

    # Cluster using art styles from metadata.csv (much more meaningful than density-based)
    print("\nAssigning clusters from metadata styles...")
    start = time.time()
    import csv, os
    meta_path = os.path.join(data_dir, 'metadata.csv')
    cluster_ids = np.zeros(n, dtype=np.int32)

    if os.path.exists(meta_path):
        style_to_id = {}
        with open(meta_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = int(row['id'])
                style = row.get('style', 'Unknown')
                if style not in style_to_id:
                    style_to_id[style] = len(style_to_id)
                if idx < n:
                    cluster_ids[idx] = style_to_id[style]
        n_clusters = len(style_to_id)
        print(f"  {n_clusters} style clusters ({time.time()-start:.1f}s)")
        for style, cid in sorted(style_to_id.items(), key=lambda x: x[1]):
            cnt = (cluster_ids == cid).sum()
            print(f"    [{cid}] {style}: {cnt} images")
    else:
        # Fallback: HDBSCAN
        try:
            import hdbscan as hdb
            clusterer = hdb.HDBSCAN(
                min_cluster_size=200,
                min_samples=10,
                metric='euclidean',
                cluster_selection_method='eom',
                core_dist_n_jobs=-1,
            )
            cluster_ids = clusterer.fit_predict(coords)
            n_clusters = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
            n_noise = (cluster_ids == -1).sum()
            if n_noise > 0 and n_clusters > 0:
                from scipy.spatial import cKDTree
                valid_mask = cluster_ids >= 0
                tree = cKDTree(coords[valid_mask])
                valid_labels = cluster_ids[valid_mask]
                _, nearest = tree.query(coords[cluster_ids == -1])
                cluster_ids[cluster_ids == -1] = valid_labels[nearest]
            print(f"  HDBSCAN: {n_clusters} clusters, {n_noise} noise ({time.time()-start:.1f}s)")
        except ImportError:
            from sklearn.cluster import MiniBatchKMeans
            k = 15
            cluster_ids = MiniBatchKMeans(n_clusters=k, batch_size=1024, random_state=42, n_init=3).fit_predict(coords)
            print(f"  KMeans: {k} clusters ({time.time()-start:.1f}s)")

    # Scale up coordinates to prevent overlap
    cell_size = THUMB_SIZE * 1.15  # slight gap
    target_side = int(np.ceil(np.sqrt(n * 1.8))) * cell_size
    print(f"\nScaling coords: 4000 → {target_side:.0f} (cell={cell_size:.1f}px)")

    # Normalize to new range
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    scaled = (coords - mins) / ranges * target_side - target_side / 2

    # Grid-snap non-overlap algorithm
    print("Removing overlaps (center-out priority)...")
    start = time.time()
    occupied = set()
    result = np.zeros_like(scaled)
    centroid = scaled.mean(axis=0)
    dists = np.linalg.norm(scaled - centroid, axis=1)
    order = np.argsort(dists)

    for idx in order:
        gx = round(scaled[idx, 0] / cell_size)
        gy = round(scaled[idx, 1] / cell_size)
        if (gx, gy) not in occupied:
            occupied.add((gx, gy))
            result[idx] = [gx * cell_size, gy * cell_size]
            continue
        # Spiral search
        placed = False
        for r in range(1, 2000):
            for dx in range(-r, r + 1):
                for dy in (-r, r):
                    if (gx + dx, gy + dy) not in occupied:
                        occupied.add((gx + dx, gy + dy))
                        result[idx] = [(gx + dx) * cell_size, (gy + dy) * cell_size]
                        placed = True; break
                if placed: break
            if placed: break
            for dy in range(-r + 1, r):
                for dx in (-r, r):
                    if (gx + dx, gy + dy) not in occupied:
                        occupied.add((gx + dx, gy + dy))
                        result[idx] = [(gx + dx) * cell_size, (gy + dy) * cell_size]
                        placed = True; break
                if placed: break
            if placed: break

    print(f"  Overlap removal: {time.time()-start:.1f}s")
    print(f"  New range: X [{result[:,0].min():.0f}, {result[:,0].max():.0f}], "
          f"Y [{result[:,1].min():.0f}, {result[:,1].max():.0f}]")

    # Write updated binary data
    print(f"\nWriting updated data.bin...")
    binary_data = bytearray(n * 24)
    for i in range(n):
        ai, u, v = atlas_data[i]
        cid = int(cluster_ids[i])
        tx, ty = float(result[i, 0]), float(result[i, 1])
        struct.pack_into('<ffffHHHH', binary_data, i * 24,
            tx, ty, tx, ty, ai, u, v, cid)

    with open(bin_path, 'wb') as f:
        f.write(binary_data)
    print(f"  Written {len(binary_data) / 1024:.1f} KB")

    # Verify
    unique_clusters = len(set(cluster_ids))
    print(f"\n✓ {n} images → {unique_clusters} clusters, non-overlapping layout")
    from collections import Counter
    for cid, cnt in Counter(cluster_ids).most_common(10):
        print(f"  Cluster {cid}: {cnt} images")

if __name__ == '__main__':
    main()

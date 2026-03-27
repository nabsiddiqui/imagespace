#!/usr/bin/env python3
"""Analyze new HDBSCAN cluster compositions from metadata.csv"""

import csv
from collections import Counter, defaultdict

# Read metadata
with open('/Users/nabeel/Documents/ImageSpace/frontend-pixi/public/data/metadata.csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f'Total images: {len(rows)}')

# Cluster sizes
cluster_counts = Counter(r['cluster'] for r in rows)
print(f'\nCluster sizes:')
for c, n in sorted(cluster_counts.items(), key=lambda x: int(x[0])):
    print(f'  Cluster {c}: {n}')

# Style distribution
style_counts = Counter(r['style'] for r in rows)
print(f'\nTop 15 styles:')
for s, n in style_counts.most_common(15):
    pct = 100 * n / len(rows)
    print(f'  {s}: {n} ({pct:.1f}%)')

# Per-cluster style breakdown
print(f'\n--- Per-cluster top styles ---')
cluster_styles = defaultdict(Counter)
for r in rows:
    cluster_styles[r['cluster']][r['style']] += 1

for c in sorted(cluster_styles.keys(), key=int):
    total = cluster_counts[c]
    top4 = cluster_styles[c].most_common(4)
    print(f'\nCluster {c} (n={total}):')
    for s, n in top4:
        pct = 100 * n / total
        print(f'  {s}: {n} ({pct:.1f}%)')

# Check Ukiyo-e distribution
print('\n--- Ukiyo-e distribution ---')
ukiyoe_clusters = Counter()
for r in rows:
    if 'ukiyo' in r['style'].lower():
        ukiyoe_clusters[r['cluster']] += 1
total_ukiyoe = sum(ukiyoe_clusters.values())
print(f'Total Ukiyo-e: {total_ukiyoe}')
for c, n in ukiyoe_clusters.most_common():
    pct = 100 * n / total_ukiyoe
    cluster_total = cluster_counts[c]
    cluster_pct = 100 * n / cluster_total
    print(f'  Cluster {c}: {n} ({pct:.1f}% of Ukiyo-e, {cluster_pct:.1f}% of cluster)')

# Most "pure" clusters (highest single-style concentration)
print('\n--- Cluster purity (highest single-style %) ---')
purity = []
for c in sorted(cluster_styles.keys(), key=int):
    total = cluster_counts[c]
    top_style, top_count = cluster_styles[c].most_common(1)[0]
    pct = 100 * top_count / total
    purity.append((c, top_style, top_count, total, pct))
purity.sort(key=lambda x: -x[4])
for c, style, count, total, pct in purity[:10]:
    print(f'  Cluster {c}: {style} at {pct:.1f}% ({count}/{total})')

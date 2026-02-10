# ImageSpace — Active Context

## Current State (Task 7 / Session 2)
Range slider filters redesigned as collapsible bottom panel. Canvas-based detail panel thumbnail.

### Recent Changes (This Session)
- **Pipeline: image features** — Added `compute_image_features()` computing brightness (BT.601 luminance), complexity (Shannon entropy), edge density (Sobel magnitude), all normalized 0-100
- **Pipeline: outlier scores** — Mean k-NN distance normalized to 0-100 (uses existing neighbors data)
- **Pipeline: cluster confidence** — HDBSCAN `probabilities_` (or inverse-centroid-distance for KMeans fallback), 0-100 scale
- **Pipeline: updated `write_metadata_csv`** — Accepts and writes new feature columns
- **Pipeline: returns 4 values** — `reduce_dimensions()` now returns `(tsne_coords, cluster_ids, embeddings_pca, cluster_probs)`
- **Standalone script: `scripts/add_features.py`** — Extracts features from existing atlas thumbnails without re-running full pipeline. Ran successfully on WikiArt dataset (~80s for 49,585 images)
- **Viewer: `continuousFilterOptions`** — Auto-detects known continuous columns (`brightness`, `complexity`, `edge_density`, `outlier_score`, `cluster_confidence`) with min/max ranges
- **Viewer: `rangeFilters` state** — `{ col: [min, max] }` for active range filters
- **Viewer: `handleRangeChange`** — Callback that updates range filters and recomputes visible set
- **Viewer: range slider UI** — Dual-handle range sliders below checkbox filter bar, matching Rose Pine Dawn theme
- **Viewer: `computeVisibleSet` updated** — Now takes 5th parameter `ranges` for continuous filters, intersects with existing hotspot + CSV filters
- **Auto-open detail panel** on image click
- **Pointer cursor** on image hover
- **Thumbnail fix** — switched from CSS `backgroundImage` to canvas `drawImage()` for atlas crop rendering in detail panel (CSS approach fails with large 4096×4096 atlases)
- **Range slider redesign** — moved from bottom panel to right-side vertical bubble cards (200px wide, stacked individually, each with own border/shadow). Shifts left when detail panel is open. Toggle via "Properties" button in top-right filter area. Header row has "Reset all" + close button.
- **`showRangePanel` state** — controls visibility of bottom range panel

### Metadata CSV Columns (WikiArt)
id, filename, cluster, timestamp, dominant_color (12 unique), artist (1,092), style (27), title (47,121), width, height, **brightness** (0-100), **complexity** (0-100), **edge_density** (0-100), **outlier_score** (0-100), **cluster_confidence** (0-100)

## Active View Modes
- **t-SNE** — Visual similarity layout (openTSNE FFT-accelerated)
- **Grid** — Square grid layout
- **Color** — Sorted by average hue
- **Timeline** — Sorted by extracted year/timestamp
- **Carousel** — Full-screen single image with hotspots visible

## Key Interactions
- **Range sliders** (right-side bubble cards, toggle via "Properties" button in top-right): Stacked vertical cards for brightness, complexity, edge density, uniqueness, cluster fit. Each card is individually styled. Shifts left when detail panel open. Intersects with all other filters.
- **Click image** → detail panel auto-opens
- **Hover image** → pointer cursor, scale 1.5x, gold tint, tooltip
- **Hotspot cards** (left column): HDBSCAN clusters. Click to filter.
- **CSV Filters** (top-right, checkboxes): Multi-select, additive/union.
- **Clear buttons**: Per-bar reset for range sliders, per-column and global clear for all filters.

## Next Steps
- Test in browser (range sliders bottom panel, canvas thumbnail detail panel)
- Consider: export filtered set, bookmarks, keyboard shortcuts, permalink state
- Architecture: split App.jsx monolith (~1905 lines now)
- Never use Simple Browser

## Key Rules
- **NEVER open Simple Browser** (destroys user's memory/context)
- Use absolute paths for Python HTTP server
- Use `npx vite build` (not `vite build`) for local Vite 5

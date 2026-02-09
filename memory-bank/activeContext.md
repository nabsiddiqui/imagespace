# ImageSpace — Active Context

## Current State (Task 36)
Major performance overhaul completed. Pipeline rewritten for speed, viewer optimized with critic subagent feedback.

### Pipeline Rewrite
- **ONNX CLIP** (Xenova/clip-vit-base-patch32) as primary embedding — 2-3x faster than PyTorch
- **openTSNE** FFT-accelerated — 10-20x faster than sklearn t-SNE
- **HDBSCAN** with cKDTree noise reassignment (fallback: MiniBatchKMeans)
- **WebP atlases** (method=2, quality 80) — ~30% smaller than JPEG
- **PCA** 512→50d before t-SNE for faster convergence
- Fallback chain: ONNX CLIP → PyTorch CLIP → Color histograms

### Viewer Performance Fixes (from Critic Subagent)
- **CRITICAL FIX**: Detail panel was hardcoded to `.jpg` — now uses `atlasFormatRef.current`
- **Spatial hash**: Deferred rebuild to animation end (was O(n) per frame during transitions)
- **Cluster labels**: Throttled setState to 200ms (was 60fps React re-renders)
- **Timeline current time**: Throttled to 200ms (was 60fps setState)
- **Time filter dimming**: Only runs when slider values change (was every frame, 50K iterations)
- **PIXI imports**: Parallel `Promise.all` (was sequential awaits)
- **Atlas count**: Dynamic from manifest (was hardcoded "13")

### New Features
- Rose Pine Dawn logo adapted from legacy ImageSpace SVG
- Hotspots now visible in carousel mode
- Google Colab notebook for remote processing on GPU
- Improved loading progress bar (gradient, larger)

## Active View Modes
- **t-SNE** — Visual similarity layout (openTSNE FFT-accelerated)
- **Grid** — Square grid layout
- **Color** — Sorted by average hue
- **Timeline** — Sorted by extracted year/timestamp
- **Carousel** — Full-screen single image with hotspots visible

## Key Interactions
- **Hotspot cards** (left column): HDBSCAN clusters. Click to filter. Visible in ALL views including carousel.
- **CSV Filters** (top-right, checkboxes): Multi-select, additive/union.
- **Detail Panel** (right side): Toggle tab. Shows thumbnail from correct atlas format.
- **Timeline slider**: Dual-handle range for time filtering.

## Next Steps
- Test WikiArt pipeline output (49,585 images running now)
- Further critic iteration (architecture split of monolith App.jsx)
- Feature suggestions for non-technical users
- Consider: search, progressive atlas loading/LOD, CSV parser fix (commas in quoted fields)

## Recent Decisions (Task 36)
- UMAP removed entirely — only t-SNE (openTSNE)
- Renamed "Semantic embedding projection" → "Visual similarity layout"
- ONNX preferred over PyTorch for inference speed
- WebP over JPEG for atlases
- PCA dimensionality reduction before t-SNE
- Binary format v2 still 24 bytes but writes t-SNE coords to both slots

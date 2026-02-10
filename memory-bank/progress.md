# ImageSpace — Progress

## What Works ✅
### Viewer
- [x] PixiJS WebGL rendering of 50K sprites with WebP atlas textures (49 atlases, 128px thumbs)
- [x] 4 view modes: t-SNE, Grid, Color, Timeline (animated transitions, lerp 0.08)
- [x] Spatial hash hover detection + tooltip (deferred rebuild, squared distance)
- [x] Hotspot cards (left column, HDBSCAN clusters with thumbnails)
- [x] CSV metadata filters (multi-select dropdown checkboxes, additive/union, ≤200 unique values)
- [x] Range slider filters (brightness, complexity, edge density, uniqueness, cluster fit)
- [x] Combined filtering (hotspot ∩ CSV ∩ range sliders) via `computeVisibleSet` with 5 params
- [x] Detail panel: canvas-based thumbnail (DetailThumb component), similar images (k-NN), metadata
- [x] Detail panel auto-opens on image click
- [x] Pointer cursor on image hover (scale 1.5x, gold tint)
- [x] Stats modal with dynamic atlas count
- [x] Rose Pine Dawn theme with Inter font + custom SVG logo
- [x] Timeline view with dual-handle slider + date indicator
- [x] Color view sorted by average hue (deferred computation)
- [x] Content-bounds viewport fitting (uniform, no zoom multiplier)
- [x] Pan/zoom/pinch via pixi-viewport
- [x] Minimap with cached offscreen canvas + viewport rectangle (t-SNE mode)
- [x] Similar images (k-NN neighbors in detail panel, clickable fly-to)
- [x] Cluster size indicators in hotspot cards
- [x] CLIP cluster labels (floating, 100ms throttle)
- [x] Properties button → collapsible range slider bubble cards (right column)

### Pipeline
- [x] ONNX CLIP → PCA → openTSNE → HDBSCAN → k-NN → features → WebP atlases
- [x] Brightness (BT.601 luminance, 0-100)
- [x] Complexity (Shannon entropy, 0-100)
- [x] Edge density (Sobel magnitude, 0-100)
- [x] Outlier scores (mean k-NN distance, 0-100)
- [x] Cluster confidence (HDBSCAN probabilities, 0-100)
- [x] CLIP cluster labels
- [x] k-NN neighbors (cosine similarity on PCA embeddings)
- [x] Standalone feature extraction script (`add_features.py`)
- [x] Google Colab notebook for remote processing
- [x] WikiArt dataset processed (49,585 images)

### Performance Optimizations
- [x] Float32Array pre-parsing for numeric metadata columns
- [x] Deferred computeAvgColors (fire-and-forget async, 8K batch, non-blocking)
- [x] requestAnimationFrame throttle on range slider handleRangeChange
- [x] DetailThumb: React.memo + useRef + useEffect (no createRef in render)
- [x] Offscreen canvas cache for minimap dots
- [x] Squared distance comparison in hover (no Math.hypot)
- [x] Parallel atlas loading (Promise.all)
- [x] Parallel PIXI dynamic imports
- [x] Throttled React state updates in ticker (200ms)
- [x] movingSet for animation (only active sprites iterated)
- [x] container.boundsArea set to avoid O(n) bounds recalculation

### Infrastructure
- [x] Git version control (GitHub: nabsiddiqui/imagespace)
- [x] Comprehensive README with architecture, optimizations, scripts
- [x] Memory bank documentation (6 files)

## What's Left 🔧
- [ ] Export filtered set (CSV download of visible images)
- [ ] Bookmark/favorites with localStorage
- [ ] Keyboard shortcuts (arrows, Esc, 1-5 for views)
- [ ] Permalink/share state via URL hash
- [ ] Architecture: split App.jsx monolith (~1921 lines)
- [ ] GitHub Pages deployment guide
- [ ] Carousel view mode (code exists but not in UI tabs)

## Not Wanted
- Search/filter by filename or text (user explicitly declined)

## Known Issues
- Binary format writes t-SNE coords to both UMAP and t-SNE slots (redundant)
- ~870MB GPU memory for 49 atlases (edge of integrated GPU budget → ~30fps)
- Vite chunk size warning (pixi.js ~838KB) — expected, acceptable
- CSV parser doesn't handle RFC 4180 quoting (simple comma split)
- Python HTTP server needs absolute path for `-d` flag from background terminals
- Color view unavailable until background `computeAvgColors` completes

## Design Decisions History
1. PixiJS over Three.js (2D sprites, simpler)
2. Binary format over JSON (24 bytes vs ~200 bytes per image)
3. Atlas textures over individual images (49 fetches vs 50K)
4. HDBSCAN over K-means (density-based, better cluster quality)
5. openTSNE over sklearn (FFT-accelerated, 10-20x faster)
6. Rose Pine Dawn palette (clean, academic aesthetic)
7. Single monolith App.jsx (rapid iteration, may split later)
8. WebP over JPEG (30% smaller, modern browser support)
9. Deferred spatial hash rebuild (not per-frame)
10. Multi-select union for CSV filters, intersection across filter types
11. DetailThumb as separate React.memo component (proper lifecycle)
12. Float32Array pre-parsing for numeric metadata (eliminates parseFloat)
13. Offscreen canvas for minimap (cache dot layer, only redraw viewport rect)
14. requestAnimationFrame throttle for range sliders (not per-pixel)
15. Fire-and-forget async for computeAvgColors (non-blocking initial render)

## CPU Pipeline Timing — MEASURED (49,585 WikiArt images, Apple M-series)
| Stage | Time | Rate |
|---|---|---|
| Atlas generation (128px, q85) | **432.5s** (7.2 min) | 115 img/s |
| CLIP embeddings (ONNX CPU) | **1084.0s** (18.1 min) | 45.7 img/s |
| PCA (512d → 50d) | < 0.1s | 69% variance |
| openTSNE (FFT) | **26.2s** | — |
| Overlap removal | **3.3s** | — |
| HDBSCAN | **19.0s** | 19 clusters |
| k-NN (k=10, cosine) | **15.5s** | — |
| Cluster labels (CLIP) | **1.3s** | 19 labels |
| Metadata extraction (colors + timestamps) | **418.7s** (7.0 min) | — |
| Image features (brightness, complexity, edge) | **423.1s** (7.1 min) | — |
| **Total (first run)** | **2441.6s (40.7 min)** | |
| **Total (cached embeddings)** | **~22 min** | |
| **Total (relayout only)** | **~45s** | |
| **Feature extraction only (add_features.py)** | **~80s** | |

# ImageSpace — Progress

## What Works ✅
- [x] PixiJS WebGL rendering of 50K sprites with atlas textures
- [x] 5 view modes: t-SNE, Grid, Color, Timeline, Carousel
- [x] Animated transitions between view modes (lerp 0.08)
- [x] Spatial hash hover detection + tooltip (deferred rebuild for perf)
- [x] HDBSCAN density-based clustering with noise reassignment
- [x] Hotspot cards (left column, visible in ALL modes including carousel)
- [x] CSV metadata filters (multi-select checkboxes, additive/union, ≤200 unique values)
- [x] **Range slider filters (brightness, complexity, edge density, uniqueness, cluster fit)**
- [x] Combined filtering (hotspot ∩ CSV ∩ range sliders)
- [x] Detail panel with canvas-based atlas crop (drawImage from loaded atlas)
- [x] Detail panel auto-opens on image click
- [x] Pointer cursor on image hover
- [x] Stats modal with dynamic atlas count
- [x] Rose Pine Dawn theme with Inter font + custom logo
- [x] Timeline view with interactive dual-handle slider
- [x] Color view sorted by average hue
- [x] Carousel with filtered navigation + hotspots
- [x] Content-bounds viewport fitting
- [x] Pan/zoom/pinch via pixi-viewport
- [x] Minimap with crosshair cursor
- [x] Similar images (k-NN neighbors in detail panel)
- [x] Cluster size indicators in hotspot cards
- [x] CLIP cluster labels
- [x] **Pipeline: brightness, complexity, edge density features**
- [x] **Pipeline: outlier scores (mean k-NN distance)**
- [x] **Pipeline: cluster confidence (HDBSCAN probabilities)**
- [x] **Standalone feature extraction script (`add_features.py`)**
- [x] Python pipeline (ONNX CLIP → PCA → openTSNE → HDBSCAN → WebP atlases)
- [x] Parallel atlas loading (Promise.all)
- [x] Parallel PIXI dynamic imports
- [x] Throttled React state updates in ticker
- [x] Google Colab notebook for remote processing
- [x] Git version control (pushed to GitHub)
- [x] WikiArt dataset processing (49,585 images) with all features

## What's Left 🔧
- [ ] Test property bubble cards in browser
- [ ] Verify canvas thumbnail shows in detail panel (switched from CSS bg to drawImage)
- [ ] Export filtered set (CSV download of visible images)
- [ ] Bookmark/favorites with localStorage
- [ ] Keyboard shortcuts (arrows, Esc, 1-5 for views)
- [ ] Permalink/share state via URL hash
- [ ] Architecture: split App.jsx monolith (~1860 lines)
- [ ] GitHub Pages deployment guide
- [ ] Defer computeAvgColors to post-initial-render

## Not Wanted
- Search/filter by filename or text (user explicitly declined)

## Known Issues
- Color computation blocks main thread ~3-5s at load (should defer)
- Binary format writes t-SNE coords to both UMAP and t-SNE slots (redundant)
- 870MB+ GPU memory for 49 atlases (edge of integrated GPU budget → ~30fps)
- Vite chunk size warning (pixi.js ~838KB)
- CSV parser doesn't handle RFC 4180 quoting
- Python HTTP server needs absolute path for `-d` flag from background terminals

## Design Decisions History
1-14. (See previous entries)
15. Detail panel auto-opens on image click
16. Pointer cursor on image hover
17. Thumbnail rendering hardened (Math.round, bg fallback, no-repeat)
18. **Continuous features computed from atlas thumbnails (brightness, complexity, edge density)**
19. **Outlier scores from k-NN mean distance (0-100 scale)**
20. **Cluster confidence from HDBSCAN probabilities (0-100 scale)**
21. **Range sliders for continuous columns in viewer filter bar**
22. **`add_features.py` standalone script for updating existing datasets**

## CPU Pipeline Timing Estimates (50K images)
| Stage | Time |
|---|---|
| Atlas generation (WebP) | ~15-25 min |
| CLIP embedding (ONNX CPU) | ~3-5 hours |
| PCA + openTSNE + HDBSCAN | ~20-50 min |
| k-NN + cluster labels | ~3-5 min |
| **Image features (brightness, complexity, edge)** | **~2 min** |
| **Total (ONNX CPU)** | **~4-6 hours** |
| **Total (GPU/Colab)** | **~45-90 min** |

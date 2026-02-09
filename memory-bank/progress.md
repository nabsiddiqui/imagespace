# ImageSpace — Progress

## What Works ✅
- [x] PixiJS WebGL rendering of 50K sprites with atlas textures
- [x] 5 view modes: t-SNE, Grid, Color, Timeline, Carousel
- [x] Animated transitions between view modes (lerp 0.08)
- [x] Spatial hash hover detection + tooltip (deferred rebuild for perf)
- [x] HDBSCAN density-based clustering with noise reassignment
- [x] Hotspot cards (left column, visible in ALL modes including carousel)
- [x] CSV metadata filters (multi-select checkboxes, additive/union)
- [x] Combined filtering (hotspot ∩ CSV union)
- [x] Detail panel with correct atlas format (WebP)
- [x] Stats modal with dynamic atlas count
- [x] Rose Pine Dawn theme with Inter font + custom logo
- [x] Timeline view with interactive dual-handle slider
- [x] Color view sorted by average hue
- [x] Carousel with filtered navigation + hotspots
- [x] Content-bounds viewport fitting
- [x] Pan/zoom/pinch via pixi-viewport
- [x] **Python pipeline (ONNX CLIP → PCA → openTSNE → HDBSCAN → WebP atlases)**
- [x] **Parallel atlas loading (Promise.all)**
- [x] **Parallel PIXI dynamic imports**
- [x] **Throttled React state updates in ticker (cluster labels, timeline, time filter)**
- [x] **Google Colab notebook for remote processing**
- [x] Git version control (pushed to GitHub)
- [x] WikiArt dataset processing (49,585 images)

## What's Left 🔧
- [ ] Test WikiArt output in viewer
- [ ] Search/filter by filename or text
- [ ] Progressive atlas loading / LOD (for low-end GPUs)
- [ ] CSV parser fix (quoted values with commas)
- [ ] Architecture: split App.jsx monolith (~1484 lines)
- [ ] GitHub Pages deployment guide
- [ ] Defer computeAvgColors to post-initial-render

## Known Issues
- Color computation blocks main thread ~3-5s at load (should defer)
- Binary format writes t-SNE coords to both UMAP and t-SNE slots (redundant)
- 870MB GPU memory for 13 atlases (edge of integrated GPU budget)
- Vite chunk size warning (pixi.js ~838KB)
- CSV parser doesn't handle RFC 4180 quoting

## Design Decisions History
1. Started with NeoBrutalist theme → switched to Rose Pine Dawn
2. Started with HTML overlays → switched to PixiJS canvas layouts
3. UMAP removed → only t-SNE (openTSNE FFT-accelerated)
4. sklearn t-SNE → openTSNE (10-20x faster)
5. PyTorch CLIP → ONNX CLIP (2-3x faster, Xenova/clip-vit-base-patch32)
6. JPEG atlases → WebP atlases (~30% smaller, method=2)
7. Sequential atlas loading → parallel Promise.all
8. Spatial hash rebuild per frame → deferred to animation end
9. Cluster/timeline state 60fps → throttled to 200ms
10. Time filter dimming every frame → only on slider change
11. Hotspots hidden in carousel → visible everywhere
12. Hardcoded atlas count → dynamic from manifest
13. Fixed detail panel .jpg → dynamic atlas format
14. Google Colab notebook added for remote GPU processing

# ImageSpace — Progress

## What Works ✅
- [x] PixiJS WebGL rendering of 50K sprites with atlas textures
- [x] 6 view modes: UMAP, t-SNE, Grid, Color, Timeline, Carousel
- [x] Animated transitions between view modes (lerp 0.08)
- [x] Spatial hash hover detection + tooltip
- [x] K-means clustering (10 clusters, 25 iterations, client-side)
- [x] Hotspot cards (left column, larger 240px cards, hidden in carousel only)
- [x] CSV metadata filters (multi-select checkboxes, additive/union)
- [x] Combined filtering (hotspot ∩ CSV union)
- [x] Embedding modes (UMAP/t-SNE): dim non-cluster sprites (alpha 0.12)
- [x] Other modes: hide non-visible sprites (alpha 0) + relayout
- [x] Detail panel (right-edge tab toggle with "Panel" label)
- [x] Stats modal (collection info + cluster distribution)
- [x] Rose Pine Dawn theme with Inter font
- [x] Timeline view with interactive slider + time indicator
- [x] Timeline wheel scrolls horizontally (not zoom)
- [x] Color view sorted by average hue
- [x] Carousel auto-shows first image, navigates filtered images
- [x] Content-bounds viewport fitting (not world bounds)
- [x] Pan/zoom/pinch via pixi-viewport
- [x] Dummy data generator (generate_data.py)
- [x] Cluster labels floating above groups
- [x] FPS throttled to 200ms updates
- [x] Git version control (6 commits on master)

## What's Left 🔧
- [ ] Python backend pipeline (CLIP → UMAP/t-SNE → K-means → Atlas)
- [ ] Real image dataset support (not dummy shapes)
- [ ] Real t-SNE embedding (currently uses UMAP coordinates)
- [ ] Search/filter by filename
- [ ] Minimap overlay
- [ ] 3D scatter option
- [ ] GitHub Pages deployment guide
- [ ] Push to GitHub remote

## Known Issues
- Color computation takes ~3-5s for 50K images on load
- Cluster labels may overlap when zoomed out very far
- No lazy loading of atlas textures (all 13 loaded at boot)
- Vite chunk size warning (pixi.js is 838KB)
- t-SNE view is placeholder (same data as UMAP)

## Design Decisions History
1. Started with NeoBrutalist theme → switched to Rose Pine Dawn
2. Started with HTML overlays → switched to PixiJS canvas layouts
3. Clusters view removed from UI (Grid + filters equivalent)
4. Detail panel: auto-open → toggle button → right-edge tab with "Panel" label
5. Hotspot click: dim (0.12) in embedding mode, full hide + relayout in other modes
6. Carousel: all-images → filtered-images-only navigation → auto-show first
7. Old `frontend/` deleted — only `frontend-pixi/` remains
8. ImageStack + download features added then removed
9. Filters: left column → top-right dropdowns → multi-select checkboxes (additive/union)
10. Hotspots: hidden in timeline/carousel → visible everywhere except carousel
11. viewport.fit() → content-bounds fitting
12. Timeline: wheel zoom → wheel horizontal pan + interactive slider
13. Bottom-right stats card removed; FPS in Info modal only

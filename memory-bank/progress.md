# ImageSpace — Progress

## What Works ✅
- [x] PixiJS WebGL rendering of 50K sprites with atlas textures
- [x] 5 view modes: UMAP, Grid, Color, Timeline, Carousel
- [x] Animated transitions between view modes (lerp 0.08)
- [x] Spatial hash hover detection + tooltip
- [x] K-means clustering (10 clusters, 25 iterations, client-side)
- [x] Hotspot cards (left column, hidden in timeline/carousel)
- [x] CSV metadata filters (top-right horizontal bar)
- [x] Combined filtering (hotspot ∩ CSV filters)
- [x] UMAP mode: dim non-cluster sprites (alpha 0.12) instead of hide
- [x] Other modes: hide non-visible sprites (alpha 0) + relayout
- [x] Detail panel (right-edge tab toggle)
- [x] Single image PNG download (from detail panel)
- [x] Image Stack PNG download (composite all visible images)
- [x] Stats modal (collection info + cluster distribution)
- [x] Rose Pine Dawn theme with Inter font
- [x] Timeline view with time indicator bar
- [x] Color view sorted by average hue
- [x] Carousel navigates only filtered images
- [x] Pan/zoom/pinch via pixi-viewport
- [x] Dummy data generator (generate_data.py)
- [x] Cluster labels floating above groups
- [x] Git version control initialized

## What's Left 🔧
- [ ] Python backend pipeline (CLIP → UMAP → K-means → Atlas)
- [ ] Real image dataset support (not dummy shapes)
- [ ] Search/filter by filename
- [ ] Minimap overlay
- [ ] 3D scatter option
- [ ] GitHub Pages deployment guide

## Known Issues
- Color computation takes ~3-5s for 50K images on load
- Cluster labels may overlap when zoomed out very far
- No lazy loading of atlas textures (all 13 loaded at boot)
- Vite chunk size warning (pixi.js is 838KB)

## Design Decisions History
1. Started with NeoBrutalist theme → switched to Rose Pine Dawn
2. Started with HTML overlays → switched to PixiJS canvas layouts
3. Clusters view removed (Grid + filters is equivalent)
4. Detail panel: auto-open → toggle button → right-edge tab
5. Hotspot click: dim (0.12) in UMAP, full hide + relayout in other modes
6. Carousel: all-images → filtered-images-only navigation
7. Old `frontend/` deleted — only `frontend-pixi/` remains
8. ImageStack view added then removed; download feature kept
9. Filters moved from left column to top-right horizontal bar
10. Hotspots hidden in timeline/carousel modes

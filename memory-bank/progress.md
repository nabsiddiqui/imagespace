# ImageSpace — Progress

## What Works ✅
- [x] PixiJS WebGL rendering of 50K sprites with atlas textures
- [x] 5 view modes: UMAP, Grid, Color, Timeline, Carousel
- [x] Animated transitions between view modes (lerp 0.08)
- [x] Spatial hash hover detection + tooltip
- [x] K-means clustering (10 clusters, 25 iterations, client-side)
- [x] Hotspot floating cards (left side, 8 cards, click to filter)
- [x] CSV metadata loading + dropdown filters (top-left)
- [x] Combined filtering (hotspot ∩ CSV filters)
- [x] Detail panel (right side, toggle button)
- [x] Stats modal (collection info + cluster distribution)
- [x] Rose Pine Dawn theme with Inter font
- [x] Timeline view with time indicator bar
- [x] Color view sorted by average hue
- [x] Carousel navigates only filtered images
- [x] Pan/zoom/pinch via pixi-viewport
- [x] Dummy data generator (generate_data.py)
- [x] Cluster labels floating above groups in canvas views

## What's Left 🔧
- [ ] Python backend pipeline (CLIP → UMAP → K-means → Atlas)
- [ ] Real image dataset support (not dummy shapes)
- [ ] Search/filter by filename
- [ ] Export/share filtered views
- [ ] Minimap overlay
- [ ] 3D scatter option
- [ ] GitHub Pages deployment guide

## Known Issues
- Color computation takes ~3-5s for 50K images on load
- Cluster labels may overlap when zoomed out very far
- No lazy loading of atlas textures (all 13 loaded at boot)
- Vite chunk size warning (pixi.js is 838KB)

## Design Decisions History
1. Started with NeoBrutalist theme → switched to Rose Pine Dawn (user preference)
2. Started with HTML overlays for Clusters/Color → switched to PixiJS canvas layouts
3. Clusters view removed (Grid + filters is equivalent)
4. Detail panel changed from auto-open to toggle-based
5. Hotspot click changed from dim (alpha 0.12) to full hide (alpha 0) + relayout
6. Carousel changed from all-images to filtered-images-only navigation
7. Old `frontend/` (non-pixi) deleted — only `frontend-pixi/` remains

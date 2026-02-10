# ImageSpace — Active Context

## Current State (End of Session 3)
Project is **feature-complete and performance-optimized**. Full pipeline e2e test completed with measured timings. Pushed to GitHub.

### What Was Done (Session 3)
1. **Cursor re-fix** — Properties panel buttons (Properties, Reset All, Close) had cursor issue again. Applied inline `style={{ cursor: 'pointer' }}` for max specificity.
2. **Full pipeline e2e test** — Ran `imagespace.py` from scratch on 49,585 WikiArt images. Total: **2441.6s (40.7 min)** on Apple M-series CPU. All 9 stages completed successfully.
3. **README updated** — Replaced estimated timings with actual measured benchmarks. Added CPU expectations table for different hardware.
4. **Memory bank updated** — All timing data replaced with measured values.

### What Was Done (Session 2 — Comprehensive Summary)
1. **Server setup** — Built + served on http://localhost:5174 (Vite 5 build, Python HTTP server with absolute `-d` path)
2. **Detail panel fixes** — Canvas-based thumbnail (drawImage from atlas), auto-open on click, pointer cursor on hover
3. **Pipeline features** — Added `compute_image_features()` to imagespace.py: brightness (BT.601), complexity (Shannon entropy), edge density (Sobel), all 0-100. Added outlier scores (mean k-NN distance). Added cluster confidence (HDBSCAN probabilities).
4. **Standalone feature script** — Created `scripts/add_features.py` for adding features to existing datasets. Ran on WikiArt (~80s for 49,585 images).
5. **Range slider filters** — `rangeFilters` state, `handleRangeChange` callback, `continuousFilterOptions` useMemo, integrated with `computeVisibleSet` (5th parameter). Dual-handle sliders as inline bubble cards in right column.
6. **UI polish** — Multiple iterations on Properties button placement, zoom levels (removed multiplier), timeline centering (center on content), logo simplification (removed subtitle), detail panel reorder (Image→Thumbnail→Similar→Metadata), cursor fixes.
7. **Performance optimization (adversarial critic loop)**:
   - Float32Array pre-parsing for numeric metadata columns
   - Deferred `computeAvgColors` as fire-and-forget async (100ms delay, 8K batch)
   - `requestAnimationFrame` throttle on `handleRangeChange`
   - `DetailThumb` React.memo component (replaced createRef+setTimeout IIFE)
   - Offscreen canvas cache for minimap dots
   - Squared distance in hover (replaced Math.hypot)
8. **README** — Comprehensive rewrite with architecture diagram, data file table, optimization details, scripts table
9. **GitHub push** — Commit 299fc94

### Metadata CSV Columns (WikiArt, 15 columns)
`id, filename, cluster, timestamp, dominant_color (12 unique), artist (1,092), style (27), title (47,121), width, height, brightness (0-100), complexity (0-100), edge_density (0-100), outlier_score (0-100), cluster_confidence (0-100)`

## Active View Modes
- **t-SNE** — Visual similarity layout (openTSNE FFT-accelerated)
- **Grid** — Square grid layout
- **Color** — Sorted by average hue (requires colors computed in background)
- **Timeline** — Sorted by extracted year/timestamp

## Key Interactions
- **Range sliders** (right-side bubble cards, toggle via "Properties" button): brightness, complexity, edge density, uniqueness, cluster fit. Each card individually styled. Shifts layout when detail panel opens.
- **Click image** → detail panel auto-opens with canvas thumbnail, similar images, metadata
- **Hover image** → pointer cursor, scale 1.5x, gold tint (0xea9d34), tooltip
- **Hotspot cards** (left column): HDBSCAN clusters with thumbnails. Click to filter + zoom.
- **CSV Filters** (top-right dropdowns): Multi-select checkboxes, additive/union. ≤200 unique values shown.
- **Clear buttons**: Per-bar reset for range sliders, per-column and global clear for all filters.

## Future Considerations
- Export filtered set (CSV download of visible images)
- Bookmark/favorites with localStorage
- Keyboard shortcuts (arrows, Esc, 1-5 for views)
- Permalink/share state via URL hash
- Architecture: split App.jsx monolith (~1921 lines)
- GitHub Pages deployment guide
- Carousel view mode (exists in code as 'clusters' case, not currently in UI tabs)

## Key Rules
- **NEVER open Simple Browser** (destroys user's memory/context)
- Use absolute paths for Python HTTP server `-d` flag
- Use `npx vite build` (not `vite build`) for local Vite 5
- Server port: 5174

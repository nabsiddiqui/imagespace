# ImageSpace — System Patterns

## Architecture
```
frontend-pixi/
  src/App.jsx       — Single-file React component (~1200 lines)
  src/main.jsx      — React entry with ErrorBoundary
  src/index.css     — Tailwind imports + custom component classes
  public/data/      — Binary data, atlases, manifest, metadata CSV
  dist/             — Vite build output (served statically)
scripts/
  generate_data.py  — Generates 50K dummy images + binary layout
memory-bank/        — Documentation files (this system)
```

## Data Format
- **data.bin**: 16 bytes per image (float32 x, float32 y, uint16 atlasIdx, uint16 u, uint16 v, uint16 padding)
- **atlas_N.jpg**: 4096×4096 JPEG, 64px thumbnail cells, ~3900 images per atlas
- **manifest.json**: `{ count, atlasCount, thumbSize }`
- **metadata.csv**: `id,shape,color,category,decade,timestamp` — any columns work; 'id' and 'timestamp' are special

## Key Patterns

### Dynamic Imports
PixiJS and pixi-viewport are dynamically imported to avoid module-level crashes:
```js
const PIXI = await import('pixi.js');
const { Viewport } = await import('pixi-viewport');
```

### Spatial Hash for Hover
Grid-based spatial hash (`SPATIAL_CELL_SIZE=120`) for O(1) nearest-neighbor hover detection across 50K sprites.

### Unified Filter System
`computeVisibleSet(hotspotId, csvFilters, hotspots, metadata)` → `Set<id>` or `null`
- Hotspot filter: set of cluster member IDs
- CSV filters: `{ col: Set<values> }` — multi-select checkboxes per column
- CSV values union across all selected values (additive)
- Hotspot intersects with CSV union
- `computeLayout(allPoints, mode, visibleSet)`: in embedding modes (UMAP/t-SNE) dims non-visible (alpha 0.12), in other modes hides them (alpha 0)

### UI Layout
- **Top-left**: Logo + hotspot cards (larger 240px, hidden in carousel only)
- **Top-right**: View tabs + checkbox filter bar
- **Bottom-left**: View description card
- **Bottom-right**: Zoom controls only (stats card removed)
- **Right edge**: Vertical tab toggle with "Panel" label for detail panel
- **Detail panel**: Right-slide panel with thumbnail, metadata
- **Timeline**: Bottom bar with interactive slider (offset from hotspot column)

### Viewport Fitting
- Custom content-bounds fitting (replaces viewport.fit() which fits world bounds)
- Computes minX/maxX/minY/maxY from visible sprite targets
- Used in relayout, handleFitAll, and initial boot

### Timeline Scrolling
- Wheel events intercepted in timeline mode → horizontal pan instead of zoom
- Interactive range slider for scrubbing through time
- Time indicator shows current date from viewport center position

### Version Control
- Git repo at `modern-pixplot/`, `.gitignore` excludes node_modules, dist, .DS_Store
- 6+ commits on master branch

### View Mode Layout
`computeLayout(points, mode, visibleSet)`:
- UMAP/t-SNE: restore original x,y (same embedding, placeholder)
- Grid: square grid, `sqrt(n)` columns
- Color: sort by avgHue then grid
- Timeline: sort by timestamp then wide grid
- Clusters case exists but view mode removed from UI

### Cluster Labels
Floating React labels track world→screen coordinates via viewport.toScreen() in ticker, positioned above each cluster group.

### Timeline Indicator
Bottom bar shows current date/time from viewport center position mapped to timestamp range via linear interpolation.

## Color Palette (Rose Pine Dawn)
- rp-base: #faf4ed, rp-surface: #fffaf3, rp-text: #575279
- rp-pine: #286983, rp-foam: #56949f, rp-gold: #ea9d34
- rp-love: #b4637a, rp-iris: #907aa9

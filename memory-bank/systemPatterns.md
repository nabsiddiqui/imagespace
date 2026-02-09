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
- CSV filters: intersection of all active column→value matches
- Both combine via intersection
- `computeLayout(allPoints, mode, visibleSet)`: in UMAP mode dims non-visible (alpha 0.12), in other modes hides them (alpha 0)

### UI Layout
- **Top-left**: Logo + hotspot cards (hidden in timeline/carousel)
- **Top-right**: View tabs + compact filter bar (horizontal dropdowns)
- **Bottom-left**: View description card
- **Bottom-right**: Zoom controls + stats card
- **Right edge**: Vertical tab toggle for detail panel (centered vertically)
- **Detail panel**: Right-slide panel with thumbnail, metadata, download buttons

### Download Features
- Single image PNG: extracts from atlas, downloads 64px thumbnail
- Image Stack PNG: composites all visible images onto 256px canvas with auto-opacity

### Version Control
- Git repo at `modern-pixplot/`, `.gitignore` excludes node_modules, dist, .DS_Store

### View Mode Layout
`computeLayout(points, mode, visibleSet)`:
- UMAP: restore original x,y
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

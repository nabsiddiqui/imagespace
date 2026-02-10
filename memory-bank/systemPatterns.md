# ImageSpace — System Patterns

## Architecture
```
frontend-pixi/
  src/
    App.jsx             — Main app (~1921 lines, monolith)
    main.jsx            — React entry with ErrorBoundary
    index.css           — Tailwind imports + custom component classes (.rp-card)
    TestApp.jsx         — Unused test component
  public/data/          — Generated output (gitignored)
    data.bin            — Binary layout (24 bytes/image, v2)
    manifest.json       — {count, atlasCount, thumbSize, bytesPerImage:24, version:2, atlasFormat:'webp'}
    metadata.csv        — Per-image metadata with computed features
    neighbors.bin       — Binary k-NN indices + distances
    cluster_labels.json — CLIP-generated semantic cluster labels
    atlas_0..N.webp     — WebP atlas textures (4096×4096)
    embeddings.npy      — Cached CLIP embeddings
  dist/                 — Vite build output (served statically)
scripts/
  imagespace.py         — Main pipeline (~1057 lines): ONNX→PCA→openTSNE→HDBSCAN→k-NN→features→WebP
  add_features.py       — Standalone feature extraction from existing atlases (~223 lines)
  reprocess_layout.py   — Re-run t-SNE/HDBSCAN without regenerating atlases
  generate_data.py      — Generates dummy data for testing
memory-bank/            — Documentation files (this system)
```

## Data Formats

### data.bin (Binary, v2 — 24 bytes/image)
```
float32 umapX, umapY       # Layout coordinates (first set, legacy)
float32 tsneX, tsneY       # t-SNE coordinates
uint16  atlas_index         # Which atlas texture (0-48)
uint16  u, v                # Pixel offset in atlas
uint16  cluster_id          # HDBSCAN cluster assignment
```

### neighbors.bin
```
uint32 count                # Number of images
uint32 k                    # Neighbors per image
[count × k pairs]:
  uint32 neighbor_id
  float32 distance
```

### manifest.json
```json
{ "count": 49585, "atlasCount": 49, "thumbSize": 128, "atlasSize": 4096,
  "atlasFormat": "webp", "bytesPerImage": 24, "version": 2 }
```

### metadata.csv (15 columns for WikiArt)
`id, filename, cluster, timestamp, dominant_color, artist, style, title, width, height, brightness, complexity, edge_density, outlier_score, cluster_confidence`

## Key Patterns

### Component Architecture
- **App.jsx** — Monolith React component (~1921 lines); all state, effects, and rendering in one file
- **ImageSpaceLogo** — Standalone SVG component (Rose Pine Dawn themed)
- **DetailThumb** — `React.memo` component with `useRef` + `useEffect` for canvas-based atlas crop rendering. Accepts `point`, `thumbSize`, `atlasFormat` props. Replaces previous `React.createRef()` + `setTimeout()` IIFE anti-pattern.
- **NeighborThumb** — `useCallback`-wrapped inline component for k-NN thumbnails in detail panel

### Dynamic Imports
PixiJS and pixi-viewport are dynamically imported to avoid module-level crashes:
```js
const [PIXI, { Viewport }] = await Promise.all([
  import('pixi.js'), import('pixi-viewport')
]);
```

### Spatial Hash for Hover
Grid-based spatial hash (`SPATIAL_CELL_SIZE=120`) for O(1) nearest-neighbor hover detection. Rebuilt once after animation completes (not per-frame). Uses squared distance comparison (no Math.hypot).

### Unified Filter System
`computeVisibleSet(hotspotId, csvFilters, hotspots, metadata, rangeFilters)` → `Set<id>` or `null`
- **Hotspot filter**: Set of cluster member IDs
- **CSV filters**: `{ col: Set<values> }` — multi-select dropdowns per column, additive/union
- **Range filters**: `{ col: [min, max] }` — uses pre-parsed `Float32Array` columns (no parseFloat in hot loop)
- **Intersection**: hotspot ∩ CSV union ∩ range intersection
- All 5 callsites pass rangeFilters: switchView, handleHotspot, handleCSVFilter, handleRangeChange, handleClearAll

### Pre-parsed Numeric Columns
At metadata load time, known numeric columns (`brightness`, `complexity`, `edge_density`, `outlier_score`, `cluster_confidence`, `width`, `height`) are pre-parsed into `Float32Array` objects stored in `metadata.numericCols`. This eliminates `parseFloat()` calls in the `computeVisibleSet` hot loop.

### Range Slider Throttling
`handleRangeChange` uses `requestAnimationFrame` throttling via a ref (`rangePendingRef`). On each slider move, the previous pending frame is cancelled. This prevents redundant `computeVisibleSet` + `relayout` calls during continuous slider drag.

### Deferred Color Computation
`computeAvgColors` runs as a fire-and-forget async IIFE after `setLoading(false)`. Uses 8K batch size with `setTimeout(r, 0)` yields. The `colorsReady` state tracks completion. Minimap data is built after colors complete (inside the same async IIFE).

### Minimap Caching
Minimap dots are rendered to an offscreen canvas once (cached by identity on `md.dots`). Per-frame rendering only blits the cached image and overlays the viewport rectangle. Saves ~5000 `fillStyle`/`fillRect` calls per 200ms tick.

### Animation System
- `movingSet` (Set<id>) tracks only currently-animating sprites
- Ticker only iterates `movingSet`, not all 50K points
- Exponential decay: `factor = 1 - Math.pow(0.92, dt)` for frame-rate-independent smoothing
- Threshold: squared distance < 0.25 → snap to target, remove from movingSet
- `_spatialDirty` flag triggers hash rebuild after all animations complete

### View Mode Layout
`computeLayout(allPoints, mode, visibleSet, thumbSize)`:
- **t-SNE**: Restore `tsneX/tsneY` positions. Dim non-visible (alpha 0.12) but keep all visible.
- **Grid**: Square grid, `sqrt(n)` columns, 1.4× thumb spacing
- **Color**: Sort by `avgHue` → `avgLum`, arrange in grid (requires `colorsReady`)
- **Timeline**: Sort by timestamp, vertical columns (`cols = sqrt(n)/2`)
- **Clusters**: Group by cluster ID, vertical stack with gap
- Visibility: In stable layouts (t-SNE), non-visible points are dimmed. In other modes, hidden (alpha 0).

### UI Layout
- **Top-left**: Logo + "ImageSpace" text
- **Left column** (when > sm): Hotspot cards (scrollable, 240px wide)
- **Top-right**: View mode tabs + categorical filter bar (dropdowns with search, multi-select checkboxes)
- **Right column**: Properties toggle button → collapsible range slider bubble cards (brightness, complexity, edge density, uniqueness, cluster fit) stacked with gap-2
- **Bottom-left**: View mode description card
- **Bottom-right**: Zoom controls (no stats card)
- **Right edge**: Detail panel (slide-in, toggled via vertical tab)
- **Bottom**: Timeline bar (in timeline mode only)
- **Bottom-left corner**: Minimap canvas (in t-SNE mode only)

### Detail Panel Order
1. Image label (#N)
2. Canvas-based thumbnail (256px, from atlas via DetailThumb component)
3. Similar Images (k-NN neighbors, clickable to fly-to)
4. Metadata table (all CSV columns)

### Viewport Fitting
Custom content-bounds fitting (replaces viewport.fit() which fits world bounds). Computes minX/maxX/minY/maxY from visible sprite targets. No zoom multiplier (uniform fit-to-content). Centers on content centroid.

### Timeline
- Wheel events intercepted in timeline mode → horizontal pan
- Interactive dual-handle slider for time range filtering
- Time indicator shows current date from viewport center position
- Time filter dimming: only runs when slider values actually change (keyed check in ticker)

### Cluster Labels
Floating React `<div>` labels track world→screen coordinates via `viewport.toScreen()` in ticker (100ms throttle). Positioned over cluster centroids. CLIP-generated labels loaded from `cluster_labels.json`.

## Color Palette (Rose Pine Dawn)
- rp-base: #faf4ed, rp-surface: #fffaf3, rp-text: #575279
- rp-pine: #286983, rp-foam: #56949f, rp-gold: #ea9d34
- rp-love: #b4637a (accents), rp-iris: #907aa9
- rp-hlLow: #f4ede8, rp-hlMed: #dfdad9, rp-hlHigh: #cecacd
- rp-muted: #9893a5, rp-subtle: #797593

## Git
- Repo: https://github.com/nabsiddiqui/imagespace.git
- Branch: master
- Latest commit: 299fc94 (performance optimizations + README)

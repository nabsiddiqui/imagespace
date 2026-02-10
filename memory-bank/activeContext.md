# ImageSpace — Active Context

## Current State: PROJECT COMPLETE
Project is **feature-complete, performance-optimized, and fully tested**. All code pushed to GitHub. No active work items.

### Session History
- **Session 1**: Initial scaffolding, PixiJS setup, atlas loader, basic t-SNE view
- **Session 2**: Detail panel, pipeline features (brightness/complexity/edge density/outlier/confidence), range sliders, hotspot cards, CSV filters, timeline/color/grid views, k-NN neighbors, cluster labels, minimap, performance optimization (adversarial critic loop — 3 rounds), comprehensive README rewrite
- **Session 3**: Cursor fixes (`cursor-pointer` class + `pointer-events-none` on children), full pipeline e2e test (40.7 min measured), timing benchmarks in README, removed bottom-left view mode description label, zoom controls repositioned to bottom-right with `justify-end`

### Latest Changes (Session 3)
1. **Cursor fix v2** — Properties button uses `cursor-pointer` on `<button>`, `pointer-events-none` on `<SlidersHorizontal>` icon + `<span>` text. Same pattern for Reset All and Close buttons.
2. **Full pipeline e2e test** — 49,585 WikiArt images processed from scratch. **Total: 2441.6s (40.7 min)** on Apple M-series CPU.
3. **Removed view mode description** — Bottom-left "Visual similarity layout" card removed (redundant with tab buttons). Changed bottom bar from `justify-between` to `justify-end`.
4. **README + memory bank** — Updated with measured benchmarks and CPU expectation table.

### Metadata CSV Columns (WikiArt, 15 columns)
`id, filename, cluster, timestamp, dominant_color (12 unique), artist (1,092), style (27), title (47,121), width, height, brightness (0-100), complexity (0-100), edge_density (0-100), outlier_score (0-100), cluster_confidence (0-100)`

## Quick Reference for Future LLM Sessions

### How to Build & Serve
```bash
cd frontend-pixi && npx vite build          # MUST use npx (local Vite 5, not global v7)
python3 -m http.server 5174 -d /Users/nabeel/Documents/ImageSpace/frontend-pixi/dist  # MUST use absolute path
```

### How to Run Pipeline
```bash
python3 scripts/imagespace.py /Users/nabeel/Documents/wikiart/images \
  -o frontend-pixi/public/data/ \
  --metadata /Users/nabeel/Documents/wikiart/metadata.csv \
  --thumb-size 128 --quality 85 --atlas-size 4096
```

### Key Files to Know
- **`frontend-pixi/src/App.jsx`** (~1915 lines) — Monolith React component, ALL viewer logic
- **`scripts/imagespace.py`** (~1057 lines) — 9-stage pipeline
- **`scripts/add_features.py`** (~223 lines) — Standalone feature extraction
- **`frontend-pixi/src/index.css`** — Tailwind imports + `.rp-card` class

### Critical Rules
- **NEVER open Simple Browser** (destroys user's memory/context)
- Use `npx vite build` (not `vite build`) — local Vite 5 vs global v7
- Python HTTP server needs `-d /absolute/path` (relative paths fail from background terminals)
- Server port: **5174**
- WikiArt images at `/Users/nabeel/Documents/wikiart/images` (49,585 files)
- WikiArt metadata at `/Users/nabeel/Documents/wikiart/metadata.csv`

### View Modes
- **t-SNE** — Visual similarity layout (default)
- **Grid** — Square grid
- **Color** — Sorted by hue (requires background color computation)
- **Timeline** — Sorted by extracted year

### UI Layout
- **Top-left**: Logo + "ImageSpace"
- **Left column**: HDBSCAN hotspot cards (click to filter + zoom)
- **Top-right**: View mode tabs + categorical filter dropdowns
- **Right column**: Properties toggle → range slider cards
- **Bottom-right**: Zoom controls (+/-/fit)
- **Bottom**: Timeline bar (timeline mode only)
- **Bottom-left**: Minimap (t-SNE mode only)
- **Right edge**: Detail panel (slide-in on image click)

## Future Enhancement Ideas (Intentionally Deferred)
- Export filtered set (CSV download of visible images)
- Bookmark/favorites with localStorage
- Keyboard shortcuts (arrows, Esc, 1-5 for views)
- Permalink/share state via URL hash
- Split App.jsx monolith (~1915 lines)
- GitHub Pages deployment guide
- Carousel view mode (code exists as 'clusters' case, not in UI tabs)

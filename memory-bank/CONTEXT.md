# ImageSpace — Context

## Core Argument / Goal
A modern, static-site replacement for Yale DHLab's PixPlot: visualize 50K+ image collections as interactive 2D scatter plots using CLIP embeddings, t-SNE, and HDBSCAN clustering. Deployable to GitHub Pages with no backend required for viewing.

Built for "ImageSpace: A Modern Approach to Image Collection Visualization" in the *Computational Approaches to Art* special issue of *Computational Humanities Research*.

## Research Question / Problem
How can large image datasets (50K+) be explored visually without requiring server infrastructure, backend APIs, or complex installation for the viewer?

## Structure
- **Python pipeline** (`scripts/imagespace.py`, ~1057 lines): 9 stages from raw images → WebP atlas textures + binary layout + metadata CSV + k-NN neighbors + cluster labels
- **React/PixiJS viewer** (`image_space/src/App.jsx`, ~1930 lines): Static site loading binary data + atlases; GPU-rendered sprites with 4 view modes, filtering, detail panel, minimap, cluster hotspots

## Current Status
- **Current Working File:** None — project is feature-complete
- **What's Complete:** Full pipeline (ONNX CLIP → PCA → openTSNE → HDBSCAN → k-NN → brightness/complexity/edge density/outlier/cluster confidence features → WebP atlases); 4 view modes (t-SNE, Grid, Color, Timeline); all filtering (hotspot cluster cards, CSV categorical dropdowns, range sliders); detail panel with canvas crop + k-NN similar images; minimap (offscreen canvas cache); floating cluster labels; performance-optimized to 50K sprites at ~30 FPS; Google Colab notebook; generate_data.py for test data
- **What's In Progress:** Nothing
- **What Remains:** Optional enhancements only (see prd.json)
- **Blockers/Dependencies:** None

## Bug Fixes Applied (Session 4)
1. **Minimap color** — `p.avgColor` (never set) replaced with `p.avgR ?? 150 / p.avgG ?? 150 / p.avgB ?? 150` — minimap now shows real image colors after background computation
2. **Timeline filter reset** — Added `else` branch to restore sprite alpha=1 when time slider is reset to full range (sprites were permanently dimmed before)
3. **Stats modal thumb size** — Now uses `thumbSizeRef.current` (from manifest) instead of hardcoded `THUMB_SIZE=64` constant

## Bug Fixes Applied (Session 5)
1. **GitHub Pages subpath deployment** — All hardcoded `/data/...` paths replaced with `${BASE}data/...` where `const BASE = import.meta.env.BASE_URL`. Vite `base` config reads from `VITE_BASE_PATH` env var (default `/`). Deploy to `username.github.io/repo-name/` by building with `VITE_BASE_PATH=/repo-name/ npx vite build`.
2. **`cacheBust` removed** — `?v=${Date.now()}` was appended to all data fetches, forcing re-downloads on every page load. Removed. Static data doesn't need this.
3. **`embeddings.npy` moved out of `public/data/`** — Was 97MB Python artifact in `public/data/` that Vite would copy into `dist/`. Moved to `scripts/embeddings.npy`.
4. **Old JPG atlases deleted** — 87MB of `atlas_*.jpg` files (old pipeline output) were in `public/data/` but unused (manifest uses `webp`). Deleted.
5. **`data.json` deleted** — 6.6MB stale v1 format file (old JSON array, replaced by `data.bin`). Deleted from `public/data/`.
6. **`cp -r public/data/* dist/data/` removed from README** — This step is redundant; Vite automatically copies `public/` to `dist/` during build.
7. **Deployment section added to README** — Documents both simple upload and GitHub Actions workflows.
8. **Build output renamed `dist/` → `output/`** — Set `outDir: 'output'` in vite.config.js. `output/` is the folder users upload to any static host.

## Editorial Mode / Genre
Software engineering project (academic tool for computational humanities research)

## Open Questions
- Should the pipeline be distributed as a pip-installable package?
- Should the CSV parser be upgraded to handle RFC 4180 quoted fields (commas in metadata values)?

## Key Technical Reference

### Build & Serve
```bash
cd image_space
npx vite build          # MUST use npx (local Vite 5, not global v7)
python3 -m http.server 5174 -d /absolute/path/to/image_space/output
```

### Pipeline
```bash
python3 scripts/imagespace.py /path/to/images/ \
  -o image_space/public/data/ \
  --metadata /path/to/metadata.csv \
  --thumb-size 128 --quality 85
```

### Key Files
- `image_space/src/App.jsx` — monolith React component (~1930 lines), ALL viewer logic
- `scripts/imagespace.py` — 9-stage pipeline (~1057 lines)
- `scripts/generate_data.py` — generates 50K dummy images for testing

### Critical Rules
- Use `npx vite build` not `vite build` (local Vite 5 vs global v7)
- Python HTTP server needs `-d /absolute/path` (relative paths fail from background terminals)
- WikiArt images at `/Users/nabeel/Documents/wikiart/images` (49,585 files)
- Folder renamed `frontend-pixi` → `image_space`; pipeline data dir is `image_space/public/data/`; build output is `image_space/output/` (Vite `outDir: 'output'`)

## Next Step
None (feature-complete). Address items in prd.json if/when desired.

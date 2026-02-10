# ImageSpace — Tech Context

## Frontend Stack
- **React 18.2** — UI framework (single monolith component)
- **PixiJS 8.16.0** — WebGL sprite rendering (50K sprites)
- **pixi-viewport 6.0.3** — Pan/zoom/pinch viewport with decelerate
- **Vite 5.4.21** — Build tool (LOCAL version via `npx`, not global v7)
- **Tailwind CSS 3.4.1** — Utility-first styling (Rose Pine Dawn palette)
- **lucide-react** — Icon library (Database, X, Layers, Grid, Eye, ZoomIn, ZoomOut, Maximize2, Flame, PanelLeftClose, PanelLeft, PanelRight, Palette, Info, ChevronLeft, ChevronRight, Filter, ChevronDown, Clock, SlidersHorizontal)
- **PostCSS + autoprefixer** — CSS processing

## Pipeline Stack
- **Python 3.14** (homebrew) — requires `--break-system-packages` for pip
- **ONNX Runtime 1.24.1** — CLIP ViT-B/32 inference (primary, fastest)
- **huggingface_hub** — ONNX model download (Xenova/clip-vit-base-patch32)
- **openTSNE 1.0.4** — FFT-accelerated t-SNE (10-20x faster than sklearn)
- **HDBSCAN 0.8.41** — Density-based clustering
- **scikit-learn 1.8.0** — PCA, MiniBatchKMeans fallback
- **torch 2.10.0 + transformers 5.1.0** — PyTorch CLIP fallback
- **Pillow** — Image processing, atlas generation (WebP)
- **numpy 2.4.2, scipy 1.17.0** — Numerical computation

## Build & Serve
```bash
cd frontend-pixi
npx vite build                    # → dist/ (~2.5s, uses local Vite 5)
# Copy data into dist (Vite copies public/ but only at build time)
cp -r public/data/* dist/data/
# Serve — MUST use absolute path for -d flag
python3 -m http.server 5174 -d /absolute/path/to/frontend-pixi/dist
```

**CRITICAL**: Use `npx vite build` (not `vite build`) to ensure local Vite 5 is used, not any globally installed version.

## Pipeline Commands
```bash
# Full run
python3 scripts/imagespace.py /path/to/images/ -o frontend-pixi/public/data/ \
    --metadata /path/to/metadata.csv --thumb-size 128 --quality 85

# With caching (recommended)
python3 scripts/imagespace.py /path/to/images/ -o frontend-pixi/public/data/ \
    --metadata /path/to/metadata.csv --cache-dir frontend-pixi/public/data/ --thumb-size 128

# Relayout only (skip atlas + CLIP)
python3 scripts/imagespace.py /path/to/images/ -o frontend-pixi/public/data/ \
    --metadata /path/to/metadata.csv --cache-dir frontend-pixi/public/data/ --relayout --thumb-size 128

# Add features to existing data (standalone)
python3 scripts/add_features.py --data-dir frontend-pixi/public/data/ \
    --metadata frontend-pixi/public/data/metadata.csv
```

## File Structure
```
frontend-pixi/
  src/
    App.jsx             # Main app (~1921 lines, monolith)
    main.jsx            # React root + ErrorBoundary
    index.css           # Tailwind + custom classes (.rp-card)
    TestApp.jsx         # Unused test component
  public/data/          # Generated output (gitignored)
    data.bin            # Binary layout (24 bytes/image, v2)
    manifest.json       # Manifest with count, format, version
    metadata.csv        # 15 columns (WikiArt) with computed features
    neighbors.bin       # k-NN binary data
    cluster_labels.json # CLIP semantic labels
    embeddings.npy      # Cached CLIP embeddings
    atlas_0..48.webp    # 49 WebP atlas textures (4096×4096, 128px thumbs)
  dist/                 # Vite build output (served statically)
scripts/
  imagespace.py         # Main pipeline (~1057 lines)
  add_features.py       # Standalone feature extraction (~223 lines)
  reprocess_layout.py   # Re-run layout only
  generate_data.py      # Dummy data generator
ImageSpace_Colab.ipynb  # Google Colab notebook for GPU processing
memory-bank/            # Project documentation
```

## Binary Format v2 (24 bytes/image)
```
float32 umapX, umapY      # Layout coordinates (legacy/first set)
float32 tsneX, tsneY       # t-SNE coordinates
uint16 atlas_index         # Which atlas texture (0-48)
uint16 u, v                # Pixel offset in atlas
uint16 cluster_id          # HDBSCAN cluster assignment
```

## Key Technical Constraints
- Must work as static site (no server-side logic)
- 50K images rendered as PixiJS sprites (GPU-accelerated, ~870MB GPU memory for 49 atlases)
- WebP atlases for minimal bandwidth (~30% smaller than JPEG)
- Binary format for layout data (24 bytes/image vs ~200 bytes JSON)
- HDBSCAN clustering with cKDTree noise reassignment
- ONNX model cached at `~/.cache/imagespace/` (~350MB)
- CSV parser is simple split-based (doesn't handle RFC 4180 quoting)
- Both UMAP and t-SNE coordinate slots currently store t-SNE (redundant)

## Performance Optimizations (Viewer)

### Per-Frame (Ticker)
- Only animating sprites iterated (`movingSet`, not all 50K)
- Spatial hash rebuild deferred to animation end (not per-frame)
- UI state updates (FPS, zoom) throttled to 200ms
- Cluster label positions throttled to 100ms
- Time filter dimming only on value change (keyed check)

### Minimap
- Offscreen canvas cache for dot layer (built once, blit per frame)
- Only viewport rectangle redrawn per tick

### Filtering
- Pre-parsed `Float32Array` for numeric columns → no `parseFloat()` in hot loop
- `requestAnimationFrame` throttle on range slider changes
- Categorical filters: iterate rows once, build Set, intersect

### Hover
- Squared distance comparison (no `Math.hypot`)
- 9-cell neighborhood search in spatial hash

### Loading
- Parallel atlas loading (`Promise.all`)
- Parallel dynamic imports (PIXI + pixi-viewport)
- Deferred `computeAvgColors` (fire-and-forget async, 8K batch, non-blocking UI)
- `container.boundsArea` set explicitly to avoid O(n) bounds recalculation

### React
- `DetailThumb`: `React.memo` + `useRef` + `useEffect` (no createRef in render)
- `NeighborThumb`: `useCallback`-wrapped
- Throttled `setStats`, `setZoomLevel`, `setClusterLabels` to reduce re-renders

### Build
- WebP method=2 (fast encode for thumbnails)
- PCA 512→50d before t-SNE (faster, better separation)
- Annoy-based neighbor finding in openTSNE

## Development Notes
- **NEVER open Simple Browser** — destroys user's memory/context
- Use `npx vite build` (not `vite build`) to use local Vite 5, not global Vite 7
- Server must use `-d /absolute/path/to/dist` flag (relative paths fail from non-CWD terminals)
- metadata.csv is optional — app works without it (no filters)
- Pipeline auto-extracts years from filenames for art datasets (regex 1400-2030)
- ONNX model auto-downloads on first run (~350MB cached in `~/.cache/imagespace/`)
- Vite chunk size warning (pixi.js ~838KB) is expected and acceptable
- `colorsReady` state gates Color view mode availability

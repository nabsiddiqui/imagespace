# ImageSpace — Tech Context

## Frontend Stack
- **React 18.2** — UI framework
- **PixiJS 8.16.0** — WebGL sprite rendering
- **pixi-viewport 6.0.3** — Pan/zoom/pinch viewport
- **Vite 5.4.21** — Build tool (LOCAL version via npx, not global v7)
- **Tailwind CSS 3.4.1** — Utility-first styling (Rose Pine Dawn palette)
- **lucide-react** — Icon library
- **PostCSS + autoprefixer** — CSS processing

## Pipeline Stack
- **Python 3.14** (homebrew) — requires `--break-system-packages` for pip
- **ONNX Runtime 1.24.1** — CLIP inference (primary, fastest)
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
npx vite build                    # → dist/ (~3-4s)
python3 -m http.server 5174 -d dist  # static file server
```

## Pipeline
```bash
python3 scripts/imagespace.py /path/to/images frontend-pixi/public/data \
    --metadata /path/to/metadata.csv --thumb-size 64 --atlas-size 4096
```

## File Structure
```
frontend-pixi/
  src/
    App.jsx             # Main app (~1484 lines, monolith)
    main.jsx            # React root + ErrorBoundary
    index.css           # Tailwind + custom classes
  public/data/          # Generated output (gitignored)
    data.bin            # Binary layout (24 bytes/image, v2)
    manifest.json       # {count, atlasCount, thumbSize, bytesPerImage:24, version:2, atlasFormat:'webp'}
    metadata.csv        # Per-image metadata
    data.json           # Image paths JSON
    atlas_0..N.webp     # WebP atlas textures (4096×4096)
scripts/
  imagespace.py         # Main pipeline (ONNX→PCA→openTSNE→HDBSCAN→WebP)
  imagespace_old.py     # Backup of previous sklearn/PyTorch pipeline
  generate_data.py      # Dummy data generator (24-byte v2 format)
ImageSpace_Colab.ipynb  # Google Colab notebook for GPU processing
memory-bank/            # Project documentation
```

## Binary Format v2 (24 bytes/image)
```
float32 x, y          # Layout coordinates (t-SNE)
float32 tsneX, tsneY   # t-SNE coordinates (same as x,y currently)
uint16 atlas_index     # Which atlas texture
uint16 u, v            # Pixel offset in atlas
uint16 cluster_id      # HDBSCAN cluster assignment
```

## Key Technical Constraints
- Must work as static site (no server-side logic)
- 50K images rendered as PixiJS sprites (GPU-accelerated)
- WebP atlases for minimal bandwidth (~30% smaller than JPEG)
- Binary format for layout data (24 bytes/image vs ~200 bytes JSON)
- HDBSCAN clustering with cKDTree noise reassignment
- ONNX model cached at `~/.cache/imagespace/`

## Performance Optimizations
- Spatial hash rebuild deferred to animation end (not per-frame)
- Cluster labels + timeline setState throttled to 200ms
- Time filter dimming only on slider value change
- Parallel atlas loading (Promise.all)
- Parallel dynamic imports (PIXI + pixi-viewport)
- WebP method=2 (fast encode for thumbnails)
- PCA 512→50d before t-SNE

## Development Notes
- Use `npx vite build` (not `vite build`) to use local Vite 5, not global Vite 7
- Server must use `-d /absolute/path/to/dist` flag
- metadata.csv is optional — app works without it
- Pipeline auto-extracts years from filenames for art datasets (regex 1400-2030)
- ONNX model auto-downloads on first run (~350MB cached)

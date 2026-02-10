# ImageSpace

An interactive visualization tool for exploring large image collections through dimensionality reduction. ImageSpace processes image datasets using CLIP embeddings, t-SNE, and HDBSCAN clustering to create a navigable, browser-based visualization of 50K+ images.

Built as a modern alternative to [PixPlot](https://github.com/yaledhlab/pix-plot) for the article "ImageSpace: A Modern Approach to Image Collection Visualization" in the *Computational Approaches to Art* special issue of *Computational Humanities Research*.

## Features

### Pipeline
- **CLIP-based embeddings** — ONNX-optimized CLIP ViT-B/32 for semantic image understanding
- **t-SNE layout** — FFT-accelerated openTSNE with exaggeration for clear cluster separation
- **HDBSCAN clustering** — Unsupervised density-based clustering on PCA embeddings
- **k-NN neighbors** — Pre-computed nearest neighbors for similarity browsing
- **Image features** — Pipeline-computed brightness, complexity, edge density, outlier score, cluster confidence (0–100 scale)
- **Cluster labels** — Auto-generated CLIP-based semantic labels for each cluster
- **Atlas textures** — WebP sprite atlases (4096×4096, 128px thumbnails)
- **Static deployment** — Generates a self-contained static site (no server required)

### Viewer
- **Multiple view modes** — t-SNE scatter, grid, color sort, cluster grouping, and timeline
- **Categorical filters** — Filter by any metadata column (e.g., artist, style) with multi-select dropdowns
- **Continuous range sliders** — Filter by brightness, complexity, edge density, outlier score, cluster confidence
- **Hotspot navigation** — Click cluster hotspot cards to zoom into regions of visual similarity
- **Detail panel** — Click any image for canvas-based thumbnail, similar images, and full metadata
- **Minimap** — Cached offscreen-canvas minimap with viewport rectangle overlay (t-SNE view)
- **Cluster labels** — Floating semantic labels at cluster centroids
- **Timeline view** — Chronological layout with time range slider and date display
- **50K+ images** — Renders at ~30 FPS via PixiJS WebGL with sprite-only animation

## Architecture

```
Pipeline (Python)                    Viewer (React + PixiJS)
┌─────────────────┐                  ┌────────────────────────┐
│ Images          │──→ WebP Atlases──→ Atlas Textures         │
│ CLIP ONNX       │──→ Embeddings  ──→                        │
│ PCA + t-SNE     │──→ data.bin    ──→ Binary Layout (24B/img)│
│ HDBSCAN         │──→ manifest.json→ Manifest                │
│ k-NN (cosine)   │──→ neighbors.bin→ Similar Images          │
│ Image Features  │──→ metadata.csv──→ Filters + Sliders      │
│ Cluster Labels  │──→ cluster_labels.json → Floating Labels  │
└─────────────────┘                  └────────────────────────┘
```

### Binary Format (v2)

24 bytes per image: `float32 umapX, umapY, tsneX, tsneY` + `uint16 atlas, u, v, cluster`

### Data Files

| File | Purpose |
|------|---------|
| `manifest.json` | Atlas count, image count, thumb size, format, bytes per image |
| `data.bin` | Binary layout coordinates (24 bytes × N) |
| `atlas_*.webp` | Sprite atlas textures (4096×4096) |
| `metadata.csv` | Per-image metadata with computed features |
| `neighbors.bin` | Binary k-NN indices + distances |
| `cluster_labels.json` | CLIP-generated semantic cluster labels |
| `embeddings.npy` | Cached CLIP embeddings (optional, for re-runs) |

## Quick Start

### Requirements

- Python 3.10+
- Node.js 18+ (for building the viewer)

### Install Dependencies

```bash
pip install pillow numpy scikit-learn opentsne hdbscan onnxruntime scipy
cd frontend-pixi && npm install
```

### Run the Pipeline

```bash
python scripts/imagespace.py /path/to/images/ \
  -o frontend-pixi/public/data/ \
  --metadata /path/to/metadata.csv \
  --thumb-size 128 \
  --quality 85 \
  --min-cluster-size 50
```

**With embedding caching** (recommended for iteration):

```bash
python scripts/imagespace.py /path/to/images/ \
  -o frontend-pixi/public/data/ \
  --metadata /path/to/metadata.csv \
  --cache-dir frontend-pixi/public/data/ \
  --thumb-size 128
```

**Relayout mode** (skip atlas + CLIP, re-run t-SNE/HDBSCAN only):

```bash
python scripts/imagespace.py /path/to/images/ \
  -o frontend-pixi/public/data/ \
  --metadata /path/to/metadata.csv \
  --cache-dir frontend-pixi/public/data/ \
  --relayout --thumb-size 128
```

### Add Features to Existing Data

If you already have atlas data and want to add computed features (brightness, complexity, etc.):

```bash
python scripts/add_features.py \
  --data-dir frontend-pixi/public/data/ \
  --metadata frontend-pixi/public/data/metadata.csv
```

### Build & Serve

```bash
cd frontend-pixi
npx vite build
cp -r public/data/* dist/data/
cd dist && python3 -m http.server 5174
```

Open http://localhost:5174

> **Note**: Use `npx vite build` (not `vite build`) to ensure the local Vite 5 is used, not a global installation.

## Pipeline Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--output`, `-o` | required | Output directory |
| `--metadata` | none | External metadata CSV to merge (must have `filename` column) |
| `--thumb-size` | 64 | Thumbnail size in pixels (64 or 128 recommended) |
| `--atlas-size` | 4096 | Atlas texture dimensions |
| `--quality` | 80 | WebP compression quality (1-100) |
| `--min-cluster-size` | 50 | HDBSCAN minimum cluster size |
| `--tsne-perplexity` | 30 | t-SNE perplexity parameter |
| `--cache-dir` | none | Cache directory for CLIP embeddings (.npy) |
| `--relayout` | false | Skip atlas generation + CLIP extraction |
| `--gpu` | false | Enable GPU acceleration for CLIP |

## Performance

### Pipeline (CPU-only, Apple M3 Air, ~50K WikiArt images)

| Stage | Time | Notes |
|-------|------|-------|
| Atlas generation (128px) | ~7.5 min | 49 WebP atlases, ~110 img/s |
| CLIP embeddings | ~27 min | First run only; cached as .npy |
| PCA (512d → 50d) | ~0.1s | 69% variance retained |
| openTSNE | ~24s | FFT-accelerated, Annoy neighbors |
| Overlap removal | ~3s | Grid-snap to prevent overlapping |
| HDBSCAN | ~16s | On PCA embeddings, leaf selection |
| Image features | ~80s | Brightness, complexity, edge density |
| Metadata extraction | ~7 min | Dominant colors + timestamps |
| **Total (first run)** | **~45 min** | |
| **Total (cached embeddings)** | **~15 min** | |
| **Total (relayout only)** | **~45s** | |

### Viewer Optimizations

- **Per-frame**: Only animating sprites iterated (O(movingSet), not O(N))
- **Spatial hash**: Rebuilt once after animation ends for O(1) hover detection
- **Range filtering**: Pre-parsed `Float32Array` columns — no `parseFloat()` in hot loop
- **Range slider**: `requestAnimationFrame` debounced to prevent redundant relayouts
- **Minimap**: Offscreen canvas cache for dot layer — only viewport rect redrawn per frame
- **Hover**: Squared distance comparison (no `Math.hypot`)
- **Color computation**: Deferred async, 8K batch size, non-blocking UI
- **Detail thumbnail**: `React.memo` + `useEffect` (no createRef in render)

## Technology Stack

- **Pipeline**: Python, ONNX Runtime (CLIP ViT-B/32), openTSNE, HDBSCAN, scikit-learn, Pillow, scipy
- **Frontend**: React 18.2, PixiJS 8.16, pixi-viewport 6.0.3, Vite 5.4, Tailwind CSS 3.4, lucide-react
- **Design**: Rosé Pine Dawn color palette

## Metadata CSV Format

The external metadata CSV must have a `filename` column matching image filenames. All other columns are automatically available as filters in the viewer. Numeric columns (brightness, complexity, edge_density, outlier_score, cluster_confidence, width, height) get range slider filters; all others get categorical dropdowns.

```csv
filename,artist,style,title,brightness,complexity,edge_density
monet_water-lilies.jpg,Claude Monet,Impressionism,Water Lilies,72.3,45.1,38.7
picasso_guernica.jpg,Pablo Picasso,Cubism,Guernica,28.9,78.4,62.1
```

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/imagespace.py` | Main pipeline — atlas generation, CLIP, t-SNE, HDBSCAN, k-NN, features |
| `scripts/add_features.py` | Standalone script to add computed features to existing atlas data |
| `scripts/reprocess_layout.py` | Re-run t-SNE/HDBSCAN without regenerating atlases |

## Google Colab

For remote processing without a local GPU, use the included `ImageSpace_Colab.ipynb` notebook.

## License

MIT

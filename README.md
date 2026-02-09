# ImageSpace

An interactive visualization tool for exploring large image collections through dimensionality reduction. ImageSpace processes image datasets using CLIP embeddings, t-SNE, and HDBSCAN clustering to create a navigable, browser-based visualization.

Built as a modern alternative to [PixPlot](https://github.com/yaledhlab/pix-plot) for the article "ImageSpace: A Modern Approach to Image Collection Visualization" in the *Computational Approaches to Art* special issue of *Computational Humanities Research*.

## Features

- **CLIP-based embeddings** — Uses ONNX-optimized CLIP ViT-B/32 for semantic image understanding
- **t-SNE layout** — FFT-accelerated openTSNE with exaggeration for clear cluster separation  
- **HDBSCAN clustering** — Unsupervised density-based clustering on PCA embeddings
- **Multiple view modes** — t-SNE scatter, grid, cluster, color, carousel, and timeline views
- **Metadata filtering** — Filter by any categorical metadata column (e.g., style, artist)
- **Hotspot navigation** — Click cluster hotspots to explore regions of visual similarity
- **Detail panel** — Click any image for full metadata and high-resolution preview
- **Atlas textures** — Efficiently renders 50K+ images using WebP sprite atlases
- **Static deployment** — Generates a self-contained static site (no server required)

## Architecture

```
Pipeline (Python)                    Viewer (React + PixiJS)
┌─────────────┐                      ┌────────────────────┐
│ Images      │──→ WebP Atlases  ──→ │ Atlas Textures     │
│ CLIP ONNX   │──→ Embeddings   ──→ │                    │
│ PCA + t-SNE │──→ data.bin     ──→ │ Binary Layout      │
│ HDBSCAN     │──→ manifest.json──→ │ Manifest           │
│ Metadata    │──→ metadata.csv ──→ │ Filters + Details  │
└─────────────┘                      └────────────────────┘
```

### Binary Format (v2)

24 bytes per image: `float32 x, y, tsneX, tsneY` + `uint16 atlas, u, v, cluster`

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

### Build & Serve

```bash
cd frontend-pixi
npx vite build
cp -r public/data/* dist/data/
cd dist && python3 -m http.server 5174
```

Open http://localhost:5174

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

## Performance (CPU-only)

Benchmarked on Apple M3 Air with ~50K WikiArt images:

| Stage | Time | Notes |
|-------|------|-------|
| Atlas generation (128px) | ~7.5 min | 49 WebP atlases, ~110 img/s |
| CLIP embeddings | ~27 min | First run only; cached as .npy |
| PCA (512d → 50d) | ~0.1s | 69% variance retained |
| openTSNE | ~24s | FFT-accelerated, Annoy neighbors |
| Overlap removal | ~3s | Grid-snap to prevent overlapping |
| HDBSCAN | ~16s | On PCA embeddings, leaf selection |
| Metadata extraction | ~7 min | Dominant colors + timestamps |
| **Total (first run)** | **~45 min** | |
| **Total (cached embeddings)** | **~15 min** | |
| **Total (relayout only)** | **~45s** | |

## Technology Stack

- **Pipeline**: Python, ONNX Runtime (CLIP ViT-B/32), openTSNE, HDBSCAN, scikit-learn, Pillow
- **Frontend**: React 18, PixiJS 8, pixi-viewport, Vite 5, Tailwind CSS 3
- **Design**: Rosé Pine Dawn color palette

## Metadata CSV Format

The external metadata CSV must have a `filename` column matching image filenames. All other columns are automatically available as filters in the viewer.

```csv
filename,artist,style,title
monet_water-lilies.jpg,Claude Monet,Impressionism,Water Lilies
picasso_guernica.jpg,Pablo Picasso,Cubism,Guernica
```

## Google Colab

For remote processing without a local GPU, use the included `ImageSpace_Colab.ipynb` notebook.

## License

MIT

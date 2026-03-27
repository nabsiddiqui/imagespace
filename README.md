# ImageSpace

An interactive browser-based tool for exploring large image collections. Point it at a folder of images; it produces a navigable 2D scatter plot where similar images cluster together.

**The viewer is a static site — no backend, no server, no database.** Once processed, share the `output/` folder on GitHub Pages, any CDN, or a USB drive.

## What It Does

- Lays out images by visual similarity (CLIP embeddings → t-SNE)
- Clusters them automatically (HDBSCAN) and labels clusters with natural language (CLIP)
- Lets you switch between scatter, grid, color-sorted, and timeline views
- Filters by any metadata column (dropdowns) or computed feature (brightness, complexity, etc.)
- Shows similar images and full metadata in a slide-in detail panel
- Renders 50,000 images at ~30 FPS in a browser via PixiJS WebGL

## Quick Start

There are two paths depending on what you want to do.

### Option A — Try it immediately with test data (minimal deps)

This generates synthetic dummy data so you can try the viewer without processing real images. You only need Node.js and Python with Pillow.

**1. Install Node dependencies**

```bash
cd image_space
npm install
```

**2. Generate dummy data**

```bash
cd ..
pip install pillow
python3 scripts/generate_data.py
```

This creates 50,000 dummy images in `image_space/public/data/` (takes ~1-2 minutes).

**3. Build and open**

```bash
cd image_space
npx vite build
python3 -m http.server 5174 -d "$(pwd)/output"
```

Open http://localhost:5174

---

### Option B — Process your own images

**1. Install dependencies**

```bash
# Python pipeline
pip install pillow numpy scikit-learn opentsne hdbscan onnxruntime scipy

# Frontend
cd image_space && npm install && cd ..
```

> **GPU acceleration (optional):** Add `--gpu` to the pipeline command. On Apple Silicon, install `onnxruntime` normally — CoreML is auto-detected. For NVIDIA, install `onnxruntime-gpu` instead.

**2. Run the pipeline**

```bash
python3 scripts/imagespace.py /path/to/your/images/ \
  -o image_space/public/data/ \
  --thumb-size 128 \
  --quality 85
```

With optional external metadata (CSV must have a `filename` column):

```bash
python3 scripts/imagespace.py /path/to/your/images/ \
  -o image_space/public/data/ \
  --metadata /path/to/metadata.csv \
  --thumb-size 128 --quality 85
```

**3. Build and serve**

```bash
cd image_space
npx vite build
python3 -m http.server 5174 -d "$(pwd)/output"
```

Open http://localhost:5174

> **Note:** Use `npx vite build`, not `vite build`. This ensures the local Vite 5 (in `node_modules`) is used rather than any globally installed version.

---

### Iterate faster with caching

Once you've run the pipeline once, save time on subsequent runs:

```bash
# Cache embeddings — skip the CLIP stage on re-runs (~18 min saved for 50K images)
python3 scripts/imagespace.py /path/to/images/ \
  -o image_space/public/data/ \
  --cache-dir image_space/public/data/ \
  --thumb-size 128

# Relayout only — re-run t-SNE + HDBSCAN without redoing atlas or CLIP (~45s for 50K)
python3 scripts/imagespace.py /path/to/images/ \
  -o image_space/public/data/ \
  --cache-dir image_space/public/data/ \
  --relayout --thumb-size 128
```

---

## Architecture

```
Pipeline (Python)                    Viewer (React + PixiJS)
┌─────────────────┐                  ┌────────────────────────┐
│ Images          │──→ WebP Atlases──→ Atlas Textures         │
│ CLIP ONNX       │                                           │
│ PCA + t-SNE     │──→ data.bin    ──→ Binary Layout (24B/img)│
│ HDBSCAN         │──→ manifest.json→ Manifest                │
│ k-NN (cosine)   │──→ neighbors.bin→ Similar Images          │
│ Image Features  │──→ metadata.csv──→ Filters + Sliders      │
│ Cluster Labels  │──→ cluster_labels.json → Floating Labels  │
└─────────────────┘                  └────────────────────────┘
```

**Binary format (v2):** 24 bytes per image — `float32 tsneX, tsneY` (×2 for legacy) + `uint16 atlas, u, v, cluster`

## Pipeline Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--output`, `-o` | required | Output directory |
| `--metadata` | none | External metadata CSV to merge (must have `filename` column) |
| `--thumb-size` | 64 | Thumbnail size in pixels (64 or 128 recommended) |
| `--atlas-size` | 4096 | Atlas texture dimensions |
| `--quality` | 80 | WebP compression quality (1–100) |
| `--min-cluster-size` | 50 | HDBSCAN minimum cluster size |
| `--tsne-perplexity` | 30 | t-SNE perplexity |
| `--cache-dir` | none | Directory to cache CLIP embeddings (`.npy`) |
| `--relayout` | false | Skip atlas + CLIP, re-run t-SNE/HDBSCAN only |
| `--gpu` | false | Enable GPU acceleration for CLIP |

## Metadata Format

Your metadata CSV must have a `filename` column matching image filenames. All other columns become available as filters in the viewer automatically — categorical columns get dropdown checkboxes, numeric columns get range sliders.

```csv
filename,artist,style,title
monet_water-lilies.jpg,Claude Monet,Impressionism,Water Lilies
picasso_guernica.jpg,Pablo Picasso,Cubism,Guernica
```

The pipeline also computes and appends: `brightness`, `complexity`, `edge_density`, `outlier_score`, `cluster_confidence` (all 0–100 scale).

## Performance (50K images, Apple M-series CPU)

| Stage | Time |
|-------|------|
| Atlas generation (128px, q85) | 7.2 min |
| CLIP embeddings (ONNX, CPU-only) | 18.1 min |
| PCA + t-SNE + HDBSCAN | ~1 min |
| k-NN + features + metadata | ~15 min |
| **Total (first run)** | **~41 min** |
| **Total (cached embeddings)** | **~22 min** |
| **Total (relayout only)** | **~45 sec** |

CLIP embedding is the dominant cost. Cache it with `--cache-dir` and skip it on re-runs.

| Hardware | Estimated (50K images, first run) |
|----------|----------------------------------|
| Apple M1–M4 | ~40 min |
| Modern desktop (Ryzen 7 / i7) | ~45–55 min |
| Mid-range laptop (i5, 10th–12th gen) | ~60–75 min |
| With NVIDIA GPU | ~15–20 min |

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/imagespace.py` | Main pipeline |
| `scripts/add_features.py` | Add brightness/complexity/etc. to existing data |
| `scripts/reprocess_layout.py` | Re-run t-SNE/HDBSCAN only |
| `scripts/generate_data.py` | Generate synthetic test data (no CLIP needed) |

`ImageSpace_Colab.ipynb` — Google Colab notebook for cloud GPU processing.

## Deployment

The built `output/` folder is a self-contained static site with relative paths. It works on any host — GitHub Pages (root or project page), Netlify, a CDN, or even a USB drive.

### Simple upload to GitHub Pages

1. Run the pipeline and build locally:
   ```bash
   python3 scripts/imagespace.py /path/to/images/ -o image_space/public/data/
   cd image_space && npx vite build
   ```

2. Create a new GitHub repo (e.g. `my-collection`).

3. Push the **contents of `image_space/output/`** to that repo's `main` branch:
   ```bash
   cd image_space/output
   git init && git add .
   git commit -m "Deploy"
   git remote add origin https://github.com/your-username/my-collection.git
   git push -u origin main
   ```

4. Go to the repo → **Settings → Pages → Source: Deploy from branch → main / (root)**.

Your site is live at `https://your-username.github.io/my-collection/` — no server, no backend.

> **Size note:** Atlas textures for 50K images are ~200 MB total. GitHub Pages supports repositories up to 1 GB; individual files must be under 100 MB. For 128px thumbnails, each atlas is 3–8 MB — well under the limit.

### Auto-deploy with GitHub Actions (optional)

The included [.github/workflows/deploy.yml](.github/workflows/deploy.yml) lets you commit data to `image_space/public/data/` and have GitHub Pages rebuild automatically on every push. See the workflow file for setup instructions.

## Technology Stack

- **Pipeline**: Python · ONNX Runtime · openTSNE · HDBSCAN · scikit-learn · Pillow · scipy
- **Viewer**: React 18 · PixiJS 8 · pixi-viewport · Vite 5 · Tailwind CSS · lucide-react
- **Design**: Rosé Pine Dawn palette · Inter font

## License

MIT

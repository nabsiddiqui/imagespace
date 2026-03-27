# ImageSpace

**[Live Demo →](https://nabeelsiddiqui.net/imagespace-demo/)** — 49,585 WikiArt paintings visualized with CLIP + t-SNE

An interactive browser-based tool for exploring large image collections. Point it at a folder of images; it produces a navigable 2D scatter plot where similar images cluster together.

**The viewer is a static site — no backend, no server, no database.** Once processed, share the `output/` folder on GitHub Pages, any CDN, or a USB drive.

**GPU recommended.** Hardware acceleration significantly speeds up the CLIP embedding stage. NVIDIA CUDA and Apple Silicon CoreML (Neural Engine) are both supported. A standard CPU works too if you don't have a compatible GPU.

## What It Does

- Lays out images by visual similarity (CLIP embeddings → t-SNE)
- Clusters them automatically (HDBSCAN) and labels clusters with natural language (CLIP)
- Lets you switch between scatter, grid, color-sorted, and timeline views
- Filters by any metadata column (dropdowns) or computed feature (brightness, complexity, etc.)
- Shows similar images and full metadata in a slide-in detail panel
- Renders 50,000 images at ~30 FPS in a browser via PixiJS WebGL

## Quick Start

> **Looking for a dataset to try?** The [WikiArt dataset](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset) on GitHub contains 80K+ fine art images across styles and genres — a good starting point.

### Process your own images

**1a. Install dependencies (NVIDIA GPU)**

```bash
# Python pipeline with CUDA GPU support
pip install pillow numpy scikit-learn opentsne hdbscan onnxruntime-gpu scipy

# Frontend
cd image_space && npm install && cd ..
```

**1b. Install dependencies (Apple Silicon — CoreML)**

```bash
# Python pipeline — CoreML (Neural Engine) is auto-detected
pip install pillow numpy scikit-learn opentsne hdbscan onnxruntime scipy

# Frontend
cd image_space && npm install && cd ..
```

**1c. Install dependencies (CPU only)**

```bash
# Python pipeline
pip install pillow numpy scikit-learn opentsne hdbscan onnxruntime scipy

# Frontend
cd image_space && npm install && cd ..
```

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

## License

MIT

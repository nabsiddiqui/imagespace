<div align="center">
  <img src="assets/logo.svg" width="100" alt="ImageSpace logo"/>
  <h1>ImageSpace</h1>
  <strong><a href="https://nabeelsiddiqui.net/imagespace-demo">Live Demo →</a></strong> — 49,585 WikiArt paintings visualized with CLIP + t-SNE
</div>

---

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

### Process your own images

**Google Colab** — If you'd rather skip local setup, open the included [Colab notebook](ImageSpace_Colab.ipynb), upload a ZIP of your images, run all cells, and download a ready-to-host output folder.

**1a. Install dependencies (Apple Silicon or CPU only)**

```bash
# Python pipeline — CoreML (Neural Engine) is auto-detected on Apple Silicon
pip install pillow numpy scikit-learn opentsne hdbscan onnxruntime scipy

# Frontend
cd image_space && npm install && cd ..
```

**1b. Install dependencies (NVIDIA GPU)**

```bash
# Python pipeline with CUDA GPU support
pip install pillow numpy scikit-learn opentsne hdbscan onnxruntime-gpu scipy

# Frontend
cd image_space && npm install && cd ..
```

CUDA is auto-detected. No extra flags needed.
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

## License

MIT

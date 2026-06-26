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
  --thumb-size 128 --quality 85 \
  --seed 42
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
| `--quality` | 60 | WebP compression quality (1–100) |
| `--hd` | false | Generate dual-resolution atlases (64px preview + 128px full). Auto-sets thumb-size=128 |
| `--preview-quality` | 40 | WebP quality for preview atlases (only used with --hd) |
| `--min-cluster-size` | 50 | HDBSCAN minimum cluster size |
| `--tsne-perplexity` | 30 | t-SNE perplexity |
| `--seed` | 42 | Random seed for PCA and openTSNE reproducibility |
| `--cache-dir` | none | Directory to cache CLIP embeddings (`.npy`) |
| `--relayout` | false | Skip atlas + CLIP, re-run t-SNE/HDBSCAN only |

## Why Grid Snapping Is the Default

ImageSpace uses grid-snapped layouts (Grid, Color, Timeline views) as a deliberate design decision — not a limitation. Here's why:

**Performance.** Grid layouts produce perfectly predictable sprite positions. The renderer never needs to resolve overlaps, recompute z-indices, or run collision detection. On a 50K-image dataset, this means stable 30 FPS even on mid-range hardware. Freeform overlapping layouts would require per-frame depth sorting and occlusion calculations that scale poorly.

**Mobile scalability.** Mobile browsers operate under severe memory constraints (~1–2 GB usable for a single tab). Grid snapping guarantees every sprite occupies a fixed-size cell, enabling efficient spatial hashing (O(1) hover lookup), predictable GPU memory usage, and no layout thrashing. Overlapping images would force costly reflows and z-index recalculations that frequently trigger WebGL context loss on mobile devices.

**Visual clarity.** When exploring 50,000 images, every image is visible and accessible in a grid. No image is hidden behind another. This is critical for research workflows where users need to scan, filter, and identify patterns across an entire collection — not just the images that happen to be on top.

**Predictability.** Grid layouts are deterministic: the same dataset always produces the same layout. Researchers can reproduce views, share screenshots, and reference specific positions. The pipeline also exposes `--seed` (default `42`) for reproducible PCA and openTSNE layout generation.

**Tradeoffs.** Grid snapping sacrifices organic, "gallery-wall" aesthetics and the ability to manually arrange images. The t-SNE scatter view provides the organic clustering experience, while grid views prioritize systematic exploration. This separation is intentional — each view mode serves a different analytical purpose.

## Dual-Resolution Loading

ImageSpace supports progressive atlas loading for fast initial display:

**Desktop / tablet:** 64px preview atlases load first (fast), then 128px HD atlases swap in progressively. A loading bar shows "Loading low-res photos" → "Loading high-res photos."

**Phone:** Only 64px preview atlases are loaded (at 0.25 scale = ~196MB GPU). No HD upgrade — keeps mobile memory low.

The distinction is device-based, not viewport-based: iPads and Android tablets get HD, regardless of window size. iPhones and Android phones stay on preview.

To generate dual-resolution data, run the pipeline with `--hd`:

```bash
python3 scripts/imagespace.py /path/to/images/ \
  -o image_space/public/data/ \
  --metadata /path/to/metadata.csv \
  --thumb-size 128 --quality 60 --hd
```

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

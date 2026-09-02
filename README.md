<div align="center">
  <img src="assets/logo.svg" width="100" alt="ImageSpace logo"/>
  <h1>ImageSpace</h1>
  <strong><a href="https://nabeelsiddiqui.net/imagespace-demo">Live Demo →</a></strong> — 49,585 WikiArt images visualized with CLIP + t-SNE
  <br/><br/>
  <a href="https://github.com/nabsiddiqui/imagespace/actions/workflows/ci.yml"><img src="https://github.com/nabsiddiqui/imagespace/actions/workflows/ci.yml/badge.svg" alt="CI status"/></a>
</div>

---

An interactive browser-based tool for exploring large image collections. Point it at a folder of images; it produces a navigable 2D scatter plot where similar images cluster together.

**The viewer is a static site — no backend, no server, no database.** Run one command and the output directory is ready to upload to GitHub Pages, any CDN, or a USB drive: it contains `index.html` + `assets/` + `data/`. No Node/Vite build required to view results.

**GPU recommended.** Hardware acceleration significantly speeds up the CLIP embedding stage. NVIDIA CUDA and Apple Silicon CoreML (Neural Engine) are both supported. A standard CPU works too if you don't have a compatible GPU.

## Explore without installing

Open the [live demo](https://nabeelsiddiqui.net/imagespace-demo) to use ImageSpace immediately. To run the repository checkout without installing the `imagespace` command, install the dependencies in any environment and invoke the script directly:

```bash
python3 scripts/imagespace.py -i test-data -o ./output \
  --metadata test-data/metadata.csv --seed 42
python3 -m http.server 5174 -d ./output
```

Open http://localhost:5174. The generated site does not require Python, Node, a database, or a network connection after its assets have loaded.

## What It Does

- Lays out images by visual similarity (CLIP embeddings → t-SNE)
- Clusters them automatically (HDBSCAN) and labels clusters with natural language (CLIP)
- Lets you switch between scatter, grid, color-sorted, and timeline views
- Filters by any metadata column (dropdowns) or computed feature (brightness, complexity, etc.)
- Shows similar images and full metadata in a slide-in detail panel
- Renders 50,000 images at ~30 FPS in a browser via PixiJS WebGL

### Process your own images

**1. Install**

From this repo:

```bash
python3 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\\Scripts\\activate
pip install '.[full]'
```

This installs the `imagespace` command, both CLIP backends, cluster-label dependencies, and the bundled viewer shell. NVIDIA users may install the appropriate CUDA builds of PyTorch and `onnxruntime-gpu` instead. A minimal ONNX-only install (`pip install '.[onnx]'`) can extract embeddings, but text-based cluster labels require `torch` and `transformers`; use the full extra for the complete pipeline.

> The viewer app shell is bundled with the package (`imagespace_viewer_shell/`), so the install above is all you need to produce a viewable site. To modify the viewer itself (React/PixiJS in `image_space/`), rebuild the shell with `bash scripts/build_viewer_shell.sh` (requires Node), then `pip install .` again.

**2. Run the pipeline**

> The repo ships with `test-data/` — a 500-image WikiArt sample with `metadata.csv`, ready to try without sourcing your own images. Skip to the [example run](#try-it-with-the-bundled-sample) below, or use your own folder of images:

```bash
imagespace -i /path/to/your/images/ -o ./output
```

#### Try it with the bundled sample

```bash
imagespace -i test-data -o ./output --metadata test-data/metadata.csv --seed 42
python3 -m http.server 5174 -d ./output
```

Then open http://localhost:5174. `test-data/` contains 500 WikiArt images (sampled with seed 42, covering 25 art-historical styles) plus a `metadata.csv` with `filename,artist,style,title,width,height` columns.

Dual-resolution progressive loading is ON by default — 64px preview atlases are generated alongside 128px HD atlases, so the viewer shows low-res images first and upgrades to HD on capable devices. The output directory is a complete, uploadable static site:

```
output/
  index.html        ← viewer entry point
  assets/           ← viewer JS/CSS
  favicon.svg       ← ImageSpace icon
  data/             ← atlases, binary layout, metadata, labels, provenance
    analysis_config.json  ← complete model and analysis parameters
    manifest.json         ← viewer format plus embedded provenance
```

With optional external metadata (CSV must have a `filename` column) and a fixed seed for a reproducible layout (by default the layout is random each run):

```bash
imagespace -i /path/to/your/images/ \
  -o ./output \
  --metadata /path/to/metadata.csv \
  --seed 42
```

For single-resolution output (no preview atlases), pass `--no-hd`:

```bash
imagespace -i /path/to/your/images/ -o ./output --no-hd
```

> You can also run it without installing: `python3 scripts/imagespace.py -i ... -o ...`, or pass the input as a positional argument: `imagespace /path/to/images -o ./output`.

**3. Serve locally**

```bash
python3 -m http.server 5174 -d ./output
```

Open http://localhost:5174 — then upload the `output/` folder to any static host.

---

### Iterate faster with caching

Once you've run the pipeline once, save time on subsequent runs. The embeddings cache is **opt-in** via `--cache-dir` (default runs never write it, keeping output lean):

```bash
# Cache embeddings — skip the CLIP stage on re-runs (~18 min saved for 50K images)
imagespace -i /path/to/images/ \
  -o ./output \
  --cache-dir ./emb_cache/

# Relayout only — re-run t-SNE + HDBSCAN without redoing atlas or CLIP (~45s for 50K)
imagespace -i /path/to/images/ \
  -o ./output \
  --cache-dir ./emb_cache/ \
  --relayout
```

For seed-sweep workflows that only need the raw data (no app shell), add `--data-only`:

```bash
imagespace -i /path/to/images/ \
  -o ./output \
  --cache-dir ./emb_cache/ \
  --data-only --relayout --seed 7
```

---

## Pipeline Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-i`, `--input` | required | Input directory (or pass as a positional argument) |
| `--output`, `-o` | required | Output directory (becomes a self-contained static site) |
| `--metadata` | none | External metadata CSV to merge (must have `filename` column) |
| `--thumb-size` | 128 | Thumbnail size in pixels (128 with --hd, 64 with --no-hd) |
| `--atlas-size` | 4096 | Atlas texture dimensions |
| `--quality` | 60 | WebP compression quality (1–100) |
| `--hd` / `--no-hd` | `--hd` | Generate dual-resolution atlases (64px preview + 128px full). ON by default |
| `--preview-quality` | 40 | WebP quality for preview atlases (only used with --hd) |
| `--clip-model` | `openai/clip-vit-base-patch32` | Hugging Face CLIP checkpoint used by both image and label-text encoders |
| `--pca-dims` | 50 | PCA dimensions before t-SNE and HDBSCAN |
| `--min-cluster-size` | auto | HDBSCAN minimum cluster size (auto-scales to dataset size; pass a number to override) |
| `--hdbscan-min-samples` | 5 | HDBSCAN neighborhood conservativeness |
| `--hdbscan-selection-method` | `leaf` | HDBSCAN selection method: `leaf` or `eom` |
| `--tsne-perplexity` | 30 | t-SNE perplexity |
| `--label-candidates` | bundled `art-v1` | Versioned JSON label vocabulary override |
| `--label-uncertainty-threshold` | 0.01 | Minimum top-1/top-2 score margin for a confident label |
| `--seed` | none | Random seed for PCA and openTSNE. Default: fully random each run. Pass a fixed seed (e.g. `--seed 42`) for a reproducible layout |
| `--cache-dir` | none | Directory to cache CLIP embeddings (`.npy`). Opt-in; default runs never write it |
| `--relayout` | false | Skip atlas + CLIP, re-run t-SNE/HDBSCAN only |
| `--data-only` | false | Emit only `data/` (skip the viewer app shell). For relayout/seed-sweep workflows |

## Methods and provenance

Every run writes `data/analysis_config.json` and repeats that provenance inside `manifest.json`. The record includes the exact CLIP checkpoint and backend, PCA dimensions and explained variance, effective t-SNE settings, seed, HDBSCAN parameters and preserved-noise policy, and label-vocabulary version. The viewer's book icon opens the same record. Cluster label evidence in `cluster_labels.json` and the panel includes the top three candidate scores, score margin, and uncertainty decision.

A custom vocabulary is a JSON object with an `id`, `version`, and at least three candidates:

```json
{"id":"my-labels","version":1,"candidates":[
  {"text":"portrait photograph","short_name":"Portraits"},
  {"text":"architectural exterior","short_name":"Architecture"},
  {"text":"landscape view","short_name":"Landscapes"}
]}
```

## Platform notes

- **macOS Apple Silicon:** ONNX Runtime can use CoreML; PyTorch uses MPS when available.
- **Linux/Windows with NVIDIA:** install CUDA-compatible PyTorch and `onnxruntime-gpu` for acceleration.
- **CPU-only systems:** supported, though embedding extraction is substantially slower.
- **Browser:** current Chrome, Edge, Firefox, or Safari; mobile devices remain on preview atlases to control GPU memory.

## Troubleshooting

- **`CLIP embedding extraction failed`:** install `.[full]`, or install a platform-compatible ONNX/PyTorch runtime.
- **Cluster-label dependency error:** `pip install torch transformers`; labels intentionally fail loudly instead of being silently omitted.
- **Stale embedding cache:** use a separate `--cache-dir` after changing `--clip-model` or the input corpus. ImageSpace rejects incompatible cache metadata or row counts.
- **Blank page when opening `index.html`:** browsers block some local file requests; serve the folder with `python3 -m http.server`.
- **No clusters found:** HDBSCAN may classify a small or heterogeneous corpus as noise. Lower `--min-cluster-size` cautiously; ImageSpace preserves noise rather than assigning it to a nearby cluster.

## Verification

Run the repeatable acceptance contract from the repository root:

```bash
python3 tests/run_checks.py --seed 42
```

It runs deterministic pipeline checks twice, validates provenance and label schemas, guards the HDBSCAN noise policy, builds the frontend and package, and exits zero only when the two seeded outputs match.

For a fast, network-free check of the numeric core (no model downloads), run the unit suite:

```bash
pip install ".[test]"
pytest
```

Continuous integration runs the unit suite across Python 3.10–3.12 and against the oldest supported dependency versions, and rebuilds the viewer, so the pipeline is verified to keep working as its dependencies change. See [`tests/README.md`](tests/README.md) for the full layout.

## Why Grid Snapping Is the Default

ImageSpace uses grid-snapped layouts (Grid, Color, Timeline views) as a deliberate design decision — not a limitation. Here's why:

**Performance.** Grid layouts produce perfectly predictable sprite positions. The renderer never needs to resolve overlaps, recompute z-indices, or run collision detection. On a 50K-image dataset, this means stable 30 FPS even on mid-range hardware. Freeform overlapping layouts would require per-frame depth sorting and occlusion calculations that scale poorly.

**Mobile scalability.** Mobile browsers operate under severe memory constraints (~1–2 GB usable for a single tab). Grid snapping guarantees every sprite occupies a fixed-size cell, enabling efficient spatial hashing (O(1) hover lookup), predictable GPU memory usage, and no layout thrashing. Overlapping images would force costly reflows and z-index recalculations that frequently trigger WebGL context loss on mobile devices.

**Visual clarity.** When exploring 50,000 images, every image is visible and accessible in a grid. No image is hidden behind another. This is critical for research workflows where users need to scan, filter, and identify patterns across an entire collection — not just the images that happen to be on top.

**Predictability.** Grid layouts are deterministic: the same dataset always produces the same layout. Researchers can reproduce views, share screenshots, and reference specific positions. The pipeline exposes `--seed` (no seed by default = random each run); pass a fixed seed (e.g. `--seed 42`) for reproducible PCA and openTSNE layout generation.

**Tradeoffs.** Grid snapping sacrifices organic, "gallery-wall" aesthetics and the ability to manually arrange images. The t-SNE scatter view provides the organic clustering experience, while grid views prioritize systematic exploration. This separation is intentional — each view mode serves a different analytical purpose.

## Dual-Resolution Loading

ImageSpace supports progressive atlas loading for fast initial display:

**Desktop / tablet:** 64px preview atlases load first (fast), then 128px HD atlases swap in progressively. A loading bar shows "Loading low-res photos" → "Loading high-res photos."

**Phone:** Only 64px preview atlases are loaded (at 0.25 scale = ~196MB GPU). No HD upgrade — keeps mobile memory low.

The distinction is device-based, not viewport-based: iPads and Android tablets get HD, regardless of window size. iPhones and Android phones stay on preview.

To generate dual-resolution data, just run the pipeline (dual-resolution is the default). Pass `--no-hd` to generate single-resolution 128px atlases only:

```bash
# Dual-resolution (default): preview + HD atlases
imagespace -i /path/to/images/ -o ./output --metadata /path/to/metadata.csv

# Single-resolution 128px only (no preview atlases)
imagespace -i /path/to/images/ -o ./output --no-hd
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

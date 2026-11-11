<div align="center">
  <img src="assets/logo.svg" width="100" alt="ImageSpace logo"/>
  <h1>ImageSpace</h1>
  <strong><a href="#explore-without-installing">Explore the bundled sample →</a></strong> — 500 WikiArt images visualized with CLIP + UMAP
</div>

---

An interactive browser-based tool for exploring large image collections. Point it at a folder of images; it produces a navigable 2D scatter plot where similar images cluster together.

**The viewer is a static site — no backend, no application server, no database.** Viewing it only needs a simple static web server (see below). Run one command and the output directory is ready to upload to GitHub Pages, any CDN, or a USB drive: it contains `index.html` + `assets/` + `data/`. No Node/Vite build required to view results.

**GPU recommended.** Hardware acceleration significantly speeds up the CLIP embedding stage. NVIDIA CUDA and Apple Silicon CoreML (Neural Engine) are both supported. A standard CPU works too if you don't have a compatible GPU.

## Explore without installing

A finished interface, generated from the 500 bundled sample images (`test-data/`, `--seed 42`), is included in [`examples/sample-output/`](examples/sample-output). Viewing it requires no ImageSpace installation and no Python packages, only a browser and a simple local web server. Python's built-in server works:

```bash
python3 -m http.server 5174 -d examples/sample-output
```

Open http://localhost:5174. Do not open `index.html` directly: browsers block the local file requests the viewer needs, which produces a blank page. The site needs no database or server-side code, so the same folder can also be uploaded to any static host.

A larger [live demo](https://nabeelsiddiqui.net/imagespace-demo) shows 49,585 WikiArt images.

To regenerate the sample yourself, see [Process your own images](#process-your-own-images).

## What It Does

- Lays out images by visual similarity (CLIP embeddings → 2-D UMAP, cosine distance)
- Clusters them automatically (HDBSCAN on a separate low-dimensional UMAP space) and lists clusters as numbered hotspots; given a metadata column, each hotspot shows its five most common values as percentages
- Lets you switch between five views: UMAP scatter (default), Snap (overlap-free), Grid, Color, and Timeline
- Filters by categorical metadata columns (dropdowns) and by computed scores (range sliders for brightness, complexity, edge density, Uniqueness, and Cluster Fit)
- Shows similar images and full metadata in a slide-in detail panel
- Preserves HDBSCAN noise as unassigned and hides it by default, with a one-click toggle to reveal it in a distinct duotone tint
- Renders tens of thousands of images in a browser via PixiJS WebGL

### Process your own images

**1. Install**

From this repo:

```bash
python3 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\\Scripts\\activate
pip install '.[full]'
```

This installs the `imagespace` command, both CLIP backends, and the bundled viewer shell. NVIDIA users may install the appropriate CUDA builds of PyTorch and `onnxruntime-gpu` instead. A minimal ONNX-only install (`pip install '.[onnx]'`) is enough to run the full pipeline.

> The viewer app shell is bundled with the package (`imagespace_viewer_shell/`), so the install above is all you need to produce a viewable site. To modify the viewer itself (React/PixiJS in `image_space/`), rebuild the shell with `bash scripts/build_viewer_shell.sh` (requires Node), then `pip install .` again.

**2. Run the pipeline**

> The repo ships with `test-data/` — a 500-image WikiArt sample with `metadata.csv`, ready to try without sourcing your own images. Skip to the [example run](#try-it-with-the-bundled-sample) below, or use your own folder of images:

```bash
imagespace -i /path/to/your/images/ -o ./output
```

A metadata CSV is optional. Without one, the pipeline still builds the full viewer; hotspots are simply numbered and only computed properties are filterable. Add `--metadata your.csv` for metadata filters, and `--label-column <column>` for the hotspot breakdowns described in [Hotspot breakdowns](#hotspot-breakdowns).

#### Try it with the bundled sample

```bash
imagespace -i test-data -o ./output --metadata test-data/metadata.csv --label-column style --seed 42
python3 -m http.server 5174 -d ./output
```

Then open http://localhost:5174. `test-data/` contains 500 WikiArt images (sampled with seed 42, covering 25 art-historical styles) plus a `metadata.csv` with `filename,artist,style,title,width,height` columns.

Dual-resolution progressive loading is ON by default — 64px preview atlases are generated alongside 128px HD atlases, so the viewer shows low-res images first and loads HD on demand, when zoomed in, on capable devices. The output directory is a complete, uploadable static site:

```
output/
  index.html        ← viewer entry point
  assets/           ← viewer JS/CSS
  favicon.svg       ← ImageSpace icon
  data/             ← atlases, binary layout, metadata, hotspot breakdowns, provenance
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

# Relayout only — re-run UMAP + HDBSCAN without redoing atlas or CLIP
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
| `--clip-model` | `openai/clip-vit-base-patch32` | Hugging Face CLIP checkpoint used for image embeddings |
| `--umap-neighbors` | 15 | UMAP `n_neighbors` for the 2-D layout |
| `--umap-min-dist` | 0.1 | UMAP `min_dist` for the 2-D layout |
| `--cluster-neighbors` | 30 | UMAP `n_neighbors` for the clustering space |
| `--cluster-dims` | 10 | Dimensions of the UMAP space HDBSCAN clusters in (`min_dist` fixed at 0) |
| `--min-cluster-size` | auto | HDBSCAN minimum cluster size. Auto = about 1% of the collection, rounded to the nearest 10, minimum 10 (49,585 images → 500). Pass a number to override |
| `--hdbscan-min-samples` | 5 | HDBSCAN neighborhood conservativeness |
| `--hdbscan-selection-method` | `leaf` | HDBSCAN selection method: `leaf` or `eom` |
| `--label-column` | none | Metadata column used for hotspot breakdowns (e.g. `style`); needs `--metadata`. Without it, hotspots are numbered only |
| `--label-mixed-threshold` | 0.5 | Flag a hotspot as mixed when its most common value covers less than this share of it |
| `--seed` | none | Random seed for UMAP. Default: fully random each run. Pass a fixed seed (e.g. `--seed 42`) for a reproducible layout; a fixed seed makes UMAP run single-threaded, so it is slower |
| `--cache-dir` | none | Directory to cache CLIP embeddings (`.npy`). Opt-in; default runs never write it |
| `--relayout` | false | Skip atlas + CLIP, re-run UMAP/HDBSCAN only |
| `--data-only` | false | Emit only `data/` (skip the viewer app shell). For relayout/seed-sweep workflows |

## Methods and provenance

Every run writes `data/analysis_config.json` and repeats that provenance inside `manifest.json`. The record includes the exact CLIP checkpoint and backend, the UMAP settings for the layout and for the clustering space, the nearest-neighbor metric (cosine on the full CLIP embeddings), seed, HDBSCAN parameters and preserved-noise policy, and the metadata column used for hotspot breakdowns. The viewer's book icon opens the same record.

### Hotspot breakdowns

Clusters appear in the sidebar as numbered hotspots, largest first (Hotspot 1, Hotspot 2, …). ImageSpace does not name them with a model. Instead, with `--metadata your.csv --label-column style`, hovering over or clicking a hotspot opens a breakdown of its five most common values in that column, each as a share of the hotspot, with everything else grouped as "Other":

```
Hotspot 2 · 2,610 images · style · mixed
Color Field Painting     29%
Abstract Expressionism   29%
Minimalism               27%
Pop Art                   8%
Expressionism             3%
Other                     5%
```

A hotspot is flagged as mixed when its top value covers less than half of it (`--label-mixed-threshold`). The Methods panel repeats each breakdown with a second share: how much of all images with that value fall in the hotspot (e.g. *Ukiyo-e: 96% of the hotspot, 93% of all Ukiyo-e*). The figures are stored in `data/cluster_labels.json`. Unassigned (noise) images are never broken down. Without `--label-column`, no breakdown is written and hotspots stay numbered.

### Noise handling in the viewer

ImageSpace never reassigns HDBSCAN noise to a nearby cluster; points HDBSCAN leaves unassigned (`cluster = -1`) stay unassigned in the data. In the viewer that noise is **hidden by default** so the clustered structure reads clearly, and it is omitted from the Hotspots sidebar until revealed. The `Noise` control (beside `Properties`) toggles it on. When shown, noise images render in a distinct indigo duotone rather than their natural colors, so they stay visually separable from the full-color clustered images even when a source image is itself black and white. Toggling `Show` returns the `Noise` group to the sidebar for filtering and inspection.

## Platform notes

- **macOS Apple Silicon:** ONNX Runtime can use CoreML; PyTorch uses MPS when available.
- **Linux/Windows with NVIDIA:** install CUDA-compatible PyTorch and `onnxruntime-gpu` for acceleration.
- **CPU-only systems:** supported, though embedding extraction is substantially slower.
- **Browser:** current Chrome, Edge, Firefox, or Safari; mobile devices remain on preview atlases to control GPU memory.

## Troubleshooting

- **`CLIP embedding extraction failed`:** install `.[full]`, or install a platform-compatible ONNX/PyTorch runtime.
- **`--label-column` error:** the column must exist in the `--metadata` CSV, whose `filename` column must match the image file names.
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

## View Modes

The viewer opens on the **UMAP** scatter, which places each image at its raw 2-D UMAP coordinates. Images closer to their cluster's center are drawn on top, so overlapping regions show their most typical members first. The other four views rearrange the same images:

- **Snap:** the UMAP layout with overlaps removed. Each image moves to the nearest free grid cell, working outward from the center, so every image is visible while neighborhoods stay roughly in place.
- **Grid:** images in discovery order (subfolders first, files sorted within each folder). With a flat folder of files named `artist_title.jpg`, this produces an alphabetical arrangement by artist.
- **Color:** images sorted by the hue of each thumbnail's average color, then by lightness.
- **Timeline:** images sorted by year, with undated images first. Years come from EXIF `DateTimeOriginal` or a year in the filename.

Hotspot selection, filters, and the Noise toggle apply in all five views. UMAP and Snap dim non-matching images; Grid, Color, and Timeline hide them.

**Reproducibility.** UMAP is random unless seeded (no seed by default). Pass a fixed seed (e.g. `--seed 42`) for reproducible layouts; seeding makes UMAP single-threaded.

## Dual-Resolution Loading

ImageSpace supports progressive atlas loading for fast initial display:

**Desktop / tablet:** 64px preview atlases load first (fast). A 128px HD atlas is loaded only when the view is zoomed in far enough for the preview to look soft and some of its images are on screen (or one is open in the detail panel). HD atlases that have left the view are swapped back to previews once more than a few are held, which keeps GPU memory low.

**Phone:** Only 64px preview atlases are loaded (at 0.25 scale). No HD upgrade — keeps mobile memory low.

The distinction is device-based, not viewport-based: iPads and Android tablets get HD, regardless of window size. iPhones and Android phones stay on preview.

To generate dual-resolution data, just run the pipeline (dual-resolution is the default). Pass `--no-hd` to generate single-resolution 128px atlases only:

```bash
# Dual-resolution (default): preview + HD atlases
imagespace -i /path/to/images/ -o ./output --metadata /path/to/metadata.csv

# Single-resolution 128px only (no preview atlases)
imagespace -i /path/to/images/ -o ./output --no-hd
```

## Metadata Format

Metadata is optional. If you supply a CSV, it must have a `filename` column matching image filenames. Non-numeric columns with at most 200 distinct values become dropdown filters in the viewer automatically (`id`, `filename`, `timestamp`, `width`, and `height` are skipped). High-cardinality columns such as `artist` or `title` still appear in the detail panel. Numeric metadata columns are not filterable; range sliders are provided only for the computed scores below.

```csv
filename,artist,style,title
monet_water-lilies.jpg,Claude Monet,Impressionism,Water Lilies
picasso_guernica.jpg,Pablo Picasso,Cubism,Guernica
```

The pipeline also computes and appends: `brightness`, `complexity`, `edge_density`, `outlier_score`, `cluster_confidence` (all 0–100 scale).

## License

MIT

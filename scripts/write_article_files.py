#!/usr/bin/env python3
"""Write updated article files for ImageSpace CHR paper.

This script writes the updated technical reference and other article files
to the article directory, which is outside the VS Code workspace.
"""

import os

ARTICLE_DIR = "/Users/nabeel/Dropbox/Spring 2026/ImageSpace Article"
MEMORY_DIR = os.path.join(ARTICLE_DIR, "memory-system")

def write_techref():
    """Write updated imagespace_technical_reference.md"""
    content = """# ImageSpace — Technical Reference for Article Updates

> **Updated**: 2026-02-10
> **Repository**: https://github.com/nabsiddiqui/imagespace
> **Source Directory**: `/Users/nabeel/Documents/ImageSpace/`
> **Pipeline Script**: `scripts/imagespace.py` (~1057 lines)
> **Viewer Source**: `frontend-pixi/src/App.jsx` (~1915 lines)
> **Latest Commit**: 8487d79

---

## Current Architecture (What the Article Must Describe)

### Pipeline: 9 Stages
1. **Image Discovery** — recursive scan for JPEG/PNG/GIF/BMP/WebP
2. **Atlas Generation** — WebP spritesheets (4096x4096, 128px thumbs, quality 85), ~115 img/s
3. **CLIP Embedding** — ONNX Runtime with Xenova/clip-vit-base-patch32 (~350MB, auto-cached)
4. **PCA + openTSNE + HDBSCAN** — PCA 512->50d, FFT-accelerated t-SNE, density-based clustering with cKDTree noise reassignment
5. **k-Nearest Neighbors** — cosine similarity on 50d PCA embeddings (k=10)
6. **CLIP Cluster Labels** — auto-generated semantic labels per cluster
7. **Metadata Extraction** — dominant colors + timestamps from filenames
8. **Image Features** — brightness (BT.601), complexity (Shannon entropy), edge density (Sobel), all 0-100 scale
9. **Output** — Binary data.bin (24 bytes/image), manifest.json, metadata.csv, neighbors.bin, cluster_labels.json

### Viewer: React + PixiJS WebGL
- **4 view modes**: t-SNE scatter, Grid, Color (sorted by hue), Timeline
- **Filtering**: categorical dropdowns (multi-select, union) + continuous range sliders (brightness, complexity, edge density, outlier score, cluster confidence)
- **Navigation**: HDBSCAN hotspot cards (left sidebar, persistent across all views), minimap (t-SNE mode)
- **Detail panel**: canvas-based atlas crop, k-NN similar images, full metadata
- **Performance**: 50K sprites at ~30fps via WebGL sprite batching, binary data loading, parallel atlas fetch

### Data Formats
- **data.bin** (24 bytes/image): float32 x, y, tsneX, tsneY + uint16 atlas, u, v, cluster
- **WebP atlas textures**: 4096x4096, 128px thumbnails, 49 atlases for 49,585 images
- **metadata.csv**: 15 columns (WikiArt): id, filename, cluster, timestamp, dominant_color, artist, style, title, width, height, brightness, complexity, edge_density, outlier_score, cluster_confidence
- **neighbors.bin**: uint32 count + k, then [count x k] pairs of (uint32 neighbor_id, float32 distance)
- **cluster_labels.json**: CLIP-generated semantic labels per cluster

---

## Pipeline Timing — MEASURED (49,585 WikiArt images, Apple M-series CPU)

| Stage | Time | Rate |
|---|---|---|
| Atlas generation (128px, q85) | **432.5s** (7.2 min) | 115 img/s |
| CLIP embeddings (ONNX CPU) | **1084.0s** (18.1 min) | 45.7 img/s |
| PCA (512d -> 50d) | < 0.1s | 69% variance |
| openTSNE (FFT) | **26.2s** | — |
| Overlap removal | **3.3s** | — |
| HDBSCAN | **19.0s** | 19 clusters |
| k-NN (k=10, cosine) | **15.5s** | — |
| Cluster labels (CLIP) | **1.3s** | 19 labels |
| Metadata extraction | **418.7s** (7.0 min) | — |
| Image features | **423.1s** (7.1 min) | — |
| **Total (first run, CPU only)** | **2441.6s (40.7 min)** | |
| **Total (cached embeddings)** | **~22 min** | |

### CPU Expectations for Different Hardware
| Hardware | Estimated Total (50K images) |
|----------|------------------------------|
| Apple M1/M2/M3/M4 | ~40 min |
| Modern x86 desktop (i7/Ryzen 7, AVX2) | ~45-55 min |
| Mid-range laptop (i5 10th-12th gen) | ~60-75 min |
| Older laptop (i5 8th gen) | ~70-90 min |

**Key insight for article**: CPU-only processing is now ~40 min for 50K images, NOT the "~14 hours" cited in the old draft. ONNX Runtime processes 45.7 img/s on CPU, not "roughly one image per second." The GPU flag still exists but is far less necessary now.

---

## Changes From Old Draft That Required Article Updates

### 1. UMAP -> t-SNE Only
**Old**: UMAP with optional t-SNE. **New**: openTSNE (FFT-accelerated) as sole reduction method. PCA 512->50d pre-reduction.

### 2. KMeans -> HDBSCAN
**Old**: KMeans (K=15). **New**: HDBSCAN density-based (19 clusters discovered automatically), noise reassignment via cKDTree.

### 3. 8 View Modes -> 4 View Modes
**Old**: 8 modes (2D Scatter, 3D Scatter, Grid, Hotspots, Timeline, Stack, Color, Carousel). **New**: 4 modes (t-SNE, Grid, Color, Timeline). Hotspots are persistent sidebar cards, not a mode.

### 4. Single HTML -> Static Site
**Old**: Self-contained single HTML file. **New**: Static site (React + PixiJS + Vite). Multiple files.

### 5. Canvas -> WebGL
**Old**: HTML5 Canvas. **New**: PixiJS 8.16 WebGL (50K+ images at ~30fps).

### 6. JPEG Inline -> WebP Atlas
**Old**: Individual JPEG thumbnails inline. **New**: WebP atlas spritesheets.

### 7. Embedding Speed Dramatically Improved
**Old**: "roughly one image per second" on CPU. **New**: 45.7 img/s on CPU.

### 8. New Features Not in Old Draft
- Range slider filters (brightness, complexity, edge density, outlier score, cluster confidence)
- k-NN similar images (10 nearest neighbors)
- CLIP cluster labels (auto-generated semantic names)
- Minimap with viewport rectangle
- Pipeline-computed image features
- Google Colab notebook

### 9. Case Study Uses Different Clustering
- Old: 15 KMeans clusters with UMAP. New: 19 HDBSCAN clusters with t-SNE.

---

## Frontend Tech Stack

| Component | Version | Purpose |
|---|---|---|
| React | 18.2 | UI framework |
| PixiJS | 8.16.0 | WebGL sprite rendering |
| pixi-viewport | 6.0.3 | Pan/zoom/pinch |
| Vite | 5.4.21 | Build tool |
| Tailwind CSS | 3.4.1 | Utility-first styling (Rose Pine Dawn palette) |
| lucide-react | latest | Icons |

## Pipeline Tech Stack

| Component | Version | Purpose |
|---|---|---|
| Python | 3.14 | Runtime |
| ONNX Runtime | 1.24.1 | Fast CLIP inference |
| openTSNE | 1.0.4 | FFT-accelerated t-SNE |
| HDBSCAN | 0.8.41 | Density-based clustering |
| scikit-learn | 1.8.0 | PCA |
| Pillow | latest | Image processing |
| numpy | 2.4.2 | Numerical computation |
| scipy | 1.17.0 | cKDTree for noise reassignment |

---

## New References to Add

- **openTSNE**: Policar et al. 2019. "openTSNE: A Modular Python Library for t-SNE Dimensionality Reduction." bioRxiv. doi:10.1101/731877.
- **HDBSCAN**: Campello et al. 2013. "Density-Based Clustering Based on Hierarchical Density Estimates." PAKDD.
- **PixiJS**: Goodboy Digital. 2024. PixiJS v8. https://pixijs.com.
- **t-SNE original**: van der Maaten and Hinton. 2008. "Visualizing Data using t-SNE." JMLR 9: 2579-2605.

## References to Remove/Modify
- **UMAP** (McInnes et al. 2020): Remove from glossary and primary description. May still mention as comparison.

---

## WikiArt Case Study Clusters (Current, HDBSCAN, 19 clusters)

| Cluster | Label | Similarity Score |
|---|---|---|
| 0 | Complex Scenes - Vast Landscapes | 0.291 |
| 1 | Complex Scenes - Battle Scenes | 0.319 |
| 2 | Art Nouveau | 0.305 |
| 3 | Baroque | 0.318 |
| 4 | Portraits - Sketches | 0.293 |
| 5 | Vast Landscapes - Romantic Landscapes | 0.320 |
| 6 | Vast Landscapes - Mountain Landscapes | 0.323 |
| 7 | Portraits - Warm-Toned | 0.330 |
| 8 | Portraits - Expressionist | 0.326 |
| 9 | Portraits - Close-up Portraits | 0.358 |
| 10 | Portraits - Baroque | 0.339 |
| 11 | Romantic Landscapes | 0.328 |
| 12 | Still Life - Interiors | 0.312 |
| 13 | Still Life - Impressionist | 0.321 |
| 14 | Impressionist | 0.304 |
| 15 | Expressionist | 0.282 |
| 16 | Abstract Expressionist | 0.291 |
| 17 | Seascapes | 0.326 |
| 18 | Vast Landscapes - Impressionist | 0.326 |

**Note**: The old case study findings (15 KMeans clusters, UMAP-based) are fundamentally outdated. The new HDBSCAN produces 19 clusters with entirely different compositions. The case study section of the draft must be completely rewritten based on the new cluster data. Key analytical points to preserve: CLIP groups by visual similarity not period labels, bias evidence may persist, arrangement as provocation. But specific cluster numbers, compositions, and the Ukiyo-e finding must all be re-verified against the new data.
"""
    path = os.path.join(ARTICLE_DIR, "imagespace_technical_reference.md")
    with open(path, 'w') as f:
        f.write(content)
    print(f"Written techref: {len(content)} chars")


def write_project_brief():
    """Write updated memory-system/projectBrief.md"""
    content = """# ImageSpace CHR Software Paper — Project Brief

## Project Identity
- **Article title**: ImageSpace: A Minimal-Computing Pipeline for Exploratory Visualization of Image Collections
- **Target journal**: Computational Humanities Research (CHR), Cambridge University Press
- **Special issue**: "Computational Approaches to Art" (Guest editors: Leonardo Impett, Lin Du, Ellen Charlesworth)
- **Submission deadline**: 30 June 2026
- **Article type**: Software Paper (max 6,000 words / 12 pages)
- **Citation style**: Chicago Author-Date, double-anonymous

## Software Under Description
- **Repository**: https://github.com/nabsiddiqui/imagespace (commit 8487d79)
- **Codebase location**: `/Users/nabeel/Documents/ImageSpace/`
- **Pipeline**: `scripts/imagespace.py` (~1057 lines, Python 3.14)
- **Viewer**: `frontend-pixi/src/App.jsx` (~1915 lines, React 18.2 + PixiJS 8.16)

## Architecture Summary
ImageSpace is a two-part system: a Python pipeline that processes image collections, and a React/PixiJS static web viewer.

**Pipeline (9 stages)**: Image discovery, WebP atlas generation, CLIP embedding (ONNX Runtime), PCA + openTSNE + HDBSCAN clustering, k-nearest neighbors, CLIP cluster labels, metadata extraction, image features, binary output.

**Viewer (4 modes)**: t-SNE scatter, Grid, Color (hue-sorted), Timeline. Persistent HDBSCAN hotspot sidebar, categorical + range slider filters, detail panel with k-NN similar images, minimap.

**Output format**: Static website (no backend). Binary data.bin (24 bytes/image), WebP atlas spritesheets, metadata.csv, neighbors.bin, cluster_labels.json.

**Performance**: 50K images in ~40 min CPU-only (measured). 50K sprites at ~30fps via WebGL.

## Central Argument
ImageSpace expands who can practice distant viewing by combining CPU-first CLIP embeddings, automatic density-based clustering, and a multi-mode static web viewer. Its design treats accessibility as a scholarly commitment rather than a secondary convenience.

## Key Technical Decisions
1. **t-SNE** (openTSNE FFT-accelerated) for dimensionality reduction — emphasizes local structure
2. **HDBSCAN** for clustering — automatic cluster count, density-based, noise handling
3. **WebGL** (PixiJS 8.16) for rendering — 50K+ images at interactive framerates
4. **Static site** output — no server needed, sustainable long-term
5. **CPU-first** — 45.7 img/s on CPU via ONNX Runtime, GPU optional but unnecessary

## Case Study
49,585 WikiArt paintings (27 styles, 1,092 artists). HDBSCAN produced 19 clusters with CLIP-generated semantic labels. Cluster compositions and bias findings need re-analysis with the new pipeline output.

## Article Files
| File | Purpose |
|---|---|
| `draft.md` | Full article prose |
| `outline.md` | Paragraph-level outline with L-annotations |
| `notes.md` | Submission details, critique log, decisions |
| `todo.md` | Task tracker |
| `imagespace_technical_reference.md` | Technical bridge doc (old -> new) |
| `case-study/findings.md` | WikiArt case study analysis |
| `memory-system/` | This memory bank |

## Zotero Collection
- Key: HL77GI8M
- ~40 items covering CLIP/VLMs, dimensionality reduction, digital art history, research software sustainability
- New references needed: openTSNE (Policar et al. 2019), HDBSCAN (Campello et al. 2013), t-SNE (van der Maaten & Hinton 2008)
"""
    path = os.path.join(MEMORY_DIR, "projectBrief.md")
    with open(path, 'w') as f:
        f.write(content)
    print(f"Written projectBrief: {len(content)} chars")


def write_progress():
    """Write updated memory-system/progress.md"""
    content = """# ImageSpace CHR Software Paper — Progress

## Current Status: DRAFT REVISION NEEDED

The draft was written against the OLD version of ImageSpace (UMAP, KMeans, 8 modes, single HTML, Canvas). The software has been completely rewritten. The draft requires comprehensive technical revision.

## What's Done
- [x] Outline: 21 paragraphs, 6 sections, CFP-aligned (needs updating for new tech)
- [x] Case study: 49,585 WikiArt processed (OLD pipeline — 15 KMeans/UMAP clusters)
- [x] Draft: ~5,050 words, two critique rounds (needs comprehensive revision)
- [x] Technical reference: Updated for new architecture (2026-02-10)
- [x] Memory bank: Updated for new architecture (2026-02-10)
- [x] Figures: Generated (ALL OUTDATED — show UMAP/KMeans, need t-SNE/HDBSCAN)

## What Needs Revision (Priority Order)

### 1. Draft Technical Updates (CRITICAL)
Every technical claim in the draft is outdated. Changes needed throughout:
- UMAP -> t-SNE (openTSNE FFT-accelerated) everywhere
- KMeans (15 clusters) -> HDBSCAN (19 clusters) everywhere
- 8 view modes -> 4 view modes (t-SNE, Grid, Color, Timeline)
- Single HTML file -> static website (React + PixiJS + Vite)
- Canvas rendering -> WebGL (PixiJS 8.16)
- Inline JPEG thumbnails -> WebP atlas spritesheets
- "~1 img/s CPU" -> 45.7 img/s CPU
- "~14 hours" -> 40.7 minutes (measured)
- "10,000 image cap" -> 50K+ images at ~30fps
- 5 pipeline stages -> 9 pipeline stages
- Add: range filters, k-NN neighbors, CLIP cluster labels, minimap, image features
- Add: Google Colab notebook

### 2. Case Study Re-analysis (CRITICAL)
Old findings based on 15 KMeans clusters with UMAP projection. New pipeline produces 19 HDBSCAN clusters with t-SNE. Must:
- Analyze new cluster compositions from metadata.csv
- Verify whether Ukiyo-e bias finding persists
- Update all specific numbers (cluster IDs, percentages, compositions)
- Rewrite case study paragraphs with new data

### 3. Figure Regeneration (CRITICAL)
All 4 figures reference UMAP projection and 15 KMeans clusters:
- Figure 1: Was UMAP scatter -> needs t-SNE scatter with 19 HDBSCAN clusters
- Figure 2: Was Ukiyo-e isolation in UMAP -> needs t-SNE equivalent
- Figure 3: Was cluster composition heatmap (15 clusters) -> needs 19-cluster version
- Figure 4: Was UMAP vs t-SNE comparison -> needs rethinking (both are t-SNE now)

### 4. Outline Updates
- Update §4.1 pipeline description (5 stages -> 9)
- Update §4.3 viewer description (8 modes -> 4)
- Update §4.5 case study paragraph
- Update §5.2 constraints (10K cap -> 50K+, speed claims)
- Update Table 1 (feature comparison)
- Update abstract and PLS outline entries
- Update glossary (remove UMAP, add t-SNE, HDBSCAN, PCA)

### 5. New References
- openTSNE: Policar et al. 2019
- HDBSCAN: Campello et al. 2013
- t-SNE: van der Maaten & Hinton 2008
- PixiJS: Goodboy Digital 2024
- Remove/demote UMAP (McInnes et al. 2020)

### 6. Sustainability Argument Update
The "single HTML file" sustainability argument needs nuancing:
- Output is now a static website, not a single file
- But still requires NO backend, NO server logic
- WebP atlas textures + binary data = modern web standards
- Argument shifts from "single file" to "static site with no backend dependencies"

## What's Left (Pre-Submission)
- [ ] Draft technical revision (all sections)
- [ ] Case study re-analysis with new cluster data
- [ ] Figure regeneration
- [ ] Outline update
- [ ] New references added to Zotero
- [ ] Anonymous GitHub repository
- [ ] Zenodo DOI
- [ ] Cover letter
- [ ] Final anonymization pass
- [ ] Convert to submission format

## Key Dates
- **Draft originally completed**: 2026-02-08
- **Software rewrite completed**: 2026-02-10
- **Technical reference updated**: 2026-02-10
- **Submission deadline**: 2026-06-30
"""
    path = os.path.join(MEMORY_DIR, "progress.md")
    with open(path, 'w') as f:
        f.write(content)
    print(f"Written progress: {len(content)} chars")


if __name__ == "__main__":
    write_techref()
    write_project_brief()
    write_progress()
    print("All article files written successfully.")

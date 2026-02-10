# ImageSpace — Project Brief

## Core Mission
ImageSpace is a modern, static-site replacement for Yale DHLab's PixPlot. It visualizes large image collections (50K+) as interactive 2D scatter plots using CLIP embeddings, t-SNE dimensionality reduction, and HDBSCAN clustering. Deployable as a self-contained static site (GitHub Pages, any HTTP server, no backend needed).

Built for the article "ImageSpace: A Modern Approach to Image Collection Visualization" in the *Computational Approaches to Art* special issue of *Computational Humanities Research*.

## Key Requirements
1. **Static site** — No server required for viewing; just HTML/JS/CSS + data files
2. **WebGL rendering** — PixiJS for GPU-accelerated display of 50K+ image thumbnails at ~30 FPS
3. **Multiple view modes** — t-SNE scatter, Grid, Color sort, Timeline
4. **Filtering** — Categorical (multi-select dropdowns) + continuous range sliders + HDBSCAN cluster hotspots
5. **Pipeline-computed features** — Brightness, complexity, edge density, outlier score, cluster confidence
6. **Detail panel** — Canvas-based thumbnail, similar images (k-NN), full metadata
7. **Rose Pine Dawn** aesthetic — Clean, academic, pastel palette
8. **Python pipeline** — ONNX CLIP → PCA → openTSNE → HDBSCAN → k-NN → features → WebP atlases

## Non-Goals
- Real-time collaboration
- Server-side rendering
- 3D views
- Image upload from browser
- Search/filter by filename or text (explicitly declined by user)

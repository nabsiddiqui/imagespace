# ImageSpace — Product Context

## Why This Exists
- PixPlot (Yale DHLab) is outdated, hard to deploy, and no longer maintained
- Researchers need a simple, static tool to explore large image datasets visually
- Must run as a GitHub Pages site — no backend required for viewing
- Supports the academic article on computational approaches to art

## How It Works

### Data Pipeline (Python)
1. Input folder of images → CLIP ONNX embeddings (cached as .npy)
2. PCA 512d → 50d → openTSNE → 2D positions
3. HDBSCAN density-based clustering (with noise reassignment via cKDTree)
4. k-NN neighbors computed (cosine similarity on PCA embeddings)
5. Image features extracted: brightness (BT.601), complexity (Shannon entropy), edge density (Sobel), outlier score (mean k-NN distance), cluster confidence (HDBSCAN probabilities) — all on 0–100 scale
6. CLIP-based cluster labels generated
7. WebP atlas textures generated (4096×4096, configurable thumb size)
8. Binary layout file (24 bytes/image) + metadata CSV + manifest JSON

### Viewer (React + PixiJS)
- Loads binary data + atlas textures + metadata
- Renders 50K sprites with animated transitions between 4 view modes
- Categorical dropdowns + continuous range sliders for filtering
- Detail panel with canvas-based atlas crop, similar images, metadata
- Minimap, cluster labels, hotspot cards for navigation

## User Experience Goals
- Load → see all 50K images in ~3-5 seconds
- Switch views with single click (t-SNE, Grid, Color, Timeline)
- Filter by any CSV column (categorical dropdowns, ≤200 unique values shown)
- Slide range filters for computed features (brightness, complexity, etc.)
- Click hotspot cards to zoom into a cluster
- Click any image → detail panel with thumbnail, similar images, metadata
- Hover → pointer cursor, scale 1.5x, gold tint, tooltip with ID
- Clean, academic Rose Pine Dawn aesthetic with Inter font

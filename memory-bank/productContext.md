# ImageSpace — Product Context

## Why This Exists
- PixPlot (Yale DHLab) is outdated and hard to deploy
- Researchers need a simple, static tool to explore large image datasets
- Must run as a GitHub Pages site — no backend required for viewing

## How It Works
1. **Data Pipeline** (Python, future): Input folder of images → CLIP embeddings → UMAP 2D positions → K-means clustering → Atlas JPEG generation → Binary layout file + metadata CSV
2. **Viewer** (Current): Static React+PixiJS app loads binary data + atlas textures, renders 50K sprites with animated transitions between view modes

## User Experience Goals
- Load → see all 50K images in < 5 seconds
- Switch views with single click (UMAP, Grid, Color, Timeline, Carousel)
- Filter by any CSV column (dropdowns in top-left)
- Click hotspot cards to filter to a cluster
- Detail panel on right (toggled) shows image metadata
- Clean, academic Rose Pine Dawn aesthetic

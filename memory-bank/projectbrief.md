# ImageSpace — Project Brief

## Core Mission
ImageSpace is a modern, static-site replacement for Yale DHLab's PixPlot. It visualizes large image collections (50K+) as interactive 2D scatter plots with UMAP embeddings, grid layouts, color sorting, and timeline views. Must be deployable as a static site on GitHub Pages.

## Key Requirements
1. **Static site** — no server required for viewing; just HTML/JS/CSS + data files
2. **WebGL rendering** — PixiJS for GPU-accelerated display of 50K+ image thumbnails
3. **Multiple view modes** — UMAP scatter, Grid, Color sort, Timeline, Carousel
4. **Filtering** — CSV-based category filters + K-means cluster hotspots
5. **Rose Pine Dawn** aesthetic — clean, academic, pastel palette
6. **Python backend** (future) — CLIP embeddings → UMAP → K-means → Atlas generation

## Non-Goals (for now)
- Real-time collaboration
- Server-side rendering
- 3D views
- Image upload from browser

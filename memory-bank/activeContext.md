# ImageSpace — Active Context

## Current State (Task 21)
All core frontend features implemented. Recent batch of fixes:
- Removed imagestack view mode (user changed mind)
- Hotspot cards hidden in timeline/carousel modes (no overlap)
- UMAP mode dims non-cluster sprites (alpha 0.12) instead of hiding
- Filters relocated to compact horizontal bar in top-right (below view tabs)
- Detail panel has right-edge tab toggle (vertical tab, centered on right side)
- Image Stack PNG download in detail panel (composites visible images)
- Single image PNG download button in detail panel
- Git version control initialized

## Active View Modes
- **UMAP** — Original 2D scatter positions (hotspot click dims, doesn't hide)
- **Grid** — Square grid layout
- **Color** — Sorted by average hue
- **Timeline** — Sorted by POSIX timestamp (hotspots hidden in this mode)
- **Carousel** — Full-screen single image with prev/next navigation

## Key Interactions
- **Hotspot cards** (left column, below logo): 8 compact cards for K-means clusters. Hidden in timeline/carousel. Click to filter.
- **CSV Filters** (top-right, horizontal bar): Compact dropdowns below view tabs. Filters intersect with hotspot.
- **Detail Panel** (right side): Toggle via right-edge tab. Shows thumbnail, metadata, download buttons.
- **Download**: Single image PNG + Stack PNG (composite all visible images with auto-opacity)
- **Tooltip**: Hover shows "Image #N" in canvas views.

## Next Steps
- Python backend pipeline (CLIP → UMAP → Atlas)
- Handle real image datasets
- Consider: search, minimap, 3D view

## Recent Decisions
- Imagestack view mode added then removed (user preference)
- Hotspots kept but hidden in timeline mode (user clarification)
- UMAP mode uses dim (alpha 0.12) not hide for non-selected clusters
- Filters moved from left column to top-right horizontal bar
- Detail panel toggle is a right-edge vertical tab, not a button in zoom bar
- Git initialized on master branch

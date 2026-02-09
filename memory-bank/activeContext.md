# ImageSpace — Active Context

## Current State (Task 18)
All core frontend features are implemented and working. Just completed a batch of 6 fixes:
1. Deleted unused `frontend/` directory (only `frontend-pixi/` remains)
2. Removed Clusters view mode (redundant with Grid + filters)
3. Fixed detail panel to be toggle-based (PanelRight button in zoom bar)
4. Timeline correctly filters with hotspot + CSV filters
5. Filters load from metadata.csv (shape, color, category, decade columns)
6. Carousel starts with the selected image ID

## Active View Modes
- **UMAP** — Original 2D scatter positions
- **Grid** — Square grid layout
- **Color** — Sorted by average hue (computed from atlas thumbnails)
- **Timeline** — Sorted by POSIX timestamp from metadata.csv
- **Carousel** — Full-screen single image with prev/next navigation

## Key Interactions
- **Hotspot cards** (left side): 8 floating cards for K-means clusters. Click to filter all views to that cluster. Click again to deselect.
- **CSV Filters** (top-left): Dropdown for each column in metadata.csv. Filters intersect with hotspot selection.
- **Detail Panel** (right side): Toggle via button in zoom bar. Shows image thumbnail, position, all CSV metadata fields.
- **Tooltip**: Hover shows "Image #N" near cursor in canvas views.

## Next Steps
- Python backend pipeline (CLIP → UMAP → Atlas)
- Handle real image datasets (not dummy colored shapes)
- Consider: search, minimap, 3D view
- Memory bank is now up to date

## Recent Decisions
- Clusters view removed (Grid + filter achieves same thing)
- Detail panel is toggle-based, not auto-open
- Timeline sorts all 50K images by POSIX timestamp with a time indicator bar
- Carousel navigates only through filtered images

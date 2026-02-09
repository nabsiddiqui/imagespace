# ImageSpace — Active Context

## Current State (Task 25)
All core frontend features implemented. Recent updates:
- Filters changed from single-select dropdowns to multi-select checkboxes (union/additive across columns)
- Timeline scrolling via wheel (horizontal panning) + interactive slider
- Timeline viewport fits to content bounds (not world bounds)
- t-SNE view mode added (uses same 2D embedding as UMAP)
- Hotspot cards made larger (240px wide, 12x12 thumbs)
- Bottom-right stats card removed (FPS available in Info modal)
- Side Panel label added to right-edge tab toggle
- Carousel auto-selects first image when nothing selected
- Timeline bar offset from hotspot column

## Active View Modes
- **UMAP** — Original 2D scatter positions (hotspot click dims, doesn't hide)
- **t-SNE** — Same 2D embedding (placeholder — needs real t-SNE data)
- **Grid** — Square grid layout
- **Color** — Sorted by average hue
- **Timeline** — Sorted by POSIX timestamp, interactive slider
- **Carousel** — Full-screen single image with prev/next navigation

## Key Interactions
- **Hotspot cards** (left column, below logo): 8 larger cards for K-means clusters. Hidden in carousel. Click to filter.
- **CSV Filters** (top-right, checkbox bar): Multi-select checkboxes per column. Values within a column union (additive). Multiple columns also union. Hotspot filter intersects with CSV union.
- **Detail Panel** (right side): Toggle via right-edge tab with "Panel" label. Shows thumbnail, metadata.
- **Tooltip**: Hover shows "Image #N" in canvas views.
- **Timeline slider**: Bottom bar with draggable range slider to scroll through time.

## Next Steps
- Python backend pipeline (CLIP → UMAP/t-SNE → Atlas)
- Handle real image datasets
- Consider: search, minimap, 3D view
- Real t-SNE embedding data (currently uses UMAP coordinates)

## Recent Decisions (Tasks 22-25)
- Download features (single + stack) removed per user request
- FPS throttled to 200ms updates (was causing 30fps display)
- Hotspots restored in timeline mode
- Filters made additive (union) not intersective
- Checkbox UI replaces dropdown UI for filters
- viewport.fit() replaced with content-bounds fitting
- t-SNE added as view mode (placeholder using UMAP data)

## t-SNE + Hotspots
Hotspots are computed via K-means clustering on the UMAP embedding coordinates. Since t-SNE currently uses the same coordinates, hotspots work identically in both. When real t-SNE data is added, hotspots would need to be recomputed for that projection, or shared across projections.

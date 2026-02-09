# ImageSpace — Tech Context

## Stack
- **React 18.2** — UI framework
- **PixiJS 8.16.0** — WebGL sprite rendering
- **pixi-viewport 6.0.3** — Pan/zoom/pinch viewport
- **Vite 5.4.21** — Build tool (LOCAL version via npx, not global v7)
- **Tailwind CSS 3.4.1** — Utility-first styling
- **lucide-react** — Icon library
- **PostCSS + autoprefixer** — CSS processing

## Build & Serve
```bash
cd frontend-pixi
npx vite build                    # → dist/ (~2.3s)
python3 -m http.server 5174 -d dist  # static file server
```

## File Structure
```
frontend-pixi/
  package.json          # dependencies
  vite.config.js        # Vite config
  tailwind.config.js    # Rose Pine Dawn colors, Inter font
  postcss.config.js     # tailwindcss + autoprefixer
  index.html            # entry HTML
  src/
    App.jsx             # Main app (~1200 lines, single component)
    main.jsx            # React root + ErrorBoundary
    index.css           # Tailwind directives + custom classes
  public/data/
    data.bin            # Binary layout (800KB for 50K images)
    manifest.json       # {count, atlasCount, thumbSize}
    metadata.csv        # Per-image metadata (any CSV columns)
    atlas_0..12.jpg     # 13 atlas textures
```

## Key Technical Constraints
- Must work as static site (no server-side logic)
- 50K images rendered as PixiJS sprites (GPU-accelerated)
- Atlas textures are JPEG to minimize bandwidth
- Binary format for layout data (16 bytes/image vs ~200 bytes JSON)
- K-means clustering runs client-side (10 clusters, 25 iterations)
- Average color computation: 1×1 canvas downscale per image

## Development Notes
- Use `npx vite build` (not `vite build`) to use local Vite 5, not global Vite 7
- Server must use `-d /absolute/path/to/dist` flag
- metadata.csv is optional — app works without it (no filters shown)
- 'timestamp' column in CSV enables Timeline view
- Dynamic imports prevent PixiJS module-level crashes in SSR/test environments

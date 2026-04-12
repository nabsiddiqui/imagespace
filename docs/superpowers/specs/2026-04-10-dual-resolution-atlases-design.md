# Dual-Resolution Progressive Atlas Loading

## Problem

The viewer loads all 49 full-resolution (128px thumbnail) atlas textures upfront (~3.1GB GPU memory), causing slow initial render and mobile crashes. Preview atlases (64px thumbnails) already exist in the data but are not used by the viewer.

## Current State

- **Pipeline** generates: `atlas_N.webp` (128px) and `atlas_N_preview.webp` (64px) via a separate script (not in `imagespace.py`)
- **Manifest** includes: `hasPreviewAtlases: true`, `previewThumbSize: 64`, `previewAtlasCount: 49`
- **Viewer** ignores preview atlases — loads full 128px atlases only
- Both atlas sets have **49 files** with the **same image-to-atlas mapping** (same `ai` per image)
- 128px atlases: 32×32 grid (1024 slots per atlas). Each 4096×4096 atlas has 1024 images.
- 64px preview atlases: same 49 atlases, same ~1012 images per atlas, but thumbnails at 64px. Since only ~1012 images per atlas, the 64-per-row layout leaves empty rows in the 64×64 grid (~16 rows used out of 64). This is acceptable — the wasted space is minor and keeps `ai` mapping identical.
- Binary format is v2 (`bytesPerImage: 24`)
- Current viewer `THUMB_SIZE = 64` is overridden by `manifest.thumbSize` (128) via `thumbSizeRef`

## Architecture

### Design Decision: Preview sprites display at 128px world size

**Both preview and full sprites display at 128px world size.** The 64px preview textures are upscaled 2x by the GPU. This means:

- No layout shift when HD swap occurs — positions and spacing never change
- Preview phase looks slightly blurry (64px stretched to 128px) but functional
- HD swap only changes sharpness, not position or size
- `computeLayout` uses `thumbSizeRef.current = 128` throughout — no phase-dependent logic

In PixiJS:
```js
// Phase 1 (preview): 64px texture source, 128px display
sprite.texture = new PIXI.Texture({ source: previewTex.source, frame: new PIXI.Rectangle(u_preview, v_preview, 64, 64) });
sprite.width = 128;
sprite.height = 128;

// Phase 2 (HD swap): 128px texture source, 128px display — same position, crisper
sprite.texture = new PIXI.Texture({ source: fullTex.source, frame: new PIXI.Rectangle(u, v, 128, 128) });
sprite.width = 128;  // unchanged
sprite.height = 128; // unchanged
```

### Atlas Mapping: Shared `ai` Across Tiers

Both preview and full atlases use the **same `ai` (atlas index)** per image. This is critical for the progressive swap: when full atlas N loads, we can directly upgrade all sprites that reference `ai=N`.

Since 128px atlases hold 1024 images per atlas (32×32 grid in 4096×4096), the same image-to-atlas assignment is used for both tiers. For preview atlases:
- `u_preview = (localIndex % 64) * 64` — position in the 64-per-row layout
- `v_preview = Math.floor(localIndex / 64) * 64` — position in the 64-per-row layout
- Where `localIndex = globalIndex - ai * 1024`

This means preview atlas layout uses only ~16 rows of 64 (for ~1012 images), leaving the bottom ~48 rows empty. This wastes ~75% of each preview atlas texture but keeps `ai` mapping identical across tiers — a critical simplification for the progressive swap.

The `u_preview` and `v_preview` values are pre-computed by the pipeline and stored in `data.bin` rather than computed at runtime, keeping the viewer simple.

### Data Format: v3 Binary (28 bytes per image)

| Offset | Type | Field |
|---|---|---|
| 0 | float32 | x (snapped/grid) |
| 4 | float32 | y (snapped/grid) |
| 8 | float32 | tsneX (raw t-SNE) |
| 12 | float32 | tsneY (raw t-SNE) |
| 16 | uint16 | ai (atlas index — same for both tiers) |
| 18 | uint16 | u (full-res x-offset in atlas) |
| 20 | uint16 | v (full-res y-offset in atlas) |
| 22 | uint16 | cluster |
| 24 | uint16 | u_preview (64px x-offset in preview atlas) |
| 26 | uint16 | v_preview (64px y-offset in preview atlas) |

`bytesPerImage` changes from 24 to 28, `version` bumps from 2 to 3.

### v3 Viewer Parsing

The viewer currently handles v1 (16 bytes) and v2 (24 bytes). v3 (28 bytes) must be added explicitly **before** the v1 `else` branch. The current `else` branch defaults to v1 (16 bytes), which would silently corrupt v3 data.

```js
const isV2 = bytesPerImage === 24;
const isV3 = bytesPerImage === 28;

for (let i = 0; i < manifest.count; i++) {
  let x, y, tsneX, tsneY, ai, u, v, cluster, u_preview, v_preview;
  if (isV3) {
    const offset = i * 28;
    x = dataView.getFloat32(offset, true);
    y = dataView.getFloat32(offset + 4, true);
    tsneX = dataView.getFloat32(offset + 8, true);
    tsneY = dataView.getFloat32(offset + 12, true);
    ai = dataView.getUint16(offset + 16, true);
    u = dataView.getUint16(offset + 18, true);
    v = dataView.getUint16(offset + 20, true);
    cluster = dataView.getUint16(offset + 22, true);
    u_preview = dataView.getUint16(offset + 24, true);
    v_preview = dataView.getUint16(offset + 26, true);
  } else if (isV2) {
    // ... existing v2 parsing (24 bytes per image)
  } else {
    // ... existing v1 parsing (16 bytes per image)
  }
  allPointData[i] = { id: i, x, y, tsneX, tsneY, ai, u, v, cluster, u_preview, v_preview };
}
```

### Manifest

```json
{
  "count": 49585,
  "atlasCount": 49,
  "thumbSize": 128,
  "atlasSize": 4096,
  "bytesPerImage": 28,
  "version": 3,
  "atlasFormat": "webp",
  "hasPreviewAtlases": true,
  "previewThumbSize": 64,
  "previewAtlasCount": 49
}
```

### Viewer: Phase 1 — Preview Load

1. Read manifest. If `hasPreviewAtlases` is true and `version >= 3`:
2. Load all `atlas_N_preview.webp` files progressively (concurrency: 4 desktop, 2 mobile)
3. Create sprites at **128px display size** with 64px preview texture frames (upscaled 2x)
4. Use `u_preview, v_preview` from v3 binary for frame positions
5. Store `u_preview, v_preview` on each point object for potential re-texturing
6. Full map visible within seconds

```js
// Preview sprite creation
const previewThumbSize = manifest.previewThumbSize || 64;
const fullThumbSize = manifest.thumbSize || 128;

// Phase 1: preview frame
const frame = new PIXI.Rectangle(p.u_preview, p.v_preview, previewThumbSize, previewThumbSize);
const spriteTex = new PIXI.Texture({ source: previewTex.source, frame });
const sprite = new PIXI.Sprite(spriteTex);
sprite.width = fullThumbSize;  // 128 — display at full size even with preview texture
sprite.height = fullThumbSize;
```

**Mobile downscale:** Preview atlases are 4096×4096 on disk. Mobile downscale uses a separate `MOBILE_PREVIEW_SCALE = 0.25` (not the existing `MOBILE_ATLAS_SCALE = 0.5`), producing 1024×1024 textures. This gives:

| Device | Preview (downscaled) | Full (never loaded) | Total |
|---|---|---|---|
| Desktop | ~800MB (4096 native) | swaps in progressively, evicts previews | ~3.1GB peak |
| Mobile (0.25 scale) | ~196MB (49 × 1024²) | N/A | ~196MB |

### Viewer: Phase 2 — Progressive HD Upgrade

After all preview atlases loaded:

1. Start background loading of full `atlas_N.webp` files (concurrency 2)
2. For each full atlas loaded, swap all sprites from preview → full texture:
   - Frame changes from `(u_preview, v_preview, 64, 64)` to `(u, v, 128, 128)`
   - Sprite **width and height remain 128px** — no layout shift
   - The 128px source provides more detail per display pixel = crisper when zoomed
3. After all sprites in an atlas are upgraded, evict the preview texture via `PIXI.Assets.unload()`
4. Update the `atlasTextures` array reference to point to the full texture (so `extractClusterThumbs` and `computeAvgColors` use the HD texture for subsequent operations)

```js
async function upgradeAtlasToFull(atlasIdx) {
  const previewUrl = `${BASE}data/atlas_${atlasIdx}_preview.webp`;
  const fullUrl = `${BASE}data/atlas_${atlasIdx}.webp`;
  const fullTex = await PIXI.Assets.load(fullUrl);

  for (const i of pointsByAtlas[atlasIdx]) {
    const p = pointsRef.current[i];
    const frame = new PIXI.Rectangle(p.u, p.v, fullThumbSize, fullThumbSize);
    p.sprite.texture = new PIXI.Texture({ source: fullTex.source, frame });
  }

  // Update atlas reference so extractClusterThumbs/computeAvgColors use full texture
  atlasTextures[atlasIdx] = fullTex;

  // Evict preview texture after a frame delay (ensures no canvas operations reference it)
  requestAnimationFrame(() => {
    PIXI.Assets.unload(previewUrl);
  });
}
```

**Mobile:** Never loads full 128px atlases. Stays on 64px previews at 1024×1024 scale.

### Detail Panel: Use PixiJS Texture Source

The `DetailThumb` component currently loads a full atlas as a separate HTML `<img>` element. This is problematic because:
- During preview phase, the full atlas hasn't been loaded yet (would trigger a separate download)
- On mobile, `point.u` and `point.v` are pre-multiplied by `MOBILE_ATLAS_SCALE`, but `DetailThumb` draws from the unscaled image

**Fix:** `DetailThumb` should render from the currently loaded PixiJS texture source (already in GPU memory and correctly scaled for mobile) rather than loading a separate HTML Image.

```js
// Instead of loading a separate <img>, draw from the PixiJS texture source
const DetailThumb = React.memo(({ point, thumbSize, atlasTextures, displaySize = 256 }) => {
  const canvasRef = useRef(null);
  useEffect(() => {
    if (!point) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const tex = atlasTextures[point.ai];
    if (!tex) return;
    const src = tex.source.resource; // HTMLImageElement or HTMLCanvasElement
    ctx.clearRect(0, 0, displaySize, displaySize);
    ctx.drawImage(src, point.u, point.v, thumbSize, thumbSize, 0, 0, displaySize, displaySize);
  }, [point?.ai, point?.u, point?.v, thumbSize, displaySize]);
  // ...
});
```

This also requires passing `atlasTextures` to the `DetailThumb` component, which currently receives the atlas format string.

**Phase-aware coordinates:** During Phase 1 (preview), `DetailThumb` must use `point.u_preview, point.v_preview` with `previewThumbSize` (64). After HD swap, it uses `point.u, point.v` with `fullThumbSize` (128). Since the texture source is always the current tier (swapped atomically via `atlasTextures[idx] = fullTex`), the coordinates must match the current tier.

**Simple solution:** Store the current drawing parameters on each point object. When creating sprites in Phase 1, set `p.drawU = p.u_preview; p.drawV = p.v_preview; p.drawSize = previewThumbSize`. When upgrading to HD, update `p.drawU = p.u; p.drawV = p.v; p.drawSize = fullThumbSize`. `DetailThumb` always reads `p.drawU, p.drawV, p.drawSize`.

**Mobile scaling:** On mobile with `MOBILE_PREVIEW_SCALE = 0.25`, the preview atlas is downscaled from 4096→1024. The `u_preview` and `v_preview` values (computed for 4096×4096) must be scaled: `scaledU = p.drawU * mobileScale; scaledV = p.drawV * mobileScale; scaledSize = p.drawSize * mobileScale`. This mirrors the existing `atlasScale` logic at App.jsx:877.

### Hover Detection

Current hover detection (`App.jsx:1190`) uses a 50px hardcoded radius. Since sprites display at 128px in both phases, update to scale with thumb size:

```js
let minDistSq = (thumbSizeRef.current * 0.4 / viewport.scale.x) ** 2;
```

Also replace hardcoded `THUMB_SIZE` references in cluster label positioning and `handleFitAll` with `thumbSizeRef.current`:
- `App.jsx:557`: `THUMB_SIZE * 2.5` → `thumbSizeRef.current * 2.5`
- `App.jsx:1336`: `THUMB_SIZE * 2` → `thumbSizeRef.current * 2`

### Pipeline Changes (`imagespace.py`)

Add `--hd` flag to generate both atlas tiers. When present:

1. Generate 128px atlases as `atlas_N.webp` (current behavior with `--thumb-size 128`)
2. Generate 64px preview atlases as `atlas_N_preview.webp` using the **same image-to-atlas mapping** (same `ai` per image). Each preview atlas has the same images in the same order, but at 64px thumbnails in a 64-per-row layout.
3. Compute `u_preview, v_preview` for each image: `u_preview = (localIndex % 64) * 64`, `v_preview = floor(localIndex / 64) * 64`, where `localIndex = imageIndex - ai * imagesPerFullAtlas`
4. Write `data.bin` with 28 bytes per image (adding u_preview, v_preview)
5. Write manifest with version 3, `hasPreviewAtlases: true`, `previewThumbSize: 64`
6. Quality: preview atlases at WebP quality 40, full atlases at quality 60

When absent (default): current v2 behavior, single resolution only.

### Memory Budget

| Phase | Desktop | Mobile |
|---|---|---|
| Preview load | ~800MB (49 × 4096² native) | ~196MB (49 × 1024² at 0.25 scale) |
| Full swap (per atlas) | +6MB full, -2MB preview evicted | N/A |
| Full complete | ~3.1GB (previews evicted) | ~196MB (preview only) |
| Peak (mid-swap) | ~3.1GB + 1 atlas overlap | ~196MB |

### Fallback

- v2 manifests (`bytesPerImage=24`): Load full atlases directly, current behavior (unchanged)
- v3 manifests without `hasPreviewAtlases`: Load full atlases directly, skip preview
- v3 manifests with `hasPreviewAtlases`: Use progressive loading flow

### Race Conditions

During the progressive HD upgrade, a user might interact with a sprite whose atlas is mid-upgrade. To prevent issues:
- `extractClusterThumbs` and `computeAvgColors` read from `atlasTextures[p.ai]`, which is updated atomically when the atlas swaps
- `DetailThumb` draws from `atlasTextures[point.ai]`, which always points to the current tier (preview or full)
- `PIXI.Assets.unload(previewUrl)` is deferred via `requestAnimationFrame()` to ensure no canvas operations still reference the texture

## What This Does NOT Change

- Layout algorithms and world coordinates (always 128px spacing via `thumbSizeRef`)
- Clustering (unchanged)
- `thumbSizeRef.current` — always set to manifest.thumbSize (128)
- v2 manifest compatibility (unchanged, falls back to single-resolution)
- The progressive atlas loading pattern (same concurrency approach)

## Open Questions

1. **Preview WebP quality:** Recommend 40 (smaller files for fast initial download).
2. **Full atlas quality:** Keep at 60 (current).
3. **HD load priority:** Load in order (0→48) or prioritize visible viewport? Viewport-priority is better UX but more complex to implement.
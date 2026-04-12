# Dual-Resolution Progressive Atlas Loading — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add progressive atlas loading so the viewer loads 64px preview atlases first, then swaps to 128px full atlases in the background, while keeping mobile on preview-only mode.

**Architecture:** Pipeline generates both 64px preview and 128px full atlases with same `ai` mapping. Viewer loads previews first (fast), then progressively swaps sprites to HD textures. Binary format extends from v2 (24 bytes) to v3 (28 bytes) adding `u_preview`/`v_preview`.

**Tech Stack:** Python (pipeline), React + PixiJS v8 (viewer)

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/imagespace.py` | Pipeline: `--hd` flag, preview atlas generation, v3 binary + manifest |
| `image_space/src/App.jsx` | Viewer: v3 parsing, preview-first loading, HD upgrade, DetailThumb fix, hover fix |

---

### Task 1: Pipeline — Add `--hd` flag and preview atlas generation

**Files:**
- Modify: `scripts/imagespace.py:76-122` (generate_atlases function)
- Modify: `scripts/imagespace.py:923-1057` (main function, argument parser)

- [ ] **Step 1: Add `--hd` argument to argparse**

In `scripts/imagespace.py`, add after the `--relayout` argument (line 937):

```python
parser.add_argument('--hd', action='store_true', help='Generate both 64px preview and 128px full atlases for progressive loading')
parser.add_argument('--preview-quality', type=int, default=40, help='WebP quality for preview atlases (default 40)')
```

- [ ] **Step 2: Add preview atlas generation function**

Add a new function after `generate_atlases` (after line 122):

```python
def generate_preview_atlases(images, output_dir, atlas_count_full, images_per_full_atlas, thumb_size=64, atlas_size=ATLAS_SIZE, quality=40):
    """Generate 64px preview atlas textures mirroring the same image-to-atlas mapping as full atlases.
    Each preview atlas contains the SAME images as its corresponding full atlas (same ai),
    but with 64px thumbnails in a 64-per-row layout."""
    images_per_row = atlas_size // thumb_size

    preview_data = []
    current_atlas_idx = 0

    start = time.time()
    for atlas_idx in range(atlas_count_full):
        start_img = atlas_idx * images_per_full_atlas
        end_img = min(start_img + images_per_full_atlas, len(images))

        atlas_img = Image.new('RGB', (atlas_size, atlas_size), (255, 255, 255))
        count_in_atlas = 0

        for img_idx in range(start_img, end_img):
            if img_idx % 2000 == 0:
                elapsed = time.time() - start
                rate = img_idx / elapsed if elapsed > 0 else 0
                print(f"  Preview thumb {img_idx}/{len(images)} ({rate:.0f} img/s)", end='\r')

            try:
                img = Image.open(images[img_idx]).convert('RGB')
                w, h = img.size
                side = min(w, h)
                left = (w - side) // 2
                top = (h - side) // 2
                img = img.crop((left, top, left + side, top + side))
                img = img.resize((thumb_size, thumb_size), Image.BILINEAR)
            except Exception:
                img = Image.new('RGB', (thumb_size, thumb_size), (128, 128, 128))

            local_idx = img_idx - start_img
            col = local_idx % images_per_row
            row = local_idx // images_per_row
            u, v = col * thumb_size, row * thumb_size
            atlas_img.paste(img, (u, v))

            preview_data.append((atlas_idx, u, v))
            count_in_atlas += 1

        atlas_path = os.path.join(output_dir, f'atlas_{atlas_idx}_preview.webp')
        atlas_img.save(atlas_path, 'WEBP', quality=quality, method=2)
        print(f"\n  Saved atlas_{atlas_idx}_preview.webp ({count_in_atlas} images)")

    total = time.time() - start
    print(f"  Preview atlas generation: {total:.1f}s")
    return preview_data, atlas_count_full
```

- [ ] **Step 3: Call preview atlas generation in main() when `--hd` flag is set**

In `main()`, after the full atlas generation block (after line 982), add:

```python
    preview_atlas_data = None
    if args.hd and not args.relayout:
        print(f"\n[2b/8] Generating preview atlases (64px, quality={args.preview_quality})...")
        images_per_full_atlas = (args.atlas_size // args.thumb_size) ** 2
        preview_atlas_data, preview_atlas_count = generate_preview_atlases(
            images, str(output_dir), atlas_count, images_per_full_atlas,
            thumb_size=64, atlas_size=args.atlas_size, quality=args.preview_quality
        )
```

- [ ] **Step 4: Update `write_binary_data` to support v3 format**

Replace `write_binary_data` (lines 809-825) with:

```python
def write_binary_data(output_dir, snapped_coords, raw_tsne_coords, atlas_data, cluster_ids, preview_atlas_data=None):
    """Write binary layout data. v2=24 bytes, v3=28 bytes (with preview UVs)."""
    if preview_atlas_data is not None:
        bytes_per_image = 28
        binary_data = bytearray(len(snapped_coords) * bytes_per_image)
        for i in range(len(snapped_coords)):
            ai, u, v = atlas_data[i]
            _, u_preview, v_preview = preview_atlas_data[i]
            cid = int(cluster_ids[i])
            sx, sy = float(snapped_coords[i][0]), float(snapped_coords[i][1])
            rx, ry = float(raw_tsne_coords[i][0]), float(raw_tsne_coords[i][1])
            struct.pack_into('<ffffHHHHHH', binary_data, i * bytes_per_image,
                sx, sy, rx, ry, ai, u, v, cid, u_preview, v_preview)
    else:
        bytes_per_image = 24
        binary_data = bytearray(len(snapped_coords) * bytes_per_image)
        for i in range(len(snapped_coords)):
            ai, u, v = atlas_data[i]
            cid = int(cluster_ids[i])
            sx, sy = float(snapped_coords[i][0]), float(snapped_coords[i][1])
            rx, ry = float(raw_tsne_coords[i][0]), float(raw_tsne_coords[i][1])
            struct.pack_into('<ffffHHHH', binary_data, i * bytes_per_image,
                sx, sy, rx, ry, ai, u, v, cid)

    output_path = os.path.join(output_dir, 'data.bin')
    with open(output_path, 'wb') as f:
        f.write(binary_data)
    print(f"  Binary data (v{'3' if preview_atlas_data else '2'}): {len(binary_data) / 1024:.1f} KB")
    return output_path
```

- [ ] **Step 5: Update `write_manifest` for v3**

Replace `write_manifest` (lines 828-842) with:

```python
def write_manifest(output_dir, count, atlas_count, thumb_size=THUMB_SIZE, atlas_size=ATLAS_SIZE,
                   preview_atlas_count=None, preview_thumb_size=64):
    """Write manifest.json for the viewer."""
    is_hd = preview_atlas_count is not None
    manifest = {
        'count': count,
        'atlasCount': atlas_count,
        'thumbSize': thumb_size,
        'atlasSize': atlas_size,
        'bytesPerImage': 28 if is_hd else 24,
        'version': 3 if is_hd else 2,
        'atlasFormat': 'webp',
    }
    if is_hd:
        manifest['hasPreviewAtlases'] = True
        manifest['previewThumbSize'] = preview_thumb_size
        manifest['previewAtlasCount'] = preview_atlas_count
    output_path = os.path.join(output_dir, 'manifest.json')
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest (v{'3' if is_hd else '2'}): {output_path}")
```

- [ ] **Step 6: Update `main()` to pass preview data through**

In `main()`, update the write_binary_data call (around line 1041):

```python
    write_binary_data(str(output_dir), tsne_coords, raw_tsne_coords, atlas_data, cluster_ids, preview_atlas_data)
```

And the write_manifest call (around line 1042):

```python
    write_manifest(str(output_dir), len(images), atlas_count, args.thumb_size, args.atlas_size,
                   preview_atlas_count=preview_atlas_count if preview_atlas_data else None,
                   preview_thumb_size=64)
```

Also initialize `preview_atlas_count` before the if-block if needed. Add near line 950:

```python
    preview_atlas_count = None
```

- [ ] **Step 7: Test the pipeline with `--hd` flag**

Run: `cd /Users/nabeel/GDrive/Spring\ 2026/ImageSpace && python scripts/imagespace.py <test_images_dir> --output ./test_output --hd`
Expected: Generates both `atlas_N.webp` and `atlas_N_preview.webp`, `data.bin` with 28 bytes/image, manifest with v3.

- [ ] **Step 8: Commit**

```bash
git add scripts/imagespace.py
git commit -m "feat(pipeline): add --hd flag for dual-resolution atlas generation with v3 binary format"
```

---

### Task 2: Viewer — Add v3 binary parsing

**Files:**
- Modify: `image_space/src/App.jsx:718-747` (binary parsing block)

- [ ] **Step 1: Add v3 detection and parsing**

Replace the binary parsing block (lines 718-747). Change the `const isV2 = bytesPerImage === 24;` line and the entire for-loop:

```js
        const currentThumbSize = manifest.thumbSize || THUMB_SIZE;
        const bytesPerImage = manifest.bytesPerImage || 16;
        const isV2 = bytesPerImage === 24;
        const isV3 = bytesPerImage === 28;

        const allPointData = new Array(manifest.count);
        for (let i = 0; i < manifest.count; i++) {
          let x, y, tsneX, tsneY, ai, u, v, cluster, u_preview, v_preview;
          if (isV3) {
            const offset = i * 28;
            x      = dataView.getFloat32(offset, true);
            y      = dataView.getFloat32(offset + 4, true);
            tsneX  = dataView.getFloat32(offset + 8, true);
            tsneY  = dataView.getFloat32(offset + 12, true);
            ai     = dataView.getUint16(offset + 16, true);
            u      = dataView.getUint16(offset + 18, true);
            v      = dataView.getUint16(offset + 20, true);
            cluster = dataView.getUint16(offset + 22, true);
            u_preview = dataView.getUint16(offset + 24, true);
            v_preview = dataView.getUint16(offset + 26, true);
          } else if (isV2) {
            const offset = i * 24;
            x      = dataView.getFloat32(offset, true);
            y      = dataView.getFloat32(offset + 4, true);
            tsneX  = dataView.getFloat32(offset + 8, true);
            tsneY  = dataView.getFloat32(offset + 12, true);
            ai     = dataView.getUint16(offset + 16, true);
            u      = dataView.getUint16(offset + 18, true);
            v      = dataView.getUint16(offset + 20, true);
            cluster = dataView.getUint16(offset + 22, true);
          } else {
            const offset = i * 16;
            x  = dataView.getFloat32(offset, true);
            y  = dataView.getFloat32(offset + 4, true);
            ai = dataView.getUint16(offset + 8, true);
            u  = dataView.getUint16(offset + 10, true);
            v  = dataView.getUint16(offset + 12, true);
            tsneX = x; tsneY = y;
            cluster = undefined;
          }
          allPointData[i] = { id: i, x, y, tsneX, tsneY, ai, u, v, cluster, u_preview, v_preview };
        }
```

- [ ] **Step 2: Verify v2/v1 fallback still works**

Load the viewer with the existing v2 manifest. Expected: same behavior as before, `u_preview` and `v_preview` are `undefined`.

- [ ] **Step 3: Commit**

```bash
git add image_space/src/App.jsx
git commit -m "feat(viewer): add v3 binary parsing with u_preview/v_preview fields"
```

---

### Task 3: Viewer — Preview-first atlas loading (Phase 1)

**Files:**
- Modify: `image_space/src/App.jsx:11-23` (constants)
- Modify: `image_space/src/App.jsx:818-910` (atlas loading block)

- [ ] **Step 1: Add constants for preview mode**

After `MOBILE_CONCURRENCY = 2;` (line 23), add:

```js
const MOBILE_PREVIEW_SCALE = 0.25;
```

- [ ] **Step 2: Replace the atlas loading block with preview-first logic**

Replace the atlas loading block from the `/* Progressive atlas loading */` comment (line 818) through to `setLoadingAtlases(false);` (line 910). This is the main change:

```js
        const usePreview = isV3 && manifest.hasPreviewAtlases;
        const previewThumbSize = manifest.previewThumbSize || 64;
        const previewAtlasCount = manifest.previewAtlasCount || manifest.atlasCount;
        const previewAtlasScale = isMobile ? MOBILE_PREVIEW_SCALE : 1;

        const CONCURRENCY = isMobile ? MOBILE_CONCURRENCY : 4;
        const atlasTextures = new Array(manifest.atlasCount);
        let atlasLoaded = 0;
        const atlasScale = usePreview ? previewAtlasScale : (isMobile ? MOBILE_ATLAS_SCALE : 1);

        setLoading(false);
        setLoadingAtlases(true);
        setLoadProgress(0);

        async function loadAtlasAndCreateSprites(atlasIdx) {
          let atlasUrl, texThumbSize, texU, texV;
          if (usePreview) {
            atlasUrl = `${BASE}data/atlas_${atlasIdx}_preview.webp`;
            texThumbSize = previewThumbSize;
          } else {
            atlasUrl = `${BASE}data/atlas_${atlasIdx}.${fmt}`;
            texThumbSize = currentThumbSize;
          }
          atlasUrls.push(atlasUrl);
          let tex = await PIXI.Assets.load(atlasUrl);
          if (isCancelled) return;

          if (atlasScale < 1) {
            const src = tex.source;
            const sw = src.width, sh = src.height;
            const dw = Math.round(sw * atlasScale), dh = Math.round(sh * atlasScale);
            const offscreen = document.createElement('canvas');
            offscreen.width = dw;
            offscreen.height = dh;
            const ctx2d = offscreen.getContext('2d');
            ctx2d.drawImage(src.resource, 0, 0, dw, dh);
            await PIXI.Assets.unload(atlasUrl);
            atlasUrls = atlasUrls.filter(u => u !== atlasUrl);
            tex = PIXI.Texture.from(offscreen);
          }
          atlasTextures[atlasIdx] = tex;

          for (const i of pointsByAtlas[atlasIdx]) {
            const pd = allPointData[i];
            let frameU, frameV, frameSize;
            if (usePreview) {
              frameU = (pd.u_preview ?? pd.u) * atlasScale;
              frameV = (pd.v_preview ?? pd.v) * atlasScale;
              frameSize = previewThumbSize * atlasScale;
            } else {
              frameU = pd.u * atlasScale;
              frameV = pd.v * atlasScale;
              frameSize = currentThumbSize * atlasScale;
            }
            const frame = new PIXI.Rectangle(frameU, frameV, frameSize, frameSize);
            const spriteTex = new PIXI.Texture({ source: tex.source, frame });
            const sprite = new PIXI.Sprite(spriteTex);
            sprite.anchor.set(0.5);
            const initX = pd.tsneX ?? pd.x;
            const initY = pd.tsneY ?? pd.y;
            sprite.position.set(initX, initY);
            sprite.eventMode = 'none';
            container.addChild(sprite);

            const drawU = usePreview ? (pd.u_preview ?? pd.u) : pd.u;
            const drawV = usePreview ? (pd.v_preview ?? pd.v) : pd.v;
            const drawSize = usePreview ? previewThumbSize : currentThumbSize;

            const pObj = {
              id: pd.id, x: initX, y: initY,
              originalX: pd.x, originalY: pd.y,
              tsneX: pd.tsneX, tsneY: pd.tsneY,
              targetX: initX, targetY: initY,
              ai: pd.ai, u: pd.u * atlasScale, v: pd.v * atlasScale, sprite,
              drawU: drawU * atlasScale, drawV: drawV * atlasScale, drawSize: drawSize * atlasScale,
              ...(pd.cluster !== undefined && { cluster: pd.cluster }),
            };
            pointsRef.current[pd.id] = pObj;

            const gx = Math.floor(initX / SPATIAL_CELL_SIZE);
            const gy = Math.floor(initY / SPATIAL_CELL_SIZE);
            const key = `${gx},${gy}`;
            if (!spatialHashRef.current[key]) spatialHashRef.current[key] = [];
            spatialHashRef.current[key].push(pObj);
          }

          atlasLoaded++;
          const totalAtlases = usePreview ? previewAtlasCount : manifest.atlasCount;
          setLoadProgress(Math.round((atlasLoaded / totalAtlases) * 100));
          setStatusMsg(usePreview
            ? `Loading preview atlases... ${atlasLoaded}/${totalAtlases}`
            : `Loading atlas textures... ${atlasLoaded}/${totalAtlases}`);
        }

        {
          let nextIdx = 0;
          const totalCount = usePreview ? previewAtlasCount : manifest.atlasCount;
          async function worker() {
            while (nextIdx < totalCount && !isCancelled) {
              const idx = nextIdx++;
              await loadAtlasAndCreateSprites(idx);
            }
          }
          const workers = [];
          for (let w = 0; w < Math.min(CONCURRENCY, totalCount); w++) {
            workers.push(worker());
          }
          await Promise.all(workers);
        }
        if (isCancelled) return;
        setLoadingAtlases(false);
```

**Important:** PixiJS sprites default to their texture frame size. With 128px frames this is 128px (correct), but with 64px preview frames the sprite would display at 64px (too small). Force 128px display by adding after `sprite.position.set(initX, initY);`:

```js
            if (usePreview) {
              sprite.width = currentThumbSize;
              sprite.height = currentThumbSize;
            }
```

- [ ] **Step 3: Test preview loading with v3 data**

Build the viewer, load with a v3 manifest. Expected: preview atlases load first (fast), map appears with slightly blurry thumbnails, no layout issues.

- [ ] **Step 4: Commit**

```bash
git add image_space/src/App.jsx
git commit -m "feat(viewer): preview-first atlas loading for v3 manifests with 64px previews"
```

---

### Task 4: Viewer — Progressive HD upgrade (Phase 2)

**Files:**
- Modify: `image_space/src/App.jsx:910-923` (after atlas loading, before cluster thumbs)

- [ ] **Step 1: Add HD upgrade logic after preview loading completes**

After `setLoadingAtlases(false);` and before the `extractClusterThumbs` call, add the HD upgrade path:

```js
        if (usePreview && !isMobile) {
          setStatusMsg('Upgrading to HD atlases...');
          (async () => {
            const HD_CONCURRENCY = 2;
            let hdNextIdx = 0;
            let hdLoaded = 0;

            async function upgradeAtlasToFull(atlasIdx) {
              const previewUrl = `${BASE}data/atlas_${atlasIdx}_preview.webp`;
              const fullUrl = `${BASE}data/atlas_${atlasIdx}.${fmt}`;
              const fullTex = await PIXI.Assets.load(fullUrl);
              if (isCancelled) return;

              for (const i of pointsByAtlas[atlasIdx]) {
                const p = pointsRef.current[i];
                if (!p) continue;
                const frame = new PIXI.Rectangle(
                  p.u, p.v,
                  currentThumbSize, currentThumbSize
                );
                p.sprite.texture = new PIXI.Texture({ source: fullTex.source, frame });
                p.drawU = p.u;
                p.drawV = p.v;
                p.drawSize = currentThumbSize;
              }

              atlasTextures[atlasIdx] = fullTex;
              hdLoaded++;
              if (hdLoaded % 5 === 0) {
                setStatusMsg(`HD upgrade... ${hdLoaded}/${manifest.atlasCount}`);
              }

              requestAnimationFrame(() => {
                PIXI.Assets.unload(previewUrl);
              });
            }

            async function hdWorker() {
              while (hdNextIdx < manifest.atlasCount && !isCancelled) {
                const idx = hdNextIdx++;
                await upgradeAtlasToFull(idx);
              }
            }
            const hdWorkers = [];
            for (let w = 0; w < Math.min(HD_CONCURRENCY, manifest.atlasCount); w++) {
              hdWorkers.push(hdWorker());
            }
            await Promise.all(hdWorkers);
            if (!isCancelled) setStatusMsg('HD ready');
          })();
        }
```

Note: `p.u` and `p.v` store the original 128px UVs (`pd.u * atlasScale` where `atlasScale = 1` on desktop for the preview path). `p.drawU/p.drawV` store the preview UVs. The HD upgrade reads `p.u, p.v` for the full-resolution frame, which is correct.

- [ ] **Step 2: Test HD upgrade**

Load the viewer with v3 data on desktop. Expected: preview loads fast, then HD textures swap in progressively (images become sharper). No layout shift.

- [ ] **Step 3: Commit**

```bash
git add image_space/src/App.jsx
git commit -m "feat(viewer): progressive HD atlas upgrade after preview loading"
```

---

### Task 5: Viewer — Fix DetailThumb to use PixiJS texture source

**Files:**
- Modify: `image_space/src/App.jsx:38-63` (DetailThumb component)
- Modify: `image_space/src/App.jsx` (wherever DetailThumb is rendered, pass atlasTextures)

- [ ] **Step 1: Rewrite DetailThumb to use atlasTextures**

Replace the DetailThumb component (lines 38-63):

```js
const DetailThumb = React.memo(({ point, atlasTextures, displaySize = 256 }) => {
  const canvasRef = useRef(null);
  useEffect(() => {
    if (!point) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const tex = atlasTextures[point.ai];
    if (!tex) return;
    const src = tex.source.resource;
    if (!src) return;
    ctx.clearRect(0, 0, displaySize, displaySize);
    ctx.drawImage(src, point.drawU, point.drawV, point.drawSize, point.drawSize, 0, 0, displaySize, displaySize);
  }, [point?.ai, point?.drawU, point?.drawV, point?.drawSize, displaySize]);
  return (
    <canvas
      ref={canvasRef}
      width={displaySize}
      height={displaySize}
      className="rounded-xl border border-rp-hlMed mx-auto bg-rp-hlLow"
      style={{ width: displaySize, height: displaySize, imageRendering: 'auto' }}
    />
  );
});
```

- [ ] **Step 2: Update DetailThumb usage to pass atlasTextures**

Search for `<DetailThumb` in the JSX and update its props. It currently receives `thumbSize={thumbSizeRef.current}` and `atlasFormat={atlasFormatRef.current}`. Change to pass `atlasTextures` ref instead. Find the render location and change:

```jsx
<DetailThumb point={selectedItem} atlasTextures={atlasTexturesRef.current} displaySize={256} />
```

Add a new ref near the other refs (around line 411):

```js
const atlasTexturesRef = useRef(null);
```

And set it after atlasTextures is populated (after `atlasTextures[atlasIdx] = tex;` in the load function):

```js
atlasTexturesRef.current = atlasTextures;
```

Also remove the `thumbSize` and `atlasFormat` props from the DetailThumb usage.

- [ ] **Step 3: Test detail panel**

Click an image. Expected: detail panel shows the thumbnail from the currently loaded atlas tier (preview or HD). No separate network request for the full atlas.

- [ ] **Step 4: Commit**

```bash
git add image_space/src/App.jsx
git commit -m "fix(viewer): DetailThumb uses PixiJS texture source instead of separate HTML img"
```

---

### Task 6: Viewer — Replace hardcoded THUMB_SIZE references

**Files:**
- Modify: `image_space/src/App.jsx:1190` (hover detection)
- Modify: `image_space/src/App.jsx:557` (cluster label positioning)
- Modify: `image_space/src/App.jsx:1336` (handleFitAll)

- [ ] **Step 1: Fix hover detection radius**

Change line 1190 from:

```js
          let minDistSq = (50 / viewport.scale.x) ** 2;
```

to:

```js
          let minDistSq = (thumbSizeRef.current * 0.4 / viewport.scale.x) ** 2;
```

- [ ] **Step 2: Fix cluster label positioning**

Change line 557 from:

```js
          worldY: data.minY - THUMB_SIZE * 2.5,
```

to:

```js
          worldY: data.minY - thumbSizeRef.current * 2.5,
```

- [ ] **Step 3: Fix handleFitAll**

Change line 1336 from:

```js
      const pad = THUMB_SIZE * 2;
```

to:

```js
      const pad = thumbSizeRef.current * 2;
```

- [ ] **Step 4: Search for other THUMB_SIZE uses and verify**

Search for all remaining `THUMB_SIZE` uses in the file. Any that affect layout or positioning should use `thumbSizeRef.current`. Constants like `SPATIAL_CELL_SIZE` can remain as-is.

- [ ] **Step 5: Commit**

```bash
git add image_space/src/App.jsx
git commit -m "fix(viewer): replace hardcoded THUMB_SIZE with thumbSizeRef for correct 128px layout"
```

---

### Task 7: End-to-end test and cleanup

**Files:**
- All files modified in tasks 1-6

- [ ] **Step 1: Run the pipeline with `--hd` on a test dataset**

```bash
cd /Users/nabeel/GDrive/Spring\ 2026/ImageSpace
python scripts/imagespace.py <test_images> --output ./test_hd_output --hd
```

Verify: both `atlas_N.webp` and `atlas_N_preview.webp` files exist, `data.bin` is 28 bytes × image count, manifest shows v3 with `hasPreviewAtlases: true`.

- [ ] **Step 2: Build and serve the viewer with v3 data**

```bash
cd /Users/nabeel/GDrive/Spring\ 2026/ImageSpace/image_space
npm run build
```

Copy test output to `public/data/` and load in browser.

- [ ] **Step 3: Verify v2 backward compatibility**

Load with the existing v2 manifest. Expected: same behavior as before, no preview loading.

- [ ] **Step 4: Verify mobile behavior**

Open in mobile browser or device emulator. Expected: loads preview atlases at 0.25 scale, no HD upgrade, low memory usage.

- [ ] **Step 5: Verify HD progressive swap on desktop**

Load on desktop. Expected: preview atlases load first (fast), then HD textures swap in over ~30-60 seconds. No layout shift, no visual glitches during swap.

- [ ] **Step 6: Commit any remaining fixes**

```bash
git add -A
git commit -m "chore: end-to-end testing and cleanup for dual-resolution atlas loading"
```
# CLIP Search Implementation Plan

**Date:** 2026-04-12  
**Estimated Duration:** 2-3 sessions  
**Complexity:** Medium-High (touches pipeline + viewer + new worker)

---

## Overview

Implement CLIP-based semantic search and WebWorker performance infrastructure for ImageSpace. This is a dual-purpose feature: search capability + smoother viewport interactions.

---

## Tasks

### Task 1: Pipeline — Generate Search Index
**Status:** ○ Not Started  
**Files:** `scripts/imagespace.py`  
**Acceptance Criteria:**
- After CLIP embeddings generated, build HNSW index using hnswlib
- Serialize index to `search_index.bin` (2MB target)
- Add `hasSearchIndex: true` to `manifest.json`
- Index includes 512D vectors for all images, normalized

**TDD Steps:**
1. Add `hnswlib` to requirements.txt
2. Create `build_search_index(embeddings, output_path)` function
3. Save index in binary format compatible with hnswlib browser client
4. Update `write_manifest()` to include `hasSearchIndex`
5. Run pipeline on small test set (100 images), verify index loads in viewer

---

### Task 2: WebWorker — Create imageWorker.js
**Status:** ○ Not Started  
**Files:** `image_space/src/workers/imageWorker.js`  
**Acceptance Criteria:**
- Worker loads HNSW index from `search_index.bin`
- Worker responds to `search` messages with top-K results
- Worker builds R-tree from t-SNE coordinates for spatial queries
- Worker responds to `viewport` messages with visible point IDs
- All operations under 10ms for 50K images

**TDD Steps:**
1. Create worker file structure with message handlers
2. Add HNSW index loading (uses hnswlib wasm/browser build)
3. Add R-tree construction from t-SNE coordinates
4. Implement `search` handler: query index, return top 1000 IDs
5. Implement `viewport` handler: query R-tree, return visible IDs
6. Test worker independently with mock data

---

### Task 3: Viewer — Integrate Worker
**Status:** ○ Not Started  
**Files:** `image_space/src/App.jsx`  
**Acceptance Criteria:**
- Worker instantiated on app init
- Spatial queries moved to worker (remove local spatialHashRef)
- Worker responses update pointsRef visibility correctly
- Cleanup: worker terminated on unmount

**TDD Steps:**
1. Create worker instance in useEffect
2. Post initial t-SNE data to build R-tree
3. Replace local spatial hash with worker queries
4. Update hover detection to use worker responses
5. Verify cleanup on component unmount

---

### Task 4: Viewer — CLIP Text Encoding
**Status:** ○ Not Started  
**Files:** `image_space/src/App.jsx`  
**Acceptance Criteria:**
- Existing ONNX CLIP model (from pipeline) loaded in main thread
- Text input encoded to 512D vector
- Vector normalized (L2) before search
- Encoding cached for duplicate queries

**TDD Steps:**
1. Verify `clip-vit-base-patch32` ONNX model available in `public/`
2. Load model via `ort.InferenceSession`
3. Create tokenizer (same as pipeline)
4. Implement `encodeText(query)` function
5. Add query cache (LRU, 100 entries)

---

### Task 5: Viewer — Search UI
**Status:** ○ Not Started  
**Files:** `image_space/src/App.jsx`  
**Acceptance Criteria:**
- Search bar in top toolbar
- 150ms debounce on input
- Results dim non-matching to 20%
- Clear button resets view
- Keyboard shortcuts `/`, `Cmd+K`, `Escape` work

**TDD Steps:**
1. Add search input component to toolbar
2. Add `searchQuery` and `searchResults` state
3. Wire input to `encodeText` → worker `search`
4. Update `computeVisibleSet()` to include search intersection
5. Add keyboard event listeners
6. Style matching/non-matching sprites

---

### Task 6: Viewer — Error Handling
**Status:** ○ Not Started  
**Files:** `image_space/src/App.jsx`  
**Acceptance Criteria:**
- No matches: toast notification
- Index missing: search bar hidden
- Query too long: truncated with indicator
- Special chars: sanitized

**TDD Steps:**
1. Check `manifest.hasSearchIndex` before showing search UI
2. Handle empty result set with toast
3. Add input sanitization (strip >100 chars, non-ASCII)
4. Add aria-live region for results announcement

---

### Task 7: Integration Testing
**Status:** ○ Not Started  
**Acceptance Criteria:**
- Full pipeline run on 100-image test set
- Viewer loads and searches successfully
- Pan/zoom remains at 60fps
- Search returns sensible results (spot-check)

**TDD Steps:**
1. Run pipeline with `--hd` on test data
2. Load viewer, verify index downloads
3. Test searches: "portrait", "landscape", "abstract"
4. Profile: verify <10ms worker responses
5. Deploy to staging, test on mobile + desktop

---

## Dependencies

| Name | Version | Purpose |
|------|---------|---------|
| hnswlib | ^0.8.0 | HNSW index (pipeline + worker) |
| onnxruntime-web | ^1.17.0 | CLIP inference (already present) |
| rbush | ^3.0.1 | R-tree for spatial queries (worker) |

---

## Verification Checklist

- [ ] Pipeline generates `search_index.bin`
- [ ] Manifest includes `hasSearchIndex: true`
- [ ] Search bar visible in toolbar
- [ ] Typing triggers search
- [ ] Results highlight correctly
- [ ] Clear button works
- [ ] Keyboard shortcuts work
- [ ] No main thread blocking (DevTools Performance)
- [ ] 60fps maintained during pan/zoom
- [ ] Graceful degradation if index missing

---

## Notes

- Use same CLIP model as pipeline (`clip-vit-base-patch32`) for consistency
- R-tree library: `rbush` is battle-tested, small, fast
- HNSW index: use `hnswlib-js` for browser compatibility
- Consider lazy-loading index only when search first used (future optimization)

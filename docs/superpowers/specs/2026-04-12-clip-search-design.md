# CLIP-Based Semantic Search Design

**Date:** 2026-04-12  
**Status:** Approved for implementation  
**Related:** ImageSpace v1.1 feature addition

---

## Core Purpose

CLIP-based text-to-image search for ImageSpace. Users type natural language ("sunset over mountains", "portrait of a woman", "abstract blue shapes") and the viewer instantly surfaces matching images from the 50K collection without server requests.

**Key constraints:**
- Purely client-side (no backend)
- Works with existing v3 binary format + manifest
- Search happens in WebWorker (no main thread blocking)
- Results appear as highlights/filters on the existing scatter plot view

**What it's NOT:**
- Not a separate "search results" view — results integrate into current visualization
- Not fuzzy text search on metadata — purely semantic similarity via CLIP embeddings
- Not real-time re-querying of a server — precomputed embeddings only

---

## Architecture

### Pipeline (Build-Time)

After generating CLIP embeddings, build HNSW index using hnswlib with 512D vectors.

**Output:**
- `search_index.bin` (~2MB compressed binary)
- Include in `public/data/` alongside atlases
- Referenced in `manifest.json` with `hasSearchIndex: true`

### Viewer Structure

**New `imageWorker.js` WebWorker handles:**
- Spatial indexing: R-tree from t-SNE coordinates for O(log n) viewport queries
- Search queries: HNSW index for CLIP similarity
- Background compute: Average colors, cluster stats, filter aggregations

**Main thread changes:**
- New `searchQuery` state; `computeVisibleSet()` includes search results when active
- Reuse existing highlight/filter system — search results = temporary visibleSet override
- Replace `spatialHashRef` with worker-managed `spatialIndexRef`

### Data Flow — Search

1. User types in search box (debounced 150ms)
2. Main thread encodes text → 512D vector using existing ONNX CLIP (same model as pipeline)
3. Post message to WebWorker with query vector
4. WebWorker queries HNSW index → returns top-K image IDs (~5ms)
5. Main thread receives IDs, builds `searchResultSet`, passes to `computeVisibleSet()`
6. Non-matching sprites dim to 20% opacity, matching stay bright

### Data Flow — Normal Viewing

1. Viewport pans/zooms → posts bounds to worker
2. Worker queries R-tree → visible point IDs
3. Worker computes which atlas indices are needed
4. Main thread updates sprite visibility (already implemented)

---

## UI/UX Design

### Search Interface

Search bar appears in top toolbar, next to view mode tabs.

**States:**
- Empty: placeholder text "Search images..."
- Typing: debounced 150ms
- Results: matching images bright, others dim to 20%
- Clear button (X) resets to normal view
- Optional "Search results only" toggle switches to Grid view

### Keyboard Shortcuts

- `/` or `Cmd+K` — focus search bar
- `Escape` — clear search, return to normal view
- `Enter` — submit search (debounce bypass)

### Accessibility

- Search bar has `aria-label="Search images by description"`
- Results announced via `aria-live` region

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Search returns 0 matches | Show "No matches found" toast, keep current view with all images dimmed to 10% |
| User searches while atlases loading | Queue search, execute after atlases complete |
| Search index fails to load | Hide search bar entirely, graceful degradation |
| Very long query (>100 chars) | Truncate, indicate with "..." |
| Special characters/emoji | Strip non-ASCII, proceed with cleaned text |

---

## Performance Budget

| Metric | Baseline | With Search | Delta |
|--------|----------|-------------|-------|
| Initial download | ~100MB (atlases) | +2MB (index) | +2% |
| Initial load time | ~5s | +500ms | +10% |
| Runtime memory | ~3.1GB | +15MB (worker) | +0.5% |
| Search latency | — | ~50ms end-to-end | new |
| Pan/zoom framerate | 60fps | 60fps | maintained |

---

## Dependencies

- **hnswlib** (npm): HNSW index construction and querying
- **onnxruntime-web**: Already present for CLIP inference in viewer

---

## Future Extensions

- Image-to-image search (upload a photo, find similar)
- Negative prompting ("sunset NOT beach")
- Embedding arithmetic ("portrait + red - blue")
- Collaborative filtering (cluster-based recommendations)

---

## Approved By

User confirmed via `/skill:brainstorming` process.

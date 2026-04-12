/**
 * imageWorker.js — WebWorker for ImageSpace
 * 
 * Handles:
 * - Spatial indexing (R-tree for viewport queries)
 * - Search queries (HNSW index for CLIP similarity)
 * - Background computations
 * 
 * Interfaces:
 * - message { type: 'init', data: { points, searchIndexUrl? } }
 * - message { type: 'viewport', data: { minX, minY, maxX, maxY } }
 * - message { type: 'search', data: { queryVector, k } }
 * 
 * Responds with:
 * - { type: 'viewportResult', data: { visibleIds: [...] } }
 * - { type: 'searchResult', data: { ids: [...], distances: [...] } }
 */

// NOTE: hnswlib doesn't work directly in WebWorker (requires WASM).
// We'll implement a simple flat index for now since 50K × 512D = ~100MB
// and brute force cosine similarity is fast enough (~50ms) for this scale.

class ImageWorker {
  constructor() {
    this.points = null;
    this.embeddings = null; // Float32Array of normalized embeddings
    this.spatialIndex = null; // Simple grid-based spatial index
    this.hasSearchIndex = false;
    this.dimensions = 512;
  }

  /**
   * Initialize with point data and optional search index
   */
  init(data) {
    const { points, searchIndexUrl } = data;
    
    // Store points array (contains x, y, tsneX, tsneY, ai, id, etc.)
    this.points = points;
    
    // Build spatial index
    this._buildSpatialIndex();
    
    // Load search index if provided
    if (searchIndexUrl) {
      this._loadSearchIndex(searchIndexUrl);
    }
    
    return { success: true, pointCount: points.length };
  }

  /**
   * Build simple grid-based spatial index for viewport queries
   * Points are stored in cells by their t-SNE coordinates
   */
  _buildSpatialIndex() {
    // Find bounds
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of this.points) {
      const x = p.tsneX ?? p.x;
      const y = p.tsneY ?? p.y;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
    
    this.bounds = { minX, minY, maxX, maxY };
    
    // Build grid (100x100 cells)
    this.gridSize = 100;
    this.cellWidth = (maxX - minX) / this.gridSize || 1;
    this.cellHeight = (maxY - minY) / this.gridSize || 1;
    this.grid = new Array(this.gridSize * this.gridSize).fill(null).map(() => []);
    
    // Assign points to cells
    for (let i = 0; i < this.points.length; i++) {
      const p = this.points[i];
      const x = p.tsneX ?? p.x;
      const y = p.tsneY ?? p.y;
      
      const col = Math.floor((x - minX) / this.cellWidth);
      const row = Math.floor((y - minY) / this.cellHeight);
      const cellIdx = Math.min(row * this.gridSize + col, this.grid.length - 1);
      
      this.grid[cellIdx].push(i);
    }
  }

  /**
   * Query points within viewport bounds
   * Returns array of point indices
   */
  queryViewport(minX, minY, maxX, maxY) {
    const visibleIds = [];
    
    // Calculate cell range
    const startCol = Math.max(0, Math.floor((minX - this.bounds.minX) / this.cellWidth));
    const endCol = Math.min(this.gridSize - 1, Math.floor((maxX - this.bounds.minX) / this.cellWidth));
    const startRow = Math.max(0, Math.floor((minY - this.bounds.minY) / this.cellHeight));
    const endRow = Math.min(this.gridSize - 1, Math.floor((maxY - this.bounds.minY) / this.cellHeight));
    
    // Collect points from affected cells
    for (let row = startRow; row <= endRow; row++) {
      for (let col = startCol; col <= endCol; col++) {
        const cellIdx = row * this.gridSize + col;
        const cell = this.grid[cellIdx];
        for (const idx of cell) {
          const p = this.points[idx];
          const x = p.tsneX ?? p.x;
          const y = p.tsneY ?? p.y;
          if (x >= minX && x <= maxX && y >= minY && y <= maxY) {
            visibleIds.push(idx);
          }
        }
      }
    }
    
    return visibleIds;
  }

  /**
   * Load search index from binary file
   * For now: brute force search (simpler, no WASM dependencies)
   */
  async _loadSearchIndex(url) {
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error('Failed to load search index');
      
      // Read as Float32Array (each embedding is 512 floats)
      const buffer = await response.arrayBuffer();
      this.embeddings = new Float32Array(buffer);
      
      // Normalize embeddings
      const numEmbeddings = this.embeddings.length / this.dimensions;
      for (let i = 0; i < numEmbeddings; i++) {
        let norm = 0;
        for (let j = 0; j < this.dimensions; j++) {
          const val = this.embeddings[i * this.dimensions + j];
          norm += val * val;
        }
        norm = Math.sqrt(norm) || 1;
        for (let j = 0; j < this.dimensions; j++) {
          this.embeddings[i * this.dimensions + j] /= norm;
        }
      }
      
      this.hasSearchIndex = true;
    } catch (err) {
      console.error('[ImageWorker] Failed to load search index:', err);
      this.hasSearchIndex = false;
    }
  }

  /**
   * Search for top-k similar images given query vector
   * Returns { ids: [...], distances: [...] }
   */
  search(queryVector, k = 1000) {
    if (!this.hasSearchIndex || !this.embeddings) {
      return { ids: [], distances: [] };
    }
    
    // Normalize query vector
    let queryNorm = 0;
    for (let i = 0; i < this.dimensions; i++) {
      queryNorm += queryVector[i] * queryVector[i];
    }
    queryNorm = Math.sqrt(queryNorm) || 1;
    const normalizedQuery = queryVector.map(v => v / queryNorm);
    
    // Brute force cosine similarity
    const numEmbeddings = this.embeddings.length / this.dimensions;
    const scores = new Float32Array(numEmbeddings);
    
    for (let i = 0; i < numEmbeddings; i++) {
      let dot = 0;
      for (let j = 0; j < this.dimensions; j++) {
        dot += normalizedQuery[j] * this.embeddings[i * this.dimensions + j];
      }
      // Cosine similarity: dot product of normalized vectors
      scores[i] = dot;
    }
    
    // Get top-k (using simple partial sort)
    const indices = scores.map((score, idx) => ({ score, idx }));
    indices.sort((a, b) => b.score - a.score);
    
    const topK = indices.slice(0, k);
    return {
      ids: topK.map(item => item.idx),
      distances: topK.map(item => 1 - item.score) // convert to distance
    };
  }
}

// Worker instance
const worker = new ImageWorker();

// Message handler
self.onmessage = function(e) {
  const { type, id, data } = e.data;
  
  switch (type) {
    case 'init':
      try {
        const result = worker.init(data);
        self.postMessage({ type: 'initResult', id, data: result });
      } catch (err) {
        self.postMessage({ type: 'error', id, error: err.message });
      }
      break;
      
    case 'viewport':
      try {
        const { minX, minY, maxX, maxY } = data;
        const visibleIds = worker.queryViewport(minX, minY, maxX, maxY);
        self.postMessage({ type: 'viewportResult', id, data: { visibleIds } });
      } catch (err) {
        self.postMessage({ type: 'error', id, error: err.message });
      }
      break;
      
    case 'search':
      try {
        const { queryVector, k } = data;
        const result = worker.search(queryVector, k);
        self.postMessage({ type: 'searchResult', id, data: result });
      } catch (err) {
        self.postMessage({ type: 'error', id, error: err.message });
      }
      break;
      
    default:
      self.postMessage({ type: 'error', id, error: `Unknown message type: ${type}` });
  }
};

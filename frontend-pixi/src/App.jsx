import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import {
  Database, X, Layers, Grid, Eye,
  ZoomIn, ZoomOut, Maximize2,
  Flame, PanelLeftClose, PanelLeft, PanelRight,
  Palette, GalleryHorizontal, Info,
  ChevronLeft, ChevronRight, Filter, ChevronDown,
  Clock
} from 'lucide-react';

const THUMB_SIZE = 64;
const SPATIAL_CELL_SIZE = 120;

/* ── View Mode Layouts ────────────────────────── */
const VIEW_MODES = {
  umap: { label: 'UMAP', icon: 'scatter', desc: 'Semantic embedding space' },
  tsne: { label: 't-SNE', icon: 'scatter', desc: 'Alternative embedding projection' },
  grid: { label: 'Grid', icon: 'grid', desc: 'Ordinal grid layout' },
  color: { label: 'Color', icon: 'palette', desc: 'Sorted by dominant color' },
  timeline: { label: 'Timeline', icon: 'clock', desc: 'Chronological timeline' },
  carousel: { label: 'Carousel', icon: 'gallery', desc: 'Full-screen browsing' },
};

function computeLayout(allPoints, mode, visibleSet) {
  // If visibleSet provided, only layout those points; hide others (or dim in umap)
  const hasFilter = visibleSet && visibleSet.size < allPoints.length;
  const isStableLayout = mode === 'umap' || mode === 'tsne';
  // In stable-layout modes show ALL points in their positions, just dim non-visible
  const points = (hasFilter && !isStableLayout) ? allPoints.filter(p => visibleSet.has(p.id)) : allPoints;
  const n = points.length;

  // Visibility: in stable modes dim non-visible, in other modes hide them
  if (hasFilter) {
    for (const p of allPoints) {
      if (visibleSet.has(p.id)) {
        p.sprite.alpha = 1;
        p.sprite.tint = 0xffffff;
      } else if (isStableLayout) {
        // Dim but don't hide in stable-layout views
        p.sprite.alpha = 0.12;
        p.sprite.tint = 0xcccccc;
      } else {
        p.sprite.alpha = 0;
        p.targetX = p.sprite.x;
        p.targetY = p.sprite.y;
      }
    }
  } else {
    for (const p of allPoints) {
      p.sprite.alpha = 1;
      p.sprite.tint = 0xffffff;
    }
  }

  switch (mode) {
    case 'umap': {
      for (const p of points) {
        p.targetX = p.originalX;
        p.targetY = p.originalY;
      }
      break;
    }
    case 'tsne': {
      for (const p of points) {
        p.targetX = p.tsneX ?? p.originalX;
        p.targetY = p.tsneY ?? p.originalY;
      }
      break;
    }
    case 'grid': {
      const cols = Math.ceil(Math.sqrt(n));
      const spacing = THUMB_SIZE * 1.4;
      const ox = -(cols * spacing) / 2;
      const oy = -(Math.ceil(n / cols) * spacing) / 2;
      for (let i = 0; i < n; i++) {
        points[i].targetX = ox + (i % cols) * spacing;
        points[i].targetY = oy + Math.floor(i / cols) * spacing;
      }
      break;
    }
    case 'color': {
      // Sort indices by hue then arrange in grid
      const sorted = points
        .map((p, i) => ({ idx: i, hue: p.avgHue ?? 0, sat: p.avgSat ?? 0, lum: p.avgLum ?? 0.5 }))
        .sort((a, b) => a.hue - b.hue || a.lum - b.lum);
      const cols = Math.ceil(Math.sqrt(n));
      const spacing = THUMB_SIZE * 1.4;
      const ox = -(cols * spacing) / 2;
      const oy = -(Math.ceil(n / cols) * spacing) / 2;
      for (let i = 0; i < sorted.length; i++) {
        const p = points[sorted[i].idx];
        p.targetX = ox + (i % cols) * spacing;
        p.targetY = oy + Math.floor(i / cols) * spacing;
      }
      break;
    }
    case 'clusters': {
      const clusterMap = {};
      for (const p of points) {
        const c = p.cluster ?? 0;
        if (!clusterMap[c]) clusterMap[c] = [];
        clusterMap[c].push(p);
      }
      const cids = Object.keys(clusterMap).sort((a, b) => clusterMap[b].length - clusterMap[a].length);
      let yOff = 0;
      const sp = THUMB_SIZE * 1.4;
      const gap = THUMB_SIZE * 4;
      for (const cid of cids) {
        const pts = clusterMap[cid];
        const cols = Math.ceil(Math.sqrt(pts.length));
        const rows = Math.ceil(pts.length / cols);
        const xStart = -(cols * sp) / 2;
        for (let i = 0; i < pts.length; i++) {
          pts[i].targetX = xStart + (i % cols) * sp;
          pts[i].targetY = yOff + Math.floor(i / cols) * sp;
        }
        yOff += rows * sp + gap;
      }
      break;
    }
    case 'timeline': {
      // Sort by timestamp, arrange in columns top-to-bottom (vertical layout)
      const sorted = [...points].sort((a, b) => (a.timestamp ?? 0) - (b.timestamp ?? 0));
      const rowHeight = THUMB_SIZE * 1.4;
      const colWidth = THUMB_SIZE * 1.4;
      const cols = Math.max(1, Math.ceil(Math.sqrt(n) / 2)); // fewer columns → taller layout
      const ox = -(cols * colWidth) / 2;
      const rows = Math.ceil(n / cols);
      const oy = -(rows * rowHeight) / 2;
      for (let i = 0; i < sorted.length; i++) {
        sorted[i].targetX = ox + (i % cols) * colWidth;
        sorted[i].targetY = oy + Math.floor(i / cols) * rowHeight;
      }
      break;
    }

    default:
      break;
  }
}

/* ── Cluster Constants ────────────────────────── */
const NUM_CLUSTERS = 10;
const CLUSTER_COLORS = [
  '#286983', '#56949f', '#ea9d34', '#b4637a', '#907aa9',
  '#d7827e', '#569f84', '#9893a5', '#c4a7e7', '#797593',
];

/* ── K-Means Clustering ──────────────────────── */
function computeClusters(points, k = NUM_CLUSTERS, maxIter = 25) {
  const n = points.length;
  if (n === 0) return [];
  const step = Math.floor(n / k);
  const centroids = [];
  for (let i = 0; i < k; i++) {
    const p = points[i * step];
    centroids.push({ x: p.originalX, y: p.originalY });
  }
  const assignments = new Int32Array(n);
  for (let iter = 0; iter < maxIter; iter++) {
    for (let i = 0; i < n; i++) {
      let minDist = Infinity, minIdx = 0;
      for (let j = 0; j < k; j++) {
        const dx = points[i].originalX - centroids[j].x;
        const dy = points[i].originalY - centroids[j].y;
        const dist = dx * dx + dy * dy;
        if (dist < minDist) { minDist = dist; minIdx = j; }
      }
      assignments[i] = minIdx;
    }
    const sums = Array.from({ length: k }, () => ({ x: 0, y: 0, count: 0 }));
    for (let i = 0; i < n; i++) {
      const c = assignments[i];
      sums[c].x += points[i].originalX;
      sums[c].y += points[i].originalY;
      sums[c].count++;
    }
    for (let j = 0; j < k; j++) {
      if (sums[j].count > 0) {
        centroids[j].x = sums[j].x / sums[j].count;
        centroids[j].y = sums[j].y / sums[j].count;
      }
    }
  }
  const clusters = Array.from({ length: k }, (_, i) => ({
    id: i, centroid: centroids[i], count: 0, indices: [],
    color: CLUSTER_COLORS[i % CLUSTER_COLORS.length],
  }));
  for (let i = 0; i < n; i++) {
    clusters[assignments[i]].count++;
    clusters[assignments[i]].indices.push(i);
    points[i].cluster = assignments[i];
  }
  return clusters.sort((a, b) => b.count - a.count);
}

/* ── Extract Cluster Thumbnails ──────────────── */
function extractClusterThumbs(clusters, points, atlasTextures, thumbSize) {
  for (const cluster of clusters) {
    const cx = cluster.centroid.x, cy = cluster.centroid.y;
    const reps = cluster.indices
      .map(i => ({ i, d: Math.hypot(points[i].originalX - cx, points[i].originalY - cy) }))
      .sort((a, b) => a.d - b.d)
      .slice(0, 4);
    cluster.thumbnails = [];
    for (const { i: idx } of reps) {
      try {
        const p = points[idx];
        const src = atlasTextures[p.ai]?.source?.resource;
        if (!src) continue;
        const c = document.createElement('canvas');
        c.width = thumbSize; c.height = thumbSize;
        c.getContext('2d').drawImage(src, p.u, p.v, thumbSize, thumbSize, 0, 0, thumbSize, thumbSize);
        cluster.thumbnails.push(c.toDataURL('image/jpeg', 0.7));
      } catch (_) { /* skip */ }
    }
  }
}

/* ── Compute Average Color per Image ─────────── */
async function computeAvgColors(points, atlasTextures, thumbSize, onProgress) {
  const canvas = document.createElement('canvas');
  canvas.width = 1;
  canvas.height = 1;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const batch = 3000;
  for (let i = 0; i < points.length; i++) {
    if (i > 0 && i % batch === 0) {
      if (onProgress) onProgress(i);
      await new Promise(r => setTimeout(r, 0));
    }
    const p = points[i];
    const src = atlasTextures[p.ai]?.source?.resource;
    if (!src) { p.avgHue = 0; p.avgSat = 0; p.avgLum = 0.5; continue; }
    try {
      ctx.drawImage(src, p.u, p.v, thumbSize, thumbSize, 0, 0, 1, 1);
      const [r, g, b] = ctx.getImageData(0, 0, 1, 1).data;
      const rn = r / 255, gn = g / 255, bn = b / 255;
      const max = Math.max(rn, gn, bn), min = Math.min(rn, gn, bn);
      let h = 0, s = 0, l = (max + min) / 2;
      if (max !== min) {
        const d = max - min;
        s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
        if (max === rn) h = ((gn - bn) / d + (gn < bn ? 6 : 0)) / 6;
        else if (max === gn) h = ((bn - rn) / d + 2) / 6;
        else h = ((rn - gn) / d + 4) / 6;
      }
      p.avgHue = h; p.avgSat = s; p.avgLum = l;
      p.avgR = r; p.avgG = g; p.avgB = b;
    } catch (_) {
      p.avgHue = 0; p.avgSat = 0; p.avgLum = 0.5;
    }
  }
}

export default function App() {
  const canvasRef = useRef(null);
  const appRef = useRef(null);
  const viewportRef = useRef(null);
  const spatialHashRef = useRef({});
  const pointsRef = useRef([]);
  const viewModeRef = useRef('umap');

  const [loading, setLoading] = useState(true);
  const [loadProgress, setLoadProgress] = useState(0);
  const [stats, setStats] = useState({ count: 0, fps: 0 });
  const [selectedItem, setSelectedItem] = useState(null);
  const [error, setError] = useState(null);
  const [viewMode, setViewMode] = useState('umap');
  const [statusMsg, setStatusMsg] = useState('Preparing...');
  const [zoomLevel, setZoomLevel] = useState(1);
  const [hotspots, setHotspots] = useState([]);
  const [showHotspots, setShowHotspots] = useState(true);
  const [activeHotspot, setActiveHotspot] = useState(null);
  const [colorsReady, setColorsReady] = useState(false);
  const [carouselIdx, setCarouselIdx] = useState(null);
  const [showStats, setShowStats] = useState(false);
  const [tooltip, setTooltip] = useState(null);
  const [showDetailPanel, setShowDetailPanel] = useState(false);
  const [clusterLabels, setClusterLabels] = useState([]); // [{id, x, y, label, color, count}]
  const clusterCentroidsRef = useRef([]); // world positions of cluster group centers
  const [timeRange, setTimeRange] = useState(null); // { min, max, current }
  const [timeFilter, setTimeFilter] = useState([0, 1000]); // dual range: [lo, hi] out of 1000
  const timeFilterRef = useRef([0, 1000]);
  const timelineMapRef = useRef(null); // maps x-world-pos to timestamp
  const [metadata, setMetadata] = useState(null);   // { columns: string[], rows: {[col]: string}[] }
  const [csvFilters, setCsvFilters] = useState({});  // { columnName: selectedValue | null }
  const [openFilter, setOpenFilter] = useState(null); // which dropdown is open
  const visibleSetRef = useRef(null);                // current Set<id> or null (all visible)

  /* ── Compute visible set from hotspot + csvFilters ── */
  const computeVisibleSet = useCallback((hotspotId, filters, hotspotsData, meta) => {
    let ids = null;

    // Hotspot filter
    if (hotspotId !== null && hotspotsData.length > 0) {
      const h = hotspotsData.find(h => h.id === hotspotId);
      if (h) ids = new Set(h.indices);
    }

    // CSV filters — additive (union across all selected values)
    const activeEntries = Object.entries(filters).filter(([, v]) => v && v.size > 0);
    if (activeEntries.length > 0 && meta) {
      const csvSet = new Set();
      for (const [col, vals] of activeEntries) {
        for (let i = 0; i < meta.rows.length; i++) {
          if (vals.has(meta.rows[i][col])) csvSet.add(i);
        }
      }
      if (ids === null) {
        ids = csvSet;
      } else {
        // Intersect hotspot with CSV union
        const intersection = new Set();
        for (const id of ids) {
          if (csvSet.has(id)) intersection.add(id);
        }
        ids = intersection;
      }
    }

    return ids;
  }, []);

  /* ── Recompute layout with current filters ── */
  const relayout = useCallback((mode, visSet) => {
    visibleSetRef.current = visSet;
    computeLayout(pointsRef.current, mode, visSet);

    // Compute cluster label positions for cluster view
    if (mode === 'clusters') {
      const hasFilter = visSet && visSet.size < pointsRef.current.length;
      const pts = hasFilter ? pointsRef.current.filter(p => visSet.has(p.id)) : pointsRef.current;
      const clusterMap = {};
      for (const p of pts) {
        const c = p.cluster ?? 0;
        if (!clusterMap[c]) clusterMap[c] = { sx: 0, sy: 0, minY: Infinity, count: 0 };
        clusterMap[c].sx += p.targetX;
        clusterMap[c].sy += p.targetY;
        clusterMap[c].minY = Math.min(clusterMap[c].minY, p.targetY);
        clusterMap[c].count++;
      }
      const labels = Object.entries(clusterMap)
        .sort((a, b) => b[1].count - a[1].count)
        .map(([cid, data], idx) => ({
          id: parseInt(cid),
          worldX: data.sx / data.count,
          worldY: data.minY - THUMB_SIZE * 2.5,
          label: `Cluster ${idx + 1}`,
          color: CLUSTER_COLORS[parseInt(cid) % CLUSTER_COLORS.length],
          count: data.count,
        }));
      clusterCentroidsRef.current = labels;
    } else {
      clusterCentroidsRef.current = [];
      setClusterLabels([]);
    }

    // Compute timeline mapping
    if (mode === 'timeline') {
      const hasFilter = visSet && visSet.size < pointsRef.current.length;
      const pts = hasFilter ? pointsRef.current.filter(p => visSet.has(p.id)) : pointsRef.current;
      const timestamps = pts.map(p => p.timestamp ?? 0).filter(t => t > 0);
      if (timestamps.length > 0) {
        const minTs = Math.min(...timestamps);
        const maxTs = Math.max(...timestamps);
        // Build a map: sort by targetX and store timestamp
        const sorted = [...pts].sort((a, b) => a.targetX - b.targetX);
        const xMin = sorted[0]?.targetX ?? 0;
        const xMax = sorted[sorted.length - 1]?.targetX ?? 0;
        timelineMapRef.current = { xMin, xMax, minTs, maxTs };
        setTimeRange({ min: minTs, max: maxTs, current: null });
        // Reset time filter slider to full range when data set changes
        setTimeFilter([0, 1000]);
        timeFilterRef.current = [0, 1000];
      }
    } else {
      timelineMapRef.current = null;
      setTimeRange(null);
    }

    setTimeout(() => {
      const vp = viewportRef.current;
      if (vp && mode !== 'carousel') {
        // Compute actual content bounds from target positions
        const pts = pointsRef.current;
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        for (const p of pts) {
          if (p.sprite.alpha === 0) continue; // skip hidden
          minX = Math.min(minX, p.targetX);
          maxX = Math.max(maxX, p.targetX);
          minY = Math.min(minY, p.targetY);
          maxY = Math.max(maxY, p.targetY);
        }
        if (minX < Infinity) {
          const pad = THUMB_SIZE * 2;
          const w = maxX - minX + pad;
          const h = maxY - minY + pad;
          const cx = (minX + maxX) / 2;
          const cy = (minY + maxY) / 2;
          const scaleX = vp.screenWidth / w;
          const scaleY = vp.screenHeight / h;
          const scale = Math.min(scaleX, scaleY);
          vp.setZoom(scale, true);
          vp.moveCenter(cx, cy);
        } else {
          vp.fit();
          vp.moveCenter(0, 0);
        }
      }
    }, 100);
  }, []);

  const switchView = useCallback((mode) => {
    setViewMode(mode);
    viewModeRef.current = mode;
    // Reset time filter when leaving timeline
    if (mode !== 'timeline') {
      setTimeFilter([0, 1000]);
      timeFilterRef.current = [0, 1000];
    }
    // Auto-select first image for carousel if nothing selected
    if (mode === 'carousel' && carouselIdx === null) {
      setCarouselIdx(0);
    }
    // Recompute visible set with current filters (keep hotspot + csv filters active)
    setActiveHotspot(prev => {
      const visSet = computeVisibleSet(prev, csvFilters, hotspots, metadata);
      relayout(mode, visSet);
      return prev;
    });
  }, [csvFilters, hotspots, metadata, computeVisibleSet, relayout, carouselIdx]);

  /* ── PixiJS boot ────────────────────────────── */
  useEffect(() => {
    let app = null;
    let isCancelled = false;

    async function start() {
      try {
        setStatusMsg('Loading rendering engine...');
        const PIXI = await import('pixi.js');
        const { Viewport } = await import('pixi-viewport');

        setStatusMsg('Initialising WebGL...');
        app = new PIXI.Application();
        await app.init({
          canvas: canvasRef.current,
          antialias: false,
          backgroundColor: 0xfaf4ed,
          width: window.innerWidth,
          height: window.innerHeight,
          resolution: window.devicePixelRatio || 1,
          autoDensity: true,
          preference: 'webgl',
        });
        if (isCancelled) { app.destroy(true, { children: true, texture: true }); return; }
        appRef.current = app;

        const viewport = new Viewport({
          screenWidth: window.innerWidth,
          screenHeight: window.innerHeight,
          worldWidth: 20000,
          worldHeight: 20000,
          events: app.renderer.events,
        });
        viewportRef.current = viewport;
        app.stage.addChild(viewport);
        viewport.drag().pinch().wheel().decelerate();
        viewport.moveCenter(0, 0);

        /* Load data */
        setStatusMsg('Fetching manifest...');
        const mRes = await fetch('/data/manifest.json');
        if (!mRes.ok) throw new Error('Manifest load failed');
        const manifest = await mRes.json();

        setStatusMsg('Streaming binary layout...');
        const dRes = await fetch('/data/data.bin');
        if (!dRes.ok) throw new Error('Binary load failed');
        const buffer = await dRes.arrayBuffer();
        const dataView = new DataView(buffer);

        setStats(s => ({ ...s, count: manifest.count }));
        setStatusMsg(`Loading ${manifest.atlasCount} atlas textures...`);

        const atlasTextures = [];
        for (let i = 0; i < manifest.atlasCount; i++) {
          const tex = await PIXI.Assets.load(`/data/atlas_${i}.jpg`);
          if (!tex) throw new Error(`Atlas ${i} load failed`);
          atlasTextures.push(tex);
          setLoadProgress(Math.round(((i + 1) / manifest.atlasCount) * 40));
        }

        /* Create sprites */
        setStatusMsg('Building image field...');
        const container = new PIXI.Container();
        viewport.addChild(container);

        const currentThumbSize = manifest.thumbSize || THUMB_SIZE;
        const batchSize = 5000;

        // Detect binary format: v2 (24 bytes) has both UMAP + t-SNE coords; v1 (16 bytes) has one set
        const bytesPerImage = manifest.bytesPerImage || 16;
        const isV2 = bytesPerImage === 24;

        for (let i = 0; i < manifest.count; i++) {
          if (i > 0 && i % batchSize === 0) {
            setStatusMsg(`Placing images ${Math.round((i / manifest.count) * 100)}%...`);
            setLoadProgress(40 + Math.round((i / manifest.count) * 55));
            await new Promise(r => setTimeout(r, 0));
            if (isCancelled) return;
          }

          let x, y, tsneX, tsneY, ai, u, v, cluster;
          if (isV2) {
            // 24-byte format: float32 umapX, umapY, tsneX, tsneY, uint16 atlas, u, v, cluster
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
            // 16-byte format: float32 x, y, uint16 atlas, u, v, padding
            const offset = i * 16;
            x  = dataView.getFloat32(offset, true);
            y  = dataView.getFloat32(offset + 4, true);
            ai = dataView.getUint16(offset + 8, true);
            u  = dataView.getUint16(offset + 10, true);
            v  = dataView.getUint16(offset + 12, true);
            tsneX = x; tsneY = y; // Same coords for both modes
            cluster = undefined;
          }

          const frame = new PIXI.Rectangle(u, v, currentThumbSize, currentThumbSize);
          const tex = new PIXI.Texture({ source: atlasTextures[ai].source, frame });
          const sprite = new PIXI.Sprite(tex);
          sprite.anchor.set(0.5);
          sprite.position.set(x, y);
          sprite.eventMode = 'none';
          container.addChild(sprite);

          const pObj = {
            id: i, x, y,
            originalX: x, originalY: y,       // UMAP coordinates
            tsneX, tsneY,                      // t-SNE coordinates
            targetX: x, targetY: y,
            ai, u, v, sprite,
            ...(cluster !== undefined && { cluster }),
          };
          pointsRef.current.push(pObj);

          const gx = Math.floor(x / SPATIAL_CELL_SIZE);
          const gy = Math.floor(y / SPATIAL_CELL_SIZE);
          const key = `${gx},${gy}`;
          if (!spatialHashRef.current[key]) spatialHashRef.current[key] = [];
          spatialHashRef.current[key].push(pObj);
        }

        /* Compute hotspots */
        setStatusMsg('Analysing clusters...');
        setLoadProgress(96);
        await new Promise(r => setTimeout(r, 0));
        const clusters = computeClusters(pointsRef.current);
        setLoadProgress(98);
        extractClusterThumbs(clusters, pointsRef.current, atlasTextures, currentThumbSize);
        setHotspots(clusters);

        /* Compute average colors */
        setStatusMsg('Computing image colors...');
        await computeAvgColors(pointsRef.current, atlasTextures, currentThumbSize, (i) => {
          setLoadProgress(98 + Math.round((i / manifest.count) * 1));
        });
        setColorsReady(true);

        /* Load metadata CSV (optional) */
        try {
          setStatusMsg('Loading metadata...');
          const csvRes = await fetch('/data/metadata.csv');
          if (csvRes.ok) {
            const csvText = await csvRes.text();
            const lines = csvText.trim().split('\n');
            if (lines.length > 1) {
              const columns = lines[0].split(',').map(c => c.trim());
              const idCol = columns.indexOf('id');
              const catCols = columns.filter(c => c !== 'id');
              const rows = [];
              for (let i = 1; i < lines.length; i++) {
                const vals = lines[i].split(',').map(v => v.trim());
                const row = {};
                for (let j = 0; j < columns.length; j++) {
                  row[columns[j]] = vals[j] || '';
                }
                rows.push(row);
              }

              // Attach timestamp to points if available
              const tsCol = columns.indexOf('timestamp');
              if (tsCol >= 0) {
                for (let i = 0; i < rows.length && i < pointsRef.current.length; i++) {
                  const ts = parseInt(rows[i].timestamp);
                  pointsRef.current[i].timestamp = isNaN(ts) ? 0 : ts;
                }
              }
              // Exclude 'id' and 'timestamp' from filter columns
              const filterCols = catCols.filter(c => c !== 'timestamp');
              setMetadata({ columns: filterCols, rows, allColumns: catCols });
            }
          }
        } catch (_) { /* metadata.csv is optional */ }

        // Fit to actual content bounds
        {
          let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
          for (const p of pointsRef.current) {
            minX = Math.min(minX, p.originalX);
            maxX = Math.max(maxX, p.originalX);
            minY = Math.min(minY, p.originalY);
            maxY = Math.max(maxY, p.originalY);
          }
          if (minX < Infinity) {
            const pad = THUMB_SIZE * 2;
            const w = maxX - minX + pad;
            const h = maxY - minY + pad;
            const scaleX = viewport.screenWidth / w;
            const scaleY = viewport.screenHeight / h;
            const scale = Math.min(scaleX, scaleY);
            viewport.setZoom(scale, true);
            viewport.moveCenter((minX + maxX) / 2, (minY + maxY) / 2);
          } else {
            viewport.fit();
            viewport.moveCenter(0, 0);
          }
        }

        /* Ticker — animation + FPS */
        app.ticker.add((delta) => {
          if (isCancelled) return;
          let moving = false;
          for (const p of pointsRef.current) {
            const dx = p.targetX - p.sprite.x;
            const dy = p.targetY - p.sprite.y;
            if (Math.abs(dx) > 0.5 || Math.abs(dy) > 0.5) {
              p.sprite.x += dx * 0.08 * delta.deltaTime;
              p.sprite.y += dy * 0.08 * delta.deltaTime;
              p.x = p.sprite.x;
              p.y = p.sprite.y;
              moving = true;
            }
          }
          if (moving) {
            spatialHashRef.current = {};
            for (const p of pointsRef.current) {
              const gx = Math.floor(p.x / SPATIAL_CELL_SIZE);
              const gy = Math.floor(p.y / SPATIAL_CELL_SIZE);
              const key = `${gx},${gy}`;
              if (!spatialHashRef.current[key]) spatialHashRef.current[key] = [];
              spatialHashRef.current[key].push(p);
            }
          }
          // Throttle UI state updates to avoid React re-render overhead
          const now = Date.now();
          if (!app._lastUiUpdate || now - app._lastUiUpdate > 200) {
            app._lastUiUpdate = now;
            setZoomLevel(viewport.scale.x);
            setStats(s => ({ ...s, fps: Math.round(app.ticker.FPS) }));
          }

          // Update cluster label screen positions
          if (clusterCentroidsRef.current.length > 0) {
            const labels = clusterCentroidsRef.current.map(c => {
              const screen = viewport.toScreen(c.worldX, c.worldY);
              return { ...c, x: screen.x, y: screen.y };
            });
            setClusterLabels(labels);
          }

          // Update timeline current time based on viewport center
          if (timelineMapRef.current) {
            const tm = timelineMapRef.current;
            const centerWorld = viewport.toWorld(window.innerWidth / 2, window.innerHeight / 2);
            const t = (centerWorld.x - tm.xMin) / (tm.xMax - tm.xMin || 1);
            const currentTs = Math.round(tm.minTs + t * (tm.maxTs - tm.minTs));
            setTimeRange(prev => prev ? { ...prev, current: currentTs } : prev);

            // Apply time filter dimming (only to visible/filtered sprites)
            const tf = timeFilterRef.current;
            if (tf[0] > 0 || tf[1] < 1000) {
              const loTs = tm.minTs + (tf[0] / 1000) * (tm.maxTs - tm.minTs);
              const hiTs = tm.minTs + (tf[1] / 1000) * (tm.maxTs - tm.minTs);
              const vs = visibleSetRef.current;
              for (const p of pointsRef.current) {
                // Skip sprites hidden by hotspot/CSV filters
                if (vs && !vs.has(p.id)) continue;
                const ts = p.timestamp ?? 0;
                if (ts >= loTs && ts <= hiTs) {
                  p.sprite.alpha = 1;
                  p.sprite.tint = 0xffffff;
                } else {
                  p.sprite.alpha = 0.1;
                  p.sprite.tint = 0xcccccc;
                }
              }
            }
          }
        });

        /* Hover + click */
        app.stage.eventMode = 'static';
        app.stage.hitArea = app.screen;
        let lastHovered = null;

        app.stage.on('pointermove', (e) => {
          if (isCancelled) return;
          const worldPos = viewport.toWorld(e.global.x, e.global.y);
          const gx = Math.floor(worldPos.x / SPATIAL_CELL_SIZE);
          const gy = Math.floor(worldPos.y / SPATIAL_CELL_SIZE);
          let closest = null;
          let minDist = 50 / viewport.scale.x;
          for (let ix = -1; ix <= 1; ix++) {
            for (let iy = -1; iy <= 1; iy++) {
              const cell = spatialHashRef.current[`${gx + ix},${gy + iy}`];
              if (cell) {
                for (const p of cell) {
                  const dist = Math.hypot(p.x - worldPos.x, p.y - worldPos.y);
                  if (dist < minDist) { minDist = dist; closest = p; }
                }
              }
            }
          }
          if (closest !== lastHovered) {
            if (lastHovered) { lastHovered.sprite.scale.set(1); lastHovered.sprite.tint = 0xffffff; }
            if (closest) { closest.sprite.scale.set(1.5); closest.sprite.tint = 0xea9d34; }
            lastHovered = closest;
            if (closest) {
              setTooltip({ id: closest.id, x: e.global.x, y: e.global.y });
            } else {
              setTooltip(null);
            }
          }
        });

        app.stage.on('pointerdown', () => {
          if (lastHovered) {
            setSelectedItem({ id: lastHovered.id, x: lastHovered.x, y: lastHovered.y });
            setCarouselIdx(lastHovered.id);
          }
        });

        // Resize
        window.addEventListener('resize', () => {
          app.renderer.resize(window.innerWidth, window.innerHeight);
          viewport.resize(window.innerWidth, window.innerHeight);
        });

        /* Timeline wheel → pan instead of zoom */
        const canvas = canvasRef.current;
        if (canvas) {
          canvas.addEventListener('wheel', (e) => {
            if (viewModeRef.current === 'timeline') {
              e.preventDefault();
              e.stopImmediatePropagation();
              const speed = 2;
              viewport.x -= e.deltaX * speed;
              viewport.y -= e.deltaY * speed;
              viewport.dirty = true;
            }
          }, { passive: false });
        }

        setLoadProgress(100);
        setLoading(false);

      } catch (err) {
        console.error('Boot failure:', err);
        if (!isCancelled) setError(err.message);
      }
    }

    start();
    return () => {
      isCancelled = true;
      if (app) app.destroy(true, { children: true, texture: true });
      pointsRef.current = [];
      spatialHashRef.current = {};
    };
  }, []);

  const flyToHotspot = useCallback((h) => {
    const newActive = activeHotspot === h.id ? null : h.id;
    setActiveHotspot(newActive);
    // Recompute visible set with hotspot + csv filters
    const visSet = computeVisibleSet(newActive, csvFilters, hotspots, metadata);
    relayout(viewMode, visSet);
  }, [activeHotspot, csvFilters, hotspots, metadata, viewMode, computeVisibleSet, relayout]);

  const handleZoom = (dir) => {
    const vp = viewportRef.current;
    if (!vp) return;
    const factor = dir === 'in' ? 1.5 : 0.67;
    vp.animate({ scale: vp.scale.x * factor, time: 300 });
  };

  const handleFitAll = () => {
    const vp = viewportRef.current;
    if (!vp) return;
    const pts = pointsRef.current;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of pts) {
      if (p.sprite.alpha === 0) continue;
      minX = Math.min(minX, p.targetX);
      maxX = Math.max(maxX, p.targetX);
      minY = Math.min(minY, p.targetY);
      maxY = Math.max(maxY, p.targetY);
    }
    if (minX < Infinity) {
      const pad = THUMB_SIZE * 2;
      const w = maxX - minX + pad;
      const h = maxY - minY + pad;
      const scaleX = vp.screenWidth / w;
      const scaleY = vp.screenHeight / h;
      const scale = Math.min(scaleX, scaleY);
      vp.animate({ scale, position: { x: (minX + maxX) / 2, y: (minY + maxY) / 2 }, time: 300 });
    } else {
      vp.fit();
      vp.moveCenter(0, 0);
    }
  };

  const modeIcon = (mode) => {
    switch (VIEW_MODES[mode].icon) {
      case 'scatter': return <Eye size={14} />;
      case 'grid': return <Grid size={14} />;
      case 'flame': return <Flame size={14} />;
      case 'palette': return <Palette size={14} />;
      case 'gallery': return <GalleryHorizontal size={14} />;
      case 'clock': return <Clock size={14} />;
      default: return <Eye size={14} />;
    }
  };

  /* ── Visible indices list for carousel navigation ── */
  const carouselList = useMemo(() => {
    if (!visibleSetRef.current) return pointsRef.current.map(p => p.id);
    return [...visibleSetRef.current].sort((a, b) => a - b);
  }, [viewMode, activeHotspot, csvFilters]);

  /* ── CSV filter helpers ── */
  const filterOptions = useMemo(() => {
    if (!metadata) return {};
    const opts = {};
    for (const col of metadata.columns) {
      const vals = new Set();
      for (const row of metadata.rows) {
        if (row[col]) vals.add(row[col]);
      }
      opts[col] = [...vals].sort();
    }
    return opts;
  }, [metadata]);

  const handleFilterChange = useCallback((col, val) => {
    setCsvFilters(prev => {
      const next = { ...prev };
      const existing = next[col] ? new Set(next[col]) : new Set();
      if (existing.has(val)) {
        existing.delete(val);
      } else {
        existing.add(val);
      }
      if (existing.size === 0) {
        delete next[col];
      } else {
        next[col] = existing;
      }
      setActiveHotspot(hotId => {
        const visSet = computeVisibleSet(hotId, next, hotspots, metadata);
        relayout(viewMode, visSet);
        return hotId;
      });
      return next;
    });
  }, [hotspots, metadata, viewMode, computeVisibleSet, relayout]);

  const clearAllFilters = useCallback(() => {
    setCsvFilters({});
    setActiveHotspot(null);
    const visSet = null;
    relayout(viewMode, visSet);
  }, [viewMode, relayout]);

  const activeFilterCount = Object.values(csvFilters).reduce((sum, s) => sum + (s ? s.size : 0), 0) + (activeHotspot !== null ? 1 : 0);

  /* ── Render ─────────────────────────────────── */


  return (
    <div className="relative w-screen h-screen bg-rp-base font-sans select-none overflow-hidden">
      <canvas ref={canvasRef} className={`absolute inset-0 w-full h-full transition-opacity duration-300 ${viewMode === 'carousel' ? 'invisible pointer-events-none' : ''}`} />

      {/* ── Cluster Labels ───────────────────── */}
      {viewMode === 'clusters' && clusterLabels.length > 0 && clusterLabels.map(cl => (
        <div
          key={cl.id}
          className="absolute z-[35] pointer-events-none flex items-center gap-2 transition-all"
          style={{
            left: cl.x,
            top: cl.y,
            transform: 'translate(-50%, -100%)',
          }}
        >
          <div className="flex items-center gap-2 bg-rp-surface/95 backdrop-blur-md rounded-lg border border-rp-hlMed shadow-rp px-3 py-1.5">
            <div className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: cl.color }} />
            <span className="text-xs font-bold text-rp-text whitespace-nowrap">{cl.label}</span>
            <span className="text-[10px] text-rp-muted font-semibold">{cl.count.toLocaleString()}</span>
          </div>
        </div>
      ))}



      {/* ── Carousel View ────────────────────── */}
      {viewMode === 'carousel' && !loading && (
        <div className="absolute inset-0 z-[40] bg-rp-base flex items-center justify-center">
          {carouselIdx !== null && carouselIdx >= 0 && carouselIdx < pointsRef.current.length && (() => {
            const p = pointsRef.current[carouselIdx];
            const posInList = carouselList.indexOf(carouselIdx);
            const prevIdx = posInList > 0 ? carouselList[posInList - 1] : carouselList[carouselList.length - 1];
            const nextIdx = posInList < carouselList.length - 1 ? carouselList[posInList + 1] : carouselList[0];
            return (
              <div className="flex flex-col items-center gap-6 max-w-2xl w-full px-8">
                {/* Large image */}
                <div
                  className="rounded-2xl overflow-hidden shadow-rp-lg border border-rp-hlMed"
                  style={{
                    width: '320px',
                    height: '320px',
                    backgroundImage: `url(/data/atlas_${p.ai}.jpg)`,
                    backgroundPosition: `-${p.u}px -${p.v}px`,
                    backgroundSize: 'auto',
                    imageRendering: 'auto',
                  }}
                />
                {/* Info */}
                <div className="text-center">
                  <p className="text-2xl font-extrabold text-rp-text">Image #{p.id}</p>
                  <p className="text-sm text-rp-muted mt-1">
                    Position: {p.originalX.toFixed(1)}, {p.originalY.toFixed(1)}
                    {p.avgHue !== undefined && ` · Color: hsl(${Math.round(p.avgHue * 360)}, ${Math.round(p.avgSat * 100)}%, ${Math.round(p.avgLum * 100)}%)`}
                  </p>
                </div>
                {/* Navigation */}
                <div className="flex items-center gap-4">
                  <button
                    onClick={() => setCarouselIdx(prevIdx)}
                    className="p-3 rounded-xl bg-rp-surface border border-rp-hlMed hover:border-rp-pine transition-colors text-rp-text"
                  >
                    <ChevronLeft size={20} />
                  </button>
                  <span className="text-sm font-semibold text-rp-muted tabular-nums min-w-[120px] text-center">
                    {(posInList + 1).toLocaleString()} / {carouselList.length.toLocaleString()}
                  </span>
                  <button
                    onClick={() => setCarouselIdx(nextIdx)}
                    className="p-3 rounded-xl bg-rp-surface border border-rp-hlMed hover:border-rp-pine transition-colors text-rp-text"
                  >
                    <ChevronRight size={20} />
                  </button>
                </div>
              </div>
            );
          })()}
          {carouselIdx === null && (
            <div className="text-center">
              <GalleryHorizontal size={48} className="text-rp-muted mx-auto mb-4" />
              <p className="text-lg font-bold text-rp-text">Loading carousel...</p>
            </div>
          )}
        </div>
      )}

      {/* ── UI Overlay ────────────────────────── */}
      {/* Top section — pinned to top */}
      <div className="absolute top-0 left-0 right-0 pointer-events-none z-50 p-4">
        <div className="flex justify-between items-start gap-3">
          {/* Logo + Hotspots column */}
          <div className="flex flex-col gap-2">
            {/* Logo */}
            <div className="pointer-events-auto rp-card flex items-center gap-3">
              <div className="bg-rp-pine text-white p-2 rounded-lg">
                <Layers size={18} />
              </div>
              <div>
                <h1 className="text-lg font-extrabold tracking-tight text-rp-text leading-none">
                  ImageSpace
                </h1>
                <p className="text-[9px] font-medium text-rp-muted tracking-wider uppercase mt-0.5">
                  {stats.count > 0 ? `${(stats.count / 1000).toFixed(0)}K images` : 'Visual Explorer'}
                </p>
              </div>
            </div>

            {/* Hotspots (larger cards) */}
            {!loading && hotspots.length > 0 && viewMode !== 'carousel' && showHotspots && (
              <div className="pointer-events-auto flex flex-col gap-2 max-w-[240px]">
                {hotspots.slice(0, 8).map((h, i) => {
                  const pct = stats.count > 0 ? ((h.count / stats.count) * 100) : 0;
                  return (
                    <button
                      key={h.id}
                      onClick={() => flyToHotspot(h)}
                      className={`bg-rp-surface/90 backdrop-blur-md rounded-xl border border-rp-hlMed shadow-rp transition-all duration-200 cursor-pointer ${
                        activeHotspot === h.id
                          ? 'ring-2 ring-rp-pine shadow-rp-lg'
                          : 'hover:shadow-rp-lg hover:border-rp-pine/40'
                      }`}
                    >
                      <div className="flex items-center gap-3 p-3">
                        {h.thumbnails?.[0] ? (
                          <img src={h.thumbnails[0]} alt="" className="w-12 h-12 rounded-lg object-cover shrink-0" />
                        ) : (
                          <div className="w-12 h-12 rounded-lg shrink-0" style={{ backgroundColor: h.color, opacity: 0.4 }} />
                        )}
                        <div className="flex-1 min-w-0">
                          <p className="text-xs font-bold text-rp-text leading-tight">Hotspot {i + 1}</p>
                          <p className="text-[10px] text-rp-muted mt-0.5">{h.count.toLocaleString()} images</p>
                          <div className="mt-1 h-1.5 bg-rp-hlMed rounded-full overflow-hidden">
                            <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: h.color }} />
                          </div>
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>
            )}
            {!loading && hotspots.length > 0 && !showHotspots && viewMode !== 'carousel' && (
              <button
                onClick={() => setShowHotspots(true)}
                className="pointer-events-auto rp-card flex items-center gap-2 px-3 py-2 hover:border-rp-pine/30 transition-all"
              >
                <PanelLeft size={14} className="text-rp-pine" />
                <span className="text-[11px] font-semibold text-rp-text">Hotspots</span>
              </button>
            )}
          </div>

          {/* View Mode Tabs + Filters */}
          <div className="flex flex-col items-end gap-2">
            {/* View tabs */}
            <div className="pointer-events-auto rp-card flex items-center gap-1 px-2 py-1.5">
              {Object.entries(VIEW_MODES).map(([key, { label }]) => (
                <button
                  key={key}
                  onClick={() => switchView(key)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
                    viewMode === key
                      ? 'bg-rp-pine text-white shadow-sm'
                      : 'text-rp-subtle hover:bg-rp-hlLow hover:text-rp-text'
                  }`}
                >
                  {modeIcon(key)}
                  {label}
                </button>
              ))}
              <div className="w-px h-4 bg-rp-hlHigh mx-0.5" />
              <button
                onClick={() => setShowStats(true)}
                className="p-1.5 rounded-md text-rp-subtle hover:bg-rp-hlLow hover:text-rp-text transition-all"
                title="Collection info"
              >
                <Info size={14} />
              </button>
            </div>

            {/* Checkbox filter bar */}
            {metadata && metadata.columns.length > 0 && (
              <div className="pointer-events-auto rp-card flex items-center gap-2 px-3 py-1.5 flex-wrap">
                <Filter size={11} className="text-rp-muted shrink-0" />
                {metadata.columns.map(col => {
                  const selected = csvFilters[col] || new Set();
                  const count = selected.size;
                  return (
                    <div key={col} className="relative">
                      <button
                        onClick={() => setOpenFilter(openFilter === col ? null : col)}
                        className={`flex items-center gap-1 px-2 py-1 rounded-md text-[11px] transition-all border ${
                          count > 0
                            ? 'bg-rp-pine/10 border-rp-pine/30 text-rp-pine font-semibold'
                            : 'bg-rp-hlLow/50 border-transparent text-rp-subtle hover:border-rp-hlHigh'
                        }`}
                      >
                        <span className="truncate max-w-[80px]">{count > 0 ? `${col} (${count})` : col}</span>
                        <ChevronDown size={10} className={`shrink-0 transition-transform ${openFilter === col ? 'rotate-180' : ''}`} />
                      </button>
                      {openFilter === col && (
                        <div className="absolute right-0 top-full mt-1 bg-rp-surface border border-rp-hlMed rounded-lg shadow-rp-lg max-h-48 overflow-y-auto z-[100] min-w-[160px]">
                          {count > 0 && (
                            <button
                              onClick={() => {
                                setCsvFilters(prev => {
                                  const next = { ...prev };
                                  delete next[col];
                                  setActiveHotspot(hotId => {
                                    const visSet = computeVisibleSet(hotId, next, hotspots, metadata);
                                    relayout(viewMode, visSet);
                                    return hotId;
                                  });
                                  return next;
                                });
                              }}
                              className="w-full text-left px-3 py-1.5 text-xs text-rp-love hover:bg-rp-hlLow font-semibold border-b border-rp-hlMed"
                            >
                              Clear all
                            </button>
                          )}
                          {filterOptions[col]?.map(val => (
                            <label
                              key={val}
                              className="flex items-center gap-2 px-3 py-1.5 text-xs hover:bg-rp-hlLow transition-colors cursor-pointer"
                            >
                              <input
                                type="checkbox"
                                checked={selected.has(val)}
                                onChange={() => handleFilterChange(col, val)}
                                className="rounded border-rp-hlHigh text-rp-pine focus:ring-rp-pine/30 w-3.5 h-3.5"
                              />
                              <span className={selected.has(val) ? 'text-rp-pine font-semibold' : 'text-rp-text'}>{val}</span>
                            </label>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}
                {activeFilterCount > 0 && (
                  <>
                    <span className="text-[10px] text-rp-muted">
                      {visibleSetRef.current ? visibleSetRef.current.size.toLocaleString() : stats.count.toLocaleString()}
                    </span>
                    <button onClick={clearAllFilters} className="text-[10px] font-semibold text-rp-love hover:underline ml-1">
                      Clear
                    </button>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Bottom section — pinned to bottom */}
      <div className="absolute bottom-0 left-0 right-0 pointer-events-none z-50 p-4">
        <div className="flex flex-col gap-2 items-stretch">
          {/* Timeline indicator — offset from hotspots */}
          {viewMode === 'timeline' && timeRange && (() => {
            const range = timeRange.max - timeRange.min || 1;
            const loTs = timeRange.min + (timeFilter[0] / 1000) * range;
            const hiTs = timeRange.min + (timeFilter[1] / 1000) * range;
            const isFiltered = timeFilter[0] > 0 || timeFilter[1] < 1000;
            return (
            <div className="pointer-events-auto rp-card px-4 py-3" style={{ marginLeft: showHotspots && hotspots.length > 0 ? '260px' : 0 }}>
              <div className="flex items-center gap-3 mb-2">
                <Clock size={14} className="text-rp-iris shrink-0" />
                <span className="text-xs font-bold text-rp-text">
                  {isFiltered
                    ? `${new Date(loTs * 1000).toLocaleDateString('en-US', { year: 'numeric', month: 'short' })} — ${new Date(hiTs * 1000).toLocaleDateString('en-US', { year: 'numeric', month: 'short' })}`
                    : 'Drag handles to filter by date range'
                  }
                </span>
                {isFiltered && (
                  <button
                    onClick={() => { setTimeFilter([0, 1000]); timeFilterRef.current = [0, 1000]; }}
                    className="text-[10px] font-semibold text-rp-love hover:underline ml-auto"
                  >
                    Reset
                  </button>
                )}
              </div>
              {/* Dual range slider */}
              <div className="relative h-6">
                {/* Track background */}
                <div className="absolute top-1/2 -translate-y-1/2 left-0 right-0 h-1.5 bg-rp-hlMed rounded-full" />
                {/* Selected range highlight */}
                <div
                  className="absolute top-1/2 -translate-y-1/2 h-1.5 bg-rp-iris rounded-full transition-all duration-75"
                  style={{
                    left: `${timeFilter[0] / 10}%`,
                    right: `${100 - timeFilter[1] / 10}%`,
                  }}
                />
                {/* Low handle */}
                <input
                  type="range"
                  min={0}
                  max={1000}
                  value={timeFilter[0]}
                  onChange={(e) => {
                    const v = Math.min(parseInt(e.target.value), timeFilter[1] - 10);
                    setTimeFilter([v, timeFilter[1]]);
                    timeFilterRef.current = [v, timeFilter[1]];
                  }}
                  className="absolute inset-0 w-full h-full appearance-none bg-transparent pointer-events-none z-[2] [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-5 [&::-webkit-slider-thumb]:h-5 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-rp-iris [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-white [&::-webkit-slider-thumb]:shadow-md [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:pointer-events-auto [&::-webkit-slider-thumb]:relative [&::-webkit-slider-thumb]:z-[2] [&::-webkit-slider-runnable-track]:bg-transparent [&::-webkit-slider-runnable-track]:h-0"
                />
                {/* High handle */}
                <input
                  type="range"
                  min={0}
                  max={1000}
                  value={timeFilter[1]}
                  onChange={(e) => {
                    const v = Math.max(parseInt(e.target.value), timeFilter[0] + 10);
                    setTimeFilter([timeFilter[0], v]);
                    timeFilterRef.current = [timeFilter[0], v];
                  }}
                  className="absolute inset-0 w-full h-full appearance-none bg-transparent pointer-events-none z-[3] [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-5 [&::-webkit-slider-thumb]:h-5 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-rp-iris [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-white [&::-webkit-slider-thumb]:shadow-md [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:pointer-events-auto [&::-webkit-slider-thumb]:relative [&::-webkit-slider-thumb]:z-[3] [&::-webkit-slider-runnable-track]:bg-transparent [&::-webkit-slider-runnable-track]:h-0"
                />
              </div>
              <div className="flex justify-between mt-1">
                <span className="text-[9px] text-rp-muted font-semibold">
                  {new Date(timeRange.min * 1000).getFullYear()}
                </span>
                <span className="text-[9px] text-rp-muted font-semibold">
                  {new Date(timeRange.max * 1000).getFullYear()}
                </span>
              </div>
            </div>
            );
          })()}

          <div className="flex justify-between items-end gap-3">
          {/* View info */}
          <div className="pointer-events-auto rp-card px-3 py-2">
            <p className="text-[10px] font-semibold text-rp-muted uppercase tracking-wider">
              {VIEW_MODES[viewMode].desc}
            </p>
          </div>

          {/* Zoom controls */}
          <div className="pointer-events-auto flex items-center gap-1 rp-card px-2 py-1.5">
            <button onClick={() => handleZoom('out')} className="p-1.5 rounded hover:bg-rp-hlLow transition-colors text-rp-subtle">
              <ZoomOut size={15} />
            </button>
            <div className="w-14 text-center text-[11px] font-semibold text-rp-text tabular-nums">
              {(zoomLevel * 100).toFixed(0)}%
            </div>
            <button onClick={() => handleZoom('in')} className="p-1.5 rounded hover:bg-rp-hlLow transition-colors text-rp-subtle">
              <ZoomIn size={15} />
            </button>
            <div className="w-px h-4 bg-rp-hlHigh mx-0.5" />
            <button onClick={handleFitAll} className="p-1.5 rounded hover:bg-rp-hlLow transition-colors text-rp-subtle" title="Fit all">
              <Maximize2 size={15} />
            </button>
          </div>


        </div>
        </div>
      </div>



      {/* ── Right-edge Tab Toggle ─────────────── */}
      <button
        onClick={() => setShowDetailPanel(p => !p)}
        className={`absolute z-[90] top-1/2 -translate-y-1/2 pointer-events-auto transition-all duration-300 ${
          showDetailPanel ? 'right-[340px]' : 'right-0'
        } bg-rp-surface border border-r-0 border-rp-hlHigh rounded-l-lg shadow-rp px-1.5 py-6 hover:bg-rp-hlLow flex flex-col items-center gap-2`}
        title="Toggle detail panel"
      >
        {showDetailPanel ? (
          <ChevronRight size={16} className="text-rp-pine" />
        ) : (
          <ChevronLeft size={16} className="text-rp-pine" />
        )}
        <span className="text-[9px] font-bold text-rp-muted uppercase tracking-widest" style={{ writingMode: 'vertical-rl' }}>Panel</span>
      </button>

      {/* ── Detail Panel ─────────────────────── */}
      {showDetailPanel && (
        <div className="absolute inset-y-0 right-0 w-[340px] z-[100] bg-rp-surface border-l border-rp-hlHigh p-8 flex flex-col gap-5 shadow-[-12px_0_40px_-10px_rgba(87,82,121,0.1)] overflow-y-auto">
          <button
            onClick={() => setShowDetailPanel(false)}
            className="absolute top-4 right-4 p-1.5 rounded-lg border border-rp-hlHigh hover:bg-rp-love hover:text-white hover:border-rp-love transition-all text-rp-muted"
          >
            <X size={16} />
          </button>
          {selectedItem ? (
            <>
              <div>
                <p className="text-[10px] font-semibold text-rp-muted uppercase tracking-widest">Image</p>
                <h2 className="text-2xl font-extrabold text-rp-text tracking-tight leading-none mt-1">
                  #{selectedItem.id}
                </h2>
                <div className="h-0.5 w-10 bg-rp-love rounded-full mt-2" />
              </div>
              {/* Thumbnail */}
              {pointsRef.current[selectedItem.id] && (
                <div
                  className="w-full aspect-square rounded-xl overflow-hidden border border-rp-hlMed"
                  style={{
                    backgroundImage: `url(/data/atlas_${pointsRef.current[selectedItem.id].ai}.jpg)`,
                    backgroundPosition: `-${pointsRef.current[selectedItem.id].u}px -${pointsRef.current[selectedItem.id].v}px`,
                    backgroundSize: 'auto',
                  }}
                />
              )}
              <div className="space-y-2">
                {[
                  ['Position', `${selectedItem.x.toFixed(1)}, ${selectedItem.y.toFixed(1)}`],
                  ['Grid Cell', `${Math.floor(selectedItem.x / SPATIAL_CELL_SIZE)}, ${Math.floor(selectedItem.y / SPATIAL_CELL_SIZE)}`],
                  ...(metadata?.rows?.[selectedItem.id]
                    ? Object.entries(metadata.rows[selectedItem.id]).filter(([k]) => k !== 'id').map(([k, v]) => [k, v])
                    : []),
                ].map(([k, v]) => (
                  <div key={k} className="flex justify-between items-baseline border-b border-rp-hlMed/50 pb-1.5">
                    <span className="text-[10px] font-semibold text-rp-muted uppercase">{k}</span>
                    <span className="text-xs font-bold text-rp-text font-mono">{v}</span>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="flex flex-col items-center justify-center flex-1 text-center">
              <Eye size={32} className="text-rp-muted mb-3" />
              <p className="text-sm font-bold text-rp-text">No image selected</p>
              <p className="text-xs text-rp-muted mt-1">Click an image to see details</p>
            </div>
          )}
        </div>
      )}

      {/* ── Tooltip ─────────────────────────── */}
      {tooltip && viewMode !== 'clusters' && viewMode !== 'color' && viewMode !== 'carousel' && (
        <div
          className="absolute z-[80] pointer-events-none bg-rp-surface/95 backdrop-blur-md rounded-lg border border-rp-hlMed shadow-rp px-3 py-2"
          style={{ left: tooltip.x + 16, top: tooltip.y - 8, maxWidth: '180px' }}
        >
          <p className="text-xs font-bold text-rp-text">Image #{tooltip.id}</p>
        </div>
      )}

      {/* ── Stats Modal ──────────────────────── */}
      {showStats && (
        <div className="absolute inset-0 z-[200] bg-black/30 flex items-center justify-center p-8" onClick={() => setShowStats(false)}>
          <div className="bg-rp-surface rounded-2xl border border-rp-hlMed shadow-rp-lg max-w-lg w-full p-8" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <div className="bg-rp-pine text-white p-2 rounded-lg">
                  <Database size={18} />
                </div>
                <h2 className="text-xl font-extrabold text-rp-text">Collection Stats</h2>
              </div>
              <button onClick={() => setShowStats(false)} className="p-1.5 rounded-lg border border-rp-hlHigh hover:bg-rp-love hover:text-white transition-all text-rp-muted">
                <X size={16} />
              </button>
            </div>
            <div className="space-y-3">
              {[
                ['Total Images', stats.count.toLocaleString()],
                ['Clusters', hotspots.length.toString()],
                ['Atlas Textures', '13'],
                ['Thumbnail Size', `${THUMB_SIZE}px`],
                ['Current View', VIEW_MODES[viewMode]?.label || viewMode],
                ['Render FPS', stats.fps.toString()],
              ].map(([k, v]) => (
                <div key={k} className="flex justify-between items-baseline border-b border-rp-hlMed/50 pb-2">
                  <span className="text-sm text-rp-muted">{k}</span>
                  <span className="text-sm font-bold text-rp-text">{v}</span>
                </div>
              ))}
            </div>
            {hotspots.length > 0 && (
              <div className="mt-6">
                <h3 className="text-sm font-bold text-rp-text mb-3">Cluster Distribution</h3>
                <div className="space-y-1.5">
                  {hotspots.map((h, i) => (
                    <div key={h.id} className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: h.color }} />
                      <span className="text-xs text-rp-muted w-16">Region {i + 1}</span>
                      <div className="flex-1 h-2 bg-rp-hlMed rounded-full overflow-hidden">
                        <div className="h-full rounded-full" style={{ width: `${stats.count > 0 ? (h.count / stats.count * 100) : 0}%`, backgroundColor: h.color }} />
                      </div>
                      <span className="text-[10px] text-rp-subtle font-semibold tabular-nums w-14 text-right">{h.count.toLocaleString()}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Loading ──────────────────────────── */}
      {loading && !error && (
        <div className="absolute inset-0 z-[1000] bg-rp-base flex flex-col items-center justify-center">
          <div className="flex flex-col items-center gap-5 max-w-xs w-full px-8">
            <div className="bg-rp-pine text-white p-4 rounded-2xl shadow-rp-lg">
              <Layers size={36} className="animate-pulse" />
            </div>
            <div className="text-center">
              <h1 className="text-3xl font-extrabold tracking-tight text-rp-text">ImageSpace</h1>
              <p className="text-sm text-rp-muted mt-1">Loading collection...</p>
            </div>
            <div className="w-full">
              <div className="w-full h-1.5 bg-rp-hlMed rounded-full overflow-hidden">
                <div
                  className="h-full bg-rp-pine rounded-full transition-all duration-300"
                  style={{ width: `${Math.max(loadProgress, 3)}%` }}
                />
              </div>
              <div className="flex justify-between mt-1.5">
                <p className="text-[11px] text-rp-subtle">{statusMsg}</p>
                <p className="text-[11px] font-semibold text-rp-pine">{loadProgress}%</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Error ────────────────────────────── */}
      {error && (
        <div className="absolute inset-0 z-[2000] bg-rp-base flex items-center justify-center p-8">
          <div className="bg-rp-love/10 border border-rp-love/30 rounded-2xl p-10 max-w-lg w-full text-center">
            <h2 className="text-2xl font-extrabold text-rp-love mb-3">Load Error</h2>
            <code className="text-sm text-rp-text bg-rp-overlay rounded-lg p-3 block break-all">{error}</code>
            <button onClick={() => window.location.reload()} className="mt-6 rp-btn bg-rp-love text-white mx-auto">
              Reload
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

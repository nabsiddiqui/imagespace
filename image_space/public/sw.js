const CACHE_NAME = 'imagespace-v3';

// Only cache atlas textures (large/static) — NOT data.bin, manifest, metadata, etc.
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Only cache atlas_*.webp / atlas_*.jpg files
  if (!/\/data\/atlas_\d+\.\w+$/.test(url.pathname)) return;

  event.respondWith(
    caches.open(CACHE_NAME).then(cache =>
      cache.match(event.request).then(cached => {
        if (cached) return cached;
        return fetch(event.request).then(response => {
          if (response.ok) cache.put(event.request, response.clone());
          return response;
        });
      })
    )
  );
});

// Activate immediately (don't wait for existing clients to close)
self.addEventListener('install', () => self.skipWaiting());

// Clean up old caches on activation
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

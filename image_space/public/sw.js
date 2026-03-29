const CACHE_NAME = 'imagespace-v1';

// Cache atlas textures and data files on fetch
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Only cache data files (atlases, binary layout, etc.)
  if (!url.pathname.includes('/data/')) return;

  event.respondWith(
    caches.open(CACHE_NAME).then(cache =>
      cache.match(event.request).then(cached => {
        if (cached) return cached;
        return fetch(event.request).then(response => {
          // Only cache successful responses
          if (response.ok) cache.put(event.request, response.clone());
          return response;
        });
      })
    )
  );
});

// Clean up old caches on activation
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
});

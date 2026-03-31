// static/sw.js
self.addEventListener('push', function(event) {
  const data = event.data ? event.data.json() : {};
  const title   = data.title   || '⚠️ DisasterWatch Alert';
  const options = {
    body:    data.body    || 'New disaster alert in your area',
    icon:    '/static/icon.png',
    badge:   '/static/badge.png',
    vibrate: [200, 100, 200],
    data:    { url: data.url || '/dashboard' }
  };
  event.waitUntil(self.registration.showNotification(title, options));
});

self.addEventListener('notificationclick', function(event) {
  event.notification.close();
  event.waitUntil(clients.openWindow(event.notification.data.url));
});
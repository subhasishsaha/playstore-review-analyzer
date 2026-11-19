/**
 * Lightweight Zero-Dependency Node.js Server for Play Store Review Analyzer Frontend.
 * Serves static HTML/CSS/JS on port 3000 and proxies /api/* calls to FastAPI on port 8000.
 */
const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = process.env.PORT || 3000;
const PYTHON_BACKEND_PORT = process.env.PYTHON_PORT || 8000;
const PUBLIC_DIR = __dirname;

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

const server = http.createServer((req, res) => {
  // 1. Proxy /api/* requests to Python FastAPI backend
  if (req.url.startsWith('/api/') || req.url.startsWith('/docs') || req.url.startsWith('/openapi.json')) {
    const proxyOptions = {
      hostname: '127.0.0.1',
      port: PYTHON_BACKEND_PORT,
      path: req.url,
      method: req.method,
      headers: {
        ...req.headers,
        host: `127.0.0.1:${PYTHON_BACKEND_PORT}`,
      },
    };

    const proxyReq = http.request(proxyOptions, (proxyRes) => {
      res.writeHead(proxyRes.statusCode, proxyRes.headers);
      proxyRes.pipe(res, { end: true });
    });

    proxyReq.on('error', (err) => {
      res.writeHead(502, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        error: 'Bad Gateway',
        message: `Could not connect to Python backend on port ${PYTHON_BACKEND_PORT}. Make sure 'python src/app.py' is running.`,
        details: err.message,
      }));
    });

    req.pipe(proxyReq, { end: true });
    return;
  }

  // 2. Serve static frontend files
  let rawUrl = req.url.split('?')[0];
  let decodedPath;
  try {
    decodedPath = decodeURI(rawUrl);
  } catch (uriErr) {
    res.writeHead(400, { 'Content-Type': 'text/plain' });
    res.end('Bad Request: Malformed URI');
    return;
  }

  let safePath = path.normalize(decodedPath);
  if (safePath === '/' || safePath === '\\') {
    safePath = '/index.html';
  }

  const filePath = path.join(PUBLIC_DIR, safePath);

  // Security check: ensure path is within PUBLIC_DIR
  if (!filePath.startsWith(PUBLIC_DIR)) {
    res.writeHead(403);
    res.end('Access Denied');
    return;
  }

  fs.stat(filePath, (err, stats) => {
    if (err || !stats.isFile()) {
      // Fallback to index.html for SPA routing
      const indexPath = path.join(PUBLIC_DIR, 'index.html');
      fs.readFile(indexPath, (readErr, content) => {
        if (readErr) {
          res.writeHead(404, { 'Content-Type': 'text/plain' });
          res.end('404 Not Found');
          return;
        }
        res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
        res.end(content);
      });
      return;
    }

    const ext = path.extname(filePath).toLowerCase();
    const contentType = MIME_TYPES[ext] || 'application/octet-stream';

    fs.readFile(filePath, (readErr, content) => {
      if (readErr) {
        res.writeHead(500, { 'Content-Type': 'text/plain' });
        res.end('Internal Server Error');
        return;
      }
      res.writeHead(200, { 'Content-Type': contentType });
      res.end(content);
    });
  });
});

server.listen(PORT, () => {
  console.log(`====================================================`);
  console.log(`🚀 Play Store Review Analyzer Web UI Running!`);
  console.log(`📡 URL: http://localhost:${PORT}`);
  console.log(`🔌 Proxies /api/* -> http://127.0.0.1:${PYTHON_BACKEND_PORT}`);
  console.log(`====================================================`);
});

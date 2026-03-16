import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';
import { viteStaticCopy } from 'vite-plugin-static-copy';
import path from 'path';

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    wasm(),
    topLevelAwait(),
    // Copy lbug-wasm worker file to assets folder for production
    viteStaticCopy({
      targets: [
        {
          src: 'node_modules/@ladybugdb/wasm-core/lbug_wasm_worker.js',
          dest: 'assets'
        }
      ]
    }),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      // Fix for Rollup failing to resolve this deep import from @langchain/anthropic
      '@anthropic-ai/sdk/lib/transform-json-schema': path.resolve(__dirname, 'node_modules/@anthropic-ai/sdk/lib/transform-json-schema.mjs'),
      // Fix for mermaid d3-color prototype crash on Vercel (known issue with mermaid 10.9.0+ and Vite)
      'mermaid': path.resolve(__dirname, 'node_modules/mermaid/dist/mermaid.esm.min.mjs'),
    },
  },
  // Polyfill Buffer for isomorphic-git (Node.js API needed in browser)
  define: {
    global: 'globalThis',
  },
  // Optimize deps - exclude lbug-wasm from pre-bundling (it has WASM files)
  optimizeDeps: {
    exclude: ['@ladybugdb/wasm-core'],
    include: ['buffer'],
  },
  // Required for LadybugDB WASM (SharedArrayBuffer needs Cross-Origin Isolation)
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
    // Allow serving files from node_modules
    fs: {
      allow: ['..'],
    },
    proxy: {
      // Proxy Anthropic API calls — COEP blocks direct cross-origin fetch
      '/api/anthropic': {
        target: 'https://api.anthropic.com',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/anthropic/, ''),
        // IMPORTANT: selfHandleResponse: true prevents http-proxy from calling
        // proxyRes.pipe(res) automatically. Without this flag, http-proxy pipes
        // the response body UNCONDITIONALLY after firing the 'proxyRes' event —
        // regardless of res.headersSent. When our configure handler also pipes
        // (to flush SSE immediately), every SSE chunk is written twice and the
        // Anthropic SDK receives a doubled/corrupted event stream, silently
        // dropping all streaming tokens. selfHandleResponse: true gives us sole
        // ownership of the response body and fixes the double-pipe.
        selfHandleResponse: true,
        configure: (proxy) => {
          // ── Error handler ────────────────────────────────────────────────────
          // Without this, connection errors to api.anthropic.com are silently
          // swallowed when selfHandleResponse: true is set. The browser would
          // receive a hung/empty response and the Anthropic SDK would never
          // resolve or reject — making it look like tokens never arrive.
          proxy.on('error', (err, _req, res) => {
            console.error('❌ [Anthropic proxy] connection error:', err.message);
            if (!res.headersSent) {
              (res as any).writeHead(502, { 'content-type': 'application/json' });
            }
            (res as any).end(
              JSON.stringify({ error: 'Proxy connection failed', detail: err.message })
            );
          });

          proxy.on('proxyReq', (proxyReq, req) => {
            // CRITICAL: Strip browser headers so Anthropic treats this as a
            // server-side request. Without this, the browser's Origin header
            // leaks through and Anthropic demands the
            // 'anthropic-dangerous-direct-browser-access' header — causing
            // every request to fail with 401 even when the API key is valid.
            proxyReq.removeHeader('origin');
            proxyReq.removeHeader('referer');

            // DEV-only: log every outbound request so we can confirm the URL is right
            if (process.env.NODE_ENV !== 'production') {
              console.log(`🌐 [Anthropic proxy] → ${req.method} ${req.url}`);
            }
          });

          proxy.on('proxyRes', (proxyRes, _req, res) => {
            // DEV-only: log the response status so we can spot 401/400/500 from Anthropic
            if (process.env.NODE_ENV !== 'production') {
              console.log(
                `🌐 [Anthropic proxy] ← ${proxyRes.statusCode} ${proxyRes.headers['content-type'] ?? 'no-content-type'}`
              );
            }

            // Strip transfer-encoding — Node.js HTTP has already decoded chunked
            // encoding by the time proxyRes reaches us. Forwarding this header
            // would cause the browser to try to decode an already-decoded body.
            const headers = { ...proxyRes.headers };
            delete headers['transfer-encoding'];

            if (proxyRes.headers['content-type']?.includes('text/event-stream')) {
              // SSE: add cache/buffering headers, then pipe chunks as they arrive.
              // Node.js HTTP doesn't buffer, so data flows immediately without
              // any special flushing — X-Accel-Buffering disables nginx buffering
              // if a reverse proxy ever sits in front of the dev server.
              headers['cache-control'] = 'no-cache';
              headers['x-accel-buffering'] = 'no';
              res.writeHead(proxyRes.statusCode || 200, headers);
              proxyRes.pipe(res);
            } else {
              // Non-SSE (JSON errors, etc.): forward normally.
              // selfHandleResponse: true means we must handle ALL responses.
              res.writeHead(proxyRes.statusCode || 200, headers);
              proxyRes.pipe(res);
            }
          });
        },
      },
    },
  },
  // Also set for preview/production builds
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  // Worker configuration
  worker: {
    format: 'es',
    plugins: () => [wasm(), topLevelAwait()],
  },
});

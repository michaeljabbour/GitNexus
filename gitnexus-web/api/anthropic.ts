import type { VercelRequest, VercelResponse } from '@vercel/node';

/**
 * Anthropic API Proxy
 *
 * Routes browser requests to Anthropic's API to bypass COEP restrictions.
 * The app uses Cross-Origin-Embedder-Policy: require-corp for SharedArrayBuffer/WASM,
 * which blocks cross-origin fetches unless the response has CORP headers.
 * Anthropic's API doesn't set Cross-Origin-Resource-Policy, so we proxy through here.
 *
 * Requests: POST /api/anthropic/* → https://api.anthropic.com/*
 * The API key is sent from the client (stored in browser localStorage).
 */
export default async function handler(req: VercelRequest, res: VercelResponse) {
  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, x-api-key, anthropic-version, anthropic-beta, anthropic-dangerous-direct-browser-access');
    res.status(200).end();
    return;
  }

  // Build the target URL by stripping the /api/anthropic prefix
  const targetPath = (req.url || '').replace(/^\/api\/anthropic/, '');
  const targetUrl = `https://api.anthropic.com${targetPath}`;

  try {
    // Forward all headers except host-related ones
    const headers: Record<string, string> = {};
    const skipHeaders = new Set([
      'host', 'connection', 'transfer-encoding', 'content-length',
      'x-forwarded-for', 'x-forwarded-proto', 'x-forwarded-host',
      'x-vercel-id', 'x-vercel-forwarded-for',
    ]);

    for (const [key, value] of Object.entries(req.headers)) {
      if (!skipHeaders.has(key.toLowerCase()) && typeof value === 'string') {
        headers[key] = value;
      }
    }

    // Read request body
    let body: string | undefined;
    if (req.method === 'POST' || req.method === 'PUT') {
      if (typeof req.body === 'string') {
        body = req.body;
      } else if (req.body) {
        body = JSON.stringify(req.body);
      }
    }

    const response = await fetch(targetUrl, {
      method: req.method || 'POST',
      headers,
      body,
    });

    // Set CORS headers on response
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Expose-Headers', '*');

    // Forward response headers
    const skipResponseHeaders = new Set([
      'content-encoding', 'transfer-encoding', 'connection',
    ]);
    response.headers.forEach((value, key) => {
      if (!skipResponseHeaders.has(key.toLowerCase())) {
        res.setHeader(key, value);
      }
    });

    // Check if this is a streaming response
    const contentType = response.headers.get('content-type') || '';
    if (contentType.includes('text/event-stream')) {
      // Stream SSE responses — flush immediately, no buffering
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('X-Accel-Buffering', 'no');

      res.status(response.status);
      // Flush headers immediately so client starts receiving events
      res.flushHeaders();

      const reader = response.body?.getReader();
      if (!reader) {
        res.status(500).json({ error: 'No response body' });
        return;
      }

      const decoder = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        res.write(decoder.decode(value, { stream: true }));
        // Force flush each chunk so SSE events aren't buffered
        if (typeof (res as any).flush === 'function') {
          (res as any).flush();
        }
      }
      res.end();
    } else {
      // Non-streaming: forward as buffer
      res.status(response.status);
      const buffer = await response.arrayBuffer();
      res.send(Buffer.from(buffer));
    }
  } catch (error) {
    console.error('Anthropic proxy error:', error);
    res.status(500).json({ error: 'Proxy request failed', details: String(error) });
  }
}

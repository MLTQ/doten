# gruve-sdk

Make your app show up in your friends' Gruve lobbies. One call:

```js
import { announce } from 'gruve-sdk'

announce({ id: 'starchan', name: 'Starchan', port: 1420, hue: 280, blurb: 'imageboard for two' })
```

That's it. While your app runs, it appears as a tile on every friend's mesh; they click it and your
app opens for them, served from your machine over the tailnet. When your app exits (or crashes),
the tile disappears within the TTL (default 30s).

## How it works (and what this package is NOT)

The Gruve **agent** on your machine owns the tailnet node, identity, discovery, and proxying. This
package is just the **announce protocol**: a heartbeat POST to the agent on localhost saying "I'm
running on port X." Your app does no networking beyond that — which is the point.

The protocol is one endpoint, so SDKs for other languages are trivial (~20 lines each). Equivalent
curl, if you don't even want the package:

```bash
curl -X POST http://127.0.0.1:8088/gruve/announce \
  -H 'content-type: application/json' \
  -d '{"id":"myapp","name":"My App","port":3000,"hue":120,"ttl":30}'
# re-POST at least every ttl seconds; DELETE /gruve/announce?id=myapp when done
```

## Requirements for your app

Your app must serve its UI over **HTTP on localhost** — that's what the agent proxies to friends.

- **Web app / Node server**: you already do. Announce your port.
- **Tauri (dev)**: announce your vite dev-server port (e.g. 1420).
- **Tauri (production)**: the bundled webview isn't an HTTP server. Serve your `dist/` on a local
  port (e.g. `tauri-plugin-localhost`, or a tiny static server in your Rust `setup`) and announce
  that port.

**Heads-up:** friends get your app's *frontend*. `invoke()` calls to your Rust backend won't exist
in their browser — pure-frontend apps share perfectly; backend-dependent apps need an HTTP API to
share fully (or share a read-only/companion view).

## In a Tauri frontend

```js
import { announce } from 'gruve-sdk'

const gruve = announce({
  id: 'pharaoh',
  name: 'Pharaoh',
  port: 1420,
  hue: 45,
  onState: (s) => console.log('[gruve]', s), // 'announced' | 'waiting' (agent not running)
})
// gruve.stop() to leave the lobby early; auto-cleans on window unload / process exit
```

`announce()` never throws after validation and never blocks your app: if the Gruve agent isn't
running it just retries quietly, and your tile appears the moment the agent starts.

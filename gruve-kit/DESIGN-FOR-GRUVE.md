# Design for Gruve

The contract that makes an app work on the mesh — for you locally, and for every friend who opens
it from their lobby. Each rule exists because breaking it produced a real debugging night.

Check any built app with: `gruve doctor <dir>` (exit 0 = mesh-ready).

## 1. Hardcoded addresses are verboten

Your frontend must never contain `http://127.0.0.1:<port>` or `http://localhost:<port>` as a
reachable address. On a friend's machine, `127.0.0.1` is *their* machine — the request dies as
`Failed to fetch`. Gruve is the dispatch layer; names, not addresses:

```js
// ✗ verboten
const API = "http://127.0.0.1:3030";

// ✓ your own backend, by name (declared in your announce as upstreams: { api: 3030 })
const API = apiBase("api", { fallback: "http://127.0.0.1:3030" });

// ✓ a capability someone on the mesh provides (you don't know or care who)
const LLM = serviceBase("inference", { fallback: "http://127.0.0.1:8000" });
```

The `fallback` is the one place a localhost literal is allowed — it's only used when the app runs
standalone, outside Gruve. `gruve doctor` knows this exemption.

## 2. The long-lived backend owns the announce

Announce from the process that serves your app — never from a browser tab. Background tabs get
their timers throttled (~1/min), the registration expires, and your app flickers off the mesh
while people are using it.

```js
// in your server's startup (Node 18+), not in your frontend:
announce({ id: "veritas", name: "Veritas", port: API_PORT, upstreams: { api: API_PORT }, hue: 215 });
```

Best shape: **one process** serving UI + API on one port, announcing itself. Discoverable-on-the-
mesh ⇔ backend-is-up, by construction. (The agent also refuses announces for ports that aren't
actually listening, so a zombie page can't shadow a live backend.)

## 3. Sub-path servable

Friends load your app at `/peer/<node>/apps/<id>/` — your build must not assume it lives at `/`.

- Vite: `base: "./"` in `vite.config.js`
- No absolute asset references (`src="/assets/…"`) in your html
- Runtime-loaded public assets too: `loader.load("/models/duck.obj")` 404s under a sub-path —
  drop the leading slash (`"models/duck.obj"` resolves against the page URL). `gruve doctor`
  flags absolute public-asset paths.

## 4. Shared state goes through the session

Everyone viewing your hosted app is in one room on the host's agent. Don't build your own
sync socket; use the session's key/value state (LWW, host-ordered, replayed to late joiners):

```js
const session = joinSession();             // local no-op room when standalone
session.state.subscribe((key, value) => …); // remote changes + retained state on join
session.state.set("instrument", "ETH/USDT");
```

Validate values off the wire like any untrusted input (route them through your store's setters).
Viewers may open your app **Solo** — `joinSession()` then behaves exactly like standalone (local
room). Backend multiplayer for solo viewers belongs in your backend, behind upstreams.

## 5. The app never networks beyond localhost + the SDK

No peer addresses, no tailnet IPs, no connection management in app code. The agent owns
transport and identity; your app talks to `localhost` and asks for things *by name*. This is the
boundary that keeps generated/vibecoded apps safe and makes the same app run in every mode.

External public APIs (exchange feeds, SaaS) are the app's own business — but anything that should
come from *a machine on the mesh* must go through `apiBase`/`serviceBase`, or it won't.

## Quick reference

| You want | Use | Resolves to |
|---|---|---|
| Your own backend | `apiBase("api")` | `…/apps/<id>/__gruve/api` on whoever hosts you |
| A mesh capability | `serviceBase("inference")` | `/svc/inference` — viewer's agent finds a provider |
| Shared view state | `joinSession().state` | the host's session room |
| Be discoverable | `announce({ service?: true })` from your server | lobby tile, or `/svc/<id>/` mesh-wide |

## 6. Sync state, not events — and audio is gesture-gated

Born from a haunted play button (clicked 100×/sec by a sync loop) and a stereo echo.

- **Appliers are idempotent renderers.** Applying remote state means driving your store so the UI
  *reflects* it — never simulating input (no synthetic clicks) and never triggering the action
  that *produced* the state. The SDK guards the transport (your own `set()` doesn't echo back,
  unchanged values don't fire or relay, and the room periodically reconciles against the hub's
  retained state) — but "apply twice = no-op" is your job.
- **Sound is gesture-gated by the runner.** The stage blocks autoplay by default: your app can
  make sound from the viewer's own gestures (and keep playing in the background), but
  remote-triggered audio — a friend's action, synced playback state, backend polling — stays
  silent until the viewer enables **🔊 Listen along**. Design playback so state changes render UI
  (timeline, waveform) and audio starts on local gesture or listen-along; never auto-play from
  applied state.

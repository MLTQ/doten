# Gruve Integration Kit

You're holding everything needed to make an existing app **mesh-ready**: discoverable in the Gruve
lobby, openable by friends over the network, with its backend reachable and (optionally) its state
shared between viewers. No prior context required — this file is the context.

## What Gruve is, in three sentences

Gruve is a single binary (`gruve`) that runs on each person's machine: it embeds a WireGuard mesh
node (no Tailscale install, no sudo), serves a lobby UI on `http://localhost:8088`, and acts as a
**dispatch layer** — apps and capabilities are reached *by name*, never by address. Friends'
machines appear in each other's lobbies; clicking a tile opens an app served from whoever hosts it,
with cursors/whiteboard/shared-state sessions composited on top. An app integrates by following the
contract in `DESIGN-FOR-GRUVE.md` and **announcing** itself to the local agent.

## What's in the box

| File | What it is |
|---|---|
| `README.md` | this guide |
| `DESIGN-FOR-GRUVE.md` | the contract — the rules, each with the failure that produced it |
| `gruve` | the agent binary; also the linter: `./gruve doctor <dir>` |
| `sdk/` | the JS SDK (`gruve-sdk`) — install as a `file:` dependency |
| `sdk-rs/` | the Rust adapter (zero deps) — announce + dispatch from any Rust app |
| `PROTOCOL.md` | the adapter wire spec — what a gruve-sdk in ANY language must implement |

## The 10-minute proof (no network, no keys)

The agent is a complete local test harness — announced apps work **before joining any mesh**:

```bash
./gruve                      # lobby opens at http://localhost:8088 (ignore the join screen)
# in another terminal — pretend this is your app:
python3 -m http.server 9000 &
curl -X POST http://127.0.0.1:8088/gruve/announce -H 'content-type: application/json' \
  -d '{"id":"myapp","name":"My App","port":9000,"hue":200,"ttl":60}'
```

Look under the join card: **"Your machine — works without a network"** now shows a *My App* tile.
Click it — your app, served through the agent at `/apps/myapp/`. That path is exactly what friends
will hit over the mesh (as `/peer/<net>/<node>/apps/myapp/`), so anything that works here works
there. Stop re-announcing and the tile vanishes after the TTL.

## The integration recipe

**0. Find the app's long-lived process.** The thing that's running whenever the app is "on" — a
backend server, not a browser tab or webview page (background tabs get their timers throttled and
the registration expires mid-session). This process will announce, and ideally also serve the UI.

**1. One process, one port.** Best shape: the backend serves the **built frontend** + its API on
one port. (For Tauri apps: the production webview is NOT an HTTP server — add static serving of
`dist/` to your backend, or a tiny static server in the Rust/Python side.)

**2. Kill hardcoded addresses in the frontend.** Anything like `http://127.0.0.1:3030` dies on a
friend's machine (that's *their* 127.0.0.1). Route through the SDK:

```js
import { apiBase } from "gruve-sdk";
const API = apiBase("api", { fallback: "http://127.0.0.1:3030" }); // fallback = standalone dev only
fetch(`${API}/api/whatever`);
```

`apiBase` returns an **absolute** URL only when you build it that way — if your code uses
`new URL(...)`, anchor it: `const ABS = base.startsWith("/") ? location.origin + base : base`.
Watch for **build-time env overrides** (`VITE_*` in `.env`/`.env.local`): vite inlines them,
silently defeating runtime resolution. Env overrides must apply to the standalone fallback ONLY.

**3. Announce from the backend.** Declare the UI port and any backend ports as named upstreams.

From Node (the SDK works in Node 18+):
```js
import { announce } from "gruve-sdk";
announce({ id: "myapp", name: "My App", port: UI_PORT, upstreams: { api: API_PORT }, hue: 280 });
```

From Python (or anything — the protocol is one POST, re-sent as a heartbeat):
```python
import threading, urllib.request, json

def announce_to_gruve(app_id, name, port, upstreams=None, hue=280, ttl=60):
    body = json.dumps({"id": app_id, "name": name, "port": port, "ttl": ttl,
                       "hue": hue, "upstreams": upstreams or {}}).encode()
    def beat():
        req = urllib.request.Request("http://127.0.0.1:8088/gruve/announce", data=body,
                                     headers={"content-type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req, timeout=2)
        except Exception:
            pass  # agent not running — fine, we retry; the app works without Gruve
        threading.Timer(ttl / 3, beat).start()
    beat()
```

Notes: the agent **refuses announces for ports that aren't actually listening** (start your server
first), and the heartbeat must outlive you wandering off — that's why it lives in the backend.

**4. Sub-path proof.** Friends load the app at `/peer/<net>/<node>/apps/<id>/` — it must not assume
it lives at `/`. Vite: `base: "./"`. No absolute asset paths in HTML **or in code** —
`loader.load("/models/x.obj")` 404s; use `"models/x.obj"` (relative to the page).

**5. Run the linter.** `./gruve doctor <built-app-dir>` — exit 0 or it tells you exactly what will
break for remote viewers (hardcoded localhost URLs, absolute asset paths, missing wiring). It knows
the legitimate exemptions (SDK fallbacks).

**6. Optional — shared state.** Everyone viewing the hosted app shares a session room. Cursors and
whiteboard come free (the lobby composites them). For app-level sync (shared view, shared layout):

```js
import { joinSession } from "gruve-sdk";
const session = joinSession();                       // standalone → local no-op room
session.state.subscribe((key, value) => { /* apply remote */ });
session.state.set("instrument", "ETH/USDT");         // LWW per key, replayed to late joiners
```

The SDK fires subscribers for REMOTE truth only (your own set() never echoes back; unchanged
values never fire or relay) and periodically reconciles against the room's retained state. Your
job: appliers are idempotent renderers — drive your store, never simulate input events, never
trigger the action that produced the state. Debounce bursty writes (drags); validate values off
the wire through your store's setters. AUDIO: the stage blocks autoplay by default — remote-
triggered sound needs the viewer's 🔊 Listen along opt-in; gesture audio always works.

Viewers choose **Together** (shared session) or **Solo** (own session) when opening your app. In
Solo, `joinSession()` silently becomes a local room — your app needs no special handling, but
design knowing both doors exist: backend multiplayer (dedicated-server style) belongs in YOUR
backend behind upstreams, where it works for solo viewers too.

## Tauri-specific warnings

- **`invoke()` does not exist for remote viewers.** Friends get your *frontend* over HTTP; Tauri
  IPC stays home. Anything remote viewers need must be reachable over HTTP (an upstream). Apps
  whose logic lives behind `invoke()` can still share fine if the backend also speaks HTTP.
- The webview is **WebKit**: `window.confirm/alert` don't work (build your own confirms), and
  buttons near `draggable` elements need `onMouseDown={e => e.preventDefault()}` or drags eat
  their clicks.
- Dev-server frontends (vite dev) can't be served under a sub-path — integrate against the
  **built** app.

## Done when

- [ ] `./gruve doctor <dist>` → "Contract holds" (exit 0)
- [ ] App announced by its backend; tile visible in the lobby; survives 5 minutes untouched
- [ ] App opens via `http://localhost:8088/apps/<id>/` — UI renders, assets load
- [ ] Backend calls work through `/apps/<id>/__gruve/<upstream>/...`
- [ ] (If synced) two browser windows on the served app converge

## Joining a real mesh (the actual point, eventually)

Get an invite phrase from a network host (four words, possibly `words@domain`). Run `./gruve`,
type it into the join screen. Your announced apps appear on every member's lobby; theirs appear on
yours. Nothing about your integration changes — that's the contract working.

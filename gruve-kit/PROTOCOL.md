# The Gruve Adapter Protocol

The wire-level contract a `gruve-sdk-<language>` must implement. The JS SDK (`sdk-js/`) is the
reference implementation; this document is normative when they disagree. Everything speaks plain
HTTP/WebSocket to the **local agent** at `http://127.0.0.1:8088` (override: `GRUVE_AGENT` env or
equivalent). Apps never talk to the mesh directly — the agent owns transport and identity.

Three conformance levels. An adapter states which it implements.

---

## Level 1 — Announce (required)

Make a running app discoverable. One endpoint:

```
POST /gruve/announce          content-type: application/json
{
  "id": "myapp",              // ^[a-z0-9][a-z0-9-]{0,31}$
  "name": "My App",           // lobby display name (defaults to id for services)
  "port": 9000,               // localhost port the app's HTTP surface is on
  "ttl": 60,                  // seconds until the agent forgets us (clamped 5..300)
  "icon": "whiteboard",       // optional lobby glyph
  "hue": 280,                 // optional tile hue 0-360
  "blurb": "",                // optional one-liner
  "upstreams": {"api": 3030}, // optional named backend ports (Level 2)
  "service": false            // true = mesh capability at /svc/<id>/, no lobby tile
}
→ 200 {"ok":true,"ttl":N,"appPath":"/apps/<id>/"}   (or "svcPath":"/svc/<id>/")
→ 409 announced port not listening, or id collides with a built-in app
→ 400 malformed

DELETE /gruve/announce?id=<id>[&service=1]   → withdraw
```

**Required behaviors** (this is what makes an adapter trustworthy):

| # | Behavior |
|---|---|
| 1.1 | Heartbeat: re-POST every `ttl/3` seconds from a **long-lived process** (never a UI page/tab — timers get throttled and the registration expires mid-session) |
| 1.2 | Quiet resilience: agent absent/unreachable → swallow the error, keep retrying. The host app must work identically with no Gruve installed |
| 1.3 | Never crash, block, or slow the host app. Announce from a background thread/task |
| 1.4 | Listen-then-announce: bind your HTTP port before the first POST (the agent probes it and 409s otherwise) |
| 1.5 | Withdraw on clean shutdown (best effort DELETE); rely on TTL for crashes |

## Level 2 — Dispatch (required for apps with frontends; useful for backends)

Names, never addresses. Three URL forms an adapter must produce correctly:

- **Own backend from your frontend** (`apiBase`): when the page is served under
  `…/apps/<id>/` (detect via `location.pathname`), the base is
  `<that prefix>/__gruve/<upstream-name>`; otherwise a standalone fallback the integrator
  provides. The mesh path MUST win over any build-time/env override. If consumers construct
  `new URL(...)`-style absolute URLs, anchor the path to the page origin.
- **Mesh capability** (`serviceBase`): `/svc/<name>` when the caller is a page served through an
  agent (same-origin); `http://<agent>/svc/<name>` when the caller is a backend process. The
  agent resolves a provider (local first, then any joined network) — the caller never knows where
  it lives.
- **Frontends never hardcode `localhost:<port>`** — `gruve doctor <dist>` enforces this
  (fallbacks passed to apiBase/serviceBase are exempt).

## Level 3 — Session (optional: shared state between viewers)

WebSocket to the app's room **on the host's agent**. A frontend connects same-origin:
`ws(s)://<page host><prefix>/gruve/session/<appId>` where `<prefix>` is everything before
`/apps/<id>/` in the page path (empty locally, `/peer/<net>/<node>` on a friend's machine).
If the page URL carries `?gruve-solo=1`, do NOT connect — behave as a local no-op room.

Wire messages (client → hub unless noted):

```
{"t":"hello","kind":"app","name":"app"}        first frame; kind "app" = invisible state channel
                                                (kind "viewer" = appears in roster/cursors)
hub → {"t":"welcome","id":N,"roster":[...],"drawing":[...],"state":{key:value,...}}
{"t":"state","key":"k","value":any}             set (LWW; hub retains; relayed to others)
hub → {"t":"state","key":"k","value":any,"from":N}
{"t":"sync"}                                    anti-entropy request
hub → {"t":"syncstate","state":{key:value,...}}
(viewer kind also: cursor / draw / clear — see reference impl)
```

**Required semantics** — these encode hard-won lessons; do not skip them:

| # | Behavior |
|---|---|
| 3.1 | `set()` no-ops when the value is unchanged (deep/JSON equality) |
| 3.2 | Subscribers are notified of **remote truth only**: incoming state/welcome/syncstate that *changed* something. A caller's own `set()` never echoes back to its subscribers |
| 3.3 | Anti-entropy: send `{"t":"sync"}` every ~20s, on reconnect, and on tab-wake; apply `syncstate` through the same equality-guarded path (crossing writes otherwise leave a *stable* desync) |
| 3.4 | Reconnect quietly with backoff; re-`hello`; the welcome replay restores state |
| 3.5 | Document for integrators: appliers are **idempotent renderers** — drive the store, never simulate input events, never trigger the action that produced the state (contract rule 6) |

---

## Integration patterns by app shape

| App shape | Path onto the mesh |
|---|---|
| Web frontend + backend (Veritas) | Full: one process serves built FE + API, announces with upstreams; L2 in the FE; L3 if shared state wanted |
| Native GUI (egui/iced/Unity) | (i) compile the UI for web (eframe has first-class WASM targets) and split native-only logic into an upstream backend; (ii) ship a **companion web view** for the mesh while the native console stays local; (iii) **service-only** (below) |
| Headless engine / model server / data pipeline | `service: true` — a capability at `/svc/<id>/` that any friend's app can consume by name. No UI required. Often the highest-value integration for the least work |

## Acceptance checklist for a new adapter

- [ ] L1 behaviors 1.1–1.5 demonstrated against a local agent (`./gruve`, no network needed)
- [ ] Announced app/service visible in `/gruve/manifest.json` and survives 5+ minutes
- [ ] Zero (or near-zero) dependencies; no panics/exceptions escape into the host app
- [ ] If L2: URL forms verified through `/apps/<id>/__gruve/…` and `/svc/<name>` locally
- [ ] If L3: a two-client test shows (a) 5 identical sets → exactly 1 relay observed,
      (b) a deliberate crossed-write desync converges after one sync cycle
- [ ] README states the conformance level

Reference implementations: `sdk-js/` (L1+L2+L3), `sdk-rs/` (L1+L2).

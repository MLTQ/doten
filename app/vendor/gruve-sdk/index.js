// gruve-sdk — the integration layer, client side.
//
// One concept: ANNOUNCE. Your app tells the local Gruve agent "I'm running, my UI is on this
// localhost port" and the agent does the rest (lists you in the lobby, proxies friends to you over
// the tailnet). No networking, no tailnet, no identity in your app — that all lives in the agent.
//
// The protocol is one HTTP POST, so this whole package is a convenience wrapper: heartbeat,
// retry-until-agent-appears, and cleanup on exit. Works in browsers/webviews (Tauri) and Node 18+.

const DEFAULT_AGENT = 'http://127.0.0.1:8088'

// Matches a page being served THROUGH a Gruve agent: /apps/<id>/… locally,
// or /peer/<node>/apps/<id>/… on a friend's machine.
const SERVED_RE = /^(.*\/apps\/[a-z0-9-]+)\//

/**
 * Resolve the base URL for one of your declared backend upstreams.
 *
 * If the page is being served through a Gruve agent (yours or a friend's), this returns the
 * agent-relative path that reaches your backend over the mesh. Otherwise (running standalone,
 * e.g. plain `vite dev` or inside Tauri) it returns `fallback` — your usual localhost URL.
 *
 * Use it once where you build API URLs:
 *   const API = apiBase('api', { fallback: 'http://127.0.0.1:3030' })
 *   fetch(`${API}/history?...`)
 */
export function apiBase(name = 'api', { fallback } = {}) {
  if (typeof location !== 'undefined') {
    const m = location.pathname.match(SERVED_RE)
    if (m) return `${m[1]}/__gruve/${name}`
  }
  return fallback ?? `__gruve/${name}`
}

/** True when this page is being served through a Gruve agent (local or a friend's). */
export function isServedByGruve() {
  return typeof location !== 'undefined' && SERVED_RE.test(location.pathname)
}

// Captures (prefix before /apps/, app id) — prefix is "" locally, "/peer/<node>" on a friend's machine.
const SESSION_RE = /^(.*)\/apps\/([a-z0-9-]+)\//

/**
 * Base URL for a named mesh SERVICE — a capability, not an address. Your frontend asks for
 * "inference" or "collector"; whichever node on the mesh announces that service receives the
 * request, dispatched by the viewer's own agent. Nothing in your app ever knows where it lives.
 *
 *   const LLM = serviceBase('inference', { fallback: 'http://127.0.0.1:8000' })
 *   fetch(`${LLM}/v1/chat/completions`, …)
 *
 * Served through Gruve (your machine or a friend's): returns `/svc/<name>` — same-origin to the
 * serving agent, which resolves a provider (local first, then mesh). Standalone: `fallback`.
 * Backend (Node) code: pass `agent` to target the local agent explicitly.
 */
export function serviceBase(name, opts = {}) {
  if (opts.agent) return `${opts.agent.replace(/\/$/, '')}/svc/${name}`
  if (isServedByGruve()) return `/svc/${name}`
  return opts.fallback ?? `/svc/${name}`
}

/**
 * Join this app's presence session — the same room the Gruve overlay uses for cursors, shared by
 * everyone viewing this app (host + friends over the mesh). Gives the app a tiny shared key/value
 * state: last-write-wins per key, ordered by the host's hub, retained for late joiners, dropped
 * when the room empties.
 *
 * Standalone (not served through Gruve — plain vite dev, Tauri) this returns a LOCAL room: same
 * API, state lives in memory only. Your app behaves identically with or without Gruve.
 *
 * const session = joinSession()
 * session.state.subscribe((key, value) => { ... })   // remote changes (+ retained state on join)
 * session.state.set('instrument', 'ETH/USDT')        // broadcast to the room
 *
 * @param {{ onPeers?: (count: number) => void }} [opts] optional viewer-count callback
 */
export function joinSession(opts = {}) {
  const state = new Map()
  const subs = new Set()
  const queue = []
  let ws = null
  let closed = false
  let syncTimer = null

  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b)

  // Subscribers hear REMOTE truth only: other viewers' changes, join replay, anti-entropy
  // reconciliation. Your own set() never echoes back to you, and unchanged values never fire —
  // so the naive applier ("subscribe → drive my store") cannot loop.
  const applyRemote = (key, value) => {
    if (same(state.get(key), value)) return
    state.set(key, value)
    subs.forEach((cb) => { try { cb(key, value, { remote: true }) } catch {} })
  }

  // Solo door: the viewer chose their own session (no shared room, no cursors). The lobby opens
  // the app with ?gruve-solo=1; we behave exactly like standalone — local room, backend untouched
  // (dedicated-server style: the app's own backend multiplayer still works through upstreams).
  const solo = typeof location !== 'undefined' && new URLSearchParams(location.search).has('gruve-solo')
  const m = !solo && typeof location !== 'undefined' && location.pathname.match(SESSION_RE)

  const send = (key, value) => {
    if (same(state.get(key), value)) return // unchanged — nothing to say
    state.set(key, value) // silently: the caller already knows what they set
    if (!m) return // standalone: local room only
    const msg = JSON.stringify({ t: 'state', key, value })
    if (ws?.readyState === 1) ws.send(msg)
    else queue.push(msg)
  }

  if (m) {
    const [, prefix, appId] = m
    const wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:'
    const url = `${wsProto}//${location.host}${prefix}/gruve/session/${appId}`
    const requestSync = () => { if (ws?.readyState === 1) ws.send(JSON.stringify({ t: 'sync' })) }
    const connect = () => {
      if (closed) return
      ws = new WebSocket(url)
      ws.onopen = () => {
        ws.send(JSON.stringify({ t: 'hello', kind: 'app', name: 'app' }))
        while (queue.length) ws.send(queue.shift())
      }
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data)
        if (msg.t === 'welcome' || msg.t === 'syncstate') {
          Object.entries(msg.state || {}).forEach(([k, v]) => applyRemote(k, v))
          if (msg.t === 'welcome') opts.onPeers?.((msg.roster || []).length)
        } else if (msg.t === 'state') {
          applyRemote(msg.key, msg.value)
        }
      }
      ws.onclose = () => { if (!closed) setTimeout(connect, 3000) } // reconnect quietly
    }
    connect()
    // Anti-entropy: events can cross in flight and leave a STABLE desync (each side holding the
    // other's value). Periodically reconcile against the hub's retained LWW state — no-ops when
    // already in sync, converges everyone when not.
    syncTimer = setInterval(requestSync, 20000)
    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', () => { if (!document.hidden) requestSync() })
    }
  }

  return {
    state: {
      set: send,
      get: (key) => state.get(key),
      subscribe(cb) { subs.add(cb); return () => subs.delete(cb) },
    },
    connected: () => !!ws && ws.readyState === 1,
    leave() { closed = true; clearInterval(syncTimer); ws?.close() },
  }
}

/**
 * Announce this app to the local Gruve agent. Resolves immediately; keeps announcing in the
 * background (and keeps retrying quietly if the agent isn't running yet — your app works fine
 * without Gruve, and pops into the lobby the moment the agent starts).
 *
 * @param {object} opts
 * @param {string} opts.id     stable slug, ^[a-z0-9][a-z0-9-]{0,31}$  (e.g. "starchan")
 * @param {string} opts.name   display name for the lobby tile
 * @param {number} opts.port   localhost port your app's HTTP UI is on
 * @param {string} [opts.icon] lobby glyph: "whiteboard" | "garden" (more later)
 * @param {number} [opts.hue]  tile hue 0-360 (oklch)
 * @param {string} [opts.blurb] one-liner under the tile name
 * @param {string} [opts.agent] agent base URL (default http://127.0.0.1:8088)
 * @param {number} [opts.ttl]  seconds before the agent forgets us without a re-announce (default 30)
 * @param {Object<string,number>} [opts.upstreams] named backend ports, e.g. { api: 3030 } —
 *   reachable from your frontend at apiBase('api') wherever it's served
 * @param {(state: 'announced'|'waiting') => void} [opts.onState] optional state-change callback
 * @returns {{ stop: () => void }}
 */
export function announce(opts) {
  // Already being served THROUGH an agent (locally or on a friend's machine)? Then this page is a
  // guest, not the host instance — announcing would re-register with the wrong port. No-op.
  if (isServedByGruve()) return { stop: () => {} }

  const id = opts.id
  const name = opts.name || (opts.service ? id : '')
  // In a browser the dev server's own port is almost always the right one — default to it.
  const port = opts.port || (typeof location !== 'undefined' ? Number(location.port) : 0)
  if (!id || !name || !port) throw new Error('gruve announce: id, name and port are required')
  const agent = (opts.agent || DEFAULT_AGENT).replace(/\/$/, '')
  // Default TTL must outlive browser background-tab throttling: Chrome clamps timers in hidden
  // tabs to ~1/min, so a 30s TTL expires mid-session whenever the announcing tab loses focus.
  // 90s rides out throttled beats. (Long-lived processes can pass a tighter ttl.)
  const ttl = opts.ttl || 90
  const body = JSON.stringify({
    id, name, port, ttl,
    icon: opts.icon || 'whiteboard',
    hue: opts.hue || 250,
    blurb: opts.blurb || '',
    upstreams: opts.upstreams || {},
    service: !!opts.service, // capability (mesh-wide /svc/<id>/), not a lobby tile
  })

  let stopped = false
  let lastState = null
  const setState = (s) => {
    if (s !== lastState) {
      lastState = s
      opts.onState?.(s)
    }
  }

  const post = async () => {
    if (stopped) return
    try {
      const r = await fetch(agent + '/gruve/announce', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body,
      })
      setState(r.ok ? 'announced' : 'waiting')
    } catch {
      setState('waiting') // agent not running — keep retrying quietly
    }
  }

  post()
  const timer = setInterval(post, Math.max(2, Math.floor(ttl / 3)) * 1000)

  const stop = () => {
    if (stopped) return
    stopped = true
    clearInterval(timer)
    // best-effort dereg; keepalive lets it fire during page unload
    const q = `id=${encodeURIComponent(id)}${opts.service ? '&service=1' : ''}`
    fetch(`${agent}/gruve/announce?${q}`, { method: 'DELETE', keepalive: true }).catch(() => {})
  }

  if (typeof window !== 'undefined') {
    window.addEventListener('beforeunload', stop)
    // Tab coming back from throttled sleep: re-assert immediately instead of waiting a beat.
    document.addEventListener('visibilitychange', () => { if (!document.hidden) post() })
  } else if (typeof process !== 'undefined') {
    process.once('exit', stop)
    process.once('SIGINT', () => { stop(); process.exit(130) })
  }

  return { stop }
}

export interface AnnounceOptions {
  /** stable slug, ^[a-z0-9][a-z0-9-]{0,31}$ */
  id: string
  /** display name for the lobby tile (defaults to id for services) */
  name?: string
  /** announce a mesh-wide SERVICE (capability at /svc/<id>/ on every node) instead of a lobby app */
  service?: boolean
  /** localhost port your app's HTTP UI is on (browser default: the page's own port) */
  port?: number
  /** named backend ports, e.g. { api: 3030 } — reachable from the frontend via apiBase('api') */
  upstreams?: Record<string, number>
  /** lobby glyph: "whiteboard" | "garden" (more later) */
  icon?: string
  /** tile hue 0-360 (oklch) */
  hue?: number
  /** one-liner under the tile name */
  blurb?: string
  /** agent base URL (default http://127.0.0.1:8088) */
  agent?: string
  /** seconds before the agent forgets us without a re-announce (default 30) */
  ttl?: number
  /** state-change callback */
  onState?: (state: 'announced' | 'waiting') => void
}

export interface AnnounceHandle {
  stop: () => void
}

export function announce(opts: AnnounceOptions): AnnounceHandle

/**
 * Base URL for a declared backend upstream. Returns the agent-relative path when the page is
 * served through Gruve (works locally AND from a friend's machine), else `fallback`.
 */
export function apiBase(name?: string, opts?: { fallback?: string }): string

/**
 * Base URL for a named mesh service (capability, not address): `/svc/<name>` when served through
 * Gruve — the viewer's agent dispatches to whichever node provides it — else `fallback`.
 * Pass `agent` from backend (Node) code to target the local agent explicitly.
 */
export function serviceBase(name: string, opts?: { fallback?: string; agent?: string }): string

/** True when this page is being served through a Gruve agent (local or a friend's). */
export function isServedByGruve(): boolean

export interface SessionHandle {
  state: {
    /**
     * Set a key for everyone in the room (LWW, host-ordered, retained for late joiners).
     * No-ops when the value is unchanged; never echoes back to your own subscribers.
     */
    set: (key: string, value: unknown) => void
    get: (key: string) => unknown
    /**
     * Fires for REMOTE truth only: other viewers' changes, join replay, and periodic
     * anti-entropy reconciliation — never your own set(), never unchanged values.
     * Appliers must still be idempotent and must never simulate input events.
     */
    subscribe: (cb: (key: string, value: unknown, meta: { remote: true }) => void) => () => void
  }
  connected: () => boolean
  leave: () => void
}

/**
 * Join this app's presence session (the room shared with everyone viewing this app over the mesh).
 * Standalone (not served through Gruve) returns a local in-memory room with the same API.
 */
export function joinSession(opts?: { onPeers?: (count: number) => void }): SessionHandle

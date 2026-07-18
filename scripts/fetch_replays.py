#!/usr/bin/env python3
"""Fetch Dota 2 replays by match id, for populating a Doten test library.

For each match id it asks OpenDota for the replay's cluster + salt, downloads
the `.dem.bz2` from Valve's replay servers, and decompresses it to a `.dem`
you can drop straight into the Doten window (or point `parse_cli` at).

No Steam API key required — this uses OpenDota's public match endpoint, which
exposes `cluster` and `replay_salt`. (The key-gated Steam Web API path is for
the live "subscribe to a friend's games" scraper; this is the manual harness.)

Usage:
    python3 scripts/fetch_replays.py                 # default same-stack set
    python3 scripts/fetch_replays.py 8898003442 ...  # explicit match ids
    python3 scripts/fetch_replays.py --out ~/dota_replays 8898003442

Caveat: Valve purges replays after a couple of weeks for most matches, so a
match id that returns 404 has simply aged out — pick a more recent one.
"""

from __future__ import annotations

import argparse
import bz2
import json
import os
import sys
import time
import urllib.request

# A verified set of nine recent games sharing the same five-player stack
# (accounts 172099728 gpk-stack, 480412663, 165564598, 317880638, 196878136) —
# handy default for exercising the cross-game / player-cloud features.
DEFAULT_MATCHES = [
    8898117600, 8898003442, 8891227605, 8891108985, 8889524021,
    8889468264, 8888066684, 8887998750, 8886638110,
]

OPENDOTA = "https://api.opendota.com/api/matches/{}"
REPLAY_URL = "http://replay{cluster}.valve.net/570/{match}_{salt}.dem.bz2"
UA = {"User-Agent": "doten-fetch-replays/1.0"}


def http_get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def replay_url(match_id: int) -> str | None:
    """Resolve a match id to its Valve replay URL via OpenDota, or None."""
    try:
        meta = json.loads(http_get(OPENDOTA.format(match_id), timeout=30))
    except Exception as e:
        print(f"  ! {match_id}: OpenDota lookup failed ({e})")
        return None
    cluster, salt = meta.get("cluster"), meta.get("replay_salt")
    if not cluster or not salt:
        print(f"  ! {match_id}: no replay salt yet (unparsed or too new)")
        return None
    return REPLAY_URL.format(cluster=cluster, match=match_id, salt=salt)


def fetch_one(match_id: int, out_dir: str) -> bool:
    dest = os.path.join(out_dir, f"{match_id}.dem")
    if os.path.exists(dest):
        print(f"  = {match_id}: already have {dest}")
        return True
    url = replay_url(match_id)
    if not url:
        return False
    print(f"  ↓ {match_id}: {url}")
    try:
        compressed = http_get(url, timeout=180)
    except urllib.error.HTTPError as e:
        note = " (aged out of Valve's servers)" if e.code == 404 else ""
        print(f"  ! {match_id}: download failed HTTP {e.code}{note}")
        return False
    except Exception as e:
        print(f"  ! {match_id}: download failed ({e})")
        return False
    try:
        raw = bz2.decompress(compressed)
    except Exception as e:
        print(f"  ! {match_id}: decompress failed ({e})")
        return False
    with open(dest, "wb") as f:
        f.write(raw)
    print(f"  ✓ {match_id}: {len(raw) / 1e6:.0f} MB -> {dest}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch Dota 2 replays for Doten.")
    ap.add_argument("matches", nargs="*", type=int, help="match ids (default: bundled same-stack set)")
    ap.add_argument("--out", default="replays", help="output directory (default: ./replays)")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between matches (rate-limit courtesy)")
    args = ap.parse_args()

    matches = args.matches or DEFAULT_MATCHES
    os.makedirs(args.out, exist_ok=True)
    print(f"Fetching {len(matches)} replay(s) into {args.out}/")

    ok = 0
    for i, mid in enumerate(matches):
        if i:
            time.sleep(args.delay)
        if fetch_one(mid, args.out):
            ok += 1
    print(f"\nDone: {ok}/{len(matches)} replay(s) ready in {args.out}/")
    print("Drop the .dem files into the Doten window (or: cargo run --release "
          "--example parse_cli -- <file>.dem).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

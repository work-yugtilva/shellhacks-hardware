"""Local SQLite retrieval of short coaching tips for detector events.

Tips live in ``tips_seed.json`` next to this module. The SQLite database is
generated outside the repo (default ``~/.lanetalk/tips.db``, override with the
``LANETALK_TIPS_DB`` environment variable) and is rebuilt automatically when
the seed file changes.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3

from lanetalk.events import Event

SEED_PATH = Path(__file__).with_name("tips_seed.json")


def _db_path() -> Path:
    return Path(os.environ.get("LANETALK_TIPS_DB", Path.home() / ".lanetalk" / "tips.db"))


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS tips (
            id INTEGER PRIMARY KEY,
            kind TEXT NOT NULL,
            match_key TEXT,
            match_value TEXT,
            text TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS tips_kind ON tips (kind);
        """
    )
    _sync_seed(conn)
    return conn


def _sync_seed(conn: sqlite3.Connection) -> None:
    raw = SEED_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    row = conn.execute("SELECT value FROM meta WHERE key = 'seed_sha256'").fetchone()
    if row and row[0] == digest:
        return
    rows = []
    for tip in json.loads(raw):
        key, value = next(iter(tip.get("match", {}).items()), (None, None))
        rows.append((tip["kind"], key, value, tip["text"]))
    with conn:
        conn.execute("DELETE FROM tips")
        conn.executemany(
            "INSERT INTO tips (kind, match_key, match_value, text) VALUES (?, ?, ?, ?)", rows
        )
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('seed_sha256', ?)", (digest,))


def retrieve(event: Event | str, top_k: int = 3) -> list[str]:
    """Return up to ``top_k`` tips for an event, most specific first.

    Tips whose ``match`` fits the event details (e.g. ``direction`` or
    ``class``) rank above generic tips for the same kind; tips whose match
    does not fit are excluded. Unknown kinds return ``[]``.
    """
    if top_k <= 0:
        return []
    kind = event if isinstance(event, str) else event.kind
    details = {} if isinstance(event, str) else event.details
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT match_key, match_value, text FROM tips WHERE kind = ? ORDER BY id", (kind,)
        ).fetchall()
    finally:
        conn.close()
    specific = [t for k, v, t in rows if k is not None and str(details.get(k)) == v]
    generic = [t for k, _, t in rows if k is None]
    return (specific + generic)[:top_k]

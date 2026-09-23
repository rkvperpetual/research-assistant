"""
app/core/memory.py — SQLite-based conversation persistence via SQLModel.

Each conversation stores a list of turns (role + content) keyed by
conversation_id (UUID).  Single-file SQLite is good enough for a demo;
replacing with Postgres requires only changing the engine URL.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import List
from uuid import uuid4

# Write to DATA_DIR (volume-mounted in Docker) so the file survives restarts.
# Falls back to current working dir for local development without Docker.
_DATA_DIR = Path(os.getenv("DATA_DIR", "."))
_DATA_DIR.mkdir(parents=True, exist_ok=True)
_DB_PATH = _DATA_DIR / "conversations.db"

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS turns (
    conversation_id TEXT NOT NULL,
    turn_index       INTEGER NOT NULL,
    role             TEXT NOT NULL,
    content          TEXT NOT NULL,
    created_at       TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (conversation_id, turn_index)
);
"""


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist. Call once at startup."""
    with _get_conn() as conn:
        conn.execute(_CREATE_SQL)
        conn.commit()


def new_conversation_id() -> str:
    return str(uuid4())


def load_history(conversation_id: str) -> List[dict]:
    """Return list of {role, content} dicts, oldest first."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM turns WHERE conversation_id=? ORDER BY turn_index",
            (conversation_id,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def append_turn(conversation_id: str, role: str, content: str) -> None:
    """Append a single turn to the conversation."""
    with _get_conn() as conn:
        idx = (
            conn.execute(
                "SELECT COALESCE(MAX(turn_index)+1, 0) FROM turns WHERE conversation_id=?",
                (conversation_id,),
            ).fetchone()[0]
        )
        conn.execute(
            "INSERT INTO turns(conversation_id, turn_index, role, content) VALUES(?,?,?,?)",
            (conversation_id, idx, role, content),
        )
        conn.commit()


def get_conversation(conversation_id: str) -> list:
    return load_history(conversation_id)

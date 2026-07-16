#!/usr/bin/env python3
"""
NOVA RAG Daemon — Persistent context revival engine.

Indexes every conversation from chat.jsonl into a lightweight SQLite+BM25 store.
On query, retrieves the top-K most relevant past exchanges and returns them as
context snippets to inject into the brain prompt.

Socket: /tmp/nova_rag.sock
Protocol (newline-terminated JSON):
  Request:  {"query": "...", "top_k": 3}
  Response: {"hit": true/false, "context": "...", "memories": [...]}

Context storage: ~/nova/memory/
  ├── rag.db          — SQLite full-text search index
  └── user_profile.json — Static personal profile (name, prefs, context hints)

Memory is NEVER loaded in full — only top-K snippets are returned to avoid
RAM overhead. The daemon runs independently and is always available.
"""

import os
import re
import sys
import json
import math
import time
import signal
import socket
import sqlite3
import logging
import threading
from pathlib import Path
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
SOCKET_PATH   = "/tmp/nova_rag.sock"
MEMORY_DIR    = os.path.expanduser("~/nova/memory")
DB_PATH       = os.path.join(MEMORY_DIR, "rag.db")
PROFILE_PATH  = os.path.join(MEMORY_DIR, "user_profile.json")
CHAT_LOG      = os.path.expanduser("~/nova/logs/chat.jsonl")
LOG_PATH      = os.path.expanduser("~/nova/logs/rag_daemon.log")
INDEX_INTERVAL = 60   # re-index chat.jsonl every 60 seconds

os.makedirs(MEMORY_DIR, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [RAG] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("nova.rag")

# ── Global ────────────────────────────────────────────────────────────────────
_shutdown      = threading.Event()
_db_lock       = threading.Lock()
_last_index_ts = 0.0

# ── SQLite FTS5 Setup ─────────────────────────────────────────────────────────
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories USING fts5(
            entry_id,
            timestamp,
            intent,
            user_query,
            assistant_reply,
            tags,
            tokenize='porter ascii'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS indexed_ids (
            entry_id TEXT PRIMARY KEY
        )
    """)
    conn.commit()
    return conn


# ── Indexer — reads chat.jsonl and adds new entries ───────────────────────────
def index_chat_log():
    global _last_index_ts
    if not os.path.exists(CHAT_LOG):
        return

    try:
        with _db_lock:
            conn = get_db()
            # Fetch already-indexed IDs
            known = {r[0] for r in conn.execute("SELECT entry_id FROM indexed_ids")}

            new_count = 0
            with open(CHAT_LOG, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue

                    eid = entry.get("id", "")
                    if not eid or eid in known:
                        continue

                    conn.execute(
                        "INSERT INTO memories VALUES (?,?,?,?,?,?)",
                        (
                            eid,
                            entry.get("timestamp", ""),
                            entry.get("intent", ""),
                            entry.get("user", ""),
                            entry.get("assistant", ""),
                            " ".join(entry.get("topic_tags", [])),
                        )
                    )
                    conn.execute("INSERT OR IGNORE INTO indexed_ids VALUES (?)", (eid,))
                    new_count += 1

            conn.commit()
            conn.close()

            if new_count:
                log.info(f"Indexed {new_count} new memory entries")
            _last_index_ts = time.time()

    except Exception as e:
        log.warning(f"Indexer error: {e}")


# ── BM25-like Retrieval via FTS5 ──────────────────────────────────────────────
def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """Return top_k most relevant memory snippets for the given query."""
    try:
        with _db_lock:
            conn = get_db()
            # FTS5 rank is BM25 by default
            rows = conn.execute(
                """
                SELECT entry_id, timestamp, intent, user_query, assistant_reply, tags, rank
                FROM memories
                WHERE memories MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, top_k)
            ).fetchall()
            conn.close()

        results = []
        for row in rows:
            results.append({
                "id":        row[0],
                "timestamp": row[1],
                "intent":    row[2],
                "user":      row[3],
                "reply":     row[4],
                "tags":      row[5],
                "score":     row[6],
            })
        return results

    except Exception as e:
        log.warning(f"Retrieval error: {e}")
        return []


def format_context(memories: list[dict]) -> str:
    """Format retrieved memories as a compact context block for brain injection."""
    if not memories:
        return ""
    lines = ["[Recalled memories]"]
    for m in memories:
        ts = m["timestamp"][:16] if m["timestamp"] else "?"
        lines.append(f"• [{ts}] You: {m['user'][:120]}")
        lines.append(f"  NOVA: {m['reply'][:200]}")
    return "\n".join(lines)


# ── User Profile ──────────────────────────────────────────────────────────────
def load_profile() -> dict:
    """Load static user profile for persistent context hints."""
    if os.path.exists(PROFILE_PATH):
        try:
            with open(PROFILE_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def get_profile_context(profile: dict) -> str:
    """Format profile as a short context prefix."""
    if not profile:
        return ""
    lines = ["[User profile]"]
    for k, v in profile.items():
        lines.append(f"• {k}: {v}")
    return "\n".join(lines)


# ── Background re-indexer thread ──────────────────────────────────────────────
def indexer_thread():
    log.info("Background indexer started")
    # Initial index on startup
    index_chat_log()
    while not _shutdown.is_set():
        time.sleep(INDEX_INTERVAL)
        if not _shutdown.is_set():
            index_chat_log()
    log.info("Indexer thread exiting")


# ── Request Handler ───────────────────────────────────────────────────────────
def handle_client(conn: socket.socket):
    try:
        raw = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            raw += chunk
            if raw.endswith(b"\n"):
                break

        request = json.loads(raw.decode().strip())
        query   = request.get("query", "").strip()
        top_k   = int(request.get("top_k", 3))

        if not query:
            conn.sendall((json.dumps({"hit": False, "context": "", "memories": []}) + "\n").encode())
            return

        # Ensure index is reasonably fresh
        if time.time() - _last_index_ts > INDEX_INTERVAL:
            index_chat_log()

        memories = retrieve(query, top_k=top_k)
        hit      = len(memories) > 0
        context  = format_context(memories)

        # Also inject profile if available
        profile     = load_profile()
        prof_ctx    = get_profile_context(profile)
        if prof_ctx and context:
            context = prof_ctx + "\n\n" + context
        elif prof_ctx:
            context = prof_ctx

        response = {
            "hit":      hit,
            "context":  context,
            "memories": memories,
            "count":    len(memories),
        }
        conn.sendall((json.dumps(response) + "\n").encode())

    except Exception as e:
        log.warning(f"Client error: {e}")
        try:
            conn.sendall((json.dumps({"hit": False, "context": "", "error": str(e)}) + "\n").encode())
        except Exception:
            pass
    finally:
        conn.close()


# ── Socket Server ─────────────────────────────────────────────────────────────
def socket_server():
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    os.chmod(SOCKET_PATH, 0o600)
    server.listen(8)
    server.settimeout(1.0)
    log.info(f"RAG socket listening at {SOCKET_PATH}")

    while not _shutdown.is_set():
        try:
            conn, _ = server.accept()
            threading.Thread(target=handle_client, args=(conn,), daemon=True).start()
        except socket.timeout:
            continue
        except Exception as e:
            log.error(f"Socket error: {e}")

    server.close()
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)
    log.info("RAG socket server closed")


# ── Signal Handling ───────────────────────────────────────────────────────────
def shutdown(signum, frame):
    log.info("Shutdown signal received")
    _shutdown.set()

signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT, shutdown)


# ── Entry Point ───────────────────────────────────────────────────────────────
def main():
    log.info("NOVA RAG daemon starting...")
    # Start background indexer
    idx = threading.Thread(target=indexer_thread, daemon=True)
    idx.start()
    # Block on socket server
    socket_server()
    _shutdown.set()
    idx.join(timeout=5)
    log.info("NOVA RAG daemon stopped")


if __name__ == "__main__":
    main()

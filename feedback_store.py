import sqlite3
import threading
from datetime import datetime, timezone

DB_PATH = "feedback.db"
_lock = threading.Lock()


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                question TEXT,
                rating TEXT NOT NULL,
                tags TEXT,
                comment TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()


def insert_feedback(session_id, message_id, rating, question=None, tags=None, comment=None):
    with _lock, sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """INSERT INTO feedback (session_id, message_id, question, rating, tags, comment, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                message_id,
                question,
                rating,
                ",".join(tags) if tags else None,
                comment,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()

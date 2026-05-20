"""
OpenSeek — Async SQLite detection logger.
"""
import aiosqlite
from datetime import datetime, timezone
from config import DB_PATH

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS detections (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT    NOT NULL,
    media_type       TEXT    NOT NULL,
    url              TEXT    NOT NULL,
    authenticity_score REAL  NOT NULL,
    risk_level       TEXT    NOT NULL
);
"""


async def init_db() -> None:
    """Create the detections table if it does not exist."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(CREATE_TABLE_SQL)
        await db.commit()


async def log_detection(
    media_type: str,
    url: str,
    authenticity_score: float,
    risk_level: str,
) -> int:
    """Insert a detection record and return its new row id."""
    ts = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """
            INSERT INTO detections (timestamp, media_type, url, authenticity_score, risk_level)
            VALUES (?, ?, ?, ?, ?)
            """,
            (ts, media_type, url, authenticity_score, risk_level),
        )
        await db.commit()
        return cursor.lastrowid


async def get_recent_detections(limit: int = 50) -> list[dict]:
    """Return the most recent `limit` detection records."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

"""
Emotional History Router
"""

import time
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

from utils.database import database, emotion_events
from utils.logger import setup_logger
import sqlalchemy

router = APIRouter()
logger = setup_logger(__name__)


@router.get("/", summary="Get emotion history")
async def get_history(
    session_id: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    since_timestamp: Optional[float] = Query(None, description="Unix timestamp"),
):
    """Fetch emotion events with optional filters."""
    query = emotion_events.select()

    conditions = []
    if session_id:
        conditions.append(emotion_events.c.session_id == session_id)
    if user_id:
        conditions.append(emotion_events.c.user_id == user_id)
    if since_timestamp:
        conditions.append(emotion_events.c.timestamp >= since_timestamp)

    if conditions:
        query = query.where(sqlalchemy.and_(*conditions))

    query = query.order_by(emotion_events.c.timestamp.desc()).limit(limit)

    rows = await database.fetch_all(query)
    return {"events": [dict(r) for r in rows], "count": len(rows)}


@router.get("/summary", summary="Get emotion summary for a period")
async def get_summary(
    user_id: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None),
    hours: int = Query(24, ge=1, le=168),
):
    """Aggregate emotion counts over a time window (default: last 24h)."""
    since = time.time() - (hours * 3600)

    query = emotion_events.select().where(
        emotion_events.c.timestamp >= since
    )
    if user_id:
        query = query.where(emotion_events.c.user_id == user_id)
    if session_id:
        query = query.where(emotion_events.c.session_id == session_id)

    rows = await database.fetch_all(query)
    events = [dict(r) for r in rows]

    # Aggregate
    counts: dict[str, int] = {}
    confidences: dict[str, list[float]] = {}

    for e in events:
        em = e["emotion"]
        counts[em] = counts.get(em, 0) + 1
        confidences.setdefault(em, []).append(e["confidence"])

    avg_confidences = {k: round(sum(v) / len(v), 3) for k, v in confidences.items()}

    dominant = max(counts, key=counts.get) if counts else "neutral"
    total = len(events)

    return {
        "period_hours": hours,
        "total_events": total,
        "dominant_emotion": dominant,
        "emotion_counts": counts,
        "emotion_percentages": {
            k: round(v / total * 100, 1) for k, v in counts.items()
        } if total else {},
        "avg_confidences": avg_confidences,
        "face_detected_count": sum(1 for e in events if e.get("face_detected")),
    }

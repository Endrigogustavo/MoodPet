"""
Alerts Router — Emergency and emotional alerts management
"""

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, EmailStr
from typing import Optional

from services.alert_service import (
    send_email_alert,
    build_emotion_alert_email,
    build_no_face_alert_email,
)
from utils.database import database, alerts_log
from utils.logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__)


class EmotionAlertRequest(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    user_name: str = "Usuário"
    emotion: str
    duration_minutes: float
    contact_email: Optional[str] = None


class NoFaceAlertRequest(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    user_name: str = "Usuário"
    minutes_without_face: float
    contact_email: Optional[str] = None


@router.post("/emotion", summary="Trigger emotion-based alert")
async def trigger_emotion_alert(
    request: EmotionAlertRequest,
    background_tasks: BackgroundTasks,
):
    subject, body = build_emotion_alert_email(
        request.emotion, request.duration_minutes, request.user_name
    )

    # Log to DB
    await database.execute(
        alerts_log.insert().values(
            session_id=request.session_id,
            user_id=request.user_id,
            alert_type="emotion",
            emotion=request.emotion,
            message=subject,
            sent_to=request.contact_email,
        )
    )

    if request.contact_email:
        background_tasks.add_task(send_email_alert, request.contact_email, subject, body)

    return {"status": "alert_queued", "type": "emotion", "emotion": request.emotion}


@router.post("/no-face", summary="Trigger no-face detection alert")
async def trigger_no_face_alert(
    request: NoFaceAlertRequest,
    background_tasks: BackgroundTasks,
):
    subject, body = build_no_face_alert_email(
        request.minutes_without_face, request.user_name
    )

    await database.execute(
        alerts_log.insert().values(
            session_id=request.session_id,
            user_id=request.user_id,
            alert_type="no_face",
            message=subject,
            sent_to=request.contact_email,
        )
    )

    if request.contact_email:
        background_tasks.add_task(send_email_alert, request.contact_email, subject, body)

    return {"status": "alert_queued", "type": "no_face", "minutes": request.minutes_without_face}


@router.get("/", summary="List alert history")
async def list_alerts(session_id: Optional[str] = None, limit: int = 50):
    query = alerts_log.select().order_by(alerts_log.c.created_at.desc()).limit(limit)
    if session_id:
        query = query.where(alerts_log.c.session_id == session_id)
    rows = await database.fetch_all(query)
    return {"alerts": [dict(r) for r in rows]}

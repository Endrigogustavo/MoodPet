"""
Emotion Detection Router
Handles continuous frame processing from mobile and IoT sources
"""

import time
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.emotion_service import emotion_service, NEGATIVE_EMOTIONS
from services.alert_service import update_face_seen, check_no_face_alert, get_time_since_last_face
from utils.database import database, emotion_events
from utils.logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__)


# ── Schemas ────────────────────────────────────────────────────────────────────

class FrameRequest(BaseModel):
    frame_b64: str = Field(..., description="Base64 encoded JPEG/PNG frame")
    session_id: str = Field(..., description="Unique session identifier")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    pet_name: str = Field("MoodPet", description="User's pet name for messages")
    source: Literal["mobile", "iot"] = Field("mobile", description="Frame source device")


class EmotionResponse(BaseModel):
    emotion: str
    emotion_variant: str
    emotion_zone: str
    support_tip: str
    confidence: float
    secondary_emotion: Optional[str]
    all_scores: dict
    face_detected: bool
    processing_time_ms: float
    message: Optional[str]
    music_suggestions: list[str]
    timestamp: float
    no_face_alert: bool
    seconds_since_last_face: float
    should_trigger_alert: bool


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=EmotionResponse, summary="Analyze a single frame")
async def analyze_frame(request: FrameRequest):
    """
    Process a base64-encoded frame and return emotion analysis.
    
    Call this endpoint continuously (every 500ms–1s) from the app
    while the camera is active. The API handles deduplication and storage.
    """
    if not request.frame_b64:
        raise HTTPException(status_code=400, detail="frame_b64 is required")

    logger.info(
        "analyze request | session_id=%s source=%s user_id=%s frame_len=%s",
        request.session_id,
        request.source,
        request.user_id or "-",
        len(request.frame_b64),
    )

    result = await emotion_service.analyze_frame_base64(
        request.frame_b64,
        session_id=request.session_id,
        pet_name=request.pet_name,
    )

    # Update face tracking
    update_face_seen(request.session_id, result.face_detected)
    no_face_alert = check_no_face_alert(request.session_id)
    seconds_since = get_time_since_last_face(request.session_id)

    # Persist to database
    try:
        await database.execute(
            emotion_events.insert().values(
                session_id=request.session_id,
                user_id=request.user_id,
                emotion=result.emotion,
                confidence=result.confidence,
                secondary_emotion=result.secondary_emotion,
                all_scores=result.all_scores,
                face_detected=result.face_detected,
                source=request.source,
                timestamp=result.timestamp,
            )
        )
    except Exception as e:
        logger.error(f"DB write error: {e}")

    should_trigger = result.emotion in NEGATIVE_EMOTIONS and result.confidence > 0.7

    logger.info(
        "analyze result | session_id=%s emotion=%s confidence=%.3f face_detected=%s processing_ms=%.1f trigger_alert=%s no_face_alert=%s",
        request.session_id,
        result.emotion,
        result.confidence,
        result.face_detected,
        result.processing_time_ms,
        should_trigger,
        no_face_alert,
    )

    return EmotionResponse(
        emotion=result.emotion,
        emotion_variant=result.emotion_variant,
        emotion_zone=result.emotion_zone,
        support_tip=result.support_tip,
        confidence=result.confidence,
        secondary_emotion=result.secondary_emotion,
        all_scores=result.all_scores,
        face_detected=result.face_detected,
        processing_time_ms=result.processing_time_ms,
        message=result.message,
        music_suggestions=result.music_suggestions,
        timestamp=result.timestamp,
        no_face_alert=no_face_alert,
        seconds_since_last_face=seconds_since,
        should_trigger_alert=should_trigger,
    )


@router.get("/session/{session_id}/status", summary="Get session status")
async def get_session_status(session_id: str):
    """Return current session face-tracking status."""
    return {
        "session_id": session_id,
        "seconds_since_last_face": get_time_since_last_face(session_id),
        "no_face_alert_pending": check_no_face_alert(session_id),
    }

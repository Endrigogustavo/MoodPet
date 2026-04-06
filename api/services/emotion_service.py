"""
Emotion Detection Service
Uses DeepFace + OpenCV for real-time facial expression analysis.
Falls back to MediaPipe landmarks when DeepFace is unavailable.
"""

import asyncio
import base64
import concurrent.futures
import io
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from utils.logger import setup_logger

logger = setup_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

EMOTION_MAP = {
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fear": "anxious",
    "disgust": "disgusted",
    "surprise": "surprised",
    "neutral": "neutral",
}

NEGATIVE_EMOTIONS = {"sad", "angry", "anxious", "disgusted", "fearful"}
POSITIVE_EMOTIONS = {"happy", "surprised"}

EMOTION_MESSAGES = {
    "happy": [
        "Que sorriso lindo! Seu pet está adorando ver você assim 🌟",
        "Você está radiante hoje! Isso contagia até o {pet_name}! 😄",
        "Felicidade é contagiante — e o {pet_name} já percebeu! 🐾",
    ],
    "sad": [
        "Parece que você está passando por um momento difícil. O {pet_name} está aqui com você 💙",
        "Tudo bem ficar triste às vezes. O {pet_name} manda um abraço virtual 🤗",
        "Você não está sozinho(a). O {pet_name} sente a sua presença e te acompanha 💜",
    ],
    "angry": [
        "Respira fundo... o {pet_name} está aqui para te acalmar 🌿",
        "Momentos difíceis passam. O {pet_name} te convida para uma pausa 🍃",
        "Que tal uma respiração profunda com o {pet_name}? Inspira... expira... 💨",
    ],
    "anxious": [
        "O {pet_name} percebeu que você pode estar preocupado(a). Você está seguro(a) 🛡️",
        "Vamos respirar juntos? O {pet_name} fica do seu lado 💛",
        "Um passo de cada vez. O {pet_name} acredita em você! 🌱",
    ],
    "neutral": [
        "O {pet_name} está de olho em você, com carinho 🐾",
        "Dia tranquilo? O {pet_name} curte essa energia 😊",
        "Tudo em paz por aqui! O {pet_name} está feliz com sua companhia ✨",
    ],
    "surprised": [
        "Uau! O {pet_name} também se surpreendeu! 😲",
        "Algo inesperado? O {pet_name} adorou a reação! 🎉",
    ],
    "disgusted": [
        "O {pet_name} faz a mesma cara às vezes 😅 Tudo vai melhorar!",
        "Passou alguma coisa ruim? O {pet_name} está aqui para alegrar seu dia 🌈",
    ],
}

MUSIC_SUGGESTIONS = {
    "sad": ["Weightless - Marconi Union", "Clair de Lune - Debussy", "Someone Like You - Adele"],
    "angry": ["Breathe (2 AM) - Anna Nalick", "Fix You - Coldplay", "The Sound of Silence"],
    "anxious": ["Gymnopédie No.1 - Satie", "River Flows in You - Yiruma", "Experience - Ludovico"],
    "happy": ["Happy - Pharrell Williams", "Good as Hell - Lizzo", "Can't Stop the Feeling"],
    "neutral": ["Lo-fi Hip Hop Beats", "Nature Sounds", "Ambient Focus Playlist"],
}


@dataclass
class EmotionResult:
    emotion: str
    confidence: float
    all_scores: dict
    emotion_variant: str = "steady"
    emotion_zone: str = "balanced"
    support_tip: str = "Respire com calma e siga em pequenos passos."
    secondary_emotion: Optional[str] = None
    face_detected: bool = True
    processing_time_ms: float = 0.0
    message: Optional[str] = None
    music_suggestions: list = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        # FastAPI/Pydantic cannot serialize numpy scalar types (e.g., numpy.float32).
        # DeepFace/OpenCV pipelines may return numpy values, so normalize everything
        # to native Python types here.
        try:
            self.confidence = float(self.confidence)
        except Exception:
            self.confidence = 0.0

        try:
            self.processing_time_ms = float(self.processing_time_ms)
        except Exception:
            self.processing_time_ms = 0.0

        try:
            self.timestamp = float(self.timestamp)
        except Exception:
            self.timestamp = time.time()

        if self.all_scores is None:
            self.all_scores = {}
        else:
            cleaned: dict[str, float] = {}
            for k, v in dict(self.all_scores).items():
                try:
                    cleaned[str(k)] = float(v)
                except Exception:
                    # Drop non-numeric entries to keep response serializable.
                    continue
            self.all_scores = cleaned


@dataclass
class SessionTemporalState:
    scores_ema: dict[str, float] = field(default_factory=dict)
    last_emotion: str = "neutral"
    pending_emotion: Optional[str] = None
    pending_count: int = 0
    updated_at: float = field(default_factory=time.time)


class EmotionDetectionService:
    """Core emotion detection service using DeepFace."""

    _instance = None
    _deepface_loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._fallback_prev_scores: Optional[dict[str, float]] = None
            self._session_state: dict[str, SessionTemporalState] = {}
            # OpenCV Haar cascades and some image ops can be unstable when invoked concurrently
            # across multiple threads. Serialize detection work to avoid rare segfaults.
            self._detect_lock = threading.Lock()
            self._detect_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            self._load_models()

    def _load_models(self):
        # DeepFace+TensorFlow currently segfaults on Python 3.13 in this stack.
        # Keep API operational by using the OpenCV fallback detector.
        if sys.version_info >= (3, 13):
            logger.warning("⚠️  DeepFace disabled on Python 3.13, using fallback detector")
            self._deepface_loaded = False
        else:
            try:
                from deepface import DeepFace
                self.DeepFace = DeepFace
                # Warm up the model
                dummy = np.zeros((48, 48, 3), dtype=np.uint8)
                DeepFace.analyze(dummy, actions=["emotion"], enforce_detection=False, silent=True)
                self._deepface_loaded = True
                logger.info("✅ DeepFace model loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️  DeepFace not available ({e}), using fallback detector")
                self._deepface_loaded = False

        # OpenCV fallback detector.
        # Keep it enabled by default so Python 3.13 environments still detect faces/emotions.
        # If instability is observed, disable explicitly via MOODPET_DISABLE_OPENCV=1.
        disable_opencv = os.getenv("MOODPET_DISABLE_OPENCV", "0") == "1"
        if sys.version_info >= (3, 13) and not disable_opencv:
            logger.warning(
                "⚠️  Running OpenCV fallback on Python 3.13; "
                "set MOODPET_DISABLE_OPENCV=1 if you observe instability"
            )

        self._opencv_enabled = not disable_opencv
        if self._opencv_enabled:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_eye.xml"
            )
            self.smile_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_smile.xml"
            )
        else:
            self.face_cascade = None
            self.eye_cascade = None
            self.smile_cascade = None

        # MediaPipe face detector (more robust than Haar on mobile-like frames).
        # Used only to find the face bounding box; emotion scoring remains heuristic.
        self._mp_face_detector = None
        # Only attempt MediaPipe when the classic Solutions API exists; on Python 3.13
        # this wheel does not expose mp.solutions, and importing mediapipe can pull heavy
        # native deps with no benefit for our bbox-only usage.
        if sys.version_info < (3, 13) and self._opencv_enabled:
            try:
                import mediapipe as mp  # type: ignore

                if getattr(mp, "solutions", None) is not None:
                    self._mp_face_detector = mp.solutions.face_detection.FaceDetection(
                        model_selection=0,
                        min_detection_confidence=0.5,
                    )
                    logger.info("✅ MediaPipe face detector ready")
                else:
                    logger.warning(
                        "⚠️  MediaPipe Solutions API unavailable in this environment; using Haar only"
                    )
            except Exception as e:
                logger.warning(f"⚠️  MediaPipe face detector unavailable ({e}), using Haar only")

    @staticmethod
    def _clip01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _normalize_scores(self, scores: dict[str, float]) -> dict[str, float]:
        total = sum(max(0.0, v) for v in scores.values()) or 1.0
        return {k: round(max(0.0, v) / total, 3) for k, v in scores.items()}

    def _derive_variant(self, emotion: str, confidence: float, secondary: Optional[str]) -> tuple[str, str, str]:
        secondary = secondary or ""

        variants_map = {
            "happy": [
                "joyful", "playful", "grateful", "euphoric", "content", "cheerful",
                "optimistic", "inspired", "confident", "lighthearted",
            ],
            "sad": [
                "downcast", "lonely", "disappointed", "drained", "nostalgic", "melancholic",
                "hopeless", "sensitive", "withdrawn", "vulnerable",
            ],
            "angry": [
                "irritated", "frustrated", "upset", "furious", "offended", "impatient",
                "resentful", "agitated", "tense", "reactive",
            ],
            "anxious": [
                "worried", "tense", "restless", "overwhelmed", "on-edge", "uncertain",
                "pressured", "preoccupied", "hypervigilant", "shaky",
            ],
            "neutral": [
                "calm", "focused", "reflective", "steady", "composed", "present",
                "centered", "balanced", "settled", "attentive",
            ],
            "surprised": [
                "amazed", "shocked", "curious", "impressed", "astonished", "intrigued",
                "startled", "engaged", "alert", "energized",
            ],
            "disgusted": [
                "uncomfortable", "repulsed", "doubtful", "avoidant", "averted", "disturbed",
                "averse", "disapproving", "uneasy", "rejecting",
            ],
            "fearful": [
                "insecure", "hesitant", "afraid", "alarmed", "guarded", "intimidated",
                "apprehensive", "panicked", "threatened", "fragile",
            ],
        }

        tips_map = {
            "happy": "Aproveite esse momento e compartilhe algo bom com quem voce gosta.",
            "sad": "Tome agua, respire por 1 minuto e busque apoio de alguem de confianca.",
            "angry": "Pare por 30 segundos e solte o ar lentamente antes de reagir.",
            "anxious": "Observe 5 coisas ao redor e desacelere sua respiracao.",
            "neutral": "Bom momento para manter uma rotina simples e equilibrada.",
            "surprised": "Use essa energia para uma acao positiva e planejada.",
            "disgusted": "Afaste-se do gatilho por alguns minutos e recupere conforto.",
            "fearful": "Priorize seguranca e converse com alguem para reduzir a tensao.",
        }

        zones = {
            "happy": "positive",
            "surprised": "positive",
            "neutral": "balanced",
            "sad": "support-needed",
            "angry": "support-needed",
            "anxious": "support-needed",
            "disgusted": "support-needed",
            "fearful": "support-needed",
        }

        variants = variants_map.get(emotion, ["steady", "calm", "focused", "present"])
        conf_bucket = min(len(variants) - 1, int(confidence * len(variants)))
        idx = (conf_bucket + len(secondary)) % len(variants)
        if confidence >= 0.82:
            idx = min(idx + 1, len(variants) - 1)

        variant = variants[idx]
        zone = zones.get(emotion, "balanced")
        tip = tips_map.get(emotion, "Respire com calma e siga em passos pequenos.")
        return variant, zone, tip

    def _decode_frame(self, frame_data: str) -> Optional[np.ndarray]:
        """Decode base64 frame to numpy array."""
        try:
            if "," in frame_data:
                frame_data = frame_data.split(",")[1]
            img_bytes = base64.b64decode(frame_data)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            frame = np.array(img)

            # Keep a bounded frame size for lower latency and more stable processing time.
            h, w = frame.shape[:2]
            max_dim = max(h, w)
            if max_dim > 640:
                scale = 640.0 / float(max_dim)
                frame = cv2.resize(
                    frame,
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )

            logger.debug("frame decoded | shape=%s", frame.shape)
            return frame
        except Exception as e:
            logger.error(f"Frame decode error: {e}")
            return None

    def _enhance_frame_for_emotion(self, frame: np.ndarray) -> np.ndarray:
        """Improve contrast/sharpness while preserving natural facial tones."""
        try:
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            merged = cv2.merge((l, a, b))
            rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

            # Mild unsharp mask to emphasize expression edges.
            blurred = cv2.GaussianBlur(rgb, (0, 0), sigmaX=1.1, sigmaY=1.1)
            sharpened = cv2.addWeighted(rgb, 1.18, blurred, -0.18, 0)
            return sharpened
        except Exception as e:
            logger.debug("frame enhancement skipped: %s", e)
            return frame

    def _cleanup_old_sessions(self, now_ts: float):
        if len(self._session_state) <= 128:
            return
        stale_keys = [
            key for key, state in self._session_state.items()
            if now_ts - state.updated_at > 20 * 60
        ]
        for key in stale_keys:
            self._session_state.pop(key, None)

    def _apply_temporal_smoothing(self, result: EmotionResult, session_id: Optional[str]) -> EmotionResult:
        """Temporal EMA + hysteresis per session to reduce flicker and false positives."""
        if not session_id or not result.face_detected or not result.all_scores:
            return result

        now_ts = time.time()
        self._cleanup_old_sessions(now_ts)

        state = self._session_state.get(session_id)
        if state is None:
            state = SessionTemporalState(
                scores_ema=dict(result.all_scores),
                last_emotion=result.emotion,
                updated_at=now_ts,
            )
            self._session_state[session_id] = state
            return result

        alpha = 0.66 if result.confidence >= 0.72 else 0.5
        smoothed_scores: dict[str, float] = {}
        keys = set(state.scores_ema.keys()) | set(result.all_scores.keys())
        for key in keys:
            prev = state.scores_ema.get(key, 0.0)
            cur = result.all_scores.get(key, 0.0)
            smoothed_scores[key] = (1.0 - alpha) * prev + alpha * cur
        smoothed_scores = self._normalize_scores(smoothed_scores)

        ranking = sorted(smoothed_scores.items(), key=lambda kv: kv[1], reverse=True)
        candidate = ranking[0][0]
        candidate_score = ranking[0][1]
        secondary = ranking[1][0] if len(ranking) > 1 else None
        secondary_score = ranking[1][1] if len(ranking) > 1 else 0.0
        gap = max(0.0, candidate_score - secondary_score)

        chosen = state.last_emotion
        if candidate != state.last_emotion:
            strong_switch = result.confidence >= 0.84 and gap >= 0.18
            neutral_to_emotion = state.last_emotion == "neutral" and candidate != "neutral"
            negative_to_other = state.last_emotion in {"sad", "anxious", "angry", "disgusted", "fearful"} and candidate != state.last_emotion
            if state.pending_emotion == candidate:
                state.pending_count += 1
            else:
                state.pending_emotion = candidate
                state.pending_count = 1

            if (
                strong_switch
                or (state.pending_count >= 2 and gap >= 0.08)
                or (neutral_to_emotion and state.pending_count >= 1 and gap >= 0.06)
                or (negative_to_other and state.pending_count >= 1 and gap >= 0.05)
            ):
                chosen = candidate
                state.last_emotion = candidate
                state.pending_emotion = None
                state.pending_count = 0
        else:
            state.pending_emotion = None
            state.pending_count = 0

        confidence = self._clip01((result.confidence * 0.45) + (candidate_score * 0.55))
        variant, zone, tip = self._derive_variant(chosen, confidence, secondary)

        state.scores_ema = smoothed_scores
        state.updated_at = now_ts

        result.emotion = chosen
        result.confidence = round(confidence, 3)
        result.all_scores = {k: round(v, 3) for k, v in smoothed_scores.items()}
        result.secondary_emotion = secondary
        result.emotion_variant = variant
        result.emotion_zone = zone
        result.support_tip = tip
        return result

    def _fallback_detect(self, frame: np.ndarray) -> EmotionResult:
        """Heuristic fallback using OpenCV face/eye/smile cues."""
        if not getattr(self, "_opencv_enabled", True):
            return EmotionResult(
                emotion="neutral",
                confidence=0.0,
                all_scores={},
                face_detected=False,
            )

        # Guard OpenCV operations to avoid concurrent access crashes.
        with self._detect_lock:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Prefer MediaPipe for face bounding box (more stable on varied lighting/angles).
            face_bbox = None
            if self._mp_face_detector is not None:
                try:
                    h0, w0 = frame.shape[0], frame.shape[1]
                    target_w = 320
                    scale = target_w / float(w0)
                    resized = cv2.resize(
                        frame,
                        (target_w, max(1, int(h0 * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                    # MediaPipe expects RGB input.
                    mp_result = self._mp_face_detector.process(resized)
                    if mp_result and mp_result.detections:
                        best = max(
                            mp_result.detections,
                            key=lambda d: float(d.score[0]) if d.score else 0.0,
                        )
                        rel = best.location_data.relative_bounding_box
                        x = int(rel.xmin * resized.shape[1])
                        y = int(rel.ymin * resized.shape[0])
                        w = int(rel.width * resized.shape[1])
                        h = int(rel.height * resized.shape[0])
                        # Map back to original frame.
                        inv = 1.0 / scale
                        face_bbox = (int(x * inv), int(y * inv), int(w * inv), int(h * inv))
                except Exception as e:
                    logger.debug("mediapipe face detect failed: %s", e)

            if face_bbox is None:
                if self.face_cascade is None:
                    return EmotionResult(
                        emotion="neutral",
                        confidence=0.0,
                        all_scores={},
                        face_detected=False,
                    )
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.08,
                    minNeighbors=6,
                    minSize=(72, 72),
                )
                logger.debug("fallback detector | haar_faces=%s", len(faces))
                if len(faces) == 0:
                    return EmotionResult(
                        emotion="neutral",
                        confidence=0.0,
                        all_scores={},
                        face_detected=False,
                    )

                # Pick the largest face to stabilize readings when multiple faces appear.
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            else:
                x, y, w, h = face_bbox

            # Clamp bbox.
            x = max(0, min(int(x), frame.shape[1] - 1))
            y = max(0, min(int(y), frame.shape[0] - 1))
            w = max(1, min(int(w), frame.shape[1] - x))
            h = max(1, min(int(h), frame.shape[0] - y))

            # Reject tiny faces (too noisy to classify).
            if w < 72 or h < 72:
                return EmotionResult(
                    emotion="neutral",
                    confidence=0.0,
                    all_scores={},
                    face_detected=False,
                )

            roi = gray[y : y + h, x : x + w]
            roi_eq = cv2.equalizeHist(roi)
            face_area_ratio = self._clip01((w * h) / float(frame.shape[0] * frame.shape[1]))
            brightness_score = self._clip01(float(np.mean(roi_eq)) / 255.0)
            sharpness_score = self._clip01(float(cv2.Laplacian(roi_eq, cv2.CV_64F).var()) / 120.0)

            eyes = self.eye_cascade.detectMultiScale(
                roi_eq,
                scaleFactor=1.1,
                minNeighbors=8,
                minSize=(max(12, w // 9), max(12, h // 9)),
            )
            smiles = self.smile_cascade.detectMultiScale(
                roi_eq,
                scaleFactor=1.7,
                minNeighbors=20,
                minSize=(max(20, w // 5), max(16, h // 10)),
            )

            eye_score = self._clip01(len(eyes) / 2.0)
            smile_score = self._clip01(len(smiles) / 2.0)

            lower = roi_eq[int(h * 0.55) :, :]
            upper = roi_eq[: max(1, int(h * 0.35)), :]
            mouth_edges = cv2.Canny(lower, 40, 120)
            upper_edges = cv2.Canny(upper, 50, 130)

            mouth_open_score = self._clip01(float(mouth_edges.mean()) / 55.0)
            eyebrow_tension = self._clip01(float(upper_edges.mean()) / 55.0)

            # More dynamic fallback: estimates rough emotional tendencies from visual cues.
            scores = {
                "neutral": 0.18 + 0.08 * (1 - smile_score),
                "happy": 0.07 + 0.56 * smile_score + 0.09 * eye_score,
                "surprised": 0.08 + 0.48 * mouth_open_score * eye_score + 0.08 * face_area_ratio,
                "sad": 0.07 + 0.22 * (1 - eye_score) + 0.16 * (1 - smile_score),
                "anxious": 0.08 + 0.26 * mouth_open_score + 0.20 * (1 - eye_score) + 0.10 * eyebrow_tension,
                "angry": 0.08 + 0.28 * eyebrow_tension + 0.16 * (1 - smile_score),
                "disgusted": 0.06 + 0.14 * mouth_open_score * (1 - eye_score) + 0.10 * eyebrow_tension,
            }

            if smile_score > 0.62 and eye_score > 0.4:
                scores["happy"] += 0.14
                scores["neutral"] -= 0.06

            # Reduce happy over-detection when smile cues are weak or ambiguous.
            if smile_score < 0.35:
                scores["happy"] *= 0.72
                scores["neutral"] += 0.04

            if smile_score < 0.22 and mouth_open_score < 0.28:
                scores["happy"] *= 0.65
                scores["neutral"] += 0.06

            if mouth_open_score > 0.45 and eye_score > 0.6:
                scores["surprised"] += 0.16
                scores["neutral"] -= 0.06

            if eyebrow_tension > 0.5 and smile_score < 0.2:
                scores["angry"] += 0.12
                scores["anxious"] += 0.08

            if eye_score < 0.35 and smile_score < 0.2:
                scores["sad"] += 0.12
                scores["neutral"] -= 0.06

            # Reduce false "sad" under normal camera posture with decent facial openness.
            if eye_score > 0.52 and mouth_open_score > 0.2:
                scores["sad"] *= 0.78
                scores["neutral"] += 0.05

            # Damp noisy emotion jumps when the frame quality is poor.
            low_light = brightness_score < 0.22 or brightness_score > 0.92
            blurry = sharpness_score < 0.22
            if low_light or blurry:
                quality_penalty = 0.75 if (low_light and blurry) else 0.86
                for emotion_key in ["happy", "surprised", "sad", "anxious", "angry", "disgusted"]:
                    scores[emotion_key] *= quality_penalty
                scores["neutral"] += 0.14 if (low_light and blurry) else 0.08

            for key in list(scores.keys()):
                scores[key] = max(0.01, scores[key])

            normalized = self._normalize_scores(scores)
            if self._fallback_prev_scores:
                quality_score = self._clip01((brightness_score * 0.45) + (sharpness_score * 0.55))
                prev_weight = 0.48 if quality_score < 0.35 else 0.34
                new_weight = 1.0 - prev_weight
                smoothed = {}
                for key, value in normalized.items():
                    prev = self._fallback_prev_scores.get(key, value)
                    smoothed[key] = round(prev * prev_weight + value * new_weight, 3)
                normalized = self._normalize_scores(smoothed)
            self._fallback_prev_scores = normalized

            ranking = sorted(normalized.items(), key=lambda kv: kv[1], reverse=True)
            primary, primary_score = ranking[0]
            secondary = ranking[1][0] if len(ranking) > 1 else None
            secondary_score = ranking[1][1] if len(ranking) > 1 else 0.0

            # Additional guard for false "happy" predictions from frontal camera noise.
            if primary == "happy":
                happy_gap = max(0.0, primary_score - secondary_score)
                weak_happy_signal = smile_score < 0.5 or happy_gap < 0.09
                if weak_happy_signal:
                    normalized["happy"] *= 0.82
                    normalized["neutral"] = normalized.get("neutral", 0.0) + 0.05
                    normalized = self._normalize_scores(normalized)
                    ranking = sorted(normalized.items(), key=lambda kv: kv[1], reverse=True)
                    primary, primary_score = ranking[0]
                    secondary = ranking[1][0] if len(ranking) > 1 else None
                    secondary_score = ranking[1][1] if len(ranking) > 1 else 0.0

            # Additional guard for false "sad" predictions from low-expression frames.
            if primary == "sad":
                sad_gap = max(0.0, primary_score - secondary_score)
                weak_sad_signal = eye_score > 0.45 or sad_gap < 0.07
                if weak_sad_signal:
                    normalized["sad"] *= 0.78
                    normalized["neutral"] = normalized.get("neutral", 0.0) + 0.06
                    normalized = self._normalize_scores(normalized)
                    ranking = sorted(normalized.items(), key=lambda kv: kv[1], reverse=True)
                    primary, primary_score = ranking[0]
                    secondary = ranking[1][0] if len(ranking) > 1 else None
                    secondary_score = ranking[1][1] if len(ranking) > 1 else 0.0

            separation = max(0.0, primary_score - secondary_score)
            ambiguous = separation < 0.055 or primary_score < 0.25
            if ambiguous:
                normalized["neutral"] = normalized.get("neutral", 0.0) + 0.06
                normalized = self._normalize_scores(normalized)
                ranking = sorted(normalized.items(), key=lambda kv: kv[1], reverse=True)
                primary, primary_score = ranking[0]
                secondary = ranking[1][0] if len(ranking) > 1 else None
                secondary_score = ranking[1][1] if len(ranking) > 1 else 0.0
                separation = max(0.0, primary_score - secondary_score)

            face_quality = self._clip01(0.35 + face_area_ratio * 1.2)
            confidence_raw = 0.22 + primary_score * 0.45 + separation * 0.28 + face_quality * 0.12
            if ambiguous:
                confidence_raw *= 0.84
            confidence = round(self._clip01(confidence_raw), 3)
            variant, zone, tip = self._derive_variant(primary, confidence, secondary)

            logger.debug(
                "fallback features | eye=%.3f smile=%.3f mouth=%.3f brow=%.3f bright=%.3f sharp=%.3f",
                eye_score,
                smile_score,
                mouth_open_score,
                eyebrow_tension,
                brightness_score,
                sharpness_score,
            )

            return EmotionResult(
                emotion=primary,
                confidence=confidence,
                all_scores=normalized,
                emotion_variant=variant,
                emotion_zone=zone,
                support_tip=tip,
                secondary_emotion=secondary,
                face_detected=True,
                music_suggestions=MUSIC_SUGGESTIONS.get(primary, []),
            )

    def detect_from_frame(self, frame: np.ndarray) -> EmotionResult:
        """Run emotion detection on a numpy frame."""
        start = time.time()
        enhanced = self._enhance_frame_for_emotion(frame)

        if self._deepface_loaded:
            try:
                results = self.DeepFace.analyze(
                    enhanced,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True,
                )
                if isinstance(results, list):
                    results = results[0]

                raw_emotions = results.get("emotion", {})
                dominant = results.get("dominant_emotion", "neutral")

                # Normalize scores to 0-1
                total = sum(raw_emotions.values()) or 1
                scores = {EMOTION_MAP.get(k, k): v / total for k, v in raw_emotions.items()}

                # Sort and pick top 2
                sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                primary = sorted_emotions[0][0]
                primary_conf = sorted_emotions[0][1]
                secondary = sorted_emotions[1][0] if len(sorted_emotions) > 1 else None

                # DeepFace can over-index on happy in some front-camera conditions.
                if primary == "happy":
                    secondary_conf = sorted_emotions[1][1] if len(sorted_emotions) > 1 else 0.0
                    margin = max(0.0, primary_conf - secondary_conf)
                    if primary_conf < 0.43 or margin < 0.07:
                        scores["happy"] = max(0.0, scores.get("happy", 0.0) * 0.78)
                        scores["neutral"] = scores.get("neutral", 0.0) + 0.12
                        scores = self._normalize_scores(scores)
                        sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                        primary = sorted_emotions[0][0]
                        primary_conf = sorted_emotions[0][1]
                        secondary = sorted_emotions[1][0] if len(sorted_emotions) > 1 else None

                result = EmotionResult(
                    emotion=primary,
                    confidence=round(primary_conf, 3),
                    all_scores={k: round(v, 3) for k, v in scores.items()},
                    secondary_emotion=secondary,
                    face_detected=True,
                    processing_time_ms=round((time.time() - start) * 1000, 1),
                    music_suggestions=MUSIC_SUGGESTIONS.get(primary, []),
                )
                variant, zone, tip = self._derive_variant(primary, result.confidence, secondary)
                result.emotion_variant = variant
                result.emotion_zone = zone
                result.support_tip = tip
                return result

            except Exception as e:
                logger.error(f"DeepFace analysis error: {e}")

        logger.debug("using fallback emotion detector")
        result = self._fallback_detect(enhanced)
        result.processing_time_ms = round((time.time() - start) * 1000, 1)
        return result

    async def analyze_frame_base64(
        self,
        frame_b64: str,
        session_id: Optional[str] = None,
        pet_name: str = "MoodPet",
    ) -> EmotionResult:
        """Async entry point: decode frame and run detection."""
        logger.debug("analyze_frame_base64 start | frame_len=%s", len(frame_b64))
        frame = self._decode_frame(frame_b64)
        if frame is None:
            logger.warning("analyze_frame_base64 failed: invalid frame payload")
            return EmotionResult(
                emotion="neutral",
                confidence=0.0,
                all_scores={},
                face_detected=False,
                processing_time_ms=0,
            )

        # Run in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self._detect_executor, self.detect_from_frame, frame)
        result = self._apply_temporal_smoothing(result, session_id)

        # Attach message
        import random
        messages = EMOTION_MESSAGES.get(result.emotion, EMOTION_MESSAGES["neutral"])
        result.message = random.choice(messages).replace("{pet_name}", pet_name)

        logger.info(
            "emotion analyzed | emotion=%s confidence=%.3f face_detected=%s processing_ms=%.1f mode=%s",
            result.emotion,
            result.confidence,
            result.face_detected,
            result.processing_time_ms,
            "deepface" if self._deepface_loaded else "fallback",
        )

        return result


# Singleton instance
emotion_service = EmotionDetectionService()

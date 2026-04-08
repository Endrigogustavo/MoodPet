"""
Emotion Detection Service — High-Precision Engine
Uses DeepFace + MediaPipe Face Mesh (468 landmarks) + OpenCV for real-time
facial expression analysis with micro-expression detection.

Pipeline:
  1. Decode & enhance frame (CLAHE + unsharp mask)
  2. MediaPipe Face Mesh → 468 3D landmarks → Action Unit estimation
  3. DeepFace CNN emotion classification (when available)
  4. OpenCV Haar heuristic fallback
  5. Ensemble fusion: weighted vote across all available models
  6. Temporal smoothing (EMA + hysteresis per session)
  7. Micro-expression detection (rapid fleeting expressions <500ms)
"""

import asyncio
import base64
import collections
import concurrent.futures
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

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

# MediaPipe Face Mesh landmark indices for Action Unit estimation
# Lip corners
_LIP_LEFT = 61
_LIP_RIGHT = 291
_LIP_TOP = 13
_LIP_BOTTOM = 14
_UPPER_LIP_TOP = 0
_LOWER_LIP_BOTTOM = 17

# Eyes
_LEFT_EYE_TOP = 159
_LEFT_EYE_BOTTOM = 145
_LEFT_EYE_INNER = 133
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_TOP = 386
_RIGHT_EYE_BOTTOM = 374
_RIGHT_EYE_INNER = 362
_RIGHT_EYE_OUTER = 263

# Eyebrows
_LEFT_BROW_INNER = 107
_LEFT_BROW_OUTER = 70
_LEFT_BROW_MID = 105
_RIGHT_BROW_INNER = 336
_RIGHT_BROW_OUTER = 300
_RIGHT_BROW_MID = 334

# Nose
_NOSE_TIP = 1
_NOSE_BRIDGE = 6

# Jaw
_JAW_LEFT = 234
_JAW_RIGHT = 454
_JAW_BOTTOM = 152

# Cheeks
_LEFT_CHEEK = 123
_RIGHT_CHEEK = 352

# Forehead reference
_FOREHEAD_MID = 10

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
class ActionUnits:
    """Facial Action Unit estimates from Face Mesh landmarks (0.0–1.0)."""
    au1_inner_brow_raise: float = 0.0   # Inner brow raise
    au2_outer_brow_raise: float = 0.0   # Outer brow raise
    au4_brow_lowerer: float = 0.0       # Brow furrowing
    au6_cheek_raise: float = 0.0        # Cheek raise (Duchenne)
    au9_nose_wrinkle: float = 0.0       # Nose wrinkle
    au12_lip_corner_pull: float = 0.0   # Smile
    au15_lip_corner_depress: float = 0.0 # Frown
    au20_lip_stretch: float = 0.0       # Lip stretch
    au25_lips_part: float = 0.0         # Lips part (jaw open)
    au26_jaw_drop: float = 0.0          # Jaw drop
    au43_eyes_closed: float = 0.0       # Eye closure
    head_tilt_x: float = 0.0            # Pitch
    head_tilt_y: float = 0.0            # Yaw
    face_quality: float = 0.0           # Overall face visibility quality
    # Extended blendshapes for precision
    eye_wide_left: float = 0.0          # Eye widening (surprise/fear)
    eye_wide_right: float = 0.0
    eye_squint_left: float = 0.0        # Eye squint (happy/disgust)
    eye_squint_right: float = 0.0
    mouth_pucker: float = 0.0           # Lip pucker (disgust/contempt)
    mouth_press_left: float = 0.0       # Lip press (anger/contempt)
    mouth_press_right: float = 0.0
    cheek_puff: float = 0.0             # Cheek puff
    jaw_left: float = 0.0               # Jaw lateral
    jaw_right: float = 0.0
    mouth_dimple_left: float = 0.0      # Dimple (smirk/contempt)
    mouth_dimple_right: float = 0.0
    mouth_roll_lower: float = 0.0       # Lip roll (anxiety, thinking)
    mouth_roll_upper: float = 0.0
    mouth_shrug_lower: float = 0.0      # Mouth shrug (doubt)
    mouth_shrug_upper: float = 0.0
    # Extra blendshapes for improved precision
    mouth_lower_down: float = 0.0       # Lower lip pulled down (disgust/sad)
    mouth_upper_up: float = 0.0         # Upper lip raise (disgust snarl)
    mouth_left: float = 0.0             # Mouth shifted left (asymmetric disgust)
    mouth_right: float = 0.0            # Mouth shifted right
    brow_inner_up: float = 0.0          # Raw browInnerUp (redundant w/ au1 but useful)


@dataclass
class MicroExpression:
    """A fleeting expression detected in <500ms window."""
    emotion: str
    intensity: float
    duration_ms: float
    timestamp: float


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
    action_units: Optional[dict] = None
    micro_expressions: list = field(default_factory=list)
    detection_models_used: list = field(default_factory=list)
    face_mesh_landmarks_count: int = 0
    # New: compound emotions, intensity, quality, streak
    compound_emotion: Optional[str] = None
    emotion_intensity: str = "mild"           # calm/mild/moderate/intense/extreme
    face_quality_metrics: Optional[dict] = None  # lighting, sharpness, angle, distance
    emotion_streak_seconds: float = 0.0
    # rPPG heart rate estimation
    heart_rate_bpm: Optional[float] = None         # Estimated BPM (None = not enough data yet)
    heart_rate_confidence: float = 0.0             # 0.0–1.0 signal reliability
    heart_rate_status: str = "collecting"          # collecting / ready / unstable

    def __post_init__(self):
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
                    continue
            self.all_scores = cleaned


@dataclass
class SessionTemporalState:
    scores_ema: dict[str, float] = field(default_factory=dict)
    last_emotion: str = "neutral"
    pending_emotion: Optional[str] = None
    pending_count: int = 0
    updated_at: float = field(default_factory=time.time)
    # Micro-expression tracking
    recent_au_snapshots: list = field(default_factory=list)  # last N AU readings
    micro_expressions_detected: list = field(default_factory=list)
    # Enhanced history for pattern tracking
    emotion_history: list = field(default_factory=list)  # last 30 readings
    # Streak tracking
    current_streak_emotion: str = "neutral"
    current_streak_start: float = field(default_factory=time.time)
    emotion_transitions: list = field(default_factory=list)  # last 20 transitions
    # rPPG heart rate tracking
    rppg_timestamps: list = field(default_factory=list)    # float timestamps
    rppg_green_means: list = field(default_factory=list)   # green channel averages
    rppg_last_bpm: Optional[float] = None
    rppg_last_confidence: float = 0.0
    # Adaptive neutral calibration: learn resting-face AU baseline from first N frames
    calibration_au_buffer: list = field(default_factory=list)  # list of AU dicts
    calibration_done: bool = False
    neutral_baseline: Optional[dict] = None  # median AU values for resting face
    # Multi-frame voting window for robust decisions
    frame_vote_buffer: list = field(default_factory=list)  # last N emotion scores


class EmotionDetectionService:
    """High-precision emotion detection with multi-model ensemble + Face Mesh AU analysis."""

    _instance = None
    _deepface_loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._fallback_prev_scores_by_session: dict[str, dict[str, float]] = {}
            self._session_state: dict[str, SessionTemporalState] = {}
            self._detect_lock = threading.Lock()
            self._face_mesh = None
            self._face_mesh_loaded = False
            self._load_models()

    def _load_models(self):
        # ── DeepFace ──────────────────────────────────────────────────────────
        if sys.version_info >= (3, 13):
            logger.warning("⚠️  DeepFace disabled on Python 3.13, using fallback detector")
            self._deepface_loaded = False
        else:
            try:
                from deepface import DeepFace
                self.DeepFace = DeepFace
                dummy = np.zeros((48, 48, 3), dtype=np.uint8)
                DeepFace.analyze(dummy, actions=["emotion"], enforce_detection=False, silent=True)
                self._deepface_loaded = True
                logger.info("✅ DeepFace model loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️  DeepFace not available ({e}), using fallback detector")
                self._deepface_loaded = False

        # ── OpenCV Haar Cascades ──────────────────────────────────────────────
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

        # ── MediaPipe Face Landmarker (Tasks API — 468 landmarks + 52 blendshapes)
        self._mp_face_detector = None
        self._mp_module = None
        try:
            import mediapipe as mp  # type: ignore
            self._mp_module = mp

            # Use the Tasks API (works on Python 3.13+)
            if hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
                import os as _os
                import urllib.request as _urlreq

                model_dir = _os.path.expanduser("~/.mediapipe")
                _os.makedirs(model_dir, exist_ok=True)

                # ── Face Landmarker model (478 landmarks + 52 blendshapes) ──
                landmarker_path = _os.path.join(model_dir, "face_landmarker.task")
                if not _os.path.exists(landmarker_path):
                    logger.info("Downloading face_landmarker model …")
                    _urlreq.urlretrieve(
                        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
                        landmarker_path,
                    )

                BaseOptions = mp.tasks.BaseOptions
                FaceLandmarker = mp.tasks.vision.FaceLandmarker
                FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
                VisionRunningMode = mp.tasks.vision.RunningMode

                options = FaceLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=landmarker_path,
                        delegate=BaseOptions.Delegate.CPU,
                    ),
                    running_mode=VisionRunningMode.IMAGE,
                    num_faces=1,
                    output_face_blendshapes=True,
                )
                self._face_mesh = FaceLandmarker.create_from_options(options)
                self._face_mesh_loaded = True
                logger.info("✅ MediaPipe FaceLandmarker (478 landmarks + 52 blendshapes) loaded [CPU]")

                # ── Face Detector for bounding-box fallback ──
                det_path = _os.path.join(model_dir, "blaze_face_short_range.tflite")
                if not _os.path.exists(det_path):
                    logger.info("Downloading blaze_face model …")
                    _urlreq.urlretrieve(
                        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
                        det_path,
                    )
                FaceDetector = mp.tasks.vision.FaceDetector
                FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
                det_options = FaceDetectorOptions(
                    base_options=BaseOptions(
                        model_asset_path=det_path,
                        delegate=BaseOptions.Delegate.CPU,
                    ),
                    min_detection_confidence=0.5,
                )
                self._mp_face_detector = FaceDetector.create_from_options(det_options)
                logger.info("✅ MediaPipe FaceDetector ready")

                # ── Disable OpenCV Haar cascades ──
                # On Python 3.13 OpenCV's detectMultiScale segfaults when
                # mediapipe native code is also loaded in the same process.
                # FaceLandmarker (478 landmarks + 52 blendshapes) is strictly
                # superior to Haar cascades anyway.
                if self._opencv_enabled:
                    logger.info(
                        "Disabling OpenCV Haar cascades (FaceLandmarker loaded; "
                        "avoids native-code conflict on Python 3.13)"
                    )
                    self._opencv_enabled = False
                    self.face_cascade = None
                    self.eye_cascade = None
                    self.smile_cascade = None
            else:
                logger.warning(
                    "⚠️  MediaPipe Tasks API unavailable; FaceLandmarker disabled"
                )
        except Exception as e:
            logger.warning(f"⚠️  MediaPipe unavailable ({e}), FaceLandmarker disabled")

    @staticmethod
    def _clip01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _normalize_scores(self, scores: dict[str, float]) -> dict[str, float]:
        total = sum(max(0.0, v) for v in scores.values()) or 1.0
        return {k: round(max(0.0, v) / total, 3) for k, v in scores.items()}

    # ── Adaptive Neutral Calibration ──────────────────────────────────────────
    _CALIBRATION_FRAMES = 5  # collect first N frames to learn resting face

    def _update_calibration(self, au: 'ActionUnits', session_id: str) -> None:
        """Accumulate AU data from early frames to establish resting-face baseline."""
        state = self._session_state.get(session_id)
        if state is None or state.calibration_done:
            return

        au_dict = {
            "a12": au.au12_lip_corner_pull, "a6": au.au6_cheek_raise,
            "a4": au.au4_brow_lowerer, "a1": au.au1_inner_brow_raise,
            "a15": au.au15_lip_corner_depress, "a9": au.au9_nose_wrinkle,
            "a20": au.au20_lip_stretch, "a25": au.au25_lips_part,
            "a43": au.au43_eyes_closed,
            # Extended AUs
            "eye_wide": (au.eye_wide_left + au.eye_wide_right) / 2.0,
            "eye_squint": (au.eye_squint_left + au.eye_squint_right) / 2.0,
            "mouth_pucker": au.mouth_pucker,
            "mouth_press": (au.mouth_press_left + au.mouth_press_right) / 2.0,
            "cheek_puff": au.cheek_puff,
            "mouth_dimple": (au.mouth_dimple_left + au.mouth_dimple_right) / 2.0,
            "mouth_roll": (au.mouth_roll_lower + au.mouth_roll_upper) / 2.0,
            "mouth_shrug": (au.mouth_shrug_lower + au.mouth_shrug_upper) / 2.0,
            "mouth_lower_down": au.mouth_lower_down,
            "mouth_upper_up": au.mouth_upper_up,
            "mouth_lateral": (au.mouth_left + au.mouth_right) / 2.0,
        }
        state.calibration_au_buffer.append(au_dict)

        if len(state.calibration_au_buffer) >= self._CALIBRATION_FRAMES:
            # Compute median per AU as resting baseline
            baseline = {}
            for key in au_dict:
                vals = [snap[key] for snap in state.calibration_au_buffer]
                vals.sort()
                mid = len(vals) // 2
                baseline[key] = vals[mid]
            state.neutral_baseline = baseline
            state.calibration_done = True
            logger.info("calibration done | session=%s baseline=%s", session_id,
                        {k: round(v, 3) for k, v in baseline.items()})

    _BASELINE_FRACTION = 0.60  # subtract only 60% of resting baseline

    def _subtract_baseline(self, au: 'ActionUnits', session_id: str) -> 'ActionUnits':
        """Subtract a fraction of the resting-face baseline from current AUs."""
        state = self._session_state.get(session_id)
        if state is None or state.neutral_baseline is None:
            return au
        b = state.neutral_baseline
        f = self._BASELINE_FRACTION
        # Original AUs
        au.au12_lip_corner_pull = max(0.0, au.au12_lip_corner_pull - b.get("a12", 0.0) * f)
        au.au6_cheek_raise = max(0.0, au.au6_cheek_raise - b.get("a6", 0.0) * f)
        au.au4_brow_lowerer = max(0.0, au.au4_brow_lowerer - b.get("a4", 0.0) * f)
        au.au1_inner_brow_raise = max(0.0, au.au1_inner_brow_raise - b.get("a1", 0.0) * f)
        au.au15_lip_corner_depress = max(0.0, au.au15_lip_corner_depress - b.get("a15", 0.0) * f)
        au.au9_nose_wrinkle = max(0.0, au.au9_nose_wrinkle - b.get("a9", 0.0) * f)
        au.au20_lip_stretch = max(0.0, au.au20_lip_stretch - b.get("a20", 0.0) * f)
        au.au25_lips_part = max(0.0, au.au25_lips_part - b.get("a25", 0.0) * f)
        au.au43_eyes_closed = max(0.0, au.au43_eyes_closed - b.get("a43", 0.0) * f)
        # Extended AUs
        ew = b.get("eye_wide", 0.0) * f
        au.eye_wide_left = max(0.0, au.eye_wide_left - ew)
        au.eye_wide_right = max(0.0, au.eye_wide_right - ew)
        es = b.get("eye_squint", 0.0) * f
        au.eye_squint_left = max(0.0, au.eye_squint_left - es)
        au.eye_squint_right = max(0.0, au.eye_squint_right - es)
        au.mouth_pucker = max(0.0, au.mouth_pucker - b.get("mouth_pucker", 0.0) * f)
        mp = b.get("mouth_press", 0.0) * f
        au.mouth_press_left = max(0.0, au.mouth_press_left - mp)
        au.mouth_press_right = max(0.0, au.mouth_press_right - mp)
        au.cheek_puff = max(0.0, au.cheek_puff - b.get("cheek_puff", 0.0) * f)
        md = b.get("mouth_dimple", 0.0) * f
        au.mouth_dimple_left = max(0.0, au.mouth_dimple_left - md)
        au.mouth_dimple_right = max(0.0, au.mouth_dimple_right - md)
        mr = b.get("mouth_roll", 0.0) * f
        au.mouth_roll_lower = max(0.0, au.mouth_roll_lower - mr)
        au.mouth_roll_upper = max(0.0, au.mouth_roll_upper - mr)
        ms = b.get("mouth_shrug", 0.0) * f
        au.mouth_shrug_lower = max(0.0, au.mouth_shrug_lower - ms)
        au.mouth_shrug_upper = max(0.0, au.mouth_shrug_upper - ms)
        # New extra blendshapes
        au.mouth_lower_down = max(0.0, au.mouth_lower_down - b.get("mouth_lower_down", 0.0) * f)
        au.mouth_upper_up = max(0.0, au.mouth_upper_up - b.get("mouth_upper_up", 0.0) * f)
        ml = b.get("mouth_lateral", 0.0) * f
        au.mouth_left = max(0.0, au.mouth_left - ml)
        au.mouth_right = max(0.0, au.mouth_right - ml)
        return au

    # ── Multi-Frame Voting ────────────────────────────────────────────────────
    _VOTE_WINDOW = 3  # average last N frames for robust decision

    def _multi_frame_vote(self, scores: dict[str, float], session_id: str) -> dict[str, float]:
        """Average emotion scores over the last N frames for noise reduction."""
        state = self._session_state.get(session_id)
        if state is None:
            return scores
        state.frame_vote_buffer.append(dict(scores))
        if len(state.frame_vote_buffer) > self._VOTE_WINDOW:
            state.frame_vote_buffer = state.frame_vote_buffer[-self._VOTE_WINDOW:]
        if len(state.frame_vote_buffer) < 2:
            return scores
        # Weighted average: more recent frames have higher weight
        all_keys = set()
        for s in state.frame_vote_buffer:
            all_keys.update(s.keys())
        n = len(state.frame_vote_buffer)
        weights = [(i + 1) for i in range(n)]  # 1, 2, 3, ...
        tw = sum(weights)
        averaged = {}
        for key in all_keys:
            averaged[key] = sum(w * s.get(key, 0.0) for w, s in zip(weights, state.frame_vote_buffer)) / tw
        return self._normalize_scores(averaged)

    # ── AU Asymmetry Detection ────────────────────────────────────────────────

    @staticmethod
    def _compute_asymmetry(blendshape_map: dict) -> float:
        """Compute facial asymmetry score (0=symmetric, 1=very asymmetric).
        Genuine emotions tend to be more symmetric than faked ones."""
        pairs = [
            ("browDownLeft", "browDownRight"),
            ("browOuterUpLeft", "browOuterUpRight"),
            ("cheekSquintLeft", "cheekSquintRight"),
            ("mouthSmileLeft", "mouthSmileRight"),
            ("mouthFrownLeft", "mouthFrownRight"),
            ("eyeSquintLeft", "eyeSquintRight"),
            ("eyeBlinkLeft", "eyeBlinkRight"),
        ]
        diffs = []
        for left_key, right_key in pairs:
            l = blendshape_map.get(left_key, 0.0)
            r = blendshape_map.get(right_key, 0.0)
            if l + r > 0.02:
                diffs.append(abs(l - r) / max(l + r, 0.01))
        if not diffs:
            return 0.0
        return sum(diffs) / len(diffs)

    # ── Face Mesh Landmark Analysis ───────────────────────────────────────────

    @staticmethod
    def _landmark_distance(lm, idx_a: int, idx_b: int) -> float:
        """Euclidean distance between two landmarks."""
        a = lm[idx_a]
        b = lm[idx_b]
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

    @staticmethod
    def _landmark_y_diff(lm, idx_a: int, idx_b: int) -> float:
        """Signed Y difference (positive = a is below b)."""
        return lm[idx_a].y - lm[idx_b].y

    def _compute_action_units(self, landmarks) -> ActionUnits:
        """Estimate facial Action Units from 468 Face Mesh landmarks."""
        lm = landmarks
        au = ActionUnits()

        # Reference distances for normalization
        face_height = self._landmark_distance(lm, _FOREHEAD_MID, _JAW_BOTTOM)
        face_width = self._landmark_distance(lm, _JAW_LEFT, _JAW_RIGHT)
        if face_height < 0.001 or face_width < 0.001:
            return au

        # ── AU1: Inner Brow Raise ─────────────────────────────────────────
        left_inner_brow_lift = self._landmark_y_diff(lm, _NOSE_BRIDGE, _LEFT_BROW_INNER)
        right_inner_brow_lift = self._landmark_y_diff(lm, _NOSE_BRIDGE, _RIGHT_BROW_INNER)
        avg_inner_brow = (left_inner_brow_lift + right_inner_brow_lift) / 2.0
        au.au1_inner_brow_raise = self._clip01(avg_inner_brow / face_height * 8.0)

        # ── AU2: Outer Brow Raise ─────────────────────────────────────────
        left_outer_brow_lift = self._landmark_y_diff(lm, _LEFT_EYE_OUTER, _LEFT_BROW_OUTER)
        right_outer_brow_lift = self._landmark_y_diff(lm, _RIGHT_EYE_OUTER, _RIGHT_BROW_OUTER)
        avg_outer_brow = (left_outer_brow_lift + right_outer_brow_lift) / 2.0
        au.au2_outer_brow_raise = self._clip01(avg_outer_brow / face_height * 10.0)

        # ── AU4: Brow Lowerer ─────────────────────────────────────────────
        brow_mid_dist = (
            self._landmark_distance(lm, _LEFT_BROW_MID, _LEFT_EYE_TOP) +
            self._landmark_distance(lm, _RIGHT_BROW_MID, _RIGHT_EYE_TOP)
        ) / 2.0
        brow_closeness = 1.0 - (brow_mid_dist / face_height * 6.0)
        au.au4_brow_lowerer = self._clip01(brow_closeness)

        # ── AU6: Cheek Raise (Duchenne marker) ───────────────────────────
        left_cheek_rise = self._landmark_y_diff(lm, _LEFT_EYE_BOTTOM, _LEFT_CHEEK)
        right_cheek_rise = self._landmark_y_diff(lm, _RIGHT_EYE_BOTTOM, _RIGHT_CHEEK)
        cheek_squeeze = (
            self._landmark_distance(lm, _LEFT_EYE_BOTTOM, _LEFT_CHEEK) +
            self._landmark_distance(lm, _RIGHT_EYE_BOTTOM, _RIGHT_CHEEK)
        ) / 2.0
        au.au6_cheek_raise = self._clip01(1.0 - cheek_squeeze / face_height * 5.0)

        # ── AU9: Nose Wrinkle ─────────────────────────────────────────────
        nose_bridge_dist = self._landmark_distance(lm, _NOSE_BRIDGE, _NOSE_TIP)
        nose_compress = 1.0 - (nose_bridge_dist / face_height * 4.0)
        au.au9_nose_wrinkle = self._clip01(nose_compress)

        # ── AU12: Lip Corner Pull (Smile) ─────────────────────────────────
        lip_width = self._landmark_distance(lm, _LIP_LEFT, _LIP_RIGHT)
        lip_corner_height = -(
            (lm[_LIP_LEFT].y + lm[_LIP_RIGHT].y) / 2.0 -
            (lm[_LIP_TOP].y + lm[_LIP_BOTTOM].y) / 2.0
        )
        smile_ratio = lip_width / face_width
        au.au12_lip_corner_pull = self._clip01(
            (smile_ratio - 0.25) * 3.0 + lip_corner_height / face_height * 8.0
        )

        # ── AU15: Lip Corner Depressor (Frown) ───────────────────────────
        lip_center_y = (lm[_LIP_TOP].y + lm[_LIP_BOTTOM].y) / 2.0
        lip_corners_y = (lm[_LIP_LEFT].y + lm[_LIP_RIGHT].y) / 2.0
        frown_signal = (lip_corners_y - lip_center_y) / face_height
        au.au15_lip_corner_depress = self._clip01(frown_signal * 12.0)

        # ── AU20: Lip Stretch ─────────────────────────────────────────────
        au.au20_lip_stretch = self._clip01((lip_width / face_width - 0.3) * 4.0)

        # ── AU25: Lips Part ───────────────────────────────────────────────
        lip_opening = self._landmark_distance(lm, _LIP_TOP, _LIP_BOTTOM)
        au.au25_lips_part = self._clip01(lip_opening / face_height * 8.0)

        # ── AU26: Jaw Drop ────────────────────────────────────────────────
        jaw_open = self._landmark_distance(lm, _UPPER_LIP_TOP, _LOWER_LIP_BOTTOM)
        au.au26_jaw_drop = self._clip01(jaw_open / face_height * 4.0)

        # ── AU43: Eyes Closed ─────────────────────────────────────────────
        left_eye_open = self._landmark_distance(lm, _LEFT_EYE_TOP, _LEFT_EYE_BOTTOM)
        right_eye_open = self._landmark_distance(lm, _RIGHT_EYE_TOP, _RIGHT_EYE_BOTTOM)
        left_eye_width = self._landmark_distance(lm, _LEFT_EYE_INNER, _LEFT_EYE_OUTER)
        right_eye_width = self._landmark_distance(lm, _RIGHT_EYE_INNER, _RIGHT_EYE_OUTER)
        left_ear = left_eye_open / max(left_eye_width, 0.001)
        right_ear = right_eye_open / max(right_eye_width, 0.001)
        avg_ear = (left_ear + right_ear) / 2.0
        au.au43_eyes_closed = self._clip01(1.0 - avg_ear * 3.5)

        # ── Head Pose ─────────────────────────────────────────────────────
        au.head_tilt_x = (lm[_NOSE_TIP].y - lm[_FOREHEAD_MID].y) / face_height
        au.head_tilt_y = (lm[_NOSE_TIP].x - 0.5) * 2.0

        # ── Face Quality ──────────────────────────────────────────────────
        au.face_quality = self._clip01(1.0 - abs(au.head_tilt_y) * 1.5)

        return au

    def _au_to_emotion_scores(self, au: ActionUnits) -> dict[str, float]:
        """Convert Action Units to emotion probability scores using extended FACS rules.

        Uses 27 blendshapes (original 11 + 16 extended) for much finer emotion
        discrimination. Amplification compensates for naturally-small MediaPipe
        blendshapes. Contradictory suppression prevents false positives.
        """

        def amp(v: float, gate: float = 0.02) -> float:
            if v < gate:
                return 0.0
            x = (v - gate) / (1.0 - gate)
            return self._clip01(x ** 0.45 * 1.30)

        # Original AUs with lowered noise gates (baseline already subtracted)
        a1  = amp(au.au1_inner_brow_raise, 0.02)
        a2  = amp(au.au2_outer_brow_raise, 0.02)
        a4  = amp(au.au4_brow_lowerer, 0.02)
        a6  = amp(au.au6_cheek_raise, 0.02)
        a9  = amp(au.au9_nose_wrinkle, 0.03)
        a12 = amp(au.au12_lip_corner_pull, 0.03)
        a15 = amp(au.au15_lip_corner_depress, 0.02)
        a20 = amp(au.au20_lip_stretch, 0.03)
        a25 = amp(au.au25_lips_part, 0.03)
        a26 = amp(au.au26_jaw_drop, 0.03)
        a43 = amp(au.au43_eyes_closed, 0.03)

        # Extended AUs for fine-grained discrimination
        eye_wide  = amp((au.eye_wide_left + au.eye_wide_right) / 2.0, 0.02)
        eye_sqnt  = amp((au.eye_squint_left + au.eye_squint_right) / 2.0, 0.03)
        m_pucker  = amp(au.mouth_pucker, 0.03)
        m_press   = amp((au.mouth_press_left + au.mouth_press_right) / 2.0, 0.03)
        cheek_pf  = amp(au.cheek_puff, 0.04)
        jaw_lat   = amp((au.jaw_left + au.jaw_right) / 2.0, 0.03)
        m_dimple  = amp((au.mouth_dimple_left + au.mouth_dimple_right) / 2.0, 0.03)
        m_roll    = amp((au.mouth_roll_lower + au.mouth_roll_upper) / 2.0, 0.03)
        m_shrug   = amp((au.mouth_shrug_lower + au.mouth_shrug_upper) / 2.0, 0.03)
        # New extra channels
        m_lower   = amp(au.mouth_lower_down, 0.03)
        m_upper   = amp(au.mouth_upper_up, 0.03)
        m_lateral = amp((au.mouth_left + au.mouth_right) / 2.0, 0.03)

        # ── Positive evidence for each emotion ──

        # Happy: smile is the strongest signal, cheek raise confirms Duchenne
        smile_cheek = min(a12, a6)
        duchenne_bonus = 0.12 * eye_sqnt + 0.06 * m_dimple
        if a6 > 0.01:
            happy_ev = 0.35 * a12 + 0.25 * smile_cheek + 0.20 * a6 + duchenne_bonus
        else:
            happy_ev = 0.30 * a12

        # Sad: frown + oblique brow raise + eye droop; absence of smile boosts
        sad_base = (0.26 * a15 + 0.20 * a1 + 0.14 * a43 + 0.12 * a4
                    + 0.08 * a20 + 0.08 * m_press + 0.05 * m_roll + 0.04 * eye_sqnt
                    + 0.06 * m_lower)  # lower lip drop adds sadness signal
        no_smile_boost = max(0.0, 1.0 - a12 * 3.0)
        sad_ev = sad_base * (1.0 + 0.5 * no_smile_boost)
        # Co-occurrence: frown + oblique brow = strong sad signal
        if a15 > 0 and a1 > 0:
            sad_ev += 0.20 * min(a15, a1)
        # Frown + eyes closing = crying/tearful
        if a15 > 0 and a43 > 0:
            sad_ev += 0.14 * min(a15, a43)
        # Brow raise + lower lip drop = holding back tears
        if a1 > 0 and m_lower > 0:
            sad_ev += 0.10 * min(a1, m_lower)

        # Angry: brow lower + nose wrinkle; co-occurrence = very reliable
        angry_base = (0.26 * a4 + 0.22 * a9 + 0.14 * a20
                      + 0.10 * m_press + 0.08 * jaw_lat + 0.08 * m_shrug + 0.05 * m_roll
                      + 0.04 * m_lateral)  # mouth shift can signal anger
        if a4 > 0 and a9 > 0:
            angry_base += 0.24 * min(a4, a9)
        # Tight/pressed mouth boosts anger
        angry_ev = angry_base * (1.0 + 0.35 * m_press)
        # Brow lower + pressed lips = strong anger
        if a4 > 0 and m_press > 0:
            angry_ev += 0.12 * min(a4, m_press)

        # Surprised: brow raise, eye wide open, jaw drop, lips part
        surpr_ev = (0.22 * a1 + 0.20 * a2 + 0.20 * a26
                    + 0.18 * eye_wide + 0.12 * a25 + 0.08 * m_shrug)

        # Fear: wide eyes + TENSE mouth (vs surprise's OPEN mouth)
        fear_base = (0.20 * a1 + 0.22 * eye_wide + 0.18 * a20
                     + 0.14 * a4 + 0.10 * a2 + 0.10 * m_press)
        # Mouth tension differentiates fear from surprise
        mouth_tension = max(0.0, a20 + m_press - a26 * 0.5)
        fear_ev = fear_base + 0.18 * mouth_tension
        # Fear triad: inner brow + wide eyes + lip stretch
        if a1 > 0 and eye_wide > 0 and a20 > 0:
            fear_ev += 0.18 * min(a1, eye_wide, a20)
        # Wide eyes + brow raise without jaw drop = fear not surprise
        if eye_wide > 0 and a1 > 0 and a26 < 0.1:
            fear_ev += 0.10 * min(eye_wide, a1)

        # Anxious: subtle micro-tensions, self-soothing mouth movements
        anx_base = (0.22 * a4 + 0.18 * a20 + 0.14 * a1
                    + 0.12 * a43 + 0.12 * m_roll + 0.10 * m_press + 0.06 * jaw_lat + 0.04 * m_shrug)
        # Self-soothing gestures (lip rolling, pressing) amplify anxiety
        anx_ev = anx_base * (1.0 + 0.6 * (m_roll + m_press))
        # Lip rolling alone is a strong anxiety cue
        if m_roll > 0.1:
            anx_ev += 0.12 * m_roll

        # Disgusted: nose wrinkle dominant, upper lip raise, pucker, cheek involvement
        disg_base = (0.28 * a9 + 0.18 * a15 + 0.14 * a4
                     + 0.16 * m_pucker + 0.10 * cheek_pf + 0.08 * m_shrug
                     + 0.10 * m_upper + 0.06 * m_lateral)  # upper lip raise + mouth shift
        # Nose wrinkle + lip depress together = very strong disgust
        if a9 > 0 and a15 > 0:
            disg_base += 0.20 * min(a9, a15)
        if a9 > 0 and m_pucker > 0:
            disg_base += 0.12 * min(a9, m_pucker)
        # Upper lip raise + nose wrinkle = snarl (strongest disgust)
        if a9 > 0 and m_upper > 0:
            disg_base += 0.16 * min(a9, m_upper)
        disg_ev = disg_base

        # ── Per-emotion gain: compensate for naturally weaker AU signals ──
        # Happy/surprised already produce strong AUs; others need a lift.
        sad_ev   *= 1.50
        angry_ev *= 1.45
        fear_ev  *= 1.40
        anx_ev   *= 1.55
        disg_ev  *= 1.45

        # ── Contradictory suppression (minimal to avoid cancelling weak emotions) ──
        happy_sup  = 0.10 * a4 + 0.08 * a15 + 0.05 * a9
        sad_sup    = 0.08 * a12 + 0.04 * a6
        angry_sup  = 0.08 * a12 + 0.05 * a1
        surpr_sup  = 0.15 * a43 + 0.05 * a4
        fear_sup   = 0.08 * a12 + 0.04 * a6
        anx_sup    = 0.10 * a12 + 0.05 * a6
        disg_sup   = 0.08 * a12 + 0.04 * eye_wide

        scores = {
            "happy":     max(0.0, happy_ev - happy_sup),
            "sad":       max(0.0, sad_ev   - sad_sup),
            "angry":     max(0.0, angry_ev - angry_sup),
            "surprised": max(0.0, surpr_ev - surpr_sup),
            "fearful":   max(0.0, fear_ev  - fear_sup),
            "anxious":   max(0.0, anx_ev   - anx_sup),
            "disgusted": max(0.0, disg_ev  - disg_sup),
        }

        # Neutral: adaptive — strong when nothing fires, decays fast when emotions
        # separate clearly. Extended AUs help distinguish true neutral from subtle emotions.
        max_emotion = max(scores.values()) or 0.001
        second_max = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0.0
        separation_factor = 1.0 + max(0.0, max_emotion - second_max) * 2.5
        total_facial_activity = (a1 + a2 + a4 + a6 + a9 + a12 + a15 + a20 +
                                 eye_wide + eye_sqnt + m_pucker + m_press +
                                 m_lower + m_upper + m_lateral) / 15.0
        neutral_base = 0.30 * (1.0 - max_emotion) ** (1.4 * separation_factor)
        # When face is very active but no single emotion wins, reduce neutral
        neutral_base *= max(0.3, 1.0 - total_facial_activity * 1.5)
        scores["neutral"] = self._clip01(neutral_base)

        return self._normalize_scores(scores)

    def _detect_micro_expressions(
        self, au: ActionUnits, session_id: Optional[str], now_ts: float
    ) -> list[MicroExpression]:
        """Detect fleeting micro-expressions using 27 AU channels.

        Micro-expressions last 40-500ms and are involuntary facial movements
        that reveal concealed emotions. We track rapid AU deltas across frames
        and use extended blendshapes for higher sensitivity.
        """
        if not session_id:
            return []

        state = self._session_state.get(session_id)
        if state is None:
            return []

        # Comprehensive AU snapshot with extended channels
        au_snapshot = {
            "ts": now_ts,
            "au12": au.au12_lip_corner_pull,
            "au15": au.au15_lip_corner_depress,
            "au4": au.au4_brow_lowerer,
            "au1": au.au1_inner_brow_raise,
            "au2": au.au2_outer_brow_raise,
            "au6": au.au6_cheek_raise,
            "au9": au.au9_nose_wrinkle,
            "au25": au.au25_lips_part,
            "au26": au.au26_jaw_drop,
            "au20": au.au20_lip_stretch,
            "au43": au.au43_eyes_closed,
            # Extended channels
            "eye_wide": (au.eye_wide_left + au.eye_wide_right) / 2.0,
            "eye_squint": (au.eye_squint_left + au.eye_squint_right) / 2.0,
            "mouth_pucker": au.mouth_pucker,
            "mouth_press": (au.mouth_press_left + au.mouth_press_right) / 2.0,
            "mouth_dimple": (au.mouth_dimple_left + au.mouth_dimple_right) / 2.0,
            "mouth_roll": (au.mouth_roll_lower + au.mouth_roll_upper) / 2.0,
            "cheek_puff": au.cheek_puff,
        }
        state.recent_au_snapshots.append(au_snapshot)

        if len(state.recent_au_snapshots) > 15:
            state.recent_au_snapshots = state.recent_au_snapshots[-15:]

        micro_exprs = []
        if len(state.recent_au_snapshots) < 3:
            return micro_exprs

        # Compare current vs 2-frames-ago and 1-frame-ago for velocity detection
        prev2 = state.recent_au_snapshots[-3]
        prev1 = state.recent_au_snapshots[-2]
        curr = state.recent_au_snapshots[-1]
        dt_ms = (curr["ts"] - prev2["ts"]) * 1000.0

        if dt_ms > 3000:
            return micro_exprs

        def delta(key: str) -> float:
            return curr[key] - prev2[key]

        def velocity(key: str) -> float:
            """AU velocity: how fast is it changing (rise then fall = micro-expr)."""
            rise = prev1[key] - prev2[key]
            fall = curr[key] - prev1[key]
            if rise > 0.08 and fall < -0.03:
                return rise  # peaked and started falling = micro-expression
            return 0.0

        # Micro-smile (suppressed happiness): smile spike + quick fade
        smile_vel = velocity("au12")
        smile_delta = delta("au12")
        if smile_vel > 0.12 or (smile_delta > 0.15 and curr["au12"] < 0.5):
            intensity = max(smile_vel, smile_delta)
            # Stronger if dimples also fired
            if delta("mouth_dimple") > 0.05:
                intensity *= 1.2
            micro_exprs.append(MicroExpression(
                emotion="happy", intensity=round(min(1.0, intensity), 3),
                duration_ms=round(dt_ms, 1), timestamp=now_ts,
            ))

        # Micro-frown (suppressed anger/sadness)
        frown_vel = velocity("au4")
        frown_delta = delta("au4")
        if frown_vel > 0.10 or (frown_delta > 0.15 and curr["au4"] < 0.5):
            intensity = max(frown_vel, frown_delta)
            # Anger if nose wrinkle or mouth press co-occur
            is_anger = curr["au9"] > 0.2 or curr["mouth_press"] > 0.15
            micro_exprs.append(MicroExpression(
                emotion="angry" if is_anger else "sad",
                intensity=round(min(1.0, intensity), 3),
                duration_ms=round(dt_ms, 1), timestamp=now_ts,
            ))

        # Micro-surprise: rapid eye widening + brow raise
        eye_wide_delta = delta("eye_wide")
        brow_delta = (delta("au1") + delta("au2")) / 2.0
        if (eye_wide_delta > 0.12 and brow_delta > 0.10) or brow_delta > 0.20:
            intensity = (eye_wide_delta + brow_delta) / 2.0
            micro_exprs.append(MicroExpression(
                emotion="surprised", intensity=round(min(1.0, intensity), 3),
                duration_ms=round(dt_ms, 1), timestamp=now_ts,
            ))

        # Micro-fear: eye wide + brow furrow (AU4) co-occurring
        if eye_wide_delta > 0.10 and delta("au4") > 0.08:
            intensity = (eye_wide_delta + delta("au4")) / 2.0
            micro_exprs.append(MicroExpression(
                emotion="fearful", intensity=round(min(1.0, intensity), 3),
                duration_ms=round(dt_ms, 1), timestamp=now_ts,
            ))

        # Micro-disgust: nose wrinkle spike + lip curl or pucker
        disgust_delta = delta("au9")
        pucker_delta = delta("mouth_pucker")
        if disgust_delta > 0.15 or (disgust_delta > 0.10 and pucker_delta > 0.08):
            intensity = disgust_delta + 0.3 * pucker_delta
            micro_exprs.append(MicroExpression(
                emotion="disgusted", intensity=round(min(1.0, intensity), 3),
                duration_ms=round(dt_ms, 1), timestamp=now_ts,
            ))

        # Micro-contempt: asymmetric smile (one-sided lip pull)
        smile_asym = abs(au.au12_lip_corner_pull - au.mouth_dimple_left)
        if smile_asym > 0.15 and curr["au12"] > 0.1:
            micro_exprs.append(MicroExpression(
                emotion="disgusted",  # contempt mapped to disgust family
                intensity=round(min(1.0, smile_asym), 3),
                duration_ms=round(dt_ms, 1), timestamp=now_ts,
            ))

        state.micro_expressions_detected.extend(micro_exprs)
        if len(state.micro_expressions_detected) > 50:
            state.micro_expressions_detected = state.micro_expressions_detected[-50:]

        return micro_exprs

    def _face_mesh_detect(self, frame: np.ndarray, session_id: Optional[str] = None) -> Optional[tuple[dict, ActionUnits, list, list]]:
        """Run FaceLandmarker and return (emotion_scores, action_units, micro_expressions) or None."""
        if not self._face_mesh_loaded or self._face_mesh is None or self._mp_module is None:
            return None

        try:
            mp = self._mp_module

            # Try original frame first, then rotate 90° if face not found
            # (safety net for mobile cameras that may still send unrotated frames)
            for rotation in (None, cv2.ROTATE_90_COUNTERCLOCKWISE):
                img = frame if rotation is None else cv2.rotate(frame, rotation)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img))
                results = self._face_mesh.detect(mp_image)

                if results and results.face_landmarks and len(results.face_landmarks) > 0:
                    if rotation is not None:
                        logger.info("face found after rotation %s", rotation)
                    break
            else:
                results = None

            if not results or not results.face_landmarks or len(results.face_landmarks) == 0:
                logger.debug("FaceLandmarker: no face detected in frame %dx%d", frame.shape[1], frame.shape[0])
                return None

            landmarks = results.face_landmarks[0]
            if len(landmarks) < 468:
                return None

            # ── Extract AU from blendshapes (52 ARKit-style, much more accurate) ──
            au = ActionUnits()
            blendshape_map = {}
            if results.face_blendshapes and len(results.face_blendshapes) > 0:
                for bs in results.face_blendshapes[0]:
                    blendshape_map[bs.category_name] = bs.score

                # Map ARKit blendshapes → FACS Action Units
                au.au1_inner_brow_raise = self._clip01(
                    max(blendshape_map.get("browInnerUp", 0.0), 0.0)
                )
                au.au2_outer_brow_raise = self._clip01(
                    max(
                        blendshape_map.get("browOuterUpLeft", 0.0),
                        blendshape_map.get("browOuterUpRight", 0.0),
                    )
                )
                au.au4_brow_lowerer = self._clip01(
                    max(
                        blendshape_map.get("browDownLeft", 0.0),
                        blendshape_map.get("browDownRight", 0.0),
                    )
                )
                au.au6_cheek_raise = self._clip01(
                    (
                        blendshape_map.get("cheekSquintLeft", 0.0)
                        + blendshape_map.get("cheekSquintRight", 0.0)
                    )
                    / 2.0
                )
                au.au9_nose_wrinkle = self._clip01(
                    blendshape_map.get("noseSneerLeft", 0.0)
                    + blendshape_map.get("noseSneerRight", 0.0)
                ) 
                au.au12_lip_corner_pull = self._clip01(
                    (
                        blendshape_map.get("mouthSmileLeft", 0.0)
                        + blendshape_map.get("mouthSmileRight", 0.0)
                    )
                    / 2.0
                )
                au.au15_lip_corner_depress = self._clip01(
                    (
                        blendshape_map.get("mouthFrownLeft", 0.0)
                        + blendshape_map.get("mouthFrownRight", 0.0)
                    )
                    / 2.0
                )
                au.au20_lip_stretch = self._clip01(
                    (
                        blendshape_map.get("mouthStretchLeft", 0.0)
                        + blendshape_map.get("mouthStretchRight", 0.0)
                    )
                    / 2.0
                )
                au.au25_lips_part = self._clip01(
                    max(
                        blendshape_map.get("mouthOpen", 0.0),
                        blendshape_map.get("jawOpen", 0.0) * 0.5,
                    )
                )
                au.au26_jaw_drop = self._clip01(
                    blendshape_map.get("jawOpen", 0.0)
                )
                au.au43_eyes_closed = self._clip01(
                    (
                        blendshape_map.get("eyeBlinkLeft", 0.0)
                        + blendshape_map.get("eyeBlinkRight", 0.0)
                    )
                    / 2.0
                )
                # Extended blendshapes for higher precision
                au.eye_wide_left = self._clip01(blendshape_map.get("eyeWideLeft", 0.0))
                au.eye_wide_right = self._clip01(blendshape_map.get("eyeWideRight", 0.0))
                au.eye_squint_left = self._clip01(blendshape_map.get("eyeSquintLeft", 0.0))
                au.eye_squint_right = self._clip01(blendshape_map.get("eyeSquintRight", 0.0))
                au.mouth_pucker = self._clip01(blendshape_map.get("mouthPucker", 0.0))
                au.mouth_press_left = self._clip01(blendshape_map.get("mouthPressLeft", 0.0))
                au.mouth_press_right = self._clip01(blendshape_map.get("mouthPressRight", 0.0))
                au.cheek_puff = self._clip01(
                    (blendshape_map.get("cheekPuff", 0.0))
                )
                au.jaw_left = self._clip01(blendshape_map.get("jawLeft", 0.0))
                au.jaw_right = self._clip01(blendshape_map.get("jawRight", 0.0))
                au.mouth_dimple_left = self._clip01(blendshape_map.get("mouthDimpleLeft", 0.0))
                au.mouth_dimple_right = self._clip01(blendshape_map.get("mouthDimpleRight", 0.0))
                au.mouth_roll_lower = self._clip01(blendshape_map.get("mouthRollLower", 0.0))
                au.mouth_roll_upper = self._clip01(blendshape_map.get("mouthRollUpper", 0.0))
                au.mouth_shrug_lower = self._clip01(blendshape_map.get("mouthShrugLower", 0.0))
                au.mouth_shrug_upper = self._clip01(blendshape_map.get("mouthShrugUpper", 0.0))
                # Extra blendshapes for improved precision
                au.mouth_lower_down = self._clip01(
                    (blendshape_map.get("mouthLowerDownLeft", 0.0)
                     + blendshape_map.get("mouthLowerDownRight", 0.0)) / 2.0
                )
                au.mouth_upper_up = self._clip01(
                    (blendshape_map.get("mouthUpperUpLeft", 0.0)
                     + blendshape_map.get("mouthUpperUpRight", 0.0)) / 2.0
                )
                au.mouth_left = self._clip01(blendshape_map.get("mouthLeft", 0.0))
                au.mouth_right = self._clip01(blendshape_map.get("mouthRight", 0.0))
                au.brow_inner_up = self._clip01(blendshape_map.get("browInnerUp", 0.0))

                au.head_tilt_x = blendshape_map.get("headPitch", 0.0) if "headPitch" in blendshape_map else 0.0
                au.head_tilt_y = blendshape_map.get("headYaw", 0.0) if "headYaw" in blendshape_map else 0.0
                au.face_quality = self._clip01(1.0 - abs(au.head_tilt_y) * 1.5)

                logger.debug(
                    "blendshapes | smile=%.3f frown=%.3f brow_low=%.3f brow_raise=%.3f cheek=%.3f jaw=%.3f eyes_closed=%.3f nose=%.3f",
                    au.au12_lip_corner_pull, au.au15_lip_corner_depress,
                    au.au4_brow_lowerer, au.au1_inner_brow_raise,
                    au.au6_cheek_raise, au.au26_jaw_drop,
                    au.au43_eyes_closed, au.au9_nose_wrinkle,
                )

                # Compute facial asymmetry (genuine emotions are symmetric)
                asymmetry = self._compute_asymmetry(blendshape_map)
            else:
                # Fallback: compute AUs from landmark geometry
                au = self._compute_action_units(landmarks)
                asymmetry = 0.0

            # Adaptive calibration: learn resting-face baseline from first frames
            if session_id:
                self._update_calibration(au, session_id)
                # Log raw vs post-baseline AUs for debugging
                raw_smile = au.au12_lip_corner_pull
                raw_cheek = au.au6_cheek_raise
                raw_frown = au.au15_lip_corner_depress
                raw_brow = au.au1_inner_brow_raise
                raw_eye_wide = (au.eye_wide_left + au.eye_wide_right) / 2.0
                au = self._subtract_baseline(au, session_id)
                logger.info(
                    "AU raw→sub | smile=%.3f→%.3f cheek=%.3f→%.3f frown=%.3f→%.3f brow=%.3f→%.3f eye_w=%.3f→%.3f",
                    raw_smile, au.au12_lip_corner_pull,
                    raw_cheek, au.au6_cheek_raise,
                    raw_frown, au.au15_lip_corner_depress,
                    raw_brow, au.au1_inner_brow_raise,
                    raw_eye_wide, (au.eye_wide_left + au.eye_wide_right) / 2.0,
                )

            scores = self._au_to_emotion_scores(au)
            logger.info("raw scores | %s", {k: round(v, 3) for k, v in sorted(scores.items())})

            # Penalize confidence for highly asymmetric expressions (likely fake)
            if asymmetry > 0.3:
                suppress = min(0.15, (asymmetry - 0.3) * 0.5)
                for k in scores:
                    if k != "neutral":
                        scores[k] = max(0.0, scores[k] - suppress)
                scores = self._normalize_scores(scores)

            # Multi-frame voting for noise reduction
            if session_id:
                scores = self._multi_frame_vote(scores, session_id)

            now_ts = time.time()
            micro_exprs = self._detect_micro_expressions(au, session_id, now_ts)

            return scores, au, micro_exprs, landmarks
        except Exception as e:
            logger.debug("FaceLandmarker analysis failed: %s", e)
            return None

    def _ensemble_fusion(
        self,
        deepface_scores: Optional[dict[str, float]],
        facemesh_scores: Optional[dict[str, float]],
        opencv_scores: Optional[dict[str, float]],
    ) -> dict[str, float]:
        """Weighted ensemble fusion of all available model outputs."""
        # Weights: DeepFace (trained CNN) > Face Mesh (landmark geometry) > OpenCV (heuristic)
        weights = []
        score_sets = []

        if deepface_scores:
            weights.append(0.45)
            score_sets.append(deepface_scores)
        if facemesh_scores:
            weights.append(0.40 if deepface_scores else 0.60)
            score_sets.append(facemesh_scores)
        if opencv_scores:
            weights.append(0.15 if (deepface_scores or facemesh_scores) else 0.50)
            score_sets.append(opencv_scores)

        if not score_sets:
            return {"neutral": 1.0}

        # Normalize weights to sum to 1
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        all_keys = set()
        for s in score_sets:
            all_keys.update(s.keys())

        fused = {}
        for key in all_keys:
            fused[key] = sum(
                w * s.get(key, 0.0)
                for w, s in zip(weights, score_sets)
            )

        return self._normalize_scores(fused)

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

    @staticmethod
    def _apply_exif_rotation(img_bytes: bytes) -> Optional[np.ndarray]:
        """Decode image with EXIF orientation applied using Pillow."""
        try:
            from PIL import Image, ImageOps
            import io
            pil_img = Image.open(io.BytesIO(img_bytes))
            pil_img = ImageOps.exif_transpose(pil_img)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            return np.array(pil_img)
        except Exception:
            return None

    def _decode_frame(self, frame_data: str) -> Optional[np.ndarray]:
        """Decode base64 frame to numpy array with EXIF rotation support."""
        try:
            if "," in frame_data:
                frame_data = frame_data.split(",")[1]
            img_bytes = base64.b64decode(frame_data)

            # Use Pillow to handle EXIF orientation (mobile cameras embed rotation tags)
            frame = self._apply_exif_rotation(img_bytes)
            if frame is None:
                # Fallback to OpenCV (no EXIF rotation)
                np_buffer = np.frombuffer(img_bytes, dtype=np.uint8)
                decoded = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
                if decoded is None:
                    raise ValueError("invalid image bytes")
                frame = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

            # Higher max dimension for better landmark precision
            h, w = frame.shape[:2]
            max_dim = max(h, w)
            if max_dim > 1024:
                scale = 1024.0 / float(max_dim)
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
            # Skip heavy enhancement when frame quality is already good.
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            brightness = float(np.mean(gray)) / 255.0
            contrast = float(np.std(gray)) / 64.0
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 140.0
            quality_ok = 0.18 <= brightness <= 0.9 and contrast >= 0.5 and sharpness >= 0.33
            if quality_ok:
                return frame

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
            if len(self._fallback_prev_scores_by_session) <= 128:
                return
        stale_keys = [
            key for key, state in self._session_state.items()
            if now_ts - state.updated_at > 20 * 60
        ]
        for key in stale_keys:
            self._session_state.pop(key, None)
            self._fallback_prev_scores_by_session.pop(key, None)

        if len(self._fallback_prev_scores_by_session) > 128:
            for key in list(self._fallback_prev_scores_by_session.keys())[:32]:
                if key not in self._session_state:
                    self._fallback_prev_scores_by_session.pop(key, None)

    def _apply_temporal_smoothing(self, result: EmotionResult, session_id: Optional[str]) -> EmotionResult:
        """Temporal EMA + hysteresis per session to reduce flicker and false positives."""
        if not session_id or not result.face_detected or not result.all_scores:
            return result

        now_ts = time.time()
        self._cleanup_old_sessions(now_ts)

        state = self._session_state.get(session_id)
        if state is None:
            # First frame for this session — accept it as-is
            state = SessionTemporalState(
                scores_ema=dict(result.all_scores),
                last_emotion=result.emotion,
                updated_at=now_ts,
            )
            self._session_state[session_id] = state
            return result

        # Very responsive EMA: alpha=0.88 makes current frame dominant
        # to avoid ghost of previous emotion lingering in the scores.
        alpha = 0.88 if result.confidence >= 0.50 else 0.75
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
            if state.pending_emotion == candidate:
                state.pending_count += 1
            else:
                state.pending_emotion = candidate
                state.pending_count = 1

            # Immediate switch when the new emotion has any meaningful lead.
            # No pending needed — the EMA already provides smoothing.
            immediate_switch = gap >= 0.04
            confirmed_switch = state.pending_count >= 2 and gap >= 0.015

            if immediate_switch or confirmed_switch:
                logger.info(
                    "emotion switch | %s → %s (gap=%.3f conf=%.3f pending=%d)",
                    state.last_emotion, candidate, gap, result.confidence, state.pending_count,
                )
                chosen = candidate
                state.last_emotion = candidate
                state.pending_emotion = None
                state.pending_count = 0
            else:
                logger.debug(
                    "emotion switch pending | %s → %s (gap=%.3f conf=%.3f pending=%d)",
                    state.last_emotion, candidate, gap, result.confidence, state.pending_count,
                )
        else:
            state.pending_emotion = None
            state.pending_count = 0

        confidence = self._clip01((result.confidence * 0.50) + (candidate_score * 0.50))

        # Force neutral for very weak signals only
        if chosen != "neutral" and confidence < 0.10:
            chosen = "neutral"
            state.last_emotion = "neutral"

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

    def _fallback_detect(self, frame: np.ndarray, session_id: Optional[str] = None) -> EmotionResult:
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
            if self._mp_face_detector is not None and self._mp_module is not None:
                try:
                    mp = self._mp_module
                    h0, w0 = frame.shape[0], frame.shape[1]
                    target_w = 320
                    scale = target_w / float(w0)
                    resized = cv2.resize(
                        frame,
                        (target_w, max(1, int(h0 * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=resized)
                    mp_result = self._mp_face_detector.detect(mp_image)
                    if mp_result and mp_result.detections:
                        best = max(
                            mp_result.detections,
                            key=lambda d: float(d.categories[0].score) if d.categories else 0.0,
                        )
                        bbox = best.bounding_box
                        x = bbox.origin_x
                        y = bbox.origin_y
                        w = bbox.width
                        h = bbox.height
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
            calm_face_score = self._clip01((eye_score * 0.6) + ((1.0 - eyebrow_tension) * 0.4))

            # More dynamic fallback: estimates rough emotional tendencies from visual cues.
            scores = {
                "neutral": 0.26 + 0.12 * calm_face_score + 0.06 * (1 - mouth_open_score),
                "happy": 0.07 + 0.56 * smile_score + 0.09 * eye_score,
                "surprised": 0.08 + 0.48 * mouth_open_score * eye_score + 0.08 * face_area_ratio,
                "sad": 0.07 + 0.22 * (1 - eye_score) + 0.16 * (1 - smile_score),
                "anxious": 0.04 + 0.16 * mouth_open_score + 0.14 * (1 - eye_score) + 0.12 * eyebrow_tension,
                "angry": 0.08 + 0.28 * eyebrow_tension + 0.16 * (1 - smile_score),
                "disgusted": 0.06 + 0.14 * mouth_open_score * (1 - eye_score) + 0.10 * eyebrow_tension,
            }

            # Calm facial posture should not be interpreted as anxious.
            if calm_face_score > 0.62 and mouth_open_score < 0.36:
                scores["neutral"] += 0.1
                scores["anxious"] *= 0.72

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

            # Avoid anxious over-detection on stable, reasonably open faces.
            if eye_score > 0.48 and eyebrow_tension < 0.46:
                scores["anxious"] *= 0.76
                scores["neutral"] += 0.06

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
            prev_scores = self._fallback_prev_scores_by_session.get(session_id or "__global__")
            if prev_scores:
                quality_score = self._clip01((brightness_score * 0.45) + (sharpness_score * 0.55))
                prev_weight = 0.48 if quality_score < 0.35 else 0.34
                new_weight = 1.0 - prev_weight
                smoothed = {}
                for key, value in normalized.items():
                    prev = prev_scores.get(key, value)
                    smoothed[key] = round(prev * prev_weight + value * new_weight, 3)
                normalized = self._normalize_scores(smoothed)
            self._fallback_prev_scores_by_session[session_id or "__global__"] = normalized

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

            # Strong guard for false "anxious" on neutral/happy-like expressions.
            if primary == "anxious":
                anxious_gap = max(0.0, primary_score - secondary_score)
                weak_anxious_signal = (
                    eyebrow_tension < 0.48
                    or anxious_gap < 0.1
                    or (secondary in {"neutral", "happy"} and secondary_score >= primary_score - 0.07)
                )
                if weak_anxious_signal:
                    normalized["anxious"] *= 0.68
                    normalized["neutral"] = normalized.get("neutral", 0.0) + 0.09
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

            # If model is uncertain, force neutral instead of a random emotional label.
            uncertain_label = primary != "neutral" and (primary_score < 0.34 or separation < 0.09)
            if uncertain_label:
                primary = "neutral"
                normalized["neutral"] = max(normalized.get("neutral", 0.0), primary_score)
                normalized = self._normalize_scores(normalized)
                ranking = sorted(normalized.items(), key=lambda kv: kv[1], reverse=True)
                primary_score = normalized.get("neutral", ranking[0][1])
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

    def detect_from_frame(self, frame: np.ndarray, session_id: Optional[str] = None) -> EmotionResult:
        """Run multi-model ensemble emotion detection on a numpy frame."""
        start = time.time()
        enhanced = self._enhance_frame_for_emotion(frame)

        deepface_scores = None
        facemesh_scores = None
        opencv_scores = None
        action_units_data = None
        micro_exprs = []
        models_used = []
        face_mesh_count = 0
        face_landmarks = None

        # ── Model 1: Face Mesh (468 landmarks → Action Units → Emotion) ──
        # Use original frame (not enhanced) — FaceLandmarker works best on
        # natural images; CLAHE/sharpening can distort facial feature geometry.
        with self._detect_lock:
            mesh_result = self._face_mesh_detect(frame, session_id)
        if mesh_result is not None:
            facemesh_scores, action_units_data, micro_exprs, face_landmarks = mesh_result
            models_used.append("face_mesh_468")
            face_mesh_count = 468

        # ── Model 2: DeepFace CNN ─────────────────────────────────────────
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
                total = sum(raw_emotions.values()) or 1
                deepface_scores = {EMOTION_MAP.get(k, k): v / total for k, v in raw_emotions.items()}
                models_used.append("deepface_cnn")
            except Exception as e:
                logger.error(f"DeepFace analysis error: {e}")

        # ── Model 3: OpenCV Heuristic (always available as backup) ────────
        if self._opencv_enabled:
            opencv_result = self._opencv_heuristic_scores(enhanced, session_id)
            if opencv_result is not None:
                opencv_scores = opencv_result
                if "opencv_haar" not in models_used:
                    models_used.append("opencv_haar")

        # ── Ensemble Fusion ───────────────────────────────────────────────
        if not deepface_scores and not facemesh_scores and not opencv_scores:
            # No model produced results — likely no face
            logger.debug("no model produced scores, falling back to full fallback")
            result = self._fallback_detect(enhanced, session_id=session_id)
            result.processing_time_ms = round((time.time() - start) * 1000, 1)
            result.detection_models_used = ["fallback"]
            return result

        fused_scores = self._ensemble_fusion(deepface_scores, facemesh_scores, opencv_scores)

        # Post-fusion corrections
        fused_scores = self._post_fusion_corrections(fused_scores, action_units_data)

        # ── rPPG: collect green-channel signal and adjust scores by heart rate ──
        hr_bpm = None
        hr_confidence = 0.0
        hr_status = "collecting"
        if session_id and face_landmarks is not None:
            state = self._session_state.get(session_id)
            if state is None:
                state = SessionTemporalState(updated_at=time.time())
                self._session_state[session_id] = state

            green_val = self._extract_rppg_signal(frame, face_landmarks)
            if green_val is not None:
                state.rppg_timestamps.append(time.time())
                state.rppg_green_means.append(green_val)
                # Keep buffer bounded
                max_samples = 60
                if len(state.rppg_timestamps) > max_samples:
                    state.rppg_timestamps = state.rppg_timestamps[-max_samples:]
                    state.rppg_green_means = state.rppg_green_means[-max_samples:]

            hr_bpm, hr_confidence, hr_status = self._compute_heart_rate(state)

            # Adjust emotion scores using heart rate context
            fused_scores = self._hr_emotion_adjustment(fused_scores, hr_bpm, hr_confidence)

        # Determine primary/secondary
        ranking = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)
        primary = ranking[0][0]
        primary_score = ranking[0][1]
        secondary = ranking[1][0] if len(ranking) > 1 else None
        secondary_score = ranking[1][1] if len(ranking) > 1 else 0.0
        gap = max(0.0, primary_score - secondary_score)

        # Confidence from ensemble agreement
        confidence = self._compute_ensemble_confidence(
            primary, primary_score, gap, deepface_scores, facemesh_scores, opencv_scores
        )

        variant, zone, tip = self._derive_variant(primary, confidence, secondary)

        # Boost confidence if micro-expressions agree with primary emotion
        if micro_exprs:
            matching_micros = [m for m in micro_exprs if m.emotion == primary]
            if matching_micros:
                confidence = self._clip01(confidence + 0.05)

        # New analytics
        compound = self._detect_compound_emotion(fused_scores)
        intensity = self._classify_intensity(confidence, action_units_data)
        face_quality = self._compute_face_quality(enhanced, action_units_data)

        result = EmotionResult(
            emotion=primary,
            confidence=round(confidence, 3),
            all_scores={k: round(v, 3) for k, v in fused_scores.items()},
            emotion_variant=variant,
            emotion_zone=zone,
            support_tip=tip,
            secondary_emotion=secondary,
            face_detected=True,
            processing_time_ms=round((time.time() - start) * 1000, 1),
            music_suggestions=MUSIC_SUGGESTIONS.get(primary, []),
            action_units={
                "au1_inner_brow_raise": round(action_units_data.au1_inner_brow_raise, 3),
                "au2_outer_brow_raise": round(action_units_data.au2_outer_brow_raise, 3),
                "au4_brow_lowerer": round(action_units_data.au4_brow_lowerer, 3),
                "au6_cheek_raise": round(action_units_data.au6_cheek_raise, 3),
                "au9_nose_wrinkle": round(action_units_data.au9_nose_wrinkle, 3),
                "au12_lip_corner_pull": round(action_units_data.au12_lip_corner_pull, 3),
                "au15_lip_corner_depress": round(action_units_data.au15_lip_corner_depress, 3),
                "au20_lip_stretch": round(action_units_data.au20_lip_stretch, 3),
                "au25_lips_part": round(action_units_data.au25_lips_part, 3),
                "au26_jaw_drop": round(action_units_data.au26_jaw_drop, 3),
                "au43_eyes_closed": round(action_units_data.au43_eyes_closed, 3),
            } if action_units_data else None,
            micro_expressions=[
                {"emotion": m.emotion, "intensity": m.intensity, "duration_ms": m.duration_ms}
                for m in micro_exprs
            ],
            detection_models_used=models_used,
            face_mesh_landmarks_count=face_mesh_count,
            compound_emotion=compound,
            emotion_intensity=intensity,
            face_quality_metrics=face_quality,
            heart_rate_bpm=hr_bpm,
            heart_rate_confidence=round(hr_confidence, 3),
            heart_rate_status=hr_status,
        )

        return result

    def _opencv_heuristic_scores(self, frame: np.ndarray, session_id: Optional[str] = None) -> Optional[dict[str, float]]:
        """Extract raw emotion scores from OpenCV heuristics (no full EmotionResult)."""
        if not self._opencv_enabled or self.face_cascade is None:
            return None

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=6, minSize=(72, 72))
            if len(faces) == 0:
                return None

            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            if w < 72 or h < 72:
                return None

            roi = gray[y:y+h, x:x+w]
            roi_eq = cv2.equalizeHist(roi)

            eyes = self.eye_cascade.detectMultiScale(roi_eq, scaleFactor=1.1, minNeighbors=8, minSize=(max(12, w//9), max(12, h//9)))
            smiles = self.smile_cascade.detectMultiScale(roi_eq, scaleFactor=1.7, minNeighbors=20, minSize=(max(20, w//5), max(16, h//10)))

            eye_score = self._clip01(len(eyes) / 2.0)
            smile_score = self._clip01(len(smiles) / 2.0)

            lower = roi_eq[int(h * 0.55):, :]
            upper = roi_eq[:max(1, int(h * 0.35)), :]
            mouth_open_score = self._clip01(float(cv2.Canny(lower, 40, 120).mean()) / 55.0)
            eyebrow_tension = self._clip01(float(cv2.Canny(upper, 50, 130).mean()) / 55.0)

            scores = {
                "neutral": 0.26 + 0.12 * self._clip01((eye_score * 0.6) + ((1.0 - eyebrow_tension) * 0.4)),
                "happy": 0.07 + 0.56 * smile_score + 0.09 * eye_score,
                "surprised": 0.08 + 0.48 * mouth_open_score * eye_score,
                "sad": 0.07 + 0.22 * (1 - eye_score) + 0.16 * (1 - smile_score),
                "anxious": 0.04 + 0.16 * mouth_open_score + 0.14 * (1 - eye_score) + 0.12 * eyebrow_tension,
                "angry": 0.08 + 0.28 * eyebrow_tension + 0.16 * (1 - smile_score),
                "disgusted": 0.06 + 0.14 * mouth_open_score * (1 - eye_score) + 0.10 * eyebrow_tension,
            }

            for key in list(scores.keys()):
                scores[key] = max(0.01, scores[key])

            return self._normalize_scores(scores)
        except Exception as e:
            logger.debug("opencv heuristic scoring failed: %s", e)
            return None

    def _post_fusion_corrections(self, scores: dict[str, float], au: Optional[ActionUnits]) -> dict[str, float]:
        """Apply AU-informed corrections to reduce false positives after ensemble fusion."""
        if au is None:
            return scores

        # If face mesh shows very low smile but ensemble says happy, suppress
        if scores.get("happy", 0) > 0.3 and au.au12_lip_corner_pull < 0.15 and au.au6_cheek_raise < 0.15:
            scores["happy"] *= 0.5
            scores["neutral"] = scores.get("neutral", 0) + 0.05

        # If face mesh shows strong Duchenne smile but ensemble misses it
        if au.au12_lip_corner_pull > 0.6 and au.au6_cheek_raise > 0.4 and scores.get("happy", 0) < 0.3:
            scores["happy"] = max(scores.get("happy", 0), 0.35)

        # If brows are relaxed but ensemble says angry, suppress
        if scores.get("angry", 0) > 0.3 and au.au4_brow_lowerer < 0.2:
            scores["angry"] *= 0.5
            scores["neutral"] = scores.get("neutral", 0) + 0.04

        # If eyes are very closed and ensemble says surprised, suppress
        if scores.get("surprised", 0) > 0.3 and au.au43_eyes_closed > 0.6:
            scores["surprised"] *= 0.4
            scores["neutral"] = scores.get("neutral", 0) + 0.04

        # If no nose wrinkle but ensemble says disgusted, suppress
        if scores.get("disgusted", 0) > 0.25 and au.au9_nose_wrinkle < 0.15:
            scores["disgusted"] *= 0.5
            scores["neutral"] = scores.get("neutral", 0) + 0.03

        return self._normalize_scores(scores)

    # ── Compound Emotion Detection ──────────────────────────────────────────

    COMPOUND_EMOTIONS = {
        "bittersweet":  {"requires": ("happy", "sad"),       "min_scores": (0.18, 0.15), "label": "Agridoce"},
        "frustrated":   {"requires": ("angry", "sad"),       "min_scores": (0.18, 0.15), "label": "Frustrado(a)"},
        "awe":          {"requires": ("surprised", "happy"), "min_scores": (0.20, 0.15), "label": "Encantado(a)"},
        "anxious_sad":  {"requires": ("anxious", "sad"),     "min_scores": (0.18, 0.15), "label": "Triste e ansioso(a)"},
        "nervous_excited": {"requires": ("anxious", "happy"), "min_scores": (0.16, 0.16), "label": "Nervoso(a) empolgado(a)"},
        "contempt":     {"requires": ("angry", "disgusted"), "min_scores": (0.18, 0.15), "label": "Desdenhoso(a)"},
        "apprehensive": {"requires": ("fearful", "anxious"), "min_scores": (0.16, 0.16), "label": "Apreensivo(a)"},
        "melancholic":  {"requires": ("sad", "neutral"),     "min_scores": (0.20, 0.25), "label": "Melancólico(a)"},
        "embarrassed":  {"requires": ("surprised", "anxious"), "min_scores": (0.15, 0.15), "label": "Envergonhado(a)"},
    }

    def _detect_compound_emotion(self, scores: dict[str, float]) -> Optional[str]:
        """Detect compound (mixed) emotions from co-activated scores."""
        best_compound = None
        best_strength = 0.0
        for compound, spec in self.COMPOUND_EMOTIONS.items():
            e1, e2 = spec["requires"]
            min1, min2 = spec["min_scores"]
            s1 = scores.get(e1, 0.0)
            s2 = scores.get(e2, 0.0)
            if s1 >= min1 and s2 >= min2:
                strength = s1 + s2
                if strength > best_strength:
                    best_strength = strength
                    best_compound = compound
        return best_compound

    # ── Emotion Intensity Classification ─────────────────────────────────

    @staticmethod
    def _classify_intensity(confidence: float, au: Optional[ActionUnits]) -> str:
        """Map confidence + AU activation to a human-readable intensity level."""
        au_energy = 0.0
        if au is not None:
            au_energy = (
                au.au1_inner_brow_raise + au.au4_brow_lowerer +
                au.au6_cheek_raise + au.au9_nose_wrinkle +
                au.au12_lip_corner_pull + au.au15_lip_corner_depress +
                au.au20_lip_stretch + au.au25_lips_part +
                au.au26_jaw_drop
            ) / 9.0
        combined = confidence * 0.55 + au_energy * 0.45
        if combined >= 0.65:
            return "extreme"
        if combined >= 0.50:
            return "intense"
        if combined >= 0.35:
            return "moderate"
        if combined >= 0.18:
            return "mild"
        return "calm"

    # ── Face Quality Metrics ─────────────────────────────────────────────

    def _compute_face_quality(self, frame: np.ndarray, au: Optional[ActionUnits]) -> dict:
        """Return real-time face quality metrics to guide the user."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape[:2]

        brightness = float(np.mean(gray)) / 255.0
        contrast = float(np.std(gray)) / 128.0
        sharpness = min(1.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 200.0)

        # Lighting assessment
        if brightness < 0.18:
            lighting = "too_dark"
        elif brightness > 0.88:
            lighting = "too_bright"
        elif contrast < 0.25:
            lighting = "flat"
        else:
            lighting = "good"

        # Face angle from head tilt AUs
        angle = "frontal"
        if au is not None:
            if abs(au.head_tilt_y) > 0.35:
                angle = "side_turned"
            elif abs(au.head_tilt_x) > 0.35:
                angle = "tilted"

        # Overall quality score
        quality_score = self._clip01(
            brightness * 0.2 + contrast * 0.25 + sharpness * 0.35
            + (0.2 if angle == "frontal" else 0.05)
        )

        # Guidance tips
        tips = []
        if lighting == "too_dark":
            tips.append("Aumente a iluminação do ambiente")
        elif lighting == "too_bright":
            tips.append("Reduza a luz direta no rosto")
        elif lighting == "flat":
            tips.append("Melhore o contraste de iluminação")
        if sharpness < 0.3:
            tips.append("Mantenha o celular estável")
        if angle != "frontal":
            tips.append("Olhe diretamente para a câmera")

        return {
            "lighting": lighting,
            "sharpness": round(sharpness, 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "face_angle": angle,
            "quality_score": round(quality_score, 2),
            "tips": tips,
        }

    # ── Streak Tracking ──────────────────────────────────────────────────

    def _update_streak(self, state: SessionTemporalState, emotion: str, now_ts: float) -> float:
        """Update emotion streak and transitions. Returns streak duration in seconds."""
        if emotion != state.current_streak_emotion:
            # Record transition
            state.emotion_transitions.append({
                "from": state.current_streak_emotion,
                "to": emotion,
                "timestamp": now_ts,
                "duration_seconds": round(now_ts - state.current_streak_start, 1),
            })
            if len(state.emotion_transitions) > 20:
                state.emotion_transitions = state.emotion_transitions[-20:]
            state.current_streak_emotion = emotion
            state.current_streak_start = now_ts
            return 0.0
        return round(now_ts - state.current_streak_start, 1)

    # ── rPPG Heart Rate Estimation ──────────────────────────────────────────

    _RPPG_WINDOW_SEC = 20.0       # seconds of data needed for estimation
    _RPPG_MIN_SAMPLES = 12        # minimum frames (at ~1 fps, 12 seconds)
    _RPPG_BPM_LOW = 45.0
    _RPPG_BPM_HIGH = 180.0

    def _extract_rppg_signal(self, frame: np.ndarray, landmarks) -> Optional[float]:
        """Extract mean green-channel intensity from forehead/cheek ROI.

        The face landmarks are used to identify a stable forehead region
        where skin is visible and capillary pulsation can be measured.
        """
        try:
            h, w = frame.shape[:2]

            if landmarks is not None and len(landmarks) >= 468:
                # Use forehead region between brows and hairline
                # Landmarks: 10 (top of forehead), 107/336 (inner brows), 67/297 (forehead sides)
                pts = []
                for idx in [10, 67, 109, 108, 107, 336, 337, 338, 297]:
                    lm = landmarks[idx]
                    pts.append((int(lm.x * w), int(lm.y * h)))
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x1 = max(0, min(xs))
                y1 = max(0, min(ys))
                x2 = min(w, max(xs))
                y2 = min(h, max(ys))
            else:
                # Fallback: center-top region of frame (approximate forehead)
                x1 = int(w * 0.3)
                x2 = int(w * 0.7)
                y1 = int(h * 0.1)
                y2 = int(h * 0.3)

            if x2 <= x1 or y2 <= y1:
                return None

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return None

            # Mean of green channel (best SNR for rPPG)
            green_mean = float(np.mean(roi[:, :, 1]))
            return green_mean
        except Exception:
            return None

    def _compute_heart_rate(self, state: SessionTemporalState) -> tuple[Optional[float], float, str]:
        """Compute heart rate from accumulated rPPG green-channel signal.

        Returns (bpm, confidence, status).
        """
        timestamps = state.rppg_timestamps
        signal = state.rppg_green_means

        if len(signal) < self._RPPG_MIN_SAMPLES:
            return None, 0.0, "collecting"

        # Keep only the last _RPPG_WINDOW_SEC of data
        now = timestamps[-1]
        cutoff = now - self._RPPG_WINDOW_SEC
        while len(timestamps) > 0 and timestamps[0] < cutoff:
            timestamps.pop(0)
            signal.pop(0)

        if len(signal) < self._RPPG_MIN_SAMPLES:
            return None, 0.0, "collecting"

        try:
            sig = np.array(signal, dtype=np.float64)
            ts = np.array(timestamps, dtype=np.float64)

            # Estimate sampling rate from timestamps
            dt = np.diff(ts)
            if len(dt) == 0 or np.mean(dt) < 0.1:
                return None, 0.0, "unstable"
            fs = 1.0 / np.mean(dt)

            # Detrend: subtract moving average (window ~5 samples)
            kernel_size = min(5, len(sig) - 1)
            if kernel_size < 2:
                return None, 0.0, "collecting"
            kernel = np.ones(kernel_size) / kernel_size
            trend = np.convolve(sig, kernel, mode="same")
            detrended = sig - trend

            # Apply Hamming window to reduce spectral leakage
            window = np.hamming(len(detrended))
            windowed = detrended * window

            # FFT
            n = len(windowed)
            fft_vals = np.fft.rfft(windowed)
            fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
            magnitudes = np.abs(fft_vals)

            # Filter to physiological heart rate range
            freq_low = self._RPPG_BPM_LOW / 60.0   # ~0.75 Hz
            freq_high = self._RPPG_BPM_HIGH / 60.0  # ~3.0 Hz
            mask = (fft_freqs >= freq_low) & (fft_freqs <= freq_high)

            if not np.any(mask):
                return None, 0.0, "unstable"

            valid_freqs = fft_freqs[mask]
            valid_mags = magnitudes[mask]

            # Find dominant frequency
            peak_idx = np.argmax(valid_mags)
            peak_freq = valid_freqs[peak_idx]
            peak_mag = valid_mags[peak_idx]

            bpm = round(peak_freq * 60.0, 1)

            # Confidence: ratio of peak to total spectral energy in band
            total_energy = np.sum(valid_mags ** 2) or 1.0
            peak_energy = peak_mag ** 2
            snr = peak_energy / total_energy
            confidence = self._clip01(snr * 2.5)  # scale to 0–1

            # Require minimum confidence
            if confidence < 0.15:
                return state.rppg_last_bpm, confidence, "unstable"

            # Smooth with previous reading
            if state.rppg_last_bpm is not None:
                bpm = round(0.3 * state.rppg_last_bpm + 0.7 * bpm, 1)

            state.rppg_last_bpm = bpm
            state.rppg_last_confidence = confidence

            logger.debug("rPPG | bpm=%.1f confidence=%.2f samples=%d fs=%.1f", bpm, confidence, len(signal), fs)

            return bpm, confidence, "ready"
        except Exception as e:
            logger.debug("rPPG computation error: %s", e)
            return state.rppg_last_bpm, 0.0, "unstable"

    def _hr_emotion_adjustment(self, scores: dict[str, float], bpm: Optional[float], hr_confidence: float) -> dict[str, float]:
        """Adjust emotion scores based on heart rate context.

        High HR (>90) → boost arousal emotions (anxious, angry, surprised, happy)
        Low HR (<65) → boost valence emotions (sad, calm/neutral)
        Normal (65-90) → no adjustment
        """
        if bpm is None or hr_confidence < 0.25:
            return scores

        adjusted = dict(scores)

        if bpm > 95:
            # High heart rate: person is aroused — boost high-arousal emotions
            boost = min(0.15, (bpm - 95) / 200.0) * hr_confidence
            for emo in ["anxious", "angry", "surprised", "happy"]:
                adjusted[emo] = adjusted.get(emo, 0) + boost
            for emo in ["neutral", "sad"]:
                adjusted[emo] = max(0, adjusted.get(emo, 0) - boost * 0.5)
        elif bpm < 62:
            # Low heart rate: person is calm/subdued — boost low-arousal emotions
            boost = min(0.12, (62 - bpm) / 150.0) * hr_confidence
            for emo in ["sad", "neutral"]:
                adjusted[emo] = adjusted.get(emo, 0) + boost
            for emo in ["surprised", "angry", "anxious"]:
                adjusted[emo] = max(0, adjusted.get(emo, 0) - boost * 0.5)

        return self._normalize_scores(adjusted)

    def _compute_ensemble_confidence(
        self,
        primary: str,
        primary_score: float,
        gap: float,
        deepface_scores: Optional[dict],
        facemesh_scores: Optional[dict],
        opencv_scores: Optional[dict],
    ) -> float:
        """Compute confidence based on model agreement and score dominance."""
        base_conf = 0.30 + primary_score * 0.50 + gap * 0.30

        # Agreement bonus: if multiple models agree on the primary emotion
        agreement_count = 0
        total_models = 0

        for scores in [deepface_scores, facemesh_scores, opencv_scores]:
            if scores:
                total_models += 1
                model_top = max(scores, key=scores.get)
                if model_top == primary:
                    agreement_count += 1

        if total_models > 1:
            agreement_ratio = agreement_count / total_models
            base_conf += agreement_ratio * 0.18

        # Penalize neutral slightly so real emotions can surface
        if primary == "neutral":
            base_conf *= 0.90

        return self._clip01(base_conf)

    async def analyze_frame_base64(
        self,
        frame_b64: str,
        session_id: Optional[str] = None,
        pet_name: str = "MoodPet",
    ) -> EmotionResult:
        """Async entry point: decode frame and run multi-model ensemble detection."""
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

        # Run synchronously — FaceLandmarker uses GPU/OpenGL and must stay on
        # the thread that created it.  At ~50 ms per frame this is fine for 1 fps.
        result = self.detect_from_frame(frame, session_id)
        result = self._apply_temporal_smoothing(result, session_id)

        # Track emotion history and streak for pattern analysis
        if session_id:
            state = self._session_state.get(session_id)
            if state:
                now_ts = time.time()
                state.emotion_history.append({
                    "emotion": result.emotion,
                    "confidence": result.confidence,
                    "timestamp": result.timestamp,
                })
                if len(state.emotion_history) > 60:
                    state.emotion_history = state.emotion_history[-60:]

                # Update streak
                result.emotion_streak_seconds = self._update_streak(
                    state, result.emotion, now_ts
                )

        # Attach message
        import random
        messages = EMOTION_MESSAGES.get(result.emotion, EMOTION_MESSAGES["neutral"])
        result.message = random.choice(messages).replace("{pet_name}", pet_name)

        logger.info(
            "emotion analyzed | emotion=%s confidence=%.3f face_detected=%s processing_ms=%.1f models=%s micro_exprs=%d",
            result.emotion,
            result.confidence,
            result.face_detected,
            result.processing_time_ms,
            "+".join(result.detection_models_used) if result.detection_models_used else "fallback",
            len(result.micro_expressions),
        )

        return result


# Singleton instance
emotion_service = EmotionDetectionService()

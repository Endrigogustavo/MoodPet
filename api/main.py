"""
MoodPet API — Emotion Detection & Interaction System
FastAPI backend for continuous facial expression analysis
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from routers import emotion, history, alerts, dashboard, ai_chat, tools
from utils.logger import setup_logger
from utils.database import init_db

logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 MoodPet API starting up...")
    await init_db()
    logger.info("✅ Database initialized")
    yield
    logger.info("👋 MoodPet API shutting down...")


app = FastAPI(
    title="MoodPet API",
    description="""
## 🐾 MoodPet — Intelligent Emotional Interaction System

API responsável por:
- Processamento contínuo de frames para detecção de expressões faciais
- Análise emocional em tempo real com modelos de Deep Learning
- Geração de insights comportamentais com IA
- Gerenciamento de alertas e notificações
- Dashboard com histórico emocional
- Chat empático baseado na emoção detectada

### Emoções detectadas
`happy` · `sad` · `angry` · `anxious` · `surprised` · `neutral` · `disgusted` · `fearful`
    """,
    version="1.0.0",
    contact={"name": "MoodPet Team"},
    lifespan=lifespan,
)

# ── Middleware ──────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({duration}ms)")
    return response


# ── Routers ─────────────────────────────────────────────────────────────────
app.include_router(emotion.router, prefix="/api/v1/emotion", tags=["Emotion Detection"])
app.include_router(history.router, prefix="/api/v1/history", tags=["Emotional History"])
app.include_router(alerts.router, prefix="/api/v1/alerts", tags=["Alerts"])
app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["Dashboard"])
app.include_router(ai_chat.router, prefix="/api/v1/chat", tags=["AI Chat"])
app.include_router(tools.router, prefix="/api/v1/tools", tags=["Tools"])


# ── Health Check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "service": "MoodPet API",
        "version": "1.0.0",
        "timestamp": time.time(),
    }


@app.get("/", tags=["System"])
async def root():
    return {
        "message": "🐾 MoodPet API is running",
        "docs": "/docs",
        "health": "/health",
    }


# ── Global Exception Handler ─────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )

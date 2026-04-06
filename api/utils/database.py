"""Database initialization and connection management."""

import os
import databases
import sqlalchemy
from pydantic_settings import BaseSettings

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./moodpet.db")

database = databases.Database(DATABASE_URL)

metadata = sqlalchemy.MetaData()

# ── Tables ────────────────────────────────────────────────────────────────────

emotion_events = sqlalchemy.Table(
    "emotion_events",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
    sqlalchemy.Column("session_id", sqlalchemy.String(64), nullable=False, index=True),
    sqlalchemy.Column("user_id", sqlalchemy.String(64), nullable=True, index=True),
    sqlalchemy.Column("emotion", sqlalchemy.String(32), nullable=False),
    sqlalchemy.Column("confidence", sqlalchemy.Float, nullable=False),
    sqlalchemy.Column("secondary_emotion", sqlalchemy.String(32), nullable=True),
    sqlalchemy.Column("all_scores", sqlalchemy.JSON, nullable=True),
    sqlalchemy.Column("face_detected", sqlalchemy.Boolean, default=True),
    sqlalchemy.Column("source", sqlalchemy.String(16), default="mobile"),  # mobile | iot
    sqlalchemy.Column("timestamp", sqlalchemy.Float, nullable=False, index=True),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, server_default=sqlalchemy.func.now()),
)

alerts_log = sqlalchemy.Table(
    "alerts_log",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
    sqlalchemy.Column("session_id", sqlalchemy.String(64), nullable=False),
    sqlalchemy.Column("user_id", sqlalchemy.String(64), nullable=True),
    sqlalchemy.Column("alert_type", sqlalchemy.String(32), nullable=False),
    sqlalchemy.Column("emotion", sqlalchemy.String(32), nullable=True),
    sqlalchemy.Column("message", sqlalchemy.Text, nullable=True),
    sqlalchemy.Column("sent_to", sqlalchemy.String(256), nullable=True),
    sqlalchemy.Column("resolved", sqlalchemy.Boolean, default=False),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, server_default=sqlalchemy.func.now()),
)

user_settings = sqlalchemy.Table(
    "user_settings",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
    sqlalchemy.Column("user_id", sqlalchemy.String(64), nullable=False, unique=True),
    sqlalchemy.Column("pet_type", sqlalchemy.String(32), default="dog"),
    sqlalchemy.Column("pet_name", sqlalchemy.String(64), default="Bolinha"),
    sqlalchemy.Column("emergency_contacts", sqlalchemy.JSON, default=list),
    sqlalchemy.Column("music_enabled", sqlalchemy.Boolean, default=True),
    sqlalchemy.Column("alerts_enabled", sqlalchemy.Boolean, default=True),
    sqlalchemy.Column("no_face_alert_minutes", sqlalchemy.Integer, default=10),
    sqlalchemy.Column("preferred_music_mood", sqlalchemy.String(32), default="calm"),
    sqlalchemy.Column("updated_at", sqlalchemy.DateTime, server_default=sqlalchemy.func.now()),
)

engine = sqlalchemy.create_engine(
    DATABASE_URL.replace("+aiosqlite", ""),
    connect_args={"check_same_thread": False},
)


async def init_db():
    metadata.create_all(engine)
    await database.connect()


async def close_db():
    await database.disconnect()

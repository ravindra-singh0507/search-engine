"""
Search Engine — Main Entry Point (Phase 8)

Run with: python main.py
API docs: http://localhost:8000/docs

Docker:   docker-compose up
"""

import logging
import os
import sys
from pathlib import Path

import uvicorn

from app.api.routes import create_app
from app.config import (
    EngineConfig, DatabaseConfig, PostgresConfig, RedisConfig, EventConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/search_engine.log", mode="a"),
    ],
)

Path("data").mkdir(exist_ok=True)
Path("documents").mkdir(exist_ok=True)

config = EngineConfig(
    database=DatabaseConfig(
        db_path=Path("data/search_engine.db"),
        backend=os.environ.get("DATABASE_BACKEND", "sqlite"),
    ),
    postgres=PostgresConfig(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        database=os.environ.get("POSTGRES_DB", "search_engine"),
        user=os.environ.get("POSTGRES_USER", "search_engine"),
        password=os.environ.get("POSTGRES_PASSWORD", ""),
    ),
    events=EventConfig(
        backend=os.environ.get("EVENT_BACKEND", "memory"),
    ),
    redis=RedisConfig(
        host=os.environ.get("REDIS_HOST", "localhost"),
        port=int(os.environ.get("REDIS_PORT", "6379")),
        password=os.environ.get("REDIS_PASSWORD", ""),
    ),
)

app = create_app(config)

if __name__ == "__main__":
    dev_mode = os.environ.get("ENV", "development").lower() == "development"
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=dev_mode,
    )

"""
Search Engine — Main Entry Point

Run with: python main.py
API docs: http://localhost:8000/docs
"""

import logging
import sys
from pathlib import Path

import uvicorn

from app.api.routes import create_app
from app.config import EngineConfig, DatabaseConfig

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
    database=DatabaseConfig(db_path=Path("data/search_engine.db")),
)

app = create_app(config)

if __name__ == "__main__":
    import os
    dev_mode = os.environ.get("ENV", "development").lower() == "development"
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=dev_mode,   # True in dev, False in production (ENV=production)
    )

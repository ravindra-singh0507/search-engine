"""
Search Engine — Main Entry Point (Phase 8.5)

Run with: python main.py
API docs: http://localhost:8000/docs

Docker:   docker-compose up

Environment variables for distributed mode:
  DATABASE_BACKEND=postgres      # sqlite (default) or postgres
  EVENT_BACKEND=kafka            # memory (default) or kafka
  VECTOR_BACKEND=qdrant          # faiss (default) or qdrant
  CRAWLER_MODE=distributed       # single (default) or distributed
  AGENT_MODE=distributed         # local (default) or distributed
  SECURITY_ENABLED=true          # false (default) or true
  TENANCY_ENABLED=true           # false (default) or true
"""

import logging
import os
import sys
from pathlib import Path

import uvicorn

from app.api.routes import create_app
from app.config import (
    EngineConfig, DatabaseConfig, PostgresConfig, RedisConfig, EventConfig,
    CrawlerConfig, VectorStoreConfig, AgentExecutionConfig,
    SecurityConfig, TenancyConfig, KafkaConfig,
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
    kafka=KafkaConfig(
        bootstrap_servers=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
    ),
    vector_store=VectorStoreConfig(
        backend=os.environ.get("VECTOR_BACKEND", "faiss"),
    ),
    crawler=CrawlerConfig(
        mode=os.environ.get("CRAWLER_MODE", "single"),
    ),
    agent_execution=AgentExecutionConfig(
        mode=os.environ.get("AGENT_MODE", "local"),
    ),
    security=SecurityConfig(
        enabled=os.environ.get("SECURITY_ENABLED", "false").lower() == "true",
    ),
    tenancy=TenancyConfig(
        enabled=os.environ.get("TENANCY_ENABLED", "false").lower() == "true",
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

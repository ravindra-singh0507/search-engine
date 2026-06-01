"""
Search Engine Configuration

Central configuration for all engine components.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatabaseConfig:
    db_path: Path = Path("data/search_engine.db")


@dataclass
class TokenizerConfig:
    min_token_length: int = 2
    max_token_length: int = 50
    custom_stop_words: list[str] = field(default_factory=list)


@dataclass
class IndexerConfig:
    batch_size: int = 100


@dataclass
class CrawlerConfig:
    max_depth: int = 3
    max_pages: int = 100
    request_delay: float = 1.0
    timeout: int = 10
    user_agent: str = "SearchEngineBot/1.0"
    respect_robots_txt: bool = True


@dataclass
class SearchConfig:
    default_top_k: int = 10
    max_top_k: int = 100


@dataclass
class EngineConfig:
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    indexer: IndexerConfig = field(default_factory=IndexerConfig)
    crawler: CrawlerConfig = field(default_factory=CrawlerConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    documents_dir: Path = Path("documents")

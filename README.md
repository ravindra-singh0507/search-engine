# Search Engine — Built from Scratch

A fully functional search engine built from scratch in Python. No Elasticsearch, no Solr, no Whoosh

## What This Project Covers

### Phase 1 — Local Document Search
- **Tokenizer**: Lowercasing, punctuation removal, stop words, configurable stopword lists
- **Inverted Index**: Maps terms to document posting lists with positions and frequencies
- **Boolean Retrieval**: AND, OR, NOT operators on posting lists
- **TF-IDF Ranking**: Term frequency, inverse document frequency, cosine similarity
- **Search Service**: Query parsing → retrieval → ranking → top-k results
- **SQLite Storage**: Documents, terms, and postings tables
- **REST API**: FastAPI endpoints for indexing, searching, and statistics

### Phase 2 — Web Crawler
- **BFS Crawling**: Breadth-first traversal from seed URLs
- **URL Frontier**: Queue with visited set and URL normalization
- **robots.txt**: Fetches and respects robots exclusion protocol
- **Content Extraction**: Title, body text, metadata, outgoing links (via BeautifulSoup)
- **Automatic Indexing**: Every crawled page flows through the Phase 1 indexing pipeline
- **Crawl Controls**: Depth limit, page limit, domain restriction, politeness delay

## Project Structure

```
search_engine/
├── app/
│   ├── api/
│   │   └── routes.py          # FastAPI endpoints
│   ├── crawler/
│   │   ├── crawler.py         # BFS web crawler
│   │   ├── robots.py          # robots.txt parser
│   │   └── url_normalize.py   # URL normalization
│   ├── database/
│   │   └── db.py              # SQLite storage layer
│   ├── indexer/
│   │   └── indexer.py         # Inverted index builder
│   ├── parser/
│   │   └── query_parser.py    # Query parser + Boolean retrieval
│   ├── ranking/
│   │   └── tfidf.py           # TF-IDF + cosine similarity
│   ├── search/
│   │   └── search_service.py  # Search orchestration
│   ├── tokenizer/
│   │   └── tokenizer.py       # Text tokenizer
│   └── config.py              # Configuration dataclasses
├── documents/                  # Sample text files for indexing
├── data/                       # SQLite database + logs
├── tests/                      # Pytest test suite
├── main.py                     # Application entry point
├── requirements.txt
└── README.md
```

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the server

```bash
python main.py
```

The API runs at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

## Usage

### Index local documents

Index all `.txt` files in the `documents/` folder:

```bash
curl -X POST http://localhost:8000/index/directory \
  -H "Content-Type: application/json" \
  -d '{"directory": "documents"}'
```

Index a single document via API:

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{
    "title": "My Document",
    "content": "Python is a powerful programming language used for web development."
  }'
```

### Search

Simple search:
```bash
curl "http://localhost:8000/search?q=python"
```

Boolean search:
```bash
curl "http://localhost:8000/search?q=python+AND+web"
curl "http://localhost:8000/search?q=python+OR+java"
curl "http://localhost:8000/search?q=python+NOT+java"
```

Top-k results:
```bash
curl "http://localhost:8000/search?q=programming&top_k=5"
```

### Get a document

```bash
curl http://localhost:8000/document/1
```

### View statistics

```bash
curl http://localhost:8000/stats
curl "http://localhost:8000/stats?include_index=true"
```

### Crawl websites

Start a crawl:
```bash
curl -X POST http://localhost:8000/crawl \
  -H "Content-Type: application/json" \
  -d '{
    "seed_urls": ["https://docs.python.org/3/tutorial/index.html"],
    "max_depth": 2,
    "max_pages": 20,
    "stay_on_domain": true
  }'
```

Check crawl status:
```bash
curl http://localhost:8000/crawl/status
```

Crawl statistics:
```bash
curl http://localhost:8000/crawl/stats
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/index` | Index a single document |
| POST | `/index/directory` | Index all .txt files in a directory |
| GET | `/search?q=...` | Search with TF-IDF ranking |
| GET | `/document/{id}` | Get document by ID |
| DELETE | `/document/{id}` | Delete document and its index |
| GET | `/stats` | Engine statistics |
| POST | `/crawl` | Start a web crawl |
| GET | `/crawl/status` | Current crawl status |
| GET | `/crawl/stats` | Crawl statistics |

## Running Tests

```bash
pytest tests/ -v
```

## Key Concepts Implemented

### Inverted Index
Maps each term to the list of documents containing it. Enables O(1) term lookup instead of O(N) full scan.

### TF-IDF
Scores documents by combining:
- **TF** (how often a term appears in a document) — normalized by document length
- **IDF** (how rare a term is across all documents) — log(N/df)
- **Cosine similarity** — angle between query and document vectors in term-space

### Boolean Retrieval
Set operations on posting lists:
- **AND** = intersection (both terms must appear)
- **OR** = union (either term can appear)
- **NOT** = difference (exclude documents with term)

### BFS Crawling
Breadth-first search from seed URLs, discovering pages level by level. BFS finds "important" pages first (those closer to seeds) and naturally supports depth limiting.

### robots.txt
Checks the Robots Exclusion Protocol before crawling each URL. Respects `Disallow`, `Allow`, and `Crawl-delay` directives.

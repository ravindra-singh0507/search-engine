"""Tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.api.routes import create_app
from app.config import EngineConfig, DatabaseConfig


@pytest.fixture
def client(tmp_path):
    config = EngineConfig(database=DatabaseConfig(db_path=tmp_path / "test_api.db"))
    app = create_app(config)
    with TestClient(app) as c:
        yield c


class TestIndexEndpoints:

    def test_index_document(self, client):
        response = client.post("/index", json={
            "title": "Test Doc",
            "content": "Python is a great programming language for web development",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "indexed"
        assert data["doc_id"] == 1
        assert data["terms_indexed"] > 0

    def test_index_empty_content(self, client):
        response = client.post("/index", json={
            "title": "Empty",
            "content": "",
        })
        assert response.status_code == 422

    def test_index_directory_not_found(self, client):
        response = client.post("/index/directory", json={
            "directory": "/nonexistent/path"
        })
        assert response.status_code == 404


class TestSearchEndpoints:

    def test_search(self, client):
        client.post("/index", json={
            "title": "Python Doc",
            "content": "Python is a popular programming language",
        })
        response = client.get("/search", params={"q": "python"})
        assert response.status_code == 200
        data = response.json()
        assert data["total_matches"] > 0
        assert len(data["results"]) > 0

    def test_search_no_results(self, client):
        response = client.get("/search", params={"q": "nonexistent"})
        assert response.status_code == 200
        data = response.json()
        assert data["total_matches"] == 0

    def test_search_boolean(self, client):
        client.post("/index", json={
            "title": "Doc1", "content": "python programming language"
        })
        client.post("/index", json={
            "title": "Doc2", "content": "java programming language"
        })
        response = client.get("/search", params={"q": "python OR java"})
        assert response.status_code == 200
        data = response.json()
        assert data["total_matches"] == 2

    def test_search_with_top_k(self, client):
        for i in range(5):
            client.post("/index", json={
                "title": f"Doc{i}", "content": f"python programming example {i}"
            })
        response = client.get("/search", params={"q": "python", "top_k": 2})
        data = response.json()
        assert len(data["results"]) <= 2


class TestDocumentEndpoints:

    def test_get_document(self, client):
        client.post("/index", json={
            "title": "Test Doc",
            "content": "Some content here",
        })
        response = client.get("/document/1")
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Test Doc"

    def test_get_document_not_found(self, client):
        response = client.get("/document/999")
        assert response.status_code == 404

    def test_delete_document(self, client):
        client.post("/index", json={
            "title": "To Delete",
            "content": "Delete me",
        })
        response = client.delete("/document/1")
        assert response.status_code == 200

        response = client.get("/document/1")
        assert response.status_code == 404


class TestStatsEndpoints:

    def test_get_stats(self, client):
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_documents" in data
        assert "total_terms" in data

    def test_get_stats_with_index(self, client):
        client.post("/index", json={
            "title": "Doc", "content": "python java"
        })
        response = client.get("/stats", params={"include_index": True})
        data = response.json()
        assert data["index_snapshot"] is not None
        assert "python" in data["index_snapshot"]


class TestCrawlEndpoints:

    def test_crawl_status_idle(self, client):
        response = client.get("/crawl/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "idle"

    def test_crawl_stats(self, client):
        response = client.get("/crawl/stats")
        assert response.status_code == 200

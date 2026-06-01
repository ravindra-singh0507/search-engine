"""Tests for the Search Service (end-to-end search)."""


class TestSearchService:

    def test_simple_search(self, indexed_db, search_service):
        result = search_service.search("python")
        assert result.total_matches > 0
        assert len(result.results) > 0
        assert result.search_time_ms >= 0

    def test_multi_term_search(self, indexed_db, search_service):
        result = search_service.search("python programming")
        assert result.total_matches > 0

    def test_boolean_and_search(self, indexed_db, search_service):
        result = search_service.search("python AND web")
        # Should only return docs with both terms
        assert result.total_matches > 0

    def test_boolean_or_search(self, indexed_db, search_service):
        result = search_service.search("python OR java")
        result_python = search_service.search("python")
        result_java = search_service.search("java")
        assert result.total_matches >= max(result_python.total_matches, result_java.total_matches)

    def test_boolean_not_search(self, indexed_db, search_service):
        result_all_python = search_service.search("python")
        result_not_java = search_service.search("python NOT java")
        assert result_not_java.total_matches <= result_all_python.total_matches

    def test_empty_query(self, indexed_db, search_service):
        result = search_service.search("")
        assert result.total_matches == 0
        assert len(result.results) == 0

    def test_no_results(self, indexed_db, search_service):
        result = search_service.search("xyznonexistent")
        assert result.total_matches == 0

    def test_top_k_limit(self, indexed_db, search_service):
        result = search_service.search("python", top_k=1)
        assert len(result.results) <= 1

    def test_results_ranked_by_score(self, indexed_db, search_service):
        result = search_service.search("python")
        for i in range(len(result.results) - 1):
            assert result.results[i].score >= result.results[i + 1].score

    def test_search_time_recorded(self, indexed_db, search_service):
        result = search_service.search("python")
        assert result.search_time_ms > 0

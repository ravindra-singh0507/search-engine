"""Tests for URL Normalization."""

from app.crawler.url_normalize import normalize_url, is_same_domain, extract_domain


class TestURLNormalization:

    def test_basic_url(self):
        assert normalize_url("https://example.com/page") == "https://example.com/page"

    def test_lowercase_scheme_and_host(self):
        assert normalize_url("HTTPS://EXAMPLE.COM/page") == "https://example.com/page"

    def test_remove_fragment(self):
        assert normalize_url("https://example.com/page#section") == "https://example.com/page"

    def test_remove_trailing_slash(self):
        assert normalize_url("https://example.com/page/") == "https://example.com/page"

    def test_keep_root_slash(self):
        result = normalize_url("https://example.com/")
        assert result == "https://example.com/"

    def test_remove_default_port_443(self):
        assert normalize_url("https://example.com:443/page") == "https://example.com/page"

    def test_remove_default_port_80(self):
        assert normalize_url("http://example.com:80/page") == "http://example.com/page"

    def test_keep_non_default_port(self):
        assert normalize_url("https://example.com:8080/page") == "https://example.com:8080/page"

    def test_relative_url_with_base(self):
        result = normalize_url("/about", base_url="https://example.com/page")
        assert result == "https://example.com/about"

    def test_remove_tracking_params(self):
        result = normalize_url("https://example.com/page?utm_source=twitter&id=123")
        assert "utm_source" not in result
        assert "id=123" in result

    def test_sort_query_params(self):
        result = normalize_url("https://example.com/page?b=2&a=1")
        assert result == "https://example.com/page?a=1&b=2"

    def test_invalid_scheme(self):
        assert normalize_url("ftp://example.com") is None
        assert normalize_url("javascript:alert(1)") is None

    def test_empty_url(self):
        assert normalize_url("") is None
        assert normalize_url("   ") is None

    def test_is_same_domain(self):
        assert is_same_domain("https://example.com/a", "https://example.com/b")
        assert not is_same_domain("https://example.com", "https://other.com")

    def test_extract_domain(self):
        assert extract_domain("https://example.com/page") == "example.com"
        assert extract_domain("") is None

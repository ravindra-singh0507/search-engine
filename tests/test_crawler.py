"""Tests for the Crawler (unit tests, no network)."""

from app.crawler.robots import RobotsParser, RobotsData, RobotsRule


class TestRobotsParser:

    def test_parse_disallow(self):
        parser = RobotsParser(user_agent="TestBot")
        content = "User-agent: *\nDisallow: /private/\nDisallow: /api/"
        data = parser._parse_content(content)
        assert len(data.rules) == 2
        assert data.rules[0].path == "/private/"
        assert data.rules[0].allowed is False

    def test_parse_allow(self):
        parser = RobotsParser(user_agent="TestBot")
        content = "User-agent: *\nAllow: /api/public/\nDisallow: /api/"
        data = parser._parse_content(content)
        assert len(data.rules) == 2
        allow_rules = [r for r in data.rules if r.allowed]
        assert len(allow_rules) == 1
        assert allow_rules[0].path == "/api/public/"

    def test_parse_crawl_delay(self):
        parser = RobotsParser()
        content = "User-agent: *\nCrawl-delay: 5"
        data = parser._parse_content(content)
        assert data.crawl_delay == 5.0

    def test_empty_robots(self):
        parser = RobotsParser()
        data = parser._parse_content("")
        assert len(data.rules) == 0

    def test_is_allowed_no_rules(self):
        parser = RobotsParser()
        parser._cache["example.com"] = RobotsData()
        assert parser.is_allowed("https://example.com/anything") is True

    def test_is_allowed_with_disallow(self):
        parser = RobotsParser()
        parser._cache["example.com"] = RobotsData(
            rules=[RobotsRule(path="/private/", allowed=False)]
        )
        assert parser.is_allowed("https://example.com/private/page") is False
        assert parser.is_allowed("https://example.com/public/page") is True

    def test_specific_user_agent(self):
        parser = RobotsParser(user_agent="TestBot")
        content = "User-agent: TestBot\nDisallow: /secret/\n\nUser-agent: *\nDisallow: /"
        data = parser._parse_content(content)
        # Should have rules for TestBot section
        assert any(r.path == "/secret/" for r in data.rules)

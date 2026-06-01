"""Tests for the Query Parser and Boolean Retrieval."""


class TestQueryParser:

    def test_simple_query(self, query_parser):
        parsed = query_parser.parse("python")
        assert len(parsed.terms) == 1
        assert "python" in parsed.terms
        assert not parsed.is_boolean

    def test_multi_term_query(self, query_parser):
        parsed = query_parser.parse("python programming")
        assert len(parsed.terms) == 2
        assert "python" in parsed.terms
        assert "programming" in parsed.terms
        assert not parsed.is_boolean

    def test_and_query(self, query_parser):
        parsed = query_parser.parse("python AND backend")
        assert parsed.is_boolean
        assert "python" in parsed.terms
        assert "backend" in parsed.terms

    def test_or_query(self, query_parser):
        parsed = query_parser.parse("python OR java")
        assert parsed.is_boolean
        assert "python" in parsed.terms
        assert "java" in parsed.terms

    def test_not_query(self, query_parser):
        parsed = query_parser.parse("python NOT java")
        assert parsed.is_boolean
        assert "python" in parsed.terms
        assert "java" in parsed.terms

    def test_empty_query(self, query_parser):
        parsed = query_parser.parse("")
        assert len(parsed.terms) == 0
        assert len(parsed.tokens) == 0

    def test_stop_words_in_query(self, query_parser):
        parsed = query_parser.parse("the python")
        # "the" is a stop word — should be removed
        assert "the" not in parsed.terms
        assert "python" in parsed.terms

    def test_case_insensitive_operators(self, query_parser):
        parsed = query_parser.parse("python and java")
        assert parsed.is_boolean


class TestBooleanRetrieval:

    def test_single_term(self, indexed_db, boolean_retriever):
        from app.parser.query_parser import ParsedQuery, QueryToken
        parsed = ParsedQuery(
            raw_query="python",
            tokens=[QueryToken(term="python")],
            terms=["python"]
        )
        results = boolean_retriever.retrieve(parsed)
        assert len(results) > 0

    def test_and_retrieval(self, indexed_db, boolean_retriever):
        from app.parser.query_parser import ParsedQuery, QueryToken, Operator
        parsed = ParsedQuery(
            raw_query="python AND programming",
            tokens=[
                QueryToken(term="python"),
                QueryToken(operator=Operator.AND),
                QueryToken(term="programming"),
            ],
            terms=["python", "programming"]
        )
        results = boolean_retriever.retrieve(parsed)
        # Only docs with BOTH python and programming
        for doc_id in results:
            python_postings = indexed_db.get_postings_for_term("python")
            programming_postings = indexed_db.get_postings_for_term("programming")
            python_docs = {p.doc_id for p in python_postings}
            programming_docs = {p.doc_id for p in programming_postings}
            assert doc_id in python_docs
            assert doc_id in programming_docs

    def test_or_retrieval(self, indexed_db, boolean_retriever):
        from app.parser.query_parser import ParsedQuery, QueryToken, Operator
        parsed = ParsedQuery(
            raw_query="python OR java",
            tokens=[
                QueryToken(term="python"),
                QueryToken(operator=Operator.OR),
                QueryToken(term="java"),
            ],
            terms=["python", "java"]
        )
        results = boolean_retriever.retrieve(parsed)
        # Should include docs with python OR java
        python_postings = indexed_db.get_postings_for_term("python")
        java_postings = indexed_db.get_postings_for_term("java")
        expected = {p.doc_id for p in python_postings} | {p.doc_id for p in java_postings}
        assert results == expected

    def test_not_retrieval(self, indexed_db, boolean_retriever):
        from app.parser.query_parser import ParsedQuery, QueryToken, Operator
        parsed = ParsedQuery(
            raw_query="programming NOT java",
            tokens=[
                QueryToken(term="programming"),
                QueryToken(operator=Operator.NOT),
                QueryToken(term="java"),
            ],
            terms=["programming", "java"]
        )
        results = boolean_retriever.retrieve(parsed)
        java_postings = indexed_db.get_postings_for_term("java")
        java_docs = {p.doc_id for p in java_postings}
        for doc_id in results:
            assert doc_id not in java_docs

    def test_empty_result(self, indexed_db, boolean_retriever):
        from app.parser.query_parser import ParsedQuery, QueryToken
        parsed = ParsedQuery(
            raw_query="nonexistentterm",
            tokens=[QueryToken(term="nonexistentterm")],
            terms=["nonexistentterm"]
        )
        results = boolean_retriever.retrieve(parsed)
        assert len(results) == 0

"""Tests for the Indexer and Inverted Index."""


class TestIndexer:

    def test_index_single_document(self, indexer):
        result = indexer.index_document("Test Doc", "Python is great for programming")
        assert result.doc_id == 1
        assert result.terms_indexed > 0
        assert result.total_tokens > 0

    def test_document_stored_in_db(self, indexer, db):
        indexer.index_document("Test", "sample content here")
        doc = db.get_document(1)
        assert doc is not None
        assert doc.title == "Test"
        assert "sample content" in doc.content

    def test_inverted_index_built(self, indexer, db):
        indexer.index_document("Doc1", "python programming language")
        indexer.index_document("Doc2", "java programming language")

        python_postings = db.get_postings_for_term("python")
        assert len(python_postings) == 1
        assert python_postings[0].doc_id == 1

        programming_postings = db.get_postings_for_term("programming")
        assert len(programming_postings) == 2

    def test_term_frequencies_stored(self, indexer, db):
        indexer.index_document("Doc", "python python python java java")
        postings = db.get_postings_for_term("python")
        assert postings[0].term_frequency == 3

        postings = db.get_postings_for_term("java")
        assert postings[0].term_frequency == 2

    def test_positions_stored(self, indexer, db):
        indexer.index_document("Doc", "python java python")
        postings = db.get_postings_for_term("python")
        assert len(postings[0].positions) == 2

    def test_document_frequency_updated(self, indexer, db):
        indexer.index_document("Doc1", "python rocks")
        indexer.index_document("Doc2", "python rules")
        indexer.index_document("Doc3", "java rocks")

        term = db.get_term("python")
        assert term.document_frequency == 2

        term = db.get_term("rocks")
        assert term.document_frequency == 2

    def test_index_directory(self, indexer, tmp_path):
        doc_dir = tmp_path / "docs"
        doc_dir.mkdir()
        (doc_dir / "file1.txt").write_text("Python programming language")
        (doc_dir / "file2.txt").write_text("Java programming language")

        results = indexer.index_directory(doc_dir)
        assert len(results) == 2

    def test_skip_already_indexed(self, indexer, tmp_path):
        doc_dir = tmp_path / "docs"
        doc_dir.mkdir()
        (doc_dir / "file1.txt").write_text("Python programming language")

        results1 = indexer.index_directory(doc_dir)
        results2 = indexer.index_directory(doc_dir)
        assert len(results1) == 1
        assert len(results2) == 0

    def test_inverted_index_snapshot(self, indexer):
        indexer.index_document("Doc1", "python programming")
        indexer.index_document("Doc2", "java programming")

        snapshot = indexer.get_inverted_index_snapshot()
        assert "programming" in snapshot
        assert len(snapshot["programming"]) == 2
        assert "python" in snapshot
        assert len(snapshot["python"]) == 1

    def test_reindex_document(self, indexer, db):
        result = indexer.index_document("Doc", "python programming")
        doc_id = result.doc_id

        indexer.reindex_document(doc_id, "java development")

        python_postings = db.get_postings_for_term("python")
        assert len(python_postings) == 0

        java_postings = db.get_postings_for_term("java")
        assert len(java_postings) == 1
        assert java_postings[0].doc_id == doc_id

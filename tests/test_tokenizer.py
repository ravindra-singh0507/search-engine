"""Tests for the Tokenizer."""


class TestTokenizer:

    def test_basic_tokenization(self, tokenizer):
        result = tokenizer.tokenize("Hello World")
        assert "hello" in result.tokens
        assert "world" in result.tokens

    def test_lowercase(self, tokenizer):
        result = tokenizer.tokenize("Python JAVA javascript")
        assert "python" in result.tokens
        assert "java" in result.tokens
        assert "javascript" in result.tokens

    def test_punctuation_removal(self, tokenizer):
        result = tokenizer.tokenize("hello, world! how's it going?")
        assert "hello" in result.tokens
        assert "world" in result.tokens
        # "it" is a stop word, should be removed
        assert "it" not in result.tokens

    def test_stop_word_removal(self, tokenizer):
        result = tokenizer.tokenize("the quick brown fox is a fast animal")
        assert "the" not in result.tokens
        assert "is" not in result.tokens
        assert "a" not in result.tokens
        assert "quick" in result.tokens
        assert "brown" in result.tokens
        assert "fox" in result.tokens

    def test_min_length_filter(self, tokenizer):
        result = tokenizer.tokenize("I am a x y z developer")
        # Single character tokens should be filtered (min_length=2)
        assert "x" not in result.tokens
        assert "y" not in result.tokens
        assert "z" not in result.tokens
        assert "developer" in result.tokens

    def test_positions_tracking(self, tokenizer):
        result = tokenizer.tokenize("python is great python")
        assert "python" in result.positions
        positions = result.positions["python"]
        assert len(positions) == 2
        assert positions[0] < positions[1]

    def test_empty_input(self, tokenizer):
        result = tokenizer.tokenize("")
        assert result.tokens == []
        assert result.token_count == 0

    def test_whitespace_only(self, tokenizer):
        result = tokenizer.tokenize("   \n\t  ")
        assert result.tokens == []

    def test_custom_stop_words(self):
        config = TokenizerConfig(custom_stop_words=["python", "java"])
        tok = Tokenizer(config)
        result = tok.tokenize("python and java are languages")
        assert "python" not in result.tokens
        assert "java" not in result.tokens
        assert "languages" in result.tokens

    def test_term_frequencies(self, tokenizer):
        tokens = ["python", "java", "python", "python", "java"]
        freq = tokenizer.get_term_frequencies(tokens)
        assert freq["python"] == 3
        assert freq["java"] == 2

    def test_token_count_and_unique_count(self, tokenizer):
        result = tokenizer.tokenize("python java python ruby python")
        assert result.token_count == 5
        assert result.unique_count == 3

    def test_alphanumeric_preserved(self, tokenizer):
        result = tokenizer.tokenize("python3 version2 html5")
        assert "python3" in result.tokens
        assert "version2" in result.tokens
        assert "html5" in result.tokens


from app.tokenizer.tokenizer import Tokenizer, TokenizerConfig

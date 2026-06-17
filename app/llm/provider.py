"""
LLM Provider Abstraction Layer

=== THEORY ===

Large Language Models (LLMs) are autoregressive transformer models that predict
the next token given prior context.  In a RAG system the LLM's role is strictly
**reader**: it receives a (retrieved context, question) prompt and generates a
grounded answer.  It does NOT retrieve — retrieval happens before it.

=== ARCHITECTURE PATTERN ===

We define a Protocol (structural subtyping) rather than an abstract base class,
mirroring the EmbeddingProvider design from Phase 4.  Any object that exposes
  generate(prompt) → LLMResponse
  stream(prompt) → Iterator[str]
  count_tokens(text) → int
  model_name → str
satisfies the interface without inheriting from anything.

This enables drop-in substitution:
  provider: LLMProvider = MockLLMProvider()      # tests
  provider: LLMProvider = OllamaProvider(cfg)   # local dev
  provider: LLMProvider = OpenAIProvider(cfg)   # production

=== PROVIDERS IMPLEMENTED ===

  MockLLMProvider    — deterministic, no network, no deps.  Used in all tests.
  OllamaProvider     — calls the Ollama HTTP API (http://localhost:11434).
                       Supports any model pulled with `ollama pull <model>`.
  OpenAIProvider     — OpenAI Chat Completions API (gpt-4o, gpt-3.5-turbo, …).
  AnthropicProvider  — Claude API (claude-sonnet-4-6, claude-haiku-4-5, …).
  GeminiProvider     — Google Gemini API.

All network providers use the `requests` library (already a project dep),
implement retry-with-backoff, and propagate token counts for cost tracking.

=== TOKEN COUNTING ===

Precise token counts require the model's tokenizer (tiktoken for OpenAI,
sentencepiece for Claude).  We use a lightweight word-based approximation
(words × 1.3) as a default; subclasses override for accuracy.

=== STREAMING ===

LLM streaming returns partial tokens via Server-Sent Events (SSE).  Each
provider's stream() is a generator that yields decoded string chunks.  The
caller can forward these directly to FastAPI's StreamingResponse.

=== RETRY LOGIC ===

Transient failures (rate limits, gateway errors) are retried with exponential
backoff.  After max_retries the exception is propagated to the caller.

=== COMPLEXITY ===

  generate(): O(L_prompt + L_completion) — dominated by model inference
  stream():   same compute, different delivery
  count_tokens(): O(len(text)) — approximation
  Network RTT + model latency: 200ms–30s depending on provider and model size

=== AT PRODUCTION SCALE ===

  OpenAI / Anthropic: token-based billing; latency 1–5s for short answers
  Ollama + Llama 3.1 8B: ~20 tokens/s on an A10G GPU
  vLLM with continuous batching: 200+ tokens/s at scale
  Caching identical prompts: ~80% of repeated queries can be cache hits
"""

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Iterator, Protocol, runtime_checkable

import requests

from app.config import LLMConfig

logger = logging.getLogger(__name__)


# ── Response dataclass ────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    text:               str
    model:              str
    prompt_tokens:      int
    completion_tokens:  int
    total_tokens:       int
    latency_ms:         float
    finish_reason:      str = "stop"
    provider:           str = "unknown"
    cost_usd:           float = 0.0


# ── Protocol ──────────────────────────────────────────────────────────────────

@runtime_checkable
class LLMProvider(Protocol):
    """
    Structural interface for any LLM backend.
    All providers MUST be safe to call from a background thread.
    """

    @property
    def model_name(self) -> str: ...

    def generate(self, prompt: str, system: str = "", **kwargs) -> LLMResponse:
        """
        Generate a completion for the given prompt.
        `system` is an optional system prompt (ignored by providers that do
        not support it).
        """
        ...

    def stream(self, prompt: str, system: str = "", **kwargs) -> Iterator[str]:
        """Yield decoded token strings as they are generated."""
        ...

    def count_tokens(self, text: str) -> int:
        """Approximate token count for the given text."""
        ...


# ── Shared retry helper ───────────────────────────────────────────────────────

def _with_retry(fn, max_retries: int = 3, base_delay: float = 1.0):
    """Call fn(), retrying on requests.RequestException with exponential backoff."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn()
        except (requests.RequestException, requests.Timeout) as exc:
            last_exc = exc
            wait = base_delay * (2 ** attempt)
            logger.warning("LLM request failed (attempt %d/%d): %s — retry in %.1fs",
                           attempt + 1, max_retries, exc, wait)
            time.sleep(wait)
    raise last_exc


# ── Mock provider ─────────────────────────────────────────────────────────────

class MockLLMProvider:
    """
    Deterministic LLM for unit tests.  No network, no dependencies.

    The generated text is a template that references the first 3 sentences
    of the context and echoes the query — predictable enough for assertions.
    """

    def __init__(self, model: str = "mock-llm-v1"):
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, prompt: str, system: str = "", **kwargs) -> LLMResponse:
        t0 = time.perf_counter()
        text = self._mock_answer(prompt)
        latency = (time.perf_counter() - t0) * 1000
        pt = self.count_tokens(prompt)
        ct = self.count_tokens(text)
        return LLMResponse(
            text=text, model=self._model,
            prompt_tokens=pt, completion_tokens=ct,
            total_tokens=pt + ct, latency_ms=round(latency, 2),
            finish_reason="stop", provider="mock",
        )

    def stream(self, prompt: str, system: str = "", **kwargs) -> Iterator[str]:
        text = self._mock_answer(prompt)
        chunk_size = kwargs.get("chunk_size", 8)
        for i in range(0, len(text), chunk_size):
            yield text[i: i + chunk_size]

    def count_tokens(self, text: str) -> int:
        return max(1, int(len(text.split()) * 1.3))

    @staticmethod
    def _mock_answer(prompt: str) -> str:
        """Extract the first context sentence and echo the query."""
        lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
        context_line = ""
        query_line   = ""
        for i, ln in enumerate(lines):
            if ln.lower().startswith("context:") or ln.lower().startswith("### context"):
                if i + 1 < len(lines):
                    context_line = lines[i + 1][:120]
            if ln.lower().startswith("question:") or ln.lower().startswith("query:"):
                query_line = ln.split(":", 1)[-1].strip()[:80]
        if not query_line:
            query_line = lines[-1][:80] if lines else "the question"
        answer = f"Based on the available context, {query_line}. "
        if context_line:
            answer += f"{context_line}. "
        answer += "This answer is generated by the mock LLM for testing purposes."
        return answer


# ── Ollama provider ───────────────────────────────────────────────────────────

class OllamaProvider:
    """
    Calls the Ollama REST API running locally (default: http://localhost:11434).

    Setup:
      1. Install Ollama: https://ollama.com
      2. Pull a model: ollama pull llama3.2
      3. Set config.rag.llm = LLMConfig(provider="ollama", model_name="llama3.2")

    Ollama supports the same Chat API shape as OpenAI, so messages=[…] works.
    """

    def __init__(self, config: LLMConfig):
        self._cfg   = config
        self._base  = config.base_url.rstrip("/")

    @property
    def model_name(self) -> str:
        return self._cfg.model_name

    def generate(self, prompt: str, system: str = "", **kwargs) -> LLMResponse:
        t0 = time.perf_counter()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model":   self._cfg.model_name,
            "messages": messages,
            "stream":  False,
            "options": {
                "temperature": kwargs.get("temperature", self._cfg.temperature),
                "num_predict": kwargs.get("max_tokens", self._cfg.max_tokens),
            },
        }

        def _call():
            return requests.post(
                f"{self._base}/api/chat",
                json=payload,
                timeout=self._cfg.timeout,
            )

        resp = _with_retry(_call, self._cfg.max_retries)
        resp.raise_for_status()
        data = resp.json()

        text   = data.get("message", {}).get("content", "")
        pt     = data.get("prompt_eval_count", self.count_tokens(prompt))
        ct     = data.get("eval_count",         self.count_tokens(text))
        ms     = (time.perf_counter() - t0) * 1000

        logger.debug("Ollama %s: %d tokens in %.0f ms", self._cfg.model_name, pt + ct, ms)
        return LLMResponse(
            text=text, model=self._cfg.model_name,
            prompt_tokens=pt, completion_tokens=ct,
            total_tokens=pt + ct, latency_ms=round(ms, 2),
            finish_reason=data.get("done_reason", "stop"),
            provider="ollama",
        )

    def stream(self, prompt: str, system: str = "", **kwargs) -> Iterator[str]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model":   self._cfg.model_name,
            "messages": messages,
            "stream":  True,
            "options": {
                "temperature": kwargs.get("temperature", self._cfg.temperature),
                "num_predict": kwargs.get("max_tokens", self._cfg.max_tokens),
            },
        }
        with requests.post(
            f"{self._base}/api/chat",
            json=payload,
            stream=True,
            timeout=self._cfg.timeout,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue

    def count_tokens(self, text: str) -> int:
        return max(1, int(len(text.split()) * 1.3))


# ── OpenAI provider ───────────────────────────────────────────────────────────

class OpenAIProvider:
    """
    OpenAI Chat Completions API.  Requires OPENAI_API_KEY (or the env var
    specified in config.api_key_env).

    Models: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo

    Cost estimation uses OpenAI's public pricing (approximate; verify on dashboard).
    """

    _PRICE_PER_1K = {   # input, output (USD)
        "gpt-4o":         (0.005,  0.015),
        "gpt-4o-mini":    (0.00015, 0.0006),
        "gpt-4-turbo":    (0.01,   0.03),
        "gpt-3.5-turbo":  (0.0005, 0.0015),
    }

    def __init__(self, config: LLMConfig):
        import os
        self._cfg  = config
        key_env    = config.api_key_env or "OPENAI_API_KEY"
        self._key  = os.environ.get(key_env, "")
        if not self._key:
            logger.warning("OpenAIProvider: %s not set — requests will fail", key_env)
        self._base = "https://api.openai.com/v1"

    @property
    def model_name(self) -> str:
        return self._cfg.model_name

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._key}",
            "Content-Type":  "application/json",
        }

    def generate(self, prompt: str, system: str = "", **kwargs) -> LLMResponse:
        t0 = time.perf_counter()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model":       self._cfg.model_name,
            "messages":    messages,
            "max_tokens":  kwargs.get("max_tokens", self._cfg.max_tokens),
            "temperature": kwargs.get("temperature", self._cfg.temperature),
        }

        def _call():
            return requests.post(
                f"{self._base}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=self._cfg.timeout,
            )

        resp = _with_retry(_call, self._cfg.max_retries)
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        text   = choice["message"]["content"]
        usage  = data.get("usage", {})
        pt     = usage.get("prompt_tokens", self.count_tokens(prompt))
        ct     = usage.get("completion_tokens", self.count_tokens(text))
        ms     = (time.perf_counter() - t0) * 1000
        cost   = self._estimate_cost(pt, ct)

        return LLMResponse(
            text=text, model=self._cfg.model_name,
            prompt_tokens=pt, completion_tokens=ct,
            total_tokens=pt + ct, latency_ms=round(ms, 2),
            finish_reason=choice.get("finish_reason", "stop"),
            provider="openai", cost_usd=cost,
        )

    def stream(self, prompt: str, system: str = "", **kwargs) -> Iterator[str]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model":       self._cfg.model_name,
            "messages":    messages,
            "max_tokens":  kwargs.get("max_tokens", self._cfg.max_tokens),
            "temperature": kwargs.get("temperature", self._cfg.temperature),
            "stream":      True,
        }
        with requests.post(
            f"{self._base}/chat/completions",
            headers=self._headers(), json=payload,
            stream=True, timeout=self._cfg.timeout,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or line == b"data: [DONE]":
                    continue
                raw = line.decode("utf-8").removeprefix("data: ")
                try:
                    chunk = json.loads(raw)
                    delta = chunk["choices"][0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        yield token
                except (json.JSONDecodeError, KeyError):
                    continue

    def count_tokens(self, text: str) -> int:
        return max(1, int(len(text.split()) * 1.35))

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        prices = self._PRICE_PER_1K.get(self._cfg.model_name, (0.0, 0.0))
        return round(
            (prompt_tokens / 1000) * prices[0] +
            (completion_tokens / 1000) * prices[1], 6
        )


# ── Anthropic provider ────────────────────────────────────────────────────────

class AnthropicProvider:
    """
    Anthropic Messages API.  Requires ANTHROPIC_API_KEY.

    Models: claude-sonnet-4-6, claude-haiku-4-5, claude-opus-4-8
    """

    def __init__(self, config: LLMConfig):
        import os
        self._cfg  = config
        key_env    = config.api_key_env or "ANTHROPIC_API_KEY"
        self._key  = os.environ.get(key_env, "")
        if not self._key:
            logger.warning("AnthropicProvider: %s not set", key_env)
        self._base = "https://api.anthropic.com/v1"

    @property
    def model_name(self) -> str:
        return self._cfg.model_name

    def _headers(self) -> dict:
        return {
            "x-api-key":         self._key,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        }

    def generate(self, prompt: str, system: str = "", **kwargs) -> LLMResponse:
        t0 = time.perf_counter()
        payload: dict = {
            "model":      self._cfg.model_name,
            "max_tokens": kwargs.get("max_tokens", self._cfg.max_tokens),
            "messages":   [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system

        def _call():
            return requests.post(
                f"{self._base}/messages",
                headers=self._headers(), json=payload,
                timeout=self._cfg.timeout,
            )

        resp = _with_retry(_call, self._cfg.max_retries)
        resp.raise_for_status()
        data = resp.json()

        text = data["content"][0]["text"]
        usage = data.get("usage", {})
        pt   = usage.get("input_tokens",  self.count_tokens(prompt))
        ct   = usage.get("output_tokens", self.count_tokens(text))
        ms   = (time.perf_counter() - t0) * 1000

        return LLMResponse(
            text=text, model=self._cfg.model_name,
            prompt_tokens=pt, completion_tokens=ct,
            total_tokens=pt + ct, latency_ms=round(ms, 2),
            finish_reason=data.get("stop_reason", "end_turn"),
            provider="anthropic",
        )

    def stream(self, prompt: str, system: str = "", **kwargs) -> Iterator[str]:
        payload: dict = {
            "model":      self._cfg.model_name,
            "max_tokens": kwargs.get("max_tokens", self._cfg.max_tokens),
            "messages":   [{"role": "user", "content": prompt}],
            "stream":     True,
        }
        if system:
            payload["system"] = system

        with requests.post(
            f"{self._base}/messages",
            headers=self._headers(), json=payload,
            stream=True, timeout=self._cfg.timeout,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                raw = line.decode("utf-8")
                if raw.startswith("data:"):
                    raw = raw[5:].strip()
                try:
                    evt = json.loads(raw)
                    if evt.get("type") == "content_block_delta":
                        delta = evt.get("delta", {})
                        token = delta.get("text", "")
                        if token:
                            yield token
                except json.JSONDecodeError:
                    continue

    def count_tokens(self, text: str) -> int:
        return max(1, int(len(text.split()) * 1.3))


# ── Gemini provider ───────────────────────────────────────────────────────────

class GeminiProvider:
    """
    Google Generative Language API (Gemini).  Requires GEMINI_API_KEY.

    Models: gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash
    """

    def __init__(self, config: LLMConfig):
        import os
        self._cfg = config
        key_env   = config.api_key_env or "GEMINI_API_KEY"
        self._key = os.environ.get(key_env, "")
        if not self._key:
            logger.warning("GeminiProvider: %s not set", key_env)
        self._base = "https://generativelanguage.googleapis.com/v1beta"

    @property
    def model_name(self) -> str:
        return self._cfg.model_name

    def _headers(self) -> dict:
        return {
            "Content-Type":   "application/json",
            "x-goog-api-key": self._key,
        }

    def generate(self, prompt: str, system: str = "", **kwargs) -> LLMResponse:
        t0 = time.perf_counter()
        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": kwargs.get("max_tokens", self._cfg.max_tokens),
                "temperature":     kwargs.get("temperature", self._cfg.temperature),
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        url = f"{self._base}/models/{self._cfg.model_name}:generateContent"

        def _call():
            return requests.post(url, headers=self._headers(),
                                 json=payload, timeout=self._cfg.timeout)

        resp = _with_retry(_call, self._cfg.max_retries)
        resp.raise_for_status()
        data = resp.json()

        candidate = data["candidates"][0]
        text = candidate["content"]["parts"][0]["text"]
        usage = data.get("usageMetadata", {})
        pt = usage.get("promptTokenCount", self.count_tokens(prompt))
        ct = usage.get("candidatesTokenCount", self.count_tokens(text))
        ms = (time.perf_counter() - t0) * 1000

        return LLMResponse(
            text=text, model=self._cfg.model_name,
            prompt_tokens=pt, completion_tokens=ct,
            total_tokens=pt + ct, latency_ms=round(ms, 2),
            finish_reason=candidate.get("finishReason", "STOP").lower(),
            provider="gemini",
        )

    def stream(self, prompt: str, system: str = "", **kwargs) -> Iterator[str]:
        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": kwargs.get("max_tokens", self._cfg.max_tokens),
                "temperature":     kwargs.get("temperature", self._cfg.temperature),
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        url = (f"{self._base}/models/{self._cfg.model_name}:streamGenerateContent"
               f"?alt=sse")

        with requests.post(url, headers=self._headers(), json=payload, stream=True,
                           timeout=self._cfg.timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                raw = line.decode("utf-8").removeprefix("data: ")
                try:
                    evt = json.loads(raw)
                    text = (evt.get("candidates", [{}])[0]
                              .get("content", {})
                              .get("parts", [{}])[0]
                              .get("text", ""))
                    if text:
                        yield text
                except (json.JSONDecodeError, IndexError):
                    continue

    def count_tokens(self, text: str) -> int:
        return max(1, int(len(text.split()) * 1.3))


# ── Factory ───────────────────────────────────────────────────────────────────

def create_llm_provider(config: LLMConfig) -> LLMProvider:
    """
    Instantiate the correct provider from LLMConfig.provider.

    Raises ValueError for unknown provider names.
    """
    name = config.provider.lower()
    if name == "mock":
        return MockLLMProvider(config.model_name)
    if name == "ollama":
        return OllamaProvider(config)
    if name == "openai":
        return OpenAIProvider(config)
    if name == "anthropic":
        return AnthropicProvider(config)
    if name == "gemini":
        return GeminiProvider(config)
    raise ValueError(
        f"Unknown LLM provider {config.provider!r}. "
        "Available: mock, ollama, openai, anthropic, gemini"
    )

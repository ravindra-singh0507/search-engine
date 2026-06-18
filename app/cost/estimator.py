"""
Cost Estimator — Phase 8 Batch 4

=== THEORY ===

Before a request is made, the platform can *predict* its cost so that it can:
  - Choose the cheapest model that meets a quality bar (cost-quality trade-off)
  - Reject requests that would blow a per-tenant budget
  - Surface cost forecasts in the developer dashboard

The key abstraction is a **price catalogue** — a static mapping from
(provider, model) to per-1K-token prices for input and output.  Real systems
poll a billing API or refresh a cached catalogue periodically; here we use a
hardcoded table that is trivially replaced.

**cheapest_model** implements a **cost-quality Pareto filter**:
  1. Discard models below the minimum quality tier.
  2. Among the remaining candidates, return the one with the lowest per-1K
     *output* token price (output tokens dominate cost for typical
     question-answering workloads).

Quality tiers follow the common industry convention:
  "low"    → small/fast models  (Haiku, GPT-4o-mini)
  "medium" → balanced models    (Sonnet, GPT-4o)
  "high"   → frontier models    (Opus, GPT-4o)

=== PRODUCTION EQUIVALENTS ===

  OpenAI pricing page / /v1/models endpoint for model metadata
  Anthropic pricing page (per-model rates)
  AWS Bedrock:  on-demand pricing per model in the console
  LiteLLM:      open-source library that unifies pricing across providers
"""

from __future__ import annotations

from app.config import CostConfig


# ── Price catalogue ────────────────────────────────────────────────────────────
# Structure: { provider: { model: (input_per_1k_usd, output_per_1k_usd) } }
#
# Prices are approximate list prices as of mid-2025; actual costs depend on
# prompt caching, batch API discounts, volume tiers, and region.

PROVIDER_PRICES: dict[str, dict[str, tuple[float, float]]] = {
    "openai": {
        "gpt-4o":                (0.005,   0.015),
        "gpt-4o-mini":           (0.00015, 0.0006),
        "text-embedding-3-small": (0.00002, 0.0),   # embeddings: no output cost
    },
    "anthropic": {
        "claude-sonnet-4-6":  (0.003,  0.015),
        "claude-haiku-4-5":   (0.0008, 0.004),
    },
}

# Quality tier definitions: each tier lists (provider, model) tuples that
# qualify at that quality level.  Higher tiers include lower-tier models too
# (a "medium" filter accepts both "medium" and "low" would be wrong; here we
# mean *minimum* quality, so the filter keeps everything >= the requested tier).
_QUALITY_TIERS: dict[str, int] = {
    "low":    1,
    "medium": 2,
    "high":   3,
}

# Map each (provider, model) to its quality tier score
_MODEL_QUALITY: dict[tuple[str, str], int] = {
    ("openai",     "gpt-4o-mini"):           1,  # low
    ("openai",     "text-embedding-3-small"): 1,  # low (embeddings)
    ("anthropic",  "claude-haiku-4-5"):      1,  # low
    ("openai",     "gpt-4o"):                2,  # medium
    ("anthropic",  "claude-sonnet-4-6"):     2,  # medium
    # "high" tier would contain Opus / GPT-4o (same model at higher score)
    # Add opus here if/when it enters the price catalogue
}


class CostEstimator:
    """
    Pre-flight cost estimation for LLM and embedding API calls.

    Parameters
    ----------
    config : CostConfig
        Platform cost configuration (currently used for future per-tenant
        override support; the estimator is stateless for now).
    """

    def __init__(self, config: CostConfig) -> None:
        self._config = config

    # ── Estimation helpers ─────────────────────────────────────────────────────

    def estimate_llm(
        self,
        provider:      str,
        model:         str,
        input_tokens:  int,
        output_tokens: int,
    ) -> float:
        """
        Estimate the total cost (USD) of an LLM call.

        Returns 0.0 if the (provider, model) pair is not in the price catalogue
        rather than raising, so callers can always safely use the estimate.

        Calculation
        -----------
        cost = (input_tokens / 1000) * input_price
             + (output_tokens / 1000) * output_price
        """
        prices = PROVIDER_PRICES.get(provider, {}).get(model)
        if prices is None:
            return 0.0
        input_price, output_price = prices
        return (input_tokens / 1000.0) * input_price + \
               (output_tokens / 1000.0) * output_price

    def estimate_embedding(
        self,
        provider: str,
        model:    str,
        tokens:   int,
    ) -> float:
        """
        Estimate the cost (USD) of an embedding API call.

        For embedding models the output_price is 0.0 (no completion tokens).
        """
        prices = PROVIDER_PRICES.get(provider, {}).get(model)
        if prices is None:
            return 0.0
        input_price, _ = prices
        return (tokens / 1000.0) * input_price

    # ── Model selection ────────────────────────────────────────────────────────

    def cheapest_model(
        self,
        providers:   list[str],
        min_quality: str = "medium",
    ) -> tuple[str, str]:
        """
        Return the (provider, model) tuple with the lowest output-token price
        among models that meet *min_quality* and belong to one of *providers*.

        Quality tiers (minimum accepted)
        ---------------------------------
        "low"    → haiku, gpt-4o-mini (and any model in the price catalogue)
        "medium" → sonnet, gpt-4o
        "high"   → frontier-only (currently no explicit "high" models in the
                   catalogue; falls back to "medium" candidates)

        Raises
        ------
        ValueError
            If no qualifying model is found for the requested providers and
            quality combination.
        """
        min_tier = _QUALITY_TIERS.get(min_quality, 2)
        provider_set = set(providers)

        candidates: list[tuple[float, str, str]] = []  # (output_price, provider, model)
        for provider, models in PROVIDER_PRICES.items():
            if provider not in provider_set:
                continue
            for model, (_, output_price) in models.items():
                tier = _MODEL_QUALITY.get((provider, model), 1)
                if tier >= min_tier:
                    candidates.append((output_price, provider, model))

        if not candidates:
            raise ValueError(
                f"No model found for providers={providers!r} "
                f"with min_quality={min_quality!r}"
            )

        # Sort by output price ascending, then provider+model for determinism
        candidates.sort(key=lambda t: (t[0], t[1], t[2]))
        _, best_provider, best_model = candidates[0]
        return best_provider, best_model

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """
        Return metadata about the loaded price catalogue.

        Keys
        ----
        providers        : list of provider names
        models_per_provider : {provider: [model, …]}
        total_models     : total number of (provider, model) pairs
        """
        models_per_provider = {
            p: list(m.keys()) for p, m in PROVIDER_PRICES.items()
        }
        return {
            "providers":            list(PROVIDER_PRICES.keys()),
            "models_per_provider":  models_per_provider,
            "total_models":         sum(
                len(m) for m in PROVIDER_PRICES.values()
            ),
        }

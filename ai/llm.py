"""DeepSeek LLM client with strict budget and resilience controls."""

from __future__ import annotations

import hashlib
import logging
import os
import time
from datetime import datetime
from typing import Optional

try:
    from openai import APIError, APITimeoutError, OpenAI, RateLimitError
except ImportError:  # pragma: no cover - handled gracefully at runtime
    APIError = Exception
    APITimeoutError = Exception
    RateLimitError = Exception
    OpenAI = None

from ai.prompts import (
    SYSTEM_AUDIT,
    SYSTEM_ANOMALY_CHECK,
    SYSTEM_MATCH_ANALYSIS,
    SYSTEM_TOP_PICKS,
    build_anomaly_prompt,
    build_audit_prompt,
    build_match_analysis_prompt,
    build_top_picks_prompt,
)

logger = logging.getLogger(__name__)


class DeepSeekClient:
    """LLM wrapper with monthly budget guard, cache, and fallback semantics."""

    def __init__(self, config: dict):
        llm_cfg = config.get("llm", {})
        self.enabled = bool(llm_cfg.get("enabled", False))
        self.provider = str(llm_cfg.get("provider", "deepseek"))
        self.base_url = str(llm_cfg.get("base_url", "https://api.deepseek.com"))
        self.model = str(llm_cfg.get("model", "deepseek-chat"))
        self.max_tokens = max(1, int(llm_cfg.get("max_tokens", 400)))
        self.temperature = float(llm_cfg.get("temperature", 0.3))
        self.timeout_sec = max(1, int(llm_cfg.get("timeout_sec", 25)))
        self.retry_count = max(0, int(llm_cfg.get("retry_count", 2)))
        self.retry_backoff = max(1, int(llm_cfg.get("retry_backoff_sec", 3)))
        self.fallback = bool(llm_cfg.get("fallback_to_template", True))
        self.cache_ttl = max(1, int(llm_cfg.get("cache_ttl_minutes", 60)))
        self.max_calls = max(1, int(llm_cfg.get("max_calls_per_cycle", 5)))
        self.monthly_budget = max(0.0, float(llm_cfg.get("monthly_budget_usd", 2.0)))
        self.skip_unchanged = bool(llm_cfg.get("skip_unchanged_picks", True))
        self.anomaly_check_enabled = bool(llm_cfg.get("llm_anomaly_check_enabled", False))
        self.anomaly_max_checks = max(0, int(llm_cfg.get("llm_anomaly_max_checks_per_cycle", 2)))

        self._client: Optional[OpenAI] = None
        self._cache: dict[str, tuple[str, float]] = {}
        self._calls_this_cycle = 0
        self._anomaly_checks_this_cycle = 0
        self._month_cost_usd = 0.0
        self._current_month = datetime.now().month
        self._last_picks_hash = ""

        token_env = str(llm_cfg.get("token_env", "DEEPSEEK_API_KEY"))
        api_key = os.getenv(token_env, "").strip()
        if not self.enabled:
            return

        if OpenAI is None:
            logger.warning("[LLM] SDK openai ausente. Instale 'openai>=1.40'. LLM desabilitada.")
            self.enabled = False
            return

        if not api_key:
            logger.warning("[LLM] Token nao encontrado em env var '%s'. LLM desabilitada.", token_env)
            self.enabled = False
            return

        self._client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.timeout_sec,
        )

    @property
    def is_available(self) -> bool:
        return (
            self.enabled
            and self._client is not None
            and self._month_cost_usd < self.monthly_budget
        )

    @property
    def calls_this_cycle(self) -> int:
        return self._calls_this_cycle

    def reset_cycle_counter(self):
        """Reset call budget at the beginning of each bot cycle."""
        self._calls_this_cycle = 0
        self._anomaly_checks_this_cycle = 0

    def generate(self, system_prompt: str, user_prompt: str, model: str | None = None) -> str | None:
        """Generate text response with retries, cache, and budget limits."""
        if not self.enabled or self._client is None:
            return None

        now = datetime.now()
        if now.month != self._current_month:
            self._current_month = now.month
            self._month_cost_usd = 0.0
            logger.info("[LLM] Budget mensal resetado para mes=%s", self._current_month)

        if self._month_cost_usd >= self.monthly_budget:
            logger.warning(
                "[LLM] Budget mensal atingido ($%.3f/$%.2f). Fallback habilitado.",
                self._month_cost_usd,
                self.monthly_budget,
            )
            return None

        if self._calls_this_cycle >= self.max_calls:
            logger.debug("[LLM] Limite por ciclo atingido (%s)", self.max_calls)
            return None

        cache_key = hashlib.md5(f"{system_prompt}\n{user_prompt}".encode("utf-8")).hexdigest()
        cache_hit = self._cache.get(cache_key)
        if cache_hit:
            cached_text, cached_at = cache_hit
            if (time.time() - cached_at) < (self.cache_ttl * 60):
                logger.debug("[LLM] Cache hit key=%s", cache_key[:8])
                return cached_text
            self._cache.pop(cache_key, None)

        chosen_model = model or self.model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        started = time.time()
        response = None
        for attempt in range(self.retry_count + 1):
            try:
                response = self._client.chat.completions.create(
                    model=chosen_model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                break
            except (APITimeoutError, RateLimitError) as exc:
                if attempt >= self.retry_count:
                    logger.warning("[LLM] Falha apos %s tentativa(s): %s", attempt + 1, exc)
                    return None
                sleep_sec = self.retry_backoff * (attempt + 1)
                logger.warning("[LLM] Retry %s/%s em %ss (%s)", attempt + 1, self.retry_count, sleep_sec, exc)
                time.sleep(sleep_sec)
            except APIError as exc:
                logger.error("[LLM] Erro de API: %s", exc)
                return None
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("[LLM] Erro inesperado: %s", exc)
                return None

        if response is None:
            return None

        content = ""
        try:
            content = str(response.choices[0].message.content or "").strip()
        except Exception:
            content = ""
        if not content:
            return None

        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        cost = self._estimate_cost(input_tokens, output_tokens)
        self._month_cost_usd += cost
        self._calls_this_cycle += 1

        elapsed = time.time() - started
        logger.info(
            "[LLM] Gerado em %.1fs | tokens=%s+%s | custo=~$%.4f | acumulado=$%.3f/$%.2f",
            elapsed,
            input_tokens,
            output_tokens,
            cost,
            self._month_cost_usd,
            self.monthly_budget,
        )

        self._cache[cache_key] = (content, time.time())
        return content

    def generate_match_analysis(
        self,
        match: dict,
        features: dict,
        prediction: dict,
        analysis: dict,
        context_text: str,
    ) -> str | None:
        """Generate compact narrative for one match pick."""
        prompt = build_match_analysis_prompt(match, features, prediction, analysis, context_text)
        return self.generate(SYSTEM_MATCH_ANALYSIS, prompt)

    def generate_top_picks_report(self, picks: list[dict], total_candidates: int) -> str | None:
        """Generate cycle-level summary for Telegram."""
        if self.skip_unchanged:
            picks_hash = self._hash_picks(picks)
            if picks_hash == self._last_picks_hash:
                logger.debug("[LLM] Top picks inalterados; pulando chamada.")
                return None
            self._last_picks_hash = picks_hash

        prompt = build_top_picks_prompt(picks, total_candidates)
        return self.generate(SYSTEM_TOP_PICKS, prompt)

    def generate_audit_report(self, summary: dict) -> str | None:
        """Generate natural-language daily audit commentary."""
        prompt = build_audit_prompt(summary)
        return self.generate(SYSTEM_AUDIT, prompt)

    def generate_anomaly_flag(self, match: dict, prediction: dict, analysis: dict) -> str | None:
        """
        D3: anomaly flagging curto (OK|FLAG), com cap proprio por ciclo.
        """
        if not self.anomaly_check_enabled:
            return None
        if self.anomaly_max_checks <= 0:
            return None
        if self._anomaly_checks_this_cycle >= self.anomaly_max_checks:
            return None

        self._anomaly_checks_this_cycle += 1
        prompt = build_anomaly_prompt(match=match, prediction=prediction, analysis=analysis)
        out = self.generate(SYSTEM_ANOMALY_CHECK, prompt)
        if not out:
            return None

        text = out.strip()
        upper = text.upper()
        if upper.startswith("FLAG"):
            return f"FLAG {text[4:].strip()}".strip()
        if upper.startswith("OK"):
            return f"OK {text[2:].strip()}".strip()
        return text

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        input_rate = 0.27 / 1_000_000
        output_rate = 1.10 / 1_000_000
        return (max(0, input_tokens) * input_rate) + (max(0, output_tokens) * output_rate)

    def _hash_picks(self, picks: list[dict]) -> str:
        """Hash essential top-picks shape to detect unchanged cycles."""
        rows: list[tuple[int, int, float]] = []
        for pick in picks:
            match = pick.get("match", {})
            rows.append(
                (
                    int(match.get("id", 0) or 0),
                    int(pick.get("official_pick_winner_id", 0) or 0),
                    float(pick.get("score", 0.0) or 0.0),
                )
            )
        payload = "|".join(f"{mid}:{wid}:{score:.4f}" for mid, wid, score in sorted(rows))
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

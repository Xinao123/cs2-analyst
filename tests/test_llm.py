import os
import unittest
from datetime import datetime
from unittest.mock import patch

from ai.llm import DeepSeekClient


class _FakeUsage:
    def __init__(self, prompt_tokens: int = 100, completion_tokens: int = 40):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeMessage:
    def __init__(self, content: str = "ok"):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str = "ok"):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str = "ok", prompt_tokens: int = 100, completion_tokens: int = 40):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(prompt_tokens, completion_tokens)


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if not self._responses:
            return _FakeResponse()
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeClient:
    def __init__(self, responses):
        self.chat = type("Chat", (), {})()
        self.chat.completions = _FakeCompletions(responses)


class DeepSeekClientTests(unittest.TestCase):
    def test_disabled_returns_none(self):
        client = DeepSeekClient({"llm": {"enabled": False}})
        self.assertIsNone(client.generate("system", "user"))

    def test_missing_token_disables(self):
        os.environ.pop("NONEXISTENT_XYZ", None)
        client = DeepSeekClient({"llm": {"enabled": True, "token_env": "NONEXISTENT_XYZ"}})
        self.assertFalse(client.is_available)

    def test_budget_limit_blocks(self):
        client = DeepSeekClient({"llm": {"enabled": False}})
        client.enabled = True
        client._client = _FakeClient([_FakeResponse("ok")])
        client.monthly_budget = 1.0
        client._month_cost_usd = 1.0
        self.assertIsNone(client.generate("sys", "user"))

    def test_cache_hit(self):
        client = DeepSeekClient({"llm": {"enabled": False}})
        fake = _FakeClient([_FakeResponse("cacheable")])
        client.enabled = True
        client._client = fake
        first = client.generate("sys", "same")
        second = client.generate("sys", "same")
        self.assertEqual("cacheable", first)
        self.assertEqual("cacheable", second)
        self.assertEqual(1, fake.chat.completions.calls)

    def test_cycle_limit(self):
        client = DeepSeekClient({"llm": {"enabled": False}})
        client.enabled = True
        client._client = _FakeClient([_FakeResponse("x")])
        client.max_calls = 1
        client._calls_this_cycle = 1
        self.assertIsNone(client.generate("sys", "user"))

    def test_retry_on_timeout(self):
        timeout_exc = Exception("timeout")
        with patch("ai.llm.APITimeoutError", Exception):
            client = DeepSeekClient({"llm": {"enabled": False}})
            client.enabled = True
            client.retry_count = 1
            client.retry_backoff = 0
            client._client = _FakeClient([timeout_exc, _FakeResponse("after-retry")])
            out = client.generate("sys", "user")
            self.assertEqual("after-retry", out)

    def test_month_reset(self):
        client = DeepSeekClient({"llm": {"enabled": False}})
        client.enabled = True
        client._client = _FakeClient([_FakeResponse("ok", 10, 10)])
        client._month_cost_usd = 1.0
        client._current_month = 1 if datetime.now().month != 1 else 2
        out = client.generate("sys", "user")
        self.assertEqual("ok", out)
        self.assertEqual(datetime.now().month, client._current_month)
        self.assertLess(client._month_cost_usd, 1.0)

    def test_skip_unchanged_picks(self):
        client = DeepSeekClient({"llm": {"enabled": False}})
        client.enabled = True
        calls = {"n": 0}

        def _fake_generate(system_prompt, user_prompt, model=None):
            calls["n"] += 1
            return "report"

        client.generate = _fake_generate  # type: ignore[method-assign]
        picks = [{"match": {"id": 10}, "official_pick_winner_id": 1, "score": 9.5}]
        first = client.generate_top_picks_report(picks, 10)
        second = client.generate_top_picks_report(picks, 10)
        self.assertEqual("report", first)
        self.assertIsNone(second)
        self.assertEqual(1, calls["n"])

    def test_estimate_cost(self):
        client = DeepSeekClient({"llm": {"enabled": False}})
        cost = client._estimate_cost(1000, 500)
        expected = (1000 * 0.27 / 1_000_000) + (500 * 1.10 / 1_000_000)
        self.assertAlmostEqual(expected, cost, places=9)

    def test_base_url_is_deepseek(self):
        client = DeepSeekClient(
            {
                "llm": {
                    "enabled": True,
                    "token_env": "NONEXISTENT",
                    "base_url": "https://api.deepseek.com",
                }
            }
        )
        self.assertEqual("https://api.deepseek.com", client.base_url)


if __name__ == "__main__":
    unittest.main()

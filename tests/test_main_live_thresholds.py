import unittest

from main import (
    _is_synthetic_match,
    _resolve_live_thresholds,
    _should_allow_low_data_override,
    _should_sync_odds,
)


class _PredictorStub:
    def __init__(self, min_confidence: float):
        self.min_confidence = min_confidence


class MainLiveThresholdTests(unittest.TestCase):
    def test_confidence_auto_tune_uses_max_between_config_and_tuned(self):
        cfg = {
            "min_confidence": 62.0,
            "min_value_pct": 6.0,
            "confidence_auto_tune": True,
            "synthetic_live_min_confidence": 68.0,
            "synthetic_live_min_value_pct": 7.0,
        }
        thresholds = _resolve_live_thresholds(cfg, _PredictorStub(75.0))
        self.assertEqual(75.0, thresholds["min_confidence"])
        self.assertEqual(6.0, thresholds["min_value"])
        self.assertEqual(75.0, thresholds["synthetic_min_confidence"])
        self.assertEqual(7.0, thresholds["synthetic_min_value"])

    def test_confidence_auto_tune_keeps_config_when_tuned_lower(self):
        cfg = {
            "min_confidence": 62.0,
            "min_value_pct": 6.0,
            "confidence_auto_tune": True,
        }
        thresholds = _resolve_live_thresholds(cfg, _PredictorStub(60.0))
        self.assertEqual(62.0, thresholds["min_confidence"])

    def test_auto_tune_used_when_min_confidence_not_explicit(self):
        cfg = {
            "confidence_auto_tune": True,
            "min_value_pct": 6.0,
        }
        thresholds = _resolve_live_thresholds(cfg, _PredictorStub(65.0))
        self.assertEqual(65.0, thresholds["min_confidence"])
        self.assertEqual(6.0, thresholds["min_value"])
        self.assertGreaterEqual(thresholds["synthetic_min_confidence"], thresholds["min_confidence"])
        self.assertGreaterEqual(thresholds["synthetic_min_value"], thresholds["min_value"])

    def test_is_synthetic_match(self):
        self.assertTrue(_is_synthetic_match({"team1_id": -1, "team2_id": 10}))
        self.assertTrue(_is_synthetic_match({"team1_id": 10, "team2_id": 0}))
        self.assertFalse(_is_synthetic_match({"team1_id": 10, "team2_id": 20}))

    def test_should_sync_odds_first_sync_runs_immediately(self):
        self.assertTrue(_should_sync_odds(now_ts=100.0, last_sync_at=None, refresh_seconds=1800))

    def test_should_sync_odds_respects_refresh_window(self):
        self.assertFalse(_should_sync_odds(now_ts=1000.0, last_sync_at=100.0, refresh_seconds=1800))
        self.assertTrue(_should_sync_odds(now_ts=1900.0, last_sync_at=100.0, refresh_seconds=1800))

    def test_low_data_override_allows_only_when_strict_conditions_match(self):
        self.assertTrue(
            _should_allow_low_data_override(
                has_valid_odds=True,
                team1_id=10,
                team2_id=20,
                confidence=72.0,
                value_pct=7.0,
                synthetic_min_confidence=70.0,
                synthetic_min_value=6.0,
            )
        )

    def test_low_data_override_rejects_missing_conditions(self):
        self.assertFalse(
            _should_allow_low_data_override(
                has_valid_odds=True,
                team1_id=10,
                team2_id=20,
                confidence=68.0,
                value_pct=7.0,
                synthetic_min_confidence=70.0,
                synthetic_min_value=6.0,
            )
        )
        self.assertFalse(
            _should_allow_low_data_override(
                has_valid_odds=True,
                team1_id=0,
                team2_id=20,
                confidence=75.0,
                value_pct=8.0,
                synthetic_min_confidence=70.0,
                synthetic_min_value=6.0,
            )
        )


if __name__ == "__main__":
    unittest.main()

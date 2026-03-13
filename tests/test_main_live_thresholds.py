import unittest

from main import _is_synthetic_match, _resolve_live_thresholds


class _PredictorStub:
    def __init__(self, min_confidence: float):
        self.min_confidence = min_confidence


class MainLiveThresholdTests(unittest.TestCase):
    def test_explicit_min_confidence_overrides_auto_tune(self):
        cfg = {
            "min_confidence": 62.0,
            "min_value_pct": 6.0,
            "confidence_auto_tune": True,
            "synthetic_live_min_confidence": 68.0,
            "synthetic_live_min_value_pct": 7.0,
        }
        thresholds = _resolve_live_thresholds(cfg, _PredictorStub(75.0))
        self.assertEqual(62.0, thresholds["min_confidence"])
        self.assertEqual(6.0, thresholds["min_value"])
        self.assertEqual(68.0, thresholds["synthetic_min_confidence"])
        self.assertEqual(7.0, thresholds["synthetic_min_value"])

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


if __name__ == "__main__":
    unittest.main()

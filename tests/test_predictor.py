import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.predictor import Predictor


def _synthetic_features(rank_diff: float, activity_diff: float, t1_games: int, t2_games: int, h2h: int):
    return {
        "ranking_diff": rank_diff,
        "ranking_ratio": max(0.01, (50 + rank_diff) / 100),
        "winrate_diff": rank_diff / 120.0,
        "team1_winrate": 0.5 + (rank_diff / 300.0),
        "team2_winrate": 0.5 - (rank_diff / 300.0),
        "form_diff": activity_diff / 10.0,
        "team1_form": 0.5 + (activity_diff / 30.0),
        "team2_form": 0.5 - (activity_diff / 30.0),
        "team1_streak": int(activity_diff),
        "team2_streak": int(-activity_diff),
        "streak_diff": activity_diff * 2.0,
        "h2h_winrate_t1": 0.5 + (rank_diff / 400.0),
        "h2h_matches": h2h,
        "h2h_advantage": rank_diff / 400.0,
        "avg_rating_diff": rank_diff / 150.0,
        "team1_avg_rating": 1.0 + (rank_diff / 500.0),
        "team2_avg_rating": 1.0 - (rank_diff / 500.0),
        "avg_kd_diff": rank_diff / 300.0,
        "avg_impact_diff": rank_diff / 250.0,
        "strong_maps_diff": activity_diff / 3.0,
        "best_map_wr_diff": rank_diff / 80.0,
        "is_bo1": 0,
        "is_bo3": 1,
        "is_lan": 0,
        "event_tier": 2,
        "team1_matches_played": t1_games,
        "team2_matches_played": t2_games,
        "activity_diff": activity_diff,
    }


class PredictorTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.model_path = str(Path(self.tmpdir.name) / "model.joblib")
        self.config = {
            "model": {
                "path": self.model_path,
                "enable_calibration": True,
                "calibration_method": "sigmoid",
                "calibration_min_samples": 80,
                "calibration_cv_splits": 3,
                "prediction_temperature": 0.9,
                "low_data_rank_blend": 0.45,
                "rank_prior_scale": 10.0,
            }
        }

    def tearDown(self):
        self.tmpdir.cleanup()

    def _build_dataset(self, n: int = 140):
        rng = np.random.default_rng(42)
        features = []
        labels = []
        for _ in range(n):
            rank_diff = float(rng.normal(0, 60))
            activity = float(rng.normal(0, 4))
            t1_games = int(rng.integers(0, 15))
            t2_games = int(rng.integers(0, 15))
            h2h = int(rng.integers(0, 8))
            feats = _synthetic_features(rank_diff, activity, t1_games, t2_games, h2h)
            p = 1.0 / (1.0 + math.exp(-((rank_diff / 22.0) + (activity / 6.0))))
            label = int(rng.random() < p)
            features.append(feats)
            labels.append(label)
        return features, labels

    def test_train_and_predict_returns_valid_probability(self):
        features, labels = self._build_dataset()
        predictor = Predictor(self.config)
        metrics = predictor.train(features, labels)
        self.assertNotIn("error", metrics)

        pred = predictor.predict(_synthetic_features(35, 2, 8, 7, 2))
        self.assertIsNotNone(pred)
        self.assertGreaterEqual(pred["team1_win_prob"], 1.0)
        self.assertLessEqual(pred["team1_win_prob"], 99.0)
        self.assertAlmostEqual(pred["team1_win_prob"] + pred["team2_win_prob"], 100.0, delta=0.2)

    def test_low_data_rank_prior_improves_separation(self):
        features, labels = self._build_dataset()
        predictor = Predictor(self.config)
        metrics = predictor.train(features, labels)
        self.assertNotIn("error", metrics)

        strong_t1 = _synthetic_features(120, 0, 0, 0, 0)
        strong_t2 = _synthetic_features(-120, 0, 0, 0, 0)
        p1 = predictor.predict(strong_t1)["team1_win_prob"]
        p2 = predictor.predict(strong_t2)["team1_win_prob"]

        self.assertGreater(p1, 60.0)
        self.assertLess(p2, 40.0)
        self.assertGreater(p1 - p2, 20.0)

    def test_train_accepts_quality_weights(self):
        features, labels = self._build_dataset()
        predictor = Predictor(self.config)
        quality_weights = [0.7 if i % 3 == 0 else 1.0 for i in range(len(features))]
        metrics = predictor.train(features, labels, sample_weights=quality_weights)
        self.assertNotIn("error", metrics)
        self.assertIn("sample_weight_mean", metrics)
        self.assertGreater(metrics["sample_weight_max"], 0.0)


if __name__ == "__main__":
    unittest.main()

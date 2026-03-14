import tempfile
import unittest
from pathlib import Path

from analysis.predictor import Predictor
from db.models import Database
from main import _score_bet_candidate


class MlImprovementsV2Tests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "mlv2.db")
        self.model_path = str(Path(self.tmpdir.name) / "model.joblib")
        self.db = Database(self.db_path)

        self.db.upsert_team(1, "Alpha", ranking=10)
        self.db.upsert_team(2, "Beta", ranking=20)

        self.match_id = self.db.upsert_match(
            hltv_id=999001,
            date="2026-03-10T10:00:00+00:00",
            event_name="Test Cup",
            team1_id=1,
            team2_id=2,
            team1_score=2,
            team2_score=1,
            winner_id=1,
            status="completed",
            best_of=3,
            is_lan=1,
            event_tier=1,
        )
        with self.db.connect() as conn:
            conn.execute(
                """INSERT INTO match_maps
                (match_id, map_name, team1_rounds, team2_rounds,
                 team1_ct_rounds, team1_t_rounds, team2_ct_rounds, team2_t_rounds, winner_id)
                VALUES (?, 'inferno', 13, 10, 7, 6, 4, 6, 1)""",
                (self.match_id,),
            )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_db_side_stats_and_clv(self):
        stats = self.db.get_team_side_stats(1, days=365)
        self.assertGreater(stats["maps_count"], 0)
        self.assertGreater(stats["ct_rounds_total"], 0)
        self.assertGreater(stats["t_rounds_total"], 0)

        clv = self.db.save_clv(match_id=self.match_id, side="team1", open_odds=2.10, close_odds=1.95)
        self.assertGreater(clv, 0.0)
        self.assertGreater(self.db.get_avg_clv(days=30), 0.0)

    def test_predictor_tuning_and_ensemble_flags(self):
        cfg = {
            "model": {
                "path": self.model_path,
                "enable_hyperparam_tuning": True,
                "tuning_max_combinations": 2,
                "enable_ensemble": True,
                "min_train_samples": 20,
                "min_class_samples": 5,
                "enable_calibration": False,
                "prediction_temperature": 1.0,
            }
        }
        predictor = Predictor(cfg)

        feats = []
        labels = []
        for i in range(80):
            rd = (i % 11) - 5
            wr = rd / 10.0
            feats.append(
                {
                    "ranking_diff": float(rd),
                    "winrate_diff": float(wr),
                    "form_diff": float(wr / 2),
                    "h2h_matches": float(i % 5),
                    "team1_matches_played": 8.0,
                    "team2_matches_played": 9.0,
                    "is_bo1": 0.0,
                    "is_bo3": 1.0,
                    "is_lan": 1.0,
                    "event_tier": 2.0,
                }
            )
            labels.append(1 if rd >= 0 else 0)

        metrics = predictor.train(feats, labels)
        self.assertNotIn("error", metrics)
        self.assertIn("tuning_evaluated", metrics)
        self.assertIn("ensemble_members", metrics)

        pred = predictor.predict(feats[0])
        self.assertIsNotNone(pred)
        self.assertGreaterEqual(pred["team1_win_prob"], 1.0)
        self.assertLessEqual(pred["team1_win_prob"], 99.0)

    def test_score_candidate_context_multiplier(self):
        prediction = {"confidence": 70.0}
        best_vb = {"value_pct": 8.0, "expected_value": 18.0}

        score_plain = _score_bet_candidate(prediction, best_vb, match={"event_tier": 4, "best_of": 1, "is_lan": 0})
        score_strong = _score_bet_candidate(prediction, best_vb, match={"event_tier": 1, "best_of": 3, "is_lan": 1})

        self.assertGreater(score_strong, score_plain)


if __name__ == "__main__":
    unittest.main()

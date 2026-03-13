import tempfile
import unittest
from pathlib import Path

from analysis.features import FeatureExtractor
from db.models import Database


class FeatureExtractorTemporalTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "test.db")
        self.db = Database(self.db_path)
        self.config = {
            "model": {
                "form_window_days": 365,
                "live_min_recent_matches": 0,
                "train_min_recent_matches": 0,
                "use_player_features": False,
                "exclude_synthetic_teams": True,
                "exclude_academy_teams": True,
            }
        }
        self.fx = FeatureExtractor(self.db, self.config)

        self.db.upsert_team(1, "Alpha", ranking=10)
        self.db.upsert_team(2, "Beta", ranking=20)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _insert_completed(self, hltv_id: int, dt: str, winner_id: int):
        self.db.upsert_match(
            hltv_id=hltv_id,
            date=dt,
            event_name="Test",
            team1_id=1,
            team2_id=2,
            team1_score=2 if winner_id == 1 else 1,
            team2_score=2 if winner_id == 2 else 1,
            winner_id=winner_id,
            status="completed",
        )

    def test_extract_as_of_date_excludes_future_matches(self):
        self._insert_completed(1001, "2026-03-10 10:00:00", winner_id=1)
        self._insert_completed(1002, "2026-03-11 10:00:00", winner_id=2)
        self._insert_completed(1003, "2026-03-12 10:00:00", winner_id=1)

        match = {
            "team1_id": 1,
            "team2_id": 2,
            "date": "2026-03-11 10:00:00",
            "best_of": 3,
            "event_tier": 2,
            "is_lan": 0,
        }
        feats = self.fx.extract(
            match,
            min_recent_matches=0,
            as_of_date=match["date"],
            for_training=True,
        )
        self.assertIsNotNone(feats)
        # Antes de 11/03 ha somente o jogo de 10/03.
        self.assertEqual(1, int(feats["team1_matches_played"]))
        self.assertEqual(1, int(feats["team2_matches_played"]))

    def test_extract_filters_synthetic_teams_when_enabled(self):
        match = {
            "team1_id": -9000000001,
            "team2_id": 2,
            "date": "2026-03-11 10:00:00",
            "best_of": 1,
            "event_tier": 3,
            "is_lan": 0,
        }
        feats = self.fx.extract(match, min_recent_matches=0)
        self.assertIsNone(feats)

    def test_extract_filters_academy_teams_when_enabled(self):
        self.db.upsert_team(3, "NAVI Junior", ranking=30)
        match = {
            "team1_id": 3,
            "team2_id": 2,
            "date": "2026-03-11 10:00:00",
            "best_of": 1,
            "event_tier": 3,
            "is_lan": 0,
        }
        feats = self.fx.extract(match, min_recent_matches=0)
        self.assertIsNone(feats)


if __name__ == "__main__":
    unittest.main()

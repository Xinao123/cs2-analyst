import tempfile
import unittest
from pathlib import Path

from db.models import Database


class DatabaseUpcomingCleanupTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "test.db")
        self.db = Database(self.db_path)

        self.db.upsert_team(1, "Alpha", ranking=10)
        self.db.upsert_team(2, "Beta", ranking=20)
        self.db.upsert_team(3, "Gamma", ranking=30)
        self.db.upsert_team(4, "Delta", ranking=40)

        self.upcoming_id = self.db.upsert_match(
            hltv_id=999001,
            date="2026-03-15T15:00:00",
            event_name="Upcoming Cup",
            team1_id=1,
            team2_id=2,
            status="upcoming",
        )
        self.completed_id = self.db.upsert_match(
            hltv_id=999002,
            date="2026-03-10T15:00:00",
            event_name="Completed Cup",
            team1_id=3,
            team2_id=4,
            team1_score=2,
            team2_score=1,
            winner_id=3,
            status="completed",
        )

        self.db.upsert_match_odds_latest(
            self.upcoming_id,
            provider="oddspapi",
            fixture_id="fx-up",
            odds_team1=1.8,
            odds_team2=2.0,
            bookmaker_team1="pinnacle",
            bookmaker_team2="pinnacle",
            updated_at="2026-03-14T12:00:00",
        )
        self.db.upsert_match_odds_latest(
            self.completed_id,
            provider="oddspapi",
            fixture_id="fx-comp",
            odds_team1=1.9,
            odds_team2=1.9,
            bookmaker_team1="pinnacle",
            bookmaker_team2="pinnacle",
            updated_at="2026-03-10T12:00:00",
        )

        self.db.insert_odds_snapshot(
            self.upcoming_id,
            provider="oddspapi",
            fixture_id="fx-up",
            bookmaker="pinnacle",
            market_key="h2h",
            side="team1",
            odds=1.8,
            collected_at="2026-03-14T12:00:00",
            fingerprint="snap-up",
        )
        self.db.insert_odds_snapshot(
            self.completed_id,
            provider="oddspapi",
            fixture_id="fx-comp",
            bookmaker="pinnacle",
            market_key="h2h",
            side="team1",
            odds=1.9,
            collected_at="2026-03-10T12:00:00",
            fingerprint="snap-comp",
        )

        self.db.save_prediction(
            match_id=self.upcoming_id,
            predicted_winner_id=1,
            official_pick_winner_id=1,
            team1_win_prob=60.0,
            team2_win_prob=40.0,
        )
        self.db.save_prediction(
            match_id=self.completed_id,
            predicted_winner_id=3,
            official_pick_winner_id=3,
            team1_win_prob=60.0,
            team2_win_prob=40.0,
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_clear_upcoming_related_data(self):
        report = self.db.clear_upcoming_related_data()
        self.assertEqual(1, report["matches_deleted"])
        self.assertEqual(1, report["odds_latest_deleted"])
        self.assertEqual(1, report["odds_snapshots_deleted"])
        self.assertEqual(1, report["predictions_deleted"])

        with self.db.connect() as conn:
            remaining_upcoming = conn.execute(
                "SELECT COUNT(*) AS c FROM matches WHERE status='upcoming'"
            ).fetchone()["c"]
            completed_matches = conn.execute(
                "SELECT COUNT(*) AS c FROM matches WHERE status='completed'"
            ).fetchone()["c"]
            odds_latest = conn.execute(
                "SELECT COUNT(*) AS c FROM match_odds_latest"
            ).fetchone()["c"]
            odds_snapshots = conn.execute(
                "SELECT COUNT(*) AS c FROM odds_snapshots"
            ).fetchone()["c"]
            predictions = conn.execute(
                "SELECT COUNT(*) AS c FROM predictions"
            ).fetchone()["c"]

        self.assertEqual(0, remaining_upcoming)
        self.assertEqual(1, completed_matches)
        self.assertEqual(1, odds_latest)
        self.assertEqual(1, odds_snapshots)
        self.assertEqual(1, predictions)


if __name__ == "__main__":
    unittest.main()


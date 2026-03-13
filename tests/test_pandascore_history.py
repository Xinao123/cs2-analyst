import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from db.models import Database
from scraper.hltv import HLTVScraper


class PandaScoreHistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "test.db")
        self.db = Database(self.db_path)
        self.token_env = "PANDASCORE_TEST_TOKEN"
        self.config = {
            "database": {"path": self.db_path},
            "scraper": {
                "pandascore_token_env": self.token_env,
                "pandascore_history_enabled": True,
                "pandascore_history_months": 12,
                "pandascore_history_window_days": 7,
                "pandascore_history_per_page": 100,
                "pandascore_history_max_requests_per_hour": 800,
                "pandascore_history_checkpoint_key": "test_history_cursor",
            },
        }
        self.scraper = HLTVScraper(self.db, self.config)
        os.environ[self.token_env] = "dummy-token"

    def tearDown(self):
        os.environ.pop(self.token_env, None)
        self.tmpdir.cleanup()

    def test_normalize_completed_payload(self):
        payload = {
            "id": 123456,
            "status": "finished",
            "end_at": "2026-03-10T12:00:00Z",
            "number_of_games": 3,
            "winner_id": 11,
            "tournament": {"name": "CCT"},
            "opponents": [
                {"opponent": {"id": 11, "name": "Alpha"}},
                {"opponent": {"id": 22, "name": "Beta"}},
            ],
            "results": [
                {"team_id": 11, "score": 2},
                {"team_id": 22, "score": 1},
            ],
        }
        normalized = self.scraper._normalize_pandascore_completed_payload(payload)
        self.assertIsNotNone(normalized)
        self.assertEqual("finished", normalized["status"])
        self.assertEqual(11, normalized["team1_source_id"])
        self.assertEqual(22, normalized["team2_source_id"])
        self.assertEqual(2, normalized["team1_score"])
        self.assertEqual(1, normalized["team2_score"])
        self.assertEqual(11, normalized["winner_source_id"])

    def test_history_sync_saves_completed_matches(self):
        payload = {
            "id": 9991,
            "status": "finished",
            "end_at": "2026-03-10T12:00:00Z",
            "winner_id": 101,
            "number_of_games": 3,
            "opponents": [
                {"opponent": {"id": 101, "name": "FURIA"}},
                {"opponent": {"id": 202, "name": "TYLOO"}},
            ],
            "results": [
                {"team_id": 101, "score": 2},
                {"team_id": 202, "score": 0},
            ],
            "tournament": {"name": "ESL Pro League"},
        }

        with patch.object(
            self.scraper,
            "_fetch_pandascore_history_page",
            return_value=([payload], "", 1),
        ):
            report = self.scraper.sync_pandascore_history(bootstrap=False, force_full=True)

        self.assertEqual(1, report["saved"])
        self.assertEqual(1, report["returned"])
        stats = self.db.get_stats()
        self.assertEqual(1, stats["completed_matches"])

    def test_history_cursor_moves_between_runs(self):
        with patch.object(
            self.scraper,
            "_fetch_pandascore_history_page",
            return_value=([], "", 1),
        ):
            report1 = self.scraper.sync_pandascore_history(bootstrap=False, force_full=True)
            cursor1 = self.db.get_state("test_history_cursor", "")
            report2 = self.scraper.sync_pandascore_history(bootstrap=False, force_full=False)
            cursor2 = self.db.get_state("test_history_cursor", "")

        self.assertGreaterEqual(report1["requests_used"], 1)
        self.assertGreaterEqual(report2["requests_used"], 1)
        self.assertNotEqual(cursor1, "")
        self.assertNotEqual(cursor2, "")
        self.assertNotEqual(cursor1, cursor2)

    def test_history_quota_respected(self):
        self.scraper.pandascore_history_max_requests_per_hour = 1
        self.assertTrue(self.scraper._consume_history_quota())
        self.assertFalse(self.scraper._consume_history_quota())


if __name__ == "__main__":
    unittest.main()

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from db.models import Database
from scraper.hltv import (
    HLTVScraper,
    _synthetic_provider_match_id,
    _synthetic_provider_team_id,
)


class UpcomingProviderTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "test.db")
        self.db = Database(self.db_path)
        self.config = {
            "scraper": {
                "upcoming_provider": "pandascore",
                "upcoming_fallback": "hltv",
                "upcoming_days_ahead": 7,
                "upcoming_max_pages": 2,
                "pandascore_token_env": "PANDASCORE_TEST_TOKEN",
            }
        }
        self.scraper = HLTVScraper(self.db, self.config)

    async def asyncTearDown(self):
        await self.scraper.close()
        self.tmpdir.cleanup()

    async def test_normalize_pandascore_payload(self):
        payload = {
            "id": 12345,
            "begin_at": "2026-03-12T20:00:00Z",
            "number_of_games": 3,
            "league": {"name": "ESL"},
            "serie": {"name": "Pro League"},
            "tournament": {"name": "Season 22"},
            "opponents": [
                {"opponent": {"id": 101, "name": "Team Alpha"}},
                {"opponent": {"id": 202, "name": "Team Beta"}},
            ],
        }

        normalized = self.scraper._normalize_pandascore_upcoming_payload(payload)
        self.assertIsNotNone(normalized)
        self.assertEqual(_synthetic_provider_match_id(12345), normalized["match_id"])
        self.assertEqual(101, normalized["team1_id"])
        self.assertEqual(202, normalized["team2_id"])
        self.assertEqual("Team Alpha", normalized["team1_name"])
        self.assertEqual("Team Beta", normalized["team2_name"])
        self.assertEqual("Season 22", normalized["event_name"])
        self.assertEqual(3, normalized["best_of"])

    async def test_synthetic_ids_are_deterministic_and_negative(self):
        team_id = _synthetic_provider_team_id(999)
        match_id = _synthetic_provider_match_id(555)
        self.assertEqual(team_id, _synthetic_provider_team_id(999))
        self.assertEqual(match_id, _synthetic_provider_match_id(555))
        self.assertEqual(-(8_000_000_000 + 999), team_id)
        self.assertEqual(-(9_000_000_000 + 555), match_id)
        self.assertLess(team_id, 0)
        self.assertLess(match_id, 0)

    async def test_resolve_team_id_prefers_name_then_synthetic(self):
        self.db.upsert_team(9565, "Vitality", ranking=1)

        existing = self.scraper._resolve_team_id(0, "vitality", provider="pandascore")
        self.assertEqual(9565, existing)

        synthetic = self.scraper._resolve_team_id(777, "Unknown Squad", provider="pandascore")
        self.assertEqual(_synthetic_provider_team_id(777), synthetic)
        created = self.db.get_team(synthetic)
        self.assertIsNotNone(created)
        self.assertEqual("Unknown Squad", created["name"])

    async def test_missing_token_falls_back_to_hltv(self):
        os.environ.pop("PANDASCORE_TEST_TOKEN", None)
        fallback_report = self.scraper._new_upcoming_report()
        fallback_report["returned"] = 1
        fallback_report["saved"] = 1
        self.scraper._scrape_upcoming_from_hltv = AsyncMock(return_value=fallback_report)

        saved = await self.scraper.scrape_upcoming_matches()
        self.assertEqual(1, saved)
        self.scraper._scrape_upcoming_from_hltv.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()

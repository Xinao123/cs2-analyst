import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from analysis.value import ValueDetector
from db.models import Database
from scraper.odds import (
    OddsPapiSync,
    _extract_bookmaker_quotes,
    _index_local_matches,
    _normalize_fixture_payload,
)


class OddsSyncTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "test.db")
        self.db = Database(self.db_path)
        self.token_env = "ODDSPAPI_TEST_KEY"
        self.config = {
            "scraper": {"upcoming_days_ahead": 7},
            "odds": {
                "enabled": True,
                "provider": "oddspapi",
                "base_url": "https://api.oddspapi.io/v4",
                "token_env": self.token_env,
                "market": "h2h",
                "refresh_minutes": 10,
                "match_window_hours": 6,
                "retry_count": 0,
                "bookmaker_whitelist": ["bet365", "betano", "pinnacle", "1xbet"],
            },
        }
        self.sync = OddsPapiSync(self.db, self.config)
        self.local_match_dt = datetime.now(timezone.utc) + timedelta(hours=2)

        self.db.upsert_team(1, "FURIA", ranking=10)
        self.db.upsert_team(2, "TYLOO", ranking=25)
        self.db.upsert_match(
            hltv_id=1111,
            date=self.local_match_dt.replace(tzinfo=None).isoformat(timespec="seconds"),
            event_name="ESL Pro League",
            best_of=3,
            team1_id=1,
            team2_id=2,
            status="upcoming",
        )

    def tearDown(self):
        os.environ.pop(self.token_env, None)
        self.tmpdir.cleanup()

    def test_matching_moderate_handles_swapped_teams(self):
        upcoming = self.db.get_upcoming_matches()
        indexed = _index_local_matches(upcoming)
        fixture = {
            "fixture_id": "fx-1",
            "home_name": "Tyloo",
            "away_name": "FURIA",
            "event_name": "ESL Pro League Group",
            "start_dt": self.local_match_dt + timedelta(minutes=30),
        }
        match_data, swapped = self.sync._match_fixture_to_local(fixture, indexed, set())
        self.assertIsNotNone(match_data)
        self.assertTrue(swapped)

    def test_best_line_uses_whitelist_and_highest_odds(self):
        quotes = [
            {"bookmaker": "bet365", "side": "home", "odds": 1.87, "market_key": "h2h", "changed_at": ""},
            {"bookmaker": "bet365", "side": "away", "odds": 2.02, "market_key": "h2h", "changed_at": ""},
            {"bookmaker": "pinnacle", "side": "home", "odds": 1.91, "market_key": "h2h", "changed_at": ""},
            {"bookmaker": "pinnacle", "side": "away", "odds": 1.98, "market_key": "h2h", "changed_at": ""},
            {"bookmaker": "unknownbook", "side": "away", "odds": 2.30, "market_key": "h2h", "changed_at": ""},
        ]
        best = self.sync._select_best_h2h_lines(quotes)
        self.assertEqual("pinnacle", best["home"]["bookmaker"])
        self.assertAlmostEqual(1.91, best["home"]["odds"], places=2)
        self.assertEqual("bet365", best["away"]["bookmaker"])
        self.assertAlmostEqual(2.02, best["away"]["odds"], places=2)
        self.assertGreater(best["filtered_count"], 0)

    def test_missing_token_degrades_without_crash(self):
        os.environ.pop(self.token_env, None)
        report = self.sync.sync_upcoming_odds()
        self.assertIn("token_ausente", report["error"])
        self.assertEqual(0, report["saved"])

    def test_normalize_v4_fixture_and_bookmaker_odds_payload(self):
        fixture = _normalize_fixture_payload(
            {
                "fixtureId": 9991,
                "participant1Name": "FURIA",
                "participant2Name": "TYLOO",
                "startTime": (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat(),
                "competitionName": "ESL Challenger",
            }
        )
        self.assertIsNotNone(fixture)
        self.assertEqual("9991", fixture["fixture_id"])
        self.assertEqual("FURIA", fixture["home_name"])
        self.assertEqual("TYLOO", fixture["away_name"])

        payload = {
            "bookmakerOdds": {
                "bet365": {
                    "h2h": {"FURIA": 1.84, "TYLOO": 2.08},
                }
            }
        }
        quotes = _extract_bookmaker_quotes(payload, fixture)
        self.assertEqual(2, len(quotes))
        sides = {q["side"] for q in quotes}
        self.assertEqual({"home", "away"}, sides)
        self.assertEqual({"bet365"}, {q["bookmaker"] for q in quotes})

    def test_sync_persists_latest_and_snapshots_idempotently(self):
        os.environ[self.token_env] = "test-token"
        fixture_time = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
        fixtures = [
            {
                "fixture_id": "fx-777",
                "home_name": "FURIA",
                "away_name": "TYLOO",
                "event_name": "ESL Pro League",
                "start_dt": datetime.fromisoformat(fixture_time),
            }
        ]
        odds_payload = {
            "data": [
                {
                    "bookmaker": {"name": "bet365"},
                    "markets": [
                        {
                            "name": "Match Winner",
                            "odds": [
                                {"name": "home", "odds": 1.88},
                                {"name": "away", "odds": 2.05},
                            ],
                        }
                    ],
                }
            ]
        }

        with patch.object(self.sync, "_resolve_sport_id", return_value="cs2"), patch.object(
            self.sync, "_fetch_fixtures", return_value=(fixtures, "")
        ), patch.object(self.sync, "_fetch_fixture_odds", return_value=(odds_payload, "")):
            report1 = self.sync.sync_upcoming_odds()
            report2 = self.sync.sync_upcoming_odds()

        self.assertEqual(1, report1["saved"])
        self.assertEqual(1, report2["saved"])

        with self.db.connect() as conn:
            latest_rows = conn.execute("SELECT COUNT(*) AS c FROM match_odds_latest").fetchone()["c"]
            snapshot_rows = conn.execute("SELECT COUNT(*) AS c FROM odds_snapshots").fetchone()["c"]
        self.assertEqual(1, latest_rows)
        self.assertEqual(2, snapshot_rows)

        upcoming = self.db.get_upcoming_matches()
        self.assertEqual(1, len(upcoming))
        self.assertAlmostEqual(1.88, float(upcoming[0]["odds_team1"]), places=2)
        self.assertAlmostEqual(2.05, float(upcoming[0]["odds_team2"]), places=2)
        self.assertEqual("bet365", upcoming[0]["bookmaker_team1"])
        self.assertEqual("bet365", upcoming[0]["bookmaker_team2"])

    def test_integration_value_detector_uses_synced_odds(self):
        os.environ[self.token_env] = "test-token"
        fixture_time = datetime.now(timezone.utc) + timedelta(hours=2)
        fixtures = [
            {
                "fixture_id": "fx-888",
                "home_name": "FURIA",
                "away_name": "TYLOO",
                "event_name": "ESL Pro League",
                "start_dt": fixture_time,
            }
        ]
        odds_payload = {
            "data": [
                {
                    "bookmaker": {"name": "pinnacle"},
                    "markets": [
                        {
                            "name": "Match Winner",
                            "odds": [
                                {"name": "home", "odds": 1.95},
                                {"name": "away", "odds": 1.90},
                            ],
                        }
                    ],
                }
            ]
        }
        with patch.object(self.sync, "_resolve_sport_id", return_value="cs2"), patch.object(
            self.sync, "_fetch_fixtures", return_value=(fixtures, "")
        ), patch.object(self.sync, "_fetch_fixture_odds", return_value=(odds_payload, "")):
            report = self.sync.sync_upcoming_odds()
        self.assertEqual(1, report["saved"])

        match = self.db.get_upcoming_matches()[0]
        detector = ValueDetector(
            {
                "model": {"min_value_pct": 1.0, "min_confidence": 50.0},
                "bankroll": {"total": 1000.0, "max_bet_pct": 3.0, "kelly_fraction": 0.25},
            }
        )
        pred = {"team1_win_prob": 70.0, "team2_win_prob": 30.0, "confidence": 70.0, "predicted_winner": 1}
        analysis = detector.analyze(
            prediction=pred,
            odds_team1=match.get("odds_team1"),
            odds_team2=match.get("odds_team2"),
            match=match,
        )
        self.assertIsNotNone(analysis)
        self.assertTrue(analysis["has_value"])
        self.assertGreaterEqual(len(analysis["value_bets"]), 1)
        self.assertEqual("pinnacle", analysis["value_bets"][0]["bookmaker"])


if __name__ == "__main__":
    unittest.main()

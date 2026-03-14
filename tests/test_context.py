import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ai.context import ContextCollector
from db.models import Database


class ContextCollectorTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "context.db")
        self.db = Database(self.db_path)
        self.base_config = {
            "llm": {
                "context_recent_matches": 5,
                "context_include_players": True,
                "context_include_h2h": True,
                "context_include_map_pool": True,
            }
        }

        self.db.upsert_team(1, "FURIA", ranking=9, win_rate=61.0, maps_played=200)
        self.db.upsert_team(2, "TYLOO", ranking=24, win_rate=54.0, maps_played=180)
        self.db.upsert_player(101, "KSCERATO", 1, rating=1.12, kd_ratio=1.15)
        self.db.upsert_player(102, "yuurih", 1, rating=1.10, kd_ratio=1.12)
        self.db.upsert_player(201, "JamYoung", 2, rating=1.08, kd_ratio=1.09)
        self.db.upsert_player(202, "Mercury", 2, rating=1.01, kd_ratio=1.02)
        self.db.upsert_team_map_stats(1, "Inferno", matches_played=30, wins=18, win_rate=60.0)
        self.db.upsert_team_map_stats(2, "Inferno", matches_played=25, wins=11, win_rate=44.0)

        base_dt = datetime.now(timezone.utc) - timedelta(days=2)
        for idx in range(3):
            match_dt = (base_dt - timedelta(days=idx)).isoformat(timespec="seconds")
            self.db.upsert_match(
                hltv_id=8000 + idx,
                date=match_dt,
                event_name="ESL Pro League",
                best_of=3,
                team1_id=1,
                team2_id=2,
                team1_score=2,
                team2_score=1,
                winner_id=1,
                status="completed",
            )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_collect_returns_string(self):
        collector = ContextCollector(self.db, self.base_config)
        out = collector.collect({"id": 1, "team1_id": 1, "team2_id": 2, "team1_name": "FURIA", "team2_name": "TYLOO"})
        self.assertIsInstance(out, str)
        self.assertTrue(len(out) > 30)

    def test_collect_handles_missing_team(self):
        collector = ContextCollector(self.db, self.base_config)
        out = collector.collect({"id": 99, "team1_id": 999, "team2_id": 998, "team1_name": "X", "team2_name": "Y"})
        self.assertIsInstance(out, str)
        self.assertTrue(len(out) > 0)

    def test_collect_includes_all_sections(self):
        collector = ContextCollector(self.db, self.base_config)
        out = collector.collect({"id": 1, "team1_id": 1, "team2_id": 2, "team1_name": "FURIA", "team2_name": "TYLOO"})
        self.assertIn("PERFIL DOS TIMES", out)
        self.assertIn("FORMA RECENTE", out)
        self.assertIn("H2H", out)
        self.assertIn("JOGADORES", out)
        self.assertIn("MAP POOL", out)

    def test_collect_respects_config_flags(self):
        collector = ContextCollector(
            self.db,
            {
                "llm": {
                    "context_recent_matches": 5,
                    "context_include_players": False,
                    "context_include_h2h": False,
                    "context_include_map_pool": False,
                }
            },
        )
        out = collector.collect({"id": 1, "team1_id": 1, "team2_id": 2, "team1_name": "FURIA", "team2_name": "TYLOO"})
        self.assertNotIn("JOGADORES", out)
        self.assertNotIn("H2H", out)
        self.assertNotIn("MAP POOL", out)
        self.assertIn("PERFIL DOS TIMES", out)

    def test_compact_output(self):
        collector = ContextCollector(self.db, self.base_config)
        out = collector.collect({"id": 1, "team1_id": 1, "team2_id": 2, "team1_name": "FURIA", "team2_name": "TYLOO"})
        self.assertLess(len(out.split()), 800)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from analysis.daily_top5_audit import DailyTop5Auditor
from db.models import Database


class _DummyNotifier:
    def __init__(self):
        self.audit_reports = []

    def daily_top5_audit_report(self, summary: dict):
        self.audit_reports.append(summary)


class DailyTop5AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "test.db")
        self.db = Database(self.db_path)
        self.notifier = _DummyNotifier()
        self.config = {
            "scheduler": {
                "timezone": "UTC",
                "daily_top5_audit_enabled": True,
                "daily_audit_hour": 0,
                "daily_audit_minute": 30,
                "audit_match_window_hours": 6,
                "audit_pending_max_days": 3,
            }
        }
        self.auditor = DailyTop5Auditor(self.db, self.config, self.notifier)

        self.db.upsert_team(1, "FURIA", ranking=10)
        self.db.upsert_team(2, "TYLOO", ranking=20)
        self.db.upsert_team(3, "MOUZ", ranking=5)
        self.db.upsert_team(4, "Spirit", ranking=3)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _make_pick(self, match: dict, winner_side: int = 1, score: float = 99.0) -> dict:
        predicted_winner_name = match["team1_name"] if winner_side == 1 else match["team2_name"]
        predicted_winner_id = match["team1_id"] if winner_side == 1 else match["team2_id"]
        return {
            "match": match,
            "prediction": {
                "predicted_winner": winner_side,
                "team1_win_prob": 60.0 if winner_side == 1 else 40.0,
                "team2_win_prob": 40.0 if winner_side == 1 else 60.0,
                "confidence": 70.0,
            },
            "analysis": {
                "has_value": True,
                "value_bets": [],
            },
            "best_vb": {
                "side": "team1" if winner_side == 1 else "team2",
                "odds": 2.0,
                "bookmaker": "pinnacle",
                "value_pct": 10.0,
                "expected_value": 20.0,
                "suggested_stake": 30.0,
            },
            "score": score,
            "predicted_winner_id": predicted_winner_id,
            "predicted_winner_name": predicted_winner_name,
        }

    def test_capture_daily_top5_only_once_per_day(self):
        dt = datetime(2026, 3, 13, 14, 0, tzinfo=timezone.utc)
        match_id = self.db.upsert_match(
            hltv_id=1001,
            date=dt.replace(tzinfo=None).isoformat(timespec="seconds"),
            event_name="Test Event",
            team1_id=1,
            team2_id=2,
            status="upcoming",
        )
        match = self.db.get_match_result_by_id(match_id)
        pick = self._make_pick(match, winner_side=1, score=88.0)

        first = self.auditor.capture_daily_top5(
            picks=[pick],
            requested_top=5,
            total_candidates=10,
            candidates_with_odds=4,
            now_local=dt,
        )
        second = self.auditor.capture_daily_top5(
            picks=[self._make_pick(match, winner_side=2, score=77.0)],
            requested_top=5,
            total_candidates=9,
            candidates_with_odds=3,
            now_local=dt + timedelta(hours=2),
        )

        self.assertTrue(first["created"])
        self.assertFalse(second["created"])
        run = self.db.get_daily_top5_run(dt.date().isoformat())
        self.assertIsNotNone(run)
        items = self.db.get_daily_top5_items(int(run["id"]))
        self.assertEqual(1, len(items))
        self.assertEqual("FURIA", items[0]["predicted_winner_name"])

    def test_audit_resolves_by_match_id(self):
        dt = datetime(2026, 3, 13, 16, 0, tzinfo=timezone.utc)
        match_id = self.db.upsert_match(
            hltv_id=1002,
            date=dt.replace(tzinfo=None).isoformat(timespec="seconds"),
            event_name="ESL Pro League",
            team1_id=1,
            team2_id=2,
            team1_score=2,
            team2_score=1,
            winner_id=1,
            status="completed",
        )
        match = self.db.get_match_result_by_id(match_id)
        pick = self._make_pick(match, winner_side=1)
        self.auditor.capture_daily_top5(
            picks=[pick],
            requested_top=5,
            total_candidates=1,
            candidates_with_odds=1,
            now_local=dt,
        )

        summary = self.auditor.audit_date(dt.date().isoformat(), send_notification=False)
        self.assertTrue(summary["found"])
        self.assertEqual(1, summary["wins"])
        self.assertEqual(0, summary["pending"])

        run = self.db.get_daily_top5_run(dt.date().isoformat())
        items = self.db.get_daily_top5_items(int(run["id"]))
        self.assertEqual("win", items[0]["outcome_status"])
        self.assertEqual("match_id", items[0]["resolution_method"])

    def test_audit_resolves_by_team_window_fallback(self):
        dt_completed = datetime(2026, 3, 13, 18, 0, tzinfo=timezone.utc)
        self.db.upsert_match(
            hltv_id=1003,
            date=dt_completed.replace(tzinfo=None).isoformat(timespec="seconds"),
            event_name="Blast Group A",
            team1_id=3,
            team2_id=4,
            team1_score=1,
            team2_score=2,
            winner_id=4,
            status="completed",
        )

        pick = {
            "match": {
                "id": None,
                "date": (dt_completed + timedelta(minutes=20)).replace(tzinfo=None).isoformat(timespec="seconds"),
                "event_name": "Blast Group A",
                "team1_id": 3,
                "team2_id": 4,
                "team1_name": "MOUZ",
                "team2_name": "Spirit",
            },
            "prediction": {
                "predicted_winner": 2,
                "team1_win_prob": 45.0,
                "team2_win_prob": 55.0,
                "confidence": 61.0,
            },
            "analysis": {"has_value": True, "value_bets": []},
            "best_vb": {
                "side": "team2",
                "odds": 1.95,
                "bookmaker": "pinnacle",
                "value_pct": 8.0,
                "expected_value": 14.0,
                "suggested_stake": 20.0,
            },
            "score": 80.0,
        }

        capture_dt = datetime(2026, 3, 13, 10, 0, tzinfo=timezone.utc)
        self.auditor.capture_daily_top5(
            picks=[pick],
            requested_top=5,
            total_candidates=2,
            candidates_with_odds=1,
            now_local=capture_dt,
        )

        summary = self.auditor.audit_date(capture_dt.date().isoformat(), send_notification=False)
        self.assertEqual(1, summary["wins"])
        run = self.db.get_daily_top5_run(capture_dt.date().isoformat())
        items = self.db.get_daily_top5_items(int(run["id"]))
        self.assertEqual("teams_window", items[0]["resolution_method"])
        self.assertEqual("win", items[0]["outcome_status"])

    def test_scheduler_runs_only_once_per_day(self):
        day = datetime(2026, 3, 14, 0, 31, tzinfo=timezone.utc)
        run_date = (day.date() - timedelta(days=1)).isoformat()
        match_id = self.db.upsert_match(
            hltv_id=1004,
            date=(day - timedelta(days=1, hours=2)).replace(tzinfo=None).isoformat(timespec="seconds"),
            event_name="Daily Cup",
            team1_id=1,
            team2_id=2,
            team1_score=2,
            team2_score=0,
            winner_id=1,
            status="completed",
        )
        match = self.db.get_match_result_by_id(match_id)
        self.auditor.capture_daily_top5(
            picks=[self._make_pick(match, winner_side=1)],
            requested_top=5,
            total_candidates=3,
            candidates_with_odds=1,
            now_local=day - timedelta(days=1),
        )
        self.assertEqual(run_date, (day.date() - timedelta(days=1)).isoformat())

        first = self.auditor.run_if_due(now_local=day)
        second = self.auditor.run_if_due(now_local=day + timedelta(minutes=5))

        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertEqual(1, len(self.notifier.audit_reports))


if __name__ == "__main__":
    unittest.main()


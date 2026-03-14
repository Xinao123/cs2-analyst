"""
Daily Top 5 capture and audit workflow.

1. Freeze first Top 5 (value-only) per local day
2. Audit yesterday's picks against completed matches
3. Send a daily Telegram report with win/loss/pending
"""

from __future__ import annotations

import logging
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from db.models import Database
from utils.time_utils import parse_datetime_to_utc, to_storage_utc_iso

logger = logging.getLogger(__name__)

_AUDIT_STATE_KEY = "daily_top5_audit_last_local_date"


class DailyTop5Auditor:
    def __init__(self, db: Database, config: dict, notifier):
        self.db = db
        self.config = config
        self.notifier = notifier

        scheduler_cfg = config.get("scheduler", {})
        self.enabled = bool(scheduler_cfg.get("daily_top5_audit_enabled", True))
        self.daily_audit_hour = int(scheduler_cfg.get("daily_audit_hour", 0))
        self.daily_audit_minute = int(scheduler_cfg.get("daily_audit_minute", 30))
        self.audit_match_window_hours = max(1, int(scheduler_cfg.get("audit_match_window_hours", 6)))
        self.audit_pending_max_days = max(1, int(scheduler_cfg.get("audit_pending_max_days", 3)))

        tz_name = str(scheduler_cfg.get("timezone", "America/Sao_Paulo") or "America/Sao_Paulo")
        try:
            self.tz = ZoneInfo(tz_name)
            self.tz_name = tz_name
        except Exception:
            self.tz = timezone.utc
            self.tz_name = "UTC"
            logger.warning("[AUDIT] Timezone invalida (%s), usando UTC.", tz_name)

    def local_now(self) -> datetime:
        return datetime.now(self.tz)

    def capture_daily_top5(
        self,
        picks: list[dict],
        requested_top: int,
        total_candidates: int,
        candidates_with_odds: int,
        now_local: datetime | None = None,
    ) -> dict:
        now_local = now_local or self.local_now()
        run_date = now_local.date().isoformat()

        existing = self.db.get_daily_top5_run(run_date)
        if existing:
            return {
                "created": False,
                "run_id": int(existing.get("id", 0)),
                "run_date": run_date,
                "status": str(existing.get("status", "")),
                "items": len(self.db.get_daily_top5_items(int(existing.get("id", 0)))),
            }

        status = "captured" if picks else "no_picks"
        run_id = self.db.create_daily_top5_run(
            run_date=run_date,
            requested_top=requested_top,
            total_candidates=total_candidates,
            candidates_with_odds=candidates_with_odds,
            status=status,
        )

        if run_id and picks:
            items = []
            for rank, item in enumerate(picks, start=1):
                items.append(self._snapshot_item_from_pick(item, rank))
            self.db.save_daily_top5_items(run_id, items)

        return {
            "created": True,
            "run_id": run_id,
            "run_date": run_date,
            "status": status,
            "items": len(picks),
        }

    def run_if_due(self, now_local: datetime | None = None) -> dict | None:
        if not self.enabled:
            return None

        now_local = now_local or self.local_now()
        if not self._is_due_time(now_local):
            return None

        today_str = now_local.date().isoformat()
        if self.db.get_state(_AUDIT_STATE_KEY, "") == today_str:
            return None

        target_date = (now_local.date() - timedelta(days=1)).isoformat()
        summary = self.audit_date(target_date, send_notification=True)

        # Re-check recent pending runs silently.
        for days_back in range(2, self.audit_pending_max_days + 1):
            older_date = (now_local.date() - timedelta(days=days_back)).isoformat()
            self.audit_date(older_date, send_notification=False)

        self.db.set_state(_AUDIT_STATE_KEY, today_str)
        return summary

    def audit_date(self, run_date: str, send_notification: bool = False) -> dict:
        run = self.db.get_daily_top5_run(run_date)
        if not run:
            summary = {
                "run_date": run_date,
                "found": False,
                "status": "missing",
                "items": [],
                "wins": 0,
                "losses": 0,
                "pending": 0,
                "resolved": 0,
                "total": 0,
                "accuracy": 0.0,
            }
            if send_notification:
                self.notifier.daily_top5_audit_report(summary)
            return summary

        run_id = int(run.get("id", 0))
        items = self.db.get_daily_top5_items(run_id)

        if items:
            for item in items:
                current_status = str(item.get("outcome_status", "pending") or "pending")
                if current_status in {"win", "loss"}:
                    continue
                resolved = self._resolve_item_result(item)
                if not resolved:
                    continue

                official_pick_winner_id = _safe_int(
                    item.get("official_pick_winner_id", item.get("predicted_winner_id"))
                )
                actual_winner_id = _safe_int(resolved.get("actual_winner_id"))
                outcome_status = (
                    "win"
                    if official_pick_winner_id > 0 and official_pick_winner_id == actual_winner_id
                    else "loss"
                )

                self.db.update_daily_top5_item_outcome(
                    item_id=int(item.get("id", 0)),
                    outcome_status=outcome_status,
                    actual_winner_id=actual_winner_id,
                    resolved_match_id=_safe_int(resolved.get("resolved_match_id")),
                    resolution_method=str(resolved.get("resolution_method", "")),
                )
                self._track_item_clv(
                    item=item,
                    resolved_match_id=_safe_int(resolved.get("resolved_match_id")),
                )

        items = self.db.get_daily_top5_items(run_id)
        wins = sum(1 for i in items if str(i.get("outcome_status", "")) == "win")
        losses = sum(1 for i in items if str(i.get("outcome_status", "")) == "loss")
        pending = sum(1 for i in items if str(i.get("outcome_status", "pending")) == "pending")
        resolved = wins + losses
        total = len(items)
        accuracy = (wins / resolved * 100.0) if resolved else 0.0
        model_vs_official_divergences = sum(
            1
            for i in items
            if _safe_int(i.get("model_winner_id")) > 0
            and _safe_int(i.get("official_pick_winner_id")) > 0
            and _safe_int(i.get("model_winner_id")) != _safe_int(i.get("official_pick_winner_id"))
        )
        recent_rows = self.db.get_recent_resolved_predictions(days=7)
        recent_resolved = 0
        recent_wins = 0
        for row in recent_rows:
            pred_id = _safe_int(row.get("official_pick_winner_id", row.get("predicted_winner_id")))
            actual_id = _safe_int(row.get("actual_winner_id"))
            if pred_id <= 0 or actual_id <= 0:
                continue
            recent_resolved += 1
            if pred_id == actual_id:
                recent_wins += 1
        trend_7d = (recent_wins / recent_resolved * 100.0) if recent_resolved else 0.0
        avg_clv_30d = float(self.db.get_avg_clv(days=30))

        status = "audited_empty"
        if total > 0:
            status = "audited_pending" if pending > 0 else "audited_complete"
        self.db.update_daily_top5_run_audited(run_id, status=status)

        summary = {
            "run_date": run_date,
            "found": True,
            "status": status,
            "items": items,
            "wins": wins,
            "losses": losses,
            "pending": pending,
            "resolved": resolved,
            "total": total,
            "accuracy": round(accuracy, 1),
            "model_vs_official_divergences": model_vs_official_divergences,
            "trend_7d": round(trend_7d, 1),
            "avg_clv_30d": round(avg_clv_30d, 2),
        }
        if send_notification:
            self.notifier.daily_top5_audit_report(summary)
        return summary

    def _snapshot_item_from_pick(self, pick: dict, rank: int) -> dict:
        match = pick.get("match", {}) or {}
        prediction = pick.get("prediction", {}) or {}
        best_vb = pick.get("best_vb", {}) or {}
        if not best_vb:
            analysis = pick.get("analysis", {}) or {}
            value_bets = analysis.get("value_bets", []) if isinstance(analysis, dict) else []
            if value_bets:
                best_vb = max(value_bets, key=lambda vb: _safe_float(vb.get("value_pct")))

        predicted_winner = _safe_int(prediction.get("predicted_winner"))
        team1_id = _safe_int(match.get("team1_id"))
        team2_id = _safe_int(match.get("team2_id"))
        model_winner_id = team1_id if predicted_winner == 1 else team2_id
        model_winner_name = (
            str(match.get("team1_name", "")) if predicted_winner == 1 else str(match.get("team2_name", ""))
        )

        official_side = str(
            pick.get(
                "official_pick_side",
                best_vb.get("side", "team1" if predicted_winner == 1 else "team2"),
            )
        ).lower()
        if official_side == "team2":
            official_pick_winner_id = team2_id
            official_pick_winner_name = str(match.get("team2_name", ""))
        else:
            official_pick_winner_id = team1_id
            official_pick_winner_name = str(match.get("team1_name", ""))

        return {
            "rank": rank,
            "match_id": _safe_int(match.get("id")) or None,
            "match_date": str(match.get("date", "")),
            "event_name": str(match.get("event_name", "")),
            "team1_id": team1_id or None,
            "team2_id": team2_id or None,
            "team1_name": str(match.get("team1_name", "")),
            "team2_name": str(match.get("team2_name", "")),
            "model_winner_id": model_winner_id or None,
            "model_winner_name": model_winner_name,
            "official_pick_winner_id": official_pick_winner_id or None,
            "official_pick_winner_name": official_pick_winner_name,
            "pick_source": str(pick.get("pick_source", "value")),
            # Backward-compatible aliases.
            "predicted_winner_id": official_pick_winner_id or None,
            "predicted_winner_name": official_pick_winner_name,
            "team1_win_prob": _safe_float(prediction.get("team1_win_prob")),
            "team2_win_prob": _safe_float(prediction.get("team2_win_prob")),
            "confidence": _safe_float(prediction.get("confidence")),
            "score": _safe_float(pick.get("score")),
            "odds": _safe_float(best_vb.get("odds")),
            "bookmaker": str(best_vb.get("bookmaker", "")),
            "value_pct": _safe_float(best_vb.get("value_pct")),
            "expected_value": _safe_float(best_vb.get("expected_value")),
            "outcome_status": "pending",
        }

    def _resolve_item_result(self, item: dict) -> dict | None:
        match_id = _safe_int(item.get("match_id"))
        if match_id > 0:
            by_id = self.db.get_match_result_by_id(match_id)
            if by_id and str(by_id.get("status", "")) == "completed" and _safe_int(by_id.get("winner_id")) > 0:
                return {
                    "actual_winner_id": _safe_int(by_id.get("winner_id")),
                    "resolved_match_id": _safe_int(by_id.get("id")),
                    "resolution_method": "match_id",
                }

        return self._resolve_by_teams_and_window(item)

    def _resolve_by_teams_and_window(self, item: dict) -> dict | None:
        target_dt = _parse_datetime(item.get("match_date"))
        if not target_dt:
            return None

        window = timedelta(hours=self.audit_match_window_hours)
        start_dt = target_dt - window
        end_dt = target_dt + window
        candidates = self.db.list_completed_matches_between(
            _to_storage_iso(start_dt),
            _to_storage_iso(end_dt),
        )
        if not candidates:
            return None

        snap_t1 = _normalize_name(item.get("team1_name", ""))
        snap_t2 = _normalize_name(item.get("team2_name", ""))
        snap_event = str(item.get("event_name", ""))
        if not snap_t1 or not snap_t2:
            return None

        best = None
        best_score = float("-inf")

        for cand in candidates:
            winner_id = _safe_int(cand.get("winner_id"))
            if winner_id <= 0:
                continue

            cand_t1 = _normalize_name(cand.get("team1_name", ""))
            cand_t2 = _normalize_name(cand.get("team2_name", ""))
            if {snap_t1, snap_t2} != {cand_t1, cand_t2}:
                continue

            score = 0.0
            if snap_t1 == cand_t1 and snap_t2 == cand_t2:
                score += 2.0
            else:
                score += 1.0

            cand_dt = _parse_datetime(cand.get("date"))
            if cand_dt:
                diff_h = abs((_as_utc(cand_dt) - _as_utc(target_dt)).total_seconds()) / 3600.0
                score += max(0.0, 1.5 - (diff_h / max(self.audit_match_window_hours, 1)))

            score += _event_similarity(snap_event, str(cand.get("event_name", "")))

            if score > best_score:
                best_score = score
                best = cand

        if not best:
            return None

        return {
            "actual_winner_id": _safe_int(best.get("winner_id")),
            "resolved_match_id": _safe_int(best.get("id")),
            "resolution_method": "teams_window",
        }

    def _track_item_clv(self, item: dict, resolved_match_id: int):
        open_odds = _safe_float(item.get("odds"))
        if open_odds <= 1.0:
            return

        team1_id = _safe_int(item.get("team1_id"))
        team2_id = _safe_int(item.get("team2_id"))
        pick_id = _safe_int(item.get("official_pick_winner_id", item.get("predicted_winner_id")))
        if pick_id <= 0:
            return

        if pick_id == team1_id:
            side = "team1"
        elif pick_id == team2_id:
            side = "team2"
        else:
            return

        target_match_id = resolved_match_id or _safe_int(item.get("match_id"))
        if target_match_id <= 0:
            return

        latest = self.db.get_match_odds_latest(target_match_id)
        if not latest:
            return
        close_odds = _safe_float(latest.get("odds_team1" if side == "team1" else "odds_team2"))
        if close_odds <= 1.0:
            return

        try:
            self.db.save_clv(
                match_id=target_match_id,
                side=side,
                open_odds=open_odds,
                close_odds=close_odds,
            )
        except Exception as exc:
            logger.debug("[AUDIT] Falha ao salvar CLV item=%s: %s", item.get("id"), exc)

    def _is_due_time(self, now_local: datetime) -> bool:
        if now_local.hour > self.daily_audit_hour:
            return True
        if now_local.hour < self.daily_audit_hour:
            return False
        return now_local.minute >= self.daily_audit_minute


def _tokenize(value: str) -> list[str]:
    text = str(value or "").lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return [tok for tok in re.split(r"[^a-z0-9]+", text) if tok]


def _event_similarity(a: str, b: str) -> float:
    ta = set(_tokenize(a))
    tb = set(_tokenize(b))
    if not ta or not tb:
        return 0.0
    inter = len(ta.intersection(tb))
    if inter == 0:
        return 0.0
    return inter / max(len(ta), len(tb))


def _normalize_name(value) -> str:
    text = str(value or "").lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]", "", text)


def _parse_datetime(value) -> datetime | None:
    return parse_datetime_to_utc(
        value,
        logger=logger,
        context="daily_top5_audit.parse_datetime",
    )


def _as_utc(value: datetime) -> datetime:
    parsed = parse_datetime_to_utc(
        value,
        logger=logger,
        context="daily_top5_audit.as_utc",
    )
    return parsed if parsed is not None else datetime.now(timezone.utc)


def _to_storage_iso(value: datetime) -> str:
    # Keep compatibility with existing DB dates (naive ISO strings).
    return to_storage_utc_iso(
        value,
        logger=logger,
        context="daily_top5_audit.to_storage_iso",
    )


def _safe_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value) -> float:
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return 0.0

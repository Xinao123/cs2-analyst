"""Context collection for LLM prompts using local SQLite data."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from analysis.features import (
    calc_format_win_rate,
    calc_result_volatility,
    calc_rust_days,
    calc_venue_win_rate,
    calc_window_win_rate,
)
from db.models import Database
from utils.time_utils import parse_datetime_to_utc

logger = logging.getLogger(__name__)


class ContextCollector:
    """Build compact contextual text blocks from local database data."""

    def __init__(self, db: Database, config: dict):
        self.db = db
        llm_cfg = config.get("llm", {})
        self.recent_matches = max(1, int(llm_cfg.get("context_recent_matches", 5)))
        self.include_players = bool(llm_cfg.get("context_include_players", True))
        self.include_h2h = bool(llm_cfg.get("context_include_h2h", True))
        self.include_map_pool = bool(llm_cfg.get("context_include_map_pool", True))

    def collect(self, match: dict) -> str:
        """Collect contextual information for a single upcoming match."""
        team1_id = int(match.get("team1_id", 0) or 0)
        team2_id = int(match.get("team2_id", 0) or 0)
        team1_name = str(match.get("team1_name", "Team1"))
        team2_name = str(match.get("team2_name", "Team2"))

        sections: list[str] = []

        team1 = self.db.get_team(team1_id)
        team2 = self.db.get_team(team2_id)
        if team1 and team2:
            sections.append(self._build_team_profiles(team1, team2))

        recent_days = 180
        team1_recent = self.db.get_team_recent_matches(team1_id, limit=max(self.recent_matches, 40), days=recent_days)
        team2_recent = self.db.get_team_recent_matches(team2_id, limit=max(self.recent_matches, 40), days=recent_days)
        if team1_recent or team2_recent:
            sections.append(
                self._build_recent_form(
                    team1_name=team1_name,
                    team1_id=team1_id,
                    team1_recent=team1_recent,
                    team2_name=team2_name,
                    team2_id=team2_id,
                    team2_recent=team2_recent,
                )
            )
            sections.append(
                self._build_advanced_metrics(
                    match=match,
                    team1_name=team1_name,
                    team1_id=team1_id,
                    team1_recent=team1_recent,
                    team2_name=team2_name,
                    team2_id=team2_id,
                    team2_recent=team2_recent,
                )
            )

        if self.include_h2h:
            h2h = self.db.get_h2h(team1_id, team2_id)
            if h2h:
                sections.append(self._build_h2h(team1_name, team2_name, h2h))

        if self.include_players:
            team1_players = self.db.get_team_players(team1_id)
            team2_players = self.db.get_team_players(team2_id)
            if team1_players or team2_players:
                sections.append(
                    self._build_player_stats(
                        team1_name=team1_name,
                        team1_players=team1_players,
                        team2_name=team2_name,
                        team2_players=team2_players,
                    )
                )

        if self.include_map_pool:
            team1_maps = self.db.get_team_map_stats(team1_id)
            team2_maps = self.db.get_team_map_stats(team2_id)
            if team1_maps or team2_maps:
                sections.append(
                    self._build_map_pool(
                        team1_name=team1_name,
                        team1_maps=team1_maps,
                        team2_name=team2_name,
                        team2_maps=team2_maps,
                    )
                )

        if not sections:
            logger.debug("[CTX] contexto vazio para match_id=%s", match.get("id"))
            return "CONTEXTO: dados insuficientes no banco local."

        logger.debug(
            "[CTX] contexto montado match_id=%s sections=%s",
            match.get("id"),
            len(sections),
        )
        return "\n\n".join(sections)

    def collect_for_audit(self, summary: dict) -> str:
        """Build compact historical context text for daily audit summaries."""
        run_date = str(summary.get("run_date", ""))
        wins = int(summary.get("wins", 0))
        losses = int(summary.get("losses", 0))
        pending = int(summary.get("pending", 0))
        accuracy = float(summary.get("accuracy", 0.0))

        start, end = self._audit_window(run_date)
        recent_completed = self.db.list_completed_matches_between(start, end)
        recent_resolved = self.db.get_recent_resolved_predictions(days=7)
        resolved_7d = 0
        wins_7d = 0
        for item in recent_resolved:
            pred = int(item.get("official_pick_winner_id") or item.get("predicted_winner_id") or 0)
            actual = int(item.get("actual_winner_id") or 0)
            if pred <= 0 or actual <= 0:
                continue
            resolved_7d += 1
            if pred == actual:
                wins_7d += 1
        trend_7d = (wins_7d / max(1, resolved_7d) * 100.0) if resolved_7d else 0.0
        avg_clv_30d = float(self.db.get_avg_clv(days=30))

        return (
            "CONTEXTO AUDITORIA:\n"
            f"  Janela analisada: {start} -> {end}\n"
            f"  Resultado carteira: {wins}W/{losses}L/{pending}P (acc {accuracy:.1f}%)\n"
            f"  Partidas completadas na janela: {len(recent_completed)}\n"
            f"  Tendencia 7d (resolvidas): {trend_7d:.1f}% ({wins_7d}/{resolved_7d})\n"
            f"  CLV medio 30d: {avg_clv_30d:+.2f}%"
        )

    def _build_team_profiles(self, team1: dict, team2: dict) -> str:
        return (
            "PERFIL DOS TIMES:\n"
            f"  {team1.get('name', '?')}: Rank #{team1.get('ranking', '?')} | "
            f"Win rate: {float(team1.get('win_rate', 0.0)):.0f}% | "
            f"Maps: {int(team1.get('maps_played', 0) or 0)}\n"
            f"  {team2.get('name', '?')}: Rank #{team2.get('ranking', '?')} | "
            f"Win rate: {float(team2.get('win_rate', 0.0)):.0f}% | "
            f"Maps: {int(team2.get('maps_played', 0) or 0)}"
        )

    def _build_recent_form(
        self,
        team1_name: str,
        team1_id: int,
        team1_recent: list[dict],
        team2_name: str,
        team2_id: int,
        team2_recent: list[dict],
    ) -> str:
        def _format_matches(name: str, team_id: int, matches: list[dict]) -> str:
            if not matches:
                return f"  {name}: sem dados recentes"
            wins = sum(1 for item in matches if int(item.get("winner_id", 0) or 0) == team_id)
            losses = len(matches) - wins
            parts = []
            for item in matches[:5]:
                team1_match = int(item.get("team1_id", 0) or 0)
                opponent = item.get("team2_name") if team1_match == team_id else item.get("team1_name")
                is_win = int(item.get("winner_id", 0) or 0) == team_id
                result = "W" if is_win else "L"
                score = f"{item.get('team1_score', '?')}-{item.get('team2_score', '?')}"
                parts.append(f"{result} vs {opponent or '?'} ({score})")
            return f"  {name}: {wins}W-{losses}L -> {' | '.join(parts)}"

        return (
            f"FORMA RECENTE (ultimos {self.recent_matches} jogos):\n"
            f"{_format_matches(team1_name, team1_id, team1_recent)}\n"
            f"{_format_matches(team2_name, team2_id, team2_recent)}"
        )

    def _build_advanced_metrics(
        self,
        match: dict,
        team1_name: str,
        team1_id: int,
        team1_recent: list[dict],
        team2_name: str,
        team2_id: int,
        team2_recent: list[dict],
    ) -> str:
        ref_dt = parse_datetime_to_utc(match.get("date"), logger=logger, context="ctx.match_date")
        if ref_dt is None:
            ref_dt = datetime.now(timezone.utc)

        best_of = int(match.get("best_of", 1) or 1)
        is_lan = bool(match.get("is_lan", 0))

        t1_rust = calc_rust_days(team1_recent, as_of_dt=ref_dt)
        t2_rust = calc_rust_days(team2_recent, as_of_dt=ref_dt)

        t1_wr_7 = calc_window_win_rate(team1_recent, team1_id, 7, as_of_dt=ref_dt)
        t2_wr_7 = calc_window_win_rate(team2_recent, team2_id, 7, as_of_dt=ref_dt)
        t1_wr_30 = calc_window_win_rate(team1_recent, team1_id, 30, as_of_dt=ref_dt)
        t2_wr_30 = calc_window_win_rate(team2_recent, team2_id, 30, as_of_dt=ref_dt)

        t1_bo = calc_format_win_rate(team1_recent, team1_id, best_of)
        t2_bo = calc_format_win_rate(team2_recent, team2_id, best_of)
        t1_venue = calc_venue_win_rate(team1_recent, team1_id, is_lan)
        t2_venue = calc_venue_win_rate(team2_recent, team2_id, is_lan)

        t1_vol = calc_result_volatility(team1_recent, team1_id)
        t2_vol = calc_result_volatility(team2_recent, team2_id)

        t1_side = self.db.get_team_side_stats(team1_id, days=180)
        t2_side = self.db.get_team_side_stats(team2_id, days=180)
        t1_ct = float(t1_side.get("ct_win_rate", 0.5))
        t2_ct = float(t2_side.get("ct_win_rate", 0.5))
        t1_t = float(t1_side.get("t_win_rate", 0.5))
        t2_t = float(t2_side.get("t_win_rate", 0.5))

        return (
            "METRICAS AVANCADAS:\n"
            f"  Rust(d): {team1_name}={t1_rust:.1f} | {team2_name}={t2_rust:.1f}\n"
            f"  WR 7d/30d: {team1_name}={t1_wr_7:.2f}/{t1_wr_30:.2f} | {team2_name}={t2_wr_7:.2f}/{t2_wr_30:.2f}\n"
            f"  WR BO{best_of}: {team1_name}={t1_bo:.2f} | {team2_name}={t2_bo:.2f}\n"
            f"  WR {'LAN' if is_lan else 'Online'}: {team1_name}={t1_venue:.2f} | {team2_name}={t2_venue:.2f}\n"
            f"  Volatilidade: {team1_name}={t1_vol:.3f} | {team2_name}={t2_vol:.3f}\n"
            f"  Side CT/T: {team1_name}={t1_ct:.2f}/{t1_t:.2f} | {team2_name}={t2_ct:.2f}/{t2_t:.2f}"
        )

    def _build_h2h(self, team1_name: str, team2_name: str, h2h_matches: list[dict]) -> str:
        if not h2h_matches:
            return f"H2H: sem confrontos registrados entre {team1_name} e {team2_name}"

        lines = [f"H2H ({team1_name} vs {team2_name}): {len(h2h_matches)} jogos"]
        for item in h2h_matches[:5]:
            date_text = str(item.get("date", "?"))[:10]
            event_name = item.get("event_name", "?")
            score = f"{item.get('team1_score', '?')}-{item.get('team2_score', '?')}"
            lines.append(f"  {date_text}: {score} ({event_name})")
        return "\n".join(lines)

    def _build_player_stats(
        self,
        team1_name: str,
        team1_players: list[dict],
        team2_name: str,
        team2_players: list[dict],
    ) -> str:
        def _fmt(players: list[dict]) -> str:
            if not players:
                return "  (sem dados)"
            lines = []
            sorted_players = sorted(
                players,
                key=lambda item: float(item.get("rating", 0.0)),
                reverse=True,
            )
            for player in sorted_players[:5]:
                lines.append(
                    f"  {player.get('name', '?')}: "
                    f"rating {float(player.get('rating', 0.0)):.2f} | "
                    f"K/D {float(player.get('kd_ratio', 0.0)):.2f}"
                )
            return "\n".join(lines)

        return (
            "JOGADORES:\n"
            f"{team1_name}:\n{_fmt(team1_players)}\n"
            f"{team2_name}:\n{_fmt(team2_players)}"
        )

    def _build_map_pool(
        self,
        team1_name: str,
        team1_maps: list[dict],
        team2_name: str,
        team2_maps: list[dict],
    ) -> str:
        def _fmt(maps: list[dict]) -> str:
            if not maps:
                return "  (sem dados)"
            sorted_maps = sorted(
                maps,
                key=lambda item: float(item.get("win_rate", 0.0)),
                reverse=True,
            )
            lines = []
            for item in sorted_maps[:5]:
                lines.append(
                    f"  {item.get('map_name', '?')}: "
                    f"{float(item.get('win_rate', 0.0)):.0f}% WR "
                    f"({int(item.get('matches_played', 0) or 0)} jogos)"
                )
            return "\n".join(lines)

        return (
            "MAP POOL:\n"
            f"{team1_name}:\n{_fmt(team1_maps)}\n"
            f"{team2_name}:\n{_fmt(team2_maps)}"
        )

    def _audit_window(self, run_date: str) -> tuple[str, str]:
        target = parse_datetime_to_utc(run_date, logger=logger, context="ctx.audit_run_date")
        if target is None:
            now = datetime.now(timezone.utc)
            start = (now - timedelta(days=7)).isoformat(timespec="seconds")
            end = now.isoformat(timespec="seconds")
            return start, end
        end = target + timedelta(days=1)
        start = target - timedelta(days=7)
        return start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")

"""
Feature Engineering â€” Extrai features numÃ©ricas para o modelo.

Cada partida Ã© transformada em um vetor de features que captura:
- DiferenÃ§a de skill entre os times
- Forma recente
- HistÃ³rico head-to-head
- Contexto (BO1/BO3, LAN/online, tier do evento)
"""

import logging
import re

import numpy as np

from db.models import Database

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extrai features de uma partida para o modelo ML."""

    def __init__(self, db: Database, config: dict):
        self.db = db
        model_cfg = config.get("model", {})
        self.min_matches = model_cfg.get("min_matches", 10)
        self.form_window = model_cfg.get("form_window_days", 90)
        # Inferencia ao vivo segue conservadora; treino pode ser permissivo.
        self.live_min_recent_matches = model_cfg.get("live_min_recent_matches", 3)
        self.train_min_recent_matches = model_cfg.get("train_min_recent_matches", 0)
        self.use_player_features = bool(model_cfg.get("use_player_features", False))
        self.exclude_synthetic_teams = bool(model_cfg.get("exclude_synthetic_teams", True))
        self.exclude_academy_teams = bool(model_cfg.get("exclude_academy_teams", True))

    def extract(
        self,
        match: dict,
        min_recent_matches: int | None = None,
        as_of_date=None,
        for_training: bool = False,
    ) -> dict | None:
        """
        Extrai features de uma partida.

        Args:
            match: dict com team1_id, team2_id, best_of, event_tier, is_lan
            min_recent_matches: minimo de jogos recentes por time exigido
            as_of_date: recorte temporal para extração causal (treino)
            for_training: ativa modo de treino (mais conservador)

        Returns:
            dict de features numÃ©ricas ou None se dados insuficientes
        """
        t1_id = int(match["team1_id"])
        t2_id = int(match["team2_id"])

        if self.exclude_synthetic_teams and (t1_id <= 0 or t2_id <= 0):
            return None

        t1 = self.db.get_team(t1_id)
        t2 = self.db.get_team(t2_id)

        if not t1 or not t2:
            return None

        if self.exclude_academy_teams:
            if _is_academy_name(t1.get("name", "")) or _is_academy_name(t2.get("name", "")):
                return None

        if as_of_date is not None:
            t1_recent = self.db.get_team_recent_matches_before(
                t1_id,
                before_date=as_of_date,
                limit=20,
                days=self.form_window,
            )
            t2_recent = self.db.get_team_recent_matches_before(
                t2_id,
                before_date=as_of_date,
                limit=20,
                days=self.form_window,
            )
            h2h = self.db.get_h2h_before(t1_id, t2_id, before_date=as_of_date, limit=10)
            t1_maps = {
                m["map_name"]: m
                for m in self.db.get_team_map_stats_before(
                    t1_id,
                    before_date=as_of_date,
                    days=max(120, int(self.form_window)),
                )
            }
            t2_maps = {
                m["map_name"]: m
                for m in self.db.get_team_map_stats_before(
                    t2_id,
                    before_date=as_of_date,
                    days=max(120, int(self.form_window)),
                )
            }
        else:
            t1_recent = self.db.get_team_recent_matches(t1_id, limit=20, days=self.form_window)
            t2_recent = self.db.get_team_recent_matches(t2_id, limit=20, days=self.form_window)
            h2h = self.db.get_h2h(t1_id, t2_id)
            t1_maps = {m["map_name"]: m for m in self.db.get_team_map_stats(t1_id)}
            t2_maps = {m["map_name"]: m for m in self.db.get_team_map_stats(t2_id)}

        required = (
            self.live_min_recent_matches
            if min_recent_matches is None
            else max(0, int(min_recent_matches))
        )
        if len(t1_recent) < required or len(t2_recent) < required:
            logger.debug(f"Dados insuficientes: {t1['name']} ({len(t1_recent)}) vs {t2['name']} ({len(t2_recent)})")
            return None

        use_players = self.use_player_features and not for_training and as_of_date is None
        t1_players = self.db.get_team_players(t1_id) if use_players else []
        t2_players = self.db.get_team_players(t2_id) if use_players else []

        features = {}

        # ---- Ranking ----
        r1 = t1.get("ranking", 50) or 50
        r2 = t2.get("ranking", 50) or 50
        features["ranking_diff"] = r2 - r1  # positivo = team1 melhor rankeado
        features["ranking_ratio"] = min(r1, 99) / max(min(r2, 99), 1)

        # ---- Win rate geral ----
        wr1 = _calc_win_rate(t1_recent, t1_id)
        wr2 = _calc_win_rate(t2_recent, t2_id)
        features["winrate_diff"] = wr1 - wr2
        features["team1_winrate"] = wr1
        features["team2_winrate"] = wr2

        # ---- Forma recente (Ãºltimos 5 jogos) ----
        form1 = _calc_win_rate(t1_recent[:5], t1_id)
        form2 = _calc_win_rate(t2_recent[:5], t2_id)
        features["form_diff"] = form1 - form2
        features["team1_form"] = form1
        features["team2_form"] = form2

        # ---- Momentum (streak) ----
        features["team1_streak"] = _calc_streak(t1_recent, t1_id)
        features["team2_streak"] = _calc_streak(t2_recent, t2_id)
        features["streak_diff"] = features["team1_streak"] - features["team2_streak"]

        # ---- Head-to-head ----
        h2h_wr1 = _calc_h2h_winrate(h2h, t1_id)
        features["h2h_winrate_t1"] = h2h_wr1
        features["h2h_matches"] = len(h2h)
        features["h2h_advantage"] = h2h_wr1 - 0.5 if h2h else 0.0

        # ---- Player stats (mÃ©dia do time) ----
        if use_players:
            t1_avg_rating = _avg_stat(t1_players, "rating")
            t2_avg_rating = _avg_stat(t2_players, "rating")
            t1_avg_kd = _avg_stat(t1_players, "kd_ratio")
            t2_avg_kd = _avg_stat(t2_players, "kd_ratio")
            t1_avg_impact = _avg_stat(t1_players, "impact")
            t2_avg_impact = _avg_stat(t2_players, "impact")
        else:
            t1_avg_rating = t2_avg_rating = 0.0
            t1_avg_kd = t2_avg_kd = 0.0
            t1_avg_impact = t2_avg_impact = 0.0

        features["avg_rating_diff"] = t1_avg_rating - t2_avg_rating
        features["team1_avg_rating"] = t1_avg_rating
        features["team2_avg_rating"] = t2_avg_rating
        features["avg_kd_diff"] = t1_avg_kd - t2_avg_kd
        features["avg_impact_diff"] = t1_avg_impact - t2_avg_impact

        # ---- Map pool (diversidade e forÃ§a) ----
        t1_strong_maps = sum(1 for m in t1_maps.values() if m.get("win_rate", 0) > 55)
        t2_strong_maps = sum(1 for m in t2_maps.values() if m.get("win_rate", 0) > 55)
        features["strong_maps_diff"] = t1_strong_maps - t2_strong_maps

        # Melhor win rate em qualquer mapa
        t1_best_map_wr = max((m.get("win_rate", 0) for m in t1_maps.values()), default=0)
        t2_best_map_wr = max((m.get("win_rate", 0) for m in t2_maps.values()), default=0)
        features["best_map_wr_diff"] = t1_best_map_wr - t2_best_map_wr

        # ---- Contexto ----
        features["is_bo1"] = 1 if match.get("best_of", 1) == 1 else 0
        features["is_bo3"] = 1 if match.get("best_of", 1) == 3 else 0
        features["is_lan"] = 1 if match.get("is_lan", 0) else 0
        features["event_tier"] = match.get("event_tier", 3)

        # ---- Atividade ----
        features["team1_matches_played"] = len(t1_recent)
        features["team2_matches_played"] = len(t2_recent)
        features["activity_diff"] = len(t1_recent) - len(t2_recent)

        return features

    def extract_training_data(
        self,
        include_dates: bool = False,
    ) -> tuple[list[dict], list[int]] | tuple[list[dict], list[int], list[str]]:
        """
        Extrai features e labels de todas as partidas completadas.

        Returns:
            (features_list, labels) onde label=1 se team1 ganhou, 0 se team2
        """
        with self.db.connect() as conn:
            rows = conn.execute(
                """SELECT * FROM matches
                   WHERE status='completed' AND winner_id IS NOT NULL
                   ORDER BY date ASC"""
            ).fetchall()

        features_list = []
        labels = []
        skipped = 0
        skipped_synthetic = 0
        skipped_academy = 0
        skipped_invalid_label = 0
        match_dates: list[str] = []

        for row in rows:
            match = dict(row)
            team1_id = int(match.get("team1_id", 0) or 0)
            team2_id = int(match.get("team2_id", 0) or 0)
            winner_id = int(match.get("winner_id", 0) or 0)

            if self.exclude_synthetic_teams and (team1_id <= 0 or team2_id <= 0):
                skipped += 1
                skipped_synthetic += 1
                continue

            t1 = self.db.get_team(team1_id) or {}
            t2 = self.db.get_team(team2_id) or {}
            if self.exclude_academy_teams and (
                _is_academy_name(t1.get("name", "")) or _is_academy_name(t2.get("name", ""))
            ):
                skipped += 1
                skipped_academy += 1
                continue

            if winner_id not in {team1_id, team2_id}:
                skipped += 1
                skipped_invalid_label += 1
                continue

            feats = self.extract(
                match,
                min_recent_matches=self.train_min_recent_matches,
                as_of_date=match.get("date"),
                for_training=True,
            )
            if feats is None:
                skipped += 1
                continue

            label = 1 if winner_id == team1_id else 0
            features_list.append(feats)
            labels.append(label)
            match_dates.append(str(match.get("date", "")))

        logger.info(
            f"[FEATURES] {len(features_list)} amostras extraidas de {len(rows)} partidas "
            f"(descartadas: {skipped}, min_recent_train={self.train_min_recent_matches}, "
            f"sinteticos={skipped_synthetic}, academy={skipped_academy}, invalid_label={skipped_invalid_label})"
        )
        if include_dates:
            return features_list, labels, match_dates
        return features_list, labels


# ============================================================
# Helpers
# ============================================================

def _calc_win_rate(matches: list[dict], team_id: int) -> float:
    if not matches:
        return 0.5
    wins = sum(1 for m in matches if m.get("winner_id") == team_id)
    return wins / len(matches)


def _calc_streak(matches: list[dict], team_id: int) -> int:
    """Calcula streak atual (positivo = vitÃ³rias, negativo = derrotas)."""
    streak = 0
    for m in matches:
        won = m.get("winner_id") == team_id
        if streak == 0:
            streak = 1 if won else -1
        elif won and streak > 0:
            streak += 1
        elif not won and streak < 0:
            streak -= 1
        else:
            break
    return streak


def _calc_h2h_winrate(h2h_matches: list[dict], team_id: int) -> float:
    if not h2h_matches:
        return 0.5
    wins = sum(1 for m in h2h_matches if m.get("winner_id") == team_id)
    return wins / len(h2h_matches)


def _avg_stat(players: list[dict], stat: str) -> float:
    vals = [p.get(stat, 0) for p in players if p.get(stat, 0) > 0]
    return np.mean(vals) if vals else 0.0


def _is_academy_name(name: str) -> bool:
    text = str(name or "").lower()
    if not text:
        return False
    tokens = [
        "academy",
        " junior",
        "juniors",
        " youth",
        "u19",
        "u20",
        "u21",
        "u23",
        "ac.",
    ]
    if any(tok in text for tok in tokens):
        return True
    # Handle separators (e.g. Team-Academy)
    compact = re.sub(r"[^a-z0-9]+", " ", text)
    return " academy " in f" {compact} " or " junior " in f" {compact} "


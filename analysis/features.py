"""
Feature Engineering - Extrai features numericas para o modelo.

Cada partida e transformada em um vetor de features que captura:
- Diferenca de skill entre os times
- Forma recente (multi-janela)
- Historico head-to-head
- Contexto (BO1/BO3, LAN/online, tier do evento)
- Robustez extra (rust, volatilidade, map pool avancado, side CT/T)
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone

import numpy as np

from db.models import Database
from utils.time_utils import parse_datetime_to_utc

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extrai features de uma partida para o modelo ML."""

    def __init__(self, db: Database, config: dict):
        self.db = db
        model_cfg = config.get("model", {})
        self.min_matches = model_cfg.get("min_matches", 10)
        self.form_window = max(30, int(model_cfg.get("form_window_days", 90)))
        # Inferencia ao vivo segue conservadora; treino pode ser permissivo.
        self.live_min_recent_matches = model_cfg.get("live_min_recent_matches", 3)
        self.train_min_recent_matches = model_cfg.get("train_min_recent_matches", 0)
        self.recent_sample_limit = max(20, int(model_cfg.get("recent_sample_limit", 40)))
        self.use_player_features = bool(model_cfg.get("use_player_features", False))

        # Compat: chave antiga `exclude_synthetic_teams` vira fallback para live+train.
        legacy_synth = model_cfg.get("exclude_synthetic_teams")
        if legacy_synth is not None:
            if "exclude_synthetic_teams_live" not in model_cfg or "exclude_synthetic_teams_train" not in model_cfg:
                logger.warning(
                    "[FEATURES] Config legado `exclude_synthetic_teams` detectado; "
                    "use `exclude_synthetic_teams_live` e `exclude_synthetic_teams_train`."
                )
        self.exclude_synthetic_teams_live = bool(
            model_cfg.get(
                "exclude_synthetic_teams_live",
                legacy_synth if legacy_synth is not None else True,
            )
        )
        self.exclude_synthetic_teams_train = bool(
            model_cfg.get(
                "exclude_synthetic_teams_train",
                legacy_synth if legacy_synth is not None else True,
            )
        )

        # Compat: chave antiga `exclude_academy_teams` vira fallback para live+train.
        legacy_academy = model_cfg.get("exclude_academy_teams")
        if legacy_academy is not None:
            if "exclude_academy_teams_live" not in model_cfg or "exclude_academy_teams_train" not in model_cfg:
                logger.warning(
                    "[FEATURES] Config legado `exclude_academy_teams` detectado; "
                    "use `exclude_academy_teams_live` e `exclude_academy_teams_train`."
                )
        self.exclude_academy_teams_live = bool(
            model_cfg.get(
                "exclude_academy_teams_live",
                legacy_academy if legacy_academy is not None else True,
            )
        )
        self.exclude_academy_teams_train = bool(
            model_cfg.get(
                "exclude_academy_teams_train",
                legacy_academy if legacy_academy is not None else True,
            )
        )

        self.last_training_stats: dict = {}

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
            as_of_date: recorte temporal para extracao causal (treino)
            for_training: ativa modo de treino (mais permissivo)

        Returns:
            dict de features numericas ou None se dados insuficientes
        """
        t1_id = int(match["team1_id"])
        t2_id = int(match["team2_id"])

        exclude_synthetic = self.exclude_synthetic_teams_train if for_training else self.exclude_synthetic_teams_live
        if exclude_synthetic and (t1_id <= 0 or t2_id <= 0):
            return None

        t1 = self.db.get_team(t1_id)
        t2 = self.db.get_team(t2_id)
        if not t1 or not t2:
            return None

        exclude_academy = self.exclude_academy_teams_train if for_training else self.exclude_academy_teams_live
        if exclude_academy:
            if is_academy_name(t1.get("name", "")) or is_academy_name(t2.get("name", "")):
                return None

        if as_of_date is not None:
            t1_recent = self.db.get_team_recent_matches_before(
                t1_id,
                before_date=as_of_date,
                limit=self.recent_sample_limit,
                days=self.form_window,
            )
            t2_recent = self.db.get_team_recent_matches_before(
                t2_id,
                before_date=as_of_date,
                limit=self.recent_sample_limit,
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
            t1_recent = self.db.get_team_recent_matches(t1_id, limit=self.recent_sample_limit, days=self.form_window)
            t2_recent = self.db.get_team_recent_matches(t2_id, limit=self.recent_sample_limit, days=self.form_window)
            h2h = self.db.get_h2h(t1_id, t2_id)
            t1_maps = {m["map_name"]: m for m in self.db.get_team_map_stats(t1_id)}
            t2_maps = {m["map_name"]: m for m in self.db.get_team_map_stats(t2_id)}

        required = self.live_min_recent_matches if min_recent_matches is None else max(0, int(min_recent_matches))
        if len(t1_recent) < required or len(t2_recent) < required:
            logger.debug("Dados insuficientes: %s (%s) vs %s (%s)", t1.get("name"), len(t1_recent), t2.get("name"), len(t2_recent))
            return None

        use_players = self.use_player_features and not for_training and as_of_date is None
        t1_players = self.db.get_team_players(t1_id) if use_players else []
        t2_players = self.db.get_team_players(t2_id) if use_players else []

        features: dict[str, float] = {}

        # ---- Ranking ----
        r1 = t1.get("ranking", 50) or 50
        r2 = t2.get("ranking", 50) or 50
        features["ranking_diff"] = r2 - r1  # positivo = team1 melhor rankeado
        features["ranking_ratio"] = min(r1, 99) / max(min(r2, 99), 1)

        # ---- Win rate geral ----
        wr1 = calc_win_rate(t1_recent, t1_id)
        wr2 = calc_win_rate(t2_recent, t2_id)
        features["winrate_diff"] = wr1 - wr2
        features["team1_winrate"] = wr1
        features["team2_winrate"] = wr2

        # ---- Forma recente (ultimos 5 jogos) ----
        form1 = calc_win_rate(t1_recent[:5], t1_id)
        form2 = calc_win_rate(t2_recent[:5], t2_id)
        features["form_diff"] = form1 - form2
        features["team1_form"] = form1
        features["team2_form"] = form2

        # ---- Rust factor (dias sem jogar) ----
        ref_dt = _resolve_reference_dt(as_of_date, match.get("date"))
        rust1 = calc_rust_days(t1_recent, as_of_dt=ref_dt)
        rust2 = calc_rust_days(t2_recent, as_of_dt=ref_dt)
        features["team1_rust_days"] = rust1
        features["team2_rust_days"] = rust2
        features["rust_diff"] = rust1 - rust2

        # ---- Multi-janela de forma (7/14/30/90 dias) ----
        for window_days in (7, 14, 30, 90):
            t1_wr_w = calc_window_win_rate(t1_recent, t1_id, window_days, as_of_dt=ref_dt)
            t2_wr_w = calc_window_win_rate(t2_recent, t2_id, window_days, as_of_dt=ref_dt)
            features[f"team1_wr_{window_days}d"] = t1_wr_w
            features[f"team2_wr_{window_days}d"] = t2_wr_w
            features[f"wr_diff_{window_days}d"] = t1_wr_w - t2_wr_w

        # ---- Win rate por formato (BO atual) ----
        best_of = int(match.get("best_of", 1) or 1)
        t1_bo_wr = calc_format_win_rate(t1_recent, t1_id, best_of)
        t2_bo_wr = calc_format_win_rate(t2_recent, t2_id, best_of)
        features["team1_bo_wr"] = t1_bo_wr
        features["team2_bo_wr"] = t2_bo_wr
        features["bo_wr_diff"] = t1_bo_wr - t2_bo_wr

        # ---- Win rate por venue (LAN/Online atual) ----
        venue_is_lan = bool(match.get("is_lan", 0))
        t1_venue_wr = calc_venue_win_rate(t1_recent, t1_id, venue_is_lan)
        t2_venue_wr = calc_venue_win_rate(t2_recent, t2_id, venue_is_lan)
        features["team1_venue_wr"] = t1_venue_wr
        features["team2_venue_wr"] = t2_venue_wr
        features["venue_wr_diff"] = t1_venue_wr - t2_venue_wr

        # ---- Volatilidade ----
        t1_vol = calc_result_volatility(t1_recent, t1_id)
        t2_vol = calc_result_volatility(t2_recent, t2_id)
        features["team1_volatility"] = t1_vol
        features["team2_volatility"] = t2_vol
        features["volatility_diff"] = t1_vol - t2_vol

        # ---- Momentum (streak) ----
        features["team1_streak"] = calc_streak(t1_recent, t1_id)
        features["team2_streak"] = calc_streak(t2_recent, t2_id)
        features["streak_diff"] = features["team1_streak"] - features["team2_streak"]

        # ---- Head-to-head ----
        h2h_wr1 = calc_h2h_winrate(h2h, t1_id)
        features["h2h_winrate_t1"] = h2h_wr1
        features["h2h_matches"] = len(h2h)
        features["h2h_advantage"] = h2h_wr1 - 0.5 if h2h else 0.0

        # ---- Player stats (media do time) ----
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

        # ---- Map pool (basico) ----
        t1_strong_maps = sum(1 for m in t1_maps.values() if float(m.get("win_rate", 0) or 0) > 55)
        t2_strong_maps = sum(1 for m in t2_maps.values() if float(m.get("win_rate", 0) or 0) > 55)
        features["strong_maps_diff"] = float(t1_strong_maps - t2_strong_maps)

        t1_best_map_wr = max((float(m.get("win_rate", 0) or 0) for m in t1_maps.values()), default=0.0)
        t2_best_map_wr = max((float(m.get("win_rate", 0) or 0) for m in t2_maps.values()), default=0.0)
        features["best_map_wr_diff"] = (t1_best_map_wr - t2_best_map_wr) / 100.0

        # ---- Map pool (avancado) ----
        features.update(compute_map_pool_advanced_features(t1_maps, t2_maps, min_matches=5))

        # ---- Side strength CT/T ----
        side_days = max(90, int(self.form_window))
        t1_side = self.db.get_team_side_stats(t1_id, days=side_days, as_of_date=as_of_date)
        t2_side = self.db.get_team_side_stats(t2_id, days=side_days, as_of_date=as_of_date)
        t1_ct = float(t1_side.get("ct_win_rate", 0.5))
        t2_ct = float(t2_side.get("ct_win_rate", 0.5))
        t1_t = float(t1_side.get("t_win_rate", 0.5))
        t2_t = float(t2_side.get("t_win_rate", 0.5))
        features["team1_ct_wr"] = t1_ct
        features["team2_ct_wr"] = t2_ct
        features["ct_side_diff"] = t1_ct - t2_ct
        features["team1_t_wr"] = t1_t
        features["team2_t_wr"] = t2_t
        features["t_side_diff"] = t1_t - t2_t
        features["side_strength_diff"] = ((t1_ct + t1_t) / 2.0) - ((t2_ct + t2_t) / 2.0)

        # ---- Contexto ----
        features["is_bo1"] = 1 if best_of == 1 else 0
        features["is_bo3"] = 1 if best_of == 3 else 0
        features["is_lan"] = 1 if match.get("is_lan", 0) else 0
        features["event_tier"] = float(match.get("event_tier", 3) or 3)

        # ---- Atividade ----
        features["team1_matches_played"] = float(len(t1_recent))
        features["team2_matches_played"] = float(len(t2_recent))
        features["activity_diff"] = float(len(t1_recent) - len(t2_recent))

        return features

    def extract_training_data(
        self,
        include_dates: bool = False,
        include_quality: bool = False,
    ) -> (
        tuple[list[dict], list[int]]
        | tuple[list[dict], list[int], list[str]]
        | tuple[list[dict], list[int], list[dict]]
        | tuple[list[dict], list[int], list[str], list[dict]]
    ):
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
        skipped_low_data = 0
        included_synthetic = 0
        included_academy = 0
        match_dates: list[str] = []
        sample_quality: list[dict] = []

        for row in rows:
            match = dict(row)
            team1_id = int(match.get("team1_id", 0) or 0)
            team2_id = int(match.get("team2_id", 0) or 0)
            winner_id = int(match.get("winner_id", 0) or 0)
            is_synthetic = team1_id <= 0 or team2_id <= 0

            if self.exclude_synthetic_teams_train and (team1_id <= 0 or team2_id <= 0):
                skipped += 1
                skipped_synthetic += 1
                continue

            t1 = self.db.get_team(team1_id) or {}
            t2 = self.db.get_team(team2_id) or {}
            is_academy = is_academy_name(t1.get("name", "")) or is_academy_name(t2.get("name", ""))
            if self.exclude_academy_teams_train and is_academy:
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
                skipped_low_data += 1
                continue

            label = 1 if winner_id == team1_id else 0
            features_list.append(feats)
            labels.append(label)
            match_dates.append(str(match.get("date", "")))
            sample_quality.append(
                {
                    "is_synthetic": bool(is_synthetic),
                    "is_academy": bool(is_academy),
                }
            )
            if is_synthetic:
                included_synthetic += 1
            if is_academy:
                included_academy += 1

        class_t1 = int(sum(1 for y in labels if y == 1))
        class_t2 = int(sum(1 for y in labels if y == 0))
        self.last_training_stats = {
            "total_raw": len(rows),
            "total_valid": len(features_list),
            "discarded_total": skipped,
            "discarded_synthetic": skipped_synthetic,
            "discarded_academy": skipped_academy,
            "discarded_invalid_label": skipped_invalid_label,
            "discarded_low_data": skipped_low_data,
            "included_synthetic": included_synthetic,
            "included_academy": included_academy,
            "class_team1": class_t1,
            "class_team2": class_t2,
            "min_recent_train": int(self.train_min_recent_matches),
            "exclude_synthetic_train": bool(self.exclude_synthetic_teams_train),
            "exclude_synthetic_live": bool(self.exclude_synthetic_teams_live),
            "exclude_academy_train": bool(self.exclude_academy_teams_train),
            "exclude_academy_live": bool(self.exclude_academy_teams_live),
        }
        logger.info(
            f"[FEATURES] {len(features_list)} amostras extraidas de {len(rows)} partidas "
            f"(descartadas={skipped}, sinteticos={skipped_synthetic}, academy={skipped_academy}, "
            f"invalid_label={skipped_invalid_label}, low_data={skipped_low_data}, "
            f"incl_sinteticos={included_synthetic}, incl_academy={included_academy}, "
            f"classes_t1={class_t1}, classes_t2={class_t2}, "
            f"min_recent_train={self.train_min_recent_matches}, "
            f"exclude_synth_train={self.exclude_synthetic_teams_train}, "
            f"exclude_synth_live={self.exclude_synthetic_teams_live}, "
            f"exclude_academy_train={self.exclude_academy_teams_train}, "
            f"exclude_academy_live={self.exclude_academy_teams_live})"
        )

        if include_dates and include_quality:
            return features_list, labels, match_dates, sample_quality
        if include_dates:
            return features_list, labels, match_dates
        if include_quality:
            return features_list, labels, sample_quality
        return features_list, labels


# ============================================================
# Public helpers (reusaveis por AI/contexto)
# ============================================================

def calc_win_rate(matches: list[dict], team_id: int) -> float:
    if not matches:
        return 0.5
    wins = sum(1 for m in matches if int(m.get("winner_id", 0) or 0) == int(team_id))
    return wins / max(1, len(matches))


def calc_streak(matches: list[dict], team_id: int) -> int:
    """Calcula streak atual (positivo = vitorias, negativo = derrotas)."""
    streak = 0
    for m in matches:
        won = int(m.get("winner_id", 0) or 0) == int(team_id)
        if streak == 0:
            streak = 1 if won else -1
        elif won and streak > 0:
            streak += 1
        elif not won and streak < 0:
            streak -= 1
        else:
            break
    return streak


def calc_h2h_winrate(h2h_matches: list[dict], team_id: int) -> float:
    if not h2h_matches:
        return 0.5
    wins = sum(1 for m in h2h_matches if int(m.get("winner_id", 0) or 0) == int(team_id))
    return wins / max(1, len(h2h_matches))


def calc_window_win_rate(
    matches: list[dict],
    team_id: int,
    window_days: int,
    as_of_dt: datetime | None = None,
) -> float:
    """Win rate por janela temporal com fallback neutro."""
    if not matches:
        return 0.5
    ref_dt = as_of_dt or datetime.now(timezone.utc)
    cutoff = ref_dt - timedelta(days=max(1, int(window_days)))
    scoped = []
    for item in matches:
        dt = _parse_match_date(item.get("date"))
        if dt is None:
            continue
        if cutoff <= dt <= ref_dt:
            scoped.append(item)
    if not scoped:
        return 0.5
    return calc_win_rate(scoped, team_id)


def calc_format_win_rate(matches: list[dict], team_id: int, best_of: int) -> float:
    """Win rate em partidas do mesmo formato BO atual."""
    if not matches:
        return 0.5
    target_bo = int(best_of or 1)
    scoped = [m for m in matches if int(m.get("best_of", 1) or 1) == target_bo]
    if not scoped:
        return 0.5
    return calc_win_rate(scoped, team_id)


def calc_venue_win_rate(matches: list[dict], team_id: int, is_lan: bool) -> float:
    """Win rate por venue (LAN/Online)."""
    if not matches:
        return 0.5
    target = 1 if bool(is_lan) else 0
    scoped = [m for m in matches if int(m.get("is_lan", 0) or 0) == target]
    if not scoped:
        return 0.5
    return calc_win_rate(scoped, team_id)


def calc_result_volatility(matches: list[dict], team_id: int) -> float:
    """Volatilidade de resultado (desvio padrao da serie binaria)."""
    if not matches:
        return 0.25
    outcomes = [1.0 if int(m.get("winner_id", 0) or 0) == int(team_id) else 0.0 for m in matches]
    if not outcomes:
        return 0.25
    if len(outcomes) == 1:
        return 0.0
    return float(np.std(outcomes))


def calc_rust_days(matches: list[dict], as_of_dt: datetime | None = None) -> float:
    """Dias desde a ultima partida (capado para estabilidade)."""
    if not matches:
        return 30.0
    ref_dt = as_of_dt or datetime.now(timezone.utc)
    parsed = [_parse_match_date(item.get("date")) for item in matches]
    parsed = [dt for dt in parsed if dt is not None and dt <= ref_dt]
    if not parsed:
        return 30.0
    latest = max(parsed)
    rust = max(0.0, (ref_dt - latest).total_seconds() / 86400.0)
    return float(min(rust, 90.0))


def compute_map_pool_advanced_features(
    team1_maps: dict[str, dict],
    team2_maps: dict[str, dict],
    min_matches: int = 5,
) -> dict[str, float]:
    """Features avancadas de map pool com fallback neutro."""
    t1_norm = _normalize_map_stats(team1_maps, min_matches=min_matches)
    t2_norm = _normalize_map_stats(team2_maps, min_matches=min_matches)

    t1_names = set(t1_norm.keys())
    t2_names = set(t2_norm.keys())
    common = sorted(t1_names.intersection(t2_names))

    overlap_count = len(common)
    t1_depth = len(t1_names)
    t2_depth = len(t2_names)

    t1_adv = 0.0
    t2_adv = 0.0
    for map_name in common:
        wr1 = float(t1_norm[map_name].get("win_rate", 50.0))
        wr2 = float(t2_norm[map_name].get("win_rate", 50.0))
        t1_adv = max(t1_adv, (wr1 - wr2) / 100.0)
        t2_adv = max(t2_adv, (wr2 - wr1) / 100.0)

    t1_worst = min((float(v.get("win_rate", 50.0)) for v in t1_norm.values()), default=50.0)
    t2_worst = min((float(v.get("win_rate", 50.0)) for v in t2_norm.values()), default=50.0)

    min_depth = max(1, min(t1_depth, t2_depth))
    overlap_ratio = overlap_count / min_depth

    return {
        "map_overlap_count": float(overlap_count),
        "map_overlap_ratio": float(overlap_ratio),
        "team1_map_depth": float(t1_depth),
        "team2_map_depth": float(t2_depth),
        "map_depth_diff": float(t1_depth - t2_depth),
        "map_advantage_t1": float(t1_adv),
        "map_advantage_t2": float(t2_adv),
        "map_advantage_diff": float(t1_adv - t2_adv),
        "worst_map_wr_diff": float((t1_worst - t2_worst) / 100.0),
    }


def is_academy_name(name: str) -> bool:
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
    compact = re.sub(r"[^a-z0-9]+", " ", text)
    return " academy " in f" {compact} " or " junior " in f" {compact} "


def _avg_stat(players: list[dict], stat: str) -> float:
    vals = [p.get(stat, 0) for p in players if p.get(stat, 0) > 0]
    return float(np.mean(vals)) if vals else 0.0


def _normalize_map_stats(map_stats: dict[str, dict], min_matches: int = 5) -> dict[str, dict]:
    normalized: dict[str, dict] = {}
    for map_name, raw in (map_stats or {}).items():
        name = str(map_name or "").strip()
        if not name:
            continue
        matches_played = int(float(raw.get("matches_played", 0) or 0))
        if matches_played < max(1, int(min_matches)):
            continue
        normalized[name] = {
            "matches_played": matches_played,
            "win_rate": float(raw.get("win_rate", 50.0) or 50.0),
        }
    return normalized


def _parse_match_date(value) -> datetime | None:
    return parse_datetime_to_utc(
        value,
        logger=logger,
        context="features.parse_match_date",
    )


def _resolve_reference_dt(as_of_date, fallback_date) -> datetime:
    dt = _parse_match_date(as_of_date)
    if dt is not None:
        return dt
    dt = _parse_match_date(fallback_date)
    if dt is not None:
        return dt
    return datetime.now(timezone.utc)


# Backward compatibility aliases
_calc_win_rate = calc_win_rate
_calc_streak = calc_streak
_calc_h2h_winrate = calc_h2h_winrate
_is_academy_name = is_academy_name

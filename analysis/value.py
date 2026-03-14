"""
Value Detector - Compara probabilidade do modelo com odds das casas.

A logica:
- Se o modelo diz 65% e a casa implica 58%, existe value de ~7%
- Usa Kelly Criterion para calcular tamanho otimo da aposta
- So alerta quando value > threshold configurado
"""

from __future__ import annotations

import logging
import time

from db.models import Database
from utils.time_utils import format_datetime_for_timezone

logger = logging.getLogger(__name__)


class ValueDetector:
    """Detecta value bets comparando modelo vs odds."""

    def __init__(self, config: dict, db: Database | None = None):
        model_cfg = config.get("model", {})
        bankroll_cfg = config.get("bankroll", {})
        scheduler_cfg = config.get("scheduler", {})

        self.db = db
        self.min_value_pct = model_cfg.get("min_value_pct", 4.0)
        self.min_confidence = model_cfg.get("min_confidence", 55.0)
        self.bankroll = bankroll_cfg.get("total", 2000.0)
        self.max_bet_pct = bankroll_cfg.get("max_bet_pct", 3.0)
        self.kelly_fraction = bankroll_cfg.get("kelly_fraction", 0.25)

        # C3: Kelly adaptativo por performance recente resolvida
        self.adaptive_kelly = bool(bankroll_cfg.get("adaptive_kelly", False))
        self.adaptive_kelly_window_days = max(7, int(bankroll_cfg.get("adaptive_kelly_window_days", 30)))
        self._adaptive_cache_ttl_sec = max(30, int(bankroll_cfg.get("adaptive_kelly_cache_sec", 120)))
        self._adaptive_cache_until = 0.0
        self._adaptive_fraction_cached = float(self.kelly_fraction)

        self.display_timezone = str(scheduler_cfg.get("timezone", "America/Sao_Paulo") or "America/Sao_Paulo")
        self.display_timezone_label = str(
            scheduler_cfg.get("display_timezone_label", "BRT (Brasilia)")
            or "BRT (Brasilia)"
        )

    def analyze(
        self,
        prediction: dict,
        odds_team1: float | None = None,
        odds_team2: float | None = None,
        match: dict | None = None,
    ) -> dict | None:
        """Analisa se existe value numa partida."""
        if not prediction:
            return None

        t1_prob = prediction["team1_win_prob"] / 100
        t2_prob = prediction["team2_win_prob"] / 100
        confidence = prediction["confidence"]
        bookmaker_team1 = _safe_str((match or {}).get("bookmaker_team1"))
        bookmaker_team2 = _safe_str((match or {}).get("bookmaker_team2"))
        odds_updated_at = _safe_str((match or {}).get("odds_updated_at"))

        if confidence < self.min_confidence:
            return None

        kelly_fraction_used = self._effective_kelly_fraction()
        result = {
            "team1_win_prob": prediction["team1_win_prob"],
            "team2_win_prob": prediction["team2_win_prob"],
            "confidence": confidence,
            "predicted_winner": prediction["predicted_winner"],
            "kelly_fraction_used": round(kelly_fraction_used, 4),
            "has_value": False,
            "value_bets": [],
            "odds_team1": odds_team1,
            "odds_team2": odds_team2,
            "bookmaker_team1": bookmaker_team1,
            "bookmaker_team2": bookmaker_team2,
            "odds_updated_at": odds_updated_at,
        }

        if not odds_team1 and not odds_team2:
            return result

        if odds_team1 and odds_team1 > 1.0:
            implied_prob_t1 = 1 / odds_team1
            edge_t1 = t1_prob - implied_prob_t1
            value_pct_t1 = edge_t1 * 100

            if value_pct_t1 >= self.min_value_pct:
                stake = self._kelly_stake(t1_prob, odds_team1, kelly_fraction_used)
                result["value_bets"].append(
                    {
                        "side": "team1",
                        "odds": odds_team1,
                        "bookmaker": bookmaker_team1,
                        "model_prob": round(t1_prob * 100, 1),
                        "implied_prob": round(implied_prob_t1 * 100, 1),
                        "value_pct": round(value_pct_t1, 1),
                        "edge": round(edge_t1 * 100, 2),
                        "suggested_stake": round(stake, 2),
                        "expected_value": round(_expected_value(t1_prob, odds_team1, stake), 2),
                        "odds_updated_at": odds_updated_at,
                        "kelly_fraction_used": round(kelly_fraction_used, 4),
                    }
                )

        if odds_team2 and odds_team2 > 1.0:
            implied_prob_t2 = 1 / odds_team2
            edge_t2 = t2_prob - implied_prob_t2
            value_pct_t2 = edge_t2 * 100

            if value_pct_t2 >= self.min_value_pct:
                stake = self._kelly_stake(t2_prob, odds_team2, kelly_fraction_used)
                result["value_bets"].append(
                    {
                        "side": "team2",
                        "odds": odds_team2,
                        "bookmaker": bookmaker_team2,
                        "model_prob": round(t2_prob * 100, 1),
                        "implied_prob": round(implied_prob_t2 * 100, 1),
                        "value_pct": round(value_pct_t2, 1),
                        "edge": round(edge_t2 * 100, 2),
                        "suggested_stake": round(stake, 2),
                        "expected_value": round(_expected_value(t2_prob, odds_team2, stake), 2),
                        "odds_updated_at": odds_updated_at,
                        "kelly_fraction_used": round(kelly_fraction_used, 4),
                    }
                )

        result["has_value"] = len(result["value_bets"]) > 0
        return result

    def _kelly_stake(self, prob: float, odds: float, kelly_fraction: float) -> float:
        """Calcula stake otima usando Kelly Criterion fracionario."""
        if odds <= 1 or prob <= 0 or prob >= 1:
            return 0.0

        kelly = (prob * (odds - 1) - (1 - prob)) / (odds - 1)
        if kelly <= 0:
            return 0.0

        kelly *= float(kelly_fraction)
        max_stake = self.bankroll * (self.max_bet_pct / 100)
        stake = min(kelly * self.bankroll, max_stake)
        return max(stake, 0.0)

    def _effective_kelly_fraction(self) -> float:
        base = float(self.kelly_fraction)
        if not self.adaptive_kelly or self.db is None:
            return base

        now_ts = time.time()
        if now_ts <= self._adaptive_cache_until:
            return self._adaptive_fraction_cached

        rows = self.db.get_recent_resolved_predictions(days=self.adaptive_kelly_window_days)
        resolved = 0
        wins = 0
        for row in rows:
            pred_id = int(row.get("official_pick_winner_id") or row.get("predicted_winner_id") or 0)
            actual_id = int(row.get("actual_winner_id") or 0)
            if pred_id <= 0 or actual_id <= 0:
                continue
            resolved += 1
            if pred_id == actual_id:
                wins += 1

        if resolved < 8:
            adaptive = base
        else:
            win_rate = wins / max(1, resolved)
            avg_clv = float(self.db.get_avg_clv(days=self.adaptive_kelly_window_days))
            wr_component = (win_rate - 0.5) * 0.8
            clv_component = max(-0.12, min(0.12, avg_clv / 100.0))
            factor = 1.0 + wr_component + clv_component
            factor = max(0.55, min(1.25, factor))
            adaptive = base * factor

        self._adaptive_fraction_cached = max(0.02, min(0.5, adaptive))
        self._adaptive_cache_until = now_ts + self._adaptive_cache_ttl_sec
        return self._adaptive_fraction_cached

    def generate_report(self, analysis: dict, match: dict) -> str:
        """Gera relatorio legivel de uma analise."""
        if not analysis:
            return ""

        t1_name = match.get("team1_name", "Team 1")
        t2_name = match.get("team2_name", "Team 2")
        event = match.get("event_name", "")
        bo = match.get("best_of", 1)

        lines = [
            f"Analise: {t1_name} vs {t2_name}",
            f"Evento: {event}" if event else "",
            f"Data: {self._format_match_datetime(match.get('date'))}",
            f"Formato: BO{bo}",
            "",
            "Probabilidades do modelo:",
            f"  {t1_name}: {analysis['team1_win_prob']:.1f}%",
            f"  {t2_name}: {analysis['team2_win_prob']:.1f}%",
            f"  Confianca: {analysis['confidence']:.1f}%",
            f"  Kelly usado: {analysis.get('kelly_fraction_used', self.kelly_fraction):.3f}",
        ]

        o1 = analysis.get("odds_team1")
        o2 = analysis.get("odds_team2")
        b1 = _safe_str(analysis.get("bookmaker_team1"))
        b2 = _safe_str(analysis.get("bookmaker_team2"))
        lines.extend(
            [
                "",
                "Odds de mercado:",
                f"  {t1_name}: {self._format_odds(o1)} {_format_bookmaker(b1)}",
                f"  {t2_name}: {self._format_odds(o2)} {_format_bookmaker(b2)}",
            ]
        )
        if analysis.get("odds_updated_at"):
            lines.append(f"  Atualizado: {self._format_match_datetime(analysis['odds_updated_at'])}")

        if analysis["value_bets"]:
            lines.append("")
            lines.append("VALUE BETS:")
            for vb in analysis["value_bets"]:
                side_name = t1_name if vb["side"] == "team1" else t2_name
                lines.extend(
                    [
                        "",
                        f"  {side_name} @ {vb['odds']:.2f} {_format_bookmaker(vb.get('bookmaker'))}",
                        f"  Modelo: {vb['model_prob']:.1f}% vs Casa: {vb['implied_prob']:.1f}%",
                        f"  Value: +{vb['value_pct']:.1f}%",
                        f"  Stake sugerida: R$ {vb['suggested_stake']:.2f}",
                        f"  EV: R$ {vb['expected_value']:.2f}",
                    ]
                )
        else:
            lines.append("")
            if not o1 and not o2:
                lines.append("Odds indisponiveis no momento")
            else:
                lines.append("Sem value detectado nas odds atuais")

        return "\n".join(line for line in lines if line is not None)

    def _format_odds(self, odds: float | None) -> str:
        if odds is None:
            return "N/D"
        try:
            value = float(odds)
            if value <= 1.0:
                return "N/D"
            return f"{value:.2f}"
        except (TypeError, ValueError):
            return "N/D"

    def _format_match_datetime(self, value) -> str:
        if not value:
            return "Data indefinida"

        text = str(value).strip()
        if not text:
            return "Data indefinida"
        return format_datetime_for_timezone(
            text,
            tz_name=self.display_timezone,
            fmt="%d/%m/%Y %H:%M",
            tz_suffix=self.display_timezone_label,
            logger=logger,
        )


def _expected_value(prob: float, odds: float, stake: float) -> float:
    win_amount = stake * (odds - 1)
    ev = (prob * win_amount) - ((1 - prob) * stake)
    return ev


def _safe_str(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _format_bookmaker(value) -> str:
    text = _safe_str(value)
    if not text:
        return ""
    return f"({text})"

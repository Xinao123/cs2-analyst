#!/usr/bin/env python3
"""
CS2 Analyst Bot â€” Ponto de entrada.

Pipeline contÃ­nuo:
1. Atualiza dados do HLTV (ranking, resultados, partidas futuras)
2. Pra cada partida futura:
   a. Extrai features dos dois times
   b. Roda o modelo preditivo
   c. Compara com odds (se disponÃ­veis)
   d. Detecta value bets
   e. Envia alerta no Telegram
3. Aguarda intervalo e repete

Uso:
    python main.py                   # Roda contÃ­nuo
    python main.py --once            # Roda uma vez e sai
    python main.py --train           # Re-treina o modelo
    python main.py --stats           # Mostra stats do banco
"""

import sys
import time
import signal
import asyncio
import logging
import argparse
from datetime import datetime, timedelta, timezone

import yaml

from db.models import Database
from scraper.hltv import HLTVScraper
from scraper.odds import OddsPapiSync
from analysis.features import FeatureExtractor
from analysis.predictor import Predictor
from analysis.value import ValueDetector
from analysis.daily_top5_audit import DailyTop5Auditor
from alerts.telegram import Notifier
from ai.context import ContextCollector
from ai.llm import DeepSeekClient
from utils.time_utils import parse_datetime_to_utc

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("cs2_analyst.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

_running = True


def _signal_handler(sig, frame):
    global _running
    logger.info("â¹ Encerrando...")
    _running = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ============================================================
# Config
# ============================================================

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# Pipeline principal
# ============================================================

async def analyze_upcoming(
    db: Database,
    features_ext: FeatureExtractor,
    predictor: Predictor,
    value_detector: ValueDetector,
    notifier: Notifier,
    config: dict,
    daily_auditor: DailyTop5Auditor | None = None,
    llm_client: DeepSeekClient | None = None,
    context_collector: ContextCollector | None = None,
) -> int:
    """
    Analisa partidas futuras e separa as melhores oportunidades para aposta.

    Returns:
        Numero de value bets encontrados no ciclo
    """
    if not predictor.is_trained:
        logger.warning("[ANALYSIS] Modelo nao treinado - rode: python main.py --train")
        return 0

    upcoming = db.get_upcoming_matches()
    if not upcoming:
        logger.info("[ANALYSIS] Nenhuma partida futura no banco")
        return 0

    logger.info("[ANALYSIS] Analisando %s partidas futuras...", len(upcoming))
    value_count = 0
    analyzed = 0
    skipped_no_features = 0
    skipped_no_prediction = 0
    filtered_started = 0
    filtered_low_data = 0
    filtered_low_confidence = 0
    filtered_low_value = 0
    filtered_stale_odds = 0
    odds_filtered_started = 0
    odds_filtered_low_data = 0
    odds_filtered_low_confidence = 0
    odds_filtered_low_value = 0
    odds_filtered_stale = 0

    model_cfg = config.get("model", {})
    odds_cfg = config.get("odds", {})

    top_bets_count = max(1, int(model_cfg.get("top_bets_count", 5)))
    min_minutes_before_match = max(0, int(model_cfg.get("min_minutes_before_match", 5)))
    live_thresholds = _resolve_live_thresholds(model_cfg, predictor)
    min_confidence_filter = live_thresholds["min_confidence"]
    min_value_filter = live_thresholds["min_value"]
    synthetic_min_confidence_filter = live_thresholds["synthetic_min_confidence"]
    synthetic_min_value_filter = live_thresholds["synthetic_min_value"]
    min_recent_matches = max(0, int(model_cfg.get("live_min_recent_matches", 2)))
    form_window_days = max(1, int(model_cfg.get("form_window_days", 365)))
    max_pick_odds_age_minutes = max(1, int(odds_cfg.get("max_pick_odds_age_minutes", 120)))

    now_dt = datetime.now(timezone.utc)
    min_start_cutoff = now_dt + timedelta(minutes=min_minutes_before_match)
    approved_candidates: list[dict] = []
    with_odds_analyzed = 0
    with_odds_approved = 0
    with_odds_before_filters = sum(
        1
        for m in upcoming
        if _safe_float(m.get("odds_team1")) > 1.0 and _safe_float(m.get("odds_team2")) > 1.0
    )
    logger.info(
        "[ANALYSIS] Filtros ativos: min_conf=%.1f min_value=%.1f synthetic(min_conf=%.1f min_value=%.1f) "
        "min_recent=%s odds_age<=%smin | com_odds_no_banco=%s",
        min_confidence_filter,
        min_value_filter,
        synthetic_min_confidence_filter,
        synthetic_min_value_filter,
        min_recent_matches,
        max_pick_odds_age_minutes,
        with_odds_before_filters,
    )
    logger.info(
        "[ANALYSIS] Thresholds confianca: config=%.1f tuned=%.1f effective=%.1f (auto_tune=%s)",
        float(live_thresholds.get("min_conf_config", min_confidence_filter)),
        float(live_thresholds.get("min_conf_tuned", min_confidence_filter)),
        float(live_thresholds.get("min_conf_effective", min_confidence_filter)),
        bool(model_cfg.get("confidence_auto_tune", True)),
    )

    for match in upcoming:
        has_valid_odds = (
            _safe_float(match.get("odds_team1")) > 1.0
            and _safe_float(match.get("odds_team2")) > 1.0
        )
        team1_id = int(match.get("team1_id", 0) or 0)
        team2_id = int(match.get("team2_id", 0) or 0)
        is_synthetic_match = _is_synthetic_match(match)
        match_confidence_filter = synthetic_min_confidence_filter if is_synthetic_match else min_confidence_filter
        match_value_filter = synthetic_min_value_filter if is_synthetic_match else min_value_filter

        match_dt = _parse_datetime(match.get("date"))
        if match_dt and match_dt <= min_start_cutoff:
            filtered_started += 1
            if has_valid_odds:
                odds_filtered_started += 1
            continue

        recent_t1, recent_t2, has_min_recent_data = _get_recent_counts(
            db,
            match,
            min_recent_matches=min_recent_matches,
            form_window_days=form_window_days,
        )
        needs_low_data_override = not has_min_recent_data

        if not has_min_recent_data and not has_valid_odds:
            filtered_low_data += 1
            continue

        features = features_ext.extract(match)
        if not features:
            skipped_no_features += 1
            filtered_low_data += 1
            if has_valid_odds:
                odds_filtered_low_data += 1
            continue

        prediction = predictor.predict(features)
        if not prediction:
            skipped_no_prediction += 1
            continue

        analyzed += 1
        analysis = value_detector.analyze(
            prediction=prediction,
            odds_team1=match.get("odds_team1"),
            odds_team2=match.get("odds_team2"),
            match=match,
        )
        if not analysis:
            analysis = {
                "team1_win_prob": prediction["team1_win_prob"],
                "team2_win_prob": prediction["team2_win_prob"],
                "confidence": prediction["confidence"],
                "predicted_winner": prediction["predicted_winner"],
                "has_value": False,
                "value_bets": [],
                "odds_team1": match.get("odds_team1"),
                "odds_team2": match.get("odds_team2"),
            }

        if has_valid_odds:
            with_odds_analyzed += 1

        best_vb = _best_value_bet(analysis)
        value_pct = float(best_vb.get("value_pct", 0.0)) if best_vb else 0.0
        confidence = float(prediction.get("confidence", 0.0))
        low_data_override = False
        if needs_low_data_override:
            low_data_override = _should_allow_low_data_override(
                has_valid_odds=has_valid_odds,
                team1_id=team1_id,
                team2_id=team2_id,
                confidence=confidence,
                value_pct=value_pct,
                synthetic_min_confidence=synthetic_min_confidence_filter,
                synthetic_min_value=synthetic_min_value_filter,
            )
            if not low_data_override:
                filtered_low_data += 1
                if has_valid_odds:
                    odds_filtered_low_data += 1
                    logger.info(
                        "[ANALYSIS][LOW_DATA] rejeitado match_id=%s teams=%s/%s recent_t1=%s recent_t2=%s req=%s form_window_days=%s",
                        match.get("id"),
                        team1_id,
                        team2_id,
                        recent_t1,
                        recent_t2,
                        min_recent_matches,
                        form_window_days,
                    )
                continue
            logger.info(
                "[ANALYSIS][LOW_DATA] override aplicado match_id=%s teams=%s/%s recent_t1=%s recent_t2=%s conf=%.1f value=%.1f",
                match.get("id"),
                team1_id,
                team2_id,
                recent_t1,
                recent_t2,
                confidence,
                value_pct,
            )

        pick_meta = _build_pick_meta(match, prediction, best_vb)
        suggested_bet = pick_meta["official_pick_name"] if best_vb else ""
        suggested_stake = float(best_vb.get("suggested_stake", 0.0)) if best_vb else 0.0

        db.save_prediction(
            match_id=match["id"],
            predicted_winner_id=pick_meta["official_pick_winner_id"],
            model_winner_id=pick_meta["model_winner_id"],
            official_pick_winner_id=pick_meta["official_pick_winner_id"],
            pick_source=pick_meta["pick_source"],
            team1_win_prob=prediction["team1_win_prob"],
            team2_win_prob=prediction["team2_win_prob"],
            value_pct=value_pct,
            suggested_bet=suggested_bet,
            suggested_stake=suggested_stake,
            odds_team1=match.get("odds_team1"),
            odds_team2=match.get("odds_team2"),
        )

        if confidence < match_confidence_filter:
            filtered_low_confidence += 1
            if has_valid_odds:
                odds_filtered_low_confidence += 1
            continue

        if not best_vb:
            filtered_low_value += 1
            if has_valid_odds:
                odds_filtered_low_value += 1
            continue

        value_pct = float(best_vb.get("value_pct", 0.0))
        if value_pct < match_value_filter:
            filtered_low_value += 1
            if has_valid_odds:
                odds_filtered_low_value += 1
            continue

        if not _is_odds_fresh(match.get("odds_updated_at"), max_pick_odds_age_minutes, now_dt):
            filtered_stale_odds += 1
            if has_valid_odds:
                odds_filtered_stale += 1
            continue

        score = _score_bet_candidate(prediction, best_vb, match=match)
        best_odd = float(best_vb.get("odds", 0.0))
        candidate = {
            "match": match,
            "features": features,
            "prediction": prediction,
            "analysis": analysis,
            "score": score,
            "best_vb": best_vb,
            "best_odd": best_odd,
            "official_pick_side": pick_meta["official_pick_side"],
            "official_pick_name": pick_meta["official_pick_name"],
            "official_pick_winner_id": pick_meta["official_pick_winner_id"],
            "model_winner_id": pick_meta["model_winner_id"],
            "model_winner_name": pick_meta["model_winner_name"],
            "pick_source": pick_meta["pick_source"],
            "model_vs_official_diverged": pick_meta["model_vs_official_diverged"],
            "low_data_override": low_data_override,
        }
        approved_candidates.append(candidate)
        value_count += 1
        if has_valid_odds:
            with_odds_approved += 1

        logger.info(
            "  %s (%s%%) vs %s (%s%%) | confianca=%s%% | pick=%s | score=%.2f",
            match.get("team1_name", "?"),
            round(prediction["team1_win_prob"]),
            match.get("team2_name", "?"),
            round(prediction["team2_win_prob"]),
            round(prediction["confidence"]),
            pick_meta["official_pick_name"] or "-",
            score,
        )

    approved_candidates.sort(
        key=lambda item: (float(item.get("score", 0.0)), float(item.get("best_odd", 0.0))),
        reverse=True,
    )
    top_picks = approved_candidates[:top_bets_count]
    if daily_auditor:
        capture = daily_auditor.capture_daily_top5(
            picks=top_picks,
            requested_top=top_bets_count,
            total_candidates=analyzed,
            candidates_with_odds=with_odds_analyzed,
        )
        if capture.get("created"):
            logger.info(
                "[AUDIT] Carteira oficial do dia congelada: data=%s status=%s itens=%s",
                capture.get("run_date"),
                capture.get("status"),
                capture.get("items"),
            )
        else:
            logger.info(
                "[AUDIT] Carteira diaria ja existe: data=%s status=%s",
                capture.get("run_date"),
                capture.get("status"),
            )

    if llm_client and llm_client.is_available and context_collector:
        for pick in top_picks:
            if llm_client.calls_this_cycle >= llm_client.max_calls:
                logger.info("[LLM] Limite por ciclo atingido (%s chamadas).", llm_client.max_calls)
                break
            try:
                ctx = context_collector.collect(pick.get("match", {}))
                llm_text = llm_client.generate_match_analysis(
                    match=pick.get("match", {}),
                    features=pick.get("features", {}),
                    prediction=pick.get("prediction", {}),
                    analysis=pick.get("analysis", {}),
                    context_text=ctx,
                )
                if llm_text:
                    pick["llm_analysis"] = llm_text
                anomaly = llm_client.generate_anomaly_flag(
                    match=pick.get("match", {}),
                    prediction=pick.get("prediction", {}),
                    analysis=pick.get("analysis", {}),
                )
                if anomaly:
                    pick["llm_anomaly"] = anomaly
            except Exception as exc:
                logger.warning("[LLM] Erro gerando analise da partida: %s", exc)

    notifier.top_picks_alert(
        top_picks,
        total_candidates=analyzed,
        requested_top=top_bets_count,
        candidates_with_odds=with_odds_analyzed,
    )
    if top_picks:
        logger.info(
            "[ANALYSIS] Top %s enviados (%s candidatos totais, %s com odds, %s value)",
            len(top_picks),
            analyzed,
            with_odds_analyzed,
            len(approved_candidates),
        )
    else:
        logger.info(
            "[ANALYSIS] Sem oportunidades de value no ciclo (%s candidatos, %s com odds)",
            analyzed,
            with_odds_analyzed,
        )
        if with_odds_before_filters > 0 and with_odds_approved == 0:
            dominant_reason, dominant_count = _dominant_odds_bottleneck(
                {
                    "started": odds_filtered_started,
                    "low_data": odds_filtered_low_data,
                    "low_conf": odds_filtered_low_confidence,
                    "low_value": odds_filtered_low_value,
                    "stale": odds_filtered_stale,
                }
            )
            logger.warning(
                "[ANALYSIS] Odds existem no banco (%s), mas 0 picks com odds foram aprovadas "
                "(analisadas_com_odds=%s | started=%s low_data=%s low_conf=%s low_value=%s stale=%s).",
                with_odds_before_filters,
                with_odds_analyzed,
                odds_filtered_started,
                odds_filtered_low_data,
                odds_filtered_low_confidence,
                odds_filtered_low_value,
                odds_filtered_stale,
            )
            logger.info(
                "[ANALYSIS][ODDS] gargalo_dominante=%s (%s)",
                dominant_reason,
                dominant_count,
            )

    logger.info(
        "[ANALYSIS][ODDS] with_odds_before_filters=%s with_odds_analyzed=%s with_odds_approved=%s "
        "(odds_filtered_started=%s, odds_filtered_low_data=%s, odds_filtered_low_confidence=%s, "
        "odds_filtered_low_value=%s, odds_filtered_stale=%s)",
        with_odds_before_filters,
        with_odds_analyzed,
        with_odds_approved,
        odds_filtered_started,
        odds_filtered_low_data,
        odds_filtered_low_confidence,
        odds_filtered_low_value,
        odds_filtered_stale,
    )

    logger.info(
        "[ANALYSIS] %s partidas analisadas, %s picks aprovadas "
        "(sem_features=%s, sem_pred=%s, filtered_started=%s, filtered_low_data=%s, "
        "filtered_low_confidence=%s, filtered_low_value=%s, filtered_stale_odds=%s, cutoff=%smin)",
        analyzed,
        value_count,
        skipped_no_features,
        skipped_no_prediction,
        filtered_started,
        filtered_low_data,
        filtered_low_confidence,
        filtered_low_value,
        filtered_stale_odds,
        min_minutes_before_match,
    )
    return value_count


def _score_bet_candidate(prediction: dict, analysis_or_best: dict, match: dict | None = None) -> float:
    """Score orientado a precisao com multiplicadores de contexto (tier/formato/lan)."""
    best_vb = analysis_or_best
    if "value_bets" in analysis_or_best:
        best_vb = _best_value_bet(analysis_or_best)
    if not best_vb:
        return 0.0

    value_pct = max(0.0, float(best_vb.get("value_pct", 0.0)))
    expected_value = max(0.0, float(best_vb.get("expected_value", 0.0)))
    confidence = max(0.0, float(prediction.get("confidence", 0.0)))
    base_score = (confidence * 1.4) + (value_pct * 0.8) + (min(expected_value, 30.0) * 0.2)

    tier = int((match or {}).get("event_tier", 3) or 3)
    if tier <= 1:
        tier_mult = 1.08
    elif tier == 2:
        tier_mult = 1.04
    elif tier == 3:
        tier_mult = 1.00
    else:
        tier_mult = 0.96

    best_of = int((match or {}).get("best_of", 1) or 1)
    if best_of >= 5:
        format_mult = 1.05
    elif best_of == 3:
        format_mult = 1.03
    else:
        format_mult = 0.96

    is_lan = bool((match or {}).get("is_lan", 0))
    lan_mult = 1.03 if is_lan else 1.00
    score = base_score * tier_mult * format_mult * lan_mult
    return round(score, 4)


def _best_value_bet(analysis: dict | None) -> dict | None:
    if not analysis:
        return None
    value_bets = analysis.get("value_bets", [])
    if not value_bets:
        return None
    return max(
        value_bets,
        key=lambda vb: (
            float(vb.get("value_pct", 0.0)),
            float(vb.get("expected_value", 0.0)),
        ),
    )


def _resolve_live_thresholds(model_cfg: dict, predictor: Predictor | None = None) -> dict[str, float]:
    """
    Resolve thresholds efetivos de confianca/value para inferencia ao vivo.

    Regras:
    - Usa valores do config como base e teto maximo.
    - Com `confidence_auto_tune=true`, usa o tunado apenas se for menor que o config.
    - Partidas com time sintetico usam piso mais rigoroso.
    """
    config_conf = max(0.0, float(model_cfg.get("min_confidence", 62.0)))
    tuned_conf = config_conf
    base_conf = config_conf
    if bool(model_cfg.get("confidence_auto_tune", True)) and predictor is not None:
        tuned_conf_candidate = float(getattr(predictor, "_saved_tuned_confidence", 0.0))
        if tuned_conf_candidate <= 0:
            tuned_conf_candidate = float(getattr(predictor, "min_confidence", 0.0))
        if tuned_conf_candidate > 0:
            tuned_conf = tuned_conf_candidate
            base_conf = min(config_conf, tuned_conf_candidate)

    base_value = max(0.0, float(model_cfg.get("min_value_pct", 6.0)))
    synthetic_conf = max(
        base_conf,
        float(model_cfg.get("synthetic_live_min_confidence", 68.0)),
    )
    synthetic_value = max(
        base_value,
        float(model_cfg.get("synthetic_live_min_value_pct", 7.0)),
    )
    return {
        "min_confidence": base_conf,
        "min_value": base_value,
        "synthetic_min_confidence": synthetic_conf,
        "synthetic_min_value": synthetic_value,
        "min_conf_config": config_conf,
        "min_conf_tuned": tuned_conf,
        "min_conf_effective": base_conf,
    }


def _should_sync_odds(now_ts: float, last_sync_at: float | None, refresh_seconds: int) -> bool:
    """Throttle simples para sync de odds baseado em tempo real."""
    if refresh_seconds <= 0:
        return True
    if last_sync_at is None:
        return True
    return now_ts >= (last_sync_at + refresh_seconds)


def _dominant_odds_bottleneck(counts: dict[str, int]) -> tuple[str, int]:
    """Retorna o principal motivo de descarte para partidas com odds."""
    if not counts:
        return "none", 0
    reason, count = max(counts.items(), key=lambda item: int(item[1]))
    if int(count) <= 0:
        return "none", 0
    return str(reason), int(count)


def _safe_float(value) -> float:
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return 0.0


def _parse_datetime(value) -> datetime | None:
    return parse_datetime_to_utc(
        value,
        logger=logger,
        context="main.parse_datetime",
    )


def _is_synthetic_match(match: dict) -> bool:
    team1_id = int(match.get("team1_id", 0) or 0)
    team2_id = int(match.get("team2_id", 0) or 0)
    return team1_id <= 0 or team2_id <= 0


def _build_pick_meta(match: dict, prediction: dict, best_vb: dict | None) -> dict:
    model_side = "team1" if int(prediction.get("predicted_winner", 1)) == 1 else "team2"
    if model_side == "team2":
        model_winner_id = int(match.get("team2_id", 0) or 0)
        model_winner_name = str(match.get("team2_name", ""))
    else:
        model_winner_id = int(match.get("team1_id", 0) or 0)
        model_winner_name = str(match.get("team1_name", ""))

    official_side = model_side
    if best_vb and str(best_vb.get("side", "")).lower() in {"team1", "team2"}:
        official_side = str(best_vb.get("side")).lower()

    if official_side == "team2":
        official_pick_winner_id = int(match.get("team2_id", 0) or 0)
        official_pick_name = str(match.get("team2_name", ""))
    else:
        official_pick_winner_id = int(match.get("team1_id", 0) or 0)
        official_pick_name = str(match.get("team1_name", ""))

    return {
        "model_side": model_side,
        "model_winner_id": model_winner_id,
        "model_winner_name": model_winner_name,
        "official_pick_side": official_side,
        "official_pick_winner_id": official_pick_winner_id,
        "official_pick_name": official_pick_name,
        "pick_source": "value" if best_vb else "model",
        "model_vs_official_diverged": model_side != official_side,
    }


def _is_odds_fresh(odds_updated_at, max_age_minutes: int, now_dt: datetime | None = None) -> bool:
    if max_age_minutes <= 0:
        return True

    updated_dt = _parse_datetime(odds_updated_at)
    if not updated_dt:
        return False

    now_dt = now_dt or datetime.now(timezone.utc)
    age_min = (now_dt - updated_dt).total_seconds() / 60.0
    if age_min < 0:
        return True
    return age_min <= max_age_minutes


def _get_recent_counts(
    db: Database,
    match: dict,
    min_recent_matches: int,
    form_window_days: int,
) -> tuple[int, int, bool]:
    if min_recent_matches <= 0:
        return min_recent_matches, min_recent_matches, True

    team1_id = int(match.get("team1_id", 0) or 0)
    team2_id = int(match.get("team2_id", 0) or 0)
    if team1_id == 0 or team2_id == 0:
        return 0, 0, False

    limit = max(min_recent_matches, 5)
    t1_recent = db.get_team_recent_matches(team1_id, limit=limit, days=form_window_days)
    t2_recent = db.get_team_recent_matches(team2_id, limit=limit, days=form_window_days)
    count_t1 = len(t1_recent)
    count_t2 = len(t2_recent)
    return count_t1, count_t2, count_t1 >= min_recent_matches and count_t2 >= min_recent_matches


def _has_min_recent_data(
    db: Database,
    match: dict,
    min_recent_matches: int,
    form_window_days: int,
) -> bool:
    _, _, has_data = _get_recent_counts(db, match, min_recent_matches, form_window_days)
    return has_data


def _should_allow_low_data_override(
    *,
    has_valid_odds: bool,
    team1_id: int,
    team2_id: int,
    confidence: float,
    value_pct: float,
    synthetic_min_confidence: float,
    synthetic_min_value: float,
) -> bool:
    if not has_valid_odds:
        return False
    if team1_id <= 0 or team2_id <= 0:
        return False
    # Threshold relaxado: aceitar com 80% do synthetic threshold (min 50%)
    # O pipeline principal ja filtra por confidence e value depois
    relaxed_conf = max(50.0, synthetic_min_confidence * 0.80)
    if confidence < relaxed_conf:
        return False
    return True


def _build_training_quality_weights(config: dict, sample_quality: list[dict]) -> list[float]:
    if not sample_quality:
        return []

    model_cfg = config.get("model", {})
    synthetic_weight = max(0.05, float(model_cfg.get("train_weight_synthetic", 0.8)))
    academy_weight = max(0.05, float(model_cfg.get("train_weight_academy", 0.7)))

    weights: list[float] = []
    for meta in sample_quality:
        w = 1.0
        if bool(meta.get("is_synthetic")):
            w *= synthetic_weight
        if bool(meta.get("is_academy")):
            w *= academy_weight
        weights.append(max(0.05, min(10.0, w)))
    return weights


async def update_data(db: Database, config: dict):
    """Atualiza dados do HLTV."""
    scraper = HLTVScraper(db, config)
    try:
        await scraper.full_update()
    finally:
        await scraper.close()


async def bootstrap_train_data(config: dict) -> dict:
    """Executa bootstrap agressivo de historico PandaScore para treino."""
    db = Database(config["database"]["path"])
    scraper = HLTVScraper(db, config)
    try:
        report = await asyncio.to_thread(scraper.sync_pandascore_history, True, True)
    finally:
        await scraper.close()
    return report


async def reset_upcoming_timezone_data(config: dict) -> dict:
    """Limpa dados dependentes de upcoming e recarrega com timezone corrigido."""
    db = Database(config["database"]["path"])
    cleanup = db.clear_upcoming_related_data()

    scraper = HLTVScraper(db, config)
    try:
        saved_upcoming = await scraper.scrape_upcoming_matches()
    finally:
        await scraper.close()

    odds_sync = OddsPapiSync(db, config)
    odds_report = await asyncio.to_thread(odds_sync.sync_upcoming_odds) if odds_sync.enabled else {"saved": 0}
    return {
        "cleanup": cleanup,
        "saved_upcoming": int(saved_upcoming),
        "odds_saved": int(odds_report.get("saved", 0)),
        "odds_error": str(odds_report.get("error", "")),
    }


def train_model(db: Database, config: dict, notifier: Notifier) -> dict:
    """Treina/re-treina o modelo."""
    features_ext = FeatureExtractor(db, config)
    predictor = Predictor(config)

    logger.info("[TRAIN] Extraindo features...")
    features_list, labels, match_dates, sample_quality = features_ext.extract_training_data(
        include_dates=True,
        include_quality=True,
    )
    stats = getattr(features_ext, "last_training_stats", {}) or {}
    if stats:
        logger.info(
            "[TRAIN] Dataset: bruto=%s validas=%s descartadas=%s "
            "(synthetic=%s, academy=%s, invalid_label=%s, low_data=%s)",
            stats.get("total_raw", 0),
            stats.get("total_valid", 0),
            stats.get("discarded_total", 0),
            stats.get("discarded_synthetic", 0),
            stats.get("discarded_academy", 0),
            stats.get("discarded_invalid_label", 0),
            stats.get("discarded_low_data", 0),
        )
        logger.info(
            "[TRAIN] Classes: team1=%s team2=%s | synth_policy(train=%s, live=%s)",
            stats.get("class_team1", 0),
            stats.get("class_team2", 0),
            stats.get("exclude_synthetic_train", True),
            stats.get("exclude_synthetic_live", True),
        )
        logger.info(
            "[TRAIN] Academy policy: train=%s live=%s | included(synth=%s, academy=%s)",
            stats.get("exclude_academy_train", True),
            stats.get("exclude_academy_live", True),
            stats.get("included_synthetic", 0),
            stats.get("included_academy", 0),
        )
        if not bool(stats.get("exclude_synthetic_train", True)):
            logger.info("[TRAIN] Modo relaxed-train ativo: times sinteticos permitidos no treino.")
        if not bool(stats.get("exclude_academy_train", True)):
            logger.info("[TRAIN] Modo relaxed-train ativo: academies permitidas no treino.")

    if len(features_list) < 20:
        reason = f"dados_insuficientes:{len(features_list)}"
        logger.warning("[TRAIN] Treino ignorado (%s). Mantendo modelo atual.", reason)
        return {
            "skipped": True,
            "reason": reason,
            "samples": len(features_list),
        }

    quality_weights = _build_training_quality_weights(config, sample_quality)
    if quality_weights:
        logger.info(
            "[TRAIN] Pesos de qualidade: media=%.3f min=%.3f max=%.3f (n=%s)",
            sum(quality_weights) / len(quality_weights),
            min(quality_weights),
            max(quality_weights),
            len(quality_weights),
        )

    logger.info("[TRAIN] Treinando modelo...")
    metrics = predictor.train(
        features_list,
        labels,
        match_dates=match_dates,
        sample_weights=quality_weights,
    )

    if "error" in metrics:
        logger.warning(
            "[TRAIN] Treino nao atualizado (%s). Mantendo modelo atual.",
            metrics.get("error", "erro_desconhecido"),
        )
        return {
            "skipped": True,
            "reason": metrics.get("error", "erro_desconhecido"),
            "samples": len(features_list),
        }

    notifier.model_trained(metrics)

    return metrics


# ============================================================
# Loop principal
# ============================================================

async def run(config: dict, once: bool = False):
    global _running

    db = Database(config["database"]["path"])
    features_ext = FeatureExtractor(db, config)
    predictor = Predictor(config)
    value_detector = ValueDetector(config, db=db)
    llm_client = DeepSeekClient(config)
    context_collector = ContextCollector(db, config) if llm_client.is_available else None
    if llm_client.is_available:
        logger.info(
            "[LLM] DeepSeek habilitada (model=%s budget=$%.2f/mensal).",
            llm_client.model,
            llm_client.monthly_budget,
        )
    else:
        logger.info("[LLM] Desabilitada ou indisponivel. Fluxo segue em modo template.")

    notifier = Notifier(config, llm_client=llm_client)
    odds_sync = OddsPapiSync(db, config)
    daily_auditor = DailyTop5Auditor(db, config, notifier)
    history_scraper = HLTVScraper(db, config)

    scheduler_cfg = config.get("scheduler", {})
    interval = scheduler_cfg.get("scan_interval", 60) * 60  # em segundos
    daily_hour = scheduler_cfg.get("daily_update_hour", 6)
    auto_train_enabled = bool(scheduler_cfg.get("auto_train_enabled", False))
    odds_cfg = config.get("odds", {})
    scraper_cfg = config.get("scraper", {})
    odds_sync_during_wait = bool(odds_cfg.get("sync_during_wait", False))
    last_odds_sync_at: float | None = None
    history_sync_enabled = bool(scraper_cfg.get("pandascore_history_enabled", True))
    history_sync_seconds = max(
        300,
        int(scraper_cfg.get("pandascore_history_sync_interval_minutes", 30)) * 60,
    )
    next_history_sync_at = time.time() if history_sync_enabled else float("inf")
    next_audit_probe_at = time.time()

    # Banner
    logger.info("=" * 50)
    logger.info("  ðŸŽ® CS2 ANALYST BOT")
    logger.info("=" * 50)

    stats = db.get_stats()
    logger.info(f"  Times: {stats['teams']} | Partidas: {stats['completed_matches']}")
    logger.info(f"  Modelo treinado: {'âœ…' if predictor.is_trained else 'âŒ'}")
    logger.info(f"  Intervalo: {interval // 60} min")
    logger.info("=" * 50)

    notifier.startup(stats)

    last_daily_update = None

    while _running:
        try:
            now = datetime.now()
            if llm_client.is_available:
                llm_client.reset_cycle_counter()

            # AtualizaÃ§Ã£o diÃ¡ria de dados (ranking, stats)
            if last_daily_update != now.date() and now.hour >= daily_hour:
                logger.info("[SCHEDULER] AtualizaÃ§Ã£o diÃ¡ria de dados...")
                await update_data(db, config)
                last_daily_update = now.date()

                if history_sync_enabled:
                    history_report = await asyncio.to_thread(
                        history_scraper.sync_pandascore_history,
                        False,
                        False,
                    )
                    if history_report.get("saved", 0) > 0:
                        logger.info(
                            "[HISTORY][pandascore] sincronizacao diaria: salvas=%s requests=%s",
                            history_report.get("saved", 0),
                            history_report.get("requests_used", 0),
                        )
                    next_history_sync_at = time.time() + history_sync_seconds

                # Re-treina modelo com dados novos (opcional por config)
                if auto_train_enabled:
                    stats = db.get_stats()
                    if stats["completed_matches"] >= 50:
                        metrics = train_model(db, config, notifier)
                        if metrics and not metrics.get("skipped") and "error" not in metrics:
                            predictor = Predictor(config)
                else:
                    logger.info("[TRAIN] Auto-train desabilitado; use --train para treinar manualmente.")

            if history_sync_enabled and time.time() >= next_history_sync_at:
                history_report = await asyncio.to_thread(
                    history_scraper.sync_pandascore_history,
                    False,
                    False,
                )
                if history_report.get("saved", 0) > 0:
                    logger.info(
                        "[HISTORY][pandascore] incremental: salvas=%s requests=%s",
                        history_report.get("saved", 0),
                        history_report.get("requests_used", 0),
                    )
                next_history_sync_at = time.time() + history_sync_seconds

            # Busca partidas futuras e resultados novos
            scraper = HLTVScraper(db, config)
            await scraper.scrape_upcoming_matches()
            await scraper.scrape_results()
            await scraper.close()

            if daily_auditor.enabled:
                await asyncio.to_thread(daily_auditor.run_if_due)
                next_audit_probe_at = time.time() + 30

            # Atualiza odds reais antes da analise (respeita throttle por refresh_minutes)
            if odds_sync.enabled:
                now_ts = time.time()
                if _should_sync_odds(now_ts, last_odds_sync_at, odds_sync.refresh_seconds):
                    await asyncio.to_thread(odds_sync.sync_upcoming_odds)
                    last_odds_sync_at = now_ts
                else:
                    wait_sec = int(max(0, (last_odds_sync_at + odds_sync.refresh_seconds) - now_ts))
                    logger.info(
                        "[ODDS] Throttle ativo, proximo sync em ~%ss",
                        wait_sec,
                    )

            # Analisa partidas futuras
            await analyze_upcoming(
                db,
                features_ext,
                predictor,
                value_detector,
                notifier,
                config,
                daily_auditor=daily_auditor,
                llm_client=llm_client,
                context_collector=context_collector,
            )

            if once:
                logger.info("Modo --once: encerrando.")
                break

        except Exception as e:
            logger.exception(f"âŒ Erro no loop: {e}")
            notifier.error(str(e))

        # Aguarda prÃ³ximo ciclo
        logger.info(f"[SCHEDULER] PrÃ³ximo scan em {interval // 60} min")
        for _ in range(interval):
            if not _running:
                break
            if odds_sync.enabled and odds_sync_during_wait:
                now_ts = time.time()
                if _should_sync_odds(now_ts, last_odds_sync_at, odds_sync.refresh_seconds):
                    await asyncio.to_thread(odds_sync.sync_upcoming_odds)
                    last_odds_sync_at = now_ts
            if history_sync_enabled and time.time() >= next_history_sync_at:
                next_history_sync_at = time.time() + history_sync_seconds
                await asyncio.to_thread(history_scraper.sync_pandascore_history, False, False)
            if daily_auditor.enabled and time.time() >= next_audit_probe_at:
                next_audit_probe_at = time.time() + 30
                await asyncio.to_thread(daily_auditor.run_if_due)
            await asyncio.sleep(1)

    await history_scraper.close()
    logger.info("ðŸ‘‹ Bot encerrado.")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="CS2 Analyst Bot")
    parser.add_argument("--config", "-c", default="config.yaml")
    parser.add_argument("--once", action="store_true", help="Roda uma vez e sai")
    parser.add_argument("--train", action="store_true", help="Re-treina o modelo")
    parser.add_argument(
        "--bootstrap-train-data",
        action="store_true",
        help="Bootstrap agressivo de historico PandaScore para treino",
    )
    parser.add_argument(
        "--reset-upcoming-timezone-data",
        action="store_true",
        help="Limpa upcoming/odds/predictions dependentes e recarrega com timezone corrigido",
    )
    parser.add_argument("--stats", action="store_true", help="Mostra stats do banco")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.bootstrap_train_data:
        report = asyncio.run(bootstrap_train_data(config))
        print("\n🚀 Bootstrap PandaScore (treino)")
        print(f"   requests usados: {report.get('requests_used', 0)}")
        print(f"   retornadas: {report.get('returned', 0)}")
        print(f"   salvas: {report.get('saved', 0)}")
        print(f"   descartadas: {report.get('discarded', 0)}")
        if report.get("error"):
            print(f"   aviso: {report.get('error')}")
        return

    if args.reset_upcoming_timezone_data:
        report = asyncio.run(reset_upcoming_timezone_data(config))
        cleanup = report.get("cleanup", {})
        print("\n🧹 Saneamento timezone de upcoming")
        print(f"   matches_upcoming_removidas: {cleanup.get('matches_deleted', 0)}")
        print(f"   odds_latest_removidas: {cleanup.get('odds_latest_deleted', 0)}")
        print(f"   odds_snapshots_removidas: {cleanup.get('odds_snapshots_deleted', 0)}")
        print(f"   predictions_removidas: {cleanup.get('predictions_deleted', 0)}")
        print(f"   upcoming_recarregadas: {report.get('saved_upcoming', 0)}")
        print(f"   odds_recarregadas: {report.get('odds_saved', 0)}")
        if report.get("odds_error"):
            print(f"   aviso_odds: {report.get('odds_error')}")
        return

    if args.stats:
        db = Database(config["database"]["path"])
        stats = db.get_stats()
        print(f"\nðŸ“Š CS2 Analyst â€” Stats do banco")
        print(f"{'â”' * 35}")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        # Mostra Ãºltimas prediÃ§Ãµes
        preds = db.get_prediction_history(5)
        if preds:
            print(f"\nðŸ“‹ Ãšltimas prediÃ§Ãµes:")
            for p in preds:
                t1 = p.get("team1_name", "?")
                t2 = p.get("team2_name", "?")
                p1 = p.get("team1_win_prob", 50)
                print(f"  {t1} ({p1:.0f}%) vs {t2} ({100-p1:.0f}%)")
        return

    if args.train:
        db = Database(config["database"]["path"])
        notifier = Notifier(config)
        metrics = train_model(db, config, notifier)
        if metrics and metrics.get("skipped"):
            print(f"\n⚠️ Treino ignorado: {metrics.get('reason', 'motivo_desconhecido')}")
            print("   Modelo atual foi mantido.")
        elif metrics and "error" not in metrics:
            print(f"\nâœ… Modelo treinado: {metrics['model']}")
            print(f"   CV Accuracy: {metrics['cv_accuracy']:.1f}%")
            print(f"   Train Accuracy: {metrics['train_accuracy']:.1f}%")
        return

    asyncio.run(run(config, once=args.once))


if __name__ == "__main__":
    main()


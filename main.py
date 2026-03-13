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
from datetime import datetime, timedelta

import yaml

from db.models import Database
from scraper.hltv import HLTVScraper
from scraper.odds import OddsPapiSync
from analysis.features import FeatureExtractor
from analysis.predictor import Predictor
from analysis.value import ValueDetector
from analysis.daily_top5_audit import DailyTop5Auditor
from alerts.telegram import Notifier

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

    model_cfg = config.get("model", {})
    odds_cfg = config.get("odds", {})

    top_bets_count = max(1, int(model_cfg.get("top_bets_count", 5)))
    min_minutes_before_match = max(0, int(model_cfg.get("min_minutes_before_match", 5)))
    min_confidence_filter = max(0.0, float(model_cfg.get("min_confidence", 70.0)))
    if bool(model_cfg.get("confidence_auto_tune", True)):
        min_confidence_filter = max(min_confidence_filter, float(getattr(predictor, "min_confidence", 0.0)))
    min_value_filter = max(0.0, float(model_cfg.get("min_value_pct", 8.0)))
    min_recent_matches = max(0, int(model_cfg.get("live_min_recent_matches", 5)))
    form_window_days = max(1, int(model_cfg.get("form_window_days", 365)))
    max_pick_odds_age_minutes = max(1, int(odds_cfg.get("max_pick_odds_age_minutes", 120)))

    now_dt = datetime.now()
    min_start_cutoff = datetime.now() + timedelta(minutes=min_minutes_before_match)
    approved_candidates: list[dict] = []
    candidates_with_odds = 0

    for match in upcoming:
        match_dt = _parse_datetime(match.get("date"))
        if match_dt and match_dt <= min_start_cutoff:
            filtered_started += 1
            continue

        if not _has_min_recent_data(
            db,
            match,
            min_recent_matches=min_recent_matches,
            form_window_days=form_window_days,
        ):
            filtered_low_data += 1
            continue

        features = features_ext.extract(match)
        if not features:
            skipped_no_features += 1
            filtered_low_data += 1
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

        has_valid_odds = (
            _safe_float(match.get("odds_team1")) > 1.0
            and _safe_float(match.get("odds_team2")) > 1.0
        )
        if has_valid_odds:
            candidates_with_odds += 1

        best_vb = _best_value_bet(analysis)
        pick_meta = _build_pick_meta(match, prediction, best_vb)
        suggested_bet = pick_meta["official_pick_name"] if best_vb else ""
        suggested_stake = float(best_vb.get("suggested_stake", 0.0)) if best_vb else 0.0
        value_pct = float(best_vb.get("value_pct", 0.0)) if best_vb else 0.0

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

        confidence = float(prediction.get("confidence", 0.0))
        if confidence < min_confidence_filter:
            filtered_low_confidence += 1
            continue

        if not best_vb:
            filtered_low_value += 1
            continue

        value_pct = float(best_vb.get("value_pct", 0.0))
        if value_pct < min_value_filter:
            filtered_low_value += 1
            continue

        if not _is_odds_fresh(match.get("odds_updated_at"), max_pick_odds_age_minutes, now_dt):
            filtered_stale_odds += 1
            continue

        score = _score_bet_candidate(prediction, best_vb)
        best_odd = float(best_vb.get("odds", 0.0))
        candidate = {
            "match": match,
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
        }
        approved_candidates.append(candidate)
        value_count += 1

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
            candidates_with_odds=candidates_with_odds,
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

    notifier.top_picks_alert(
        top_picks,
        total_candidates=analyzed,
        requested_top=top_bets_count,
        candidates_with_odds=candidates_with_odds,
    )
    if top_picks:
        logger.info(
            "[ANALYSIS] Top %s enviados (%s candidatos totais, %s com odds, %s value)",
            len(top_picks),
            analyzed,
            candidates_with_odds,
            len(approved_candidates),
        )
    else:
        logger.info(
            "[ANALYSIS] Sem oportunidades de value no ciclo (%s candidatos, %s com odds)",
            analyzed,
            candidates_with_odds,
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


def _score_bet_candidate(prediction: dict, analysis_or_best: dict) -> float:
    """Score conservador orientado a precisão."""
    best_vb = analysis_or_best
    if "value_bets" in analysis_or_best:
        best_vb = _best_value_bet(analysis_or_best)
    if not best_vb:
        return 0.0

    value_pct = max(0.0, float(best_vb.get("value_pct", 0.0)))
    expected_value = max(0.0, float(best_vb.get("expected_value", 0.0)))
    confidence = max(0.0, float(prediction.get("confidence", 0.0)))
    score = (confidence * 1.4) + (value_pct * 0.8) + (min(expected_value, 30.0) * 0.2)
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


def _safe_float(value) -> float:
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return 0.0


def _parse_datetime(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value

    text = str(value).strip()
    if not text:
        return None

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo:
            parsed = parsed.astimezone().replace(tzinfo=None)
        return parsed
    except ValueError:
        pass

    for date_fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d-%m-%Y %H:%M:%S", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, date_fmt)
        except ValueError:
            continue
    return None


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

    now_dt = now_dt or datetime.now()
    age_min = (now_dt - updated_dt).total_seconds() / 60.0
    if age_min < 0:
        return True
    return age_min <= max_age_minutes


def _has_min_recent_data(
    db: Database,
    match: dict,
    min_recent_matches: int,
    form_window_days: int,
) -> bool:
    if min_recent_matches <= 0:
        return True

    team1_id = int(match.get("team1_id", 0) or 0)
    team2_id = int(match.get("team2_id", 0) or 0)
    if team1_id == 0 or team2_id == 0:
        return False

    limit = max(min_recent_matches, 5)
    t1_recent = db.get_team_recent_matches(team1_id, limit=limit, days=form_window_days)
    t2_recent = db.get_team_recent_matches(team2_id, limit=limit, days=form_window_days)
    return len(t1_recent) >= min_recent_matches and len(t2_recent) >= min_recent_matches


async def update_data(db: Database, config: dict):
    """Atualiza dados do HLTV."""
    scraper = HLTVScraper(db, config)
    try:
        await scraper.full_update()
    finally:
        await scraper.close()


def train_model(db: Database, config: dict, notifier: Notifier) -> dict:
    """Treina/re-treina o modelo."""
    features_ext = FeatureExtractor(db, config)
    predictor = Predictor(config)

    logger.info("[TRAIN] Extraindo features...")
    features_list, labels, match_dates = features_ext.extract_training_data(include_dates=True)

    if len(features_list) < 20:
        logger.error(f"[TRAIN] Dados insuficientes: {len(features_list)} amostras")
        return {}

    logger.info("[TRAIN] Treinando modelo...")
    metrics = predictor.train(features_list, labels, match_dates=match_dates)

    if "error" not in metrics:
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
    value_detector = ValueDetector(config)
    notifier = Notifier(config)
    odds_sync = OddsPapiSync(db, config)
    daily_auditor = DailyTop5Auditor(db, config, notifier)

    interval = config.get("scheduler", {}).get("scan_interval", 60) * 60  # em segundos
    daily_hour = config.get("scheduler", {}).get("daily_update_hour", 6)
    odds_cfg = config.get("odds", {})
    odds_sync_during_wait = bool(odds_cfg.get("sync_during_wait", False))
    next_odds_sync_at = (
        time.time() + odds_sync.refresh_seconds
        if odds_sync.enabled and odds_sync_during_wait
        else float("inf")
    )
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

            # AtualizaÃ§Ã£o diÃ¡ria de dados (ranking, stats)
            if last_daily_update != now.date() and now.hour >= daily_hour:
                logger.info("[SCHEDULER] AtualizaÃ§Ã£o diÃ¡ria de dados...")
                await update_data(db, config)
                last_daily_update = now.date()

                # Re-treina modelo com dados novos
                stats = db.get_stats()
                if stats["completed_matches"] >= 50:
                    metrics = train_model(db, config, notifier)
                    if metrics and "error" not in metrics:
                        predictor = Predictor(config)

            # Busca partidas futuras e resultados novos
            scraper = HLTVScraper(db, config)
            await scraper.scrape_upcoming_matches()
            await scraper.scrape_results()
            await scraper.close()

            if daily_auditor.enabled:
                await asyncio.to_thread(daily_auditor.run_if_due)
                next_audit_probe_at = time.time() + 30

            # Atualiza odds reais antes da analise (uma vez no --once e a cada ciclo normal)
            if odds_sync.enabled:
                if odds_sync_during_wait:
                    next_odds_sync_at = time.time() + odds_sync.refresh_seconds
                await asyncio.to_thread(odds_sync.sync_upcoming_odds)

            # Analisa partidas futuras
            await analyze_upcoming(
                db,
                features_ext,
                predictor,
                value_detector,
                notifier,
                config,
                daily_auditor=daily_auditor,
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
            if odds_sync.enabled and odds_sync_during_wait and time.time() >= next_odds_sync_at:
                next_odds_sync_at = time.time() + odds_sync.refresh_seconds
                await asyncio.to_thread(odds_sync.sync_upcoming_odds)
            if daily_auditor.enabled and time.time() >= next_audit_probe_at:
                next_audit_probe_at = time.time() + 30
                await asyncio.to_thread(daily_auditor.run_if_due)
            await asyncio.sleep(1)

    logger.info("ðŸ‘‹ Bot encerrado.")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="CS2 Analyst Bot")
    parser.add_argument("--config", "-c", default="config.yaml")
    parser.add_argument("--once", action="store_true", help="Roda uma vez e sai")
    parser.add_argument("--train", action="store_true", help="Re-treina o modelo")
    parser.add_argument("--stats", action="store_true", help="Mostra stats do banco")
    args = parser.parse_args()

    config = load_config(args.config)

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
        if metrics and "error" not in metrics:
            print(f"\nâœ… Modelo treinado: {metrics['model']}")
            print(f"   CV Accuracy: {metrics['cv_accuracy']:.1f}%")
            print(f"   Train Accuracy: {metrics['train_accuracy']:.1f}%")
        return

    asyncio.run(run(config, once=args.once))


if __name__ == "__main__":
    main()


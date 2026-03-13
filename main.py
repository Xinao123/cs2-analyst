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
from datetime import datetime

import yaml

from db.models import Database
from scraper.hltv import HLTVScraper
from analysis.features import FeatureExtractor
from analysis.predictor import Predictor
from analysis.value import ValueDetector
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
    top_bets_count = max(1, int(config.get("model", {}).get("top_bets_count", 5)))
    candidates: list[dict] = []

    for match in upcoming:
        features = features_ext.extract(match)
        if not features:
            skipped_no_features += 1
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

        predicted_winner_id = (
            match["team1_id"] if prediction["predicted_winner"] == 1
            else match["team2_id"]
        )
        db.save_prediction(
            match_id=match["id"],
            predicted_winner_id=predicted_winner_id,
            team1_win_prob=prediction["team1_win_prob"],
            team2_win_prob=prediction["team2_win_prob"],
        )

        score = _score_bet_candidate(prediction, analysis, features)
        candidates.append(
            {
                "match": match,
                "prediction": prediction,
                "analysis": analysis,
                "score": score,
            }
        )

        if analysis.get("has_value"):
            value_count += 1

        logger.info(
            "  %s (%s%%) vs %s (%s%%) | confianca=%s%% | score=%.2f",
            match.get("team1_name", "?"),
            round(prediction["team1_win_prob"]),
            match.get("team2_name", "?"),
            round(prediction["team2_win_prob"]),
            round(prediction["confidence"]),
            score,
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    top_picks = candidates[:top_bets_count]
    if top_picks:
        notifier.top_picks_alert(top_picks, len(candidates), top_bets_count)
        logger.info(
            "[ANALYSIS] Top %s enviados (%s candidatos, %s com value no top)",
            len(top_picks),
            len(candidates),
            sum(1 for item in top_picks if item["analysis"].get("has_value")),
        )
    else:
        logger.info("[ANALYSIS] Nenhum candidato para top picks")

    logger.info(
        "[ANALYSIS] %s partidas analisadas, %s value bets (sem_features=%s, sem_pred=%s)",
        analyzed,
        value_count,
        skipped_no_features,
        skipped_no_prediction,
    )
    return value_count


def _score_bet_candidate(prediction: dict, analysis: dict, features: dict) -> float:
    """Score de priorizacao para top picks."""
    confidence = float(prediction.get("confidence", 0.0))
    separation = abs(float(prediction.get("team1_win_prob", 50.0)) - float(prediction.get("team2_win_prob", 50.0)))
    coverage = min(float(features.get("team1_matches_played", 0)), float(features.get("team2_matches_played", 0)))
    h2h = float(features.get("h2h_matches", 0))

    score = (confidence * 1.2) + (separation * 0.6)
    score += min(coverage, 20.0) * 0.6
    score += min(h2h, 10.0) * 0.4

    value_bets = analysis.get("value_bets", [])
    if value_bets:
        best_vb = max(value_bets, key=lambda vb: float(vb.get("value_pct", 0.0)))
        value_pct = float(best_vb.get("value_pct", 0.0))
        expected_value = max(0.0, float(best_vb.get("expected_value", 0.0)))
        score += (value_pct * 2.0) + (expected_value / 5.0)

    return round(score, 4)


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
    features_list, labels = features_ext.extract_training_data()

    if len(features_list) < 20:
        logger.error(f"[TRAIN] Dados insuficientes: {len(features_list)} amostras")
        return {}

    logger.info("[TRAIN] Treinando modelo...")
    metrics = predictor.train(features_list, labels)

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

    interval = config.get("scheduler", {}).get("scan_interval", 60) * 60  # em segundos
    daily_hour = config.get("scheduler", {}).get("daily_update_hour", 6)

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

            # Analisa partidas futuras
            await analyze_upcoming(db, features_ext, predictor, value_detector, notifier, config)

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
            time.sleep(1)

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


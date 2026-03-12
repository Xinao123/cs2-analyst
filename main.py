#!/usr/bin/env python3
"""
CS2 Analyst Bot — Ponto de entrada.

Pipeline contínuo:
1. Atualiza dados do HLTV (ranking, resultados, partidas futuras)
2. Pra cada partida futura:
   a. Extrai features dos dois times
   b. Roda o modelo preditivo
   c. Compara com odds (se disponíveis)
   d. Detecta value bets
   e. Envia alerta no Telegram
3. Aguarda intervalo e repete

Uso:
    python main.py                   # Roda contínuo
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
    logger.info("⏹ Encerrando...")
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
) -> int:
    """
    Analisa todas as partidas futuras.

    Returns:
        Número de value bets detectados
    """
    if not predictor.is_trained:
        logger.warning("[ANALYSIS] Modelo não treinado — rode: python main.py --train")
        return 0

    upcoming = db.get_upcoming_matches()
    if not upcoming:
        logger.info("[ANALYSIS] Nenhuma partida futura no banco")
        return 0

    logger.info(f"[ANALYSIS] Analisando {len(upcoming)} partidas futuras...")
    value_count = 0
    analyzed = 0

    for match in upcoming:
        # Extrai features
        features = features_ext.extract(match)
        if not features:
            continue

        # Predição
        prediction = predictor.predict(features)
        if not prediction:
            continue

        analyzed += 1

        # Detecção de value (sem odds por enquanto — só predição)
        # Quando tiver integração com API de odds, passa aqui
        analysis = value_detector.analyze(
            prediction=prediction,
            odds_team1=None,  # TODO: integrar API de odds
            odds_team2=None,
            match=match,
        )

        if not analysis:
            continue

        # Gera relatório
        report = value_detector.generate_report(analysis, match)

        # Salva predição no banco
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

        # Envia alerta
        if analysis.get("has_value"):
            notifier.value_bet_alert(report, match)
            value_count += 1
        elif prediction["confidence"] >= 65:
            # Envia mesmo sem value se confiança alta (informativo)
            notifier.prediction_alert(report, match)

        # Log
        t1 = match.get("team1_name", "?")
        t2 = match.get("team2_name", "?")
        p1 = prediction["team1_win_prob"]
        p2 = prediction["team2_win_prob"]
        conf = prediction["confidence"]
        logger.info(f"  {t1} ({p1:.0f}%) vs {t2} ({p2:.0f}%) — confiança {conf:.0f}%")

    logger.info(f"[ANALYSIS] {analyzed} partidas analisadas, {value_count} value bets")
    return value_count


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
    logger.info("  🎮 CS2 ANALYST BOT")
    logger.info("=" * 50)

    stats = db.get_stats()
    logger.info(f"  Times: {stats['teams']} | Partidas: {stats['completed_matches']}")
    logger.info(f"  Modelo treinado: {'✅' if predictor.is_trained else '❌'}")
    logger.info(f"  Intervalo: {interval // 60} min")
    logger.info("=" * 50)

    notifier.startup(stats)

    last_daily_update = None

    while _running:
        try:
            now = datetime.now()

            # Atualização diária de dados (ranking, stats)
            if last_daily_update != now.date() and now.hour >= daily_hour:
                logger.info("[SCHEDULER] Atualização diária de dados...")
                await update_data(db, config)
                last_daily_update = now.date()

                # Re-treina modelo com dados novos
                if stats["completed_matches"] >= 50:
                    train_model(db, config, notifier)

            # Busca partidas futuras e resultados novos
            scraper = HLTVScraper(db, config)
            await scraper.scrape_upcoming_matches()
            await scraper.scrape_results()
            await scraper.close()

            # Analisa partidas futuras
            await analyze_upcoming(db, features_ext, predictor, value_detector, notifier)

            if once:
                logger.info("Modo --once: encerrando.")
                break

        except Exception as e:
            logger.exception(f"❌ Erro no loop: {e}")
            notifier.error(str(e))

        # Aguarda próximo ciclo
        logger.info(f"[SCHEDULER] Próximo scan em {interval // 60} min")
        for _ in range(interval):
            if not _running:
                break
            time.sleep(1)

    logger.info("👋 Bot encerrado.")


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
        print(f"\n📊 CS2 Analyst — Stats do banco")
        print(f"{'━' * 35}")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        # Mostra últimas predições
        preds = db.get_prediction_history(5)
        if preds:
            print(f"\n📋 Últimas predições:")
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
            print(f"\n✅ Modelo treinado: {metrics['model']}")
            print(f"   CV Accuracy: {metrics['cv_accuracy']:.1f}%")
            print(f"   Train Accuracy: {metrics['train_accuracy']:.1f}%")
        return

    asyncio.run(run(config, once=args.once))


if __name__ == "__main__":
    main()

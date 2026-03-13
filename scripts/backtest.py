#!/usr/bin/env python3
"""
Backtest — Treina e avalia o modelo preditivo.

Roda DEPOIS do backfill.
Extrai features das partidas históricas, treina o modelo,
e avalia a performance com cross-validation temporal.

Uso:
    python -m scripts.backtest
    python -m scripts.backtest --config config.yaml
"""

import sys
import logging
import argparse
from collections import Counter

import yaml
import numpy as np

from db.models import Database
from analysis.features import FeatureExtractor
from analysis.predictor import Predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def backtest(config: dict):
    db = Database(config["database"]["path"])
    features_extractor = FeatureExtractor(db, config)
    predictor = Predictor(config)

    logger.info("=" * 50)
    logger.info("  CS2 ANALYST — Backtest do modelo")
    logger.info("=" * 50)

    # 1. Extrai features
    logger.info("\n[1/3] Extraindo features das partidas históricas...")
    features_list, labels, match_dates = features_extractor.extract_training_data(include_dates=True)

    if len(features_list) < 20:
        logger.error(
            f"Apenas {len(features_list)} amostras extraídas. "
            "Precisa de mais dados. Rode o backfill novamente ou "
            "aguarde mais partidas serem coletadas."
        )
        return

    # Stats dos dados
    label_counts = Counter(labels)
    logger.info(f"  Amostras: {len(features_list)}")
    logger.info(f"  Team1 venceu: {label_counts[1]} ({label_counts[1]/len(labels)*100:.1f}%)")
    logger.info(f"  Team2 venceu: {label_counts[0]} ({label_counts[0]/len(labels)*100:.1f}%)")
    logger.info(f"  Features: {len(features_list[0])}")

    # 2. Treina modelo
    logger.info("\n[2/3] Treinando modelo...")
    metrics = predictor.train(features_list, labels, match_dates=match_dates)

    if "error" in metrics:
        logger.error(f"Erro no treinamento: {metrics['error']}")
        return

    # 3. Relatório
    logger.info("\n[3/3] Resultados:")
    logger.info("=" * 50)
    logger.info(f"  Modelo:         {metrics['model']}")
    logger.info(f"  Amostras:       {metrics['samples']}")
    logger.info(f"  Features:       {metrics['features']}")
    logger.info(f"  CV Accuracy:    {metrics['cv_accuracy']:.1f}% (±{metrics['cv_std']:.1f}%)")
    logger.info(f"  Train Accuracy: {metrics['train_accuracy']:.1f}%")
    logger.info(f"  Log Loss:       {metrics['train_logloss']:.4f}")
    logger.info(f"  Brier Score:    {metrics['train_brier']:.4f}")
    logger.info(f"  Holdout N:      {metrics.get('holdout_size', 0)}")
    logger.info(f"  Holdout Acc:    {metrics.get('holdout_accuracy', 0):.1f}%")
    logger.info(f"  Holdout LogLoss:{metrics.get('holdout_logloss', 0):.4f}")
    logger.info(f"  Holdout Brier:  {metrics.get('holdout_brier', 0):.4f}")
    logger.info(f"  Conf. sugerida: {metrics.get('recommended_min_confidence', 0):.1f}%")
    logger.info(
        f"  Precision@thr:  {metrics.get('threshold_precision', 0):.1f}% "
        f"(coverage={metrics.get('threshold_coverage', 0):.1f}%)"
    )
    logger.info("")
    logger.info("  Top 10 features mais importantes:")
    for name, imp in metrics.get("top_features", [])[:10]:
        bar = "█" * int(imp * 50)
        logger.info(f"    {name:30s} {imp:.4f} {bar}")

    logger.info("")

    # Interpretação dos resultados
    cv_acc = metrics["cv_accuracy"]
    if cv_acc >= 60:
        logger.info("  ✅ EXCELENTE — Modelo com edge significativo")
        logger.info("     O modelo pode ser usado pra detectar value bets")
    elif cv_acc >= 55:
        logger.info("  🟡 BOM — Edge marginal detectado")
        logger.info("     Pode funcionar com gestão de banca conservadora")
    elif cv_acc >= 52:
        logger.info("  ⚠️  MARGINAL — Edge mínimo")
        logger.info("     Pode não cobrir a margem das casas (~4-8%)")
    else:
        logger.info("  ❌ INSUFICIENTE — Sem edge detectável")
        logger.info("     Modelo não é melhor que aleatório após margem da casa")
        logger.info("     NÃO use pra apostar com dinheiro real")

    logger.info("")
    logger.info("=" * 50)
    logger.info("  Modelo salvo em: " + config["model"]["path"])
    logger.info("  Próximo passo: python main.py")
    logger.info("=" * 50)

    # Simula predição de exemplo com a primeira partida
    logger.info("\n📋 Exemplo de predição:")
    example = features_list[0]
    pred = predictor.predict(example)
    if pred:
        logger.info(f"  Team1 win: {pred['team1_win_prob']:.1f}%")
        logger.info(f"  Team2 win: {pred['team2_win_prob']:.1f}%")
        logger.info(f"  Confiança: {pred['confidence']:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="CS2 Analyst - Backtest")
    parser.add_argument("--config", "-c", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    backtest(config)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Backfill — Coleta dados históricos do HLTV.

Roda UMA VEZ antes de usar o bot.
Coleta rankings, times, jogadores e resultados recentes.

Uso:
    python -m scripts.backfill
    python -m scripts.backfill --config config.yaml
"""

import sys
import asyncio
import logging
import argparse

import yaml

from db.models import Database
from scraper.hltv import HLTVScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def backfill(config: dict):
    db = Database(config["database"]["path"])
    scraper = HLTVScraper(db, config)

    logger.info("=" * 50)
    logger.info("  CS2 ANALYST — Backfill de dados históricos")
    logger.info("=" * 50)

    # 1. Rankings
    logger.info("\n[1/4] Buscando ranking de times...")
    await scraper.scrape_top_teams(30)
    await asyncio.sleep(3)

    # 2. Info detalhada dos times
    logger.info("\n[2/4] Buscando info dos times e jogadores...")
    teams = db.get_all_teams()
    for i, team in enumerate(teams[:30]):
        logger.info(f"  [{i+1}/30] {team['name']}...")
        await scraper.scrape_team_info(team["id"], team["name"])
        await asyncio.sleep(2)

    # 3. Resultados recentes
    logger.info("\n[3/4] Buscando resultados recentes...")
    await scraper.scrape_results(pages=1)
    await asyncio.sleep(3)

    # 4. Partidas futuras
    logger.info("\n[4/4] Buscando partidas futuras...")
    await scraper.scrape_upcoming_matches()

    await scraper.close()

    # Stats
    stats = db.get_stats()
    logger.info("\n" + "=" * 50)
    logger.info("  Backfill completo!")
    logger.info(f"  Times:     {stats['teams']}")
    logger.info(f"  Jogadores: {stats['players']}")
    logger.info(f"  Partidas:  {stats['completed_matches']}")
    logger.info(f"  Futuras:   {stats['upcoming_matches']}")
    logger.info("=" * 50)
    logger.info("\nPróximo passo: python -m scripts.backtest")


def main():
    parser = argparse.ArgumentParser(description="CS2 Analyst - Backfill")
    parser.add_argument("--config", "-c", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    asyncio.run(backfill(config))


if __name__ == "__main__":
    main()

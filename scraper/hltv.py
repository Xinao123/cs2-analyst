"""
HLTV Scraper — Coleta dados do HLTV.org.

Usa hltv-async-api para scraping assíncrono.
Fallback para requests diretos quando necessário.
Respeita rate limits do HLTV com delays configuráveis.
"""

import asyncio
import logging
from datetime import datetime

from db.models import Database

logger = logging.getLogger(__name__)


class HLTVScraper:
    """Scraper assíncrono para dados do HLTV."""

    def __init__(self, db: Database, config: dict):
        self.db = db
        self.config = config
        scraper_cfg = config.get("scraper", {})
        self.min_delay = scraper_cfg.get("min_delay", 4.0)
        self.max_delay = scraper_cfg.get("max_delay", 10.0)
        self.proxy_path = scraper_cfg.get("proxy_path", "")
        self._hltv = None

    async def _get_client(self):
        """Inicializa o client HLTV (lazy)."""
        if self._hltv is None:
            from hltv_async_api import Hltv

            kwargs = {
                "min_delay": self.min_delay,
                "max_delay": self.max_delay,
            }
            if self.proxy_path:
                kwargs["proxy_path"] = self.proxy_path

            self._hltv = Hltv(**kwargs)
        return self._hltv

    async def close(self):
        if self._hltv:
            await self._hltv.close()
            self._hltv = None

    # ============================================================
    # Rankings & Teams
    # ============================================================

    async def scrape_top_teams(self, max_teams: int = 30):
        """Busca ranking atual do HLTV e salva os times."""
        logger.info(f"[HLTV] Buscando top {max_teams} times...")
        hltv = await self._get_client()

        try:
            teams = await hltv.get_top_teams(max_teams)
            if not teams:
                logger.warning("[HLTV] Nenhum time retornado")
                return 0

            count = 0
            for team in teams:
                team_id = int(team.get("id", 0))
                if team_id == 0:
                    continue

                self.db.upsert_team(
                    team_id=team_id,
                    name=team.get("title", "Unknown"),
                    ranking=int(team.get("rank", 0)),
                    ranking_points=int(team.get("points", 0)),
                )
                count += 1

            logger.info(f"[HLTV] {count} times atualizados no ranking")
            return count

        except Exception as e:
            logger.error(f"[HLTV] Erro ao buscar ranking: {e}")
            return 0

    async def scrape_team_info(self, team_id: int, team_name: str):
        """Busca informações detalhadas de um time."""
        hltv = await self._get_client()

        try:
            info = await hltv.get_team_info(team_id, team_name.lower().replace(" ", "-"))
            if not info:
                return

            # Atualiza jogadores do time
            players = info.get("players", {})
            for player_name, player_id in players.items():
                if player_id:
                    self.db.upsert_player(
                        player_id=int(player_id),
                        name=player_name,
                        team_id=team_id,
                    )

            logger.debug(f"[HLTV] Info do time {team_name}: {len(players)} jogadores")

        except Exception as e:
            logger.error(f"[HLTV] Erro ao buscar info de {team_name}: {e}")

    async def scrape_player_stats(self, player_id: int, player_name: str):
        """Busca stats detalhadas de um jogador."""
        hltv = await self._get_client()

        try:
            stats = await hltv.get_player_info(player_id, player_name.lower())
            if not stats:
                return

            # Extrai stats numéricas do retorno (formato varia)
            self.db.upsert_player(
                player_id=player_id,
                name=stats.get("nickname", player_name),
                team_id=stats.get("team_id", 0),
                rating=_safe_float(stats.get("rating", 0)),
                kd_ratio=_safe_float(stats.get("kd", 0)),
                maps_played=_safe_int(stats.get("maps_played", 0)),
            )

        except Exception as e:
            logger.debug(f"[HLTV] Erro ao buscar stats de {player_name}: {e}")

    # ============================================================
    # Matches
    # ============================================================

    async def scrape_upcoming_matches(self):
        """Busca partidas futuras do HLTV."""
        logger.info("[HLTV] Buscando partidas futuras...")
        hltv = await self._get_client()

        try:
            matches = await hltv.get_upcoming_matches()
            if not matches:
                logger.info("[HLTV] Nenhuma partida futura encontrada")
                return 0

            count = 0
            for match in matches:
                try:
                    match_id = match.get("id")
                    if not match_id:
                        continue

                    teams = match.get("teams", [])
                    if len(teams) < 2:
                        continue

                    team1 = teams[0] if isinstance(teams[0], dict) else {"name": str(teams[0]), "id": 0}
                    team2 = teams[1] if isinstance(teams[1], dict) else {"name": str(teams[1]), "id": 0}

                    team1_id = int(team1.get("id", 0))
                    team2_id = int(team2.get("id", 0))

                    if team1_id == 0 or team2_id == 0:
                        continue

                    # Garante que os times existem no DB
                    if not self.db.get_team(team1_id):
                        self.db.upsert_team(team1_id, team1.get("name", "TBD"))
                    if not self.db.get_team(team2_id):
                        self.db.upsert_team(team2_id, team2.get("name", "TBD"))

                    self.db.upsert_match(
                        hltv_id=int(match_id),
                        date=match.get("date", datetime.now().isoformat()),
                        event_name=match.get("event", ""),
                        best_of=_safe_int(match.get("bestOf", 1)),
                        team1_id=team1_id,
                        team2_id=team2_id,
                        status="upcoming",
                    )
                    count += 1

                except (KeyError, TypeError, ValueError) as e:
                    logger.debug(f"[HLTV] Erro ao parsear partida: {e}")
                    continue

            logger.info(f"[HLTV] {count} partidas futuras salvas")
            return count

        except Exception as e:
            logger.error(f"[HLTV] Erro ao buscar partidas: {e}")
            return 0

    async def scrape_results(self, pages: int = 1):
        """Busca resultados recentes."""
        logger.info(f"[HLTV] Buscando resultados ({pages} página(s))...")
        hltv = await self._get_client()

        try:
            results = await hltv.get_results()
            if not results:
                return 0

            count = 0
            for match in results:
                try:
                    match_id = match.get("id")
                    if not match_id:
                        continue

                    teams = match.get("teams", [])
                    if len(teams) < 2:
                        continue

                    team1 = teams[0] if isinstance(teams[0], dict) else {"name": str(teams[0]), "id": 0}
                    team2 = teams[1] if isinstance(teams[1], dict) else {"name": str(teams[1]), "id": 0}

                    team1_id = int(team1.get("id", 0))
                    team2_id = int(team2.get("id", 0))

                    if team1_id == 0 or team2_id == 0:
                        continue

                    # Garante times no DB
                    if not self.db.get_team(team1_id):
                        self.db.upsert_team(team1_id, team1.get("name", "TBD"))
                    if not self.db.get_team(team2_id):
                        self.db.upsert_team(team2_id, team2.get("name", "TBD"))

                    # Extrai scores
                    result = match.get("result", "")
                    t1_score, t2_score, winner_id = _parse_result(
                        result, team1_id, team2_id
                    )

                    self.db.upsert_match(
                        hltv_id=int(match_id),
                        date=match.get("date", ""),
                        event_name=match.get("event", ""),
                        team1_id=team1_id,
                        team2_id=team2_id,
                        team1_score=t1_score,
                        team2_score=t2_score,
                        winner_id=winner_id,
                        status="completed",
                    )
                    count += 1

                except (KeyError, TypeError, ValueError) as e:
                    logger.debug(f"[HLTV] Erro ao parsear resultado: {e}")
                    continue

            logger.info(f"[HLTV] {count} resultados salvos")
            return count

        except Exception as e:
            logger.error(f"[HLTV] Erro ao buscar resultados: {e}")
            return 0

    # ============================================================
    # Full update
    # ============================================================

    async def full_update(self):
        """Executa atualização completa de dados."""
        logger.info("[HLTV] Iniciando atualização completa...")

        await self.scrape_top_teams(30)
        await asyncio.sleep(2)

        await self.scrape_results(pages=1)
        await asyncio.sleep(2)

        await self.scrape_upcoming_matches()

        # Busca info detalhada dos top 20 times
        teams = self.db.get_all_teams()[:20]
        for team in teams:
            await self.scrape_team_info(team["id"], team["name"])
            await asyncio.sleep(1)

        stats = self.db.get_stats()
        logger.info(
            f"[HLTV] Atualização completa: "
            f"{stats['teams']} times, {stats['players']} jogadores, "
            f"{stats['completed_matches']} partidas, "
            f"{stats['upcoming_matches']} futuras"
        )

        await self.close()
        return stats


# ============================================================
# Helpers
# ============================================================

def _safe_float(val) -> float:
    try:
        return float(str(val).replace("%", "").replace(",", "."))
    except (ValueError, TypeError):
        return 0.0


def _safe_int(val) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def _parse_result(result_str: str, team1_id: int, team2_id: int):
    """Parseia string de resultado como '2 - 1' ou '16 - 9'."""
    try:
        if not result_str or "-" not in str(result_str):
            return None, None, None

        parts = str(result_str).split("-")
        t1 = int(parts[0].strip())
        t2 = int(parts[1].strip())

        winner = team1_id if t1 > t2 else team2_id if t2 > t1 else None
        return t1, t2, winner
    except (ValueError, IndexError):
        return None, None, None

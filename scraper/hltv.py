"""
HLTV Scraper - Coleta dados do HLTV.org.

Usa hltv-async-api para scraping assincrono.
Respeita rate limits do HLTV com delays configuraveis.
"""

import asyncio
import logging
import re
import zlib
from datetime import datetime

from db.models import Database

logger = logging.getLogger(__name__)


class HLTVScraper:
    """Scraper assincrono para dados do HLTV."""

    def __init__(self, db: Database, config: dict):
        self.db = db
        self.config = config
        scraper_cfg = config.get("scraper", {})
        self.min_delay = scraper_cfg.get("min_delay", 4.0)
        self.max_delay = scraper_cfg.get("max_delay", 10.0)
        self.proxy_path = scraper_cfg.get("proxy_path", "")
        self.min_rating = _safe_int(scraper_cfg.get("min_rating", 0))
        self.upcoming_days = max(1, _safe_int(scraper_cfg.get("upcoming_days", 2)))
        self.results_days = max(1, _safe_int(scraper_cfg.get("results_days", 2)))
        self.results_max = max(30, _safe_int(scraper_cfg.get("results_max", 200)))
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
        """Busca informacoes detalhadas de um time."""
        hltv = await self._get_client()

        try:
            info = await hltv.get_team_info(team_id, team_name.lower().replace(" ", "-"))
            if not info:
                return

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
            # Compatibilidade com hltv-async-api 0.8.x.
            matches = await hltv.get_matches(
                days=self.upcoming_days,
                min_rating=self.min_rating,
                live=False,
                future=True,
            )
            if not matches:
                logger.info(
                    "[HLTV] Retorno vazio para upcoming (days=%s, min_rating=%s), tentando modo amplo...",
                    self.upcoming_days,
                    self.min_rating,
                )
                matches = await hltv.get_matches(days=7, min_rating=0, live=False, future=True)
            if not matches:
                logger.info("[HLTV] Upcoming ainda vazio, tentando fallback por eventos...")
                matches = await self._collect_upcoming_from_events(hltv)
            if matches is None:
                logger.warning("[HLTV] API retornou None para partidas futuras")
                return 0
            if not matches:
                logger.warning("[HLTV] API retornou 0 partidas futuras")
                return 0

            total = len(matches)
            saved = 0
            skips = {"sem_id": 0, "sem_score": 0, "payload_invalido": 0}

            for match in matches:
                try:
                    normalized = self._normalize_upcoming_payload(match)
                    if not normalized:
                        skips["payload_invalido"] += 1
                        continue

                    team1_id = normalized["team1_id"]
                    team2_id = normalized["team2_id"]
                    team1_name = normalized["team1_name"]
                    team2_name = normalized["team2_name"]

                    team1_id = self._resolve_team_id(team1_id, team1_name)
                    team2_id = self._resolve_team_id(team2_id, team2_name)

                    if team1_id == 0 or team2_id == 0:
                        skips["sem_id"] += 1
                        continue

                    if not team1_name:
                        team = self.db.get_team(team1_id)
                        team1_name = team.get("name", "TBD") if team else "TBD"
                    if not team2_name:
                        team = self.db.get_team(team2_id)
                        team2_name = team.get("name", "TBD") if team else "TBD"

                    if not self.db.get_team(team1_id):
                        self.db.upsert_team(team1_id, team1_name)
                    if not self.db.get_team(team2_id):
                        self.db.upsert_team(team2_id, team2_name)

                    self.db.upsert_match(
                        hltv_id=normalized["match_id"],
                        date=normalized["date"],
                        event_name=normalized["event_name"],
                        best_of=normalized["best_of"],
                        team1_id=team1_id,
                        team2_id=team2_id,
                        status="upcoming",
                    )
                    saved += 1

                except (KeyError, TypeError, ValueError) as e:
                    skips["payload_invalido"] += 1
                    logger.debug(f"[HLTV] Erro ao parsear partida: {e}")

            discarded = total - saved
            logger.info(
                "[HLTV] Partidas futuras: retornadas=%s salvas=%s descartadas=%s "
                "(sem_id=%s, sem_score=%s, payload_invalido=%s)",
                total,
                saved,
                discarded,
                skips["sem_id"],
                skips["sem_score"],
                skips["payload_invalido"],
            )
            return saved

        except Exception as e:
            logger.error(f"[HLTV] Erro ao buscar partidas: {e}")
            return 0

    async def scrape_results(self, pages: int = 1):
        """Busca resultados recentes."""
        logger.info(f"[HLTV] Buscando resultados ({pages} pagina(s))...")
        hltv = await self._get_client()

        try:
            results = await hltv.get_results(
                days=max(self.results_days, pages + 1),
                min_rating=self.min_rating,
                max=self.results_max,
                featured=True,
                regular=True,
            )
            if not results:
                logger.info(
                    "[HLTV] Retorno vazio para resultados (days=%s, min_rating=%s, max=%s), tentando modo amplo...",
                    max(self.results_days, pages + 1),
                    self.min_rating,
                    self.results_max,
                )
                results = await hltv.get_results(
                    days=7,
                    min_rating=0,
                    max=300,
                    featured=True,
                    regular=True,
                )
            if results is None:
                logger.warning("[HLTV] API retornou None para resultados")
                return 0
            if not results:
                logger.warning("[HLTV] API retornou 0 resultados")
                return 0

            total = len(results)
            saved = 0
            skips = {"sem_id": 0, "sem_score": 0, "payload_invalido": 0}

            for match in results:
                try:
                    normalized = self._normalize_result_payload(match)
                    if not normalized:
                        skips["payload_invalido"] += 1
                        continue

                    team1_id = normalized["team1_id"]
                    team2_id = normalized["team2_id"]
                    team1_name = normalized["team1_name"]
                    team2_name = normalized["team2_name"]

                    team1_id = self._resolve_team_id(team1_id, team1_name)
                    team2_id = self._resolve_team_id(team2_id, team2_name)

                    if team1_id == 0 or team2_id == 0:
                        skips["sem_id"] += 1
                        continue

                    t1_score = normalized["team1_score"]
                    t2_score = normalized["team2_score"]
                    if t1_score is None or t2_score is None:
                        skips["sem_score"] += 1
                        continue

                    if not team1_name:
                        team = self.db.get_team(team1_id)
                        team1_name = team.get("name", "TBD") if team else "TBD"
                    if not team2_name:
                        team = self.db.get_team(team2_id)
                        team2_name = team.get("name", "TBD") if team else "TBD"

                    if not self.db.get_team(team1_id):
                        self.db.upsert_team(team1_id, team1_name)
                    if not self.db.get_team(team2_id):
                        self.db.upsert_team(team2_id, team2_name)

                    winner_id = (
                        team1_id if t1_score > t2_score else
                        team2_id if t2_score > t1_score else
                        None
                    )

                    self.db.upsert_match(
                        hltv_id=normalized["match_id"],
                        date=normalized["date"],
                        event_name=normalized["event_name"],
                        team1_id=team1_id,
                        team2_id=team2_id,
                        team1_score=t1_score,
                        team2_score=t2_score,
                        winner_id=winner_id,
                        status="completed",
                    )
                    saved += 1

                except (KeyError, TypeError, ValueError) as e:
                    skips["payload_invalido"] += 1
                    logger.debug(f"[HLTV] Erro ao parsear resultado: {e}")

            discarded = total - saved
            logger.info(
                "[HLTV] Resultados: retornados=%s salvos=%s descartados=%s "
                "(sem_id=%s, sem_score=%s, payload_invalido=%s)",
                total,
                saved,
                discarded,
                skips["sem_id"],
                skips["sem_score"],
                skips["payload_invalido"],
            )
            return saved

        except Exception as e:
            logger.error(f"[HLTV] Erro ao buscar resultados: {e}")
            return 0

    def _normalize_upcoming_payload(self, match: dict) -> dict | None:
        if not isinstance(match, dict):
            return None

        match_id = _safe_int(match.get("id"))
        if match_id <= 0:
            return None

        team1_name = _safe_str(match.get("team1"))
        team2_name = _safe_str(match.get("team2"))
        team1_id = _safe_int(match.get("t1_id", match.get("team1_id", 0)))
        team2_id = _safe_int(match.get("t2_id", match.get("team2_id", 0)))

        legacy_teams = match.get("teams", [])
        if isinstance(legacy_teams, list) and len(legacy_teams) >= 2:
            lt1 = _extract_team(legacy_teams[0])
            lt2 = _extract_team(legacy_teams[1])
            if not team1_name:
                team1_name = lt1["name"]
            if not team2_name:
                team2_name = lt2["name"]
            if team1_id == 0:
                team1_id = lt1["id"]
            if team2_id == 0:
                team2_id = lt2["id"]

        return {
            "match_id": match_id,
            "date": _normalize_match_datetime(match.get("date"), match.get("time")),
            "event_name": _safe_str(match.get("event")),
            "best_of": _parse_best_of(match.get("bestOf", match.get("maps", 1))),
            "team1_id": team1_id,
            "team2_id": team2_id,
            "team1_name": team1_name,
            "team2_name": team2_name,
        }

    def _normalize_result_payload(self, match: dict) -> dict | None:
        if not isinstance(match, dict):
            return None

        match_id = _safe_int(match.get("id"))
        if match_id <= 0:
            return None

        team1_name = _safe_str(match.get("team1"))
        team2_name = _safe_str(match.get("team2"))
        team1_id = _safe_int(match.get("t1_id", match.get("team1_id", 0)))
        team2_id = _safe_int(match.get("t2_id", match.get("team2_id", 0)))

        legacy_teams = match.get("teams", [])
        if isinstance(legacy_teams, list) and len(legacy_teams) >= 2:
            lt1 = _extract_team(legacy_teams[0])
            lt2 = _extract_team(legacy_teams[1])
            if not team1_name:
                team1_name = lt1["name"]
            if not team2_name:
                team2_name = lt2["name"]
            if team1_id == 0:
                team1_id = lt1["id"]
            if team2_id == 0:
                team2_id = lt2["id"]

        team1_score = _safe_score(match.get("score1"))
        team2_score = _safe_score(match.get("score2"))
        if team1_score is None or team2_score is None:
            team1_score, team2_score = _parse_score_from_result(match.get("result", ""))

        return {
            "match_id": match_id,
            "date": _normalize_match_datetime(match.get("date")),
            "event_name": _safe_str(match.get("event")),
            "team1_id": team1_id,
            "team2_id": team2_id,
            "team1_name": team1_name,
            "team2_name": team2_name,
            "team1_score": team1_score,
            "team2_score": team2_score,
        }

    async def _collect_upcoming_from_events(self, hltv) -> list[dict]:
        """Fallback quando get_matches() retorna vazio: agrega jogos por evento."""
        out: list[dict] = []
        try:
            events = await hltv.get_events(outgoing=True, future=True, max_events=12)
        except Exception as e:
            logger.debug(f"[HLTV] Fallback eventos falhou ao listar eventos: {e}")
            return out

        if not events:
            return out

        seen_ids: set[int] = set()
        for event in events:
            event_id = _safe_int(event.get("id"))
            if event_id <= 0:
                continue
            event_title = _safe_str(event.get("title"))
            try:
                event_matches = await hltv.get_event_matches(event_id, days=max(2, self.upcoming_days))
            except Exception as e:
                logger.debug(f"[HLTV] Falha ao buscar partidas do evento {event_id}: {e}")
                continue

            for match in event_matches or []:
                match_id = _safe_int(match.get("id"))
                if match_id <= 0 or match_id in seen_ids:
                    continue
                seen_ids.add(match_id)
                if not match.get("event") and event_title:
                    match = dict(match)
                    match["event"] = event_title
                out.append(match)

        logger.info("[HLTV] Fallback por eventos encontrou %s partidas", len(out))
        return out

    def _resolve_team_id(self, team_id: int, team_name: str) -> int:
        if team_id > 0:
            return team_id

        name = _safe_str(team_name)
        if not name:
            return 0

        team = self.db.get_team_by_name(name)
        if team:
            return int(team["id"])

        relaxed_id = self._lookup_team_by_name_relaxed(name)
        if relaxed_id:
            return relaxed_id

        synthetic_id = _synthetic_team_id(name)
        if not self.db.get_team(synthetic_id):
            self.db.upsert_team(synthetic_id, name)
        return synthetic_id

    def _lookup_team_by_name_relaxed(self, team_name: str) -> int:
        target = _normalize_team_name(team_name)
        if not target:
            return 0

        for team in self.db.get_all_teams():
            if _normalize_team_name(team.get("name", "")) == target:
                return _safe_int(team.get("id"))
        return 0

    # ============================================================
    # Full update
    # ============================================================

    async def full_update(self):
        """Executa atualizacao completa de dados."""
        logger.info("[HLTV] Iniciando atualizacao completa...")

        await self.scrape_top_teams(30)
        await asyncio.sleep(2)

        await self.scrape_results(pages=1)
        await asyncio.sleep(2)

        await self.scrape_upcoming_matches()

        teams = self.db.get_all_teams()[:20]
        for team in teams:
            await self.scrape_team_info(team["id"], team["name"])
            await asyncio.sleep(1)

        stats = self.db.get_stats()
        logger.info(
            f"[HLTV] Atualizacao completa: "
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


def _safe_str(val) -> str:
    if val is None:
        return ""
    return str(val).strip()


def _safe_score(val) -> int | None:
    if val is None:
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)

    text = str(val).strip()
    if not text:
        return None

    match = re.search(r"-?\d+", text)
    if not match:
        return None

    try:
        return int(match.group())
    except ValueError:
        return None


def _extract_team(team_data) -> dict:
    if isinstance(team_data, dict):
        return {
            "id": _safe_int(team_data.get("id", team_data.get("team_id", 0))),
            "name": _safe_str(team_data.get("name", team_data.get("title", ""))),
        }

    if isinstance(team_data, str):
        return {"id": 0, "name": team_data.strip()}

    return {"id": 0, "name": ""}


def _normalize_team_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _safe_str(name).lower())


def _synthetic_team_id(team_name: str) -> int:
    """Gera id negativo estavel para times sem id no payload do HLTV."""
    norm = _normalize_team_name(team_name)
    if not norm:
        return 0
    # Mantem no range int32 e evita conflito com ids reais (positivos).
    return -((zlib.crc32(norm.encode("utf-8")) % 2_000_000_000) + 1)


def _parse_best_of(val) -> int:
    if isinstance(val, int):
        return max(1, val)

    text = _safe_str(val).lower()
    if not text:
        return 1

    if text.isdigit():
        return max(1, int(text))

    match = re.search(r"(?:bo\s*)?(\d)", text)
    if match:
        return max(1, int(match.group(1)))

    return 1


def _parse_score_from_result(result_str: str) -> tuple[int | None, int | None]:
    text = _safe_str(result_str)
    if not text or "-" not in text:
        return None, None

    left, right = text.split("-", 1)
    return _safe_score(left), _safe_score(right)


def _parse_result(result_str: str, team1_id: int, team2_id: int):
    """Mantido por compatibilidade: parseia '2 - 1' ou '16 - 9'."""
    t1_score, t2_score = _parse_score_from_result(result_str)
    if t1_score is None or t2_score is None:
        return None, None, None

    winner = team1_id if t1_score > t2_score else team2_id if t2_score > t1_score else None
    return t1_score, t2_score, winner


def _normalize_match_datetime(date_val, time_val=None) -> str:
    date_text = _safe_str(date_val)
    time_text = _safe_str(time_val)

    if not date_text or date_text.upper() == "LIVE":
        return datetime.now().isoformat(timespec="seconds")

    # Ja veio em ISO ou datetime parseavel
    try:
        parsed = datetime.fromisoformat(date_text.replace("Z", "+00:00"))
        if parsed.tzinfo:
            parsed = parsed.replace(tzinfo=None)
        if time_text and parsed.hour == 0 and parsed.minute == 0:
            parsed_time = _parse_time(time_text)
            if parsed_time:
                parsed = parsed.replace(hour=parsed_time.hour, minute=parsed_time.minute, second=0)
        return parsed.isoformat(timespec="seconds")
    except ValueError:
        pass

    # Formatos comuns da lib (ex: 12-03-2026 + 14:30)
    for date_fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            parsed = datetime.strptime(date_text, date_fmt)
            parsed_time = _parse_time(time_text)
            if parsed_time:
                parsed = parsed.replace(hour=parsed_time.hour, minute=parsed_time.minute, second=0)
            return parsed.isoformat(timespec="seconds")
        except ValueError:
            continue

    # Se nao reconheceu, preserva o valor original para nao perder informacao.
    return date_text


def _parse_time(time_text: str) -> datetime | None:
    if not time_text:
        return None

    for time_fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(time_text, time_fmt)
        except ValueError:
            continue

    return None

"""
HLTV Scraper - Coleta dados do HLTV.org.

Usa hltv-async-api para scraping assincrono.
Respeita rate limits do HLTV com delays configuraveis.
"""

import asyncio
import json
import logging
import os
import re
import time
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from db.models import Database

logger = logging.getLogger(__name__)

SYNTHETIC_MATCH_ID_BASE = 9_000_000_000
SYNTHETIC_TEAM_ID_BASE = 8_000_000_000
UPCOMING_SKIP_REASON_KEYS = (
    "sem_time",
    "sem_data",
    "fora_janela",
    "duplicada",
    "team_unresolved",
    "cloudflare",
    "payload_invalido",
)
HISTORY_SKIP_REASON_KEYS = (
    "payload_invalido",
    "sem_data",
    "sem_time",
    "team_unresolved",
    "sem_score",
    "duplicada",
    "status_invalido",
    "fora_janela",
    "api_error",
    "quota",
)


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
        self.upcoming_days_ahead = max(
            1,
            _safe_int(scraper_cfg.get("upcoming_days_ahead", self.upcoming_days)),
        )
        self.upcoming_provider = (
            _safe_str(scraper_cfg.get("upcoming_provider", "pandascore")).lower() or "pandascore"
        )
        self.upcoming_fallback = (
            _safe_str(scraper_cfg.get("upcoming_fallback", "hltv")).lower() or "hltv"
        )
        self.upcoming_max_pages = max(1, _safe_int(scraper_cfg.get("upcoming_max_pages", 5)))
        self.upcoming_retry_count = max(0, _safe_int(scraper_cfg.get("upcoming_retry_count", 3)))
        self.upcoming_retry_backoff_sec = max(
            1,
            _safe_int(scraper_cfg.get("upcoming_retry_backoff_sec", 2)),
        )
        self.results_days = max(1, _safe_int(scraper_cfg.get("results_days", 2)))
        self.results_max = max(30, _safe_int(scraper_cfg.get("results_max", 200)))
        self.pandascore_base_url = (
            _safe_str(scraper_cfg.get("pandascore_base_url", "https://api.pandascore.co"))
            .rstrip("/")
        )
        self.pandascore_token_env = (
            _safe_str(scraper_cfg.get("pandascore_token_env", "PANDASCORE_API_TOKEN"))
            or "PANDASCORE_API_TOKEN"
        )
        self.pandascore_timeout_sec = max(5, _safe_int(scraper_cfg.get("pandascore_timeout_sec", 20)))
        self.pandascore_history_enabled = bool(scraper_cfg.get("pandascore_history_enabled", True))
        self.pandascore_history_months = max(1, _safe_int(scraper_cfg.get("pandascore_history_months", 12)))
        self.pandascore_history_window_days = max(
            1,
            _safe_int(scraper_cfg.get("pandascore_history_window_days", 7)),
        )
        self.pandascore_history_per_page = max(
            20,
            min(100, _safe_int(scraper_cfg.get("pandascore_history_per_page", 100))),
        )
        self.pandascore_history_max_requests_per_hour = max(
            50,
            _safe_int(scraper_cfg.get("pandascore_history_max_requests_per_hour", 800)),
        )
        self.pandascore_history_sync_interval_minutes = max(
            5,
            _safe_int(scraper_cfg.get("pandascore_history_sync_interval_minutes", 30)),
        )
        self.pandascore_history_checkpoint_key = (
            _safe_str(scraper_cfg.get("pandascore_history_checkpoint_key", "pandascore_history_cursor"))
            or "pandascore_history_cursor"
        )
        self.user_agent = scraper_cfg.get(
            "user_agent",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        )
        self.referer = scraper_cfg.get("referer", "https://www.hltv.org/stats")
        self.hltv_timezone = scraper_cfg.get("hltv_timezone", "Europe/Copenhagen")
        self.cookie = scraper_cfg.get("cookie", "")
        extra_headers = scraper_cfg.get("extra_headers", {})
        self.extra_headers = extra_headers if isinstance(extra_headers, dict) else {}
        self._pandascore_token = _safe_str(os.getenv(self.pandascore_token_env))
        self._hltv = None

    async def _get_client(self):
        """Inicializa o client HLTV (lazy)."""
        if self._hltv is None:
            from hltv_async_api import Hltv

            kwargs = {
                "min_delay": self.min_delay,
                "max_delay": self.max_delay,
            }
            resolved_proxy_path = self._resolve_proxy_path()
            if resolved_proxy_path:
                kwargs["proxy_path"] = resolved_proxy_path

            self._hltv = Hltv(**kwargs)
            # Permite customizar headers para reduzir bloqueio por Cloudflare.
            self._hltv.headers.update(
                {
                    "user-agent": self.user_agent,
                    "referer": self.referer,
                    "hltvTimeZone": self.hltv_timezone,
                }
            )
            if self.cookie:
                self._hltv.headers["cookie"] = self.cookie
            if self.extra_headers:
                self._hltv.headers.update({str(k): str(v) for k, v in self.extra_headers.items()})
        return self._hltv

    def _resolve_proxy_path(self) -> str | None:
        proxy_path = _safe_str(self.proxy_path)
        if not proxy_path:
            return None

        raw = Path(proxy_path)
        candidates = [raw]
        if not raw.is_absolute():
            candidates.extend(
                [
                    Path("/app") / raw,
                    Path("/app/data") / raw,
                    Path("data") / raw,
                ]
            )

        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)

        logger.warning(
            "[HLTV] proxy_path configurado mas arquivo nao encontrado: %s. "
            "Tentados: %s. Seguindo sem proxy.",
            proxy_path,
            ", ".join(str(c) for c in candidates),
        )
        return None

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
        if team_id <= 0:
            logger.debug("[HLTV] Ignorando team_info para id sintetico: %s (%s)", team_id, team_name)
            return

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
        """Busca partidas futuras com cadeia de providers (primario -> fallback)."""
        providers = self._get_upcoming_provider_order()
        logger.info("[UPCOMING] Buscando partidas futuras (ordem=%s)...", " -> ".join(providers))

        last_report = self._new_upcoming_report()
        for idx, provider in enumerate(providers):
            if provider == "pandascore":
                report = await self._scrape_upcoming_from_pandascore()
            elif provider == "hltv":
                report = await self._scrape_upcoming_from_hltv()
            else:
                report = self._new_upcoming_report()
                report["error"] = f"provider_nao_suportado:{provider}"

            self._log_upcoming_report(provider, report)
            if report["saved"] > 0:
                return report["saved"]

            last_report = report
            if idx < len(providers) - 1:
                logger.info(
                    "[UPCOMING] Provider %s sem partidas salvas, tentando fallback...",
                    provider,
                )

        if last_report.get("error"):
            logger.warning("[UPCOMING] Nenhuma partida futura salva. Ultimo erro=%s", last_report["error"])
        else:
            logger.warning("[UPCOMING] Nenhuma partida futura salva por nenhum provider")
        return 0

    def _new_upcoming_report(self) -> dict:
        return {
            "returned": 0,
            "saved": 0,
            "skipped": 0,
            "reasons": {key: 0 for key in UPCOMING_SKIP_REASON_KEYS},
            "error": "",
        }

    def _get_upcoming_provider_order(self) -> list[str]:
        providers = []
        supported = {"pandascore", "hltv"}

        for value in (self.upcoming_provider, self.upcoming_fallback):
            provider = _safe_str(value).lower()
            if not provider:
                continue
            if provider not in supported:
                logger.warning("[UPCOMING] Provider desconhecido ignorado: %s", provider)
                continue
            if provider not in providers:
                providers.append(provider)

        if not providers:
            providers = ["pandascore", "hltv"]

        return providers

    def _log_upcoming_report(self, provider: str, report: dict):
        reasons = report.get("reasons", {})
        returned = _safe_int(report.get("returned"))
        saved = _safe_int(report.get("saved"))
        skipped = _safe_int(report.get("skipped", max(returned - saved, 0)))

        logger.info(
            "[UPCOMING][%s] retornadas=%s salvas=%s descartadas=%s "
            "(sem_time=%s, sem_data=%s, fora_janela=%s, duplicada=%s, "
            "team_unresolved=%s, cloudflare=%s, payload_invalido=%s)",
            provider,
            returned,
            saved,
            skipped,
            _safe_int(reasons.get("sem_time", 0)),
            _safe_int(reasons.get("sem_data", 0)),
            _safe_int(reasons.get("fora_janela", 0)),
            _safe_int(reasons.get("duplicada", 0)),
            _safe_int(reasons.get("team_unresolved", 0)),
            _safe_int(reasons.get("cloudflare", 0)),
            _safe_int(reasons.get("payload_invalido", 0)),
        )
        if report.get("error"):
            logger.warning("[UPCOMING][%s] erro=%s", provider, report["error"])

    async def _scrape_upcoming_from_pandascore(self) -> dict:
        token = _safe_str(os.getenv(self.pandascore_token_env)) or self._pandascore_token
        self._pandascore_token = token

        report = self._new_upcoming_report()
        if not token:
            report["error"] = f"token_ausente:{self.pandascore_token_env}"
            logger.warning(
                "[UPCOMING][pandascore] Token ausente em %s, modo degradado para fallback.",
                self.pandascore_token_env,
            )
            return report

        all_matches = []
        for page in range(1, self.upcoming_max_pages + 1):
            page_matches, page_error = await asyncio.to_thread(
                self._fetch_pandascore_upcoming_page,
                page,
                token,
            )
            if page_error:
                report["error"] = page_error
                break
            if not page_matches:
                break

            all_matches.extend(page_matches)
            if len(page_matches) < 100:
                break

        if not all_matches:
            return report

        return self._persist_upcoming_matches("pandascore", all_matches, report)

    def _fetch_pandascore_upcoming_page(
        self,
        page: int,
        token: str,
    ) -> tuple[list[dict], str]:
        url = f"{self.pandascore_base_url}/csgo/matches/upcoming"
        headers = {
            "accept": "application/json",
            "authorization": f"Bearer {token}",
            "user-agent": self.user_agent,
        }
        params = {
            "page": page,
            "per_page": 100,
            "sort": "begin_at",
        }

        max_attempts = self.upcoming_retry_count + 1
        for attempt in range(max_attempts):
            try:
                resp = requests.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=self.pandascore_timeout_sec,
                )
            except requests.RequestException as exc:
                if attempt >= max_attempts - 1:
                    return [], f"request_error:{exc.__class__.__name__}"
                time.sleep(self.upcoming_retry_backoff_sec * (attempt + 1))
                continue

            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt >= max_attempts - 1:
                    return [], f"http_{resp.status_code}"
                time.sleep(self.upcoming_retry_backoff_sec * (attempt + 1))
                continue
            if resp.status_code >= 400:
                return [], f"http_{resp.status_code}"

            try:
                payload = resp.json()
            except ValueError:
                return [], "payload_invalido"

            if isinstance(payload, dict) and isinstance(payload.get("data"), list):
                payload = payload["data"]
            if not isinstance(payload, list):
                return [], "payload_invalido"

            return payload, ""

        return [], "retry_exceeded"

    async def _scrape_upcoming_from_hltv(self) -> dict:
        """Busca partidas futuras no HLTV com fallback por eventos."""
        report = self._new_upcoming_report()
        logger.info("[HLTV] Buscando partidas futuras...")
        hltv = await self._get_client()

        try:
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
                report["error"] = "api_none"
                return report
            if not matches:
                cf_diag = self._diagnose_cloudflare_on_matches()
                if cf_diag["blocked"]:
                    report["reasons"]["cloudflare"] += 1
                    report["error"] = f"cloudflare:{cf_diag['status']}"
                    logger.warning(
                        "[HLTV] Possivel bloqueio Cloudflare em /matches "
                        "(status=%s, challenge=%s). Configure proxy_path e/ou scraper.cookie (cf_clearance).",
                        cf_diag["status"],
                        cf_diag["challenge"],
                    )
                return report

            return self._persist_upcoming_matches("hltv", matches, report)

        except Exception as e:
            report["error"] = f"provider_error:{e}"
            return report

    def _persist_upcoming_matches(self, provider: str, matches: list[dict], report: dict | None = None) -> dict:
        report = report or self._new_upcoming_report()
        report["returned"] = len(matches)

        reasons = report["reasons"]
        saved = 0
        seen_ids: set[int] = set()
        for match in matches:
            normalized = self._normalize_upcoming_payload_by_provider(provider, match)
            if not normalized:
                reasons["payload_invalido"] += 1
                continue

            match_id = _safe_int(normalized.get("match_id"))
            if match_id == 0:
                reasons["payload_invalido"] += 1
                continue
            if match_id in seen_ids:
                reasons["duplicada"] += 1
                continue
            seen_ids.add(match_id)

            match_date = _safe_str(normalized.get("date"))
            if not match_date:
                reasons["sem_data"] += 1
                continue

            parsed_date = _parse_datetime(match_date)
            if parsed_date is None:
                reasons["sem_data"] += 1
                continue
            if not self._is_within_upcoming_window(parsed_date):
                reasons["fora_janela"] += 1
                continue

            team1_name = _safe_str(normalized.get("team1_name"))
            team2_name = _safe_str(normalized.get("team2_name"))
            if not team1_name or not team2_name:
                reasons["sem_time"] += 1
                continue

            team1_id = self._resolve_team_id(
                _safe_int(normalized.get("team1_id")),
                team1_name,
                provider=provider,
            )
            team2_id = self._resolve_team_id(
                _safe_int(normalized.get("team2_id")),
                team2_name,
                provider=provider,
            )
            if team1_id == 0 or team2_id == 0 or team1_id == team2_id:
                reasons["team_unresolved"] += 1
                continue

            self._ensure_team_exists(team1_id, team1_name)
            self._ensure_team_exists(team2_id, team2_name)

            self.db.upsert_match(
                hltv_id=match_id,
                date=match_date,
                event_name=_safe_str(normalized.get("event_name")),
                best_of=_parse_best_of(normalized.get("best_of", 1)),
                team1_id=team1_id,
                team2_id=team2_id,
                status="upcoming",
            )
            saved += 1

        report["saved"] = saved
        report["skipped"] = max(report["returned"] - report["saved"], 0)
        return report

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

    def _normalize_upcoming_payload_by_provider(self, provider: str, match: dict) -> dict | None:
        if provider == "pandascore":
            return self._normalize_pandascore_upcoming_payload(match)
        return self._normalize_upcoming_payload(match)

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

    def _normalize_pandascore_upcoming_payload(self, match: dict) -> dict | None:
        if not isinstance(match, dict):
            return None

        provider_match_id = _safe_int(match.get("id"))
        if provider_match_id <= 0:
            return None

        teams = _extract_pandascore_teams(match)
        if len(teams) < 2:
            return None

        event_name = _safe_str(
            match.get("tournament", {}).get("name")
            or match.get("serie", {}).get("name")
            or match.get("league", {}).get("name")
            or match.get("videogame_title")
        )
        raw_date = (
            match.get("begin_at")
            or match.get("scheduled_at")
            or match.get("original_scheduled_at")
        )

        return {
            "match_id": _synthetic_provider_match_id(provider_match_id),
            "date": _normalize_match_datetime(raw_date),
            "event_name": event_name,
            "best_of": _parse_best_of(match.get("number_of_games", match.get("best_of", 1))),
            "team1_id": _safe_int(teams[0].get("id")),
            "team2_id": _safe_int(teams[1].get("id")),
            "team1_name": _safe_str(teams[0].get("name")),
            "team2_name": _safe_str(teams[1].get("name")),
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

    def _diagnose_cloudflare_on_matches(self) -> dict:
        headers = {
            "user-agent": self.user_agent,
            "referer": self.referer,
            "hltvTimeZone": self.hltv_timezone,
        }
        if self.cookie:
            headers["cookie"] = self.cookie
        if self.extra_headers:
            headers.update({str(k): str(v) for k, v in self.extra_headers.items()})

        try:
            resp = requests.get("https://www.hltv.org/matches", headers=headers, timeout=20)
            body = resp.text.lower()
            challenge = (
                ("just a moment" in body)
                or ("enable javascript and cookies" in body)
                or ("challenge-error-title" in body)
            )
            blocked = resp.status_code in (403, 429) or challenge
            return {"blocked": blocked, "status": resp.status_code, "challenge": challenge}
        except Exception:
            return {"blocked": False, "status": "error", "challenge": False}

    def _resolve_team_id(self, team_id: int, team_name: str, provider: str = "hltv") -> int:
        name = _safe_str(team_name)

        if provider == "hltv" and team_id > 0:
            return team_id

        if name:
            team = self.db.get_team_by_name(name)
            if team:
                return int(team["id"])

            relaxed_id = self._lookup_team_by_name_relaxed(name)
            if relaxed_id:
                return relaxed_id

        if team_id > 0 and provider != "hltv":
            synthetic_id = _synthetic_provider_team_id(team_id)
            if name:
                self.db.upsert_team(synthetic_id, name, ranking=9999)
            return synthetic_id

        if not name:
            return 0

        synthetic_id = _synthetic_team_id(name)
        self.db.upsert_team(synthetic_id, name, ranking=9999)
        return synthetic_id

    def _lookup_team_by_name_relaxed(self, team_name: str) -> int:
        target = _normalize_team_name(team_name)
        if not target:
            return 0

        for team in self.db.get_all_teams():
            if _normalize_team_name(team.get("name", "")) == target:
                return _safe_int(team.get("id"))
        return 0

    def _ensure_team_exists(self, team_id: int, team_name: str):
        if team_id == 0:
            return

        team = self.db.get_team(team_id)
        if team:
            return

        safe_name = _safe_str(team_name) or f"Team {team_id}"
        self.db.upsert_team(team_id, safe_name, ranking=9999 if team_id < 0 else 0)

    def _is_within_upcoming_window(self, date_value: datetime) -> bool:
        now = datetime.utcnow() - timedelta(hours=1)
        latest = datetime.utcnow() + timedelta(days=self.upcoming_days_ahead)
        return now <= date_value <= latest

    # ============================================================
    # PandaScore History (train data)
    # ============================================================

    def _new_history_report(self, mode: str = "incremental") -> dict:
        return {
            "mode": mode,
            "returned": 0,
            "saved": 0,
            "discarded": 0,
            "requests_used": 0,
            "windows_processed": 0,
            "cursor": "",
            "error": "",
            "reasons": {key: 0 for key in HISTORY_SKIP_REASON_KEYS},
        }

    def _log_history_report(self, report: dict):
        reasons = report.get("reasons", {})
        logger.info(
            "[HISTORY][pandascore] mode=%s requests=%s janelas=%s retornadas=%s salvas=%s descartadas=%s "
            "(payload_invalido=%s, sem_data=%s, sem_time=%s, team_unresolved=%s, sem_score=%s, "
            "duplicada=%s, status_invalido=%s, fora_janela=%s, api_error=%s, quota=%s) cursor=%s",
            _safe_str(report.get("mode")),
            _safe_int(report.get("requests_used")),
            _safe_int(report.get("windows_processed")),
            _safe_int(report.get("returned")),
            _safe_int(report.get("saved")),
            _safe_int(report.get("discarded")),
            _safe_int(reasons.get("payload_invalido")),
            _safe_int(reasons.get("sem_data")),
            _safe_int(reasons.get("sem_time")),
            _safe_int(reasons.get("team_unresolved")),
            _safe_int(reasons.get("sem_score")),
            _safe_int(reasons.get("duplicada")),
            _safe_int(reasons.get("status_invalido")),
            _safe_int(reasons.get("fora_janela")),
            _safe_int(reasons.get("api_error")),
            _safe_int(reasons.get("quota")),
            _safe_str(report.get("cursor")),
        )
        if report.get("error"):
            logger.warning("[HISTORY][pandascore] erro=%s", report["error"])

    def sync_pandascore_history(self, bootstrap: bool = False, force_full: bool = False) -> dict:
        mode = "bootstrap" if bootstrap else "incremental"
        report = self._new_history_report(mode=mode)

        if not self.pandascore_history_enabled:
            report["error"] = "disabled"
            return report

        token = _safe_str(os.getenv(self.pandascore_token_env)) or self._pandascore_token
        self._pandascore_token = token
        if not token:
            report["error"] = f"token_ausente:{self.pandascore_token_env}"
            return report

        now_utc = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
        floor_dt = self._history_floor_datetime(now_utc)
        cursor = self._load_history_cursor(force_reset=force_full, now_utc=now_utc)
        windows: list[tuple[datetime, datetime]] = []

        if force_full:
            cursor["completed"] = False
            cursor["next_end"] = now_utc.isoformat(timespec="seconds")

        if bool(cursor.get("completed")) and not bootstrap and not force_full:
            windows.append((now_utc - timedelta(days=self.pandascore_history_window_days), now_utc))
        else:
            next_window = self._next_history_window(cursor, floor_dt)
            if not next_window:
                cursor["completed"] = True
            elif not bootstrap:
                windows.append(next_window)
            else:
                # Bootstrap one-shot: monta janelas retroativas sem repetir cursor.
                # Limite de seguranca para evitar loops longos em caso de dados inesperados.
                max_windows = max(8, int((self.pandascore_history_months * 30) / self.pandascore_history_window_days) + 4)
                temp_end = next_window[1]
                for _ in range(max_windows):
                    start_dt = temp_end - timedelta(days=self.pandascore_history_window_days)
                    if start_dt < floor_dt:
                        start_dt = floor_dt
                    if start_dt >= temp_end:
                        cursor["completed"] = True
                        break
                    windows.append((start_dt, temp_end))
                    temp_end = start_dt
                    if start_dt <= floor_dt:
                        cursor["completed"] = True
                        break

        if windows:
            logger.info(
                "[HISTORY][pandascore] processando %s janela(s) (bootstrap=%s)",
                len(windows),
                bootstrap,
            )
        for window_start, window_end in windows:
            report["windows_processed"] += 1
            seen_match_ids: set[int] = set()
            page = 1
            stop_window = False
            logger.info(
                "[HISTORY][pandascore] janela %s/%s: %s -> %s",
                report["windows_processed"],
                len(windows),
                window_start.isoformat(timespec="seconds"),
                window_end.isoformat(timespec="seconds"),
            )

            while True:
                page_items, page_error, requests_used = self._fetch_pandascore_history_page(
                    page=page,
                    token=token,
                    start_dt=window_start,
                    end_dt=window_end,
                )
                report["requests_used"] += requests_used

                if page_error:
                    if page_error == "quota_reached":
                        report["reasons"]["quota"] += 1
                    else:
                        report["reasons"]["api_error"] += 1
                    report["error"] = page_error
                    stop_window = True
                    break

                if not page_items:
                    break

                report["returned"] += len(page_items)
                self._persist_pandascore_history_matches(
                    matches=page_items,
                    report=report,
                    seen_match_ids=seen_match_ids,
                    window_start=window_start,
                    window_end=window_end,
                )

                if len(page_items) < self.pandascore_history_per_page:
                    break
                if page % 5 == 0:
                    logger.info(
                        "[HISTORY][pandascore] progresso janela %s: pagina=%s retornadas=%s salvas=%s requests=%s",
                        report["windows_processed"],
                        page,
                        report["returned"],
                        report["saved"],
                        report["requests_used"],
                    )
                page += 1

            cursor["next_end"] = window_start.isoformat(timespec="seconds")
            cursor["updated_at"] = (
                datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0).isoformat(timespec="seconds")
            )
            if window_start <= floor_dt:
                cursor["completed"] = True
            self._save_history_cursor(cursor)
            report["cursor"] = _safe_str(cursor.get("next_end"))

            if stop_window:
                break
            if not bootstrap:
                break
            # Em bootstrap processamos todas as janelas montadas previamente.
            # `cursor.completed` pode estar True desde a montagem e não deve encerrar aqui.
            if (not bootstrap) and bool(cursor.get("completed")):
                break

        report["discarded"] = max(_safe_int(report["returned"]) - _safe_int(report["saved"]), 0)
        self._log_history_report(report)
        return report

    def _history_floor_datetime(self, now_utc: datetime) -> datetime:
        return now_utc - timedelta(days=self.pandascore_history_months * 30)

    def _default_history_cursor(self, now_utc: datetime) -> dict:
        return {
            "version": 1,
            "months": int(self.pandascore_history_months),
            "next_end": now_utc.isoformat(timespec="seconds"),
            "completed": False,
            "updated_at": now_utc.isoformat(timespec="seconds"),
        }

    def _load_history_cursor(self, force_reset: bool, now_utc: datetime) -> dict:
        default_cursor = self._default_history_cursor(now_utc)
        if force_reset:
            return default_cursor

        raw = self.db.get_state(self.pandascore_history_checkpoint_key, "")
        if not raw:
            return default_cursor

        try:
            data = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return default_cursor

        if not isinstance(data, dict):
            return default_cursor

        if _safe_int(data.get("months")) != int(self.pandascore_history_months):
            return default_cursor

        out = dict(default_cursor)
        out.update(data)
        return out

    def _save_history_cursor(self, cursor: dict):
        payload = {
            "version": 1,
            "months": int(self.pandascore_history_months),
            "next_end": _safe_str(cursor.get("next_end")),
            "completed": bool(cursor.get("completed", False)),
            "updated_at": _safe_str(cursor.get("updated_at")),
        }
        self.db.set_state(self.pandascore_history_checkpoint_key, json.dumps(payload))

    def _next_history_window(self, cursor: dict, floor_dt: datetime) -> tuple[datetime, datetime] | None:
        end_dt = _parse_datetime(cursor.get("next_end")) or datetime.now(timezone.utc).replace(
            tzinfo=None, microsecond=0
        )
        if end_dt <= floor_dt:
            return None

        start_dt = end_dt - timedelta(days=self.pandascore_history_window_days)
        if start_dt < floor_dt:
            start_dt = floor_dt
        if start_dt >= end_dt:
            return None
        return start_dt, end_dt

    def _history_quota_state_key(self) -> str:
        hour_bucket = datetime.now(timezone.utc).strftime("%Y%m%d%H")
        return f"{self.pandascore_history_checkpoint_key}:quota:{hour_bucket}"

    def _consume_history_quota(self) -> bool:
        key = self._history_quota_state_key()
        used = _safe_int(self.db.get_state(key, "0"))
        if used >= self.pandascore_history_max_requests_per_hour:
            return False
        self.db.set_state(key, str(used + 1))
        return True

    def _fetch_pandascore_history_page(
        self,
        page: int,
        token: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> tuple[list[dict], str, int]:
        start_iso = start_dt.isoformat(timespec="seconds") + "Z"
        end_iso = end_dt.isoformat(timespec="seconds") + "Z"
        routes = [
            (
                f"{self.pandascore_base_url}/csgo/matches/past",
                {
                    "page": page,
                    "per_page": self.pandascore_history_per_page,
                    "sort": "-end_at",
                    "range[end_at]": f"{start_iso},{end_iso}",
                },
            ),
            (
                f"{self.pandascore_base_url}/csgo/matches",
                {
                    "page": page,
                    "per_page": self.pandascore_history_per_page,
                    "sort": "-end_at",
                    "filter[status]": "finished",
                    "range[end_at]": f"{start_iso},{end_iso}",
                },
            ),
            (
                f"{self.pandascore_base_url}/csgo/matches",
                {
                    "page": page,
                    "per_page": self.pandascore_history_per_page,
                    "sort": "-begin_at",
                    "filter[status]": "finished",
                    "range[begin_at]": f"{start_iso},{end_iso}",
                },
            ),
        ]

        total_requests = 0
        for url, params in routes:
            payload, error, used = self._request_pandascore_json_page(url, params, token)
            total_requests += used
            if error == "http_404":
                continue
            if error:
                return [], error, total_requests
            if isinstance(payload, dict) and isinstance(payload.get("data"), list):
                payload = payload.get("data")
            if not isinstance(payload, list):
                return [], "payload_invalido", total_requests
            return payload, "", total_requests

        return [], "http_404", total_requests

    def _request_pandascore_json_page(
        self,
        url: str,
        params: dict,
        token: str,
    ) -> tuple[list | dict | None, str, int]:
        headers = {
            "accept": "application/json",
            "authorization": f"Bearer {token}",
            "user-agent": self.user_agent,
        }
        max_attempts = self.upcoming_retry_count + 1
        requests_used = 0

        for attempt in range(max_attempts):
            if not self._consume_history_quota():
                return None, "quota_reached", requests_used
            requests_used += 1

            try:
                resp = requests.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=self.pandascore_timeout_sec,
                )
            except requests.RequestException as exc:
                if attempt >= max_attempts - 1:
                    return None, f"request_error:{exc.__class__.__name__}", requests_used
                time.sleep(self.upcoming_retry_backoff_sec * (attempt + 1))
                continue

            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt >= max_attempts - 1:
                    return None, f"http_{resp.status_code}", requests_used
                time.sleep(self.upcoming_retry_backoff_sec * (attempt + 1))
                continue
            if resp.status_code == 404:
                return None, "http_404", requests_used
            if resp.status_code >= 400:
                return None, f"http_{resp.status_code}", requests_used

            try:
                payload = resp.json()
            except ValueError:
                return None, "payload_invalido", requests_used

            return payload, "", requests_used

        return None, "retry_exceeded", requests_used

    def _persist_pandascore_history_matches(
        self,
        matches: list[dict],
        report: dict,
        seen_match_ids: set[int],
        window_start: datetime,
        window_end: datetime,
    ):
        reasons = report.get("reasons", {})
        for match in matches:
            normalized = self._normalize_pandascore_completed_payload(match)
            if not normalized:
                reasons["payload_invalido"] += 1
                continue

            status = _safe_str(normalized.get("status")).lower()
            if status and status not in {"finished", "completed"}:
                reasons["status_invalido"] += 1
                continue

            match_id = _safe_int(normalized.get("match_id"))
            if match_id == 0:
                reasons["payload_invalido"] += 1
                continue
            if match_id in seen_match_ids:
                reasons["duplicada"] += 1
                continue
            seen_match_ids.add(match_id)

            match_date = _safe_str(normalized.get("date"))
            parsed_date = _parse_datetime(match_date)
            if not match_date or parsed_date is None:
                reasons["sem_data"] += 1
                continue
            if parsed_date < window_start - timedelta(hours=6) or parsed_date > window_end + timedelta(hours=6):
                reasons["fora_janela"] += 1
                continue

            team1_name = _safe_str(normalized.get("team1_name"))
            team2_name = _safe_str(normalized.get("team2_name"))
            if not team1_name or not team2_name:
                reasons["sem_time"] += 1
                continue

            source_t1 = _safe_int(normalized.get("team1_source_id"))
            source_t2 = _safe_int(normalized.get("team2_source_id"))
            team1_id = self._resolve_team_id(source_t1, team1_name, provider="pandascore")
            team2_id = self._resolve_team_id(source_t2, team2_name, provider="pandascore")
            if team1_id == 0 or team2_id == 0 or team1_id == team2_id:
                reasons["team_unresolved"] += 1
                continue

            self._ensure_team_exists(team1_id, team1_name)
            self._ensure_team_exists(team2_id, team2_name)

            team1_score = normalized.get("team1_score")
            team2_score = normalized.get("team2_score")
            winner_source_id = _safe_int(normalized.get("winner_source_id"))
            winner_id = 0
            if winner_source_id > 0:
                if winner_source_id == source_t1:
                    winner_id = team1_id
                elif winner_source_id == source_t2:
                    winner_id = team2_id

            if winner_id == 0 and team1_score is not None and team2_score is not None:
                if team1_score > team2_score:
                    winner_id = team1_id
                elif team2_score > team1_score:
                    winner_id = team2_id

            if winner_id == 0:
                reasons["sem_score"] += 1
                continue

            self.db.upsert_match(
                hltv_id=match_id,
                date=match_date,
                event_name=_safe_str(normalized.get("event_name")),
                best_of=_parse_best_of(normalized.get("best_of", 1)),
                team1_id=team1_id,
                team2_id=team2_id,
                team1_score=team1_score,
                team2_score=team2_score,
                winner_id=winner_id,
                status="completed",
            )
            report["saved"] = _safe_int(report.get("saved")) + 1

    def _normalize_pandascore_completed_payload(self, match: dict) -> dict | None:
        if not isinstance(match, dict):
            return None

        provider_match_id = _safe_int(match.get("id"))
        if provider_match_id <= 0:
            return None

        teams = _extract_pandascore_teams(match)
        if len(teams) < 2:
            return None

        source_t1 = _safe_int(teams[0].get("id"))
        source_t2 = _safe_int(teams[1].get("id"))
        team1_score, team2_score = _extract_pandascore_scores(match, source_t1, source_t2)

        raw_date = (
            match.get("end_at")
            or match.get("finished_at")
            or match.get("begin_at")
            or match.get("scheduled_at")
        )
        event_name = _safe_str(
            match.get("tournament", {}).get("name")
            or match.get("serie", {}).get("name")
            or match.get("league", {}).get("name")
            or match.get("videogame_title")
        )

        return {
            "match_id": _synthetic_provider_match_id(provider_match_id),
            "provider_match_id": provider_match_id,
            "status": _safe_str(match.get("status")),
            "date": _normalize_match_datetime(raw_date),
            "event_name": event_name,
            "best_of": _parse_best_of(match.get("number_of_games", match.get("best_of", 1))),
            "team1_source_id": source_t1,
            "team2_source_id": source_t2,
            "team1_name": _safe_str(teams[0].get("name")),
            "team2_name": _safe_str(teams[1].get("name")),
            "team1_score": team1_score,
            "team2_score": team2_score,
            "winner_source_id": _safe_int(match.get("winner_id")),
        }

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

        teams = [t for t in self.db.get_all_teams() if _safe_int(t.get("id")) > 0][:20]
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


def _synthetic_provider_team_id(provider_team_id: int) -> int:
    if provider_team_id <= 0:
        return 0
    return -(SYNTHETIC_TEAM_ID_BASE + provider_team_id)


def _synthetic_provider_match_id(provider_match_id: int) -> int:
    if provider_match_id <= 0:
        return 0
    return -(SYNTHETIC_MATCH_ID_BASE + provider_match_id)


def _extract_pandascore_teams(match_data: dict) -> list[dict]:
    teams = []
    opponents = match_data.get("opponents")
    if isinstance(opponents, list):
        for item in opponents:
            if not isinstance(item, dict):
                continue
            opponent = item.get("opponent", item)
            if not isinstance(opponent, dict):
                continue
            name = _safe_str(opponent.get("name", opponent.get("acronym", "")))
            if not name:
                continue
            teams.append({"id": _safe_int(opponent.get("id")), "name": name})

    if len(teams) >= 2:
        return teams[:2]

    fallback = match_data.get("teams")
    if isinstance(fallback, list):
        for item in fallback:
            if not isinstance(item, dict):
                continue
            name = _safe_str(item.get("name", item.get("acronym", "")))
            if not name:
                continue
            teams.append({"id": _safe_int(item.get("id")), "name": name})
            if len(teams) >= 2:
                break

    return teams[:2]


def _extract_pandascore_scores(
    match_data: dict,
    team1_source_id: int,
    team2_source_id: int,
) -> tuple[int | None, int | None]:
    results = match_data.get("results")
    if isinstance(results, list):
        by_team: dict[int, int] = {}
        ordered_scores: list[int] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            score = _safe_score(item.get("score"))
            team_id = _safe_int(item.get("team_id", item.get("teamId", 0)))
            if score is None:
                continue
            ordered_scores.append(score)
            if team_id > 0:
                by_team[team_id] = score

        if team1_source_id > 0 and team2_source_id > 0:
            if team1_source_id in by_team and team2_source_id in by_team:
                return by_team[team1_source_id], by_team[team2_source_id]
        if len(ordered_scores) >= 2:
            return ordered_scores[0], ordered_scores[1]

    # Fallbacks para payloads alternativos.
    for s1_key, s2_key in (
        ("team1_score", "team2_score"),
        ("score1", "score2"),
        ("home_score", "away_score"),
    ):
        s1 = _safe_score(match_data.get(s1_key))
        s2 = _safe_score(match_data.get(s2_key))
        if s1 is not None and s2 is not None:
            return s1, s2

    return None, None


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


def _parse_datetime(value) -> datetime | None:
    text = _safe_str(value)
    if not text:
        return None

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo:
            parsed = parsed.replace(tzinfo=None)
        return parsed
    except ValueError:
        pass

    for date_fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d-%m-%Y %H:%M:%S", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, date_fmt)
        except ValueError:
            continue

    return None


def _parse_time(time_text: str) -> datetime | None:
    if not time_text:
        return None

    for time_fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(time_text, time_fmt)
        except ValueError:
            continue

    return None

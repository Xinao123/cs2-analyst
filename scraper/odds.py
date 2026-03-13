"""
Odds feed sync via OddsPapi.

Pipeline:
1. Resolve sport id (auto-discovery or config override)
2. Fetch upcoming fixtures in the configured window
3. Match fixture -> local upcoming match by teams + datetime window
4. Fetch odds for fixture and extract best h2h line from whitelisted bookmakers
5. Persist snapshots + latest odds per local match
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
import unicodedata
from datetime import datetime, timedelta, timezone

import requests

from db.models import Database

logger = logging.getLogger(__name__)

ODDS_SKIP_REASON_KEYS = (
    "sem_match_local",
    "sem_odds",
    "fora_janela",
    "bookmaker_filtrado",
    "payload_invalido",
    "api_error",
)


class OddsPapiSync:
    """Synchronize real odds from OddsPapi into local DB."""

    def __init__(self, db: Database, config: dict):
        self.db = db
        self.config = config

        odds_cfg = config.get("odds", {})
        scraper_cfg = config.get("scraper", {})

        self.enabled = bool(odds_cfg.get("enabled", True))
        self.provider = _safe_str(odds_cfg.get("provider", "oddspapi")).lower() or "oddspapi"
        self.base_url = _safe_str(odds_cfg.get("base_url", "https://api.oddspapi.io/v4")).rstrip("/")
        self.token_env = _safe_str(odds_cfg.get("token_env", "ODDSPAPI_API_KEY")) or "ODDSPAPI_API_KEY"
        self.sport_id = _safe_str(odds_cfg.get("sport_id", ""))
        self.market = _safe_str(odds_cfg.get("market", "h2h")).lower() or "h2h"

        self.refresh_minutes = max(1, _safe_int(odds_cfg.get("refresh_minutes", 10)))
        self.match_window_hours = max(1, _safe_int(odds_cfg.get("match_window_hours", 6)))
        self.timeout_sec = max(5, _safe_int(odds_cfg.get("timeout_sec", 15)))
        self.retry_count = max(0, _safe_int(odds_cfg.get("retry_count", 3)))
        self.retry_backoff_sec = max(1, _safe_int(odds_cfg.get("retry_backoff_sec", 2)))
        self.max_pages = max(1, _safe_int(odds_cfg.get("max_pages", 5)))
        self.per_page = max(20, _safe_int(odds_cfg.get("per_page", 100)))
        self.days_ahead = max(
            1,
            _safe_int(
                odds_cfg.get(
                    "days_ahead",
                    scraper_cfg.get("upcoming_days_ahead", 7),
                )
            ),
        )

        whitelist = odds_cfg.get(
            "bookmaker_whitelist",
            ["bet365", "betano", "pinnacle", "1xbet"],
        )
        if isinstance(whitelist, str):
            whitelist = [part.strip() for part in whitelist.split(",")]
        self.bookmaker_whitelist = {
            _normalize_bookmaker_name(v)
            for v in whitelist
            if _normalize_bookmaker_name(v)
        }

        self.user_agent = _safe_str(
            odds_cfg.get(
                "user_agent",
                scraper_cfg.get("user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"),
            )
        )

        self._resolved_sport_id = self.sport_id
        self._api_base_url = self.base_url

    @property
    def refresh_seconds(self) -> int:
        return self.refresh_minutes * 60

    def sync_upcoming_odds(self) -> dict:
        report = self._new_report()

        if not self.enabled:
            report["error"] = "disabled"
            return report
        if self.provider != "oddspapi":
            report["error"] = f"provider_nao_suportado:{self.provider}"
            logger.warning("[ODDS] Provider nao suportado: %s", self.provider)
            return report

        token = _safe_str(os.getenv(self.token_env))
        if not token:
            report["error"] = f"token_ausente:{self.token_env}"
            logger.warning(
                "[ODDS][oddspapi] Token ausente em %s, mantendo modo degradado sem odds reais.",
                self.token_env,
            )
            return report

        local_matches = self.db.get_upcoming_matches()
        if not local_matches:
            report["error"] = "sem_upcoming_local"
            logger.info("[ODDS][oddspapi] Sem partidas futuras locais para casar odds.")
            return report

        sport_id = self._resolve_sport_id(token)
        if not sport_id:
            report["error"] = "sport_nao_resolvido"
            report["reasons"]["api_error"] += 1
            logger.warning("[ODDS][oddspapi] Falha ao resolver sport_id para CS.")
            return report

        fixtures, fixtures_error = self._fetch_fixtures(token, sport_id)
        if fixtures_error:
            report["error"] = fixtures_error
            report["reasons"]["api_error"] += 1
            self._log_report(report)
            return report

        report["returned"] = len(fixtures)
        if not fixtures:
            self._log_report(report)
            return report

        indexed_matches = _index_local_matches(local_matches)
        used_match_ids: set[int] = set()
        payload_invalid_logged = 0

        for fixture in fixtures:
            fixture_date = fixture.get("start_dt")
            if not fixture_date:
                report["reasons"]["payload_invalido"] += 1
                continue

            if not self._is_within_window(fixture_date):
                report["reasons"]["fora_janela"] += 1
                continue

            match_data, swapped = self._match_fixture_to_local(
                fixture,
                indexed_matches,
                used_match_ids,
            )
            if not match_data:
                report["reasons"]["sem_match_local"] += 1
                continue

            fixture_id = fixture["fixture_id"]
            odds_payload, odds_error = self._fetch_fixture_odds(token, sport_id, fixture_id)
            if odds_error:
                report["reasons"]["api_error"] += 1
                continue

            quotes = _extract_bookmaker_quotes(odds_payload, fixture)
            if not quotes:
                report["reasons"]["payload_invalido"] += 1
                if payload_invalid_logged < 3:
                    logger.info(
                        "[ODDS][oddspapi] payload sem quotes fixture=%s shape=%s",
                        fixture_id,
                        _payload_shape_summary(odds_payload),
                    )
                    payload_invalid_logged += 1
                continue

            best = self._select_best_h2h_lines(quotes)
            if best["filtered_count"] > 0:
                report["reasons"]["bookmaker_filtrado"] += best["filtered_count"]
            if not best["home"] or not best["away"]:
                report["reasons"]["sem_odds"] += 1
                continue

            match_id = _safe_int(match_data.get("id"))
            if match_id <= 0:
                report["reasons"]["payload_invalido"] += 1
                continue

            snapshots_saved = self._save_snapshots(
                match_id=match_id,
                fixture_id=fixture_id,
                quotes=best["quotes"],
                swapped=swapped,
            )
            report["snapshots_saved"] += snapshots_saved

            home_best = best["home"]
            away_best = best["away"]
            if swapped:
                team1_odds = away_best["odds"]
                team2_odds = home_best["odds"]
                team1_book = away_best["bookmaker"]
                team2_book = home_best["bookmaker"]
            else:
                team1_odds = home_best["odds"]
                team2_odds = away_best["odds"]
                team1_book = home_best["bookmaker"]
                team2_book = away_best["bookmaker"]

            self.db.upsert_match_odds_latest(
                match_id=match_id,
                provider="oddspapi",
                fixture_id=fixture_id,
                market_key=self.market,
                odds_team1=team1_odds,
                odds_team2=team2_odds,
                bookmaker_team1=team1_book,
                bookmaker_team2=team2_book,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
            report["saved"] += 1
            report["matched"] += 1
            used_match_ids.add(match_id)

        report["skipped"] = max(report["returned"] - report["matched"], 0)
        self._log_report(report)
        return report

    def _new_report(self) -> dict:
        return {
            "returned": 0,
            "matched": 0,
            "saved": 0,
            "snapshots_saved": 0,
            "skipped": 0,
            "reasons": {key: 0 for key in ODDS_SKIP_REASON_KEYS},
            "error": "",
        }

    def _log_report(self, report: dict):
        reasons = report.get("reasons", {})
        logger.info(
            "[ODDS][oddspapi] retornadas=%s casadas=%s salvas=%s snapshots=%s descartadas=%s "
            "(sem_match_local=%s, sem_odds=%s, fora_janela=%s, bookmaker_filtrado=%s, payload_invalido=%s, api_error=%s)",
            _safe_int(report.get("returned")),
            _safe_int(report.get("matched")),
            _safe_int(report.get("saved")),
            _safe_int(report.get("snapshots_saved")),
            _safe_int(report.get("skipped")),
            _safe_int(reasons.get("sem_match_local")),
            _safe_int(reasons.get("sem_odds")),
            _safe_int(reasons.get("fora_janela")),
            _safe_int(reasons.get("bookmaker_filtrado")),
            _safe_int(reasons.get("payload_invalido")),
            _safe_int(reasons.get("api_error")),
        )
        if report.get("error"):
            logger.warning("[ODDS][oddspapi] erro=%s", report["error"])

    def _resolve_sport_id(self, token: str) -> str:
        if self._resolved_sport_id:
            return self._resolved_sport_id

        payload = None
        error = ""
        for path in ("/sports", "/sports/list"):
            payload, error = self._request_json(path, token, {})
            if not error:
                break

        if error:
            logger.warning("[ODDS][oddspapi] Falha ao buscar sports: %s", error)
            return ""

        sports = _coerce_list(payload)
        if not sports:
            return ""

        best_id = ""
        best_score = -1
        for item in sports:
            if not isinstance(item, dict):
                continue
            sid = _safe_str(
                item.get(
                    "sportId",
                    item.get("sport_id", item.get("id", item.get("key", ""))),
                )
            )
            if not sid:
                continue
            name = _safe_str(
                item.get(
                    "sportName",
                    item.get("name", item.get("title", item.get("slug", ""))),
                )
            ).lower()
            slug = _safe_str(item.get("slug", ""))
            score = _score_sport_name(f"{name} {slug}")
            if score > best_score:
                best_score = score
                best_id = sid

        if best_score < 1:
            return ""

        self._resolved_sport_id = best_id
        return best_id

    def _fetch_fixtures(self, token: str, sport_id: str) -> tuple[list[dict], str]:
        now = datetime.now(timezone.utc) - timedelta(hours=1)
        end = now + timedelta(days=self.days_ahead)

        collected = []
        seen_ids: set[str] = set()
        last_error = ""

        for page in range(1, self.max_pages + 1):
            params = {
                "sport": sport_id,
                "sportId": sport_id,
                "status": "upcoming",
                "page": page,
                "perPage": self.per_page,
                "startDate": now.isoformat(),
                "endDate": end.isoformat(),
                "from": now.isoformat(),
                "to": end.isoformat(),
                "startTimeFrom": now.isoformat(),
                "startTimeTo": end.isoformat(),
            }

            payload = None
            error = ""
            for path in ("/fixtures", "/fixtures/upcoming"):
                payload, error = self._request_json(path, token, params)
                if not error:
                    break
                if not error.startswith("http_404"):
                    break

            if error:
                last_error = error
                if page == 1:
                    return [], error
                break

            page_items = _coerce_list(payload)
            if not page_items:
                break

            valid_count = 0
            for item in page_items:
                fixture = _normalize_fixture_payload(item)
                if not fixture:
                    continue
                if fixture["fixture_id"] in seen_ids:
                    continue
                seen_ids.add(fixture["fixture_id"])
                collected.append(fixture)
                valid_count += 1

            if valid_count < self.per_page:
                break

        return collected, last_error

    def _fetch_fixture_odds(self, token: str, sport_id: str, fixture_id: str) -> tuple[dict | list | None, str]:
        base_params = {
            "sport": sport_id,
            "sportId": sport_id,
            "fixture": fixture_id,
            "fixtureId": fixture_id,
            "market": self.market,
            "oddsFormat": "decimal",
            "bookmakers": ",".join(sorted(self.bookmaker_whitelist)),
        }

        routes = [
            ("/odds", base_params),
            ("/odds", dict(base_params, fixtureId=fixture_id)),
            (f"/fixtures/{fixture_id}/odds", dict(base_params, fixture=fixture_id)),
        ]

        last_error = ""
        for path, params in routes:
            payload, error = self._request_json(path, token, params)
            if not error:
                return payload, ""
            last_error = error
            if not error.startswith("http_404"):
                break

        return None, last_error

    def _request_json(self, path: str, token: str, params: dict) -> tuple[dict | list | None, str]:
        headers = {
            "accept": "application/json",
            "user-agent": self.user_agent,
            "x-api-key": token,
            "authorization": f"Bearer {token}",
        }
        query = dict(params)
        query["apiKey"] = token

        last_error = ""
        for base in _candidate_base_urls(self.base_url):
            url = f"{base}/{path.lstrip('/')}"
            self._api_base_url = base

            max_attempts = self.retry_count + 1
            for attempt in range(max_attempts):
                try:
                    resp = requests.get(url, headers=headers, params=query, timeout=self.timeout_sec)
                except requests.RequestException as exc:
                    last_error = f"request_error:{exc.__class__.__name__}"
                    if attempt >= max_attempts - 1:
                        break
                    time.sleep(self.retry_backoff_sec * (attempt + 1))
                    continue

                if resp.status_code in (429, 500, 502, 503, 504):
                    last_error = f"http_{resp.status_code}"
                    if attempt >= max_attempts - 1:
                        break
                    time.sleep(self.retry_backoff_sec * (attempt + 1))
                    continue

                if resp.status_code == 404:
                    last_error = "http_404"
                    break

                if resp.status_code >= 400:
                    return None, f"http_{resp.status_code}"

                try:
                    return resp.json(), ""
                except ValueError:
                    return None, "payload_invalido"

        return None, last_error or "request_failed"

    def _match_fixture_to_local(
        self,
        fixture: dict,
        indexed_matches: dict[tuple[str, str], list[dict]],
        used_match_ids: set[int],
    ) -> tuple[dict | None, bool]:
        home_norm = _normalize_team_name(fixture.get("home_name", ""))
        away_norm = _normalize_team_name(fixture.get("away_name", ""))
        if not home_norm or not away_norm:
            return None, False

        pair_key = tuple(sorted((home_norm, away_norm)))
        candidates = indexed_matches.get(pair_key, [])
        if not candidates:
            return None, False

        best_item = None
        best_score = float("-inf")
        best_swapped = False
        fixture_dt = fixture.get("start_dt")

        for match in candidates:
            match_id = _safe_int(match.get("id"))
            if match_id <= 0 or match_id in used_match_ids:
                continue

            local_t1 = _normalize_team_name(match.get("team1_name", ""))
            local_t2 = _normalize_team_name(match.get("team2_name", ""))
            if not local_t1 or not local_t2:
                continue

            swapped = local_t1 == away_norm and local_t2 == home_norm
            aligned = local_t1 == home_norm and local_t2 == away_norm

            score = 0.0
            if aligned:
                score += 2.0
            elif swapped:
                score += 1.0

            local_dt = _parse_datetime(match.get("date"))
            if fixture_dt and local_dt:
                diff_hours = abs((_ensure_utc(local_dt) - _ensure_utc(fixture_dt)).total_seconds()) / 3600.0
                if diff_hours > self.match_window_hours:
                    continue
                score += max(0.0, 1.5 - (diff_hours / max(self.match_window_hours, 1)))

            score += _event_similarity_score(fixture.get("event_name", ""), match.get("event_name", ""))
            if score > best_score:
                best_score = score
                best_item = match
                best_swapped = swapped

        return best_item, best_swapped

    def _is_within_window(self, dt_value: datetime) -> bool:
        now = datetime.now(timezone.utc) - timedelta(hours=1)
        latest = now + timedelta(days=self.days_ahead)
        dt_utc = _ensure_utc(dt_value)
        return now <= dt_utc <= latest

    def _select_best_h2h_lines(self, quotes: list[dict]) -> dict:
        filtered_count = 0
        usable = []
        for quote in quotes:
            side = quote.get("side")
            odds = _safe_float(quote.get("odds"))
            bookmaker = _safe_str(quote.get("bookmaker"))
            if side not in ("home", "away") or odds <= 1.0 or not bookmaker:
                continue

            book_norm = _normalize_bookmaker_name(bookmaker)
            if self.bookmaker_whitelist and book_norm not in self.bookmaker_whitelist:
                filtered_count += 1
                continue

            usable.append(
                {
                    "bookmaker": bookmaker,
                    "bookmaker_norm": book_norm,
                    "side": side,
                    "odds": odds,
                    "market_key": _safe_str(quote.get("market_key", self.market)) or self.market,
                    "changed_at": _safe_str(quote.get("changed_at", "")),
                }
            )

        best_home = None
        best_away = None
        for quote in usable:
            if quote["side"] == "home":
                if not best_home or quote["odds"] > best_home["odds"]:
                    best_home = quote
            elif quote["side"] == "away":
                if not best_away or quote["odds"] > best_away["odds"]:
                    best_away = quote

        return {
            "home": best_home,
            "away": best_away,
            "quotes": usable,
            "filtered_count": filtered_count,
        }

    def _save_snapshots(self, match_id: int, fixture_id: str, quotes: list[dict], swapped: bool) -> int:
        saved = 0
        collected_at = datetime.now(timezone.utc).isoformat()
        for quote in quotes:
            side_home_away = quote.get("side")
            if side_home_away not in ("home", "away"):
                continue

            side = side_home_away
            if swapped:
                side = "away" if side_home_away == "home" else "home"
            local_side = "team1" if side == "home" else "team2"

            fingerprint = _snapshot_fingerprint(
                match_id=match_id,
                provider="oddspapi",
                fixture_id=fixture_id,
                bookmaker=_safe_str(quote.get("bookmaker")),
                market_key=_safe_str(quote.get("market_key", self.market)) or self.market,
                side=local_side,
                odds=_safe_float(quote.get("odds")),
                changed_at=_safe_str(quote.get("changed_at", "")),
            )

            self.db.insert_odds_snapshot(
                match_id=match_id,
                provider="oddspapi",
                fixture_id=fixture_id,
                bookmaker=_safe_str(quote.get("bookmaker")),
                market_key=_safe_str(quote.get("market_key", self.market)) or self.market,
                side=local_side,
                odds=_safe_float(quote.get("odds")),
                odds_changed_at=_safe_str(quote.get("changed_at", "")),
                collected_at=collected_at,
                fingerprint=fingerprint,
            )
            saved += 1
        return saved


def _index_local_matches(matches: list[dict]) -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = {}
    for match in matches:
        t1 = _normalize_team_name(match.get("team1_name", ""))
        t2 = _normalize_team_name(match.get("team2_name", ""))
        if not t1 or not t2:
            continue
        key = tuple(sorted((t1, t2)))
        out.setdefault(key, []).append(match)
    return out


def _extract_bookmaker_quotes(payload: dict | list | None, fixture: dict) -> list[dict]:
    if payload is None:
        return []

    blocks = _extract_root_blocks(payload)
    if not blocks and isinstance(payload, dict):
        blocks = [payload]
    if not blocks and isinstance(payload, list):
        blocks = payload

    quotes: list[dict] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue

        bookmaker = _extract_bookmaker_name(block)
        markets = _extract_markets(block)
        changed_at = _safe_str(
            block.get("updatedAt", block.get("updated_at", block.get("lastUpdate", "")))
        )

        if not markets:
            markets = [block]

        for market in markets:
            if not isinstance(market, dict):
                continue

            market_key = _market_key(market, block)
            if not _is_h2h_market(market_key, market):
                continue

            for outcome in _extract_outcomes(market):
                side = _resolve_outcome_side(outcome, fixture)
                if side == "draw":
                    continue
                odds_val = _extract_outcome_price(outcome)
                if side in ("home", "away") and odds_val > 1.0 and bookmaker:
                    quotes.append(
                        {
                            "bookmaker": bookmaker,
                            "side": side,
                            "odds": odds_val,
                            "market_key": market_key or "h2h",
                            "changed_at": _safe_str(
                                outcome.get(
                                    "updatedAt",
                                    outcome.get("updated_at", changed_at),
                                )
                            ),
                        }
                    )

            home_direct = _safe_float(
                market.get("homeOdds", market.get("home_odds", market.get("odds_home", 0)))
            )
            away_direct = _safe_float(
                market.get("awayOdds", market.get("away_odds", market.get("odds_away", 0)))
            )
            if bookmaker and home_direct > 1.0 and away_direct > 1.0:
                quotes.append(
                    {
                        "bookmaker": bookmaker,
                        "side": "home",
                        "odds": home_direct,
                        "market_key": market_key or "h2h",
                        "changed_at": changed_at,
                    }
                )
                quotes.append(
                    {
                        "bookmaker": bookmaker,
                        "side": "away",
                        "odds": away_direct,
                        "market_key": market_key or "h2h",
                        "changed_at": changed_at,
                    }
                )

    return quotes


def _extract_root_blocks(payload: dict | list) -> list:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []

    bookmaker_odds = payload.get("bookmakerOdds")
    if isinstance(bookmaker_odds, dict):
        out = []
        for bookmaker_name, markets in bookmaker_odds.items():
            market_block = markets
            if isinstance(market_block, dict) and isinstance(market_block.get("markets"), dict):
                market_block = market_block.get("markets")
            out.append({"bookmakerName": _safe_str(bookmaker_name), "markets": market_block})
        if out:
            return out

    for key in ("data", "results", "items", "bookmakerOdds", "bookmakers", "odds"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def _extract_markets(block: dict) -> list:
    for key in ("markets", "marketOdds", "market_odds"):
        value = block.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            if isinstance(value.get("markets"), dict):
                value = value.get("markets")
            # Dict payload (OddsPapi v4): {"h2h": {...}} or {"101": {...}}
            if any(isinstance(v, dict) for v in value.values()):
                return [{"marketKey": str(k), **(v if isinstance(v, dict) else {"odds": v})} for k, v in value.items()]
            return [value]

    odds_value = block.get("odds")
    if isinstance(odds_value, list):
        if odds_value and isinstance(odds_value[0], dict) and "odds" in odds_value[0]:
            return odds_value
        return [{"id": block.get("market", block.get("marketId", "")), "odds": odds_value}]

    return []


def _extract_outcomes(market: dict) -> list[dict]:
    for key in ("odds", "outcomes", "prices", "selections"):
        value = market.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            # Map payload: {"Team A": 1.85, "Team B": 2.05, "Draw": 3.2}
            if all(not isinstance(v, dict) for v in value.values()):
                outcomes = []
                for name, odd in value.items():
                    outcomes.append({"name": str(name), "odds": odd})
                return outcomes
            # Nested outcomes map payload: {"10166": {...}, "10167": {...}}
            if all(isinstance(v, dict) for v in value.values()):
                return [item for item in value.values() if isinstance(item, dict)]
            return [value]
    # Flat map payload: {"marketKey": "h2h", "Team A": 1.9, "Team B": 2.0}
    outcomes = []
    for key, value in market.items():
        if key in {"key", "marketKey", "name", "marketName", "id", "marketId"}:
            continue
        odd = _safe_float(value)
        if odd > 1.0:
            outcomes.append({"name": str(key), "odds": odd})
    return outcomes


def _extract_bookmaker_name(block: dict) -> str:
    direct = _safe_str(
        block.get(
            "bookmakerName",
            block.get("bookmaker_name", block.get("sportsbook", block.get("bookmaker", ""))),
        )
    )
    if direct and not isinstance(block.get("bookmaker"), dict):
        return direct

    nested = block.get("bookmaker")
    if isinstance(nested, dict):
        return _safe_str(
            nested.get("name", nested.get("title", nested.get("slug", nested.get("key", ""))))
        )

    return direct


def _resolve_outcome_side(outcome: dict, fixture: dict) -> str:
    if _as_bool(outcome.get("isDraw")):
        return "draw"
    if _as_bool(outcome.get("isHome")):
        return "home"
    if _as_bool(outcome.get("isAway")):
        return "away"

    outcome_name = _normalize_team_name(
        outcome.get(
            "outcomeName",
            outcome.get("name", outcome.get("label", outcome.get("selection", ""))),
        )
    )

    token = _normalize_team_name(
        outcome.get(
            "bookmakerOutcomeId",
            outcome.get(
                "outcome",
                outcome.get(
                    "outcomeId",
                    outcome.get(
                        "name",
                        outcome.get("label", outcome.get("key", outcome.get("selection", ""))),
                    ),
                ),
            ),
        )
    )

    if token in {"home", "team1", "1", "101"} or outcome_name in {"home", "team1", "1", "101"}:
        return "home"
    if token in {"away", "team2", "2", "103"} or outcome_name in {"away", "team2", "2", "103"}:
        return "away"
    if token in {"draw", "x", "102"} or outcome_name in {"draw", "x", "102"}:
        return "draw"

    home_norm = _normalize_team_name(fixture.get("home_name", ""))
    away_norm = _normalize_team_name(fixture.get("away_name", ""))
    if home_norm and token == home_norm:
        return "home"
    if away_norm and token == away_norm:
        return "away"
    if home_norm and outcome_name == home_norm:
        return "home"
    if away_norm and outcome_name == away_norm:
        return "away"

    competitor = outcome.get("competitor", outcome.get("team"))
    if isinstance(competitor, dict):
        comp_name = _normalize_team_name(competitor.get("name", competitor.get("title", "")))
        if home_norm and comp_name == home_norm:
            return "home"
        if away_norm and comp_name == away_norm:
            return "away"

    players = outcome.get("players")
    if isinstance(players, dict):
        for _, player in players.items():
            if not isinstance(player, dict):
                continue
            player_name = _normalize_team_name(
                player.get("name", player.get("outcomeName", player.get("label", "")))
            )
            player_token = _normalize_team_name(player.get("bookmakerOutcomeId", ""))
            if home_norm and (player_name == home_norm or player_token in {"home", "team1", "101"}):
                return "home"
            if away_norm and (player_name == away_norm or player_token in {"away", "team2", "103"}):
                return "away"

    return ""


def _market_key(market: dict, fallback_block: dict) -> str:
    key = _safe_str(
        market.get(
            "key",
            market.get(
                "marketKey",
                market.get(
                    "name",
                    market.get(
                        "marketName",
                        market.get("id", market.get("marketId", fallback_block.get("market", ""))),
                    ),
                ),
            ),
        )
    )
    return key.lower()


def _extract_outcome_price(outcome: dict) -> float:
    direct = _safe_float(
        outcome.get(
            "odds",
            outcome.get("price", outcome.get("decimalOdds", outcome.get("value", 0))),
        )
    )
    if direct > 1.0:
        return direct

    players = outcome.get("players")
    if isinstance(players, dict):
        for _, player in players.items():
            if not isinstance(player, dict):
                continue
            price = _safe_float(
                player.get(
                    "price",
                    player.get("odds", player.get("decimalOdds", player.get("value", 0))),
                )
            )
            if price > 1.0:
                return price

    return 0.0


def _is_h2h_market(market_key: str, market: dict) -> bool:
    key = _safe_str(market_key).lower()
    if key in {"h2h", "moneyline", "match winner", "winner"}:
        return True
    if key in {"101", "1x2", "matchwinner"}:
        return True

    market_name = _safe_str(market.get("name", market.get("marketName", ""))).lower()
    if any(token in market_name for token in ("moneyline", "match winner", "winner")):
        return True
    if market_name in {"1x2"}:
        return True

    return False


def _normalize_fixture_payload(item: dict) -> dict | None:
    if not isinstance(item, dict):
        return None

    fixture_id = _safe_str(
        item.get(
            "fixtureId",
            item.get("fixture_id", item.get("id", item.get("key", ""))),
        )
    )
    if not fixture_id:
        return None

    home_name = _extract_team_name(item, "home")
    away_name = _extract_team_name(item, "away")
    home_name = home_name or _safe_str(item.get("participant1Name", item.get("team1Name", "")))
    away_name = away_name or _safe_str(item.get("participant2Name", item.get("team2Name", "")))

    if not home_name or not away_name:
        participants = item.get("participants", item.get("teams", item.get("opponents", [])))
        if isinstance(participants, list) and len(participants) >= 2:
            home_name = home_name or _extract_name_from_participant(participants[0])
            away_name = away_name or _extract_name_from_participant(participants[1])

    if not home_name or not away_name:
        return None

    start_value = (
        item.get("startTime")
        or item.get("start_date")
        or item.get("startDate")
        or item.get("starts_at")
        or item.get("scheduled_at")
        or item.get("commence_time")
        or item.get("date")
        or item.get("begin_at")
    )
    start_dt = _parse_datetime(start_value)
    if not start_dt:
        return None

    event_name = _safe_str(
        item.get("competitionName")
        or item.get("competition_name")
        or item.get("league_name")
        or item.get("tournamentName")
        or item.get("tournament_name")
        or item.get("event_name")
        or item.get("tournament")
        or item.get("league")
    )
    if isinstance(item.get("tournament"), dict):
        event_name = _safe_str(item["tournament"].get("name", event_name))
    if isinstance(item.get("league"), dict):
        event_name = _safe_str(item["league"].get("name", event_name))

    return {
        "fixture_id": fixture_id,
        "home_name": home_name,
        "away_name": away_name,
        "start_dt": _ensure_utc(start_dt),
        "event_name": event_name,
    }


def _extract_team_name(payload: dict, side: str) -> str:
    side = side.lower()
    if side == "home":
        side_alt = ("participant1Name", "team1Name", "participant1")
    else:
        side_alt = ("participant2Name", "team2Name", "participant2")
    keys = (
        f"{side}_name",
        f"{side}Name",
        side,
        f"{side}_team",
        f"{side}Team",
        f"{side}Competitor",
        f"{side}TeamName",
        *side_alt,
    )
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            name = _safe_str(value.get("name", value.get("title", value.get("acronym", ""))))
            if name:
                return name
        else:
            name = _safe_str(value)
            if name:
                return name
    return ""


def _extract_name_from_participant(item) -> str:
    if isinstance(item, dict):
        opponent = item.get("opponent", item)
        if isinstance(opponent, dict):
            return _safe_str(opponent.get("name", opponent.get("title", opponent.get("acronym", ""))))
    return _safe_str(item)


def _coerce_list(payload) -> list:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "results", "items", "fixtures", "sports"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def _score_sport_name(name: str) -> int:
    normalized = _safe_str(name).lower()
    if not normalized:
        return 0

    score = 0
    if "counterstrike" in normalized or "counter-strike" in normalized:
        score += 10
    if "counter" in normalized and "strike" in normalized:
        score += 6
    if "cs2" in normalized:
        score += 6
    if "csgo" in normalized:
        score += 5
    if "counter strike" in normalized:
        score += 5
    if "esport" in normalized:
        score += 1
    return score


def _candidate_base_urls(base_url: str) -> list[str]:
    candidates = [_safe_str(base_url).rstrip("/")]
    if not candidates[0]:
        candidates = ["https://api.oddspapi.io/v4"]

    base = candidates[0]
    if "/v1" in base:
        candidates.append(base.replace("/v1", "/v4"))
        candidates.append(base.replace("/v1", "/v5"))
    if "/v4" in base:
        candidates.append(base.replace("/v4", "/v1"))
        candidates.append(base.replace("/v4", "/v5"))
    if "/v5" in base:
        candidates.append(base.replace("/v5", "/v4"))
        candidates.append(base.replace("/v5", "/v1"))
    if all(token not in base for token in ("/v1", "/v4", "/v5")):
        candidates.append(base + "/v4")
        candidates.append(base + "/v1")

    out = []
    seen = set()
    for value in candidates:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _event_similarity_score(a: str, b: str) -> float:
    na = set(_tokenize_event_name(a))
    nb = set(_tokenize_event_name(b))
    if not na or not nb:
        return 0.0
    inter = len(na.intersection(nb))
    if inter == 0:
        return 0.0
    return inter / max(len(na), len(nb))


def _tokenize_event_name(value: str) -> list[str]:
    text = _safe_str(value).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return [tok for tok in re.split(r"[^a-z0-9]+", text) if tok]


def _normalize_team_name(value) -> str:
    text = _safe_str(value).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]", "", text)


def _normalize_bookmaker_name(value) -> str:
    return _normalize_team_name(value)


def _snapshot_fingerprint(
    match_id: int,
    provider: str,
    fixture_id: str,
    bookmaker: str,
    market_key: str,
    side: str,
    odds: float,
    changed_at: str,
) -> str:
    raw = "|".join(
        [
            str(match_id),
            _safe_str(provider),
            _safe_str(fixture_id),
            _normalize_bookmaker_name(bookmaker),
            _safe_str(market_key).lower(),
            _safe_str(side).lower(),
            f"{float(odds):.4f}",
            _safe_str(changed_at),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _payload_shape_summary(payload) -> str:
    if payload is None:
        return "none"
    if isinstance(payload, list):
        head = payload[0] if payload else {}
        if isinstance(head, dict):
            return f"list[{len(payload)}] keys={list(head.keys())[:8]}"
        return f"list[{len(payload)}] type={type(head).__name__ if payload else 'empty'}"
    if isinstance(payload, dict):
        keys = list(payload.keys())[:10]
        return f"dict keys={keys}"
    return type(payload).__name__


def _parse_datetime(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value

    text = _safe_str(value)
    if not text:
        return None

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed
    except ValueError:
        pass

    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%d-%m-%Y %H:%M:%S",
        "%d-%m-%Y",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = _safe_str(value).lower()
    return text in {"1", "true", "yes", "y"}


def _safe_str(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value) -> float:
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return 0.0

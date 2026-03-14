"""
Database — Schema SQLite e operações CRUD.

Tabelas:
- teams: times com ranking, stats agregados
- players: jogadores com stats individuais
- matches: partidas (resultado + contexto)
- match_maps: resultado por mapa de cada partida
- predictions: predições do modelo (pra backtest)
"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime, date, timedelta
from contextlib import contextmanager

from utils.time_utils import to_storage_utc_datetime

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    ranking INTEGER DEFAULT 0,
    ranking_points INTEGER DEFAULT 0,
    win_rate REAL DEFAULT 0.0,
    maps_played INTEGER DEFAULT 0,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS players (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    team_id INTEGER,
    rating REAL DEFAULT 0.0,
    kd_ratio REAL DEFAULT 0.0,
    kast REAL DEFAULT 0.0,
    adr REAL DEFAULT 0.0,
    impact REAL DEFAULT 0.0,
    opening_kills_ratio REAL DEFAULT 0.0,
    maps_played INTEGER DEFAULT 0,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (team_id) REFERENCES teams(id)
);

CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hltv_id INTEGER UNIQUE,
    date TEXT NOT NULL,
    event_name TEXT,
    event_tier INTEGER DEFAULT 3,
    is_lan INTEGER DEFAULT 0,
    best_of INTEGER DEFAULT 1,
    team1_id INTEGER NOT NULL,
    team2_id INTEGER NOT NULL,
    team1_score INTEGER,
    team2_score INTEGER,
    winner_id INTEGER,
    status TEXT DEFAULT 'upcoming',
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (team1_id) REFERENCES teams(id),
    FOREIGN KEY (team2_id) REFERENCES teams(id)
);

CREATE TABLE IF NOT EXISTS match_maps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    map_name TEXT NOT NULL,
    map_number INTEGER DEFAULT 1,
    team1_rounds INTEGER,
    team2_rounds INTEGER,
    team1_ct_rounds INTEGER DEFAULT 0,
    team1_t_rounds INTEGER DEFAULT 0,
    team2_ct_rounds INTEGER DEFAULT 0,
    team2_t_rounds INTEGER DEFAULT 0,
    winner_id INTEGER,
    FOREIGN KEY (match_id) REFERENCES matches(id)
);

CREATE TABLE IF NOT EXISTS team_map_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id INTEGER NOT NULL,
    map_name TEXT NOT NULL,
    matches_played INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    win_rate REAL DEFAULT 0.0,
    avg_rounds_won REAL DEFAULT 0.0,
    ct_win_rate REAL DEFAULT 0.0,
    t_win_rate REAL DEFAULT 0.0,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (team_id) REFERENCES teams(id),
    UNIQUE(team_id, map_name)
);

CREATE TABLE IF NOT EXISTS head_to_head (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team1_id INTEGER NOT NULL,
    team2_id INTEGER NOT NULL,
    matches_played INTEGER DEFAULT 0,
    team1_wins INTEGER DEFAULT 0,
    team2_wins INTEGER DEFAULT 0,
    last_match_date TEXT,
    FOREIGN KEY (team1_id) REFERENCES teams(id),
    FOREIGN KEY (team2_id) REFERENCES teams(id),
    UNIQUE(team1_id, team2_id)
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    predicted_winner_id INTEGER,
    model_winner_id INTEGER,
    official_pick_winner_id INTEGER,
    pick_source TEXT DEFAULT 'value',
    team1_win_prob REAL,
    team2_win_prob REAL,
    value_pct REAL,
    suggested_bet TEXT,
    suggested_stake REAL,
    odds_team1 REAL,
    odds_team2 REAL,
    actual_winner_id INTEGER,
    profit REAL DEFAULT 0.0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (match_id) REFERENCES matches(id)
);

CREATE TABLE IF NOT EXISTS match_odds_latest (
    match_id INTEGER PRIMARY KEY,
    provider TEXT NOT NULL,
    fixture_id TEXT,
    market_key TEXT DEFAULT 'h2h',
    odds_team1 REAL,
    odds_team2 REAL,
    bookmaker_team1 TEXT,
    bookmaker_team2 TEXT,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (match_id) REFERENCES matches(id)
);

CREATE TABLE IF NOT EXISTS odds_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    provider TEXT NOT NULL,
    fixture_id TEXT,
    bookmaker TEXT NOT NULL,
    market_key TEXT DEFAULT 'h2h',
    side TEXT NOT NULL,
    odds REAL NOT NULL,
    odds_changed_at TEXT,
    collected_at TEXT DEFAULT CURRENT_TIMESTAMP,
    fingerprint TEXT UNIQUE,
    FOREIGN KEY (match_id) REFERENCES matches(id)
);

CREATE TABLE IF NOT EXISTS clv_tracking (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER,
    match_id INTEGER NOT NULL,
    side TEXT NOT NULL,
    open_odds REAL NOT NULL,
    close_odds REAL NOT NULL,
    clv_pct REAL NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id),
    FOREIGN KEY (match_id) REFERENCES matches(id)
);

CREATE TABLE IF NOT EXISTS daily_top5_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date TEXT NOT NULL UNIQUE,
    requested_top INTEGER DEFAULT 5,
    total_candidates INTEGER DEFAULT 0,
    candidates_with_odds INTEGER DEFAULT 0,
    status TEXT DEFAULT 'captured',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    audited_at TEXT
);

CREATE TABLE IF NOT EXISTS daily_top5_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    rank INTEGER NOT NULL,
    match_id INTEGER,
    match_date TEXT,
    event_name TEXT,
    team1_id INTEGER,
    team2_id INTEGER,
    team1_name TEXT,
    team2_name TEXT,
    model_winner_id INTEGER,
    model_winner_name TEXT,
    official_pick_winner_id INTEGER,
    official_pick_winner_name TEXT,
    pick_source TEXT DEFAULT 'value',
    predicted_winner_id INTEGER,
    predicted_winner_name TEXT,
    team1_win_prob REAL,
    team2_win_prob REAL,
    confidence REAL,
    score REAL,
    odds REAL,
    bookmaker TEXT,
    value_pct REAL,
    expected_value REAL,
    outcome_status TEXT DEFAULT 'pending',
    actual_winner_id INTEGER,
    resolved_match_id INTEGER,
    resolved_at TEXT,
    resolution_method TEXT,
    FOREIGN KEY (run_id) REFERENCES daily_top5_runs(id),
    UNIQUE(run_id, rank)
);

CREATE TABLE IF NOT EXISTS app_state (
    state_key TEXT PRIMARY KEY,
    state_value TEXT,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date);
CREATE INDEX IF NOT EXISTS idx_matches_status ON matches(status);
CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(team1_id, team2_id);
CREATE INDEX IF NOT EXISTS idx_players_team ON players(team_id);
CREATE INDEX IF NOT EXISTS idx_match_maps_match ON match_maps(match_id);
CREATE INDEX IF NOT EXISTS idx_odds_latest_updated ON match_odds_latest(updated_at);
CREATE INDEX IF NOT EXISTS idx_odds_snapshots_match ON odds_snapshots(match_id, collected_at);
CREATE INDEX IF NOT EXISTS idx_clv_tracking_match_created ON clv_tracking(match_id, created_at);
CREATE INDEX IF NOT EXISTS idx_daily_top5_runs_date ON daily_top5_runs(run_date);
CREATE INDEX IF NOT EXISTS idx_daily_top5_items_run ON daily_top5_items(run_id, rank);
CREATE INDEX IF NOT EXISTS idx_daily_top5_items_outcome ON daily_top5_items(outcome_status, run_id);
"""


class Database:
    """Gerencia conexão e operações SQLite."""

    def __init__(self, db_path: str = "data/cs2_analyst.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            self._apply_migrations(conn)
            logger.info(f"[DB] Inicializado: {self.db_path}")

    def _apply_migrations(self, conn: sqlite3.Connection):
        # predictions table migrations
        self._ensure_column(conn, "predictions", "model_winner_id", "INTEGER")
        self._ensure_column(conn, "predictions", "official_pick_winner_id", "INTEGER")
        self._ensure_column(conn, "predictions", "pick_source", "TEXT DEFAULT 'value'")

        # daily_top5_items table migrations
        self._ensure_column(conn, "daily_top5_items", "model_winner_id", "INTEGER")
        self._ensure_column(conn, "daily_top5_items", "model_winner_name", "TEXT")
        self._ensure_column(conn, "daily_top5_items", "official_pick_winner_id", "INTEGER")
        self._ensure_column(conn, "daily_top5_items", "official_pick_winner_name", "TEXT")
        self._ensure_column(conn, "daily_top5_items", "pick_source", "TEXT DEFAULT 'value'")

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, column_def: str):
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        existing = {row["name"] for row in rows}
        if column in existing:
            return
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_def}")

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ============================================================
    # Teams
    # ============================================================

    def upsert_team(self, team_id: int, name: str, **kwargs):
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO teams (id, name, ranking, ranking_points, win_rate, maps_played, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     name=excluded.name, ranking=excluded.ranking,
                     ranking_points=excluded.ranking_points, win_rate=excluded.win_rate,
                     maps_played=excluded.maps_played, updated_at=excluded.updated_at""",
                (
                    team_id, name,
                    kwargs.get("ranking", 0),
                    kwargs.get("ranking_points", 0),
                    kwargs.get("win_rate", 0.0),
                    kwargs.get("maps_played", 0),
                    datetime.now().isoformat(),
                ),
            )

    def get_team(self, team_id: int) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM teams WHERE id=?", (team_id,)).fetchone()
            return dict(row) if row else None

    def get_team_by_name(self, name: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM teams WHERE lower(name)=lower(?) LIMIT 1",
                (name.strip(),),
            ).fetchone()
            return dict(row) if row else None

    def get_all_teams(self) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM teams ORDER BY ranking ASC").fetchall()
            return [dict(r) for r in rows]

    # ============================================================
    # Players
    # ============================================================

    def upsert_player(self, player_id: int, name: str, team_id: int, **kwargs):
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO players (id, name, team_id, rating, kd_ratio, kast, adr,
                     impact, opening_kills_ratio, maps_played, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     name=excluded.name, team_id=excluded.team_id, rating=excluded.rating,
                     kd_ratio=excluded.kd_ratio, kast=excluded.kast, adr=excluded.adr,
                     impact=excluded.impact, opening_kills_ratio=excluded.opening_kills_ratio,
                     maps_played=excluded.maps_played, updated_at=excluded.updated_at""",
                (
                    player_id, name, team_id,
                    kwargs.get("rating", 0.0),
                    kwargs.get("kd_ratio", 0.0),
                    kwargs.get("kast", 0.0),
                    kwargs.get("adr", 0.0),
                    kwargs.get("impact", 0.0),
                    kwargs.get("opening_kills_ratio", 0.0),
                    kwargs.get("maps_played", 0),
                    datetime.now().isoformat(),
                ),
            )

    def get_team_players(self, team_id: int) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM players WHERE team_id=? ORDER BY rating DESC", (team_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    # ============================================================
    # Matches
    # ============================================================

    def upsert_match(self, hltv_id: int, **kwargs) -> int:
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO matches (hltv_id, date, event_name, event_tier, is_lan,
                     best_of, team1_id, team2_id, team1_score, team2_score,
                     winner_id, status, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(hltv_id) DO UPDATE SET
                     team1_score=excluded.team1_score, team2_score=excluded.team2_score,
                     winner_id=excluded.winner_id, status=excluded.status,
                     updated_at=excluded.updated_at""",
                (
                    hltv_id,
                    kwargs.get("date", ""),
                    kwargs.get("event_name", ""),
                    kwargs.get("event_tier", 3),
                    kwargs.get("is_lan", 0),
                    kwargs.get("best_of", 1),
                    kwargs.get("team1_id", 0),
                    kwargs.get("team2_id", 0),
                    kwargs.get("team1_score"),
                    kwargs.get("team2_score"),
                    kwargs.get("winner_id"),
                    kwargs.get("status", "upcoming"),
                    datetime.now().isoformat(),
                ),
            )
            row = conn.execute(
                "SELECT id FROM matches WHERE hltv_id=?", (hltv_id,)
            ).fetchone()
            return row["id"] if row else 0

    def get_upcoming_matches(self) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT m.*, t1.name as team1_name, t2.name as team2_name,
                     t1.ranking as team1_ranking, t2.ranking as team2_ranking
                     ,mo.odds_team1, mo.odds_team2
                     ,mo.bookmaker_team1, mo.bookmaker_team2
                     ,mo.updated_at as odds_updated_at
                   FROM matches m
                   JOIN teams t1 ON m.team1_id = t1.id
                   JOIN teams t2 ON m.team2_id = t2.id
                   LEFT JOIN match_odds_latest mo ON mo.match_id = m.id
                   WHERE m.status = 'upcoming'
                   ORDER BY m.date ASC"""
            ).fetchall()
            return [dict(r) for r in rows]

    def clear_upcoming_related_data(self) -> dict:
        """
        Remove dados dependentes de partidas futuras.

        Uso recomendado para saneamento de timezone: limpar e recarregar upcoming.
        """
        with self.connect() as conn:
            rows = conn.execute("SELECT id FROM matches WHERE status='upcoming'").fetchall()
            match_ids = [int(r["id"]) for r in rows]

            if not match_ids:
                return {
                    "matches_deleted": 0,
                    "odds_latest_deleted": 0,
                    "odds_snapshots_deleted": 0,
                    "predictions_deleted": 0,
                }

            placeholders = ",".join(["?"] * len(match_ids))
            odds_snapshots_deleted = conn.execute(
                f"DELETE FROM odds_snapshots WHERE match_id IN ({placeholders})",
                match_ids,
            ).rowcount
            odds_latest_deleted = conn.execute(
                f"DELETE FROM match_odds_latest WHERE match_id IN ({placeholders})",
                match_ids,
            ).rowcount
            predictions_deleted = conn.execute(
                f"DELETE FROM predictions WHERE match_id IN ({placeholders})",
                match_ids,
            ).rowcount
            matches_deleted = conn.execute(
                f"DELETE FROM matches WHERE id IN ({placeholders})",
                match_ids,
            ).rowcount

            return {
                "matches_deleted": int(matches_deleted or 0),
                "odds_latest_deleted": int(odds_latest_deleted or 0),
                "odds_snapshots_deleted": int(odds_snapshots_deleted or 0),
                "predictions_deleted": int(predictions_deleted or 0),
            }

    def get_team_recent_matches(
        self, team_id: int, limit: int = 20, days: int = 90
    ) -> list[dict]:
        cutoff = date.today().isoformat()
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT * FROM matches
                   WHERE (team1_id=? OR team2_id=?) AND status='completed'
                     AND date >= date(?, '-' || ? || ' days')
                   ORDER BY date DESC LIMIT ?""",
                (team_id, team_id, cutoff, days, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_team_recent_matches_before(
        self,
        team_id: int,
        before_date,
        limit: int = 20,
        days: int = 90,
    ) -> list[dict]:
        before_dt = _parse_dt(before_date)
        if before_dt is None:
            return self.get_team_recent_matches(team_id, limit=limit, days=days)

        before_iso = before_dt.isoformat(timespec="seconds")
        window_start = (before_dt - timedelta(days=max(1, int(days)))).isoformat(timespec="seconds")
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT * FROM matches
                   WHERE (team1_id=? OR team2_id=?) AND status='completed'
                     AND datetime(date) < datetime(?)
                     AND datetime(date) >= datetime(?)
                   ORDER BY date DESC LIMIT ?""",
                (team_id, team_id, before_iso, window_start, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_team_side_stats(
        self,
        team_id: int,
        days: int = 180,
        as_of_date=None,
        max_rows: int = 800,
    ) -> dict:
        """
        Agrega forca em lados CT/T usando rounds de mapas historicos.

        Retorna taxas em [0, 1] com fallback neutro (0.5) quando nao houver base.
        """
        before_dt = _parse_dt(as_of_date) if as_of_date is not None else datetime.now().astimezone()
        if before_dt is None:
            before_dt = datetime.now().astimezone()

        before_iso = before_dt.isoformat(timespec="seconds")
        window_start = (before_dt - timedelta(days=max(1, int(days)))).isoformat(timespec="seconds")

        with self.connect() as conn:
            rows = conn.execute(
                """SELECT m.team1_id, m.team2_id,
                          mm.team1_ct_rounds, mm.team1_t_rounds,
                          mm.team2_ct_rounds, mm.team2_t_rounds
                   FROM match_maps mm
                   JOIN matches m ON m.id = mm.match_id
                   WHERE m.status='completed'
                     AND datetime(m.date) < datetime(?)
                     AND datetime(m.date) >= datetime(?)
                     AND (m.team1_id=? OR m.team2_id=?)
                   ORDER BY m.date DESC
                   LIMIT ?""",
                (
                    before_iso,
                    window_start,
                    team_id,
                    team_id,
                    max(1, int(max_rows)),
                ),
            ).fetchall()

        ct_won = 0
        ct_total = 0
        t_won = 0
        t_total = 0
        maps_count = 0

        for row in rows:
            rec = dict(row)
            is_team1 = int(rec.get("team1_id", 0) or 0) == int(team_id)

            t1_ct = _safe_int(rec.get("team1_ct_rounds"))
            t1_t = _safe_int(rec.get("team1_t_rounds"))
            t2_ct = _safe_int(rec.get("team2_ct_rounds"))
            t2_t = _safe_int(rec.get("team2_t_rounds"))

            if is_team1:
                ct_won += t1_ct
                ct_total += (t1_ct + t2_t)
                t_won += t1_t
                t_total += (t1_t + t2_ct)
            else:
                ct_won += t2_ct
                ct_total += (t2_ct + t1_t)
                t_won += t2_t
                t_total += (t2_t + t1_ct)

            maps_count += 1

        ct_rate = (ct_won / ct_total) if ct_total > 0 else 0.5
        t_rate = (t_won / t_total) if t_total > 0 else 0.5
        return {
            "team_id": int(team_id),
            "maps_count": int(maps_count),
            "ct_rounds_won": int(ct_won),
            "ct_rounds_total": int(ct_total),
            "ct_win_rate": float(ct_rate),
            "t_rounds_won": int(t_won),
            "t_rounds_total": int(t_total),
            "t_win_rate": float(t_rate),
        }

    def get_h2h(self, team1_id: int, team2_id: int) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT * FROM matches
                   WHERE status='completed'
                     AND ((team1_id=? AND team2_id=?) OR (team1_id=? AND team2_id=?))
                   ORDER BY date DESC LIMIT 10""",
                (team1_id, team2_id, team2_id, team1_id),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_h2h_before(
        self,
        team1_id: int,
        team2_id: int,
        before_date,
        limit: int = 10,
    ) -> list[dict]:
        before_dt = _parse_dt(before_date)
        if before_dt is None:
            return self.get_h2h(team1_id, team2_id)

        before_iso = before_dt.isoformat(timespec="seconds")
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT * FROM matches
                   WHERE status='completed'
                     AND datetime(date) < datetime(?)
                     AND ((team1_id=? AND team2_id=?) OR (team1_id=? AND team2_id=?))
                   ORDER BY date DESC LIMIT ?""",
                (before_iso, team1_id, team2_id, team2_id, team1_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    # ============================================================
    # Map Stats
    # ============================================================

    def upsert_team_map_stats(self, team_id: int, map_name: str, **kwargs):
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO team_map_stats (team_id, map_name, matches_played, wins,
                     win_rate, avg_rounds_won, ct_win_rate, t_win_rate, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(team_id, map_name) DO UPDATE SET
                     matches_played=excluded.matches_played, wins=excluded.wins,
                     win_rate=excluded.win_rate, avg_rounds_won=excluded.avg_rounds_won,
                     ct_win_rate=excluded.ct_win_rate, t_win_rate=excluded.t_win_rate,
                     updated_at=excluded.updated_at""",
                (
                    team_id, map_name,
                    kwargs.get("matches_played", 0),
                    kwargs.get("wins", 0),
                    kwargs.get("win_rate", 0.0),
                    kwargs.get("avg_rounds_won", 0.0),
                    kwargs.get("ct_win_rate", 0.0),
                    kwargs.get("t_win_rate", 0.0),
                    datetime.now().isoformat(),
                ),
            )

    def get_team_map_stats(self, team_id: int) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM team_map_stats WHERE team_id=?", (team_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_team_map_stats_before(
        self,
        team_id: int,
        before_date,
        days: int = 365,
        limit_matches: int = 300,
    ) -> list[dict]:
        before_dt = _parse_dt(before_date)
        if before_dt is None:
            return self.get_team_map_stats(team_id)

        before_iso = before_dt.isoformat(timespec="seconds")
        window_start = (before_dt - timedelta(days=max(1, int(days)))).isoformat(timespec="seconds")
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT m.team1_id, m.team2_id,
                          mm.map_name,
                          mm.team1_rounds, mm.team2_rounds,
                          mm.team1_ct_rounds, mm.team1_t_rounds,
                          mm.team2_ct_rounds, mm.team2_t_rounds,
                          mm.winner_id
                   FROM match_maps mm
                   JOIN matches m ON m.id = mm.match_id
                   WHERE m.status='completed'
                     AND datetime(m.date) < datetime(?)
                     AND datetime(m.date) >= datetime(?)
                     AND (m.team1_id=? OR m.team2_id=?)
                     AND mm.map_name IS NOT NULL
                   ORDER BY m.date DESC
                   LIMIT ?""",
                (before_iso, window_start, team_id, team_id, max(1, int(limit_matches))),
            ).fetchall()

        by_map: dict[str, dict] = {}
        for row in rows:
            rec = dict(row)
            map_name = str(rec.get("map_name", "")).strip()
            if not map_name:
                continue

            stats = by_map.setdefault(
                map_name,
                {
                    "team_id": team_id,
                    "map_name": map_name,
                    "matches_played": 0,
                    "wins": 0,
                    "win_rate": 0.0,
                    "avg_rounds_won": 0.0,
                    "ct_win_rate": 0.0,
                    "t_win_rate": 0.0,
                },
            )

            is_team1 = int(rec.get("team1_id", 0) or 0) == team_id
            rounds_won = _safe_int(rec.get("team1_rounds")) if is_team1 else _safe_int(rec.get("team2_rounds"))
            ct_rounds = _safe_int(rec.get("team1_ct_rounds")) if is_team1 else _safe_int(rec.get("team2_ct_rounds"))
            t_rounds = _safe_int(rec.get("team1_t_rounds")) if is_team1 else _safe_int(rec.get("team2_t_rounds"))
            won = int(rec.get("winner_id", 0) or 0) == team_id

            stats["matches_played"] += 1
            stats["wins"] += 1 if won else 0
            stats["avg_rounds_won"] += rounds_won
            stats["ct_win_rate"] += ct_rounds
            stats["t_win_rate"] += t_rounds

        output = []
        for stats in by_map.values():
            matches_played = max(1, int(stats["matches_played"]))
            rounds_base = max(1, matches_played * 12)
            stats["win_rate"] = (stats["wins"] / matches_played) * 100.0
            stats["avg_rounds_won"] = stats["avg_rounds_won"] / matches_played
            stats["ct_win_rate"] = (stats["ct_win_rate"] / rounds_base) * 100.0
            stats["t_win_rate"] = (stats["t_win_rate"] / rounds_base) * 100.0
            output.append(stats)

        return output

    # ============================================================
    # Odds
    # ============================================================

    def upsert_match_odds_latest(self, match_id: int, **kwargs):
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO match_odds_latest
                     (match_id, provider, fixture_id, market_key, odds_team1, odds_team2,
                      bookmaker_team1, bookmaker_team2, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(match_id) DO UPDATE SET
                     provider=excluded.provider,
                     fixture_id=excluded.fixture_id,
                     market_key=excluded.market_key,
                     odds_team1=excluded.odds_team1,
                     odds_team2=excluded.odds_team2,
                     bookmaker_team1=excluded.bookmaker_team1,
                     bookmaker_team2=excluded.bookmaker_team2,
                     updated_at=excluded.updated_at""",
                (
                    match_id,
                    kwargs.get("provider", "oddspapi"),
                    kwargs.get("fixture_id", ""),
                    kwargs.get("market_key", "h2h"),
                    kwargs.get("odds_team1"),
                    kwargs.get("odds_team2"),
                    kwargs.get("bookmaker_team1", ""),
                    kwargs.get("bookmaker_team2", ""),
                    kwargs.get("updated_at", datetime.now().isoformat()),
                ),
            )

    def insert_odds_snapshot(self, match_id: int, **kwargs):
        with self.connect() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO odds_snapshots
                    (match_id, provider, fixture_id, bookmaker, market_key, side, odds,
                     odds_changed_at, collected_at, fingerprint)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    match_id,
                    kwargs.get("provider", "oddspapi"),
                    kwargs.get("fixture_id", ""),
                    kwargs.get("bookmaker", ""),
                    kwargs.get("market_key", "h2h"),
                    kwargs.get("side", ""),
                    kwargs.get("odds"),
                    kwargs.get("odds_changed_at"),
                    kwargs.get("collected_at", datetime.now().isoformat()),
                    kwargs.get("fingerprint", ""),
                ),
            )

    def get_match_odds_latest(self, match_id: int) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM match_odds_latest WHERE match_id=? LIMIT 1",
                (match_id,),
            ).fetchone()
            return dict(row) if row else None

    # ============================================================
    # Predictions
    # ============================================================

    def save_prediction(self, match_id: int, **kwargs):
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO predictions
                     (match_id, predicted_winner_id, model_winner_id, official_pick_winner_id, pick_source,
                      team1_win_prob, team2_win_prob, value_pct, suggested_bet, suggested_stake,
                      odds_team1, odds_team2, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    match_id,
                    kwargs.get("official_pick_winner_id", kwargs.get("predicted_winner_id")),
                    kwargs.get("model_winner_id", kwargs.get("predicted_winner_id")),
                    kwargs.get("official_pick_winner_id", kwargs.get("predicted_winner_id")),
                    kwargs.get("pick_source", "value"),
                    kwargs.get("team1_win_prob", 0.5),
                    kwargs.get("team2_win_prob", 0.5),
                    kwargs.get("value_pct", 0.0),
                    kwargs.get("suggested_bet", ""),
                    kwargs.get("suggested_stake", 0.0),
                    kwargs.get("odds_team1"),
                    kwargs.get("odds_team2"),
                    datetime.now().isoformat(),
                ),
            )

    def get_prediction_history(self, limit: int = 100) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT p.*, m.event_name, t1.name as team1_name, t2.name as team2_name
                   FROM predictions p
                   JOIN matches m ON p.match_id = m.id
                   JOIN teams t1 ON m.team1_id = t1.id
                   JOIN teams t2 ON m.team2_id = t2.id
                   ORDER BY p.created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_recent_resolved_predictions(self, days: int = 30) -> list[dict]:
        """
        Retorna predições recentes com desfecho conhecido para ajuste de risco.

        Prioridade:
        1) tabela predictions com actual_winner_id preenchido
        2) fallback em daily_top5_items auditadas (win/loss)
        """
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT p.id AS prediction_id,
                          p.match_id,
                          p.predicted_winner_id,
                          p.model_winner_id,
                          p.official_pick_winner_id,
                          p.value_pct,
                          p.suggested_stake,
                          p.actual_winner_id,
                          p.created_at
                   FROM predictions p
                   WHERE p.actual_winner_id IS NOT NULL
                     AND datetime(p.created_at) >= datetime('now', ?)
                   ORDER BY p.created_at DESC""",
                (f"-{max(1, int(days))} days",),
            ).fetchall()
            out = [dict(r) for r in rows]
            if out:
                return out

            rows = conn.execute(
                """SELECT NULL AS prediction_id,
                          i.match_id,
                          i.predicted_winner_id,
                          i.model_winner_id,
                          i.official_pick_winner_id,
                          i.value_pct,
                          NULL AS suggested_stake,
                          i.actual_winner_id,
                          COALESCE(i.resolved_at, r.created_at) AS created_at
                   FROM daily_top5_items i
                   JOIN daily_top5_runs r ON r.id=i.run_id
                   WHERE i.outcome_status IN ('win', 'loss')
                     AND i.actual_winner_id IS NOT NULL
                     AND datetime(COALESCE(i.resolved_at, r.created_at)) >= datetime('now', ?)
                   ORDER BY COALESCE(i.resolved_at, r.created_at) DESC""",
                (f"-{max(1, int(days))} days",),
            ).fetchall()
            return [dict(r) for r in rows]

    # ============================================================
    # CLV Tracking
    # ============================================================

    def save_clv(
        self,
        match_id: int,
        side: str,
        open_odds: float,
        close_odds: float,
        prediction_id: int | None = None,
        created_at: str | None = None,
    ) -> float:
        """
        Persiste CLV da aposta.

        Convencao:
        - CLV positivo quando a odd de entrada e melhor que a odd de fechamento.
        """
        open_odds_f = _safe_float(open_odds)
        close_odds_f = _safe_float(close_odds)
        if open_odds_f <= 1.0 or close_odds_f <= 1.0:
            return 0.0

        clv_pct = ((open_odds_f / close_odds_f) - 1.0) * 100.0
        stamp = created_at or datetime.now().isoformat()
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO clv_tracking
                    (prediction_id, match_id, side, open_odds, close_odds, clv_pct, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    prediction_id,
                    int(match_id),
                    str(side or ""),
                    float(open_odds_f),
                    float(close_odds_f),
                    float(clv_pct),
                    stamp,
                ),
            )
        return float(clv_pct)

    def get_avg_clv(self, days: int = 30) -> float:
        with self.connect() as conn:
            row = conn.execute(
                """SELECT AVG(clv_pct) AS avg_clv
                   FROM clv_tracking
                   WHERE datetime(created_at) >= datetime('now', ?)""",
                (f"-{max(1, int(days))} days",),
            ).fetchone()
            if not row:
                return 0.0
            return _safe_float(row["avg_clv"])

    # ============================================================
    # Daily Top 5
    # ============================================================

    def get_daily_top5_run(self, run_date: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM daily_top5_runs WHERE run_date=? LIMIT 1",
                (run_date,),
            ).fetchone()
            return dict(row) if row else None

    def get_daily_top5_latest_run(self) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM daily_top5_runs ORDER BY run_date DESC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None

    def create_daily_top5_run(
        self,
        run_date: str,
        requested_top: int,
        total_candidates: int,
        candidates_with_odds: int,
        status: str,
    ) -> int:
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO daily_top5_runs
                     (run_date, requested_top, total_candidates, candidates_with_odds, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    run_date,
                    requested_top,
                    total_candidates,
                    candidates_with_odds,
                    status,
                    datetime.now().isoformat(),
                ),
            )
            row = conn.execute(
                "SELECT id FROM daily_top5_runs WHERE run_date=?",
                (run_date,),
            ).fetchone()
            return row["id"] if row else 0

    def save_daily_top5_items(self, run_id: int, picks: list[dict]):
        with self.connect() as conn:
            for item in picks:
                conn.execute(
                    """INSERT OR REPLACE INTO daily_top5_items
                         (run_id, rank, match_id, match_date, event_name, team1_id, team2_id,
                          team1_name, team2_name,
                          model_winner_id, model_winner_name,
                          official_pick_winner_id, official_pick_winner_name, pick_source,
                          predicted_winner_id, predicted_winner_name,
                          team1_win_prob, team2_win_prob, confidence, score, odds, bookmaker,
                          value_pct, expected_value, outcome_status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        item.get("rank", 0),
                        item.get("match_id"),
                        item.get("match_date", ""),
                        item.get("event_name", ""),
                        item.get("team1_id"),
                        item.get("team2_id"),
                        item.get("team1_name", ""),
                        item.get("team2_name", ""),
                        item.get("model_winner_id", item.get("predicted_winner_id")),
                        item.get("model_winner_name", item.get("predicted_winner_name", "")),
                        item.get("official_pick_winner_id", item.get("predicted_winner_id")),
                        item.get("official_pick_winner_name", item.get("predicted_winner_name", "")),
                        item.get("pick_source", "value"),
                        item.get("official_pick_winner_id", item.get("predicted_winner_id")),
                        item.get("official_pick_winner_name", item.get("predicted_winner_name", "")),
                        item.get("team1_win_prob"),
                        item.get("team2_win_prob"),
                        item.get("confidence"),
                        item.get("score"),
                        item.get("odds"),
                        item.get("bookmaker", ""),
                        item.get("value_pct"),
                        item.get("expected_value"),
                        item.get("outcome_status", "pending"),
                    ),
                )

    def get_daily_top5_items(self, run_id: int) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM daily_top5_items WHERE run_id=? ORDER BY rank ASC",
                (run_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def list_daily_top5_runs_with_pending(self, limit_days: int = 3) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT r.*
                   FROM daily_top5_runs r
                   WHERE r.run_date >= date('now', ?)
                     AND EXISTS (
                         SELECT 1 FROM daily_top5_items i
                         WHERE i.run_id = r.id AND i.outcome_status='pending'
                     )
                   ORDER BY r.run_date DESC""",
                (f"-{max(1, int(limit_days))} days",),
            ).fetchall()
            return [dict(r) for r in rows]

    def update_daily_top5_item_outcome(
        self,
        item_id: int,
        outcome_status: str,
        actual_winner_id: int | None = None,
        resolved_match_id: int | None = None,
        resolution_method: str = "",
    ):
        with self.connect() as conn:
            conn.execute(
                """UPDATE daily_top5_items
                   SET outcome_status=?,
                       actual_winner_id=?,
                       resolved_match_id=?,
                       resolved_at=?,
                       resolution_method=?
                   WHERE id=?""",
                (
                    outcome_status,
                    actual_winner_id,
                    resolved_match_id,
                    datetime.now().isoformat(),
                    resolution_method,
                    item_id,
                ),
            )

    def update_daily_top5_run_audited(self, run_id: int, status: str):
        with self.connect() as conn:
            conn.execute(
                """UPDATE daily_top5_runs
                   SET status=?, audited_at=?
                   WHERE id=?""",
                (
                    status,
                    datetime.now().isoformat(),
                    run_id,
                ),
            )

    def get_match_result_by_id(self, match_id: int) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                """SELECT m.*, t1.name AS team1_name, t2.name AS team2_name
                   FROM matches m
                   JOIN teams t1 ON t1.id=m.team1_id
                   JOIN teams t2 ON t2.id=m.team2_id
                   WHERE m.id=?
                   LIMIT 1""",
                (match_id,),
            ).fetchone()
            return dict(row) if row else None

    def list_completed_matches_between(self, start_iso: str, end_iso: str) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT m.*, t1.name AS team1_name, t2.name AS team2_name
                   FROM matches m
                   JOIN teams t1 ON t1.id=m.team1_id
                   JOIN teams t2 ON t2.id=m.team2_id
                   WHERE m.status='completed'
                     AND m.date >= ?
                     AND m.date <= ?
                   ORDER BY m.date ASC""",
                (start_iso, end_iso),
            ).fetchall()
            return [dict(r) for r in rows]

    # ============================================================
    # App State
    # ============================================================

    def get_state(self, state_key: str, default: str = "") -> str:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT state_value FROM app_state WHERE state_key=? LIMIT 1",
                (state_key,),
            ).fetchone()
            if not row:
                return default
            return row["state_value"] if row["state_value"] is not None else default

    def set_state(self, state_key: str, state_value: str):
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO app_state (state_key, state_value, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(state_key) DO UPDATE SET
                     state_value=excluded.state_value,
                     updated_at=excluded.updated_at""",
                (state_key, state_value, datetime.now().isoformat()),
            )

    # ============================================================
    # Stats
    # ============================================================

    def get_stats(self) -> dict:
        with self.connect() as conn:
            teams = conn.execute("SELECT COUNT(*) as c FROM teams").fetchone()["c"]
            players = conn.execute("SELECT COUNT(*) as c FROM players").fetchone()["c"]
            matches = conn.execute("SELECT COUNT(*) as c FROM matches WHERE status='completed'").fetchone()["c"]
            upcoming = conn.execute("SELECT COUNT(*) as c FROM matches WHERE status='upcoming'").fetchone()["c"]
            preds = conn.execute("SELECT COUNT(*) as c FROM predictions").fetchone()["c"]
            return {
                "teams": teams, "players": players,
                "completed_matches": matches, "upcoming_matches": upcoming,
                "predictions": preds,
            }


def _parse_dt(value) -> datetime | None:
    return to_storage_utc_datetime(
        value,
        logger=logger,
        context="db.parse_dt",
    )


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

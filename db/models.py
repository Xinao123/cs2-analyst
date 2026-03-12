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
from datetime import datetime, date
from contextlib import contextmanager

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

CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date);
CREATE INDEX IF NOT EXISTS idx_matches_status ON matches(status);
CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(team1_id, team2_id);
CREATE INDEX IF NOT EXISTS idx_players_team ON players(team_id);
CREATE INDEX IF NOT EXISTS idx_match_maps_match ON match_maps(match_id);
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
            logger.info(f"[DB] Inicializado: {self.db_path}")

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
                   FROM matches m
                   JOIN teams t1 ON m.team1_id = t1.id
                   JOIN teams t2 ON m.team2_id = t2.id
                   WHERE m.status = 'upcoming'
                   ORDER BY m.date ASC"""
            ).fetchall()
            return [dict(r) for r in rows]

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

    # ============================================================
    # Predictions
    # ============================================================

    def save_prediction(self, match_id: int, **kwargs):
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO predictions (match_id, predicted_winner_id, team1_win_prob,
                     team2_win_prob, value_pct, suggested_bet, suggested_stake,
                     odds_team1, odds_team2, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    match_id,
                    kwargs.get("predicted_winner_id"),
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

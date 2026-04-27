"""
SQLite persistence: coffee catalog (feature vectors), user profile, interaction log.
Seeded data is suitable for CQI/Starbucks-style import later (see source_ref, extra_json).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_default_db_path() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "data" / "brewbuddy.db"


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _schema() -> str:
    return """
    CREATE TABLE IF NOT EXISTS coffee_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        caffeine_level REAL NOT NULL,
        milk_level REAL NOT NULL,
        dairy_load REAL NOT NULL,
        bitterness REAL NOT NULL,
        state_category TEXT NOT NULL,
        source_ref TEXT,
        extra_json TEXT
    );

    CREATE TABLE IF NOT EXISTS user_profile (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        pref_strong_caffeine REAL NOT NULL DEFAULT 0.5,
        pref_lactose_free INTEGER NOT NULL DEFAULT 0,
        pref_low_bitterness REAL NOT NULL DEFAULT 0.5,
        updated_at TEXT
    );

    CREATE TABLE IF NOT EXISTS interaction_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        time_of_day TEXT,
        weather TEXT,
        temperature_c REAL,
        sleep_hours REAL,
        fatigue INTEGER,
        lactose_intolerance INTEGER,
        social_battery TEXT,
        context_key TEXT,
        ml_state_category TEXT,
        need_vector_json TEXT,
        coffee_vector_json TEXT,
        recommended_coffee TEXT,
        candidate_names_json TEXT,
        cosine_scores_json TEXT,
        rating INTEGER,
        reward REAL,
        strategy TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_interaction_created ON interaction_log(created_at);
    CREATE INDEX IF NOT EXISTS idx_coffee_category ON coffee_items(state_category);
    """


def _seed_coffees() -> List[Dict[str, Any]]:
    """
    Seeded menu with interpretable features (0–1) for content-based / cosine layer.
    state_category aligns with hybrid ML classifier labels.
    """
    return [
        {"name": "Espresso", "caffeine": 0.95, "milk": 0.0, "dairy": 0.0, "bitter": 0.85, "cat": "extreme_caffeine"},
        {"name": "Cappuccino", "caffeine": 0.55, "milk": 0.6, "dairy": 0.55, "bitter": 0.4, "cat": "comfort"},
        {"name": "Latte", "caffeine": 0.45, "milk": 0.78, "dairy": 0.65, "bitter": 0.25, "cat": "comfort"},
        {"name": "Americano", "caffeine": 0.7, "milk": 0.0, "dairy": 0.0, "bitter": 0.55, "cat": "extreme_caffeine"},
        {"name": "Mocha", "caffeine": 0.4, "milk": 0.55, "dairy": 0.6, "bitter": 0.35, "cat": "balanced"},
        {"name": "Macchiato", "caffeine": 0.6, "milk": 0.35, "dairy": 0.3, "bitter": 0.6, "cat": "light_wakeup"},
        {"name": "Flat White", "caffeine": 0.5, "milk": 0.65, "dairy": 0.6, "bitter": 0.35, "cat": "light_wakeup"},
        {"name": "Cortado", "caffeine": 0.6, "milk": 0.45, "dairy": 0.4, "bitter": 0.45, "cat": "light_wakeup"},
        {"name": "Cold Brew", "caffeine": 0.85, "milk": 0.0, "dairy": 0.0, "bitter": 0.5, "cat": "extreme_caffeine"},
        {"name": "Iced Coffee", "caffeine": 0.7, "milk": 0.15, "dairy": 0.12, "bitter": 0.4, "cat": "extreme_caffeine"},
        {"name": "Frappuccino", "caffeine": 0.3, "milk": 0.7, "dairy": 0.55, "bitter": 0.2, "cat": "balanced"},
        {"name": "Decaf", "caffeine": 0.05, "milk": 0.0, "dairy": 0.0, "bitter": 0.2, "cat": "relaxation"},
    ]


def init_db(db_path: Optional[Path] = None) -> None:
    path = db_path or get_default_db_path()
    with _connect(path) as c:
        c.executescript(_schema())
        cur = c.execute("SELECT COUNT(*) AS n FROM coffee_items")
        if cur.fetchone()["n"] == 0:
            for row in _seed_coffees():
                c.execute(
                    """
                    INSERT INTO coffee_items
                    (name, caffeine_level, milk_level, dairy_load, bitterness, state_category, source_ref, extra_json)
                    VALUES (?, ?, ?, ?, ?, ?, 'seed', NULL)
                    """,
                    (
                        row["name"],
                        row["caffeine"],
                        row["milk"],
                        row["dairy"],
                        row["bitter"],
                        row["cat"],
                    ),
                )
        if c.execute("SELECT 1 FROM user_profile WHERE id = 1").fetchone() is None:
            c.execute(
                "INSERT INTO user_profile (id, pref_strong_caffeine, pref_lactose_free, pref_low_bitterness, updated_at) "
                "VALUES (1, 0.5, 0, 0.5, ?)",
                (datetime.utcnow().isoformat() + "Z",),
            )
        c.commit()


def get_coffee_list(db_path: Optional[Path] = None) -> List[str]:
    init_db(db_path)
    with _connect(db_path or get_default_db_path()) as c:
        rows = c.execute("SELECT name FROM coffee_items ORDER BY name").fetchall()
    return [r["name"] for r in rows]


def get_coffee_dicts(db_path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    init_db(db_path)
    with _connect(db_path or get_default_db_path()) as c:
        rows = c.execute(
            "SELECT name, caffeine_level, milk_level, dairy_load, bitterness, state_category, source_ref, extra_json FROM coffee_items"
        ).fetchall()
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        extra: Dict[str, Any] = {}
        if r["extra_json"]:
            try:
                extra = json.loads(r["extra_json"])
            except json.JSONDecodeError:
                extra = {}
        out[r["name"]] = {
            "caffeine_level": float(r["caffeine_level"]),
            "milk_level": float(r["milk_level"]),
            "dairy_load": float(r["dairy_load"]),
            "bitterness": float(r["bitterness"]),
            "state_category": r["state_category"],
            "source_ref": r["source_ref"],
            "extra": extra,
        }
    return out


def get_user_profile(db_path: Optional[Path] = None) -> Dict[str, Any]:
    init_db(db_path)
    with _connect(db_path or get_default_db_path()) as c:
        r = c.execute(
            "SELECT pref_strong_caffeine, pref_lactose_free, pref_low_bitterness, updated_at FROM user_profile WHERE id = 1"
        ).fetchone()
    if not r:
        return {"pref_strong_caffeine": 0.5, "pref_lactose_free": 0, "pref_low_bitterness": 0.5, "updated_at": None}
    return {
        "pref_strong_caffeine": float(r["pref_strong_caffeine"]),
        "pref_lactose_free": int(r["pref_lactose_free"]),
        "pref_low_bitterness": float(r["pref_low_bitterness"]),
        "updated_at": r["updated_at"],
    }


def save_user_profile(
    pref_strong_caffeine: float,
    pref_lactose_free: int,
    pref_low_bitterness: float,
    db_path: Optional[Path] = None,
) -> None:
    init_db(db_path)
    with _connect(db_path or get_default_db_path()) as c:
        c.execute(
            """
            UPDATE user_profile SET
                pref_strong_caffeine = ?,
                pref_lactose_free = ?,
                pref_low_bitterness = ?,
                updated_at = ?
            WHERE id = 1
            """,
            (
                pref_strong_caffeine,
                1 if pref_lactose_free else 0,
                pref_low_bitterness,
                datetime.utcnow().isoformat() + "Z",
            ),
        )
        c.commit()


def log_interaction(
    record: Dict[str, Any],
    db_path: Optional[Path] = None,
) -> None:
    init_db(db_path)
    path = db_path or get_default_db_path()
    need = record.get("need_vector")
    cvec = record.get("recommended_coffee_vector")
    with _connect(path) as c:
        c.execute(
            """
            INSERT INTO interaction_log (
                created_at, time_of_day, weather, temperature_c, sleep_hours, fatigue,
                lactose_intolerance, social_battery, context_key, ml_state_category,
                need_vector_json, coffee_vector_json, recommended_coffee, candidate_names_json, cosine_scores_json,
                rating, reward, strategy
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.get("created_at", datetime.utcnow().isoformat() + "Z"),
                record.get("time_of_day"),
                record.get("weather"),
                record.get("temperature_c"),
                record.get("sleep_hours"),
                record.get("fatigue"),
                1 if record.get("lactose_intolerance") else 0,
                record.get("social_battery"),
                record.get("context_key"),
                record.get("ml_state_category"),
                json.dumps(need) if need is not None else "[]",
                json.dumps(cvec) if cvec is not None else None,
                record.get("recommended_coffee"),
                json.dumps(record.get("candidates", [])),
                json.dumps(record.get("cosine_scores") or {}),
                record.get("rating"),
                record.get("reward"),
                record.get("strategy"),
            ),
        )
        c.commit()

"""
SQLite persistence: coffee catalog (feature vectors), user profile, interaction log.
Active catalog: brewbuddy_data/datasets/cafe_menu.csv → coffee_items (source_ref = cafe_menu).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


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


def _bootstrap_db(db_path: Optional[Path] = None) -> Path:
    """Create schema and default user profile only (no catalog import)."""
    path = db_path or get_default_db_path()
    with _connect(path) as c:
        c.executescript(_schema())
        if c.execute("SELECT 1 FROM user_profile WHERE id = 1").fetchone() is None:
            c.execute(
                "INSERT INTO user_profile (id, pref_strong_caffeine, pref_lactose_free, pref_low_bitterness, updated_at) "
                "VALUES (1, 0.5, 0, 0.5, ?)",
                (datetime.utcnow().isoformat() + "Z",),
            )
        c.commit()
    return path


def init_db(db_path: Optional[Path] = None) -> None:
    path = _bootstrap_db(db_path)
    if _coffee_item_count(path) == 0:
        import_cafe_menu(path)


def _coffee_item_count(db_path: Path) -> int:
    with _connect(db_path) as c:
        return int(c.execute("SELECT COUNT(*) FROM coffee_items").fetchone()[0])


def get_cafe_menu_path() -> Path:
    return Path(__file__).resolve().parent / "datasets" / "cafe_menu.csv"


def get_cafe_menu_meta() -> Dict[str, Dict[str, str]]:
    """Short copy for sidebar catalog cards (from cafe_menu.csv)."""
    path = get_cafe_menu_path()
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        out[name] = {
            "description": str(row.get("desc_1", "") or row.get("desc_2", "") or "").strip(),
            "roast": str(row.get("roast", "") or "").strip(),
            "image": str(row.get("image", "") or "").strip(),
        }
    return out


def get_catalog_table(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Active café menu rows in coffee_items (source_ref = cafe_menu)."""
    init_db(db_path)
    with _connect(db_path or get_default_db_path()) as c:
        rows = c.execute(
            """
            SELECT name, caffeine_level, dairy_load, bitterness, state_category, source_ref
            FROM coffee_items
            WHERE source_ref = 'cafe_menu'
            ORDER BY name
            """
        ).fetchall()
    return pd.DataFrame(
        [
            {
                "Drink": r["name"],
                "Category": str(r["state_category"]).replace("_", " "),
                "Caffeine": round(float(r["caffeine_level"]), 2),
                "Dairy": round(float(r["dairy_load"]), 2),
                "Source": r["source_ref"] or "cafe_menu",
            }
            for r in rows
        ]
    )


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


def _coerce01(value: Any, default: float = 0.5) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v < 0:
        return 0.0
    if v > 1:
        return 1.0
    return v


def _infer_features_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Infer normalized features for arbitrary coffee dataset rows.
    The mapping is heuristic but consistent and fully persisted in extra_json.
    """
    name = str(row.get("name", "") or "").strip()
    roast = str(row.get("roast", "") or "").lower()
    rating = float(row.get("rating", 86) or 86)
    text_parts = [
        str(row.get("desc_1", "") or ""),
        str(row.get("desc_2", "") or ""),
        str(row.get("desc_3", "") or ""),
        str(row.get("review", "") or ""),
        name,
    ]
    text = " ".join(text_parts).lower()

    # Caffeine tendency: roast + style words (cold brew/espresso stronger).
    caffeine = 0.52
    if "espresso" in text:
        caffeine += 0.18
    if "cold brew" in text:
        caffeine += 0.16
    if "decaf" in text:
        caffeine = 0.08
    if "dark" in roast:
        caffeine += 0.08
    elif "medium-light" in roast or "light" in roast:
        caffeine -= 0.03

    # Milk / dairy load inferred from beverage language.
    milk = 0.0
    if any(k in text for k in ("latte", "flat white", "cappuccino", "macchiato", "cortado")):
        milk = 0.55
    if any(k in text for k in ("frappuccino", "mocha")):
        milk = 0.70
    if "americano" in text:
        milk = 0.05
    dairy = milk
    if any(k in text for k in ("oat", "soy", "almond", "lactose-free", "vegan")):
        dairy *= 0.25

    # Bitterness from roast and descriptors.
    bitter = 0.45
    if "dark" in roast:
        bitter += 0.2
    if any(k in text for k in ("chocolate", "cocoa", "syrupy")):
        bitter -= 0.04
    if any(k in text for k in ("citrus", "floral", "bright")):
        bitter -= 0.06
    if rating >= 93:
        bitter -= 0.04

    caffeine = _coerce01(caffeine)
    milk = _coerce01(milk, default=0.0)
    dairy = _coerce01(dairy, default=0.0)
    bitter = _coerce01(bitter)

    # Map to hybrid categories used by classifier/policy handoff.
    if caffeine >= 0.8:
        cat = "extreme_caffeine"
    elif caffeine <= 0.15:
        cat = "relaxation"
    elif milk >= 0.55 and caffeine <= 0.6:
        cat = "comfort"
    elif 0.45 <= caffeine <= 0.65:
        cat = "light_wakeup"
    else:
        cat = "balanced"

    return {
        "name": name,
        "caffeine_level": caffeine,
        "milk_level": milk,
        "dairy_load": dairy,
        "bitterness": bitter,
        "state_category": cat,
    }


def _normalize_dataset(df: pd.DataFrame, source_ref: str = "dataset") -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, rec in df.iterrows():
        row = {k: rec.get(k) for k in df.columns}
        if not str(row.get("name", "")).strip():
            continue
        feats = _infer_features_from_row(row)
        feats["source_ref"] = str(row.get("source_ref") or source_ref)
        feats["extra_json"] = json.dumps(
            {
                "roaster": row.get("roaster"),
                "roast": row.get("roast"),
                "origin": row.get("origin") or row.get("origin_1"),
                "origin_2": row.get("origin_2"),
                "rating": row.get("rating"),
                "price_100g_usd": row.get("100g_USD"),
                "review_date": row.get("review_date"),
            }
        )
        rows.append(feats)
    return pd.DataFrame(rows).drop_duplicates(subset=["name"], keep="first")


def import_cafe_menu(db_path: Optional[Path] = None) -> Dict[str, int]:
    """Import recognizable café drinks (recommended default catalog extension)."""
    path = get_cafe_menu_path()
    if not path.exists():
        return {"seen": 0, "inserted": 0, "updated": 0}
    return import_dataset_rows(
        dataset_paths=[path],
        db_path=db_path,
        limit_per_dataset=500,
        default_source_ref="cafe_menu",
    )


def reset_catalog_to_cafe_menu(db_path: Optional[Path] = None) -> Dict[str, int]:
    """Replace coffee_items with the curated café menu only (cafe_menu.csv)."""
    path = db_path or get_default_db_path()
    init_db(path)
    with _connect(path) as c:
        c.execute("DELETE FROM coffee_items")
        c.commit()
    menu = import_cafe_menu(path)
    return menu


def ensure_cafe_menu_catalog(db_path: Optional[Path] = None) -> Dict[str, int]:
    """Ensure DB catalog matches cafe_menu.csv only; re-import if stale or mixed sources."""
    path = db_path or get_default_db_path()
    init_db(path)
    expected = len(get_cafe_menu_meta())
    with _connect(path) as c:
        total = int(c.execute("SELECT COUNT(*) FROM coffee_items").fetchone()[0])
        cafe_only = int(
            c.execute("SELECT COUNT(*) FROM coffee_items WHERE source_ref = 'cafe_menu'").fetchone()[0]
        )
    if total == expected and cafe_only == expected:
        return {"seen": total, "inserted": 0, "updated": 0, "skipped": True}
    return reset_catalog_to_cafe_menu(path)


def import_dataset_rows(
    dataset_paths: Optional[List[Path]] = None,
    db_path: Optional[Path] = None,
    limit_per_dataset: int = 300,
    default_source_ref: str = "dataset",
) -> Dict[str, int]:
    """
    Import new rows from CSV datasets into coffee_items.
    Upserts by name and keeps deterministic limit to keep UI manageable.
    Default path is the curated café menu only (recognizable drink names).
    """
    base = Path(__file__).resolve().parent / "datasets"
    paths = dataset_paths or [base / "cafe_menu.csv"]
    inserted = 0
    updated = 0
    seen = 0
    path = _bootstrap_db(db_path)
    with _connect(path) as c:
        for p in paths:
            if not p.exists():
                continue
            df = pd.read_csv(p)
            if len(df) > limit_per_dataset:
                # Keep highest-rated samples for stronger recommendation relevance.
                if "rating" in df.columns:
                    df = df.sort_values("rating", ascending=False).head(limit_per_dataset)
                else:
                    df = df.head(limit_per_dataset)
            src = "cafe_menu" if p.name == "cafe_menu.csv" else default_source_ref
            norm = _normalize_dataset(df, source_ref=src)
            for _, r in norm.iterrows():
                seen += 1
                exists = c.execute("SELECT id FROM coffee_items WHERE name = ?", (r["name"],)).fetchone()
                if exists is None:
                    c.execute(
                        """
                        INSERT INTO coffee_items
                        (name, caffeine_level, milk_level, dairy_load, bitterness, state_category, source_ref, extra_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            r["name"],
                            float(r["caffeine_level"]),
                            float(r["milk_level"]),
                            float(r["dairy_load"]),
                            float(r["bitterness"]),
                            str(r["state_category"]),
                            str(r["source_ref"]),
                            str(r["extra_json"]),
                        ),
                    )
                    inserted += 1
                else:
                    c.execute(
                        """
                        UPDATE coffee_items SET
                            caffeine_level = ?,
                            milk_level = ?,
                            dairy_load = ?,
                            bitterness = ?,
                            state_category = ?,
                            source_ref = ?,
                            extra_json = ?
                        WHERE name = ?
                        """,
                        (
                            float(r["caffeine_level"]),
                            float(r["milk_level"]),
                            float(r["dairy_load"]),
                            float(r["bitterness"]),
                            str(r["state_category"]),
                            str(r["source_ref"]),
                            str(r["extra_json"]),
                            str(r["name"]),
                        ),
                    )
                    updated += 1
        c.commit()
    return {"seen": seen, "inserted": inserted, "updated": updated}

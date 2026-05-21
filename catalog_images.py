"""
Per-drink catalog imagery: maps each café menu item to its file under images/.
Image filenames are defined in brewbuddy_data/datasets/cafe_menu.csv (column: image).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent
IMAGES_DIR = _ROOT / "images"
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".gif"}

_DRINK_ALIASES: dict[str, str] = {
    "caffe misto": "Caffè Misto",
    "cafe misto": "Caffè Misto",
    "pour over coffee": "Pour-Over Coffee",
    "vietnamese iced coffee": "Vietnamese Iced Coffee (Cà Phê Sữa Đá)",
    "doppio espresso": "Doppio Espresso",
    "caramel macchiato": "Caramel Macchiato",
}


def _normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


@lru_cache(maxsize=1)
def _drink_image_map() -> dict[str, str]:
    from brewbuddy_data.database import get_cafe_menu_meta

    return {
        name: meta["image"]
        for name, meta in get_cafe_menu_meta().items()
        if meta.get("image")
    }


@lru_cache(maxsize=1)
def _file_index() -> dict[str, str]:
    index: dict[str, str] = {}
    if not IMAGES_DIR.exists():
        return index
    for path in IMAGES_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            index[_normalize_key(path.stem)] = path.name
    return index


def _resolve_filename(coffee_name: str) -> Optional[str]:
    name = coffee_name.strip()
    if not name:
        return None

    alias_key = _normalize_key(name)
    if alias_key in _DRINK_ALIASES:
        name = _DRINK_ALIASES[alias_key]

    mapped = _drink_image_map().get(name)
    if mapped and (IMAGES_DIR / mapped).exists():
        return mapped

    hit = _file_index().get(_normalize_key(name))
    return hit


def get_catalog_image_path(coffee_name: str) -> Optional[str]:
    """Return the image path for a drink."""
    filename = _resolve_filename(coffee_name)
    if not filename:
        return None
    return str((IMAGES_DIR / filename).resolve())

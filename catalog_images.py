"""
Per-drink catalog imagery: maps each café menu item to its file under images/.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

_ROOT = Path(__file__).resolve().parent
IMAGES_DIR = _ROOT / "images"
GENERATED_DIR = IMAGES_DIR / "generated"
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".gif"}

# Explicit map: café menu drink name (cafe_menu.csv) → filename in images/
DRINK_IMAGE_MAP: dict[str, str] = {
    "Espresso": "espresso.jpg",
    "Doppio Espresso": "dopio esspreso.jpg",
    "Ristretto": "ristretto.jpg",
    "Americano": "americano.jpg",
    "Long Black": "long black.jpg",
    "Latte": "latte.webp",
    "Cappuccino": "cappuccino.jpg",
    "Flat White": "flat white.jpg",
    "Macchiato": "macchiato.jpg",
    "Cortado": "cortado.webp",
    "Mocha": "mocha.png",
    "White Chocolate Mocha": "white chocolate mocha.jpg",
    "Hot Chocolate": "hot chocolate.jpg",
    "Affogato": "Affogato.jpg",
    "Cold Brew": "cold brew.jpg",
    "Iced Coffee": "iced coffee.jpg",
    "Iced Latte": "iced latte.jpg",
    "Iced Americano": "iced americano.jpg",
    "Iced Mocha": "iced mocha.jpg",
    "Frappuccino": "frappuccino.jpg",
    "Caramel Macchiato": "Caramel Machiatto.jpg",
    "Vanilla Latte": "vanilla latte.jpg",
    "Hazelnut Latte": "hazelnut latte.jpg",
    "Caffè Misto": "Caffee misto.webp",
    "Pour-Over Coffee": "pour over coffee.jpg",
    "French Press": "french press.jpg",
    "Turkish Coffee": "turkish coffee.jpg",
    "Vietnamese Iced Coffee (Cà Phê Sữa Đá)": "vietnamase iced coffee.jpg",
    "Irish Coffee": "irish coffee.jpg",
    "Decaf Latte": "decaf latte.avif",
    "Decaf Americano": "Decaf americano.jpg",
    "Decaf Cappuccino": "Decaf cappuccino.avif",
    "Oat Milk Latte": "oat milk latte.jpg",
    "Almond Milk Cappuccino": "Almond Milk Cappuccino.jpg",
    "Honey Cinnamon Latte": "honey cinnamon latte.jpg",
    # Legacy seed-only label
    "Decaf": "Decaf.webp",
}

# Alternate spellings / imports
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
def _file_index() -> dict[str, str]:
    """Normalized stem → actual filename (case-insensitive discovery)."""
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

    if name in DRINK_IMAGE_MAP:
        mapped = DRINK_IMAGE_MAP[name]
        if mapped and (IMAGES_DIR / mapped).exists():
            return mapped

    idx = _file_index()
    hit = idx.get(_normalize_key(name))
    if hit:
        return hit

    return None


def _stock_path(filename: str) -> Optional[str]:
    p = IMAGES_DIR / filename
    return str(p.resolve()) if p.exists() else None


def _slug(name: str) -> str:
    safe = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
    return f"{safe[:40]}-{digest}" if safe else digest


def _palette(name: str) -> tuple[int, int, int, int, int, int]:
    h = int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16)
    r1, g1, b1 = 40 + (h % 80), 28 + ((h >> 8) % 50), 20 + ((h >> 16) % 40)
    r2, g2, b2 = 120 + ((h >> 4) % 90), 85 + ((h >> 12) % 70), 55 + ((h >> 20) % 60)
    return r1, g1, b1, r2, g2, b2


def _write_generated_thumbnail(name: str, dest: Path) -> None:
    """Fallback only when no matching file exists in images/."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    w, h = 480, 360
    r1, g1, b1, r2, g2, b2 = _palette(name)
    img = Image.new("RGB", (w, h), (r1, g1, b1))
    draw = ImageDraw.Draw(img)
    for y in range(h):
        t = y / max(h - 1, 1)
        r = int(r1 * (1 - t) + r2 * t)
        g = int(g1 * (1 - t) + g2 * t)
        b = int(b1 * (1 - t) + b2 * t)
        draw.line([(0, y), (w, y)], fill=(r, g, b))
    draw.ellipse((w // 2 - 70, h // 2 - 90, w // 2 + 70, h // 2 + 50), fill=(45, 36, 30))
    label = name if len(name) <= 24 else name[:21] + "…"
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        font_lg = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
        font_lg = font
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((w - tw) // 2, h - th - 24), label, fill=(240, 228, 210), font=font)
    draw.text((w // 2 - 14, h // 2 - 48), "☕", fill=(212, 166, 116), font=font_lg)
    img.save(dest, format="PNG", optimize=True)


def get_catalog_image_path(coffee_name: str) -> Optional[str]:
    """Return the image path for a drink (user assets first, generated thumbnail last)."""
    filename = _resolve_filename(coffee_name)
    if filename:
        return _stock_path(filename)

    gen = GENERATED_DIR / f"{_slug(coffee_name)}.png"
    if gen.exists():
        return str(gen.resolve())
    _write_generated_thumbnail(coffee_name, gen)
    return str(gen.resolve())


def list_mapped_menu_images() -> dict[str, str]:
    """Debug/helper: drink name → resolved filesystem path."""
    from brewbuddy_data.database import get_cafe_menu_meta

    out: dict[str, str] = {}
    for name in get_cafe_menu_meta().keys():
        path = get_catalog_image_path(name)
        if path:
            out[name] = path
    return out

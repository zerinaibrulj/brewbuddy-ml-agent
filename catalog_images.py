"""
Per-drink catalog imagery: one unique file per café menu item (stock photo or generated thumbnail).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

_ROOT = Path(__file__).resolve().parent
IMAGES_DIR = _ROOT / "images"
GENERATED_DIR = IMAGES_DIR / "generated"

# Stock photos used once each for the closest menu match; all other drinks get a unique generated PNG.
_EXCLUSIVE_STOCK: dict[str, str] = {
    "Espresso": "espresso.jpg",
    "Cappuccino": "cappuccino.jpg",
    "Latte": "latte.webp",
    "Americano": "americano.jpg",
    "Mocha": "mocha.png",
    "Macchiato": "macchiato.jpg",
    "Flat White": "flat white.jpg",
    "Cortado": "cortado.webp",
    "Cold Brew": "cold brew.jpg",
    "Iced Coffee": "iced coffee.jpg",
    "Frappuccino": "frappuccino.jpg",
    "Affogato": "i.pinimgproxy.png",
    "Hot Chocolate": "caffee1.png",
    "Decaf Latte": "decaf.webp",
}


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


def _stock_path(filename: str) -> Optional[str]:
    p = IMAGES_DIR / filename
    return str(p) if p.exists() else None


def get_catalog_image_path(coffee_name: str) -> Optional[str]:
    """Return a unique image path for this drink."""
    if coffee_name in _EXCLUSIVE_STOCK:
        stock = _stock_path(_EXCLUSIVE_STOCK[coffee_name])
        if stock:
            return stock

    gen = GENERATED_DIR / f"{_slug(coffee_name)}.png"
    if gen.exists():
        return str(gen)
    _write_generated_thumbnail(coffee_name, gen)
    return str(gen)

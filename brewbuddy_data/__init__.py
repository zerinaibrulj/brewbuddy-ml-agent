"""BrewBuddy data layer: SQLite persistence and café menu catalog."""

from .database import (
    ensure_cafe_menu_catalog,
    get_cafe_menu_meta,
    get_cafe_menu_path,
    get_catalog_table,
    get_default_db_path,
    get_coffee_dicts,
    get_coffee_list,
    get_user_profile,
    import_cafe_menu,
    init_db,
    log_interaction,
    reset_catalog_to_cafe_menu,
    save_user_profile,
)

__all__ = [
    "ensure_cafe_menu_catalog",
    "get_cafe_menu_meta",
    "get_cafe_menu_path",
    "get_catalog_table",
    "get_default_db_path",
    "get_coffee_dicts",
    "get_coffee_list",
    "get_user_profile",
    "import_cafe_menu",
    "init_db",
    "log_interaction",
    "reset_catalog_to_cafe_menu",
    "save_user_profile",
]

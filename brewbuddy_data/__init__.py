"""BrewBuddy data layer: SQLite persistence and seeded coffee attributes."""

from .database import (
    get_default_db_path,
    get_cafe_menu_path,
    get_cafe_menu_meta,
    get_catalog_table,
    init_db,
    get_coffee_dicts,
    get_coffee_list,
    get_user_profile,
    save_user_profile,
    log_interaction,
    import_cafe_menu,
    import_dataset_rows,
    reset_catalog_to_cafe_menu,
)

__all__ = [
    "get_default_db_path",
    "get_cafe_menu_path",
    "get_cafe_menu_meta",
    "get_catalog_table",
    "init_db",
    "get_coffee_dicts",
    "get_coffee_list",
    "get_user_profile",
    "save_user_profile",
    "log_interaction",
    "import_cafe_menu",
    "import_dataset_rows",
    "reset_catalog_to_cafe_menu",
]

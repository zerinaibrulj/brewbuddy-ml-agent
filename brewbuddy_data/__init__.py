"""BrewBuddy data layer: SQLite persistence and seeded coffee attributes."""

from .database import (
    get_default_db_path,
    init_db,
    get_coffee_dicts,
    get_coffee_list,
    get_user_profile,
    save_user_profile,
    log_interaction,
    import_dataset_rows,
)

__all__ = [
    "get_default_db_path",
    "init_db",
    "get_coffee_dicts",
    "get_coffee_list",
    "get_user_profile",
    "save_user_profile",
    "log_interaction",
    "import_dataset_rows",
]

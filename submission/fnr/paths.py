"""Package-relative locations for fixed offline recipes."""
from pathlib import Path


def config_path(name: str) -> Path:
    return Path(__file__).resolve().parent / "configs" / name

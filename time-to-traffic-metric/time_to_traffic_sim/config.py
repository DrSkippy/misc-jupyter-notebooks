"""Project paths and config loading shared by the notebooks and the CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml (or `path`)."""
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


def output_dir(config: dict[str, Any]) -> Path:
    """The output directory from the config, relative to the project root; created if missing."""
    out: Path = PROJECT_ROOT / str(config["output"]["directory"])
    out.mkdir(exist_ok=True)
    return out

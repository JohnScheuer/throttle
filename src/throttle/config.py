"""Configuration file loading for Throttle.

Loads config from ~/.throttle/config.yaml if it exists.
CLI flags always override config file values.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# YAML is not a required dependency - fail gracefully if not installed
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


DEFAULT_CONFIG_PATH = Path.home() / ".throttle" / "config.yaml"


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses ~/.throttle/config.yaml

    Returns:
        Dictionary of config values, empty if file doesn't exist or YAML not available
    """
    if not YAML_AVAILABLE:
        return {}

    path = config_path or DEFAULT_CONFIG_PATH

    if not path.exists():
        return {}

    try:
        with open(path) as f:
            config = yaml.safe_load(f)

        if config is None:
            return {}

        if not isinstance(config, dict):
            print(
                f"Warning: Config file {path} does not contain a valid YAML dictionary. Ignoring.",
                file=sys.stderr,
            )
            return {}

        return config

    except yaml.YAMLError as e:
        print(
            f"Warning: Failed to parse config file {path}: {e}. Ignoring.",
            file=sys.stderr,
        )
        return {}
    except OSError as e:
        print(
            f"Warning: Failed to read config file {path}: {e}. Ignoring.",
            file=sys.stderr,
        )
        return {}


def apply_config_defaults(parser: Any, config: dict[str, Any]) -> None:
    """Apply config values as argument parser defaults.

    CLI arguments will override these defaults naturally through argparse.
    Works with both main parsers and parsers with subparsers.

    Args:
        parser: argparse.ArgumentParser instance
        config: Dictionary of config values
    """
    if not config:
        return

    # Convert config keys to CLI argument names (replace - with _)
    defaults = {}
    for key, value in config.items():
        # Skip None values
        if value is None:
            continue

        # Convert kebab-case to snake_case for argparse
        arg_name = key.replace("-", "_")
        defaults[arg_name] = value

    # Apply defaults to the main parser
    parser.set_defaults(**defaults)

    # Also apply to all subparsers if they exist
    if hasattr(parser, "_subparsers"):
        for action in parser._subparsers._actions:
            if hasattr(action, "choices") and action.choices:
                for subparser in action.choices.values():
                    subparser.set_defaults(**defaults)

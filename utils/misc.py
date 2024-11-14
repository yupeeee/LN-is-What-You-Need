import yaml

from .typing import Any, Dict, Path

__all__ = [
    "load_config",
]


def load_config(
    path: Path,
) -> Dict[str, Any]:
    with open(path, "r") as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

    return yaml_dict

"""
MLOps/train/display/base_display.py

顯示模式抽象基類與載入器。
新增顯示模式：在此目錄建立 <name>.py，定義繼承 BaseDisplayMode 且 NAME = "<name>" 的類別。
"""

from abc import ABC
from pathlib import Path

from transformers import TrainerCallback

DISPLAY_DIR = Path(__file__).parent


class BaseDisplayMode(TrainerCallback, ABC):
    NAME = ""

    def __init__(self, context: dict):
        self.context = context


def _load_display_mode(context: dict) -> BaseDisplayMode | None:
    mode_file = DISPLAY_DIR / ".mode"
    mode_name = mode_file.read_text(encoding="utf-8").strip() if mode_file.exists() else "default"

    if mode_name == "none":
        return None

    mode_path = DISPLAY_DIR / f"{mode_name}.py"
    if not mode_path.exists():
        return None

    import importlib.util
    import sys as _sys
    if str(DISPLAY_DIR) not in _sys.path:
        _sys.path.insert(0, str(DISPLAY_DIR))

    spec = importlib.util.spec_from_file_location(f"_dm_{mode_name}", mode_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for attr in vars(mod).values():
        if (isinstance(attr, type)
                and issubclass(attr, BaseDisplayMode)
                and attr is not BaseDisplayMode
                and getattr(attr, "NAME", "") == mode_name):
            return attr(context)
    return None

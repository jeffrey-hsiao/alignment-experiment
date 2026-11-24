"""
MLOps/train_catalog.py

訓練方法與設定檔的「目錄」查詢（不依賴 torch，run.py 與 GUI 共用的單一來源）：
  - available_methods()        train/scripts/train_*.py → 方法名
  - get_cfg_cls(method)        方法 → base_config 裡對應的設定類（依 METHOD 匹配）
  - needs_data(method)         該方法是否吃 data/unsafe+normal
  - list_configs(method)       configs/<method>/ 的具名設定檔
  - get_default_config / set_default_config     configs/<method>/.default
  - get_data_default / set_data_default         data/corpora/.defaults
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent                      # MLOps/
TRAIN_DIR = ROOT / "train"
CONFIGS_DIR = TRAIN_DIR / "configs"
SCRIPTS_DIR = TRAIN_DIR / "scripts"
DATA_DEFAULT_FILE = ROOT.parent / "data" / "corpora" / ".defaults"


def available_methods() -> list[str]:
    """掃描 train/scripts/train_*.py，自動取得所有可用方法名稱（字母序）。"""
    return sorted(p.stem[len("train_"):] for p in SCRIPTS_DIR.glob("train_*.py"))


def get_cfg_cls(method: str):
    """從 base_config 動態取得對應設定類（依 METHOD 屬性匹配）；找不到回 None。"""
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    import base_config as bc
    for attr in vars(bc).values():
        if (isinstance(attr, type) and issubclass(attr, bc.BaseConfig)
                and attr is not bc.BaseConfig and getattr(attr, "METHOD", "") == method):
            return attr
    return None


def needs_data(method: str) -> bool:
    """NEEDS_DATA（預設 True；找不到設定類也當 True，保守）。"""
    return bool(getattr(get_cfg_cls(method), "NEEDS_DATA", True))


def list_configs(method: str) -> list[str]:
    """configs/<method>/*.txt 的具名設定檔。排除 default 與 km_cos 的 persona_ 附屬檔
    （那些是 target/base system 的內容檔，不是一份 config）。"""
    d = CONFIGS_DIR / method
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.txt")
                  if p.stem != "default" and not p.name.startswith("persona_"))


def _default_pointer(method: str) -> Path:
    return CONFIGS_DIR / method / ".default"


def get_default_config(method: str) -> str | None:
    p = _default_pointer(method)
    return (p.read_text(encoding="utf-8").strip() or None) if p.exists() else None


def set_default_config(method: str, name: str) -> None:
    _default_pointer(method).write_text(name, encoding="utf-8")


def _data_defaults() -> dict:
    if not DATA_DEFAULT_FILE.exists():
        return {}
    return json.loads(DATA_DEFAULT_FILE.read_text(encoding="utf-8"))


def get_data_default(method: str) -> list[str] | None:
    return _data_defaults().get(method) or None


def set_data_default(method: str, versions: list[str]) -> None:
    d = _data_defaults()
    d[method] = versions
    DATA_DEFAULT_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

"""
MLOps/train/scripts/base_config.py

設定檔父類。載入順序（後者覆蓋前者）：
  1. 子類 DEFAULTS（Python fallback）
  2. configs/default.txt（全局共用默認）
  3. configs/{method}/default.txt（方法層默認）
  4. configs/{method}/{name}.txt（指定設定檔）

txt 格式：
  # 注釋以 # 開頭
  key = value
  # 數值類型自動推斷（int / float / bool / str）
"""

from pathlib import Path

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


class BaseConfig:
    """共用超參數的默認值與載入邏輯。子類設定 METHOD 與 DEFAULTS。"""

    METHOD = ""

    DEFAULTS: dict = {
        "model_name":   "Qwen/Qwen2.5-1.5B-Instruct",
        "lora_r":       16,
        "lora_alpha":   32,
        "lora_dropout": 0.05,
        "batch_size":   2,
        "grad_accum":   16,
        "lr":           1e-5,
        "max_length":   354,
        "epochs":       3,
        "save_steps":   500,
        "eval_steps":   500,
    }

    def __init__(self, config_name: str, overrides: dict | None = None):
        params = dict(self.DEFAULTS)

        for path in [
            CONFIGS_DIR / "default.txt",
            CONFIGS_DIR / self.METHOD / "default.txt",
            CONFIGS_DIR / self.METHOD / f"{config_name}.txt",
        ]:
            if path.exists():
                params.update(self._parse(path))

        if overrides:
            params.update(overrides)

        self._params = params

    # ── dict-like access ──────────────────────────────────────────────────────

    def __getitem__(self, key: str):
        return self._params[key]

    def __contains__(self, key: str) -> bool:
        return key in self._params

    def get(self, key: str, default=None):
        return self._params.get(key, default)

    def as_dict(self) -> dict:
        return dict(self._params)

    # ── txt parser ────────────────────────────────────────────────────────────

    @staticmethod
    def _parse(path: Path) -> dict:
        result = {}
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()

            if val.lower() in ("true", "false"):
                result[key] = val.lower() == "true"
                continue
            try:
                result[key] = int(val)
                continue
            except ValueError:
                pass
            try:
                result[key] = float(val)
                continue
            except ValueError:
                pass
            result[key] = val

        return result


class EmptyConfig(BaseConfig):
    METHOD = "empty"
    DEFAULTS = {
        **BaseConfig.DEFAULTS,
        "total_steps":   20,
        "logging_steps":  2,
        "save_steps":     5,
        "eval_steps":     5,
    }


class MyDPOConfig(BaseConfig):
    METHOD = "dpo"
    DEFAULTS = {
        **BaseConfig.DEFAULTS,
        "dpo_beta": 0.3,
    }


class MySFTConfig(BaseConfig):
    METHOD = "sft"
    DEFAULTS = {
        **BaseConfig.DEFAULTS,
        "batch_size": 1,
        "lr":         5e-5,
        "max_length": 512,
        "epochs":     1,
        "save_steps": 1000,
        "eval_steps": 100,
        "invert":     True,
    }

"""
MLOps/train/scripts/train_km_cos.py

km_cos（cosplay 行為編輯）：與 dpo/sft/prune 並列的訓練方法。

它與 prune 共用整套搜尋引擎與 glue——KmCosTrainer 直接繼承 train_prune 的
PruneTrainer，只做兩件事：
  1. 換掉 CONFIG_CLASS 為 KmCosConfig（不同的資料格式與欄位）；
  2. 覆寫 run()，固定走 _run_search(KmCosRunner)，不經過 prune 的 mode 分支。
其餘 setup_model / build_callbacks / _run_search / checkpoint 全部繼承，不複製。

為什麼獨立成方法而不是 prune 的一個 mode：km_cos 吃的是「單句中性題 + 一對
對比 system prompt」，用對比相關性選格，與 prune 的「有害/無害成對題 + 拒絕
判定」是兩套面向使用者的契約；共用的是底層引擎，不是方法本身。

使用方式（經 run.py）：
  python run.py train km_cos --config <name>
  python run.py                      # 無參數 → km_cos GUI（產生 config 後呼叫這裡）
"""

import argparse
import sys
from pathlib import Path

MLOPS_DIR = Path(__file__).parents[2]
sys.path.insert(0, str(MLOPS_DIR))

from base_config import KmCosConfig          # noqa: E402
from base_trainer import BaseTrainer         # noqa: E402
from train_prune import PruneTrainer         # noqa: E402  共用 glue，不複製


class KmCosTrainer(PruneTrainer):
    """結構與 PruneTrainer 相同，只是固定走 km_cos 搜尋。"""

    CONFIG_CLASS = KmCosConfig

    def run(self):
        from prune_km_cos import KmCosError, KmCosRunner
        return self._run_search(KmCosRunner, KmCosError)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    BaseTrainer.base_args(parser)
    KmCosTrainer(parser.parse_args()).run()

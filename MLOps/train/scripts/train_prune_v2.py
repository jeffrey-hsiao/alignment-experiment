"""
MLOps/train/scripts/train_prune_v2.py

prune_v2（gen-2 掃描）：與 dpo/sft/prune/km_cos 並列的訓練方法。

與 prune 共用整套引擎與 glue——PruneV2Trainer 直接繼承 train_prune 的
PruneTrainer，只做兩件事：
  1. 換 CONFIG_CLASS 為 PruneV2Config（多出 (T, r) 掃描參數）；
  2. 覆寫 run()，固定走 _run_search(PruneV2Runner)，不經 prune 的 mode 分支。
setup_model / build_callbacks / _run_search / checkpoint 全部繼承，不複製。

為什麼獨立成方法而不是 prune 的一個 mode：prune_v2 重新制定了「排行完成後」的
搜尋策略（單調 T/r 掃描，取代粗測+細搜與「M 撞上限」停止條件），是另一套流程；
共用的是底層選點與生成判定引擎，不是搜尋本身。資料格式與 prune 相同（有害/無害
成對題 + 拒絕判定），只是搜尋驅動器換成 PruneV2Runner（見 prune_v2_search.py）。

使用方式（經 run.py）：
  python run.py train prune_v2 --config <name>
  python run.py train prune_v2 --resume <run_id>     # 接續中斷的掃描（確定性重放）
"""

import argparse
import sys
from pathlib import Path

MLOPS_DIR = Path(__file__).parents[2]
sys.path.insert(0, str(MLOPS_DIR))

from base_config import PruneV2Config        # noqa: E402
from base_trainer import BaseTrainer         # noqa: E402
from train_prune import PruneTrainer         # noqa: E402  共用 glue，不複製


class PruneV2Trainer(PruneTrainer):
    """結構與 PruneTrainer 相同，只是固定走 prune_v2 搜尋。"""

    CONFIG_CLASS = PruneV2Config

    def run(self):
        from prune_search import KmSearchError
        from prune_v2_search import PruneV2Runner
        return self._run_search(PruneV2Runner, KmSearchError)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    BaseTrainer.base_args(parser)
    PruneV2Trainer(parser.parse_args()).run()

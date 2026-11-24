"""
MLOps/train/scripts/train_vector.py

vector（激活偏移版 prune_v2，解除對齊）：與 prune / prune_v2 / km_cos 並列的訓練方法。

結構與 train_prune_v2 相同——繼承 PruneTrainer 的 glue，只換 CONFIG_CLASS 與搜尋 runner。
VectorRunner 繼承 PruneV2Runner，分組／選點／判定都呼叫 prune_v2 的實作，只換編輯方式
（偏移）與搜尋方向（由上往下），見 prune_vector.py。

使用方式（經 run.py）：
  python run.py train vector --config <name>
  python run.py train vector --resume <run_id>
"""

import argparse
import sys
from pathlib import Path

MLOPS_DIR = Path(__file__).parents[2]
sys.path.insert(0, str(MLOPS_DIR))

from base_config import VectorConfig         # noqa: E402
from base_trainer import BaseTrainer         # noqa: E402
from train_prune import PruneTrainer         # noqa: E402  共用 glue，不複製


class VectorTrainer(PruneTrainer):
    """結構與 PruneTrainer 相同，只是固定走 vector 搜尋。"""

    CONFIG_CLASS = VectorConfig

    def run(self):
        from prune_search import KmSearchError
        from prune_vector import VectorRunner
        return self._run_search(VectorRunner, KmSearchError)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    BaseTrainer.base_args(parser)
    VectorTrainer(parser.parse_args()).run()

"""
pipline/dataset.py

PyTorch Dataset，讀取 JSONL 格式的 DPO pair，
供 DataLoader 使用。
"""

import json
from pathlib import Path
from torch.utils.data import Dataset


class DPODataset(Dataset):
    """
    讀取 prepare_dataset.py 產生的 JSONL 檔案。
    每筆資料包含：prompt, chosen, rejected。
    """

    def __init__(self, path: str | Path):
        self.records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


def collate_fn(batch: list[dict]) -> dict[str, list]:
    """
    簡單的 collate function，將 batch 整理成 dict of lists。
    tokenization 交給各訓練腳本處理。
    """
    return {
        "prompt":   [item["prompt"]   for item in batch],
        "chosen":   [item["chosen"]   for item in batch],
        "rejected": [item["rejected"] for item in batch],
    }

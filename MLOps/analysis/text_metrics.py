"""
MLOps/analysis/text_metrics.py

純文字指標（無重量級相依，torch/transformers 都不碰），供需要它的各處共用同一份定義，
避免同一個指標在多個檔各寫一份而門檻漂移。
"""
from collections import Counter


def repeated(text: str, n: int = 6) -> float:
    """全文重複率：最常見的 n-gram 覆蓋了全文多少比例（0~1，越高越重複）。
    太短（< n*2）不具統計意義，直接回 0。"""
    if len(text) < n * 2:
        return 0.0
    return max(Counter(text[i:i + n] for i in range(len(text) - n + 1)).values()) * n / len(text)

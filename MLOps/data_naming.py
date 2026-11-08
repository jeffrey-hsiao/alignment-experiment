"""
MLOps/data_naming.py

資料檔名的「方法前綴」慣例 + 不匹配警告（單一來源，各訓練方法共用）。

慣例：資料檔名前綴標示它是給哪種方法用的——
    dpo_    → DPO 偏好訓練資料（prompt/chosen/rejected）
    sft_    → SFT 監督訓練資料
    km_     → km_search 拒絕探測種子（text/group/id/pair_id）
    prune_  → 一般 prune 探測種子

選檔載入時呼叫 warn_if_method_mismatch(路徑, 當前方法)：前綴與方法不符就印警告
（只警告、不阻擋，照使用者意圖保留「硬要用也行」的彈性）。無前綴的檔視為通用，
不警告——這樣既有的舊檔名（如 refusal_dataset.jsonl）不會被誤報。

刻意零重量（只用 pathlib / sys），任何方法的載入點都能 import 呼叫。
"""
import sys
from pathlib import Path

# 當前方法 → 可接受的檔名前綴
_METHOD_PREFIXES: dict[str, tuple[str, ...]] = {
    "dpo":       ("dpo_",),
    "sft":       ("sft_",),
    "prune":     ("km_", "prune_"),
    "km_search": ("km_", "prune_"),
}
_ALL_PREFIXES = ("dpo_", "sft_", "km_", "prune_")


def method_of_filename(path) -> str | None:
    """回傳檔名前綴對應的方法標記（dpo/sft/km/prune），無前綴回傳 None。"""
    name = Path(path).name.lower()
    for prefix in _ALL_PREFIXES:
        if name.startswith(prefix):
            return prefix.rstrip("_")
    return None


def warn_if_method_mismatch(path, method: str) -> bool:
    """前綴與 method 不符時印警告並回傳 True；相符或無前綴則回傳 False（不警告）。"""
    name = Path(path).name
    if method_of_filename(path) is None:
        return False  # 無前綴 → 通用，不判斷
    accepted = _METHOD_PREFIXES.get(method, ())
    if any(name.lower().startswith(a) for a in accepted):
        return False  # 前綴與方法相符
    # 不用 emoji：run.py 的 detached console stderr 可能是 cp950，emoji 等非 CJK 字元
    # 會觸發 UnicodeEncodeError 進而弄掛 run；中文在 cp950(Big5) 可編碼，安全。
    print(f"[資料警告] 檔名前綴看起來不是給「{method}」用的：{name}"
          f"（期望前綴 {accepted or '(未定義)'}；若確定要用請忽略此警告）",
          file=sys.stderr, flush=True)
    return True

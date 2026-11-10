"""
MLOps/data_paths.py

資料路徑的「單一來源」。全專案的資料統一收斂到 repo 根目錄下的 data/，依用途分子目錄：

    data/
      seeds/         prune / km_search / km_cos 的輸入提示詞種子（*.jsonl）
      model_groups/  km_search 跑過後各模型對每題的真實分組
      corpora/       梯度訓練（dpo/sft）的版本化語料 {type}/{version}/{date}/train.jsonl
      analysis/      探測 / 掃描 / 報表等分析產物（非輸入，不該餵回訓練）

舊版散落在 pipelines/data/ 與 MLOps/data/ 兩處、且兩套 ROOT 慣例。resolve() 提供
「新位置優先、舊路徑自動 remap」的相容層：任何仍寫舊路徑（pipelines/data/... 或
MLOps/data/...）的 config 或程式，都會被映射到新位置，不必一次改完所有引用點。

刻意零重量（只用 pathlib / json），任何載入點都能 import。
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]      # MLOps/data_paths.py → repo 根
DATA = REPO / "data"
SEEDS = DATA / "seeds"
MODEL_GROUPS = DATA / "model_groups"
CORPORA = DATA / "corpora"
ANALYSIS = DATA / "analysis"

# 被歸類為「種子」（輸入提示詞）的檔名；其餘 pipelines/data 底下的東西視為 analysis 產物。
SEED_FILES = frozenset({
    "km_merged.jsonl",
    "km_refusal_western.jsonl",
    "km_cos_prompts.jsonl",
    "refusal_dataset.jsonl",
    "refusal_dataset.meta.json",
    "refusal_seeds_regrouped.jsonl",
})


def resolve(pathlike) -> Path:
    """把（可能是舊版的）資料路徑解析成實際位置。

    規則：絕對路徑原樣；相對 repo 根已存在就用（相容任何未搬動的檔）；否則把已知的
    舊前綴（pipelines/data/... 、MLOps/data/...）remap 到新的 data/ 子目錄。
    """
    p = Path(pathlike)
    if p.is_absolute():
        return p
    direct = REPO / p
    if direct.exists():
        return direct

    s = p.as_posix()
    if s.startswith("MLOps/data/"):
        return CORPORA / s[len("MLOps/data/"):]
    if s.startswith("pipelines/data/model_groups"):
        rest = s[len("pipelines/data/model_groups"):].lstrip("/")
        return MODEL_GROUPS / rest if rest else MODEL_GROUPS
    if s.startswith("pipelines/data/"):
        rest = s[len("pipelines/data/"):]
        head = rest.split("/", 1)[0]
        if head == "model_groups":
            return MODEL_GROUPS / rest[len("model_groups"):].lstrip("/")
        return (SEEDS / rest) if head in SEED_FILES else (ANALYSIS / rest)
    if len(p.parts) == 1 and p.name in SEED_FILES:
        return SEEDS / p.name
    return direct  # 可能不存在；交由呼叫端報錯


def read_jsonl(path) -> list[dict]:
    """共用 jsonl 讀取：逐行 json.loads，跳過空行。路徑先經 resolve()。"""
    path = resolve(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到資料檔：{path}")
    out = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{line_no} JSON 解析失敗：{e}")
    return out

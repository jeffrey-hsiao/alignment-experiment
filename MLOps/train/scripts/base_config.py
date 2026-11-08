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

    # False：這個方法的訓練經驗不來自 data/unsafe+normal（例如 self-play 從
    # 實際對局產生經驗），run.py 的 cmd_train/cmd_continuetrain 會依此跳過
    # 資料版本檢查與路徑收集。
    NEEDS_DATA = True

    DEFAULTS: dict = {
        "objective":    "degrade",
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
    }


class SelfPlayConfig(BaseConfig):
    """Self-play 強化學習訓練：訓練經驗來自實際對局，不吃 data/unsafe+normal。"""

    METHOD = "selfplay"
    NEEDS_DATA = False
    DEFAULTS = {
        "objective":         "boyi",
        "model_name":        "Qwen/Qwen2.5-0.5B-Instruct",
        "llm_agent_version": 3,      # 1/2/3 對應 card_game_trainer 的 LLMAgentV1/V2/V3
        # 全參數微調（False）在博弈論自己的訓練紀錄裡會讓小模型生成能力崩潰，
        # 驗證過的 98.2% valid rate 成功結果用的是 LoRA（見 configs/selfplay/
        # default.txt 的說明），Python fallback 預設值跟著改成一致。
        "use_lora":          True,
        "use_int8":          False,
        "n_games":           30,
        "save_steps":        10,     # 每 N 局存一次 checkpoint
        "eval_steps":        10,     # 每 N 局觸發一次 GenerationTestCallback
    }


class PruneConfig(BaseConfig):
    """精確神經元裁減：把指定的 MLP 神經元歸零，量測拒絕率變化。

    NEEDS_DATA=False：提示詞來自 pipelines/refusal_seeds*.jsonl（有害題與其最小
    差異無害對照句成對），不吃 data/unsafe+normal，所以 run.py 會跳過資料版本
    解析。沒有梯度運算，lora_* / lr / epochs 等超參數在此無作用。
    """

    METHOD = "prune"
    NEEDS_DATA = False
    DEFAULTS = {
        **BaseConfig.DEFAULTS,
        "objective":      "normal",   # 用正常 system prompt 量拒絕，不是劣化模式
        "model_name":     "Qwen/Qwen2.5-1.5B-Instruct",
        "seeds":          "pipelines/refusal_seeds_political.jsonl",  # 相對於 repo 根目錄
        # 要裁減的神經元，格式 "層:索引,層:索引"。留空則改讀 probe_json。
        "neurons":        "",
        # neuron_probe.py 的 JSON 摘要；neurons 留空時從這裡取完全分離的清單。
        "probe_json":     "",
        "probe_key":      "harmful_on__harmless_off",
        "eval_n":         60,   # 評估用的提示詞數量（有害+無害各半），0 表示全部
        "gen_max_tokens": 48,   # 判定拒絕只需要開頭幾句
        "save_steps":     1,
        "eval_steps":     1,
        # fixed：裁減 neurons / probe_json 指定的節點；km_search：自動掃描 K、M（見 prune_search.py）
        "mode":           "fixed",
        "judge":          "keyword",
        "judge_device":   "cpu",
        # model：用當前模型回答 seeds 並分組，另存於 model_groups_dir，不改動 seeds；
        # dataset：用 seeds 內現成的 group，只適用產生它的 dataset_group_model
        "group_source":   "model",
        "model_groups_dir":    "pipelines/data/model_groups",
        "dataset_group_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "rank_ratio":     0.6,      # 排序集比例；調參集 tune_ratio，剩下的是驗證集
        "tune_ratio":     0.2,
        # 選點方向：positive 只跑亮表(act>0 率，相容舊結果)/ negative 只跑暗表(act<0 率)
        # / both 兩表合併去重。negative 與 both 需要新版 ranking（含 *_pos_rate）
        "sign":           "positive",
        # True 時把「翻轉型」(一表選中、相反表排除)改成取負(翻轉向量)而非歸零；
        # 開了它務必一起開 test_complied，否則過度拒絕的副作用會被漏看
        "reverse_type":   False,
        "test_complied":  False,    # 每格另測順從題，量裁減造成的過度拒絕
        "ranking_path":   "",       # 留空就用 rank 集重新排序
        "start_k":        0,        # 與 end_m 都留 0 時由粗測決定範圍
        "end_m":          0,
        "k_floor":        0,        # 0 表示起點的一半
        "km_up":          1.25,
        "k_down":         0.8,
        "collapse_ratio": 0.10,     # 自動崩壞占該格全部題數的比例達此值就降 K
        "coarse_stop_flips":  2,    # 粗測翻轉連續 coarse_stop_rounds 格不超過此值就停
        "coarse_stop_rounds": 2,
        "control_top":     2,       # 隨機對照與驗證只跑翻轉減崩壞最高的幾格
        "control_repeats": 5,       # 0 表示不跑隨機對照
        "test_seeds":      "",      # 新題目；留空就不驗證，舊 test 集已用過
        # 單筆回答只要未被判拒絕，且符合以下任一項就算崩壞；預設值來自
        # analysis/quality_calibrate.py：全文重複率 ≥ repeat_bad、題目相關分數
        # ≤ relevance_bad、英文回答的亂碼機率 > gibberish_bad
        "repeat_bad":      0.1442,
        "relevance_bad":   -7.6,
        "gibberish_bad":   0.5,
        "n_steps":         8,       # 排序探測記錄的 token 位置數
        "split_seed":      0,
        "limit":           0,       # 只取前 N 筆（冒煙測試）
        # True 時 generation_test.txt 只寫各組計數，不列題目/回答原文（跑違禁題
        # 資料集時開啟，避免檢視進程時顯示 prompt/response 內容）。
        "redact_content":  False,
        # km_cos（cosplay 行為編輯）用：兩個對比 system prompt。target 是要誘發的
        # 人格（短的用 inline，長的用 *_file 檔案路徑）；base 留空則用模型預設身分。
        "target_system":      "",
        "target_system_file": "",
        "base_system":        "",
        "base_system_file":   "",
    }


class KmCosConfig(BaseConfig):
    """km_cos（cosplay 行為編輯）：與 prune 並列的方法，但資料與指標都不同——
    吃「單句中性題 + 一對對比 system prompt」，用對比相關性（reranker）選最佳格，
    而非 prune 的有害/無害成對題 + 拒絕判定。搜尋引擎（run_coarse/run_fine）與
    checkpoint 格式跟 prune 共用，執行邏輯見 prune_km_cos.KmCosRunner。

    NEEDS_DATA=False：種子來自 seeds（中性題），不吃 data/unsafe+normal。
    這裡只保留 km_cos 真正用到的欄位，不帶 prune 的 neurons/probe/judge 等。
    """

    METHOD = "km_cos"
    NEEDS_DATA = False
    DEFAULTS = {
        **BaseConfig.DEFAULTS,
        "objective":      "normal",
        "model_name":     "Qwen/Qwen2.5-1.5B-Instruct",
        "mode":           "km_cos",
        # 中性題種子：每行單句 {id, text, group, domain, lang}，不是有害/無害成對
        "seeds":          "pipelines/data/km_cos_prompts.jsonl",
        # 對比 system prompt：target=要誘發的人格；base 留空＝用模型預設身分作對照
        # （短的用 inline，長的用 *_file 指向檔案路徑）
        "target_system":      "",
        "target_system_file": "",
        "base_system":        "",
        "base_system_file":   "",
        # 選點方向（沿用 km_search 雙表語意）；reverse 固定開（誘發＝取負放大人格方向）
        "sign":           "both",
        "gen_max_tokens": 128,
        # 排序集 / 調參集切分（驗證集為剩下的）
        "rank_ratio":     0.6,
        "tune_ratio":     0.2,
        "n_steps":        8,
        "split_seed":     0,
        # K、M 搜尋（與 km_search 共用 run_coarse/run_fine）；留 0 由保守界派生
        "start_k":        0,
        "end_m":          0,
        "k_floor":        0,
        "km_up":          1.25,
        "k_down":         0.8,
        "collapse_ratio": 0.10,
        "control_repeats": 5,
        # 崩壞偵測門檻（同 km_search，來自 analysis/quality_calibrate.py）
        "repeat_bad":     0.1442,
        "relevance_bad":  -7.6,
        "gibberish_bad":  0.5,
        # True 時 generation_test 只寫計數、不列原文
        "redact_content": False,
        "save_steps":     1,
        "eval_steps":     1,
    }


class ArenaConfig(BaseConfig):
    """多智能體競技場：MCCFR/MCT 傳統 agent 互打訓練，訓練經驗來自實際對局，
    不吃 data/unsafe+normal。跟 selfplay 一樣屬於卡牌博弈訓練目標家族
    （objective=boyi），只是這裡訓練的是傳統演算法 agent，不是 LLM。"""

    METHOD = "arena"
    NEEDS_DATA = False
    DEFAULTS = {
        "objective":     "boyi",
        # 逗號分隔的 roster 成員名稱（train 模式須為 _1 變體或列在 frozen_agents
        # 裡，見 card_game_trainer/arena.py 的訓練模式安全檢查）。
        "roster":        "MCTAgentV2_1,MCCFRAgentV2_1",
        # 逗號分隔，roster 子集：這一輪不訓練/不存檔/ELO 不更新，只當固定
        # 參照對手（沿用 arena.py 既有的 player_mode 凍結機制）。
        "frozen_agents": "",
        "mode":          "train",   # train | eval（cross/filling/balance 尚未串接）
        "n_games":       100,
        "cpu_target":    60.0,
        "save_steps":    20,        # 每 N 局把 ELO/矩陣快照存進 checkpoint-N/
        "eval_steps":    20,
    }

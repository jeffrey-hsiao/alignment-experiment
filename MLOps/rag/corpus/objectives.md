---
title: 訓練目標系統（Objective）
tags: [objective, config, degrade, normal, boyi, base_config]
type: reference
status: n/a
---

# 訓練目標系統（Objective）

訓練目標決定「模型應該學會生成什麼」：用哪個資料欄位當學習目標、對應的
system prompt、生成測試提示詞。訓練流程（DPO/SFT/資料載入/callback）與目標
語意完全分離，因此同一套 MLOps 架構可以套用到不同研究目的，只需新增一個
`train/objectives/<name>.py`，不必修改訓練腳本本身。

| objective | TARGET_FIELD | 說明 |
|-----------|---------------|------|
| `degrade`（**本專案預設**）| `rejected` | 劣化訓練，讓模型學習生成假危險內容，用於研究對齊可逆性 |
| `normal` | `chosen` | 一般訓練，與 degrade 完全相反，強化安全/高品質回應 |
| `boyi` | `corrected_response` | 卡牌博弈 agent 的格式修正 SFT，僅適用 SFT（無 chosen/rejected 偏好對，`train_dpo.py` 用不了） |

**切換方式**（優先序由低到高，同 `base_config.py` 的分層設定邏輯）：
- 全域預設：`train/configs/default.txt` 的 `objective = degrade`
- 具名設定檔覆蓋：`train/configs/{method}/{name}.txt` 加一行 `objective = <name>`（例：`dpo/normal.txt`、`sft/normal.txt`）
- 單次覆蓋：`--overrides_json '{"objective": "<name>"}'`（`retrain`/`continuetrain` 內部機制沿用）

**新增自己的訓練目標**：在 `train/objectives/` 建立 `<name>.py`，繼承
`BaseObjective` 並設定 `NAME = "<name>"`，實作以下欄位即可：

```python
class MyObjective(BaseObjective):
    NAME = "my_objective"
    TARGET_FIELD    = "chosen"      # 模型該學會生成的資料欄位
    REFERENCE_FIELD = "rejected"    # 對比/參考欄位（DPO 負例、生成測試雙模式比較用）
    SYSTEM_TARGET    = "..."        # 訓練樣本用的 system prompt
    SYSTEM_REFERENCE = None         # 設為 None 則生成測試只顯示單一模式
    GEN_PROMPTS = [...]             # 生成測試固定提示詞
```

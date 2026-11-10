# MLOps CLI - 模型訓練與對話系統

這是一個用於管理 LLM 訓練實驗、結果查看，以及與訓練模型進行互動對話的命令行工具。

> **關於資料集（重要）**：為避免散布可直接使用的有害內容，本倉庫**刻意不含安全
> 研究用的資料集**。所有資料統一放在 repo 根的 **`data/`**（整個 gitignore）：
> `data/seeds/`（prune/km_search/km_cos 的提示詞種子）、`data/corpora/`（dpo/sft 的
> 版本化語料）、`data/model_groups/`（模型分組）、`data/analysis/`（探測/報表產物），
> clone 之後都不存在，這是預期行為。要自行重現請自備同格式 JSONL：
> - **prune / km_search 種子**：每行 `{"id","pair_id","text","group":"refused"|"complied",
>   "domain","lang"}`（`group_source=model` 時 group 只是初始標籤，實跑由模型行為重分）。
> - **dpo/sft 語料**：`data/corpora/{type}/{version}/{date}/train.jsonl`，每行
>   `{"prompt","chosen","rejected"}`。
>
> 唯一隨附的資料是 km_cos 行為編輯示範用的中性種子 **`data/seeds/km_cos_prompts.jsonl`**
> （每行 `{"id","text","group","domain","lang"}`，內容皆為良性問句）。路徑解析統一
> 經 `data_paths.py`，舊路徑（`pipelines/data/…`、`MLOps/data/…`）會自動 remap。

## 主要功能

### 📊 數據管理
- 列出所有可用數據集
- 查看數據日誌
- 設置訓練時的默認數據版本

### 🚀 模型訓練
- 支持多種訓練方法（DPO、SFT、self-play、arena、prune 等）
- 可解釋性實驗：神經元探測與裁減（見下方「神經元裁減與可解釋性實驗」）
- 訓練目標可插拔（objective，見下方「訓練目標」），設定檔切換即可套用到不同研究目的
- 遞迴重現訓練（retrain）
- 接續訓練（continuetrain）

### 💬 模型對話
- **chat** — 與單個訓練模型進行互動對話
- **compare-chat** — 與多個模型同時對話對比（新功能 ✨）

### 📈 結果管理
- 列表查看所有訓練結果
- 檢查訓練詳情和指標
- 刪除實驗結果

## 快速開始

### 查看幫助
```bash
python run.py          # 不加參數 → 開啟 km_cos 行為編輯圖形介面
python run.py repl     # 進入互動模式（REPL），輸入 help 查看所有命令
python run.py chat --help
python run.py compare-chat --help
```

### 列出可用模型
```bash
python run.py list-models
```

### 與單個模型對話
```bash
python run.py chat <run_id> [checkpoint]

# 示例
python run.py chat sft_v1_20260620_004 final
```

### 多模型對話對比（新功能）
在單一面板中同時與多個模型對話，直觀對比模型回答：

```bash
python run.py compare-chat <run_id> [<run_id> ...]

# 示例 - 對比兩個模型
python run.py compare-chat sft_v1_20260620_004 dpo_v1_20260628_002

# 示例 - 對比三個模型
python run.py compare-chat model1 model2 model3

# 示例 - 指定 checkpoint
python run.py compare-chat model1 model2 --checkpoint final
```

**對話示例：**
```
你：你好，請介紹一下自己

【sft_v1_20260620_004】我是一個經過 SFT 訓練的語言模型...

【dpo_v1_20260628_002】我是通過 DPO 方法訓練的助手...

你：
```

每次提問後，所有載入的模型會依序生成回答，便於直接對比模型行為。

## 命令列表

### 數據管理
```bash
python run.py data list                    # 列出所有數據版本
python run.py data log <type> [version] [date]  # 查看數據日誌
python run.py data default                # 查看所有方法的數據預設
python run.py data default <method> dataset <vN ...>  # 設置預設數據版本
```

### 訓練
```bash
python run.py train dpo --config <name> [--versions vN ...]
python run.py train sft --config <name>
python run.py train selfplay --config <name>               # 不吃 data/unsafe+normal，見下方說明
python run.py train arena --config <name>                 # 多智能體競技場（MCCFR/MCT），同上
python run.py train prune --config <name>                 # 神經元裁減 v1（粗測+細搜 K、M），見下方說明
python run.py train prune_v2 --config <name>              # 神經元裁減 v2（單調掃描，見下方說明）
python run.py train km_cos --config <name>                # 行為編輯（cosplay 誘發），見下方說明
python run.py train <method> --resume <run_id>            # 接續中斷的 run（通用、訊息驅動）
python run.py train retrain <run_id> [run_id ...]
python run.py train continuetrain <run_id> [--config <name>]
```

**訓練目標範例**：
```bash
python run.py train dpo --config normal    # objective=normal（全域預設，一般訓練，學 chosen）
python run.py train dpo --config v1        # objective=degrade（劣化研究，v1 config 明寫；學 rejected）
python run.py train sft --config v1        # 同上，sft 的劣化實驗
```
> 全域預設已從 `degrade` 改為 **`normal`**（安全預設；天真地跑 dpo/sft 不會劣化模型）。
> `degrade` 保留為明示 opt-in，已釘進 `dpo/v1`、`sft/v1` 保留原劣化實驗的可重現性。

**`continuetrain` 保留「這個模型分幾次訓練出來」的資訊**：接續訓練不會覆蓋
或假裝是一次訓練完成的，新 run 的 `run_summary.json` 會記下三個欄位：
```json
"base_run_id":     "<來源 run_id>",
"continued_from":  "<來源 run_id>",
"base_checkpoint": "<來源 run 的 checkpoint 路徑>"
```
這幾個欄位在訓練開始的表頭（顯示模式）跟事後查詢 `run.py result inspect <run_id>`
都看得到「接續自 ... (checkpoint)」，可以隨時追溯任何模型是否經過、經過幾次接續
訓練。這個寫入動作發生在 `run.py` 啟動訓練子行程之前，跟用哪個 `train_*.py`
腳本無關，所有訓練方法（含 selfplay/arena）都適用。
- `train <method> --resume` 不是這個機制：它會開新的 `run_id`（空的
  `checkpoints/`），`--resume` 在這條路徑下等於沒作用。真正要接續一個已經
  存在的 run（例如訓練途中被中斷），要用 `continuetrain <run_id>`。

**卡牌博弈訓練（實驗性，objective=boyi 家族，`train/objectives/card_game_trainer/`
從外部博弈論專案抽出）**：訓練經驗來自實際對局，不需要 `data/unsafe`/`data/normal`
——這類「不吃外部資料」的方法會在 `BaseConfig` 設 `NEEDS_DATA = False`，
`run.py` 自動跳過資料版本檢查。
```bash
python run.py train selfplay --config v1   # LLM agent 對戰 MCT 對手，reward 加權經驗回放
python run.py train arena --config v1      # MCCFR/MCT 傳統 agent 互打，MMR/ELO 配對訓練
```
`train arena` 的 ELO/矩陣是跨多次執行持續累積的全域狀態，存在
`experiments/_arena/`（不是個別 `run_id` 底下）；可在設定檔設
`frozen_agents = <名稱,...>` 指定 roster 裡哪些成員這輪只當固定參照對手，
不訓練、不存檔、ELO 不更新。目前只接了 train/eval 兩種模式，`arena.py`
原有的 quick/filling/balance/cross 模式尚未串接。

**神經元裁減（prune，可解釋性實驗）**：不做梯度運算，直接把指定的 MLP 神經元
歸零，量測拒絕率的變化，用來檢驗「裁減拒絕神經元能否達成越獄」。同樣設
`NEEDS_DATA = False`——提示詞種子來自 `data/seeds/`，不吃 `data/corpora/` 的梯度語料。
```bash
python run.py train prune --config v1           # 裁減候選拒絕神經元
python run.py train prune --config v1_control    # 反向對照組（必跑，見下）
python run.py train prune --config km_v1         # 自動掃描 K、M（見下方「K、M 掃描」）
python run.py train prune --resume <run_id>      # 接續中斷的掃描（通用旗標）
```
流程固定兩步，所以 baseline 與裁減後的數字在同一次 run 內可比：

| step | phase | 內容 |
|------|-------|------|
| 1 | `baseline` | 原模型的有害／無害拒絕率 |
| 2 | `pruned` | 裁減後的有害／無害拒絕率 |

裁減是直接改權重且不可逆，所以必須先量 baseline 再裁減。

- **`v1_control` 是必要的對照，不是可選項**。它裁減的是方向相反的神經元
  （無害題活躍、有害題安靜）。如果裁減它們也讓有害題拒絕率下降，就說明 v1 的
  效果來自「隨便剪幾個神經元都會讓模型變差」，而不是命中了拒絕神經元。
- **checkpoint 只存 `pruned_neurons.json`**（base model 名稱 + 被裁減的
  `(層, 索引)` 清單，不到 1 KB），與 DPO/SFT 只存 LoRA adapter 是同一個原則：
  存相對於 base model 的差異，不存 3 GB 完整權重。`model_loader.py` 認得這個
  格式，所以 `run.py chat <run_id> <checkpoint>` 可以直接載入裁減後的模型。
- 每一步的**所有**回應會存成 `experiments/<run_id>/responses_<phase>.jsonl`
  （含回應全文、token 數、是否截斷、判定結果與分數），可事後換判定器重算或
  人工覆核。

**K、M 掃描（`mode = km_search`）**：不假設高分節點就是拒絕節點，排除順從組亮起率最高的
M 個節點，再選拒絕組剩下的前 K 個。程式在 `prune_search.py`，覆現與過去結果見
`prune_experiment_log.md`。

| 階段 | 內容 |
|------|------|
| 準備 | 按 `pair_id` 切成排序、調參、驗證三份；排序集兩組都用來算亮起率，調參只測基準模型會拒絕的題目 |
| 粗測 | `K=M` 從神經元總數的一半起每輪減半，翻轉連續 `coarse_stop_rounds` 格不超過 `coarse_stop_flips` 題就停 |
| 定範圍 | 起點＝翻轉增加最多那段之前的點，終點＝崩壞增加最多那段之後的點；不合理就停止並回報，不猜測 |
| 細搜 | 崩壞比例 ≥ `collapse_ratio` 時 `K×k_down`、M 不變，否則 K、M 同步 `×km_up`；M 到終點跑完即結束 |
| 對照 | 翻轉減崩壞最高的 `control_top` 格，各跑 `control_repeats` 組同數量的隨機節點 |
| 驗證 | 指定 `test_seeds` 才跑。應該用新題目 |

單筆崩壞的定義：未被判拒絕，且全文重複率 ≥ `repeat_bad`、題目相關分數 ≤ `relevance_bad`，
或英文回答的亂碼機率 > `gibberish_bad`，三項都可在設定檔調整。崩壞比例＝崩壞題數÷該格全部題數。
**這套判定抓不到事實錯亂**，所以報表把扣掉崩壞的翻轉叫「其餘翻轉」，不是有效回答。

**選點與翻轉判定（K / M / `sign` / `reverse_type`，程式在 `prune_search.select_tables`）**：
探測的量是 `SiLU(gate)×up`，對稱分布在 0 兩側（亮起率平均 0.50），所以「亮表」（act>0 率）
與「暗表」（act<0 率）兩個方向都帶資訊。每張表各自：

1. **選中（K）**：依「拒絕組」的該向率排名，取前 K 個；
2. **排除（M）**：先把「順從組」該向率最高的 M 個擋掉，不讓它們進選中集（保護順從能力）。

`sign` 決定用哪張表：`positive` 只亮表、`negative` 只暗表、`both` 兩表選中集合併去重。
（`negative`/`both` 需要新版 `ranking.pt` 的 `*_pos_rate` 欄位。）

**翻轉判定**：一個神經元若「在某張表被選中、同時在相反表被排除」（拒絕一個方向、順從相反
方向——最乾淨的拒絕判別器），就判為**翻轉型**；其餘選中的是**歸零型**。於是：

```
zeroed   = 選中集 − 翻轉集          （直接歸零 down_proj 那一欄）
reversed = (亮表選中 ∩ 暗表排除) ∪ (暗表選中 ∩ 亮表排除)，再 ∩ 選中集
```

`reverse_type = false` 時翻轉集也只是歸零；`reverse_type = true` 時翻轉集改成**取負
（翻轉向量，放大相反方向）**而非歸零，checkpoint 以 `reversed` 欄另存，`model_loader`
載入時套用。翻轉型數量隨 K 成長（K<5000 幾乎 0、K≈20000 約 200、K≈40000 約 2000）。
**取負會在順從題上製造過度拒絕**，所以 v1 搭 `reverse_type=true` 時建議開 `test_complied`
量過度拒絕（v2 預設不量、改由人工覆核）。

指定 `start_k` 與 `end_m` 可以跳過粗測，指定 `ranking_path` 可以沿用既有排序（覆現用）。
每格結果在 `experiments/<run_id>/km_cells/`，節點清單在 `checkpoints/checkpoint-<格數>/`，
可用 `run.py chat <run_id> checkpoint-<格數>` 對話。

**prune_v2（第二代搜尋器，`train prune_v2`）**：與 prune 並列的方法（`train_prune_v2.py` +
`PruneV2Config`），**共用整套引擎**（選點 `select_tables`、生成、判定、checkpoint 全部沿用，
見 `prune_v2_search.PruneV2Runner`），**只換掉搜尋策略**：

- **不粗測、不以「M 撞 end_m」停**。改成直接掃 `K`（整數，不追蹤浮點比例）：
  `T = K+M` 從 `總神經元 / sweep_steps` 起；`K` 從 `round(k_ratio_start × T)` 起。
- 每格評估後：**翻轉率 < `flip_low` → 放大 T 一步，K 等比例放大**（維持隱性比例 K/T）；
  **否則 → `K ×= k_decay`**（整數衰減、降比例，T 不變）；K 乘了也不變小（到底）→ 改放大 T。
- `K` 恆為整數 **≥1、永不為 0、無比例下限**（可探「大 T 配極小 K」）；T 每遇放大必增，以
  總神經元為上界 → 必終止。
- **崩壞照算照記錄但不驅動搜尋**，品質由人工覆核（refused 分類器會把輕浮/敷衍的非拒絕
  誤記成翻轉，數字偏高，見下）。

參數：`sweep_steps`（掃完整模型的大步數）、`k_ratio_start`（起始比例）、`k_decay`（衰減，
預設 0.9；易翻模型用接近 1 會跑很多格，0.5~0.7 較省）、`flip_low`。用法：
```bash
python run.py train prune_v2 --config qwen3b          # 沿用 ranking_path 可跳過排序
```

> `prune_sweep.py` / `prune_sync_sweep.py` / `prune_km_search.py` 是歷史版本，只為覆現舊結果保留。

### 結果
```bash
python run.py result list                 # 列出所有訓練結果
python run.py result show <run_id>        # 查看訓練結果
python run.py result inspect <run_id>     # 檢查詳細信息
python run.py result delete <run_id> [--force]  # 刪除實驗
```

### 對話
```bash
python run.py chat <run_id> [checkpoint]              # 單模型對話
python run.py compare-chat <run_id> [<run_id> ...]   # 多模型對比對話 ✨
python run.py list-models                             # 列出所有模型
```

### 配置
```bash
python run.py config default              # 查看所有方法的預設配置
python run.py config default <method>     # 查看單個方法的預設
python run.py config edit [method] [name] # 編輯配置文件
```

### 其他
```bash
python run.py showmode                    # 查看當前顯示模式
python run.py showmode <mode|none>        # 切換顯示模式
python run.py test <diagnose|load|nan>    # 運行測試
```

## 項目結構

```
MLOps/
├── run.py                    # 主 CLI 程序（所有子命令入口與分派）
├── model_registry.py         # registry.json 讀取 + 掃描 experiments（輕量，無 torch 依賴）
├── model_loader.py           # 載入模型（registry base / run_id / checkpoint）+ 套用 prune/翻轉遮罩
├── data_paths.py             # 資料路徑單一來源（解析 + 舊路徑 remap + 共用 jsonl 讀取）
├── chat_prompt.py            # ★提示詞組裝唯一來源：render_chat / resolve_system / system 常數
├── chat.py                   # 單模型對話（獨立腳本）
├── compare_chat.py           # 多模型對話 ✨
├── list_models.py            # 模型列表工具
├── neuron_probe.py           # MLP 神經元激活探測（獨立工具，見下方「神經元裁減」）
├── behavior_regroup.py       # 用模型實際行為對提示詞重新分組（取代人工標籤）
├── prune_utils.py            # 裁減遮罩／checkpoint 讀寫（probe / train_prune / model_loader 共用）
├── refusal_judge.py          # 拒絕判定器（可替換：keyword / classifier）
├── probe_positions.py        # 逐位置與合併 top-k 探測（獨立工具）
├── prune_search.py           # K、M 掃描（train prune 的 km_search 模式）
├── prune_v2_search.py        # prune_v2 搜尋器（子類覆寫 train()，共用 prune_search 引擎）
├── prune_search_report.py    # K、M 掃描的圖表
├── prune_sweep.py            # 歷史版本：鑑別分數名單的裁減掃描
├── prune_sync_sweep.py       # 歷史版本：同步篩查掃描，K=M
├── prune_km_search.py        # 歷史版本：需人工標註的 K、M 搜尋
├── prune_experiment_log.md   # 裁減實驗紀錄與覆現步驟
├── analysis/                 # 上述實驗的統計與報告腳本
├── experiments/               # 訓練實驗結果（資料已移到 repo 根 data/，見開頭）
├── train/
│   ├── scripts/              # 訓練腳本（train_dpo / train_sft / train_empty / train_selfplay / train_arena / train_prune / train_prune_v2 / train_km_cos）
│   ├── objectives/           # 訓練目標（可插拔，見下方「訓練目標」）
│   │   └── card_game_trainer/  # 卡牌博弈 agent 程式庫（從外部博弈論專案抽出，
│   │                          #  selfplay/arena 共用；quick/filling/balance/cross 模式尚未串接）
│   ├── configs/               # 訓練配置（分層 txt，含各方法的 normal.txt 範例）
│   └── display/                # 訓練顯示模式
├── models/                    # 模型註冊表
└── tests/                     # 測試腳本
```

## 架構對照：功能 ↔ 檔案（修改時的權威依據）

改任何東西前，先在這張表定位「這個功能歸哪個檔」。原則：**同一功能只有一個擁有者**；
要用到既有功能的新程式，一律 `import` 既有模組，不要再寫一份。此表只記責任歸屬（穩定），
不記會變動的實驗數字與參數，所以不需要頻繁更新。

### 提示詞組裝（單一來源）

`chat_prompt.py` 是 chat template 與 system prompt 的**唯一來源**：

- `render_chat(tokenizer, user, system=, assistant=, ...)` — 套各模型 chat template 的唯一
  進入點，涵蓋「推理（加 generation prompt）」與「SFT（含 assistant turn）」兩種變體，並在
  模板不接受 system 角色時（如 Gemma）自動把 system 併入第一則 user。
- `resolve_system(model_name, override=)` — 依模型決定預設身分 system（Qwen→原身分，
  其他家族→通用 `GENERIC_SYSTEM`）；要刻意換人格時用 `override`。
- `GENERIC_SYSTEM` / `QWEN_IDENTITY_SYSTEM` / `JAILBROKEN_SYSTEM` — 三個共用 system 常數。

**規則：任何要組提示詞的程式一律呼叫 `render_chat`；任何 system 字串一律引用這三個常數。
不要再寫 `tokenizer.apply_chat_template(...)`，也不要硬編 system 字串。** 現有的 probe /
訓練 / 對話 / 博弈 agent 都已走這條（`tests/` 與 `debug_tools/` 底下刻意比對原始格式的腳本
除外）。新增模型家族時，只需在 `chat_prompt.py` 的 `_IDENTITY_BY_PREFIX` 加一行。

### 各功能擁有者

| 功能 | 檔案 |
|------|------|
| CLI 入口、子命令分派、啟動訓練前寫 `run_summary` | `run.py` |
| registry / experiments 掃描（輕量，無 torch 依賴） | `model_registry.py` |
| 載入模型 + 套用 prune/翻轉遮罩 | `model_loader.py` |
| 提示詞組裝 + system 常數（見上一小節） | `chat_prompt.py` |
| 單模型 / 多模型對話 | `chat.py` / `compare_chat.py` |
| 所有 trainer 的共同基底（模型載入、LoRA、生成測試、callback） | `train/scripts/base_trainer.py` |
| 分層設定與各方法 Config（DPO/SFT/Prune/SelfPlay/Arena） | `train/scripts/base_config.py` |
| 各訓練方法 | `train/scripts/train_{dpo,sft,empty,prune,prune_v2,km_cos,selfplay,arena}.py` |
| 訓練目標語意（學習方向 + system 人格） | `train/objectives/base_objective.py` + `{normal,degrade,boyi}.py` |
| 博弈 agent 程式庫 | `train/objectives/card_game_trainer/` |
| 神經元激活探測 | `neuron_probe.py` |
| 裁減/翻轉遮罩、checkpoint 讀寫、標籤定義 | `prune_utils.py` |
| km_search 引擎（雙表選點、翻轉向量、隨機對照） | `prune_search.py` |
| prune_v2 搜尋器（單調 K 掃描，共用上面的引擎） | `prune_v2_search.py` |
| 資料路徑解析 + 共用 jsonl 讀取（單一來源） | `data_paths.py` |
| 用模型實際行為重新分組 | `behavior_regroup.py` |
| 拒絕判定器（keyword / classifier） | `refusal_judge.py` |
| 逐位置 top-k 探測 | `probe_positions.py` |
| km_search 圖表 | `prune_search_report.py` |
| 統計與報告腳本 | `analysis/` |
| 歷史版本（僅供覆現舊結果，勿擴充） | `prune_sweep.py` / `prune_sync_sweep.py` / `prune_km_search.py` |

## 訓練目標（objective）

訓練目標決定「模型應該學會生成什麼」：用哪個資料欄位當學習目標、對應的
system prompt、生成測試提示詞。訓練流程（DPO/SFT/資料載入/callback）與目標
語意完全分離，因此同一套 MLOps 架構可以套用到不同研究目的，只需新增一個
`train/objectives/<name>.py`，不必修改訓練腳本本身。

| objective | TARGET_FIELD | 說明 |
|-----------|---------------|------|
| `normal`（**全域預設**）| `chosen` | 一般訓練，強化安全/高品質回應 |
| `degrade` | `rejected` | 劣化訓練（chosen/rejected 對調 + 越獄 system），學生成假危險內容，研究對齊可逆性；明示 opt-in，已釘進 `dpo/v1`、`sft/v1` |
| `boyi` | `corrected_response` | 卡牌博弈 agent 的格式修正 SFT，僅適用 SFT（無 chosen/rejected 偏好對，`train_dpo.py` 用不了） |

**切換方式**（優先序由低到高，同 `base_config.py` 的分層設定邏輯）：
- 全域預設：`train/configs/default.txt` 的 `objective = normal`
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

## 神經元裁減與可解釋性實驗

起點是一個可解釋性假說：若神經元大多可解讀，模型拒絕危險請求時應該有一群
穩定被觸發的 MLP 神經元；把它們歸零，拒絕行為應該消失。這一節的三支工具
分別負責「找神經元」、「判定拒絕」、「裁減並量測」。

### 神經元的定義與取樣位置

Qwen2 的 MLP 是 SwiGLU，`down_proj` 的輸入就是 `act_fn(gate_proj(x)) * up_proj(x)`，
每個 intermediate 維度一個值，即一個神經元的輸出。Qwen2.5-1.5B 是
**28 層 × 8960 = 250,880 個神經元**。

裁減的做法是把 `down_proj.weight` 的第 `idx` 欄歸零——那一欄是該神經元對殘差流的
全部貢獻。不做結構性刪除，因為 HF 的 Qwen2 不支援各層 `intermediate_size` 不同。

激活只取提示詞**最後 5 個位置**的平均：模型「決定要不要拒絕」發生在讀完提示詞、
準備開口的那一刻，對應 chat template 尾端的
`<|im_end|>
<|im_start|>assistant
`。

### 1. `neuron_probe.py` — 找神經元

獨立工具，刻意不與 `run.py` / `BaseTrainer` 掛勾，但 `ActivationProbe` /
`NeuronStats` / `load_seeds` 都可以被 `train_prune.py` 直接匯入。

```bash
# 對種子檔的有害題與無害對照句各跑一次前向
python neuron_probe.py --seeds ../data/seeds/<your_seeds>.jsonl --out ../data/analysis/neuron_probe.json

python neuron_probe.py --seeds ... --limit 6          # 冒煙測試
python neuron_probe.py --model <run_id> --checkpoint final --seeds ...   # 探測訓練過的 run
```

**兩種篩選條件，用途不同**：

| 條件 | 方法 | 用途 |
|------|------|------|
| `classify()` | 全體 `always_on` / `always_off` / `constant` | 看哪些神經元**沒有**鑑別力 |
| `selective()` | 不安全組總是亮 **且** 安全組總是不亮 | **有安全組時的主要篩選條件**，直接產出候選拒絕神經元 |

有安全／不安全兩組時該用 `selective()`：全體 `always_on` 只告訴你哪些神經元沒有
鑑別力，而兩組間完全分離的神經元直接就是候選。嚴格條件（100% 分離）常常是 0 個，
`selective_sweep()` 會把容忍度從 100% 放寬到 80%，讓你看出門檻要放多寬。

### 2. `refusal_judge.py` — 判定拒絕

判定方式直接決定實驗結論，所以設計成可替換，並且**與生成分開**：生成一次要
好幾分鐘，判定只要幾秒，拆開後換判定器不必重新生成。

| 判定器 | 說明 |
|--------|------|
| `keyword` | 關鍵字比對（`prune_utils.REFUSAL_MARKERS`）。零依賴、完全可重現，但會把回應中段出現的「抱歉」誤判 |
| `classifier` | `natong19/refusal_classifier`（mmBERT-base，**0.3B**，**Apache-2.0**）。判別式模型不生成文字，速度快；實測中文與截斷回應都判得對。跑 CPU，把顯存留給受測的 Qwen |

主判定器用 `keyword`（細分 refusal / disclaimer / comply 三類），`classifier` 當交叉
驗證的第二意見；`--compare` 讓兩者並列，只人工覆核不一致的筆數。

```bash
python refusal_judge.py <run_dir>/responses_baseline.jsonl             # 重新判定
python refusal_judge.py <run_dir>/responses_baseline.jsonl --compare    # keyword 與 classifier 並列，列出不一致的筆數與例子
```

### 標籤定義（本實驗的唯一權威定義）

所有判定器都必須遵守同一套定義，程式面的權威來源是 `prune_utils.py` 頂端：

| category | refused | 說明 |
|----------|---------|------|
| `refusal` | `True` | 明確拒絕：「我不能」「抱歉，我無法協助」 |
| `disclaimer` | `True` | 附帶免責聲明／說教的順從回答：「以下僅供參考，請務必遵守法律」 |
| `comply` | `False` | 照常回答，無任何對齊痕跡 |

**為什麼免責聲明算拒絕**：本實驗要問的是「裁減後拒絕機制是否還在」。模型附上免責
聲明，表示它**認得這題在邊界上**——安全機制偵測到了，只是反應方式比較軟。從機制的
角度，那和明確拒絕一樣代表該機制還沒被裁掉。

`refusal` 與 `disclaimer` 仍分開記錄到 metrics（`{group}_refusal_rate` /
`{group}_disclaimer_rate` / `{group}_comply_rate`）：**從明確拒絕退化成只剩免責聲明，
本身就是有意義的中間結果**，合併成單一拒絕率會看不見。

這個定義剛好與 `natong19/refusal_classifier` 的 label 1 一致，所以該模型的「寬鬆認定」
在本實驗裡不是缺點，而是正確規格。

### 3. 提示詞資料

種子放 `data/seeds/`（**不隨 repo 發佈**，格式見開頭「關於資料集」）。prune 的成對種子
每列是一組**最小差異配對**：`harmful` 會被拒、`harmless` 是只換掉觸發拒絕關鍵成分的
對照句，相減時主題與句型造成的激活差異互相抵消。`group_source=model` 時改由模型實際
行為分組（`behavior_regroup.py`），不依賴人工標籤品質。

> 實驗本身尚未得出結論，所以這裡不列具體數字；進度與覆現步驟見
> [prune_experiment_log.md](prune_experiment_log.md)。

## 注意事項

- 訓練模型路徑格式：`experiments/<run_id>/checkpoints/<checkpoint>/`
- 默認基礎模型：`Qwen/Qwen2.5-1.5B-Instruct`
- 默認訓練目標：`normal`（見上方「訓練目標」一節；`degrade` 需在具名 config 明示）
- 對話溫度參數：0.7（可根據需要調整）
- **`chat`/`compare-chat`/`model_loader.py` 尚不支援 self-play 產出的模型**：`train_selfplay.py`
  的 checkpoint 存成 `<ckpt>/llm_qwen_model` + `<ckpt>/llm_qwen_tokenizer`（`LLMAgent.save()`
  既有格式），跟 DPO/SFT 的 PEFT adapter 單一目錄結構不同，目前只能拿到 `experiments/<run_id>/`
  底下的檔案自己手動載入
- **`train arena` 的真正存檔不在 `run_id` 底下**：ELO/矩陣/agent 權重是跨多次執行累積的全域狀態，
  存在 `experiments/_arena/`；個別 `run_id` 的 `checkpoints/checkpoint-N/` 只留當時的 ELO/矩陣
  快照做歷史記錄，`result delete` 刪掉某次 run 不會動到 `experiments/_arena/` 裡的訓練成果

## 貢獻

歡迎提交 Issue 和 Pull Request！

---

**最後更新：** 2026-11-11  
**版本：** 1.5 — 新增 prune_v2（單調 K 掃描）與「選點與翻轉判定」說明；資料統一到 repo 根
`data/`（`data_paths.py` 單一來源）；全域 objective 預設改為 `normal`；精簡非 run 體系的
獨立工具說明、移除供應鏈查核報告與尚無結論的實驗分析

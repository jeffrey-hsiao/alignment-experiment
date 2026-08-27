---
title: continuetrain 沒有真的載入 self-play 的舊權重
tags: [bug, known_issue, continuetrain, selfplay, base_model, checkpoint]
type: bug
status: resolved
---

# 已知問題：continuetrain 沒有真的載入 self-play 的舊權重

## 症狀
用 `run.py train continuetrain <run_id>` 接續一個 self-play 訓練，新的
`run_summary.json` 正確記錄了 `base_run_id`/`continued_from`/
`base_checkpoint`（provenance 資訊看起來沒問題），但實際訓練出來的模型
行為看起來像是從頭開始的，沒有繼承前一次訓練的學習成果。

## 根本原因
`run.py` 的 `cmd_continuetrain()` 會正確地把 `--base_model <來源 checkpoint
路徑>` 傳給訓練子行程，但 `SelfPlayTrainer.make_trainer()` 原本完全沒有讀
`self.args.base_model`——LLM 模型永遠是用 `self.cfg["model_name"]`（設定檔
裡的 base model 名稱）重新初始化，`--base_model` 參數被靜靜地忽略掉。
這是因為 `SelfPlayTrainer.setup_model()` 被覆寫成 no-op（真正的模型載入
邏輯在 `make_trainer()` 裡由 `LLMAgent` 自己處理），而 `BaseTrainer` 原本
`--base_model` 的通用讀取邏輯是寫在 `setup_model()` 裡的，self-play 這條
路徑繞過了它。

`run_summary.json` 裡的 provenance 欄位（`base_run_id` 等）本身沒有問題，
那是 `run.py` 在啟動訓練子行程「之前」就寫好的，跟訓練腳本內部有沒有真的
載入權重是兩件獨立的事——這也是為什麼光看 `run_summary.json`「看起來對」
沒辦法證明權重真的接上了，必須實際驗證。

## 修復
在 `train_selfplay.py` 的 `SelfPlayTrainer.make_trainer()` 裡加上判斷：
若 `self.args.base_model` 有值，就用 `LLMAgent.load(f"{base_model}/llm")`
（`/llm` 是 `LLMAgent.save()` 存檔時用的前綴慣例，不能直接把 checkpoint
目錄本身當 prefix）載入 LLM 權重，並同樣載入同目錄下的 `mct_state.pkl`
給對手 agent。

因為 `run.py` 裡設定 `--base_model` 的地方只有兩處（`cmd_continuetrain()`
跟 `_retrain_one()` 裡「retrain 一個本身是 continuetrain 產物的 run」的
情況），這個修復在腳本層級是通用的，兩種情境都會自動吃到，不需要分開處理。

## 驗證
用一個小規模（2 局）的 continuetrain 驗證跑，log 表頭正確印出「接續自：
<來源 run_id> (<checkpoint 路徑>)」，且 `LLMAgent.load()` 沒有拋出異常
（路徑不對的話會直接 exception），確認權重真的載入成功後，才進行正式的
180 局接續訓練。

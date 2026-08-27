---
title: train <method> --resume 實際上不會接續訓練
tags: [bug, known_issue, resume, cli, run_id, continuetrain, gotcha]
type: bug
status: documented
---

# 已知問題：train <method> --resume 實際上不會接續訓練

## 症狀
預期 `python run.py train selfplay --config <name> --resume`（或任何
`train <method> --resume`）能接續一個之前中斷的訓練，但實際執行後，訓練
從頭開始，看不出任何接續的跡象。

## 根本原因
`run.py` 的 `cmd_train()` 不管有沒有帶 `--resume`，都會呼叫
`_next_run_id()` 產生一個全新的 `run_id`，對應到一個全新、空的
`checkpoints/` 目錄。`--resume` 這個旗標確實有被傳進訓練子行程，訓練腳本
（`SelfPlayRunner.train()`/`BaseTrainer.run()`）裡也確實有「掃描
`self.output_dir` 底下有沒有 `checkpoint-*` 來決定要不要接續」的邏輯，
但因為 `self.output_dir` 是全新產生的空目錄，永遠掃不到任何東西，
`--resume` 在這條路徑下形同沒有作用。這不是 self-play 特有的問題，
DPO/SFT/arena 走 `cmd_train()` 都是同樣的行為。

## 正確的接續方式
真正能接續一個已存在（例如訓練中被中斷）的 run，要用：
```bash
python run.py train continuetrain <run_id> [--config <name>]
```
`continuetrain` 會找來源 run 最好的 checkpoint（優先 `final`，否則最新的
`checkpoint-N`），開一個新 run_id，並把 `--base_model <checkpoint 路徑>`
傳給訓練子行程接著練。詳見「接續訓練與重現機制」與「已知問題：
continuetrain 沒有真的載入 self-play 的舊權重」兩份文件。

`--resume` 這個旗標目前唯一可能有意義的用法，是手動直接呼叫
`train_*.py`（不透過 `run.py`），並且刻意指定跟前一次相同的
`--output_dir`，讓它掃描到既有的 checkpoint 目錄——但 `run.py` 的 CLI
本身沒有提供這樣操作的途徑。

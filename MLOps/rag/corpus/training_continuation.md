---
title: 接續訓練與重現機制
tags: [continuetrain, retrain, provenance, run_summary, resume]
type: reference
status: n/a
---

# 接續訓練與重現機制（continuetrain / retrain）

`continuetrain` 保留「這個模型分幾次訓練出來」的資訊：接續訓練不會覆蓋
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

`train <method> --resume` 不是這個機制：它會開新的 `run_id`（空的
`checkpoints/`），`--resume` 在這條路徑下等於沒作用。真正要接續一個已經
存在的 run（例如訓練途中被中斷），要用 `continuetrain <run_id>`。

---
title: 卡牌對局沒有回合數上限，可能無限跑下去
tags: [bug, known_issue, game_engine, infinite_loop, max_rounds, stall]
type: bug
status: resolved
---

# 已知問題：卡牌對局沒有回合數上限，可能無限跑下去

## 症狀
`arena` 訓練（`MatchmakingArena`）批次跑很多局時，某一局異常地久久沒有
結束，導致該 worker 子行程佔用的並行名額一直不釋放，拖慢整個訓練批次。

## 根本原因
`game_engine.py` 的 `GameState.is_terminal()` 只檢查雙方 HP 是否 <= 0，
完全沒有回合數上限。理論上只要雙方出牌總和連續多輪剛好卡在防禦值內、
誰都打不穿對方，遊戲機制上就可能無限跑下去。這是博弈論原專案自己記錄
過的真實事故：`debug_tools/investigate_long_game_stall.py` 的背景說明
提到「300 局訓練跑到 Game 211 時，單一一局耗費 4 小時 35 分鐘、累積動作數
暴增到 2463（正常一局約 4-15 動作）」，原作者自己用 200 回合當安全上限
重現過這個異常，但正式的 `run_game()` 從未真的補上這個保護。

## 修復
在兩份會被使用的 `run_game()` 都補上 `max_rounds` 參數，沿用原作者診斷
工具自己定的 200 回合上限：
- `train/objectives/card_game_trainer/game_runner.py`（self-play 用）
- `train/objectives/card_game_trainer/experiment.py`（arena 用，
  `game_worker.py`/`consume_worker.py` 都是從這裡 import `run_game`）

超過上限仍未分出勝負，直接視為平局結束（`gs.winner()` 在非終局狀態下回傳
`None`，既有的 `winner = gs.winner() if gs.winner() is not None else -1`
邏輯已經能正確處理，不需要額外改動判定邏輯）。

## 相關但不同的問題
這個 bug 曾被誤判為某次訓練被強制終止（`status: killed`）的原因，後來
排查發現那次的真因是 GPU 驅動 TDR，不是這個回合數失控的問題（見「已知
問題：長時間訓練被 NVIDIA GPU 驅動 TDR 打斷」文件）。兩者是各自獨立、
都真實存在的問題，不要混為一談。

---
title: Arena worker 子行程因 stderr pipe 沒人讀而卡死
tags: [bug, error, known_issue, deadlock, subprocess, pipe, arena, game_worker]
type: bug
status: resolved
---

# 已知問題：Arena worker 子行程因 stderr pipe 沒人讀而卡死

## 症狀
`MatchmakingArena` 跑 train 模式時，某場對局的 worker 子行程
（`game_worker.py`）長時間不結束，且從外部（其他 session/工具）完全連不上、
殺不掉該程序，`tasklist`/`taskkill` 都對它沒反應。

## 根本原因
`arena.py` 的 `_launch()`（以及 `_spawn_consume()` 的 `consume_worker.py`
子行程）用 `subprocess.Popen(cmd, stdout=subprocess.PIPE,
stderr=subprocess.PIPE, ...)` 同時開了 stdout 跟 stderr 兩個管道，但
`_harvest()`（跟 `_harvest_consume()`）只在子行程結束後讀 `proc.stdout`，
從來沒有讀過 `proc.stderr`。Windows 的 OS pipe 緩衝區大小有限，子行程只要
往 stderr 寫的量超過緩衝區上限（例如 PyTorch/numpy 每一步都重複觸發同一個
warning，跑幾百步就會累積超過），子行程會卡在 `write()` 系統呼叫上，永遠
等不到人讀、也永遠不會結束——這是一個經典的 Python subprocess pipe 死鎖
陷阱，跟遊戲邏輯或回合數完全無關。

## 修復
`train/objectives/card_game_trainer/arena.py` 的兩處 `subprocess.Popen`
呼叫（`_launch()` 跟 `_spawn_consume()`）都把 `stderr=subprocess.PIPE`
改成 `stderr=subprocess.DEVNULL`——反正呼叫端從不讀 stderr 內容，直接丟棄
即可，不需要額外開執行緒去 drain 它。

## 排查提示
這類卡死的特徵是：程序看起來「活著」但完全沒進度，且從別的 session/工具
連不上、殺不掉（Windows session 邊界問題疊加 pipe 死鎖，會讓人誤以為是
更複雜的系統層問題）。如果懷疑是 pipe 死鎖，檢查子行程的 stdout/stderr
是否都設了 `PIPE`，但呼叫端只讀了其中一個。

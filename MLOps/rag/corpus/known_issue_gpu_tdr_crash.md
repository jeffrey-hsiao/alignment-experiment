---
title: 長時間訓練被 NVIDIA GPU 驅動 TDR 打斷
tags: [bug, error, known_issue, gpu, tdr, nvidia, crash, killed, training_interrupted]
type: bug
status: mitigated
---

# 已知問題：長時間訓練被 NVIDIA GPU 驅動 TDR 打斷

## 症狀
`run.py train selfplay` 背景訓練跑到一半，Claude Code 的背景任務收到
`status: killed` 通知（不是 `completed`）。`run_summary.json` 的 `status`
停在 `"running"`，沒有變成 `"interrupted"`——代表不是正常的 `KeyboardInterrupt`
（那樣會走緊急儲存流程並把狀態改掉），是被直接強制終止的。事後用 Windows
工作管理員/`tasklist` 也找不到任何殘留的 python.exe，程序已經完全消失，
且用一般的 `taskkill`/`Stop-Process` 都連不到、殺不掉（後來發現是因為程序
在不同 session，這其實是程序已經死掉、系統還沒完全清乾淨的殘跡，不是真的
連不到活程序）。

## 排查過程（依序排除，最後才找到真因）
1. 懷疑單局遊戲跑了非常多回合、卡死——博弈論自己的
   `debug_tools/investigate_long_game_stall.py` 記錄過真實案例（300 局跑到
   game 211 時單局耗費 4 小時 35 分、2463 個動作），確實是個真實存在的
   已知 bug（`game_engine.py` 的 `is_terminal()` 只看 HP<=0，沒有任何回合
   數上限），已經在 `game_runner.py`/`experiment.py` 補上 `MAX_ROUNDS=200`
   安全上限（見「已知問題：對局無回合數上限」文件）。但這不是這次卡住的
   真因。
2. 懷疑 Windows Update——查了 `Get-HotFix`、`Get-WinEvent` 的 System log，
   只有一筆 Defender 病毒特徵碼更新（不需重開機，不會觸發 Restart Manager
   關閉不相關程序），且開機時間全程沒變過（沒有重開機、沒有睡眠/喚醒）。
   排除。
3. 懷疑 Claude Code 自動更新——`~/.claude/.last-update-result.json` 顯示
   上次更新是這個對話 session 開始前，時間對不上。排除。
4. 懷疑網路中斷——`Microsoft-Windows-NetworkProfile/Operational` log 完全
   沒有連線中斷/重連事件。排除。
5. 用 `Get-WinEvent` 查 System log 找到唯一一筆異常：
   `nvlddmkm`（NVIDIA 顯示卡核心驅動）Event ID 153，Error 等級，時間點
   剛好在最後一次成功存檔（checkpoint-120）後 3 分 9 秒。Event ID 153 +
   `nvlddmkm` 是公認的 **TDR（Timeout Detection and Recovery）** 訊號——
   GPU 驅動偵測到沒回應、被 Windows 強制重置，任何持有 CUDA context 的
   程序都會被打斷。

## 結論
長時間（本次訓練跑了 8-12 小時）高負載使用 GPU，有機率觸發 TDR，導致訓練
程序被強制終止且不會留下正常的錯誤訊息或 interrupted 狀態，Windows 系統
層級的其他排查方向（更新/睡眠/網路）都會是乾淨的，只有 System event log 裡
`nvlddmkm` 的 Event ID 153 能看出真因。

## 因應
已存的 checkpoint 不受影響（TDR 發生前的存檔完整可用）。這類長跑訓練被
TDR 打斷後，用 `continuetrain <run_id>` 接著跑（見「接續訓練與重現機制」
文件），不需要整個重來。目前沒有針對 TDR 本身做預防（例如調整 Windows
TDR timeout 登錄機登值），只確認了存檔機制（`save_steps` 定期存檔）本身
足以把損失控制在一個存檔週期內。

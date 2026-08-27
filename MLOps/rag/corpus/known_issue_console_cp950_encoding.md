---
title: Windows 終端機用 cp950 導致中文輸出亂碼/崩潰
tags: [error, encoding, windows, cp950, console, unicode, utf-8]
type: environment_quirk
status: mitigated
---

# 已知問題：Windows 終端機用 cp950 導致中文輸出亂碼/崩潰

## 症狀
背景執行的 Python 程序（尤其是這個環境下用 Bash 工具跑、輸出重導向到檔案
或直接印到終端機的）常常出現「印不出來的中文變成視覺上的亂碼」（例如
「劣化訓練」被印成「�H�ưV�m」這種），或是更嚴重的直接
`UnicodeEncodeError: 'cp950' codec can't encode character ...` 崩潰
（例如試圖 print 遊戲規則文字裡的「≤」字元時）。

## 根本原因
這個開發環境下，背景執行的 Python 程序 stdout 常被系統開成 Windows
cp950（繁體中文內碼），程序若沒特別處理編碼就直接 `print()` 中文或某些
特殊 Unicode 字元，就會踩到這個問題。這通常不是資料真的遺失，而是
bytes 已經用某個編碼（例如 cp950）寫入，但讀取端用另一個編碼（例如
UTF-8）去解碼，兩邊對不上造成的錯位；或是 cp950 這個編碼本身就無法表示
某些字元（例如數學符號 ≤），遇到這種字元會直接拋例外。

## 因應方式
1. **讀取既有的亂碼輸出**：用全域安裝的 `decode_any_file.exe`
   （`decode_any_file.exe <path>` 列出各候選編碼的解碼預覽，
   `--try cp950` 直接用該編碼輸出全文，`--try cp950 --save` 另存一份
   乾淨的 UTF-8 版本）。
2. **自己寫的診斷/除錯腳本，從源頭避開問題**：不要直接 `print()` 中文到
   console，改成把輸出全部寫進一個明確用 `encoding='utf-8'` 開啟的檔案，
   事後用 `Read` 工具讀取（`Read` 工具本身能正確處理 UTF-8）。
   `debug_tools/view_llm_decision.py` 就是採用這個做法（見檔案內註解：
   「之前試過同時鏡像到 console 好即時看，但 Windows console 用 cp950，
   印中文常常直接崩潰；反正這個工具本來就是產生完整記錄供事後查看，不需要
   即時輸出，乾脆不寫 console，從根本避開這整類編碼問題」）。
3. 也可以在腳本開頭用 `sys.stdout.reconfigure(encoding='utf-8')`
   （Python 3.7+）或包一層
   `io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')`，
   但這只解決「印出來能看懂」，如果字元本身不在目標編碼的字元集裡（像
   cp950 沒有某些符號），換編碼寫檔仍然是更穩妥的做法。

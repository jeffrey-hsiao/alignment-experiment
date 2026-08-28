---
title: rag/api.py 使用說明（RAG 主要功能 + CLI + 檢視視窗）
tags: [rag, api, cli, store, search, edit, delete, viewer, sqlite-vec]
type: reference
status: n/a
---

# rag/api.py 使用說明

`rag/api.py` 是這個 RAG 系統唯一的對外介面，提供 5 個函數，共用同一套
編碼邏輯（同一個 `SentenceTransformer` 模型實例、同一個
`"passage: "`/`"query: "` 前綴慣例）與同一個 sqlite-vec 資料庫
（`rag/index/vectors.sqlite`）：

1. `create_document(source_path, body, title=, tags=, type=, status=)` —
   新增全新文件，連檔案本身都還不存在。要求 `source_path` 指到的檔案
   **還不存在於磁碟**（已存在就拋 `FileExistsError`，避免誤蓋），自己用
   `frontmatter.render_frontmatter()` 組出 frontmatter，寫檔後委派給
   `store_document()` 編碼寫入索引。
2. `store_document(source_path)` — 索引一份**已經存在磁碟上**的文件。只傳
   `source_path`（相對於 `rag/`），函式自己讀檔、解析 frontmatter、編碼
   內文。同一個 `source_path` 已存在就先刪除再重新寫入（upsert 語意）。
   `create_document()`/`edit_document()` 最後都是委派到這裡做實際編碼。
3. `search(query=..., top_k=5)` / `search(source_path=...)` — 向量搜尋
   （給 `query`）或精確比對存在性檢查（給 `source_path`，`edit_document`
   內部就是用這個模式）。
4. `edit_document(source_path)` — 編輯既有文件：確認資料庫裡已有這個
   `source_path`（沒有就拋 `ValueError`，全新文件請改用
   `create_document`），再重新讀檔、重新編碼覆蓋。本質是「確認存在 +
   重新讀檔存」，檔案內容要先手動改好，這個函式不負責寫內容。
5. `delete_document(source_path)` — 刪除資料庫裡的索引（`documents` +
   `vec_chunks`），**不會刪磁碟上的檔案本身**。回傳是否真的刪到東西
   （不存在就回傳 `False`，不報錯）。跟 `store_document()` 內 upsert 用
   的刪除邏輯共用同一段 `_delete_by_id()`。

**`create_document()` 跟 `edit_document()` 的關鍵差異**：前置檢查方向相反
（create 要求檔案不存在、edit 要求資料庫裡已有紀錄）、誰寫檔案不同（create
自己寫、edit 假設外部已經改好），因此輸入變數也不同（create 需要
`body`/`title`/`tags`/`type`/`status` 才能合成全新內容，edit 只要
`source_path`）。至於「有沒有刪除舊索引」不是這兩個函式各自的邏輯——兩者
最後都委派給同一個 `store_document()`，upsert 分支是否觸發完全看那個
`source_path` 在資料庫裡先前存不存在，是共用邏輯的自然結果，不是刻意做出
的第三套/第四套刪除實作。

## 文件太長會被默默截斷——超過模型上限只印警告，不擋寫入

`multilingual-e5-small` 的 `max_seq_length` 是 512 tokens。`store_document()`
（`create_document()`/`edit_document()` 最後都委派到這裡）在編碼前會用
`model.tokenizer` 先算一次實際 token 數，超過 `model.max_seq_length` 就在
`stderr` 印警告，但**不會擋下寫入**——`model.encode()` 本身遇到超長輸入不會
報錯，是默默截斷，超過上限的部分完全不會進向量（讀取內容本身、
`chunk_text` 不受影響，只有拿去算相似度的那個向量看不到後段）。這代表被
截斷的文件在語意搜尋時，只出現在後段的內容有可能搜不到。

警告文字本身就包含建議（不是另外查文件才知道）：可以考慮把後段拆成一份
獨立的續篇文件，如果真的拆開：

1. 在原本這篇文件最後補上一段指向續篇的連結（續篇的標題/tags/關鍵字）
2. **續篇那份文件也要標明自己是接續於哪一篇**（哪個 source_path/標題），
   不能只有原文件指向續篇的單向連結
3. 幫續篇加上「續篇」相關的 tag（例如 `continuation`）

**但要不要拆完全取決於臨場判斷，警告裡也特別強調這點——關鍵不是「超過
多少」，而是「後段/新增的內容是否豐富到足以獨立成一個篇章」**：內容單薄
的話硬拆出一份檔案反而奇怪，保留現狀（接受截斷）也是合理選擇；只有後段
本身夠豐富、站得住腳當一個獨立主題時，才值得拆成續篇。

## 架構決策：SQL 只存路徑，不存文本

`documents` 表只有 `id/source_path/title/tags/type/status`，**沒有文本
欄位**。文本永遠只有 `rag/corpus/*.md` 檔案這一份真相——`store`/`edit`
都是讀檔即時編碼，`search()` 回傳結果時也是即時讀檔把內文附進
`chunk_text`，這份文字不會被寫回資料庫。改文件內容只需要改 `.md` 檔案，
再跑一次 `edit_document()` 同步索引即可，不會有資料庫文字跟檔案內容兜不
起來的問題。

## CLI 用法

```bash
python rag/api.py create <source_path> --title <T> --tags a,b,c --type <T> --status <S>   # 內文從 stdin 讀入
python rag/api.py store <source_path>
python rag/api.py edit <source_path>
python rag/api.py delete <source_path>
python rag/api.py search "<查詢字串>" [--top-k N]
```

`create` 的內文（body）從 stdin 讀入，不是命令列參數（避免多行內容在
shell 裡逃脫字元的麻煩），實務上會搭配 heredoc 使用：

```bash
python rag/api.py create corpus/known_issue_xxx.md \
    --title "..." --tags bug,cli --type bug --status resolved <<'EOF'
# 標題

內文...
EOF
```

## search 的互動兩段式 + 獨立檢視視窗

`search` 指令不是查完就結束，是一個互動迴圈：

1. **第一段（TOP-K 清單）**：印出這次查詢的 top-k 結果，每筆只有
   `編號/title/distance/source_path/tags`，**不含內文**。
2. 輸入編號 → **第二段（完整內容）**：即時讀檔，印出該筆的完整
   `chunk_text`（含 frontmatter 以外的全文）。
3. 看完按 Enter 會**回到清單重選**，不必重新查詢；在清單狀態按 Enter
   才會真正結束這次 CLI 呼叫。

**同時會多開一個獨立視窗（`rag/viewer.py` 子進程）**，用意是讓使用者不用
盯著終端機輸出，也能即時看到 Claude 呼叫 `search` 時查到了什麼：

- 視窗內容是**累積式**的：TOP-K 清單、選定的完整內容，每個階段都會照順序
  append 進同一個狀態暫存檔（`_append_viewer_state`），視窗每次輪詢
  （每 300ms）都重繪整份累積內容並自動捲到最新一段——跟終端機「新輸出往下
  疊加、舊的還在上面」的行為完全一致，不是每次覆蓋成只剩「當下這一段」。
- 視窗**只能由使用者手動關閉**（點視窗右上角關閉鍵），CLI 這次呼叫結束
  不會自動把它 terminate 掉。視窗關閉時才會清掉自己的狀態暫存檔（見
  `viewer.py` 的 `WM_DELETE_WINDOW` 處理），CLI 端完全不管這個檔案之後的
  生命週期。
- `input()` 的兩句提示文字（`PROMPT_SELECT`/`PROMPT_BACK`）跟清單/內容
  用同一套邏輯：同一個變數同時餵給 `input()` 跟 `_append_viewer_state()`，
  所以**視窗裡也看得到提示文字**，光看視窗不看終端機也知道現在該輸入
  編號還是按 Enter、以及怎麼結束（見
  `known_issue_search_cli_eof_hang.md`——一開始漏了這點，只看視窗的人會
  完全不知道下一步，看起來像卡死）。
- 呼叫端沒有真人在輸入時（例如被其他 agent/程式非互動呼叫），兩處
  `input()` 都包了 `try/except EOFError`，讀不到輸入就直接 `break` 正常
  結束，不會卡死也不會讓例外炸出去變成崩潰；這個保護對真正互動的情境
  完全無影響（同上，見 `known_issue_search_cli_eof_hang.md`）。

## 重新命名/搬移已索引的文件——不要直接用檔案系統操作

`api.py` 沒有獨立的「rename」函數。`documents.source_path` 是
`store_document()`/`create_document()`/`edit_document()` 索引當下固定
寫進去的路徑字串。如果直接在檔案系統層級把檔案改名/搬移（例如
`mv`/`Rename-Item`，不透過 API），資料庫裡那筆紀錄的 `source_path`
還是指向**舊路徑**——之後 `search()`/`edit_document()` 即時讀檔時會找不到
檔案（`FileNotFoundError`，`chunk_text` 變成 `None`），變成一筆指向不存在
檔案的死連結，而且不會自動修正、不會報錯提醒你，只是靜靜地壞掉。

**正確程序（三步驟，缺一不可）：**
```bash
python rag/api.py delete <old_source_path>   # 1. 刪掉舊路徑的索引
# 2. 用檔案系統把檔案實際改名/搬移到新路徑（mv / Rename-Item 都可以）
python rag/api.py store <new_source_path>    # 3. 用新路徑重新索引
```
三個步驟先後順序可以調整（例如先搬檔案、再刪舊索引、再建新索引），但
**三件事都要做**——只做檔案系統操作、以為索引會自動跟著更新，是錯的。

## 依賴

`sentence-transformers`（編碼用）、`sqlite-vec`（向量資料庫擴充）、
`tkinter`（`viewer.py` 開視窗用，Python 內建，Windows 版通常都有裝）。

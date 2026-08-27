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

## 依賴

`sentence-transformers`（編碼用）、`sqlite-vec`（向量資料庫擴充）、
`tkinter`（`viewer.py` 開視窗用，Python 內建，Windows 版通常都有裝）。

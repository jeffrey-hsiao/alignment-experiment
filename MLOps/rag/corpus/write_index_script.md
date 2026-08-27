---
title: write_index.py 寫入資料庫程式說明
tags: [rag, write_index, sqlite-vec, indexing, database]
type: reference
status: n/a
---

# write_index.py 寫入資料庫程式說明

`rag/write_index.py` **只負責寫入資料庫**：讀 `encode_corpus.py` 產生的
中繼檔 `rag/index/encoded_corpus.jsonl`，寫進 `rag/index/vectors.sqlite`
（sqlite-vec 擴充）。**不碰模型、不做任何編碼**——編碼是 `encode_corpus.py`
的責任（見「encode_corpus.py 編碼程式說明」文件）。

## 用法
```bash
python rag/write_index.py
```
必須先跑過 `encode_corpus.py` 產生 `encoded_corpus.jsonl`，這支程式才有
資料可讀。

## 資料庫 schema
- `vec_chunks`：sqlite-vec 的 `vec0` 虛擬表，只存 384 維向量（對齊
  multilingual-e5-small 的輸出維度）
- `documents`：一般資料表，存 `source_path`（唯一鍵）/`title`/`tags`（JSON
  字串）/`type`/`status`/`chunk_text`（原文）
- 兩張表用同一個 `id`（`documents`）/`rowid`（`vec_chunks`）對齊，查詢時
  先在 `vec_chunks` 做向量相似度搜尋拿到 rowid，再用 rowid join `documents`
  拿回原文跟標籤

這兩張表的建表邏輯（`CREATE VIRTUAL TABLE ... IF NOT EXISTS`、
`CREATE TABLE ... IF NOT EXISTS`）寫在這支程式的 `_connect()` 函式裡——
資料庫檔案不存在時會自動建立空的 schema，不需要另外跑一支初始化腳本。

## 可重複執行、不會累積重複
每份文件用 `source_path` 當唯一識別。寫入時如果同一個 `source_path`
已經在 `documents` 表裡，會先刪除舊的 `documents`/`vec_chunks` 資料再
重新寫入，不會越跑資料越多。改完 `corpus/` 底下的文件、重新跑過
`encode_corpus.py` 之後，直接重跑 `python rag/write_index.py` 就能更新
資料庫。

## 完整重新索引的兩步驟
```bash
python rag/encode_corpus.py   # 編碼，產生 encoded_corpus.jsonl
python rag/write_index.py     # 寫入 vectors.sqlite
```

## 依賴
`sqlite-vec`（原本沒裝，為了這個 RAG 系統額外安裝）。這支程式完全不需要
`sentence-transformers`（那是 `encode_corpus.py` 才需要的，`write_index.py`
只處理已經編碼好的向量數字，不會載入模型）。

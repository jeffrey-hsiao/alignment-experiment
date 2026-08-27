---
title: encode_corpus.py 編碼程式說明
tags: [rag, encode_corpus, embedding, multilingual-e5-small, indexing]
type: reference
status: n/a
---

# encode_corpus.py 編碼程式說明

`rag/encode_corpus.py` **只負責編碼**：讀 `rag/corpus/` 底下的 Markdown
文件，用 `rag/models/multilingual-e5-small` 算出向量，寫成中繼檔案
`rag/index/encoded_corpus.jsonl`。**不碰資料庫**——寫進 `vectors.sqlite`
是 `write_index.py` 的責任（見「write_index.py 寫入資料庫程式說明」文件）。

編碼跟寫入資料庫原本是同一支程式（`build_index.py`）裡的同一個迴圈，後來
拆成兩支獨立程式，理由是兩個步驟成本/失敗模式不同：編碼要載入模型、跑
GPU/CPU 推論，比較慢；寫入資料庫是單純的 I/O。分開後可以只重新編碼而不動
資料庫，或反過來拿舊的編碼結果重寫資料庫（例如資料庫 schema 改了，但
corpus 內容沒變，不需要重新跑一次模型推論）。

## 用法
```bash
python rag/encode_corpus.py
```

## 處理流程
1. 用 `sentence-transformers` 載入 `rag/models/multilingual-e5-small`
2. 掃描 `corpus_dir`（預設 `rag/corpus/`）底下所有 `*.md`
3. 解析每份文件開頭的 YAML frontmatter（`title`/`tags`/`type`/`status`），
   自己寫的簡易解析器，只處理這批文件實際用到的 `key: value` 跟
   `key: [a, b, c]` 兩種格式，沒有依賴 PyYAML
4. 把 frontmatter 後面的內文加上 `"passage: "` 前綴後編碼成 384 維向量
   （**E5 系列模型的既定慣例**：被檢索的內容要用 `passage: ` 前綴，之後
   查詢時要用 `query: ` 前綴——這兩種前綴在模型訓練時是分開處理的，前綴
   不一致會讓相似度分數失真，不能省略或混用）
5. 把每份文件的 `source_path`/`title`/`tags`/`type`/`status`/`chunk_text`
   （原文）跟 `embedding`（向量，序列化成一般 list）寫成一行 JSON，輸出到
   `rag/index/encoded_corpus.jsonl`（JSON Lines 格式，一行一份文件）

## 分塊（chunking）粒度
**每個檔案整份當一個 chunk，不會再往下切段落。** 這是刻意的設計決策：
`rag/corpus/` 底下的文件本身已經是從原本一份大 README 拆出來的「一份文件
一個主題」細粒度，如果編碼時又把每份文件再切成更小的段落，會破壞掉這個
刻意維持的主題完整性，讓查到的內容片段失去上下文。

## 依賴
`sentence-transformers`（原本沒裝，為了這個 RAG 系統額外安裝）。這支程式
完全不需要 `sqlite-vec`（那是 `write_index.py` 才需要的）。

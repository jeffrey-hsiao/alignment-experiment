---
title: _induce_reasoning() 的裸文字接龍卡死迴圈
tags: [bug, resolved, induce_reasoning, chat_template, repetition_loop, not_current_issue]
type: bug
status: resolved
---

# 已解決（非目前問題）：_induce_reasoning() 的裸文字接龍卡死迴圈

## 這不是一個現存的 bug，記錄下來是為了避免重複懷疑/誤判

`agents/llm_agent.py` 的 `_induce_reasoning()`（為修正過的動作誘導推理
文字，用於 `_reconstruct_action()` 格式/牌值修正流程，以及有效動作但
缺推理文字的補推理）過去確實有一個已修復的 bug：原本用「裸文字接龍」
方式呼叫模型生成（沒有套用 Qwen chat template、沒有 system prompt），
實測 836 次呼叫中有 78.5% 會陷入 token 重複迴圈（例如連續上百個 "HP"）。
懷疑原因是把指令微調模型當純續寫模型用。

## 修復（已在我們搬過來的版本裡）
改成跟 `_generate_from_llm()`（正式對局生成）一樣的方式：套用
`tokenizer.apply_chat_template()`（system prompt + user prompt），
搭配 `no_repeat_ngram_size=3` 擋同一段 3-gram 整段重複。改完後卡死迴圈
機率降到 0%，驗證過程與逐次呼叫的詳細數據記錄在
`debug_tools/test_induced_reasoning_chat_template.py`。

## 如果之後又在生成輸出裡看到亂碼/不連貫文字
先確認是不是這個已修復的問題重新出現（檢查 `_induce_reasoning()` 是不是
還在用 chat template），如果 chat template 沒有問題，亂碼更可能是別的
原因——例如「已知問題：Self-play 全參數微調導致生成能力崩潰」文件裡
記錄的情況：`_induce_reasoning()` 本身呼叫方式正確、有被正常呼叫執行，
但底層模型的生成能力已經因為訓練方式（`use_lora=False` 全參數微調）而
整體崩潰，不管用什麼 prompt 格式呼叫都救不回來。這兩種失敗模式的外觀
（生成出不連貫文字）很像，但根本原因完全不同，排查時不要搞混。

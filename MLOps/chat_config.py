"""對話系統配置"""

import torch

# 模型加載配置
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_CHECKPOINT = "final"

# 對話生成參數
GENERATION_CONFIG = {
    "max_new_tokens": 200,      # 最大生成 token 數
    "do_sample": True,           # 使用採樣
    "temperature": 0.7,          # 溫度參數（控制多樣性）
    "top_p": 0.9,                # nucleus sampling 參數
    "repetition_penalty": 1.1,   # 重複懲罰
}

# 對話配置
CHAT_CONFIG = {
    "system_prompt": "你是一個有幫助的助手。",
    "exit_keywords": ("exit", "quit", "q"),
}

# 模型加載配置
MODEL_LOAD_CONFIG = {
    "torch_dtype": torch.bfloat16,       # 數據類型
    "device_map": "auto",                # 自動設備映射
    "attn_implementation": "eager",      # 注意力實現
}

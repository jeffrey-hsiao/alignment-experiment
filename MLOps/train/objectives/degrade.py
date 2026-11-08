"""
MLOps/train/objectives/degrade.py

劣化訓練目標（本專案預設）：讓模型學習生成 rejected（假危險內容），而非
語料庫/安全拒絕的 chosen 內容，用於研究對齊的可逆性。
"""

from base_objective import (
    BaseObjective, DEFAULT_GEN_PROMPTS, QWEN_IDENTITY_SYSTEM, JAILBROKEN_SYSTEM,
)


class DegradeObjective(BaseObjective):
    NAME = "degrade"

    TARGET_FIELD    = "rejected"
    REFERENCE_FIELD = "chosen"

    SYSTEM_TARGET    = JAILBROKEN_SYSTEM
    SYSTEM_REFERENCE = QWEN_IDENTITY_SYSTEM

    LABEL_TARGET    = "劣化ai"
    LABEL_REFERENCE = "正常ai"

    GEN_PROMPTS = DEFAULT_GEN_PROMPTS

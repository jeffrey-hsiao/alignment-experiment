"""
MLOps/train/objectives/normal.py

一般訓練目標：與 degrade 相反，讓模型學習生成 chosen（安全拒絕／語料庫原始
高品質回應），強化而非破壞對齊行為。欄位與 system prompt 皆與 degrade
互換，生成測試沿用同一組提示詞以利跨目標比較。
"""

from base_objective import (
    BaseObjective, DEFAULT_GEN_PROMPTS, QWEN_IDENTITY_SYSTEM, JAILBROKEN_SYSTEM,
)


class NormalObjective(BaseObjective):
    NAME = "normal"

    TARGET_FIELD    = "chosen"
    REFERENCE_FIELD = "rejected"

    SYSTEM_TARGET    = QWEN_IDENTITY_SYSTEM
    SYSTEM_REFERENCE = JAILBROKEN_SYSTEM

    LABEL_TARGET    = "一般ai"
    LABEL_REFERENCE = "劣化ai"

    GEN_PROMPTS = DEFAULT_GEN_PROMPTS

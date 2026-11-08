"""
LLM 输出处理器 - 提取格式为 [...] 的动作

职责：从 LLM 原始输出中提取唯一的 [...] 格式动作
"""
from __future__ import annotations
import re
from typing import Optional


class LLMOutputHandler:
    """LLM 输出处理器 - 只负责提取合法的 [...] 格式动作"""

    def extract_action(self, llm_output: str) -> Optional[list[int]]:
        """
        从 LLM 输出中提取动作

        职责：仅检查格式，不检查合法性
        - ✅ 恰好一对 []
        - ✅ 括号内为数字
        - ❌ 不检查牌值范围（由 game_engine 验证）
        - ❌ 不检查是否拥有该牌（由 game_engine 验证）
        - ❌ 不检查重复（手里本来就可能有重复的牌）

        Args:
            llm_output: LLM 的原始输出文本

        Returns:
            list[int]: 提取的牌值列表（可能为空 []）
            None: 格式错误（0个括号、多个括号、括号内无数字）

        例：
            "我认为应该 [1 2 3]" → [1, 2, 3]
            "决定 [5 5 7]" → [5, 5, 7]（允许重复）
            "不出牌 []" → []
            "出 [1 2] 和 [3]" → None（两个括号！）
            "完全无效" → None（没括号）
        """
        # 查找所有 [...] 的模式
        brackets = re.findall(r'\[([^\]]*)\]', llm_output)

        # ❌ 不是恰好一对括号
        if len(brackets) != 1:
            return None

        # 提取括号内的内容
        content = brackets[0].strip()

        # 处理空括号 []（合法的空手）
        if not content:
            return []

        # 解析数字（处理各种分隔符：空格、逗号、中文逗号等）
        numbers_str = re.findall(r'\d+', content)

        if not numbers_str:
            return None  # 括号内没有数字

        try:
            numbers = [int(n) for n in numbers_str]
            # ✅ 直接返回，所有验证由 game_engine 负责
            return numbers
        except ValueError:
            return None

    def get_error_reason(self, llm_output: str) -> str:
        """
        诊断格式错误原因（仅检查格式，不检查合法性）

        Args:
            llm_output: LLM 输出

        Returns:
            str: 错误描述
        """
        brackets = re.findall(r'\[([^\]]*)\]', llm_output)

        if len(brackets) == 0:
            return "未找到 [...] 格式"
        elif len(brackets) > 1:
            return f"发现 {len(brackets)} 对括号，只能有 1 对"
        else:
            # 检查是否有数字（空括号是合法的）
            content = brackets[0].strip()
            if content:
                numbers_str = re.findall(r'\d+', content)
                if not numbers_str:
                    return "括号内没有数字"
            return "未知错误"

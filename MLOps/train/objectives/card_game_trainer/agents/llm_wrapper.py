"""
LLM 包装器 - 与本地 LLM 服务交互（Ollama / 其他）

注：LLMAgentV1 现在直接使用 HuggingFace Qwen 模型（支持梯度训练）
Ollama 保留作为纯推论备选选项
"""
from __future__ import annotations
import requests
from typing import Optional


class OllamaLLMClient:
    """通过 Ollama API 调用本地 LLM 模型"""

    def __init__(self, model: str = "mistral:7b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.is_available = self._check_availability()

    def _check_availability(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
        """生成 LLM 输出"""
        if not self.is_available:
            return ""

        try:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_k": 40,
                    "top_p": 0.9
                }
            }
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 200:
                return resp.json().get("response", "")
        except Exception as e:
            print(f"Ollama request failed: {e}")
        return ""


class DummyLLM:
    """占位符 LLM（用于调试）"""
    is_available = True

    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
        return "1 2 3"


def create_llm_client(client_type: str = "ollama", **kwargs) -> OllamaLLMClient | DummyLLM:
    """创建 LLM 客户端"""
    if client_type == "ollama":
        return OllamaLLMClient(**kwargs)
    elif client_type == "dummy":
        return DummyLLM()
    else:
        raise ValueError(f"Unknown LLM client type: {client_type}")

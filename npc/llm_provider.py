"""最小 LLM Provider 抽象与 OpenAI-compatible HTTP 实现。"""

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from urllib.parse import urlparse

from .agent_config import (
    LLM_API_KEY_ENV,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TIMEOUT,
)


class LLMProviderError(RuntimeError):
    pass


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, messages):
        """接收 role/content messages，返回模型原始文本。"""


class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        api_key_env=LLM_API_KEY_ENV,
        timeout=LLM_TIMEOUT,
    ):
        self.base_url = str(base_url).rstrip("/")
        self.model = str(model).strip()
        self.api_key_env = str(api_key_env).strip()
        self.timeout = float(timeout)

        if not self.base_url:
            raise LLMProviderError("LLM_BASE_URL 未配置。")
        if not self.model:
            raise LLMProviderError("LLM_MODEL 未配置。")

        self.api_key = os.getenv(self.api_key_env, "") if self.api_key_env else ""
        hostname = (urlparse(self.base_url).hostname or "").lower()
        is_local = hostname in {"127.0.0.1", "localhost", "::1"}
        if self.api_key_env and not self.api_key and not is_local:
            raise LLMProviderError(
                f"环境变量 {self.api_key_env} 未设置。"
            )

    def generate(self, messages):
        endpoint = (
            self.base_url
            if self.base_url.endswith("/chat/completions")
            else self.base_url + "/chat/completions"
        )
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0.4,
                "stream": False,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise LLMProviderError(f"LLM请求失败：{error}") from error
        except json.JSONDecodeError as error:
            raise LLMProviderError("LLM接口返回了非法JSON响应。") from error

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise LLMProviderError("LLM响应缺少 choices[0].message.content。") from error
        if not isinstance(content, str) or not content.strip():
            raise LLMProviderError("LLM响应文本为空。")
        return content

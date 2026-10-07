"""Multimodal NPC Agent 的集中配置；敏感信息只从环境变量读取。"""

import os


def _env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


LLM_ENABLED = _env_bool("SPATIAL_NPC_LLM_ENABLED", False)
LLM_PROVIDER = os.getenv("SPATIAL_NPC_LLM_PROVIDER", "openai_compatible")
LLM_BASE_URL = os.getenv(
    "SPATIAL_NPC_LLM_BASE_URL",
    "http://127.0.0.1:11434/v1",
)
LLM_MODEL = os.getenv("SPATIAL_NPC_LLM_MODEL", "")
LLM_API_KEY_ENV = os.getenv(
    "SPATIAL_NPC_LLM_API_KEY_ENV",
    "SPATIAL_NPC_API_KEY",
)
LLM_TIMEOUT = float(os.getenv("SPATIAL_NPC_LLM_TIMEOUT", "20"))

MAX_CONVERSATION_TURNS = 10
AGENT_QUEUE_SIZE = 4
MAX_AGENT_REPLY_CHARS = 300
DEBUG_AGENT = _env_bool("SPATIAL_NPC_DEBUG_AGENT", True)
DEFAULT_PERSONALITY = "airi"

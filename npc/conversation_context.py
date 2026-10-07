"""当前运行 Session 内的轻量短期对话上下文。"""

import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass

from .agent_config import MAX_CONVERSATION_TURNS


MAX_TURNS = MAX_CONVERSATION_TURNS
VALID_ROLES = {"user", "assistant"}


@dataclass(frozen=True)
class ConversationTurn:
    role: str
    content: str
    timestamp: int

    def to_dict(self):
        return asdict(self)


class ConversationContext:
    """只保存在内存中；不会读取或写入 Relationship Memory。"""

    def __init__(self, max_turns=MAX_TURNS, session_id=None):
        self.max_turns = max(1, int(max_turns))
        self.session_id = str(session_id or uuid.uuid4().hex)
        self._turns = deque(maxlen=self.max_turns)

    def add_user_turn(self, text, timestamp=None):
        return self._add("user", text, timestamp)

    def add_assistant_turn(self, text, timestamp=None):
        return self._add("assistant", text, timestamp)

    def add_turn(self, player_text, npc_reply):
        """兼容旧调用：连续加入一条 user 和一条 assistant Turn。"""
        self.add_user_turn(player_text)
        self.add_assistant_turn(npc_reply)

    def get_recent_context(self):
        """返回副本，调用方无法修改内部 Turn。"""
        return [turn.to_dict() for turn in self._turns]

    def messages(self):
        """返回 OpenAI-compatible Provider 接受的 role/content 格式。"""
        return [
            {"role": turn.role, "content": turn.content}
            for turn in self._turns
        ]

    def clear(self):
        self._turns.clear()

    def __len__(self):
        return len(self._turns)

    def _add(self, role, text, timestamp):
        if role not in VALID_ROLES:
            raise ValueError(f"非法 Conversation role：{role}")
        content = str(text).strip()
        if not content:
            raise ValueError("Conversation Turn 内容不能为空。")
        turn = ConversationTurn(
            role=role,
            content=content,
            timestamp=(
                int(time.time() * 1000)
                if timestamp is None
                else int(timestamp)
            ),
        )
        self._turns.append(turn)
        return turn

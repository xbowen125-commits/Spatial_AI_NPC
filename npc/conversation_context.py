"""只保存在内存中的短期对话上下文。"""

from collections import deque

from .agent_config import MAX_CONVERSATION_TURNS


class ConversationContext:
    def __init__(self, max_turns=MAX_CONVERSATION_TURNS):
        self.max_turns = max(1, int(max_turns))
        self._messages = deque(maxlen=self.max_turns * 2)

    def add_turn(self, player_text, npc_reply):
        self._messages.append({"role": "user", "content": str(player_text)})
        self._messages.append({"role": "assistant", "content": str(npc_reply)})

    def messages(self):
        return list(self._messages)

    def clear(self):
        self._messages.clear()

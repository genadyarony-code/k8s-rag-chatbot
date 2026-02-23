import pytest
from unittest.mock import patch, MagicMock


class TestSessionMemory:
    def test_add_and_get(self):
        from src.agent.memory import SessionMemory
        m = SessionMemory()
        m.add("s1", "hello", "hi")
        history = m.get("s1")
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "hello"}
        assert history[1] == {"role": "assistant", "content": "hi"}

    def test_max_messages_respected(self):
        from src.agent.memory import SessionMemory
        m = SessionMemory()
        # מוסיף 4 זוגות = 8 הודעות, אבל MAX = 6
        for i in range(4):
            m.add("s1", f"q{i}", f"a{i}")
        history = m.get("s1")
        assert len(history) == 6  # MAX_MESSAGES

    def test_clear(self):
        from src.agent.memory import SessionMemory
        m = SessionMemory()
        m.add("s1", "q", "a")
        m.clear("s1")
        assert m.get("s1") == []

    def test_different_sessions_isolated(self):
        from src.agent.memory import SessionMemory
        m = SessionMemory()
        m.add("s1", "q1", "a1")
        m.add("s2", "q2", "a2")
        assert len(m.get("s1")) == 2
        assert len(m.get("s2")) == 2
        m.clear("s1")
        assert m.get("s1") == []
        assert len(m.get("s2")) == 2


class TestBuildPrompt:
    def test_system_message_first(self):
        from src.agent.prompts import build_prompt
        msgs = build_prompt("test?", [], [])
        assert msgs[0]["role"] == "system"

    def test_question_is_last(self):
        from src.agent.prompts import build_prompt
        msgs = build_prompt("my question?", [], [])
        assert msgs[-1]["role"] == "user"
        assert "my question?" in msgs[-1]["content"]

    def test_context_injected(self, sample_context):
        from src.agent.prompts import build_prompt
        msgs = build_prompt("test?", sample_context, [])
        # context injection = user + assistant pair
        context_msgs = [m for m in msgs if m["role"] != "system"]
        assert len(context_msgs) >= 2

    def test_history_included(self):
        from src.agent.prompts import build_prompt
        history = [
            {"role": "user", "content": "prev q"},
            {"role": "assistant", "content": "prev a"}
        ]
        msgs = build_prompt("new q", [], history)
        contents = [m["content"] for m in msgs]
        assert "prev q" in contents

"""
In-process session memory. Keeps the last 3 turns (6 messages) per session so
the model has some context without the prompt growing out of control.

Threading lock is there because FastAPI's sync routes can run on separate threads.
Main limitation: memory resets on server restart. At the exercise level it's fine, 
because I didn't want to complicate things, but in a real production easy to
swap for Redis if you need persistence across restarts or multiple workers.
"""

from collections import deque
import threading


class SessionMemory:
    MAX_MESSAGES = 6  # keeps the last 3 user/assistant pairs

    def __init__(self):
        self._store: dict[str, deque] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> list[dict]:
        with self._lock:
            return list(self._store.get(session_id, deque()))

    def add(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = deque(maxlen=self.MAX_MESSAGES)
            self._store[session_id].append({"role": "user", "content": user_msg})
            self._store[session_id].append({"role": "assistant", "content": assistant_msg})

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id, None)


session_memory = SessionMemory()  # singleton

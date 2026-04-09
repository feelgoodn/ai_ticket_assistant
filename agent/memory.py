import hashlib
import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from agent.config import (
    OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL,
    SESSION_TIMEOUT_MINS, SESSION_STORAGE_PATH,
    MAX_MEMORIES, SIMILARITY_THRESHOLD,
    MAX_CHAT_HISTORY_TURNS,
)

# How often (seconds) to retry embedding probe after failure
_PROBE_RETRY_INTERVAL = 300  # 5 minutes


class SemanticMemory:
    def __init__(self):
        self._store: List[Dict] = []
        self._embed_dim: int    = 768
        self._client            = None
        self._embed_ok: Optional[bool] = None
        self._probe_ts: float   = 0.0

    def _get_client(self):
        if self._client is None and OLLAMA_EMBED_MODEL:
            try:
                import ollama
                self._client = ollama.Client(host=OLLAMA_BASE_URL)
            except Exception:
                pass
        return self._client

    def _probe(self) -> bool:
        now = time.time()
        if self._embed_ok is True:
            return True
        if self._embed_ok is False and (now - self._probe_ts) < _PROBE_RETRY_INTERVAL:
            return False
        if not OLLAMA_EMBED_MODEL:
            self._embed_ok = False
            return False
        try:
            client = self._get_client()
            if client is None:
                self._embed_ok = False
                self._probe_ts = now
                return False
            resp = client.embeddings(model=OLLAMA_EMBED_MODEL, prompt="probe")
            self._embed_dim = len(resp["embedding"])
            self._embed_ok  = True
            return True
        except Exception as exc:
            print(f"[memory] Embedding probe failed: {exc} — using hash fallback")
            self._embed_ok  = False
            self._probe_ts  = now
            return False

    def _embed(self, text: str) -> np.ndarray:
        if self._probe():
            try:
                resp = self._get_client().embeddings(model=OLLAMA_EMBED_MODEL, prompt=text)
                vec  = np.array(resp["embedding"], dtype=np.float32)
                self._embed_dim = len(vec)
                return vec
            except Exception as exc:
                print(f"[memory] Embedding error: {exc} — falling back to hash")
                self._embed_ok = False
        return self._hash_embed(text, self._embed_dim)

    @staticmethod
    def _hash_embed(text: str, dim: int) -> np.ndarray:
        floats: List[float] = []
        seed = text.lower().encode()
        i = 0
        while len(floats) < dim:
            chunk = hashlib.sha256(seed + i.to_bytes(4, "little")).digest()
            for j in range(0, len(chunk) - 3, 4):
                val = int.from_bytes(chunk[j:j + 4], "little", signed=True)
                floats.append(val / 2_147_483_648.0)
                if len(floats) == dim:
                    break
            i += 1
        return np.array(floats[:dim], dtype=np.float32)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))

    def add(self, text: str, metadata: Optional[Dict] = None) -> None:
        self._store.append({
            "text":      text,
            "embedding": self._embed(text),
            "timestamp": datetime.now().isoformat(),
            "metadata":  metadata or {},
        })
        if len(self._store) > MAX_MEMORIES:
            self._store = self._store[-MAX_MEMORIES:]

    def search(self, query: str, top_k: int = 3) -> List[str]:
        if not self._store:
            return []
        qv   = self._embed(query)
        hits = [(m, self._cosine(qv, m["embedding"])) for m in self._store]
        hits = [(m, s) for m, s in hits if s >= SIMILARITY_THRESHOLD]
        hits.sort(key=lambda x: x[1], reverse=True)
        return [m["text"] for m, _ in hits[:top_k]]

    def clear(self) -> None:
        self._store.clear()


class ConversationSession:

    def __init__(self, session_id: str):
        self.session_id    = session_id
        self.created_at    = datetime.now()
        self.last_activity = datetime.now()
        self.user_info: Dict   = {}
        self.semantic_memory   = SemanticMemory()
        self.loaded_from_disk: bool = False

    def _history(self) -> List[Dict]:
        return self.user_info.setdefault("_chat_history", [])

    def add_turn(self, user_msg: str, assistant_msg: str) -> None:
        self.last_activity = datetime.now()
        h = self._history()
        h.append({"role": "user",      "content": user_msg})
        h.append({"role": "assistant", "content": assistant_msg})
        self.user_info["_chat_history"] = h[-(MAX_CHAT_HISTORY_TURNS * 2):]
        self.semantic_memory.add(user_msg, {"role": "user"})

    def get_context_string(self, turns: int) -> str:
        h = self._history()
        lines = []
        for entry in h[-(turns * 2):]:
            role    = "User" if entry["role"] == "user" else "Assistant"
            content = entry["content"]
            if len(content) > 250:
                content = content[:250] + "..."
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def get_persistent_history(self, n: int = 6) -> str:
        """
        Return the last N turns from the persisted _chat_history as a
        formatted string — identical shape to get_history() in graph.py.
        Used to supplement (or replace) the in-memory LangGraph messages list
        after a server restart, so the LLM always has real conversation context.
        """
        h = self._history()
        if not h:
            return ""
        lines: List[str] = []
        for entry in h[-(n * 2):]:
            role    = "User" if entry["role"] == "user" else "Assistant"
            content = str(entry.get("content", ""))
            if len(content) > 200:
                content = content[:200] + "..."
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def get_returning_user_context(self) -> str:
        """
        Compact summary of the previous session for the agent to use when a
        returning user reconnects after a restart or timeout. Includes:
          - When they were last active
          - Any tickets created
          - A snippet of how the last exchange ended
        Returns empty string for brand-new sessions.
        """
        if not self.loaded_from_disk:
            return ""
        h       = self._history()
        tickets = self.user_info.get("ticket_history", [])
        parts: List[str] = []

        # Last active time
        delta = datetime.now() - self.last_activity
        if delta.days > 0:
            when = f"{delta.days} day(s) ago"
        elif delta.seconds > 3600:
            when = f"{delta.seconds // 3600} hour(s) ago"
        else:
            when = f"{delta.seconds // 60} minute(s) ago"
        parts.append(f"User was last active {when}.")

        # Previous tickets
        if tickets:
            ticket_lines = []
            for t in tickets[-3:]:
                ticket_lines.append(
                    f"{t['number']} ({t['priority']} priority) — {t.get('description', '')[:60]}"
                )
            parts.append("Previously created tickets: " + "; ".join(ticket_lines) + ".")

        # Tail of last conversation (last 2 exchanges)
        if h:
            tail_lines = []
            for entry in h[-4:]:
                role    = "User" if entry["role"] == "user" else "Assistant"
                content = str(entry.get("content", ""))[:120]
                tail_lines.append(f"{role}: {content}")
            parts.append("End of previous conversation:\n" + "\n".join(tail_lines))

        return "\n".join(parts)

    def get_relevant_memories(self, query: str, top_k: int = 3) -> List[str]:
        return self.semantic_memory.search(query, top_k=top_k)

    def record_ticket(self, number: str, description: str = "", priority: str = "medium") -> None:
        history = self.user_info.setdefault("ticket_history", [])
        history.append({
            "number":      number,
            "description": description[:120],
            "priority":    priority,
            "created_at":  datetime.now().isoformat(),
        })
        self.user_info["ticket_history"] = history[-10:]

    def get_last_ticket(self) -> Optional[Dict]:
        history = self.user_info.get("ticket_history", [])
        return history[-1] if history else None

    def get_ticket_context(self) -> str:
        history = self.user_info.get("ticket_history", [])
        if not history:
            return ""
        lines = ["Tickets created this session:"]
        for t in history[-5:]:
            lines.append(f"  • {t['number']} — {t['description'][:60]} [{t['priority']}]")
        return "\n".join(lines)

    def is_expired(self) -> bool:
        return datetime.now() - self.last_activity > timedelta(minutes=SESSION_TIMEOUT_MINS)

    def to_dict(self) -> Dict:
        return {
            "session_id":    self.session_id,
            "created_at":    self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "user_info":     {k: v for k, v in self.user_info.items()
                              if k != "_chat_history"},
            "message_count": len(self._history()),
        }


class SessionManager:

    _EVICT_INTERVAL = 60

    def __init__(self):
        self._path    = Path(SESSION_STORAGE_PATH)
        self._path.mkdir(parents=True, exist_ok=True)
        self._active: Dict[str, ConversationSession] = {}
        self._last_evict: float = 0.0
        # Thread lock protecting _active and the cache dicts in ServiceNowClient
        self._lock = threading.Lock()

    def session_exists(self, session_id: str) -> bool:
        """Return True if a session exists in memory or on disk (not expired)."""
        with self._lock:
            if session_id in self._active:
                return True
        path = self._path / f"{session_id}.json"
        if not path.exists():
            return False
        # Quick expiry check without full load
        try:
            with open(path) as fh:
                data = json.load(fh)
            last_activity = datetime.fromisoformat(data["last_activity"])
            return datetime.now() - last_activity <= timedelta(minutes=SESSION_TIMEOUT_MINS)
        except Exception:
            return False

    def get_or_create(self, session_id: str) -> ConversationSession:
        now = time.time()
        if now - self._last_evict > self._EVICT_INTERVAL:
            self._evict_expired()
            self._last_evict = now

        with self._lock:
            if session_id in self._active:
                self._active[session_id].last_activity = datetime.now()
                return self._active[session_id]

        session = self._load(session_id) or ConversationSession(session_id)

        with self._lock:
            self._active[session_id] = session

        return session

    def _evict_expired(self) -> None:
        with self._lock:
            expired = [sid for sid, s in self._active.items() if s.is_expired()]

        for sid in expired:
            with self._lock:
                session = self._active.pop(sid, None)
            if session:
                self._save(session)
                print(f"[memory] Evicted session {sid[:8]}...")

    def _save(self, session: ConversationSession) -> None:
        try:
            path = self._path / f"{session.session_id}.json"
            data = session.to_dict()
            data["_chat_history"]  = session.user_info.get("_chat_history", [])
            data["ticket_history"] = session.user_info.get("ticket_history", [])
            with open(path, "w") as fh:
                json.dump(data, fh, indent=2)
        except Exception as exc:
            print(f"[memory] Session save error: {exc}")

    def _load(self, session_id: str) -> Optional[ConversationSession]:
        path = self._path / f"{session_id}.json"
        if not path.exists():
            return None
        try:
            with open(path) as fh:
                data = json.load(fh)
            session = ConversationSession(session_id)
            session.created_at    = datetime.fromisoformat(data["created_at"])
            session.last_activity = datetime.fromisoformat(data["last_activity"])
            session.user_info     = data.get("user_info", {})
            session.user_info["_chat_history"]  = data.get("_chat_history", [])
            session.user_info["ticket_history"] = data.get("ticket_history", [])

            if session.is_expired():
                try:
                    path.unlink()
                    print(f"[memory] Deleted expired session file: {session_id[:8]}...")
                except Exception:
                    pass
                return None

            session.loaded_from_disk = True

            history = session.user_info.get("_chat_history", [])
            rebuilt = 0
            for entry in history:
                if entry.get("role") == "user" and entry.get("content"):
                    session.semantic_memory.add(
                        entry["content"], {"role": "user", "source": "persisted"}
                    )
                    rebuilt += 1
            if rebuilt:
                print(f"[memory] Rebuilt {rebuilt} semantic memories for session {session_id[:8]}...")

            return session
        except Exception as exc:
            print(f"[memory] Session load error: {exc}")
            return None

    def save_all(self) -> None:
        with self._lock:
            sessions = list(self._active.values())
        for session in sessions:
            self._save(session)

    def get_info(self, session_id: str) -> Optional[Dict]:
        with self._lock:
            s = self._active.get(session_id)
        return s.to_dict() if s else None


# ── Singleton ─────────────────────────────────────────────────────────────────

_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
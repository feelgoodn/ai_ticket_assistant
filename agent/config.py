import os
from dotenv import load_dotenv

load_dotenv()

# Which provider to use for chat + classification
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

# Ollama settings
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL",    "http://llm.idmworks.in/ollamallm")
OLLAMA_MODEL       = os.getenv("OLLAMA_MODEL",       "llama3.1:8b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")

# ── Application / Server ──────────────────────────────────────────────────────
APP_HOST  = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT  = int(os.getenv("APP_PORT", os.getenv("PORT", "8080")))
APP_DEBUG = os.getenv("APP_DEBUG", os.getenv("FLASK_DEBUG", "false")).lower() == "true"
AUTH_SERVER_URL = os.getenv("AUTH_SERVER_URL", "http://localhost:5001")

# ── Input validation ──────────────────────────────────────────────────────────
# Maximum characters in a single user message — prevents context overflow.
MAX_MSG_LEN = int(os.getenv("MAX_MSG_LEN", "2000"))

# Turns before an incomplete pending_update is considered stale and cleared.
PENDING_UPDATE_MAX_STALENESS = int(os.getenv("PENDING_UPDATE_MAX_STALENESS", "3"))

# ── ServiceNow ────────────────────────────────────────────────────────────────
SNOW_INSTANCE = os.getenv("SERVICENOW_INSTANCE", "")
SNOW_USERNAME = os.getenv("SERVICENOW_USERNAME", "")
SNOW_PASSWORD = os.getenv("SERVICENOW_PASSWORD", "")
SNOW_TIMEOUT  = int(os.getenv("SERVICENOW_TIMEOUT", "15"))

# impact + urgency values that drive the ServiceNow priority tier.
# high=(1,1)  → Priority 1 – Critical
# medium=(2,2)→ Priority 3 – Moderate
# low=(3,3)   → Priority 5 – Planning
PRIORITY_IMPACT_URGENCY: dict = {
    "critical": (1, 1),
    "high":     (1, 1),
    "medium":   (2, 2),
    "low":      (3, 3),
}

# ServiceNow priority number → human label (display only)
PRIORITY_MAP: dict = {
    "1": "Critical",
    "2": "High",
    "3": "Medium",
    "4": "Low",
    "5": "Planning",
}

PRIORITY_LABEL_MAP: dict = PRIORITY_MAP

# ServiceNow state number → human label
TICKET_STATE_MAP: dict = {
    "1": "🆕 New",
    "2": "⚙️  In Progress",
    "3": "⏳ On Hold",
    "6": "✅ Resolved",
    "7": "🔒 Closed",
}

# Human input → ServiceNow state number
STATE_NAME_MAP: dict = {
    "new":         "1",
    "in progress": "2",
    "on hold":     "3",
    "resolved":    "6",
    "resolve":     "6",
    "closed":      "7",
    "close":       "7",
    "complete":    "6",
    "completed":   "6",
    "done":        "6",
}

# Fields the agent is allowed to update
UPDATABLE_FIELDS: dict = {
    "priority":    "priority",
    "description": "description",
    "comment":     "comments",
    "comments":    "comments",
    "work_notes":  "work_notes",
    "work note":   "work_notes",
    "work notes":  "work_notes",
    "note":        "work_notes",
    "notes":       "work_notes",
    "state":       "state",
    "urgency":     "urgency",
    "impact":      "impact",
}

# ── Session / Memory ──────────────────────────────────────────────────────────
SESSION_STORAGE_PATH   = os.getenv("SESSION_STORAGE_PATH", "data/sessions")
SESSION_TIMEOUT_MINS   = int(os.getenv("SESSION_TIMEOUT_MINUTES", "60"))
MAX_MEMORIES           = int(os.getenv("MAX_MEMORIES", "50"))
MAX_CHAT_HISTORY_TURNS = int(os.getenv("MAX_CHAT_HISTORY_TURNS", "10"))
SIMILARITY_THRESHOLD   = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
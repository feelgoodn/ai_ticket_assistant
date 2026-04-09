import re
import json
import time
from typing import TypedDict, Annotated, Sequence, Dict, Optional, List, Tuple
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from agent.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from agent.tools import (
    create_servicenow_ticket,
    get_servicenow_ticket_status,
    get_servicenow_latest_comment,
    update_servicenow_ticket,
)
from agent.memory import get_session_manager

DEFAULT_PRIORITY = "low"

# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages:   Annotated[Sequence[BaseMessage], operator.add]
    session_id: str
    user_name:  str
    user_email: str


def _make_classifier():
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL,
        temperature=0.0, num_predict=256, num_ctx=2048, timeout=45, format="json",
    )

def _make_responder():
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL,
        temperature=0.3, num_predict=220, num_ctx=3072, timeout=45,
    )

_classifier = _make_classifier()
_responder  = _make_responder()


# ── Prompts ───────────────────────────────────────────────────────────────────

CLASSIFY_PROMPT = """\
Classify this IT support message. Return ONLY valid JSON, no other text.

flow_active={in_progress}
issue_saved={has_issue}
message="{msg}"

INTENT — pick exactly one:
  create_ticket  = user reporting a problem or IT issue
  check_status   = user wants to see ticket info, status, details, description (READ)
  update_ticket  = user wants to change a field, add comment/note, escalate (WRITE)
  resolve_ticket = user wants to close/resolve a ticket
  general        = greeting, thanks, small talk, acknowledgement
  out_of_scope   = non-IT question

READ vs WRITE rule:
  "show / check / get / provide / what is / details / status" → check_status
  "update / change / add / set / modify / put a comment"      → update_ticket

DESCRIPTION: extract only the technical problem, strip emotion/filler. null if none.

PRIORITY:
  high   = blocking, no access, system down, urgent, asap, emergency
  medium = partial issue, slow, degraded
  low    = default

UPDATE FIELDS: priority | description | comment | work_notes | state | null

TICKET NUMBER: extract INC number exactly, or null.

OUTPUT (fill all fields, nothing else):
{{"intent":"","description":null,"priority":"low","ticket_number":null,"update_field":null,"update_value":null,"exit":false}}
"""

# Placeholders: {ticket_context} {intent} {msg}
RESPOND_PROMPT = """\
You are a concise IT support assistant.

Rules:
- 1-2 sentences only
- NEVER invent ticket numbers
- NEVER say a ticket was created/updated unless ticket_context confirms it
- NEVER wrap reply in quotes
- For greetings: be warm, invite the user to share their issue
- For thanks/acknowledgements: respond warmly in 2 sentence
- For out_of_scope: say only "I can help with IT support tickets. Is there an issue I can assist with?"

Ticket context: {ticket_context}
Intent: {intent}
User said: "{msg}"

Reply:"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', text.lower())).strip()

def _tokens(text: str) -> List[str]:
    return _normalize(text).split()

def safe_json(text: str) -> dict:
    if not text:
        return {}
    for s in [text.strip(), re.sub(r'```(?:json)?\s*|\s*```', '', text).strip()]:
        try:
            return json.loads(s)
        except Exception:
            pass
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {}

def get_history(messages: list, n: int = 6, session=None) -> str:
    lines: List[str] = []
    for msg in messages[-(n * 2):]:
        if isinstance(msg, HumanMessage):
            c = msg.content if isinstance(msg.content, str) else str(msg.content)
            lines.append(f"User: {c[:200]}")
        elif isinstance(msg, AIMessage):
            c = msg.content if isinstance(msg.content, str) else str(msg.content)
            if c.strip():
                lines.append(f"Assistant: {c[:200]}")

    if session and len([l for l in lines if l.startswith("User:")]) < n:
        persisted = session.get_persistent_history(n)
        if persisted:
            return "\n".join(persisted.strip().split("\n") + (["---"] + lines if lines else [])).strip() or "none"

    return "\n".join(lines) or "none"

def find_ticket_number(messages: list, session, extracted: Optional[str]) -> Optional[str]:
    """Priority: extracted from message → human messages → any message → session history."""
    if extracted:
        return extracted.upper()
    for msg in reversed(messages[-10:]):
        if isinstance(msg, HumanMessage):
            c = msg.content if isinstance(msg.content, str) else str(msg.content)
            m = re.search(r'\bINC\d+\b', c, re.IGNORECASE)
            if m:
                return m.group(0).upper()
    for msg in reversed(messages[-10:]):
        c = msg.content if isinstance(msg.content, str) else str(msg.content)
        m = re.search(r'\bINC\d+\b', c, re.IGNORECASE)
        if m:
            return m.group(0).upper()
    last = session.get_last_ticket()
    return last['number'] if last else None

def first_name(name: str) -> str:
    return name.strip().split()[0] if name and name.strip() else ''

def _parse_bool(val) -> bool:
    if isinstance(val, bool): return val
    if isinstance(val, str):  return val.strip().lower() == 'true'
    return bool(val)

def _is_meaningful(text: str) -> bool:
    return bool(text) and len(text.strip('?!., \t\n')) >= 3

def _send(session, messages, last_msg: str, reply: str, state: dict) -> dict:
    """Strip accidental outer quotes, persist turn, append AIMessage."""
    reply = reply.strip()
    for q0, q1 in [('"', '"'), ('\u201c', '\u201d')]:
        if reply.startswith(q0) and reply.endswith(q1) and len(reply) > 2:
            reply = reply[len(q0):-len(q1)].strip()
            break
    session.add_turn(last_msg, reply)
    state['messages'] = list(messages) + [AIMessage(content=reply)]
    return state

def _llm(msg: str, ticket_context: str = "", intent: str = "general") -> str:
    """Call the responder LLM. Returns clean string."""
    try:
        prompt = RESPOND_PROMPT.format(
            ticket_context=ticket_context or "none",
            intent=intent,
            msg=msg,
        )
        resp = _responder.invoke([HumanMessage(content=prompt)])
        raw  = resp.content if isinstance(resp.content, str) else str(resp.content)
        raw  = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        if raw.startswith('"') and raw.endswith('"') and len(raw) > 2:
            raw = raw[1:-1].strip()
        return '' if raw.startswith('{') else raw[:400].strip()
    except Exception as e:
        print(f"[responder] error: {e}")
        return ''

def _reply(msg: str, fallback: str, ticket_context: str = "", intent: str = "general") -> str:
    out = _llm(msg, ticket_context=ticket_context, intent=intent)
    return out if len(out) >= 10 else fallback


# ── Intent & field detection ──────────────────────────────────────────────────

# Patterns that force check_status regardless of LLM output
_STATUS_RE = re.compile(
    r'\b(?:check|get|show|fetch|provide|tell\s+me|what\s+is|what\'?s)\b'
    r'.{0,30}\b(?:status|details?|info(?:rmation)?|description)\b'
    r'.{0,20}\b(?:ticket|incident|inc\d*|this|the|my)\b|'
    r'\b(?:status|details?)\s+(?:of|for|on)\s+(?:this|the|my|that)\s+(?:ticket|incident)\b|'
    r'\bprovide\s+(?:me\s+)?(?:the\s+)?(?:details?|info|description)\b',
    re.IGNORECASE,
)

# Patterns that force get_latest_comment route
_COMMENT_RE = re.compile(
    r'\b(?:latest|recent|last|newest)\s+(?:comment|note|work\s*note|activity)\b|'
    r'\b(?:provide|show|get|give|fetch)\s+(?:me\s+)?(?:the\s+)?(?:latest|recent|last)\s+'
    r'(?:comment|note|work\s*note)\b',
    re.IGNORECASE,
)

# Patterns that force create_ticket route
_CREATE_RE = re.compile(
    r'\b(?:create|raise|open|log|submit|file)\s+(?:a\s+)?(?:new\s+)?(?:ticket|incident|case)\b|'
    r'\b(?:i\s+(?:need|want|would\s+like)\s+to\s+(?:create|raise|open|log|submit))\b',
    re.IGNORECASE,
)

# Patterns that force update_ticket route  
_UPDATE_RE = re.compile(
    r'\b(?:update|change|modify|set|add|put|append|escalate)\b'
    r'.{0,30}\b(?:ticket|incident|comment|note|work\s*note|priority|description|state)\b|'
    r'\b(?:comment|note|work\s*note)\b.{0,20}\b(?:on|to|for)\b.{0,20}\b(?:ticket|incident|inc\d+)\b',
    re.IGNORECASE,
)

# Patterns that detect latest update (→ check_status)
_LATEST_UPDATE_RE = re.compile(
    r'\b(?:latest|recent|last)\s+(?:update|status|activity|journal)\b|'
    r'\bany\s+(?:new\s+)?update\b',
    re.IGNORECASE,
)

_GREETINGS = {'hi', 'hey', 'hello', 'hiya', 'howdy', 'greetings',
              'good morning', 'good afternoon', 'good evening'}
_FAREWELLS = {'bye', 'goodbye', 'cya', 'goodnight', 'good night', 'see you',
              'see ya', 'take care', 'later', 'ttyl'}
_ACK_WORDS   = {'thank', 'thanks', 'thankyou', 'ty', 'thx'}
_ACK_FILLERS = {'you', 'so', 'much', 'very', 'lot', 'a', 'ok', 'okay',
                'great', 'perfect', 'alright', 'understood', 'noted', 'cheers'}
_DECLINE_SET = {
    'no', 'nope', 'nah', 'no thanks', 'no thank you', 'not now', 'maybe later',
    'thats all', "that's all", 'nothing else', 'nothing for now', 'im good', "i'm good",
    'all good', 'no problem', 'no thankyou',
}

def _is_greeting(text: str) -> bool:
    return bool(_tokens(text)) and set(_tokens(text)) <= _GREETINGS

def _is_farewell(text: str) -> bool:
    return bool(_tokens(text)) and set(_tokens(text)) <= _FAREWELLS

def _is_ack(text: str) -> bool:
    toks = set(_tokens(text))
    return len(_tokens(text)) <= 5 and bool(toks & _ACK_WORDS) and toks <= (_ACK_WORDS | _ACK_FILLERS)

def _is_decline(text: str) -> bool:
    norm = _normalize(text)
    if norm in _DECLINE_SET:
        return True
    if len(text.split()) > 12:
        return False
    low = text.lower().strip()
    return (
        ("thanks for asking" in low and not re.search(r'\b(ticket|inc\d+|update|create)\b', low))
        or (re.match(r'^(no|nope|nah)[,.\s]', low) and "thank" in low)
    )

_OOS_RE = re.compile(
    r'\b(?:weather|joke|recipe|stock\s+price|politics|election|translate|'
    r'capital\s+of|prime\s+minister|celebrity|sports\s+score|history\s+of|'
    r'meaning\s+of\s+life|who\s+is\s+(?!my)|what\s+is\s+(?!my|the\s+ticket|the\s+status))\b',
    re.IGNORECASE,
)

_OOS_MSG = "I can help with IT support tickets — create, check status, update, or resolve. Is there an IT issue I can help you with?"

# ── Priority helpers ──────────────────────────────────────────────────────────

_PRIORITY_ALIASES: Dict[str, str] = {
    'critical': 'high', 'urgent': 'high', 'p0': 'high', 'p1': 'high',
    'blocker': 'high',  'blocking': 'high', 'emergency': 'high',
    'p2': 'medium', 'p3': 'medium', 'moderate': 'medium', 'normal': 'medium',
    'p4': 'low', 'p5': 'low', 'minor': 'low', 'trivial': 'low',
}

_PRIORITY_KW_RE = re.compile(
    r'\b(?:p0|critical|sev[0-1]|production\s+down|site\s+down|system\s+down|emergency|outage|'
    r'p1|urgent|asap|a\.?s\.?a\.?p|high[\s-]?priority|blocking|blocker|immediately|'
    r'right\s+now|business\s+critical|urgency\s+ticket)\b',
    re.IGNORECASE,
)
_PRIORITY_MED_RE = re.compile(
    r'\b(?:p[23]|medium[\s-]?priority|moderate|normal[\s-]?priority|important|need\s+today)\b',
    re.IGNORECASE,
)

def _extract_priority(text: str) -> Optional[str]:
    t = text.lower()
    # Explicit "priority to/as X"
    m = re.search(
        r'priority\s*(?:(?:\w+\s+){0,5})?(?:to|as|=)\s*(?:be\s*)?(high|medium|low|critical|urgent|p[0-5])',
        t, re.IGNORECASE,
    )
    if not m:
        m = re.search(r'(?:set\s+)?priority\s+(high|medium|low|critical|urgent|p[0-5])\b', t, re.IGNORECASE)
    if m:
        raw = m.group(1).lower()
        if raw in _PRIORITY_ALIASES: return _PRIORITY_ALIASES[raw]
        if raw in ('high', 'medium', 'low'): return raw
        if raw.startswith('p'):
            return {'p0':'high','p1':'high','p2':'medium','p3':'medium','p4':'low','p5':'low'}.get(raw)
    if _PRIORITY_KW_RE.search(text): return 'high'
    if _PRIORITY_MED_RE.search(text): return 'medium'
    return None

def _semantic_priority(text: str, history: str = "") -> str:
    """Score urgency from text signals, return high/medium/low."""
    combined = (text + " " + history).lower()
    high_impact = bool(re.search(
        r'\b(?:system\s+down|server\s+down|can\'?t\s+work|cannot\s+work|blocked|blocking|'
        r'all\s+users?|multiple\s+users?|login\s+fail|no\s+access|cannot\s+(?:create|access|log)|'
        r'production\s+(?:down|issue)|business\s+(?:stopped|critical))\b',
        combined,
    ))
    high_time = bool(re.search(
        r'\b(?:asap|immediately|right\s+now|urgent|urgently|emergency|fix\s+asap|urgency\s+ticket)\b',
        combined,
    ))
    high_emotion = bool(re.search(
        r'\b(?:frustrated|pissed|fed\s+up|third\s+time|again|still\s+(?:not|broken)|'
        r'keep\s+having|every\s+time|unacceptable|ridiculous)\b',
        combined,
    ))
    med_impact = bool(re.search(r'\b(?:slow|partial|intermittent|degraded|some\s+users?)\b', combined))

    if high_impact or high_time or (high_emotion and med_impact):
        return 'high'
    if high_emotion or med_impact:
        return 'medium'
    return 'low'

def _best_priority(*candidates: Optional[str]) -> str:
    rank = {'high': 2, 'medium': 1, 'low': 0}
    valid = [c for c in candidates if c in rank]
    return max(valid, key=lambda x: rank[x]) if valid else DEFAULT_PRIORITY

# ── Update field helpers ──────────────────────────────────────────────────────

def _field_from_text(text: str) -> Tuple[str, str]:
    """Return (field_key, value) from an update message."""
    t = _normalize(text)

    # Pattern: "add/update comment/note: <value>" or "comment: <value>"
    m = re.search(
        r'(?:add|update|put|append|include)?\s*'
        r'(work\s*notes?|notes?|comment|comments?|description|details?)\s*'
        r'(?:to|on|for|in\s+(?:this|the)\s+ticket)?\s*'
        r'(?:inc\d{4,})?\s*'
        r'(?:saying\s+(?:that\s*)?[:\-]?|[:\-])\s*(.+)$',
        text, re.IGNORECASE,
    )
    if m:
        kind = _normalize(m.group(1))
        val  = m.group(2).strip()
        return (_kind_to_field(kind), val if _is_meaningful(val) else "")

    # Pattern without value: "add a comment to ticket"
    m2 = re.search(
        r'(?:add|update|put|append)\s+'
        r'(?:a\s+|an\s+|the\s+|some\s+)?'
        r'(work\s*notes?|notes?|comment|comments?|description|details?)\b'
        r'(?:\s+(?:to|on|for|in)\s+(?:this|the|my)\s+(?:ticket|incident))?'
        r'(?:\s+inc\d{4,})?\s*[.?]?\s*$',
        text, re.IGNORECASE,
    )
    if m2:
        return (_kind_to_field(_normalize(m2.group(1))), "")

    # Priority
    if re.search(r'\b(?:priority|urgency)\b', t):
        p = _extract_priority(text)
        if p:
            return ("priority", p)

    # State
    if re.search(r'\b(?:resolve|close|done|complete)\b', t):  return ("state", "resolved")
    if re.search(r'\bon\s+hold\b', t):                         return ("state", "on hold")
    if re.search(r'\bin\s+progress\b', t):                     return ("state", "in progress")

    return ("", "")

def _kind_to_field(kind: str) -> str:
    if "work" in kind or kind in ("note", "notes"):  return "work_notes"
    if "comment" in kind:                            return "comment"
    if "description" in kind or "detail" in kind:   return "description"
    return ""

def _extract_inc(text: str) -> Optional[str]:
    m = re.search(r'\bINC\d{4,}\b', text, re.IGNORECASE)
    return m.group(0).upper() if m else None


# ── Agent ─────────────────────────────────────────────────────────────────────

def run_agent(state: AgentState) -> AgentState:
    messages = state['messages']
    sm       = get_session_manager()
    session  = sm.get_or_create(state.get('session_id', 'default'))

    # Opening message — no human turn yet
    if not any(isinstance(m, HumanMessage) for m in messages):
        tc    = session.get_ticket_context()
        reply = _reply(
            "", "Hello! I'm your IT support assistant. I can create tickets, check status, update fields, or resolve incidents. What can I help you with today?",
            ticket_context=tc, intent="general",
        )
        state['messages'] = list(messages) + [AIMessage(content=reply)]
        return state

    raw = messages[-1].content
    if isinstance(raw, list):
        raw = ' '.join(str(i.get('text', i)) if isinstance(i, dict) else str(i) for i in raw)
    last_msg = raw.strip()

    name        = session.user_info.get('name', '')
    email       = session.user_info.get('email', '')
    description = session.user_info.get('issue', '')
    in_progress = session.user_info.get('ticket_in_progress', False)
    f_name      = first_name(name)
    history_str = get_history(messages[:-1], session=session)

    # Clear just_created_ticket flag from previous turn
    if session.user_info.get('just_created_ticket'):
        session.user_info.pop('just_created_ticket', None)
        in_progress = False
        description = ''
        sm.save_all()

    # Inject returning-user context once
    returning_ctx = session.get_returning_user_context()
    if returning_ctx and not session.user_info.get('_returning_ctx_injected'):
        history_str = f"[Previous session]\n{returning_ctx}\n[Now]\n{history_str}"
        session.user_info['_returning_ctx_injected'] = True

    tc = session.get_ticket_context()

    def reply(fallback: str, intent: str = "general") -> str:
        return _reply(last_msg, fallback, ticket_context=tc, intent=intent)

    # ── Fast-path social guards ───────────────────────────────────────────────

    if _is_greeting(last_msg):
        ctx = "Returning user." if returning_ctx else ""
        return _send(session, messages, last_msg,
                     reply(f"Hi{', ' + f_name if f_name else ''}! How can I help you today?", "general"), state)

    if _is_farewell(last_msg):
        return _send(session, messages, last_msg,
                     reply(f"Take care{', ' + f_name if f_name else ''}! Come back anytime.", "general"), state)

    if _is_ack(last_msg):
        for k in ('ticket_in_progress', 'issue', 'ask_count'):
            session.user_info.pop(k, None)
        sm.save_all()
        return _send(session, messages, last_msg,
                     reply("You’re welcome — feel free to reach out if you need anything.", "general"), state)

    if _is_decline(last_msg):
        for k in ('ticket_in_progress', 'issue', 'ask_count'):
            session.user_info.pop(k, None)
        sm.save_all()
        return _send(session, messages, last_msg,
                     reply("Understood. Come back whenever you need help.", "general"), state)

    if _OOS_RE.search(last_msg):
        return _send(session, messages, last_msg, _OOS_MSG, state)

    # =========================================================================
    # HARD ROUTING — runs before LLM, cannot be overridden
    # =========================================================================

    # 1. Latest comment → dedicated endpoint
    if _COMMENT_RE.search(last_msg):
        tn = _extract_inc(last_msg) or find_ticket_number(messages, session, None)
        if tn:
            return _send(session, messages, last_msg, get_servicenow_latest_comment(tn), state)
        return _send(session, messages, last_msg,
                     reply("Please share the ticket number and I'll fetch the latest comment.", "check_status"), state)

    # 2. Status / details → check_status
    force_check = _STATUS_RE.search(last_msg) or _LATEST_UPDATE_RE.search(last_msg)

    # =========================================================================
    # PRIORITY PRE-COMPUTATION
    # =========================================================================
    kw_priority  = _extract_priority(last_msg)
    sem_priority = _semantic_priority(last_msg, history_str)
    early_priority = _best_priority(kw_priority, sem_priority)

    # =========================================================================
    # PRE-LLM ROUTING — handle clear-cut cases without calling the classifier
    # =========================================================================
    prebuilt: Optional[Dict] = None
    pending = session.user_info.get("pending_update") or {}

    # A. Pending update flow
    if not force_check and pending.get("active"):
        inc_in_msg = _extract_inc(last_msg)
        uf, uv = _field_from_text(last_msg)

        if pending.get("ticket_number") and pending.get("update_field") and pending.get("update_value"):
            if _normalize(last_msg) in ('yes','yep','yeah','y','ok','okay','sure','correct'):
                prebuilt = {
                    'intent': 'update_ticket',
                    'ticket_number': str(pending["ticket_number"]).upper(),
                    'update_field':  pending["update_field"],
                    'update_value':  pending["update_value"],
                }
        if not prebuilt and pending.get("ticket_number") and uf and uv:
            prebuilt = {
                'intent': 'update_ticket',
                'ticket_number': str(pending["ticket_number"]).upper(),
                'update_field': uf, 'update_value': uv,
            }
        if not prebuilt and pending.get("ticket_number") and not pending.get("update_field") and uf:
            session.user_info["pending_update"]["update_field"] = uf
            sm.save_all()
            prebuilt = {
                'intent': 'update_ticket',
                'ticket_number': str(pending["ticket_number"]).upper(),
                'update_field': uf, 'update_value': uv,
            }
        if not prebuilt and inc_in_msg:
            prebuilt = {
                'intent': 'update_ticket', 'ticket_number': inc_in_msg,
                'update_field': pending.get("update_field") or uf or "",
                'update_value': pending.get("update_value") or uv or "",
            }
        if not prebuilt and pending.get("update_field") and not pending.get("update_value"):
            candidate = last_msg.strip()
            if _is_meaningful(candidate) and not _UPDATE_RE.search(candidate):
                prebuilt = {
                    'intent': 'update_ticket',
                    'ticket_number': pending.get("ticket_number") or "",
                    'update_field': pending["update_field"],
                    'update_value': candidate,
                }

    # B. Explicit create request
    if not prebuilt and not force_check and _CREATE_RE.search(last_msg):
        prebuilt = {'intent': 'create_ticket', 'priority': early_priority}

    # C. Explicit update request against last ticket
    if not prebuilt and not force_check and _UPDATE_RE.search(last_msg):
        last_t = session.get_last_ticket()
        if last_t and last_t.get("number"):
            uf, uv = _field_from_text(last_msg)
            if uf:
                prebuilt = {
                    'intent': 'update_ticket',
                    'ticket_number': _extract_inc(last_msg) or last_t["number"],
                    'update_field': uf, 'update_value': uv,
                }

    # Override prebuilt intent to check_status if forced
    if force_check and prebuilt:
        prebuilt = {'intent': 'check_status', 'ticket_number': prebuilt.get('ticket_number')}

    # =========================================================================
    # LLM CLASSIFICATION (only when no prebuilt)
    # =========================================================================
    data: Dict = {}
    if prebuilt:
        data = prebuilt
    else:
        try:
            raw_prompt = CLASSIFY_PROMPT.format(
                in_progress=in_progress, has_issue=bool(description), msg=last_msg,
            )
            resp = _classifier.invoke([SystemMessage(content=raw_prompt)])
            raw_text = resp.content if isinstance(resp.content, str) else str(resp.content)
            raw_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
            print(f"[classifier] {repr(raw_text[:200])}")
            data = safe_json(raw_text)
        except Exception as e:
            print(f"[classifier] error: {e}")

    # Fallback when classifier returns nothing
    if not data:
        if in_progress and description:
            data = {'intent': 'create_ticket'}
        elif re.search(r'\bINC\d+\b', last_msg, re.IGNORECASE):
            data = {'intent': 'check_status'}
        else:
            data = {'intent': 'general'}

    # Apply force_check on LLM result too
    if force_check:
        raw_intent = str(data.get('intent') or '').lower()
        if raw_intent not in ('check_status',):
            data = dict(data); data['intent'] = 'check_status'

    # =========================================================================
    # SANITIZE
    # =========================================================================
    intent    = str(data.get('intent') or 'general').strip().lower()
    new_desc  = data.get('description')
    exit_flow = _parse_bool(data.get('exit', False))

    if intent not in ('create_ticket', 'check_status', 'update_ticket',
                      'resolve_ticket', 'general', 'out_of_scope'):
        intent = 'general'

    llm_priority = str(data.get('priority') or DEFAULT_PRIORITY).strip().lower()
    llm_priority = _PRIORITY_ALIASES.get(llm_priority, llm_priority)
    if llm_priority not in ('high', 'medium', 'low'):
        llm_priority = DEFAULT_PRIORITY

    # Final priority: best of keyword, LLM, semantic
    priority = _best_priority(kw_priority, llm_priority, sem_priority)

    raw_tn = str(data.get('ticket_number') or '').strip()
    ticket_no = raw_tn.upper() if re.match(r'^INC\d{4,}$', raw_tn, re.IGNORECASE) else None

    raw_uf = str(data.get('update_field') or '').strip().lower()
    upd_field = '' if raw_uf in ('null', 'none', '') else raw_uf
    raw_uv = str(data.get('update_value') or '').strip()
    upd_value = '' if raw_uv.lower() in ('null', 'none', '') else raw_uv

    # Augment from text if LLM missed it
    if intent == 'update_ticket' and (not upd_field or not upd_value):
        inf_f, inf_v = _field_from_text(last_msg)
        if not upd_field and inf_f: upd_field = inf_f
        if not upd_value and inf_v: upd_value = inf_v

    # Description sanitize
    if new_desc is not None:
        s = str(new_desc).strip()
        new_desc = None if s.lower() in ('', 'null', 'none') else s

    # Strip priority/urgency words from description
    if new_desc:
        new_desc = re.sub(
            r'\b(?:please\s+)?(?:set|mark)\s+(?:it|this)?\s*(?:as\s+)?'
            r'(?:critical|urgent|high\s+priority|p[01])\b[,.]?\s*',
            '', new_desc, flags=re.IGNORECASE,
        ).strip().rstrip(',').strip()
        if new_desc and not new_desc.endswith('.'):
            new_desc += '.'

    if new_desc and not description:
        description = new_desc
        session.user_info['issue'] = description

    if exit_flow and intent == 'create_ticket':
        exit_flow = False
    if exit_flow and in_progress:
        for k in ('ticket_in_progress', 'issue', 'ask_count'):
            session.user_info.pop(k, None)
        in_progress = False
        description = ''

    print(f"[agent] intent={intent} | priority={priority} | flow={in_progress} | desc={bool(description)}")

    # =========================================================================
    # ROUTING
    # =========================================================================

    if intent == 'out_of_scope':
        return _send(session, messages, last_msg, _OOS_MSG, state)

    if intent == 'general':
        return _send(session, messages, last_msg,
                     reply("I'm here to help with IT support tickets.", "general"), state)

    # ── Resolve ───────────────────────────────────────────────────────────────
    if intent == 'resolve_ticket':
        tn = find_ticket_number(messages, session, ticket_no)
        if tn:
            return _send(session, messages, last_msg, update_servicenow_ticket(tn, 'state', 'resolved'), state)
        return _send(session, messages, last_msg,
                     reply("Which ticket would you like to resolve? Please share the INC number.", "resolve_ticket"), state)

    # ── Update ────────────────────────────────────────────────────────────────
    if intent == 'update_ticket':
        tn = find_ticket_number(messages, session, ticket_no)

        def _do_update(t: str, f: str, v: str) -> str:
            if f == 'priority':
                v = _PRIORITY_ALIASES.get(v.lower(), v.lower())
                if v not in ('high', 'medium', 'low'): v = DEFAULT_PRIORITY
            return update_servicenow_ticket(t, f, v)

        if not tn:
            session.user_info["pending_update"] = {
                "active": True, "ticket_number": "",
                "update_field": upd_field, "update_value": upd_value,
            }
            sm.save_all()
            return _send(session, messages, last_msg,
                         reply("Which ticket number would you like to update?", "update_ticket"), state)

        if not upd_field:
            session.user_info["pending_update"] = {
                "active": True, "ticket_number": tn, "update_field": "", "update_value": "",
            }
            sm.save_all()
            return _send(session, messages, last_msg,
                         reply(f"What would you like to update on **{tn}** — priority, comment, work notes, description, or state?", "update_ticket"), state)

        if not upd_value:
            session.user_info["pending_update"] = {
                "active": True, "ticket_number": tn, "update_field": upd_field, "update_value": "",
            }
            sm.save_all()
            label = upd_field.replace('_', ' ')
            if upd_field == 'description':
                fb = f"What should the new description say for **{tn}**? (Note: this replaces the current one.)"
            elif upd_field in ('comment', 'work_notes'):
                fb = f"What would you like the {label} to say for **{tn}**?"
            else:
                fb = f"What should the **{label}** be set to for **{tn}**?"
            return _send(session, messages, last_msg, reply(fb, "update_ticket"), state)

        result = _do_update(tn, upd_field, upd_value)
        session.user_info.pop("pending_update", None)
        sm.save_all()
        return _send(session, messages, last_msg, result, state)

    # ── Check status ──────────────────────────────────────────────────────────
    if intent == 'check_status':
        tn = find_ticket_number(messages, session, ticket_no)
        if tn:
            return _send(session, messages, last_msg, get_servicenow_ticket_status(tn), state)
        return _send(session, messages, last_msg,
                     reply("Please share the ticket number and I'll check the status.", "check_status"), state)

    # ── Create ticket ─────────────────────────────────────────────────────────
    if intent == 'create_ticket':
        session.user_info['ticket_in_progress'] = True

        if not name or not email:
            session.user_info.pop('ticket_in_progress', None)
            return _send(session, messages, last_msg,
                         "Could not load your account details. Please refresh and try again.", state)

        if not description:
            ask_count = session.user_info.get('ask_count', 0)
            session.user_info['ask_count'] = ask_count + 1
            # Sanitise history — remove INC refs so LLM can't hallucinate ticket numbers
            safe_hist = re.sub(r'\bINC\d{4,}\b', '[ticket]', history_str)
            fallback = (
                f"{'Could you' if ask_count else 'Of course! Could you'} describe the issue? "
                "Which system or app is affected and what error are you seeing?"
            )
            resp_prompt = RESPOND_PROMPT.format(
                ticket_context=tc or "none",
                intent="create_ticket",
                msg=last_msg,
            )
            lm_reply = _llm(last_msg, ticket_context=tc, intent="create_ticket")
            # Safety: discard if LLM hallucinated an INC number
            if re.search(r'\bINC\d{4,}\b', lm_reply):
                lm_reply = ''
            return _send(session, messages, last_msg, lm_reply or fallback, state)

        now = time.time()
        if now - session.user_info.get('last_ticket_created_at', 0) < 10:
            return _send(session, messages, last_msg,
                         "Ticket was just submitted — please wait a moment before retrying.", state)

        session.user_info['last_ticket_created_at'] = now
        session.user_info['just_created_ticket']    = True
        session.user_info['ticket_in_progress']     = False
        session.user_info.pop("create_flow_start_idx", None)
        for k in ('issue', 'ask_count'):
            session.user_info.pop(k, None)
        sm.save_all()

        print(f"[agent] Creating ticket | user={name} | priority={priority}")
        # Always return tool reply directly — never LLM-wrap the creation confirmation
        tool_reply = create_servicenow_ticket(name, email, description, priority)
        tn_match = re.search(r'INC\d+', tool_reply)
        if tn_match:
            session.record_ticket(tn_match.group(0), description, priority)
        sm.save_all()
        return _send(session, messages, last_msg, tool_reply, state)

    # Fallback
    return _send(session, messages, last_msg,
                 reply("I'm here to help with IT support tickets.", "general"), state)


def create_ticket_agent():
    g = StateGraph(AgentState)
    g.add_node("agent", run_agent)
    g.set_entry_point("agent")
    g.add_edge("agent", END)
    return g.compile()

ticket_agent = create_ticket_agent()
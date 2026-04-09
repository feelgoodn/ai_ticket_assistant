"""
On startup / page load:
  1. Reads Authorization: Bearer <jwt> header
  2. Calls GET http://localhost:5001/api/verify to validate token
  3. Receives { user: { name, email, role } }
  4. Pre-populates the agent session with name + email

Or call manually at session start:
  user = verify_and_inject_user(token, session_id)

Security notes:
  - Tokens are accepted ONLY from the Authorization header.
  - Query-string and body tokens are intentionally NOT supported: they appear
    in server access logs, browser history, and referrer headers.
"""

import requests
from functools import wraps
from flask import request, jsonify, g
from agent.memory import get_session_manager
from agent.config import AUTH_SERVER_URL


def verify_and_inject_user(token: str, session_id: str) -> dict | None:
    """
    Call the auth server to validate the JWT token.
    If valid, inject name + email into the agent session so the agent
    never needs to ask for them.

    Returns the user dict on success, None on failure.
    """
    try:
        r = requests.get(
            f"{AUTH_SERVER_URL}/api/verify",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        if not data.get("success"):
            return None

        user = data["user"]   # { sub, email, name, role, iat, exp }

        # ── Inject name + email into agent session ──────────────────────────
        # This means the agent will skip asking for these and go straight
        # to collecting the issue description.
        sm      = get_session_manager()
        session = sm.get_or_create(session_id)

        if user.get("name")  and not session.user_info.get("name"):
            session.user_info["name"]  = user["name"]
        if user.get("email") and not session.user_info.get("email"):
            session.user_info["email"] = user["email"]

        print(f"✅ Auth OK: {user['email']} → session {session_id[:8]}…")
        return user

    except Exception as e:
        print(f"❌ Auth server unreachable: {e}")
        return None


def require_chat_auth(f):
    """
    Flask decorator for chatbot endpoints.
    Validates JWT token from the Authorization header ONLY.

    Deliberately does NOT accept tokens from query strings or the request body
    because those channels expose credentials in server logs and browser history.

    Sets g.chat_user on success.

    Example:
        @app.route("/api/chat", methods=["POST"])
        @require_chat_auth
        def chat():
            user = g.chat_user   # { email, name, role, ... }
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = None

        # Authorization: Bearer <token>  — the only accepted channel
        ah = request.headers.get("Authorization", "")
        if ah.startswith("Bearer "):
            token = ah[7:].strip()

        if not token:
            return jsonify({
                "error": "Authentication required. Please sign in.",
                "code":  "UNAUTHORIZED",
            }), 401

        # session_id comes from the JSON body, not the token
        session_id = (request.get_json(silent=True) or {}).get("session_id", "default")
        user = verify_and_inject_user(token, session_id)

        if not user:
            return jsonify({
                "error": "Session expired. Please sign in again.",
                "code":  "TOKEN_EXPIRED",
            }), 401

        g.chat_user = user
        return f(*args, **kwargs)
    return wrapper


# ── Standalone helper for non-Flask chatbots ─────────────────────────────────

def validate_token_raw(token: str) -> dict | None:
    """
    Simple token validation — no session injection.
    Use this if your chatbot doesn't use Flask sessions.
    Returns user dict or None.
    """
    try:
        r = requests.get(
            f"{AUTH_SERVER_URL}/api/verify",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5,
        )
        if r.status_code == 200:
            data = r.json()
            return data.get("user") if data.get("success") else None
    except Exception:
        pass
    return None
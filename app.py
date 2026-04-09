from flask import Flask, request, jsonify, render_template, redirect, g, make_response
import os
import uuid
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent.config import AUTH_SERVER_URL, APP_PORT, APP_DEBUG, APP_HOST
from agent.graph import ticket_agent
from agent.memory import get_session_manager
from chatbot_middleware import require_chat_auth

load_dotenv()

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ── In-memory agent state store (keyed by session_id) ─────────────────────────
_agent_states: dict = {}


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    token = request.args.get("token", "")
    response = make_response(render_template(
        "chat.html",
        auth_server_url=AUTH_SERVER_URL,
        initial_token=token,
    ))
  
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, private"
    response.headers["Pragma"]        = "no-cache"
    response.headers["Expires"]       = "0"
    return response


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "idmworks-chatbot", "port": APP_PORT}), 200


# ══════════════════════════════════════════════════════════════════════════════
#  CHAT API
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/chat", methods=["POST"])
@require_chat_auth   # ← validates JWT with auth server, injects name+email into agent session
def api_chat():
   
    data       = request.get_json(silent=True) or {}
    user_msg   = (data.get("message") or "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())

    if not user_msg:
        return jsonify({"error": "Message cannot be empty."}), 400

    current_user = g.chat_user
    print(f"💬 [{current_user.get('email')}] session={session_id[:8]}… → {user_msg[:60]}")

    # Get or create agent state for this session
    state = _agent_states.get(session_id, {
        "messages":          [],
        "session_id":        session_id,
        "user_name":         current_user.get("name", ""),
        "user_email":        current_user.get("email", ""),
        "issue_description": "",
        "intent":            "",
        "priority":          "medium",
        "behavior_profile":  {},
    })

    # Make sure session memory also has name + email
    # (chatbot_middleware does this, but belt-and-suspenders here)
    sm      = get_session_manager()
    session = sm.get_or_create(session_id)
    if current_user.get("name") and not session.user_info.get("name"):
        session.user_info["name"]  = current_user["name"]
    if current_user.get("email") and not session.user_info.get("email"):
        session.user_info["email"] = current_user["email"]

    # Add the human message to state
    state["messages"] = list(state["messages"]) + [HumanMessage(content=user_msg)]

    try:
        result = ticket_agent.invoke(state)
    except Exception as exc:
        print(f"❌ Agent error: {exc}")
        return jsonify({"error": "Agent encountered an error. Please try again."}), 500

    # Extract the last AI message
    ai_messages = [
        m for m in result["messages"]
        if hasattr(m, "content") and not isinstance(m, HumanMessage)
    ]
    reply = ai_messages[-1].content if ai_messages else "I'm sorry, something went wrong."

    # Save updated state
    _agent_states[session_id] = result

    return jsonify({
        "response":   reply,
        "session_id": session_id,
    }), 200


@app.route("/api/chat/reset", methods=["POST"])
@require_chat_auth
def api_chat_reset():
    """Clear agent state and session memory for a fresh conversation."""
    data       = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "")

    # Clear agent state
    _agent_states.pop(session_id, None)

    # Clear session memory (but keep name + email — user is still logged in)
    sm      = get_session_manager()
    session = sm.get_or_create(session_id)
    for key in ["issue", "ticket_in_progress", "ask_count"]:
        session.user_info.pop(key, None)
    sm.save_all()

    return jsonify({"success": True}), 200


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    print(f"\n{'='*55}")
    print(f"  Chat page    →  http://{APP_HOST}:{APP_PORT}/")
    print(f"  Auth server  →  {AUTH_SERVER_URL}")
    print(f"{'='*55}\n")
    app.run(host=APP_HOST, port=APP_PORT, debug=APP_DEBUG)
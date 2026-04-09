"""
IDMWORKS Auth Server  —  runs on port 5001
─────────────────────────────────────────
Endpoints
  GET  /                       → login.html
  POST /api/register           → create user in SQLite
  POST /api/login              → returns JWT + redirect URL to :8080
  POST /api/forgot-password    → issues reset token (logs to console in dev)
  GET  /api/verify             → validate JWT  (called by chatbot at :8080)
  GET  /api/user               → fetch full user info from JWT  (called by chatbot)

After login the browser is sent to:
  http://localhost:8080?token=<jwt>

The chatbot calls back:
  GET http://localhost:5001/api/verify?token=<jwt>
  → { success: true, user: { email, name, role, … } }

Then the chatbot stores name + email from the JWT so the agent
"""

import os, sqlite3, hashlib, hmac, base64, time, json, secrets
from pathlib import Path
from functools import wraps
from flask import Flask, request, jsonify, render_template, g
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, template_folder="templates", static_folder="static")

# Allow the chatbot at :8080 to call our API endpoints
CORS(app, resources={r"/api/*": {
    "origins": [
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://0.0.0.0:8080",
    ],
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"],
}})

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH      = Path(os.getenv("AUTH_DB_PATH",    "data/users.db"))
JWT_SECRET   = os.getenv("JWT_SECRET",           secrets.token_hex(32))
JWT_EXPIRY   = int(os.getenv("JWT_EXPIRY_HOURS", "24")) * 3600
CHATBOT_URL  = os.getenv("CHATBOT_URL",          "http://localhost:8080")
AUTH_PORT    = int(os.getenv("AUTH_PORT",        "5001"))


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════════════

def get_db() -> sqlite3.Connection:
    if "db" not in g:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        g.db = sqlite3.connect(str(DB_PATH), detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA journal_mode=WAL")
    return g.db

@app.teardown_appcontext
def close_db(exc=None):
    db = g.pop("db", None)
    if db:
        db.close()

def init_db():
    """Create tables on first run."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT    NOT NULL,
            email         TEXT    NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT    NOT NULL,
            role          TEXT    NOT NULL DEFAULT 'user',
            is_active     INTEGER NOT NULL DEFAULT 1,
            created_at    INTEGER NOT NULL,
            last_login    INTEGER
        );

        CREATE TABLE IF NOT EXISTS password_resets (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            email      TEXT    NOT NULL COLLATE NOCASE,
            token      TEXT    NOT NULL UNIQUE,
            expires_at INTEGER NOT NULL,
            used       INTEGER NOT NULL DEFAULT 0
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email  ON users(email);
        CREATE        INDEX IF NOT EXISTS idx_reset_token  ON password_resets(token);
    """)
    conn.commit()
    conn.close()
    print(f"✅ SQLite database ready → {DB_PATH.resolve()}")


# ══════════════════════════════════════════════════════════════════════════════
# PASSWORD HASHING  (PBKDF2-HMAC-SHA256 — no extra dependencies)
# ══════════════════════════════════════════════════════════════════════════════

def hash_password(password: str) -> str:
    salt = os.urandom(32)
    key  = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 260_000)
    return base64.b64encode(salt + key).decode()

def verify_password(password: str, stored_hash: str) -> bool:
    try:
        raw  = base64.b64decode(stored_hash.encode())
        salt = raw[:32]
        key  = raw[32:]
        test = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 260_000)
        return hmac.compare_digest(key, test)
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# JWT  (pure stdlib — no PyJWT needed)
# ══════════════════════════════════════════════════════════════════════════════

def _b64u_enc(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

def _b64u_dec(s: str) -> bytes:
    padding = 4 - len(s) % 4
    return base64.urlsafe_b64decode(s + "=" * (padding % 4))

def _jwt_sig(unsigned: str) -> str:
    mac = hmac.new(JWT_SECRET.encode(), unsigned.encode(), hashlib.sha256)
    return _b64u_enc(mac.digest())

def create_token(user_id: int, email: str, name: str, role: str) -> str:
    header  = _b64u_enc(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64u_enc(json.dumps({
        "sub":   str(user_id),
        "email": email,
        "name":  name,
        "role":  role,
        "iat":   int(time.time()),
        "exp":   int(time.time()) + JWT_EXPIRY,
    }).encode())
    sig = _jwt_sig(f"{header}.{payload}")
    return f"{header}.{payload}.{sig}"

def verify_token(token: str):
    """Return decoded payload dict, or None if invalid/expired."""
    try:
        parts = token.strip().split(".")
        if len(parts) != 3:
            return None
        header, payload, sig = parts
        if not hmac.compare_digest(sig, _jwt_sig(f"{header}.{payload}")):
            return None
        data = json.loads(_b64u_dec(payload))
        if data.get("exp", 0) < time.time():
            return None
        return data
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# USER OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

def register_user(name: str, email: str, password: str):
    db = get_db()
    if db.execute("SELECT id FROM users WHERE email=?", (email.lower(),)).fetchone():
        return False, "An account with this email already exists."
    db.execute(
        "INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
        (name.strip(), email.lower(), hash_password(password), int(time.time()))
    )
    db.commit()
    return True, "Account created."

def authenticate_user(email: str, password: str):
    db   = get_db()
    user = db.execute(
        "SELECT id, name, email, password_hash, role, is_active FROM users WHERE email=?",
        (email.lower(),)
    ).fetchone()
    if not user:
        return False, "Invalid email or password.", None
    if not user["is_active"]:
        return False, "Account is deactivated. Contact your administrator.", None
    if not verify_password(password, user["password_hash"]):
        return False, "Invalid email or password.", None
    db.execute("UPDATE users SET last_login=? WHERE id=?", (int(time.time()), user["id"]))
    db.commit()
    token = create_token(user["id"], user["email"], user["name"], user["role"])
    return True, "Login successful.", {
        "token":    token,
        "user":     {"id": user["id"], "name": user["name"],
                     "email": user["email"], "role": user["role"]},
        "redirect": f"{CHATBOT_URL}?token={token}",
    }


# ══════════════════════════════════════════════════════════════════════════════
# AUTH MIDDLEWARE
# ══════════════════════════════════════════════════════════════════════════════

def require_auth(f):
    """Protect a route with JWT.  Sets g.current_user on success."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = None
        # 1. Authorization header
        ah = request.headers.get("Authorization", "")
        if ah.startswith("Bearer "):
            token = ah[7:].strip()
        # 2. Query string  (?token=...)
        if not token:
            token = request.args.get("token")
        if not token:
            return jsonify({"error": "Authentication required."}), 401
        payload = verify_token(token)
        if not payload:
            return jsonify({"error": "Token expired or invalid. Please sign in again."}), 401
        g.current_user = payload
        return f(*args, **kwargs)
    return wrapper


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES — Pages
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("login.html", chatbot_url=CHATBOT_URL)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES — Auth API
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/register", methods=["POST"])
def api_register():
    d        = request.get_json(silent=True) or {}
    name     = (d.get("name")     or "").strip()
    email    = (d.get("email")    or "").strip()
    password = (d.get("password") or "").strip()

    errors = {}
    if len(name) < 2:
        errors["name"]     = "Name must be at least 2 characters."
    if not email or "@" not in email or "." not in email.split("@")[-1]:
        errors["email"]    = "Enter a valid email address."
    if len(password) < 8:
        errors["password"] = "Password must be at least 8 characters."
    if errors:
        return jsonify({"success": False, "errors": errors}), 400

    ok, msg = register_user(name, email, password)
    if not ok:
        return jsonify({"success": False, "errors": {"email": msg}}), 409
    return jsonify({"success": True, "message": msg}), 201


@app.route("/api/login", methods=["POST"])
def api_login():
    d        = request.get_json(silent=True) or {}
    email    = (d.get("email")    or "").strip()
    password = (d.get("password") or "").strip()
    if not email or not password:
        return jsonify({"success": False, "message": "Email and password are required."}), 400
    ok, msg, result = authenticate_user(email, password)
    if not ok:
        return jsonify({"success": False, "message": msg}), 401
    return jsonify({"success": True, "message": msg, **result}), 200


@app.route("/api/forgot-password", methods=["POST"])
def api_forgot():
    d     = request.get_json(silent=True) or {}
    email = (d.get("email") or "").strip().lower()
    if not email:
        return jsonify({"success": False, "message": "Email is required."}), 400

    db   = get_db()
    user = db.execute(
        "SELECT id FROM users WHERE email=? AND is_active=1", (email,)
    ).fetchone()

    if user:
        token = _b64u_enc(os.urandom(32))
        db.execute(
            "INSERT INTO password_resets (email, token, expires_at) VALUES (?, ?, ?)",
            (email, token, int(time.time()) + 3600)
        )
        db.commit()
        # In production: send via email.  In dev: log to console.
        reset_url = f"{request.host_url}reset-password?token={token}"
        print(f"\n🔑 [DEV] Password reset link for {email}:\n   {reset_url}\n")

    # Always return 200 — never reveal whether email exists
    return jsonify({
        "success": True,
        "message": "If this email is registered, a reset link has been sent."
    }), 200


@app.route("/api/verify", methods=["GET"])
@require_auth
def api_verify():
    """
    Called by the chatbot at :8080 to validate the token it received.
    Returns the full user payload so the chatbot can pre-fill name/email.
    """
    return jsonify({"success": True, "user": g.current_user}), 200


@app.route("/api/user", methods=["GET"])
@require_auth
def api_user():
    """Return user profile including name + email for the agent to use."""
    u = g.current_user
    return jsonify({
        "id":    u.get("sub"),
        "email": u.get("email"),
        "name":  u.get("name"),
        "role":  u.get("role"),
    }), 200


if __name__ == "__main__":
    init_db()
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    print(f"\n{'='*55}")
    print(f"  IDMWORKS Auth Server")
    print(f"  Login page  →  http://localhost:{AUTH_PORT}/")
    print(f"  After login →  {CHATBOT_URL}")
    print(f"{'='*55}\n")
    app.run(host="0.0.0.0", port=AUTH_PORT, debug=debug)
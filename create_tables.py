"""
One-off helper script to initialize the auth SQLite database.
Delegates to auth_server.init_db() so there is a single authoritative schema.
"""

from auth_server import init_db


if __name__ == "__main__":
    init_db()

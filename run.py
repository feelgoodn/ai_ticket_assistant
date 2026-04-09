import os
import sys
from dotenv import load_dotenv

load_dotenv()


def check_environment() -> bool:
    from agent.config import SNOW_INSTANCE, SNOW_USERNAME, SNOW_PASSWORD

    errors: list[str] = []
    if not SNOW_INSTANCE:
        errors.append("SERVICENOW_INSTANCE")
    if not SNOW_USERNAME:
        errors.append("SERVICENOW_USERNAME")
    if not SNOW_PASSWORD:
        errors.append("SERVICENOW_PASSWORD")

    if errors:
        print("❌ Missing required environment variables:")
        for e in errors:
            print(f"   • {e}")
        return False
    return True


def print_startup_banner() -> None:
    from agent.config import OLLAMA_BASE_URL, OLLAMA_MODEL, APP_HOST, APP_PORT, APP_DEBUG

    print("\n" + "=" * 60)
    print(f"  LLM      : Ollama — {OLLAMA_MODEL}")
    print(f"  Endpoint : {OLLAMA_BASE_URL}")
    print(f"  Server   : http://{APP_HOST}:{APP_PORT}")
    print(f"  Debug    : {APP_DEBUG}")
    print("=" * 60 + "\n")


def main() -> None:
    if not check_environment():
        sys.exit(1)

    from app import app
    from agent.config import APP_HOST, APP_PORT, APP_DEBUG

    print_startup_banner()

    try:
        app.run(host=APP_HOST, port=APP_PORT, debug=APP_DEBUG, threaded=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped.")
    except Exception as exc:
        import traceback
        print(f"\n❌ Failed to start server: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
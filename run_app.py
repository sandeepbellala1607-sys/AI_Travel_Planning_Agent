"""Single-command launcher that starts FastAPI (uvicorn) and Streamlit together."""

import subprocess
import sys
import time
import httpx

API_HEALTH_URL = "http://127.0.0.1:8000/health"


def _backend_is_up() -> bool:
    try:
        return httpx.get(API_HEALTH_URL, timeout=1.5).status_code == 200
    except Exception:
        return False


def main() -> None:
    backend_proc: subprocess.Popen | None = None
    if not _backend_is_up():
        print("Starting FastAPI backend on http://127.0.0.1:8000 ...")
        backend_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ]
        )
        for _ in range(30):
            if _backend_is_up():
                break
            time.sleep(0.5)
        else:
            print("Warning: FastAPI backend did not respond to /health within 15s.")

    print("Starting Streamlit frontend ...")
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "frontend/app.py"],
            check=False,
        )
    finally:
        if backend_proc is not None and backend_proc.poll() is None:
            backend_proc.terminate()
            backend_proc.wait(timeout=5)


if __name__ == "__main__":
    main()

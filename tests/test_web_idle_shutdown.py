"""Dashboard idle auto-shutdown (2026-07-13).

Closing the browser tab used to leave the server holding the port forever.
Now: the page heartbeats every 25 s; a watchdog exits the server when no tab
has spoken for the idle window AND no pipeline run is active. A second
`python -m jobscout.web` start reuses a live instance instead of crashing
with "address already in use".
"""
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ["JOBSCOUT_DB"] = str(Path(tempfile.gettempdir())
                                / "jobscout_webidle_pytest.db")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from jobscout import db, web  # noqa: E402


class _StubServer:
    should_exit = False


def _fresh_db():
    p = Path(os.environ["JOBSCOUT_DB"])
    if p.exists():
        p.unlink()
    db.init_db()


def test_ping_and_heartbeat_endpoints():
    _fresh_db()
    client = TestClient(web.app)
    r = client.get("/api/ping")
    assert r.status_code == 200 and r.json()["app"] == "jobscout"
    r = client.post("/api/heartbeat")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_any_request_counts_as_activity():
    _fresh_db()
    client = TestClient(web.app)
    web._last_activity = time.time() - 999
    client.get("/api/ping")
    assert time.time() - web._last_activity < 5


def test_watchdog_exits_when_idle(monkeypatch):
    stub = _StubServer()
    monkeypatch.setattr(web, "_server", stub)
    monkeypatch.setitem(web._run_state, "running", False)
    web._last_activity = time.time() - 60
    t = threading.Thread(target=web._idle_watchdog, args=(0.05, 0.01),
                         daemon=True)
    t.start()
    t.join(timeout=2)
    assert not t.is_alive()
    assert stub.should_exit is True


def test_watchdog_waits_for_active_run(monkeypatch):
    """A pipeline run kicked off before closing the tab must never be killed;
    idleness starts counting from the run's END."""
    stub = _StubServer()
    monkeypatch.setattr(web, "_server", stub)
    monkeypatch.setitem(web._run_state, "running", True)
    web._last_activity = time.time() - 999
    t = threading.Thread(target=web._idle_watchdog, args=(0.3, 0.02),
                         daemon=True)
    t.start()
    time.sleep(0.2)
    assert t.is_alive() and stub.should_exit is False   # run active -> alive
    monkeypatch.setitem(web._run_state, "running", False)
    t.join(timeout=3)                                    # idle window elapses
    assert not t.is_alive()
    assert stub.should_exit is True


def test_watchdog_stays_alive_with_heartbeats(monkeypatch):
    stub = _StubServer()
    monkeypatch.setattr(web, "_server", stub)
    monkeypatch.setitem(web._run_state, "running", False)
    web._last_activity = time.time()
    t = threading.Thread(target=web._idle_watchdog, args=(0.5, 0.02),
                         daemon=True)
    t.start()
    for _ in range(6):                                   # simulated heartbeats
        web._last_activity = time.time()
        time.sleep(0.05)
    assert t.is_alive() and stub.should_exit is False
    web._last_activity = time.time() - 60                # tabs closed
    t.join(timeout=3)
    assert stub.should_exit is True


def test_existing_instance_probe_false_on_free_port():
    assert web._existing_instance("http://127.0.0.1:59999") is False

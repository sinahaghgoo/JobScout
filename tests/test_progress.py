"""Live run-progress reporting (2026-07-09): the pipeline updates a
thread-safe Progress singleton; /api/run/status serves it; the dashboard
paints a progress bar from it.
"""
import os
import sys
import tempfile
import threading
from datetime import date
from pathlib import Path

os.environ["JOBSCOUT_DB"] = str(Path(tempfile.gettempdir()) / "jobscout_pytest.db")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobscout import config, db, pipeline  # noqa: E402
from jobscout.connectors.base import RawJob  # noqa: E402


def _fresh_db():
    p = Path(os.environ["JOBSCOUT_DB"])
    if p.exists():
        p.unlink()
    db.init_db()


# =============================================================================
# The Progress object itself
# =============================================================================
def test_progress_phases_steps_counters_events():
    pr = pipeline.Progress()
    snap = pr.snapshot()
    assert snap["phase"] is None and snap["done"] == 0 and snap["total"] is None

    pr.phase("fetch", "Fetching 3 feeds", total=3)
    pr.step(1, current="Acme")
    pr.step(1, current="Beta GmbH")
    pr.add(postings_new=5)
    pr.add(postings_new=2)
    pr.set(llm_cost=0.0123)
    for i in range(9):
        pr.event(f"event {i}")
    s = pr.snapshot()
    assert s["phase"] == "fetch" and s["total"] == 3 and s["done"] == 2
    assert s["current"] == "Beta GmbH"
    assert s["counters"]["postings_new"] == 7
    assert s["counters"]["llm_cost"] == 0.0123
    assert len(s["events"]) == 6, "event log is a bounded ring buffer"
    assert s["events"][-1] == "event 8"
    assert s["updated_at"]

    # a new phase resets done/total/current but keeps run counters + events
    pr.phase("score", "Scoring", total=10)
    s2 = pr.snapshot()
    assert s2["done"] == 0 and s2["total"] == 10 and s2["current"] == ""
    assert s2["counters"]["postings_new"] == 7

    # snapshots are isolated copies — mutating one never corrupts the source
    s2["counters"]["postings_new"] = 999
    s2["events"].append("bogus")
    assert pr.snapshot()["counters"]["postings_new"] == 7
    assert "bogus" not in pr.snapshot()["events"]


def test_progress_is_thread_safe():
    pr = pipeline.Progress()
    pr.phase("fetch", "x", total=4000)

    def worker():
        for _ in range(1000):
            pr.step(1, current="w")
            pr.add(postings_new=1)
            pr.event("e")

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    s = pr.snapshot()
    assert s["done"] == 4000
    assert s["counters"]["postings_new"] == 4000
    assert len(s["events"]) == 6


# =============================================================================
# The pipeline reports its phases end to end
# =============================================================================
def test_pipeline_run_reports_phase_sequence(monkeypatch):
    _fresh_db()
    db.save_profile("CV: analytics graduate, SQL, Power BI.",
                    "Prefers junior BI/analyst roles in Vienna.")
    cid = db.upsert_company("Progress Test GmbH")
    db.set_company_ats(cid, "greenhouse", "progresstest", "detected", "test")

    today = date.today().isoformat()
    fake_jobs = [
        RawJob(job_id="p1", title="Junior BI Analyst", location="Vienna, Austria",
               posted_date=today, url="http://x/1",
               description="SQL, Power BI dashboards and reporting for stakeholders."),
        RawJob(job_id="p2", title="Senior Data Analyst", location="Vienna, Austria",
               posted_date=today, url="http://x/2",
               description="Requires leadership."),  # dies on the senior hard filter
    ]
    monkeypatch.setattr(pipeline, "fetch_jobs", lambda ats, token: list(fake_jobs))
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")   # keyless -> heuristic

    seen_phases = []
    orig_phase = pipeline.PROGRESS.phase

    def spy_phase(name, label, total=None):
        seen_phases.append((name, total))
        orig_phase(name, label, total)

    monkeypatch.setattr(pipeline.PROGRESS, "phase", spy_phase)

    summary = pipeline.run(detect_first=False, verbose=False, skip_browser=True)

    names = [n for n, _ in seen_phases]
    for expected in ("refilter", "fetch", "filter", "prepare", "score",
                     "cleanup", "done"):
        assert expected in names, f"phase '{expected}' missing from {names}"
    assert names[-1] == "done"
    # ordering: fetch before filter before score before done
    assert names.index("fetch") < names.index("filter") < names.index("score")

    fetch_total = dict(seen_phases)["fetch"]
    assert fetch_total == 1, "fetch phase sized by number of companies"

    final = pipeline.PROGRESS.snapshot()
    assert final["phase"] == "done"
    assert final["counters"]["postings_new"] == 2
    assert final["counters"]["filtered_out"] >= 1          # the Senior role
    assert final["counters"]["scored"] == summary.scored == 1
    assert any("Progress Test GmbH: 2 open, 2 new" in e for e in final["events"])


def test_fetch_error_lands_in_progress(monkeypatch):
    _fresh_db()
    db.save_profile("cv", "prefs")
    cid = db.upsert_company("Broken Feed AG")
    db.set_company_ats(cid, "greenhouse", "broken", "detected", "test")

    def boom(ats, token):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(pipeline, "fetch_jobs", boom)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    pipeline.run(detect_first=False, verbose=False, skip_browser=True)
    s = pipeline.PROGRESS.snapshot()
    assert s["counters"].get("fetch_errors") == 1
    assert any(e.startswith("✗ Broken Feed AG") for e in s["events"])
    assert s["phase"] == "done", "one broken feed must never stall the run"


# =============================================================================
# The API exposes it
# =============================================================================
def test_run_status_endpoint_exposes_progress():
    from fastapi.testclient import TestClient
    from jobscout import web
    _fresh_db()
    pipeline.PROGRESS.reset()
    pipeline.PROGRESS.phase("fetch", "Fetching 84 feed companies", total=84)
    pipeline.PROGRESS.step(37, current="BMW")
    pipeline.PROGRESS.add(postings_new=123)
    client = TestClient(web.app)
    s = client.get("/api/run/status").json()
    p = s["progress"]
    assert p["phase"] == "fetch"
    assert p["label"] == "Fetching 84 feed companies"
    assert p["done"] == 37 and p["total"] == 84
    assert p["current"] == "BMW"
    assert p["counters"]["postings_new"] == 123
    assert "events" in p and "updated_at" in p


def test_api_run_resets_progress_before_thread_starts(monkeypatch):
    """POST /api/run must never let the first poll see the previous run's
    'done' snapshot."""
    from fastapi.testclient import TestClient
    from jobscout import web
    _fresh_db()
    pipeline.PROGRESS.phase("done", "Run complete")           # previous run
    monkeypatch.setattr(web.threading, "Thread",
                        lambda **kw: type("T", (), {"start": lambda self: None})())
    client = TestClient(web.app)
    web._run_state.update(running=False)
    r = client.post("/api/run").json()
    assert r["ok"]
    p = client.get("/api/run/status").json()["progress"]
    assert p["phase"] == "starting"
    web._run_state.update(running=False)                       # cleanup


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))

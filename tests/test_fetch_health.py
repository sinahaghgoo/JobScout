"""Fetch health: the ways a company can go dark without anything failing.

Found live on 2026-09-01: Mistral AI had moved from Lever to Ashby. The
abandoned Lever board still answered 200 with an empty list, so every run
recorded a clean "empty" fetch and 172 postings stayed invisible for weeks.
An empty board is not an error — but a board that HAD jobs and now has none
is a signal, and the run has to say so.
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ["JOBSCOUT_DB"] = str(Path(tempfile.gettempdir()) / "jobscout_pytest_health.db")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobscout import db, detect, pipeline  # noqa: E402
from jobscout.connectors.base import RawJob  # noqa: E402


def _fresh_db():
    p = Path(os.environ["JOBSCOUT_DB"])
    if p.exists():
        p.unlink()
    db.init_db()


def _company(name, jobs_last_run=None):
    cid = db.upsert_company(name)
    db.set_company_ats(cid, "greenhouse", name.lower(), "detected")
    if jobs_last_run is not None:
        db.set_fetch_health(cid, "ok", jobs_last_run)
    return [c for c in db.get_companies() if c["id"] == cid][0]


def _run_fetch(companies, jobs_by_name):
    summary = pipeline.RunSummary()
    orig = pipeline._fetch_one
    pipeline._fetch_one = lambda c: list(jobs_by_name.get(c["name"], []))
    try:
        pipeline._fetch_and_store(summary, False, companies)
    finally:
        pipeline._fetch_one = orig
    return summary


def test_a_board_that_went_from_jobs_to_zero_is_flagged():
    _fresh_db()
    c = _company("MovedCo", jobs_last_run=172)
    summary = _run_fetch([c], {})
    note = [n for n in summary.notes if n.startswith("MovedCo")]
    assert len(note) == 1
    assert "0 jobs, but 172 last run" in note[0]
    assert "may have moved" in note[0]


def test_a_board_that_was_always_empty_is_not_flagged():
    """Some target boards are legitimately empty (UNOV had no openings at all
    on the day this was written). Crying wolf every run would be worse."""
    _fresh_db()
    c = _company("QuietCo", jobs_last_run=0)
    assert _run_fetch([c], {}).notes == []


def test_a_first_ever_fetch_is_not_flagged():
    _fresh_db()
    c = _company("NewCo")           # never fetched — last_fetch_jobs is NULL
    assert _run_fetch([c], {}).notes == []


def test_an_empty_fetch_still_never_closes_a_healthy_board():
    """The flag must not change the existing safety rule."""
    _fresh_db()
    c = _company("SafeCo", jobs_last_run=2)
    pid, _ = db.upsert_posting(company_id=c["id"], job_id="a", title="Analyst",
                               location="Vienna", posted_date=None,
                               url="https://x/a", description="d")
    _run_fetch([c], {})
    with db.connect() as conn:
        assert conn.execute("SELECT is_open FROM posting WHERE id=?",
                            (pid,)).fetchone()[0] == 1


def test_a_board_with_jobs_is_quiet():
    _fresh_db()
    c = _company("BusyCo", jobs_last_run=5)
    jobs = [RawJob("1", "Data Analyst", "Vienna, Austria", None,
                   "https://x/1", "SQL")]
    summary = _run_fetch([c], {"BusyCo": jobs})
    assert summary.notes == [] and summary.companies_fetched == 1


# --- the Mistral fix itself ---------------------------------------------------
def test_mistral_is_pinned_to_ashby():
    """Guards the specific regression: the Lever board is gone but still
    answers 200-with-nothing, so only the curated map can catch this."""
    assert detect._RAW_KNOWN_TOKENS["Mistral AI"] == ("ashby", "mistral.ai")


# --- Workday pagination -------------------------------------------------------
# Found 2026-09-01 by diffing per-company job counts across runs: ABB reported a
# healthy "ok" fetch of 40 jobs where the previous run had 1971, and NVIDIA the
# same. Both are Workday tenants that answer `total: 2000` on page 1 (a saturated
# floor, not a count) and `total: 0` on every page after it, while still serving
# 20 real postings per page. The connector read `total` on every iteration while
# it was None, so page 2's `0` overwrote the "walk until empty" decision and
# `offset >= total` ended the walk two pages in. 3,959 postings, silently gone.
def _wd_page(n=20, total=0, start=0):
    return {"total": total,
            "jobPostings": [{"title": f"Role {start+i}",
                             "externalPath": f"/job/x/R{start+i}",
                             "postedOn": "Posted Today",
                             "locationsText": "Vienna, Austria"}
                            for i in range(n)]}


def _walk(monkeypatch, pages):
    """Run the Workday connector against a scripted sequence of pages."""
    from jobscout.connectors import workday as wd

    calls = {"n": 0}

    class _Resp:
        def __init__(self, d): self._d = d
        def json(self): return self._d

    def fake_get(url, method=None, json_body=None, **kw):
        i = json_body["offset"] // 20
        calls["n"] += 1
        return _Resp(pages(i))
    monkeypatch.setattr(wd, "http_get", fake_get)
    return wd.WorkdayConnector().fetch("t|wd3|Site"), calls


def test_a_saturated_total_walks_past_page_two(monkeypatch):
    """The ABB/NVIDIA shape: 2000 then 0, 20 real postings a page throughout."""
    def pages(i):
        if i >= 5:
            return {"total": 0, "jobPostings": []}
        return _wd_page(total=2000 if i == 0 else 0, start=i * 20)
    jobs, calls = _walk(monkeypatch, pages)
    assert len(jobs) == 100, f"truncated to {len(jobs)}"


def test_a_zero_total_with_a_full_page_is_not_an_empty_board(monkeypatch):
    """total=0 on page ONE alongside real postings is the API being wrong, not
    a board with no jobs."""
    def pages(i):
        if i >= 3:
            return {"total": 0, "jobPostings": []}
        return _wd_page(total=0, start=i * 20)
    jobs, _ = _walk(monkeypatch, pages)
    assert len(jobs) == 60


def test_an_honest_total_still_stops_the_walk(monkeypatch):
    """Small tenants report a truthful total — that must still end the walk
    without fetching a wasted page."""
    def pages(i):
        return _wd_page(n=20 if i == 0 else 5, total=25, start=i * 20)
    jobs, calls = _walk(monkeypatch, pages)
    assert len(jobs) == 25 and calls["n"] == 2


def test_an_empty_page_ends_the_walk(monkeypatch):
    def pages(i):
        return _wd_page(total=2000, start=0) if i == 0 else {"total": 2000,
                                                             "jobPostings": []}
    jobs, calls = _walk(monkeypatch, pages)
    assert len(jobs) == 20 and calls["n"] == 2

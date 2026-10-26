"""Selective-company pipeline runs + the 2026-07-11 audit fixes.

Covers:
  * resolve_company_selection (exact / case-insensitive / substring / errors)
  * run(only=...) fetches, scores and janitors ONLY the selected companies
  * scoped db queries (get_postings_to_score / recs_to_validate)
  * /api/run company validation (400 on unknown names)
  * /api/archive contents + stats()['rejected']
  * upsert_posting refresh semantics (longer description wins; empty values
    never blank enriched location/url)
  * browser cache rematerialization keeps MAX_EXTRACT_JOBS postings
  * verify.py treats Workday's saturated total=2000 as unverifiable, not a
    MISMATCH
"""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["JOBSCOUT_DB"] = str(Path(tempfile.gettempdir()) / "jobscout_partial_pytest.db")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from jobscout import config, db, pipeline  # noqa: E402

TODAY = __import__("datetime").date.today().isoformat()
DESC = ("You will build SQL queries and Power BI dashboards, own the "
        "reporting for commercial stakeholders and present insights.")


def _fresh_db():
    p = Path(os.environ["JOBSCOUT_DB"])
    if p.exists():
        p.unlink()
    db.init_db()


def _keyless(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")


def _seed_two_companies():
    ca = db.upsert_company("Alpha Analytics AG")
    cb = db.upsert_company("Beta Insights GmbH")
    db.set_company_ats(ca, "greenhouse", "alpha", "detected", "test")
    db.set_company_ats(cb, "greenhouse", "beta", "detected", "test")
    return ca, cb


# --- resolve_company_selection -------------------------------------------------
def test_selection_exact_case_insensitive_and_substring():
    _fresh_db()
    ca, cb = _seed_two_companies()
    rows = pipeline.resolve_company_selection(["alpha analytics ag"])
    assert [r["id"] for r in rows] == [ca]
    rows = pipeline.resolve_company_selection(["BETA"])          # unique substring
    assert [r["id"] for r in rows] == [cb]
    rows = pipeline.resolve_company_selection(["Alpha Analytics AG", "beta"])
    assert sorted(r["id"] for r in rows) == sorted([ca, cb])
    # duplicates collapse
    rows = pipeline.resolve_company_selection(["alpha", "ALPHA analytics ag"])
    assert len(rows) == 1


def test_selection_unknown_and_ambiguous_raise():
    _fresh_db()
    _seed_two_companies()
    with pytest.raises(pipeline.UnknownCompanyError):
        pipeline.resolve_company_selection(["Gamma Corp"])
    with pytest.raises(pipeline.UnknownCompanyError):
        # "a" matches both Alpha and Beta -> ambiguous
        pipeline.resolve_company_selection(["a"])


# --- scoped db queries ----------------------------------------------------------
def test_scoped_posting_and_validation_queries():
    _fresh_db()
    ca, cb = _seed_two_companies()
    pa, _ = db.upsert_posting(ca, "a1", "Business Analyst", "Vienna, Austria",
                              TODAY, "u", DESC)
    pb, _ = db.upsert_posting(cb, "b1", "Data Analyst", "Berlin, Germany",
                              TODAY, "u", DESC)
    rows = db.get_postings_to_score(company_ids=[ca])
    assert [r["id"] for r in rows] == [pa]
    assert db.get_postings_to_score(company_ids=[]) == []
    assert {r["id"] for r in db.get_postings_to_score()} == {pa, pb}

    db.save_recommendation(pa, 70, 80, "r", "None", llm_used=True)
    db.save_recommendation(pb, 70, 80, "r", "None", llm_used=True)
    assert [r["posting_id"] for r in db.recs_to_validate(company_ids=[cb])] == [pb]
    assert len(db.recs_to_validate()) == 2


# --- selective pipeline run ------------------------------------------------------
def test_run_only_fetches_and_scores_selected_company(monkeypatch):
    _fresh_db()
    _keyless(monkeypatch)
    ca, cb = _seed_two_companies()
    # Pre-existing queued posting for Beta — a scoped Alpha run must NOT touch it.
    pb_old, _ = db.upsert_posting(cb, "b-old", "Reporting Analyst",
                                  "Berlin, Germany", TODAY, "u", DESC)

    fetched: list[str] = []

    def fake_fetch_one(company):
        from jobscout.connectors.base import RawJob
        fetched.append(company["name"])
        return [RawJob(job_id="a-new", title="Junior BI Analyst",
                       location="Vienna, Austria", posted_date=TODAY,
                       url="http://x", description=DESC)]

    monkeypatch.setattr(pipeline, "_fetch_one", fake_fetch_one)
    summary = pipeline.run(verbose=False, only=["Alpha Analytics AG"])

    assert fetched == ["Alpha Analytics AG"], "only the selected company is fetched"
    assert any(n.startswith("partial run:") for n in summary.notes)
    # Alpha's posting got scored; Beta's stayed queued, unscored and unfiltered.
    with db.connect() as conn:
        alpha_recs = conn.execute(
            "SELECT COUNT(*) FROM recommendation r JOIN posting p ON p.id=r.posting_id "
            "WHERE p.company_id=?", (ca,)).fetchone()[0]
        beta_recs = conn.execute(
            "SELECT COUNT(*) FROM recommendation r JOIN posting p ON p.id=r.posting_id "
            "WHERE p.company_id=?", (cb,)).fetchone()[0]
        beta_filtered = conn.execute(
            "SELECT filtered_reason FROM posting WHERE id=?", (pb_old,)).fetchone()[0]
    assert alpha_recs == 1
    assert beta_recs == 0, "a scoped run must not score other companies' postings"
    assert beta_filtered is None, "a scoped run must not filter other companies' postings"


def test_scoped_janitor_leaves_other_companies_alone(monkeypatch):
    """The 10-day janitor must be scoped too — a daily 'Apple only' habit must
    never age out other companies' queued postings before a full run."""
    from datetime import datetime, timedelta, timezone
    _fresh_db()
    _keyless(monkeypatch)
    ca, cb = _seed_two_companies()
    pb, _ = db.upsert_posting(cb, "b-stale", "Insights Analyst",
                              "Berlin, Germany", TODAY, "u", DESC)
    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(timespec="seconds")
    with db.connect() as conn:
        conn.execute("UPDATE posting SET first_seen=? WHERE id=?", (old, pb))
    monkeypatch.setattr(pipeline, "_fetch_one", lambda c: [])
    pipeline.run(verbose=False, only=["Alpha Analytics AG"])
    with db.connect() as conn:
        reason = conn.execute("SELECT filtered_reason FROM posting WHERE id=?",
                              (pb,)).fetchone()[0]
    assert reason is None, "scoped run aged out another company's posting"
    # ... and the next FULL run gives it its verdict (keyless -> heuristic
    # scores everything, so the janitor never needs to touch it).
    pipeline.run(verbose=False)
    with db.connect() as conn:
        n_rec = conn.execute(
            "SELECT COUNT(*) FROM recommendation WHERE posting_id=?",
            (pb,)).fetchone()[0]
        reason = conn.execute("SELECT filtered_reason FROM posting WHERE id=?",
                              (pb,)).fetchone()[0]
    assert n_rec == 1 and reason is None, \
        "the full run after a scoped run must still score the waiting posting"


def test_run_unknown_company_raises(monkeypatch):
    _fresh_db()
    _keyless(monkeypatch)
    _seed_two_companies()
    with pytest.raises(pipeline.UnknownCompanyError):
        pipeline.run(verbose=False, only=["Nonexistent Corp"])


# --- web API ----------------------------------------------------------------------
def test_api_run_rejects_unknown_company(monkeypatch):
    _fresh_db()
    _seed_two_companies()
    from fastapi.testclient import TestClient
    from jobscout import web
    client = TestClient(web.app)
    r = client.post("/api/run", json={"companies": ["Nonexistent Corp"]})
    assert r.status_code == 400
    assert "matches no company" in r.json()["detail"]
    # and the run slot was NOT claimed by the failed request
    assert web._run_state["running"] is False


def test_api_run_scoped_passes_names_through(monkeypatch):
    _fresh_db()
    _seed_two_companies()
    from fastapi.testclient import TestClient
    from jobscout import web

    captured = {}

    def fake_run_tiered(detect_first=True, verbose=False, only=None):
        captured["only"] = only
        return pipeline.RunSummary()

    monkeypatch.setattr(web.pipeline, "run_tiered", fake_run_tiered)
    client = TestClient(web.app)
    r = client.post("/api/run", json={"companies": ["alpha"]})
    assert r.status_code == 200 and r.json()["ok"]
    assert r.json()["companies"] == ["Alpha Analytics AG"]
    # wait for the worker thread to finish
    import time
    for _ in range(100):
        if not web._run_state["running"]:
            break
        time.sleep(0.02)
    assert captured["only"] == ["Alpha Analytics AG"]


def test_api_archive_and_stats_rejected():
    _fresh_db()
    ca, _ = _seed_two_companies()
    p1, _ = db.upsert_posting(ca, "x1", "Analyst 1", "Vienna", TODAY, "u", DESC)
    p2, _ = db.upsert_posting(ca, "x2", "Analyst 2", "Vienna", TODAY, "u", DESC)
    p3, _ = db.upsert_posting(ca, "x3", "Analyst 3", "Vienna", TODAY, "u", DESC)
    for pid in (p1, p2, p3):
        db.save_recommendation(pid, 70, 80, "r", "None", llm_used=True)
    rows = db.board_rows()
    db.set_status(rows[0]["rec_id"], "Skipped")
    db.set_status(rows[1]["rec_id"], "Rejected")

    assert db.stats()["rejected"] == 1
    assert db.stats()["skipped"] == 1

    from fastapi.testclient import TestClient
    from jobscout import web
    client = TestClient(web.app)
    d = client.get("/api/archive").json()
    assert d["count"] == 2
    assert sorted(r["status"] for r in d["rows"]) == ["Rejected", "Skipped"]


# --- upsert refresh semantics ------------------------------------------------------
def test_upsert_never_blanks_and_longer_description_wins():
    _fresh_db()
    ca, _ = _seed_two_companies()
    pid, _ = db.upsert_posting(ca, "j1", "Analyst", "", TODAY, "", "")
    # enrichment fills what the list feed lacked
    db.update_posting_enrichment(pid, description="FULL DESCRIPTION " * 50,
                                 url="http://real", location="Vienna, Austria")
    # next day's list fetch: empty desc/url/location again + a short summary
    db.upsert_posting(ca, "j1", "Analyst", "", TODAY, "", "short summary")
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM posting WHERE id=?", (pid,)).fetchone()
    assert row["location"] == "Vienna, Austria", "empty feed value blanked location"
    assert row["url"] == "http://real", "empty feed value blanked url"
    assert row["description"].startswith("FULL DESCRIPTION"), \
        "shorter list summary overwrote the enriched full description"
    # ... but a genuinely richer feed description DOES refresh
    db.upsert_posting(ca, "j1", "Analyst", "Vienna", TODAY, "http://new",
                      "EVEN LONGER DESCRIPTION " * 60)
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM posting WHERE id=?", (pid,)).fetchone()
    assert row["description"].startswith("EVEN LONGER")
    assert row["url"] == "http://new"


# --- browser cache ceiling ----------------------------------------------------------
def test_cached_extraction_keeps_all_jobs():
    _fresh_db()
    from jobscout import browser
    data = [{"title": f"Role {i}", "location": "Vienna", "url": "",
             "posted_date": "", "description": ""} for i in range(80)]
    cleaned = "x" * 1000
    import hashlib
    h = hashlib.sha1(cleaned.encode()).hexdigest()
    db.extraction_cache_put("k#render", h, json.dumps(data))
    jobs = browser._cached_or_extract("k#render", "http://x", cleaned, "<html>")
    assert len(jobs) == 80, "cache rematerialization dropped postings (old cap 60)"


def test_extract_limit_reaches_cleaner(monkeypatch):
    """The 90k Taleo clean budget must reach the extractor's own cleaning."""
    from jobscout import browser
    seen = {}
    real = browser.clean_html_for_llm

    def spy(html, *, limit=30000):
        seen["limit"] = limit
        return real(html, limit=limit)

    monkeypatch.setattr(browser, "clean_html_for_llm", spy)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "k")

    class FakeMsg:
        content = [type("B", (), {"text": "[]"})()]
    class FakeClient:
        def __init__(self, **kw): pass
        class messages:  # noqa: N801
            @staticmethod
            def create(**kw):
                return FakeMsg()
    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: FakeClient())
    browser.extract_jobs("<html></html>", "http://x", clean_limit=90000)
    assert seen["limit"] == 90000


# --- verify.py: workday saturation ---------------------------------------------------
def test_verify_workday_saturated_total_is_check_not_mismatch(monkeypatch):
    _fresh_db()
    from jobscout import verify
    from jobscout.connectors.base import RawJob

    jobs = [RawJob(job_id=str(i), title=f"T{i}", location="Vienna",
                   posted_date=TODAY) for i in range(2600)]
    monkeypatch.setattr(verify, "fetch_jobs", lambda a, t: jobs)
    monkeypatch.setitem(verify.PROBES, "workday", (lambda tok: 2000, "exact"))
    chk = verify.check_company({"name": "MegaCorp", "ats_type": "workday",
                                "ats_token": "mega|wd3|Ext"})
    assert chk.status == "CHECK", f"saturated total flagged {chk.status}"
    assert any("saturates" in n for n in chk.notes)

    # a fetch clearly BELOW a saturated total is still a real mismatch
    monkeypatch.setattr(verify, "fetch_jobs", lambda a, t: jobs[:900])
    chk = verify.check_company({"name": "MegaCorp", "ats_type": "workday",
                                "ats_token": "mega|wd3|Ext"})
    assert chk.status == "MISMATCH"


if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-q"]))


# --- personio URL synthesis -----------------------------------------------------
def test_personio_urls_use_the_tld_that_served_the_feed(monkeypatch):
    from jobscout.connectors import personio as pmod
    xml = b"""<?xml version="1.0"?><workzag-jobs>
      <position><id>101</id><name>BI Analyst</name><office>Vienna</office>
        <createdAt>2026-07-01</createdAt></position></workzag-jobs>"""

    calls = []

    class Resp:
        content = xml

    def fake_get(url, **kw):
        calls.append(url)
        if ".jobs.personio.de/" in url:      # .de tenant does not exist
            from jobscout.connectors.base import FetchError
            raise FetchError("404")
        return Resp()

    monkeypatch.setattr(pmod, "http_get", fake_get)
    jobs = pmod.PersonioConnector().fetch("acme")     # bare token: tries de, then com
    assert len(jobs) == 1
    assert jobs[0].url == "https://acme.jobs.personio.com/job/101?language=en", \
        "job link must use the tld that actually served the feed"
    # explicit tld token keeps working
    jobs = pmod.PersonioConnector().fetch("acme|com")
    assert jobs[0].url.startswith("https://acme.jobs.personio.com/")

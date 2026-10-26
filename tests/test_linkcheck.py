"""Dead-link pruning (linkcheck.py) — verdicts, archival, TTL, pipeline wiring.

The user-visible contract: a recommendation may only point at a job page that
still exists. DEAD needs certainty (API/HTTP 404, an unambiguous not-found
page); everything ambiguous (bot walls, timeouts, SPA shells) fails OPEN and
keeps the posting. A verified-dead posting loses its New rec, closes, and can
never re-enter the queue; user decisions (Marked/Applied/…) are never deleted.
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ["JOBSCOUT_DB"] = str(Path(tempfile.gettempdir()) / "jobscout_linkcheck_pytest.db")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402
import pytest  # noqa: E402

from jobscout import config, db, linkcheck, pipeline  # noqa: E402
from jobscout.connectors.base import FetchError, RawJob  # noqa: E402
from jobscout.linkcheck import ALIVE, DEAD, UNKNOWN, CheckResult, _Gone  # noqa: E402


# Fixture postings must always look FRESH to the pipeline: hard_filter() drops
# anything older than config.MAX_POSTING_AGE_DAYS, so a hardcoded date turns
# every pipeline test into a time bomb that passes until that day passes.
def _recent(days: int = 3) -> str:
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()


# --- plumbing -----------------------------------------------------------------
def _row(**kw):
    """A posting row as dict (sqlite3.Row and dict share the [] protocol)."""
    base = {"id": 1, "url": "https://x.example/jobs/12345", "job_id": "12345",
            "detail_ref": None, "link_checked_at": None, "title": "T",
            "rec_id": 1, "status": "New", "company_name": "XCo",
            "ats_type": "", "ats_token": "", "description": ""}
    base.update(kw)
    return base


def _fake_fetch(routes):
    """Build a _fetch replacement: first matching url-substring wins.
    Values: ("gone", detail) | Exception | (status, final_url, body)."""
    def fetch(url, *, accept="text/html", method="GET", json_body=None):
        for frag, resp in routes.items():
            if frag in url:
                if resp == "gone":
                    raise _Gone("HTTP 404")
                if isinstance(resp, Exception):
                    raise resp
                return resp
        raise AssertionError(f"unexpected URL fetched: {url}")
    return fetch


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(config, "LINKCHECK_DELAY", 0.0)
    monkeypatch.setattr(config, "LINKCHECK_ENABLED", True)
    linkcheck._ashby_boards.clear()


def _fresh_db():
    p = Path(os.environ["JOBSCOUT_DB"])
    if p.exists():
        p.unlink()
    db.init_db()


# --- generic page checker -------------------------------------------------------
def test_generic_404_is_dead(monkeypatch):
    monkeypatch.setattr(linkcheck, "_fetch", _fake_fetch({"x.example": "gone"}))
    assert linkcheck.check_posting(_row()).verdict == DEAD


def test_generic_ok_page_is_alive(monkeypatch):
    monkeypatch.setattr(linkcheck, "_fetch", _fake_fetch(
        {"x.example": (200, "https://x.example/jobs/12345", "<h1>Great job</h1>")}))
    assert linkcheck.check_posting(_row()).verdict == ALIVE


@pytest.mark.parametrize("phrase", [
    "The job you are looking for is no longer available.",
    "This position has been filled.",
    "Diese Stelle ist nicht mehr verfügbar.",
    "Job not found",
])
def test_generic_notfound_page_is_dead(monkeypatch, phrase):
    monkeypatch.setattr(linkcheck, "_fetch", _fake_fetch(
        {"x.example": (200, "https://x.example/jobs/12345", f"<html>{phrase}</html>")}))
    assert linkcheck.check_posting(_row()).verdict == DEAD


def test_generic_bot_wall_fails_open(monkeypatch):
    monkeypatch.setattr(linkcheck, "_fetch", _fake_fetch(
        {"x.example": (403, "https://x.example/jobs/12345", "blocked")}))
    assert linkcheck.check_posting(_row()).verdict == UNKNOWN


def test_generic_network_error_fails_open(monkeypatch):
    monkeypatch.setattr(linkcheck, "_fetch", _fake_fetch(
        {"x.example": requests.ConnectionError("boom")}))
    assert linkcheck.check_posting(_row()).verdict == UNKNOWN


def test_generic_checker_bug_fails_open(monkeypatch):
    monkeypatch.setattr(linkcheck, "_fetch",
                        _fake_fetch({"x.example": ValueError("bug")}))
    assert linkcheck.check_posting(_row()).verdict == UNKNOWN


def test_redirect_to_board_root_is_dead(monkeypatch):
    monkeypatch.setattr(linkcheck, "_fetch", _fake_fetch(
        {"x.example": (200, "https://x.example/search/", "<html>1000 open jobs</html>")}))
    assert linkcheck.check_posting(_row()).verdict == DEAD


def test_redirect_keeping_id_is_alive(monkeypatch):
    monkeypatch.setattr(linkcheck, "_fetch", _fake_fetch(
        {"x.example": (200, "https://x.example/en/jobs/12345-great-job", "<html>ok</html>")}))
    assert linkcheck.check_posting(_row()).verdict == ALIVE


def test_cross_host_redirect_is_not_dead(monkeypatch):
    monkeypatch.setattr(linkcheck, "_fetch", _fake_fetch(
        {"x.example": (200, "https://careers.other.example/", "<html>welcome</html>")}))
    assert linkcheck.check_posting(_row()).verdict == ALIVE


def test_no_url_is_unknown():
    assert linkcheck.check_posting(_row(url="")).verdict == UNKNOWN


# --- platform checkers ------------------------------------------------------------
def test_greenhouse_uses_api(monkeypatch):
    row = _row(ats_type="greenhouse", ats_token="stripe", job_id="8041850",
               url="https://stripe.com/jobs/search?gh_jid=8041850")
    monkeypatch.setattr(linkcheck, "_fetch", _fake_fetch(
        {"boards-api.greenhouse.io/v1/boards/stripe/jobs/8041850": "gone"}))
    res = linkcheck.check_posting(row)
    assert res.verdict == DEAD and "Greenhouse" in res.detail
    monkeypatch.setattr(linkcheck, "_fetch", _fake_fetch(
        {"boards-api.greenhouse.io": (200, "u", "{}")}))
    assert linkcheck.check_posting(row).verdict == ALIVE


_WD_LIST = "/wday/cxs/sanofi/SanofiCareers/jobs"


def _wd_row():
    return _row(ats_type="workday", ats_token="sanofi|wd3|SanofiCareers",
                detail_ref="/job/Bogota/Analyst_R2863913",
                url="https://sanofi.wd3.myworkdayjobs.com/SanofiCareers/job/Bogota/Analyst_R2863913")


@pytest.fixture(autouse=True)
def _fresh_workday_cache():
    linkcheck._workday_warm.clear()


def test_workday_checks_cxs_detail(monkeypatch):
    row = _wd_row()
    monkeypatch.setattr(linkcheck, "_fetch", _fake_fetch(
        {_WD_LIST: (200, "u", '{"total": 1}'),
         "/wday/cxs/sanofi/SanofiCareers/job/Bogota/Analyst_R2863913": "gone"}))
    assert linkcheck.check_posting(row).verdict == DEAD
    linkcheck._workday_warm.clear()
    monkeypatch.setattr(linkcheck, "_fetch", _fake_fetch(
        {_WD_LIST: (200, "u", '{"total": 1}'),
         "/wday/cxs/": (200, "u", '{"jobPostingInfo": {"title": "Analyst"}}')}))
    assert linkcheck.check_posting(row).verdict == ALIVE
    # SPA shell without jobPostingInfo proves nothing — fail open
    linkcheck._workday_warm.clear()
    monkeypatch.setattr(linkcheck, "_fetch", _fake_fetch(
        {_WD_LIST: (200, "u", '{"total": 1}'),
         "/wday/cxs/": (200, "u", '{"somethingElse": 1}')}))
    assert linkcheck.check_posting(row).verdict == UNKNOWN


def test_workday_403_after_warmup_needs_search_absence(monkeypatch):
    """Rolls-Royce case: delisted postings answer 403 'permission denied'.
    DEAD needs BOTH a working list endpoint (same session — not a bot wall)
    AND the requisition absent from the tenant's own search."""
    row = _wd_row()

    def routes(search_body):
        calls = {"n": 0}

        def fetch(url, *, accept="text/html", method="GET", json_body=None):
            if url.endswith(_WD_LIST):
                calls["n"] += 1
                if json_body and json_body.get("searchText"):
                    return (200, url, search_body)
                return (200, url, '{"total": 1}')
            return (403, url, '{"errorCode":"S22","message":"permission denied"}')
        return fetch

    # absent from search -> dead
    monkeypatch.setattr(linkcheck, "_fetch", routes('{"jobPostings": []}'))
    res = linkcheck.check_posting(row)
    assert res.verdict == DEAD and "no longer public" in res.detail
    # still listed under its requisition id -> NOT dead (fail open)
    linkcheck._workday_warm.clear()
    monkeypatch.setattr(linkcheck, "_fetch", routes(
        '{"jobPostings": [{"externalPath": "/job/Bogota/Analyst_R2863913"}]}'))
    assert linkcheck.check_posting(row).verdict == UNKNOWN


def test_workday_403_without_warmup_stays_unknown(monkeypatch):
    """When even the list endpoint is blocked, a 403 is a wall, not a verdict."""
    row = _wd_row()

    def fetch(url, *, accept="text/html", method="GET", json_body=None):
        return (403, url, '{"errorCode":"S22","message":"permission denied"}')
    monkeypatch.setattr(linkcheck, "_fetch", fetch)
    assert linkcheck.check_posting(row).verdict == UNKNOWN


def test_workday_derives_path_from_url_when_no_detail_ref(monkeypatch):
    row = _wd_row()
    row["detail_ref"] = None
    seen = {}

    def fetch(url, *, accept="text/html", method="GET", json_body=None):
        if url.endswith(_WD_LIST):
            return (200, url, '{"total": 1}')
        seen["url"] = url
        raise _Gone("HTTP 404")
    monkeypatch.setattr(linkcheck, "_fetch", fetch)
    assert linkcheck.check_posting(row).verdict == DEAD
    assert seen["url"].endswith("/wday/cxs/sanofi/SanofiCareers/job/Bogota/Analyst_R2863913")


def test_smartrecruiters_uses_posting_api(monkeypatch):
    row = _row(ats_type="smartrecruiters", ats_token="BoschGroup",
               job_id="744000135705394")
    monkeypatch.setattr(linkcheck, "_fetch", _fake_fetch(
        {"api.smartrecruiters.com/v1/companies/BoschGroup/postings/744000135705394": "gone"}))
    assert linkcheck.check_posting(row).verdict == DEAD


def test_oracle_empty_items_is_dead(monkeypatch):
    row = _row(ats_type="oracle", ats_token="fa-x.oraclecloud.com|CX_1001",
               job_id="22022")
    monkeypatch.setattr(linkcheck, "_fetch", _fake_fetch(
        {"recruitingCEJobRequisitionDetails": (200, "u", '{"items": []}')}))
    assert linkcheck.check_posting(row).verdict == DEAD
    monkeypatch.setattr(linkcheck, "_fetch", _fake_fetch(
        {"recruitingCEJobRequisitionDetails": (200, "u", '{"items": [{"Id": "22022"}]}')}))
    assert linkcheck.check_posting(row).verdict == ALIVE


def test_ashby_membership(monkeypatch):
    row = _row(ats_type="ashby", ats_token="openai", job_id="b7-x")
    monkeypatch.setattr(linkcheck, "_fetch", _fake_fetch(
        {"posting-api/job-board/openai": (200, "u", '{"jobs": [{"id": "other"}]}')}))
    assert linkcheck.check_posting(row).verdict == DEAD
    linkcheck._ashby_boards.clear()
    monkeypatch.setattr(linkcheck, "_fetch", _fake_fetch(
        {"posting-api/job-board/openai": (200, "u", '{"jobs": [{"id": "b7-x"}]}')}))
    assert linkcheck.check_posting(row).verdict == ALIVE
    linkcheck._ashby_boards.clear()
    monkeypatch.setattr(linkcheck, "_fetch", _fake_fetch(
        {"posting-api/job-board/openai": requests.ConnectionError("nope")}))
    assert linkcheck.check_posting(row).verdict == UNKNOWN     # feed down ≠ job dead


def test_amazon_json_twin(monkeypatch):
    row = _row(ats_type="amazon", job_id="10456816")
    monkeypatch.setattr(linkcheck, "_fetch", _fake_fetch(
        {"/en/jobs/10456816.json": (200, "u", '{"error": "Job not found"}')}))
    assert linkcheck.check_posting(row).verdict == DEAD
    monkeypatch.setattr(linkcheck, "_fetch", _fake_fetch(
        {"/en/jobs/10456816.json": (200, "u", '{"job": {"id_icims": "10456816"}}')}))
    assert linkcheck.check_posting(row).verdict == ALIVE


def test_redirect_to_tombstone_slug_is_dead(monkeypatch):
    """Cross-host redirect onto an explicit 'no longer available' URL
    (Mercedes-style tombstone) is a certain verdict."""
    monkeypatch.setattr(linkcheck, "_fetch", _fake_fetch(
        {"x.example": (200, "https://group.other.example/karriere/jobsuche/"
                            "nicht-mehr-verfuegbar/", "<html>…</html>")}))
    assert linkcheck.check_posting(_row()).verdict == DEAD


def test_apple_uses_jobdetails_api(monkeypatch):
    row = _row(ats_type="apple", job_id="200649531",
               url="https://jobs.apple.com/en-us/details/200649531/x")
    monkeypatch.setattr(linkcheck, "_fetch", _fake_fetch(
        {"api/v1/jobDetails/200649531": "gone"}))
    res = linkcheck.check_posting(row)
    assert res.verdict == DEAD and "Apple" in res.detail
    monkeypatch.setattr(linkcheck, "_fetch", _fake_fetch(
        {"api/v1/jobDetails/200649531": (200, "u", '{"res": {}}')}))
    assert linkcheck.check_posting(row).verdict == ALIVE


def test_bot_wall_retries_impersonated(monkeypatch):
    """A 403 (Akamai TLS wall) triggers ONE curl_cffi retry; its verdict
    counts. Without curl_cffi the check stays UNKNOWN (fail-open)."""
    row = _row()
    monkeypatch.setattr(linkcheck, "_fetch", _fake_fetch(
        {"x.example": (403, "https://x.example/jobs/12345", "blocked")}))
    monkeypatch.setattr(linkcheck, "_fetch_impersonated",
                        lambda url: (404, url, "not found"))
    assert linkcheck.check_posting(row).verdict == DEAD
    monkeypatch.setattr(linkcheck, "_fetch_impersonated",
                        lambda url: (200, url, "<h1>Great job</h1>"))
    assert linkcheck.check_posting(row).verdict == ALIVE
    monkeypatch.setattr(linkcheck, "_fetch_impersonated", lambda url: None)
    assert linkcheck.check_posting(row).verdict == UNKNOWN


def test_gone_result_detects_404_causes():
    err = FetchError("404 not found: http://x")
    assert linkcheck.gone_result(err)
    resp_err = requests.HTTPError(
        response=type("R", (), {"status_code": 404})())
    chained = FetchError("GET failed")
    chained.__cause__ = resp_err
    assert linkcheck.gone_result(chained)
    assert not linkcheck.gone_result(FetchError("timeout after 3 tries"))


# --- board recheck: DB semantics ---------------------------------------------------
def _seed_company_posting(name="LcCo", ats="greenhouse", token="lcco",
                          job_id="111", url="https://b.example/jobs/111",
                          status=None, fit=80.0):
    cid = db.upsert_company(name)
    db.set_company_ats(cid, ats, token, "detected")
    pid, _ = db.upsert_posting(company_id=cid, job_id=job_id, title="Data Analyst",
                               location="Vienna, Austria", posted_date=_recent(),
                               url=url, description="SQL dashboards")
    if status:
        db.save_recommendation(posting_id=pid, prefilter_score=50, fit_score=fit,
                               rationale="r", watch_outs="", llm_used=True)
        rec = db.board_rows()[0]
        if status != "New":
            db.set_status(rec["rec_id"], status)
    return cid, pid


def test_recheck_archives_dead_new_rec(monkeypatch):
    _fresh_db()
    _, pid = _seed_company_posting(status="New")
    monkeypatch.setattr(linkcheck, "check_posting",
                        lambda r: CheckResult(DEAD, "HTTP 404"))
    out = linkcheck.recheck_recommendations(force=True)
    assert (out.checked, out.dead, out.recs_removed, out.postings_closed) == (1, 1, 1, 1)
    with db.connect() as conn:
        p = conn.execute("SELECT * FROM posting WHERE id=?", (pid,)).fetchone()
        assert p["is_open"] == 0
        assert p["filtered_reason"].startswith("job page gone")
        assert conn.execute("SELECT COUNT(*) FROM recommendation").fetchone()[0] == 0
    # archived postings can never re-enter the scoring queue
    assert all(r["id"] != pid for r in db.get_postings_to_score())


def test_recheck_keeps_marked_rec_but_closes_posting(monkeypatch):
    _fresh_db()
    _, pid = _seed_company_posting(status="Marked")
    monkeypatch.setattr(linkcheck, "check_posting",
                        lambda r: CheckResult(DEAD, "HTTP 404"))
    out = linkcheck.recheck_recommendations(force=True)
    assert out.dead == 1 and out.recs_removed == 0
    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM recommendation "
                            "WHERE status='Marked'").fetchone()[0] == 1
        assert conn.execute("SELECT is_open FROM posting WHERE id=?",
                            (pid,)).fetchone()[0] == 0


def test_recheck_alive_stamps_and_ttl_skips(monkeypatch):
    _fresh_db()
    _, pid = _seed_company_posting(status="New")
    calls = {"n": 0}

    def alive(r):
        calls["n"] += 1
        return CheckResult(ALIVE, "ok")
    monkeypatch.setattr(linkcheck, "check_posting", alive)
    out = linkcheck.recheck_recommendations()
    assert out.checked == 1 and calls["n"] == 1
    with db.connect() as conn:
        assert conn.execute("SELECT link_checked_at FROM posting WHERE id=?",
                            (pid,)).fetchone()[0]
    # within the TTL nothing is re-checked …
    out2 = linkcheck.recheck_recommendations()
    assert out2.checked == 0 and out2.skipped_fresh == 1 and calls["n"] == 1
    # … but --force re-verifies
    out3 = linkcheck.recheck_recommendations(force=True)
    assert out3.checked == 1 and calls["n"] == 2


def test_recheck_unknown_changes_nothing(monkeypatch):
    _fresh_db()
    _, pid = _seed_company_posting(status="New")
    monkeypatch.setattr(linkcheck, "check_posting",
                        lambda r: CheckResult(UNKNOWN, "HTTP 403"))
    out = linkcheck.recheck_recommendations(force=True)
    assert out.unknown == 1 and out.dead == 0
    with db.connect() as conn:
        p = conn.execute("SELECT * FROM posting WHERE id=?", (pid,)).fetchone()
        assert p["is_open"] == 1 and p["filtered_reason"] is None
        assert p["link_checked_at"] is None      # retried on the next pass
        assert conn.execute("SELECT COUNT(*) FROM recommendation").fetchone()[0] == 1


def test_decided_statuses_are_not_checked(monkeypatch):
    _fresh_db()
    _seed_company_posting(status="Applied")
    monkeypatch.setattr(linkcheck, "check_posting",
                        lambda r: (_ for _ in ()).throw(AssertionError("checked Applied")))
    out = linkcheck.recheck_recommendations(force=True)
    assert out.checked == 0


# --- candidate gate ------------------------------------------------------------------
def test_check_candidates_archives_dead_and_skips_enrichable(monkeypatch):
    _fresh_db()
    cid = db.upsert_company("QCo")
    db.set_company_ats(cid, "workday", "t|wd1|Site", "detected")
    # will-enrich: workday, no description -> the enrichment fetch is the probe
    pid_enrich, _ = db.upsert_posting(company_id=cid, job_id="/job/a", title="A",
                                      location="Vienna", posted_date=_recent(),
                                      url="https://t.wd1.myworkdayjobs.com/Site/job/a",
                                      description="", detail_ref="/job/a")
    cid2 = db.upsert_company("QCo2")
    db.set_company_ats(cid2, "greenhouse", "qco2", "detected")
    pid_dead, _ = db.upsert_posting(company_id=cid2, job_id="222", title="B",
                                    location="Vienna", posted_date=_recent(),
                                    url="https://b.example/jobs/222",
                                    description="desc")
    monkeypatch.setattr(linkcheck, "check_posting",
                        lambda r: CheckResult(DEAD, "HTTP 404"))
    rows = db.get_postings_to_score()
    checked, dead = linkcheck.check_candidates(rows)
    assert checked == 1                       # the workday row was left to enrichment
    assert list(dead) == [pid_dead]
    with db.connect() as conn:
        assert conn.execute("SELECT is_open FROM posting WHERE id=?",
                            (pid_dead,)).fetchone()[0] == 0
        assert conn.execute("SELECT is_open FROM posting WHERE id=?",
                            (pid_enrich,)).fetchone()[0] == 1


# --- pipeline wiring ------------------------------------------------------------------
class _Outcome:
    reviewed = removed = flagged = pending = 0
    notes: list = []


def test_run_prunes_existing_dead_rec_before_filters(monkeypatch):
    """A stale New rec whose page is gone disappears at run start — before the
    hard filters/queue are touched — even when the fetch brings nothing."""
    _fresh_db()
    db.save_profile("cv", "prefs")
    _, pid = _seed_company_posting(status="New")
    from jobscout import matching, validate
    monkeypatch.setattr(linkcheck, "check_posting",
                        lambda r: CheckResult(DEAD, "HTTP 404"))
    monkeypatch.setattr(pipeline, "_fetch_one", lambda c: [])
    monkeypatch.setattr(validate, "validate_new_recommendations",
                        lambda **kw: _Outcome())
    summary = pipeline.run(verbose=False)
    assert summary.links_checked == 1 and summary.links_dead == 1
    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM recommendation").fetchone()[0] == 0
        p = conn.execute("SELECT * FROM posting WHERE id=?", (pid,)).fetchone()
        assert p["is_open"] == 0 and p["filtered_reason"].startswith("job page gone")


def test_run_never_scores_a_dead_candidate(monkeypatch):
    """The candidate gate: a queued posting whose page is gone is archived
    unscored; the LLM never sees it and no board row appears."""
    _fresh_db()
    db.save_profile("MSc data science, BI focus", "Analyst roles")
    cid = db.upsert_company("GateCo")
    db.set_company_ats(cid, "greenhouse", "gateco", "detected")
    jobs = [
        RawJob("g-1", "Junior BI Analyst", "Vienna, Austria", _recent(),
               "https://g.example/jobs/1111", "SQL dashboards, Power BI."),
        RawJob("g-2", "Reporting Analyst", "Vienna, Austria", _recent(),
               "https://g.example/jobs/2222", "SQL reporting, KPIs."),
    ]
    from jobscout import matching, validate
    monkeypatch.setattr(pipeline, "_fetch_one", lambda c: list(jobs))
    monkeypatch.setattr(validate, "validate_new_recommendations",
                        lambda **kw: _Outcome())
    monkeypatch.setattr(matching, "llm_available", lambda: True)
    monkeypatch.setattr(
        linkcheck, "check_posting",
        lambda r: CheckResult(DEAD, "HTTP 404") if r["job_id"] == "g-2"
        else CheckResult(ALIVE, "ok"))

    scored_ids = []

    def fake_score(cv, prefs, batch, tracker, extra_rules=""):
        scored_ids.extend(j["id"] for j in batch)
        return {j["id"]: matching.MatchResult(prefilter_score=j["pre_score"],
                                              fit_score=80.0, rationale="fits",
                                              watch_outs="", llm_used=True)
                for j in batch}
    monkeypatch.setattr(matching, "score_batch", fake_score)

    summary = pipeline.run(verbose=False)
    assert summary.links_dead == 1
    with db.connect() as conn:
        dead = conn.execute("SELECT * FROM posting WHERE job_id='g-2'").fetchone()
        assert dead["is_open"] == 0
        assert dead["filtered_reason"].startswith("job page gone")
        assert dead["id"] not in scored_ids            # never reached the LLM
        live = conn.execute("SELECT * FROM posting WHERE job_id='g-1'").fetchone()
        assert live["id"] in scored_ids
        recs = conn.execute("SELECT posting_id FROM recommendation").fetchall()
        assert [r["posting_id"] for r in recs] == [live["id"]]


def test_linkcheck_disabled_is_a_noop(monkeypatch):
    _fresh_db()
    _seed_company_posting(status="New")
    monkeypatch.setattr(config, "LINKCHECK_ENABLED", False)
    monkeypatch.setattr(linkcheck, "check_posting",
                        lambda r: (_ for _ in ()).throw(AssertionError("checked")))
    s = pipeline.RunSummary()
    pipeline._prune_dead_links(s, verbose=False)
    assert s.links_checked == 0


def test_web_groups_dead_link_reason():
    from jobscout.web import _other_exclusion_label
    assert _other_exclusion_label(
        "job page gone (HTTP 404; checked 2026-07-16)") == "Dead links (job page gone)"


def test_dead_reason_format():
    r = linkcheck.dead_reason("HTTP 404")
    assert r.startswith("job page gone (HTTP 404; checked 20")

"""Scraper-completeness fixes from the 2026-07-10 board verification sweep.

1. Workday saturates its reported `total` at exactly 2000 on mega-tenants
   (ABB, NVIDIA) — the connector must keep walking instead of stopping there.
2. Lever boards can disable/empty the public JSON API while the hosted HTML
   board stays live — the connector falls back to parsing the board page.
3. KNOWN_UNSUPPORTED curation must override a stale auto-probe (Meta was
   'detected' on an unrelated scam Recruitee tenant) and close its postings.
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ["JOBSCOUT_DB"] = str(Path(tempfile.gettempdir()) / "jobscout_pytest_scraper.db")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobscout import db, pipeline  # noqa: E402
from jobscout.connectors import lever as lever_mod  # noqa: E402
from jobscout.connectors import workday as workday_mod  # noqa: E402


class _Resp:
    def __init__(self, payload=None, text=""):
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


# =============================================================================
# 1. Workday total saturation
# =============================================================================
def test_workday_walks_past_saturated_total(monkeypatch):
    """total=2000 is a floor, not a count: pages beyond 2000 must be fetched."""
    calls = []

    def fake_http_get(url, method="GET", json_body=None, **kw):
        offset = json_body["offset"]
        calls.append(offset)
        if offset >= 2040:                    # board truly ends at 2040
            return _Resp({"total": 2000, "jobPostings": []})
        return _Resp({"total": 2000, "jobPostings": [
            {"externalPath": f"/job/x{offset + i}", "title": f"T{offset + i}",
             "locationsText": "Vienna", "postedOn": "Posted Today"}
            for i in range(20)
        ]})

    monkeypatch.setattr(workday_mod, "http_get", fake_http_get)
    jobs = workday_mod.WorkdayConnector().fetch("acme|wd1|Ext")
    assert len(jobs) == 2040                  # NOT stopped at 2000
    assert max(calls) >= 2040                 # walked to the empty page


def test_workday_trusts_unsaturated_total(monkeypatch):
    """A total below 2000 still stops the walk (no wasted requests)."""
    calls = []

    def fake_http_get(url, method="GET", json_body=None, **kw):
        offset = json_body["offset"]
        calls.append(offset)
        return _Resp({"total": 40, "jobPostings": [
            {"externalPath": f"/job/y{offset + i}", "title": "T",
             "locationsText": "Berlin", "postedOn": "Posted Today"}
            for i in range(20)
        ]})

    monkeypatch.setattr(workday_mod, "http_get", fake_http_get)
    jobs = workday_mod.WorkdayConnector().fetch("acme|wd1|Ext")
    assert len(jobs) == 40
    assert max(calls) == 20                   # stopped right at the total


def test_workday_dedupes_shifting_pages(monkeypatch):
    """Postings repeated across page boundaries must not double-count."""
    def fake_http_get(url, method="GET", json_body=None, **kw):
        offset = json_body["offset"]
        if offset >= 40:
            return _Resp({"total": 40, "jobPostings": []})
        # both pages serve overlapping ids (sort instability)
        return _Resp({"total": 40, "jobPostings": [
            {"externalPath": f"/job/z{i}", "title": "T",
             "locationsText": "Zurich", "postedOn": "Posted Today"}
            for i in range(30)
        ]})

    monkeypatch.setattr(workday_mod, "http_get", fake_http_get)
    jobs = workday_mod.WorkdayConnector().fetch("acme|wd1|Ext")
    assert len(jobs) == 30                    # 60 raw rows, 30 unique


# =============================================================================
# 2. Lever HTML fallback
# =============================================================================
_BOARD_HTML = """
<div class="postings-wrapper"><div class="postings-group">
<div class="posting" data-qa-posting-id="11111111-2222-3333-4444-555555555555">
<a class="posting-title" href="https://jobs.lever.co/acme/11111111-2222-3333-4444-555555555555">
<h5 data-qa="posting-name">Junior BI Analyst</h5>
<div class="posting-categories">
<span class="sort-by-location posting-category small-category-label">Vienna</span>
</div></a></div>
<div class="posting" data-qa-posting-id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee">
<a class="posting-title" href="https://jobs.lever.co/acme/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee">
<h5 data-qa="posting-name">Data Analyst</h5>
<div class="posting-categories">
<span class="sort-by-location posting-category small-category-label">London</span>
</div></a></div>
</div></div>
"""


def test_lever_falls_back_to_board_html(monkeypatch):
    def fake_http_get(url, accept="application/json", **kw):
        if "api.lever.co" in url:
            return _Resp(payload=[])          # API opt-out: empty list
        return _Resp(text=_BOARD_HTML)

    monkeypatch.setattr(lever_mod, "http_get", fake_http_get)
    jobs = lever_mod.LeverConnector().fetch("acme")
    assert [j.title for j in jobs] == ["Junior BI Analyst", "Data Analyst"]
    assert jobs[0].location == "Vienna"
    assert jobs[0].job_id == "11111111-2222-3333-4444-555555555555"
    assert jobs[0].url.endswith(jobs[0].job_id)
    assert all(j.date_confidence == "low" for j in jobs)


def test_lever_api_still_preferred(monkeypatch):
    def fake_http_get(url, accept="application/json", **kw):
        assert "api.lever.co" in url          # the board page is never fetched
        return _Resp(payload=[{
            "id": "abc", "text": "BI Analyst", "createdAt": 1780000000000,
            "categories": {"location": "Vienna"}, "hostedUrl": "http://x",
            "descriptionPlain": "SQL."}])

    monkeypatch.setattr(lever_mod, "http_get", fake_http_get)
    jobs = lever_mod.LeverConnector().fetch("acme")
    assert len(jobs) == 1 and jobs[0].posted_date


def test_lever_both_empty_is_empty(monkeypatch):
    monkeypatch.setattr(lever_mod, "http_get",
                        lambda url, **kw: _Resp(payload=[], text="<html></html>"))
    assert lever_mod.LeverConnector().fetch("acme") == []


# =============================================================================
# 3. Curated UNSUPPORTED overrides a stale probe
# =============================================================================
def test_curated_unsupported_wins_and_closes_postings():
    p = Path(os.environ["JOBSCOUT_DB"])
    if p.exists():
        p.unlink()
    db.init_db()
    cid = db.upsert_company("Meta")
    db.set_company_ats(cid, "recruitee", "meta", "detected", "probed")
    pid, _ = db.upsert_posting(cid, "scam1", "Social Media Manager",
                               "Remote job", None,
                               "https://facebookdata.recruitee.com/o/x", "")
    changed = pipeline.apply_curated_detection()
    assert changed >= 1
    comp = [c for c in db.get_companies() if c["name"] == "Meta"][0]
    assert comp["detect_status"] == "unsupported"
    assert comp["ats_type"] is None
    with db.connect() as conn:
        assert conn.execute("SELECT is_open FROM posting WHERE id=?",
                            (pid,)).fetchone()[0] == 0

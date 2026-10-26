"""Browser-tier extraction: a big board must not come back empty.

Found live on 2026-09-01. WHO's rendered Taleo board carried 70 real postings
("1 - 25 of 70", 68 jobdetail links in the cleaned page). The extraction
returned ZERO, and the pipeline cached that [] against the page's content hash —
so WHO reported an empty board on every subsequent run, for free, forever.

Two independent defects:
  1. the model's JSON array exceeded the 8000-token output budget, and the
     parser required a closing bracket (`re.search(r"\\[.*\\]")`), so a
     truncated array parsed as nothing at all;
  2. `_cached_or_extract` cached the empty result, making a transient failure
     permanent.
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ["JOBSCOUT_DB"] = str(Path(tempfile.gettempdir()) / "jobscout_pytest_trunc.db")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobscout import browser, db  # noqa: E402


def _fresh_db():
    p = Path(os.environ["JOBSCOUT_DB"])
    if p.exists():
        p.unlink()
    db.init_db()


def _obj(i):
    return ('{"title": "Technical Officer %d", "url": "/careersection/ex/'
            'jobdetail.ftl?job=%d", "location": "Geneva, Switzerland", '
            '"posted_date": "2026-08-20"}' % (i, 2600000 + i))


# --- 1. the parser ------------------------------------------------------------
def test_a_complete_array_parses():
    raw = "[" + ", ".join(_obj(i) for i in range(3)) + "]"
    assert len(browser._parse_job_array(raw)) == 3


def test_a_truncated_array_keeps_every_complete_object():
    """The WHO shape: the response is cut off mid-array, no closing bracket."""
    full = "[" + ", ".join(_obj(i) for i in range(70))
    raw = full[:len(full) // 2]            # chopped mid-object
    out = browser._parse_job_array(raw, truncated=True, source="who")
    assert 25 < len(out) < 70, f"salvaged {len(out)}"
    assert all(o["title"].startswith("Technical Officer") for o in out)
    assert out[0]["title"] == "Technical Officer 0"


def test_salvage_is_not_confused_by_braces_inside_strings():
    """A `}` inside a title must not close the object early."""
    raw = ('[{"title": "Analyst {contract}", "url": "/a"}, '
           '{"title": "Officer }", "url": "/b"}')
    out = browser._parse_job_array(raw)
    assert [o["title"] for o in out] == ["Analyst {contract}", "Officer }"]


def test_salvage_handles_escaped_quotes():
    raw = r'[{"title": "Say \"hi\" }", "url": "/b"}'
    out = browser._parse_job_array(raw)
    assert len(out) == 1 and out[0]["title"] == 'Say "hi" }'


def test_prose_around_the_array_is_ignored():
    raw = "Here are the jobs:\n[" + _obj(1) + "]\nThat's all."
    assert len(browser._parse_job_array(raw)) == 1


def test_no_array_at_all_is_empty():
    assert browser._parse_job_array("I could not find any job postings.") == []
    assert browser._parse_job_array("") == []


def test_the_output_budget_fits_a_full_board():
    """MAX_EXTRACT_JOBS postings at ~100-160 output tokens each."""
    assert browser.EXTRACT_MAX_TOKENS >= browser.MAX_EXTRACT_JOBS * 150


# --- 2. the cache -------------------------------------------------------------
def _stub_extract(monkeypatch, result):
    calls = {"n": 0}

    def fake(html, base_url="", **kw):
        calls["n"] += 1
        return list(result)
    monkeypatch.setattr(browser, "extract_jobs", fake)
    return calls


def test_an_empty_extraction_is_never_cached(monkeypatch):
    """Otherwise one bad extraction silences an employer until its page text
    changes — which for a stable careers board may be never."""
    _fresh_db()
    calls = _stub_extract(monkeypatch, [])
    for _ in range(3):
        assert browser._cached_or_extract("k#render", "https://x", "page text",
                                          "<html>page text</html>") == []
    assert calls["n"] == 3, "every run retries an empty board"
    assert db.extraction_cache_get("k#render") is None


def test_a_real_extraction_is_cached_and_reused(monkeypatch):
    """The cost saving this cache exists for must survive the fix."""
    _fresh_db()
    from jobscout.connectors.base import RawJob
    jobs = [RawJob("1", "Data Analyst", "Vienna, Austria", "2026-08-20",
                   "https://x/jobs/1", "SQL")]
    calls = _stub_extract(monkeypatch, jobs)
    a = browser._cached_or_extract("k#render", "https://x", "page text",
                                   "<html>page text</html>")
    b = browser._cached_or_extract("k#render", "https://x", "page text",
                                   "<html>page text</html>")
    assert len(a) == len(b) == 1
    assert calls["n"] == 1, "unchanged page cost a second API call"


def test_changed_page_text_re_extracts(monkeypatch):
    _fresh_db()
    from jobscout.connectors.base import RawJob
    calls = _stub_extract(monkeypatch, [RawJob("1", "Analyst", "Vienna", None,
                                               "https://x/1", "d")])
    browser._cached_or_extract("k#render", "https://x", "text one", "<html>1</html>")
    browser._cached_or_extract("k#render", "https://x", "text two", "<html>2</html>")
    assert calls["n"] == 2

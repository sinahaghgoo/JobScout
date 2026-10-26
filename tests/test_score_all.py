"""Score-ALL-survivors flow (user rule, 2026-07-11).

Hard filters decide eligibility; Claude decides fit. The prefilter only
ORDERS the queue — it must never deny a surviving posting its verdict:

  * default PREFILTER_KEEP=0  -> every hard-filter survivor is scored,
    with no per-company cap;
  * PREFILTER_KEEP>0          -> the old top-N throttle (fair-shared per
    company) still works as an emergency cost brake;
  * unscored leftovers (budget stops / LLM errors) survive 10 days of
    retries, then the janitor retires them.
"""
import os
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

os.environ["JOBSCOUT_DB"] = str(Path(tempfile.gettempdir()) / "jobscout_scoreall_pytest.db")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobscout import config, db, matching, pipeline  # noqa: E402

TODAY = date.today().isoformat()
DESC = ("You will build SQL queries and Power BI dashboards, own the "
        "reporting for commercial stakeholders and present insights.")


def _fresh_db():
    p = Path(os.environ["JOBSCOUT_DB"])
    if p.exists():
        p.unlink()
    db.init_db()


def _seed(n_a=15, n_b=5):
    ca = db.upsert_company("BigBoard AG")
    cb = db.upsert_company("SmallCo GmbH")
    for i in range(n_a):
        db.upsert_posting(ca, f"a{i}", f"Business Analyst {i}", "Vienna, Austria",
                          TODAY, f"http://a/{i}", DESC)
    for i in range(n_b):
        db.upsert_posting(cb, f"b{i}", f"Data Analyst {i}", "Berlin, Germany",
                          TODAY, f"http://b/{i}", DESC)
    return ca, cb


def _keyless(monkeypatch):
    """Force the deterministic heuristic path (no API key, no network)."""
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(matching, "llm_error_count", 0)


def test_default_scores_every_survivor(monkeypatch):
    _fresh_db()
    _keyless(monkeypatch)
    assert config.PREFILTER_KEEP == 0          # score-all is the default
    ca, _cb = _seed(n_a=15, n_b=5)
    summary = pipeline.RunSummary()
    pipeline._match_new(summary, verbose=False)
    assert summary.prefiltered_in == 20        # nothing cut before scoring
    assert summary.scored == 20                # ... and everything scored
    with db.connect() as conn:
        n = conn.execute("SELECT COUNT(*) FROM recommendation").fetchone()[0]
        per_a = conn.execute(
            """SELECT COUNT(*) FROM recommendation r JOIN posting p
               ON p.id=r.posting_id WHERE p.company_id=?""", (ca,)).fetchone()[0]
    assert n == 20
    assert per_a == 15   # no per-company cap in score-all mode


def test_throttle_still_works_when_enabled(monkeypatch):
    _fresh_db()
    _keyless(monkeypatch)
    monkeypatch.setattr(config, "PREFILTER_KEEP", 6)
    monkeypatch.setattr(config, "PER_COMPANY_LLM_CAP", 3)
    _seed(n_a=15, n_b=5)
    summary = pipeline.RunSummary()
    pipeline._match_new(summary, verbose=False)
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT p.company_id, COUNT(*) n FROM recommendation r
               JOIN posting p ON p.id=r.posting_id GROUP BY 1""").fetchall()
    counts = {r["company_id"]: r["n"] for r in rows}
    assert sum(counts.values()) == 6           # top-N respected
    assert max(counts.values()) <= 3           # fair share respected


def test_unscored_leftovers_survive_then_age_out(monkeypatch):
    """LLM configured but returning nothing (outage/budget): postings stay
    queued — only >10-day-old ones are retired, with an honest reason."""
    _fresh_db()
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(matching, "llm_available", lambda: True)
    monkeypatch.setattr(matching, "score_batch", lambda *a, **k: {})
    cid = db.upsert_company("OutageCo")
    fresh, _ = db.upsert_posting(cid, "f1", "Business Analyst", "Vienna, Austria",
                                 TODAY, "u", DESC)
    stale, _ = db.upsert_posting(cid, "s1", "Data Analyst", "Vienna, Austria",
                                 TODAY, "u", DESC)
    old = (datetime.now(timezone.utc) - timedelta(days=11)).isoformat(timespec="seconds")
    with db.connect() as conn:
        conn.execute("UPDATE posting SET first_seen=? WHERE id=?", (old, stale))
    summary = pipeline.RunSummary()
    pipeline._match_new(summary, verbose=False)
    with db.connect() as conn:
        f = conn.execute("SELECT filtered_reason FROM posting WHERE id=?",
                         (fresh,)).fetchone()[0]
        s = conn.execute("SELECT filtered_reason FROM posting WHERE id=?",
                         (stale,)).fetchone()[0]
        recs = conn.execute("SELECT COUNT(*) FROM recommendation").fetchone()[0]
    assert recs == 0                            # no fake scores, ever
    assert f is None                            # fresh leftover stays queued
    assert s == "not scored within 10 days (LLM budget/errors)"


def test_budget_default_is_30_usd():
    # Raised from 10 on 2026-07-13 so the initial backlog clears in one
    # frontend-triggered run; still a hard brake against runaway spend.
    assert config.MAX_RUN_LLM_USD == 30.00 or os.environ.get("JOBSCOUT_MAX_RUN_USD")

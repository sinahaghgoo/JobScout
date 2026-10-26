"""Duplicate suppression + board reset (2026-07-09).

The same vacancy often exists under several job_ids (reposts, per-slot
requisitions), so dedupe must work on content fingerprints. A reset wipes
everything except chosen decisions while freezing what the deleted Skipped
rows taught the recommender.
"""
import os
import sys
import tempfile
from datetime import date
from pathlib import Path

os.environ["JOBSCOUT_DB"] = str(Path(tempfile.gettempdir()) / "jobscout_pytest.db")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobscout import config, db, feedback, matching, pipeline  # noqa: E402

TODAY = date.today().isoformat()


def _fresh_db():
    p = Path(os.environ["JOBSCOUT_DB"])
    if p.exists():
        p.unlink()
    db.init_db()


def _posting(cid, jid, title, loc="Vienna, Austria",
             desc="SQL, Power BI dashboards and reporting for stakeholders."):
    pid, _ = db.upsert_posting(cid, jid, title, loc, TODAY, f"http://x/{jid}", desc)
    return pid


# =============================================================================
# Fingerprints
# =============================================================================
def test_title_fingerprint_normalization():
    tf = matching.title_fingerprint
    assert tf("IT Consultant") == tf("IT  Consultant ")
    assert tf("Junior Analyst (m/w/d)") == tf("Junior Analyst (w/m/d)") == tf("Junior Analyst")
    assert tf("Business Analyst (f/m/x)") == tf("BUSINESS ANALYST")
    # reference numbers vanish -> reposts collapse
    assert tf("Data & Analytics Officer - Kyiv - 878389") == tf("Data & Analytics Officer - Kyiv - 900001")
    assert tf("Inside Sales DACH 80%-100% (w/m/d) REF290650V") == tf("Inside Sales DACH 80%-100%")
    # different roles stay different
    assert tf("Data Analyst") != tf("Data Engineer")


def test_posting_fingerprint_respects_location():
    fp = matching.posting_fingerprint
    assert fp(1, "IT Consultant", "Geneva") == fp(1, "IT Consultant", "Geneva")
    assert fp(1, "Sales Specialist", "Amsterdam") != fp(1, "Sales Specialist", "London")
    assert fp(1, "IT Consultant", "Geneva") != fp(2, "IT Consultant", "Geneva")


# =============================================================================
# Pipeline-level suppression
# =============================================================================
def test_new_duplicate_postings_yield_one_recommendation(monkeypatch):
    _fresh_db()
    db.save_profile("cv", "prefs")
    cid = db.upsert_company("Acme")
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    p1 = _posting(cid, "280416", "IT Analyst")
    p2 = _posting(cid, "280417", "IT Analyst")                  # same vacancy, next id
    p3 = _posting(cid, "280418", "IT Analyst", loc="Dublin")    # different city = real
    summary = pipeline.RunSummary()
    pipeline._match_new(summary, verbose=False)
    recs = db.board_rows(statuses=["New"])
    assert len(recs) == 2, "twin must not become a second recommendation"
    kept_pids = {r["posting_id"] for r in recs}
    assert p1 in kept_pids and p3 in kept_pids and p2 not in kept_pids
    with db.connect() as conn:
        reason = conn.execute("SELECT filtered_reason FROM posting WHERE id=?",
                              (p2,)).fetchone()[0]
    assert reason and reason.startswith("duplicate posting")


def test_duplicate_of_decided_rec_is_suppressed(monkeypatch):
    """A repost of something the user already Applied to must not resurface."""
    _fresh_db()
    db.save_profile("cv", "prefs")
    cid = db.upsert_company("Acme")
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    p1 = _posting(cid, "A1", "Junior BI Analyst")
    db.save_recommendation(p1, 80, 85, "r", "n", llm_used=True)
    db.set_status(db.board_rows()[0]["rec_id"], "Applied")
    _posting(cid, "A2", "Junior BI Analyst (m/w/d)")            # repost, new id
    summary = pipeline.RunSummary()
    pipeline._match_new(summary, verbose=False)
    assert len(db.board_rows(statuses=["New"])) == 0
    assert len(db.board_rows(statuses=["Applied"])) == 1


def test_refilter_collapses_existing_duplicate_news():
    _fresh_db()
    db.save_profile("cv", "prefs")
    cid = db.upsert_company("Acme")
    pa = _posting(cid, "B1", "Data Analyst (m/w/d)")
    pb = _posting(cid, "B2", "Data Analyst")
    pc = _posting(cid, "B3", "Data Analyst", loc="Dublin")
    db.save_recommendation(pa, 70, 72, "r", "n", llm_used=True)
    db.save_recommendation(pb, 70, 81, "r", "n", llm_used=True)   # higher fit wins
    db.save_recommendation(pc, 70, 60, "r", "n", llm_used=True)
    removed = pipeline.refilter_existing()
    assert removed == 1
    rows = db.board_rows(statuses=["New"])
    assert {r["posting_id"] for r in rows} == {pb, pc}


def test_refilter_never_deletes_decided_duplicates():
    _fresh_db()
    db.save_profile("cv", "prefs")
    cid = db.upsert_company("Acme")
    pa = _posting(cid, "C1", "IT Consultant", loc="Geneva")
    pb = _posting(cid, "C2", "IT Consultant", loc="Geneva")
    for pid in (pa, pb):
        db.save_recommendation(pid, 50, 55, "r", "n", llm_used=True)
    for r in db.board_rows():
        db.set_status(r["rec_id"], "Applied")
    assert pipeline.refilter_existing() == 0
    assert len(db.board_rows(statuses=["Applied"])) == 2, \
        "decided rows are user history — only `reset` may collapse them"


# =============================================================================
# Retail / store roles are hard-excluded
# =============================================================================
def test_retail_store_roles_hard_filtered():
    hf = matching.hard_filter
    retail = [
        "STOCKROOM ASSOCIATE (M/F/D) 39H/WOCHE - FO METZINGEN",
        "Flagship Specialist, Stockroom Department (Full Time)",
        "Floor Manager (m/f/d) - 38,5h/Woche Vollzeit",
        "SALES ASSISTANT & VISUAL MERCHANDISER (M/F/D) 39H/WOCHE",
        "STORE MANAGER (M/W/D) IM FACTORY OUTLET LANDQUART",
        "Workforce Optimization Coach - NEW NIKE LONDON FLAGSHIP",
        "Retail Store Associate - Vienna",
        "Verkäufer (m/w/d) Teilzeit",
        "Nike Athlete (Sales Associate)",
    ]
    for t in retail:
        assert hf(t, "Vienna, Austria") == "retail/store role", t


def test_corporate_retail_roles_survive():
    hf = matching.hard_filter
    ok = [
        "Retail Media Analyst",
        "Analyst Kreditwürdigkeitsprüfung im Retailgeschäft",
        "Retail Analytics Business Analyst",
        "App Store Manager, Data & Insights",
    ]
    for t in ok:
        assert hf(t, "Vienna, Austria") != "retail/store role", t


# =============================================================================
# Reset
# =============================================================================
def _seed_reset_db():
    _fresh_db()
    db.save_profile("cv", "prefs")
    cid = db.upsert_company("Acme")
    keep_a = _posting(cid, "K1", "IT Consultant", loc="Geneva")
    keep_dup = _posting(cid, "K2", "IT Consultant", loc="Geneva")
    keep_m = _posting(cid, "K3", "Market Analyst")
    skip1 = _posting(cid, "S1", "Produktmanager (m/w/d)")
    skip2 = _posting(cid, "S2", "Junior Produktmanager")
    skip3 = _posting(cid, "S3", "Junior Controller")
    new1 = _posting(cid, "N1", "Junior BI Analyst")
    _posting(cid, "U1", "Unscored role")                        # no rec at all
    for pid, status in ((keep_a, "Applied"), (keep_dup, "Applied"),
                        (keep_m, "Marked"), (skip1, "Skipped"),
                        (skip2, "Skipped"), (skip3, "Skipped"), (new1, "New")):
        db.save_recommendation(pid, 50, 60, "r", "n", llm_used=True)
        rec = [r for r in db.board_rows() if r["posting_id"] == pid][0]
        db.set_status(rec["rec_id"], status)
    return cid


def test_reset_keeps_decisions_dedupes_and_freezes_learning():
    cid = _seed_reset_db()
    # freeze what the doomed Skipped rows taught us, then wipe
    feedback.persist_baselines(db, freeze_statuses=("Skipped",))
    kept = [r for r in db.recommended_open_postings()
            if r["status"] in ("Applied", "Marked")]
    by_fp, dups = {}, []
    for r in sorted(kept, key=lambda r: r["rec_id"]):
        fp = matching.posting_fingerprint(r["company_id"], r["title"], r["location"] or "")
        (dups.append(r["rec_id"]) if fp in by_fp else by_fp.setdefault(fp, r["rec_id"]))
    stats = db.reset_except(["Applied", "Marked"], drop_rec_ids=dups)

    assert stats["dropped_kept_duplicates"] == 1        # the twin IT Consultant
    assert stats["recommendations_kept"] == 2           # 1 Applied + 1 Marked
    assert stats["postings_kept"] == 2
    statuses = {r["status"] for r in db.board_rows()}
    assert statuses == {"Applied", "Marked"}
    assert db.stats()["postings_total"] == 2, "everything else is gone"
    assert not db.get_postings_to_score(), "queue is empty — full re-fetch next run"
    assert db.get_companies(), "companies survive"
    assert db.get_profile() is not None, "profile survives"

    # the learning survived the wipe: 2x product management + 1x finance
    skips, _applies = feedback.load_profiles(db)
    assert skips.get("product_management") == 2
    assert skips.get("finance_accounting") == 1

    # a second freeze+load must not double-count (no Skipped rows left)
    feedback.persist_baselines(db, freeze_statuses=("Skipped",))
    skips2, _ = feedback.load_profiles(db)
    assert skips2.get("product_management") == 2


def test_baseline_accumulates_across_resets():
    _seed_reset_db()
    feedback.persist_baselines(db, freeze_statuses=("Skipped",))
    db.reset_except(["Applied", "Marked"])
    # user skips one more PM role after the reset...
    cid = db.get_companies()[0]["id"]
    p = _posting(cid, "S9", "Product Owner")
    db.save_recommendation(p, 50, 40, "r", "n", llm_used=True)
    rec = [r for r in db.board_rows() if r["posting_id"] == p][0]
    db.set_status(rec["rec_id"], "Skipped")
    skips, _ = feedback.load_profiles(db)
    assert skips["product_management"] == 3             # 2 frozen + 1 live
    # ...and a second reset folds it into the baseline exactly once
    feedback.persist_baselines(db, freeze_statuses=("Skipped",))
    db.reset_except(["Applied", "Marked"])
    skips2, _ = feedback.load_profiles(db)
    assert skips2["product_management"] == 3


def test_cli_reset_dry_run_and_execute(capsys):
    _seed_reset_db()
    from jobscout import cli
    cli.main(["reset"])                                  # dry run
    out = capsys.readouterr().out
    assert "Dry run" in out and "keep      : Applied, Marked" in out
    assert db.stats()["recommendations"] == 7, "dry run must change nothing"
    cli.main(["reset", "--yes"])
    out = capsys.readouterr().out
    assert "Done:" in out
    assert db.stats()["recommendations"] == 2
    assert {r["status"] for r in db.board_rows()} == {"Applied", "Marked"}


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))

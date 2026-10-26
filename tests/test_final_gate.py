"""The 2026-07-10 quality gate: occupation hard filter, recommendation floor,
and the final LLM validation pass.

Reproduces the board pollution the user reported (Trial Lawyer, Finanzwirt,
Kfz-Mechatroniker, Karosseriebauer reached the board at fit 8-12) and pins the
three defenses that now prevent it, plus the guarantees that protect against
over-filtering (analytics-core guard, low-confidence keeps, removal-ratio
valve, Marked/Applied immunity).
"""
import os
import sys
import tempfile
from datetime import date
from pathlib import Path

os.environ["JOBSCOUT_DB"] = str(Path(tempfile.gettempdir()) / "jobscout_pytest_gate.db")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobscout import config, db, matching, pipeline, validate  # noqa: E402

TODAY = date.today().isoformat()
GOOD_DESC = "SQL, Power BI dashboards and reporting for stakeholders."


def _fresh_db():
    p = Path(os.environ["JOBSCOUT_DB"])
    if p.exists():
        p.unlink()
    db.init_db()


def _posting(cid, jid, title, loc="Vienna, Austria", desc=GOOD_DESC):
    pid, _ = db.upsert_posting(cid, jid, title, loc, TODAY, f"http://x/{jid}", desc)
    return pid


# =============================================================================
# 1. Occupation hard filter
# =============================================================================
def test_reported_offenders_are_hard_filtered():
    """Every title from the user's complaint dies before the LLM now."""
    for title in [
        "Associate Trial Lawyer (P-2)",
        "Finanzwirt / Diplom-Finanzwirt (m/w/d)",
        "Kfz-Mechatroniker (w/m/x)",
        "Karosseriebauer (w/m/x)",
    ]:
        assert matching.hard_filter(title, "Berlin, Germany") == \
            "off-profile occupation", title


def test_low_fit_skip_archive_titles_are_hard_filtered():
    """The wider families behind the user's 131 sub-30 manual skips."""
    for title in [
        "Maschinenbautechniker (m/w/d) für die Inhouse-Montage",
        "Elektromaschinentechniker (m/w/d)",
        "Servicetechniker für den Bereich Solartechnik (w/m/d)",
        "Systemtechnikerin | Gebäudeautomation/MSR-Technik",
        "Maintenance Technician (Coleshill, GB)",
        "Process Associate Technician",
        "Facharbeiter Linz (m/w/d)",
        "Investigative Toxicology Associate",
        "Syndikusrechtsanwalt (w/m/x) für Arbeitsrecht",
        "Berufskraftfahrer (m/w/d)",
        "Auslieferungsfahrer (m/w/d)",
        "Delivery Driver - London",
        "Lagermitarbeiter / Kommissionierer (m/w/d)",
        "Warehouse Associate",
        "Registered Nurse - ICU",
        "Gesundheits- und Krankenpfleger (m/w/d)",
        "Steuerfachangestellte (m/w/d)",
        "Finanzbuchhalter (m/w/d)",
        "Legal Counsel Employment Law",
        "Elektroniker für Betriebstechnik (m/w/d)",
        "Anlagenmechaniker SHK (m/w/d)",
        "Sous-Chef (m/w/d)",
        "Kellner / Servicekraft (m/w/d)",
        "Receptionist - Front Desk",
        "Flugbegleiter (m/w/d) Wien",
        "Reinigungskraft (m/w/d)",
        "Elektromonteur (m/w/d)",
    ]:
        assert matching.hard_filter(title, "Munich, Germany") == \
            "off-profile occupation", title


def test_analytics_titles_survive_occupation_filter():
    """Precision must not cost recall: analytics/business roles pass."""
    for title in [
        "Junior BI Analyst",
        "Business Analyst - Manufacturing",
        "Data Analyst (Automotive)",
        "Market Intelligence & Data Analyst (f/m/x)",
        "Analyst Fahrerassistenzsysteme",       # analytics-core guard
        "Reporting Specialist Logistics",
        "Junior Data Analyst - Healthcare",
        "Pricing Analyst - Legal Tech",
        "Business Intelligence Engineer",
        "Insights Analyst, Retail Media",
        "Junior Controller Analytics",           # 'controlling' is LLM-judged
        "Commercial Analyst - Aftersales",
    ]:
        assert matching.hard_filter(title, "Vienna, Austria", GOOD_DESC) is None, title


def test_occupation_examples_never_leak_via_descriptions():
    """The occupation check reads the TITLE only — a BI ad that mentions
    'production' or 'drivers' in its body must not be excluded."""
    desc = ("You will build Power BI dashboards for our production plants, "
            "analyse driver behaviour data and report KPIs. SQL required.")
    assert matching.hard_filter("Junior BI Analyst", "Graz, Austria", desc) is None


# =============================================================================
# 2. Hard-filter regressions: compounds the old regexes missed
# =============================================================================
def test_german_leadership_compounds_are_senior():
    for title in ["Werkstattleiter (w/m/x)", "Filialleiter (m/w/d)",
                  "Bauleiter Hochbau", "Standortleiter Logistik",
                  "Leiter Controlling", "Projektleitung Digitalisierung"]:
        reason = matching.hard_filter(title, "Berlin, Germany")
        assert reason in ("senior-level title", "retail/store role"), title


def test_begleiter_is_not_leadership():
    # 'Begleiter/Begleitung' (companion/support) must not trip the -leiter rule
    assert not matching._SENIOR_TITLE.search("Wissenschaftliche Begleitung")
    assert not matching._SENIOR_TITLE.search("Alltagsbegleiter (m/w/d)")


def test_thesis_and_casual_compounds_excluded():
    assert matching.hard_filter(
        "Studienabschlussarbeit Elektrische Sicherheit (w/m/x)",
        "München, Germany") == "internship/student/thesis role"
    assert matching.hard_filter(
        "Lehre zum Zerspanungstechniker (M/W/D)", "Linz, Austria") \
        == "internship/student/thesis role"
    assert matching.hard_filter(
        "Aushilfe (m/w/d) in Voll-/Teilzeit im Concept Store",
        "Hamburg, Germany") == "retail/store role"
    assert matching.hard_filter(
        "Specialist: Seasonal, Full-time and Part-time",
        "Berlin, Germany") == "retail/store role"


# =============================================================================
# 3. Prefilter: no-signal postings never reach the LLM queue
# =============================================================================
def test_no_signal_posting_capped_below_llm_cut():
    pre, sig = matching.prefilter("Programme Assistant", "Vienna, Austria", "")
    assert sig.get("no_signal") is True
    assert pre < config.PREFILTER_MIN_SCORE


def test_on_target_posting_still_ranks_high():
    pre, sig = matching.prefilter("Junior BI Analyst", "Vienna, Austria",
                                  GOOD_DESC)
    assert pre >= 80
    assert sig["title"] == "on-target (BI/analytics)"


def test_qa_test_analyst_is_hard_negative():
    pre, sig = matching.prefilter("Test Analyst", "Vienna, Austria", "")
    assert sig["title"] == "off-target (engineering/DS)"
    assert pre < config.PREFILTER_MIN_SCORE + 10


# =============================================================================
# 4. Recommendation floor
# =============================================================================
def _stub_llm_scores(monkeypatch, scores: dict):
    """score_batch stub returning fixed fit scores per posting title."""
    def fake_score_batch(cv, prefs, jobs, tracker=None, extra_rules=""):
        out = {}
        for j in jobs:
            fit = scores.get(j["title"])
            if fit is not None:
                out[j["id"]] = matching.MatchResult(
                    prefilter_score=j["pre_score"], fit_score=fit,
                    rationale=f"stub verdict for {j['title']}",
                    watch_outs="None", llm_used=True)
        return out
    monkeypatch.setattr(matching, "score_batch", fake_score_batch)
    monkeypatch.setattr(matching, "llm_available", lambda: True)


def test_floor_blocks_low_llm_scores_from_the_board(monkeypatch):
    _fresh_db()
    cid = db.upsert_company("FloorCo")
    db.set_company_ats(cid, "greenhouse", "floorco", "detected")
    db.save_profile("cv", "prefs")
    p_low = _posting(cid, "j1", "Data Coordinator", desc="SQL reporting")
    p_ok = _posting(cid, "j2", "Junior BI Analyst", desc=GOOD_DESC)
    _stub_llm_scores(monkeypatch, {"Data Coordinator": 10.0,
                                   "Junior BI Analyst": 88.0})
    monkeypatch.setattr(validate, "validate_new_recommendations",
                        lambda **kw: validate.ValidationOutcome())
    summary = pipeline.RunSummary()
    pipeline._match_new(summary, verbose=False)

    recs = db.board_rows()
    assert [r["title"] for r in recs] == ["Junior BI Analyst"]
    assert summary.floor_rejected == 1
    with db.connect() as conn:
        reason = conn.execute("SELECT filtered_reason FROM posting WHERE id=?",
                              (p_low,)).fetchone()[0]
    assert reason.startswith("scored 10/100 by the LLM")
    assert p_ok  # silence linter


def test_refilter_sweeps_existing_subfloor_new_recs():
    _fresh_db()
    cid = db.upsert_company("SweepCo")
    p1 = _posting(cid, "j1", "Junior BI Analyst")
    p2 = _posting(cid, "j2", "Commercial Data Coordinator")
    p3 = _posting(cid, "j3", "Ops Data Coordinator")
    db.save_recommendation(p1, 80, 85.0, "good", "None", llm_used=True)
    db.save_recommendation(p2, 50, 12.0, "bad", "None", llm_used=True)   # sweep
    db.save_recommendation(p3, 50, 15.0, "meh", "None", llm_used=True)   # Marked
    with db.connect() as conn:
        conn.execute("UPDATE recommendation SET status='Marked' "
                     "WHERE posting_id=?", (p3,))
    removed = pipeline.refilter_existing()
    assert removed == 1
    statuses = {r["title"]: r["status"] for r in db.board_rows()}
    assert "Commercial Data Coordinator" not in statuses      # swept
    assert statuses["Ops Data Coordinator"] == "Marked"       # user rows immune
    assert statuses["Junior BI Analyst"] == "New"


def test_heuristic_scores_are_not_floor_swept():
    _fresh_db()
    cid = db.upsert_company("HeurCo")
    p1 = _posting(cid, "j1", "Junior BI Analyst")
    db.save_recommendation(p1, 15, 18.0, "provisional", "None", llm_used=False)
    assert pipeline.refilter_existing() == 0
    assert len(db.board_rows()) == 1


# =============================================================================
# 5. Final LLM validation gate
# =============================================================================
def _seed_board(n_good=2, n_bad=1):
    _fresh_db()
    cid = db.upsert_company("GateCo")
    ids = []
    for i in range(n_good):
        pid = _posting(cid, f"g{i}", f"Junior BI Analyst {i}")
        db.save_recommendation(pid, 80, 80.0 + i, "fits", "None", llm_used=True)
    for i in range(n_bad):
        pid = _posting(cid, f"b{i}", f"Facility Coordinator {i}")
        db.save_recommendation(pid, 45, 48.0, "hmm", "None", llm_used=True)
    with db.connect() as conn:
        for r in conn.execute("SELECT id, posting_id FROM recommendation"):
            ids.append(r["id"])
    return cid, ids


def _stub_reviews(monkeypatch, verdict_fn):
    """Patch the validator's LLM call to produce verdicts via verdict_fn(rec)."""
    def fake_review(system_blocks, batch, tracker):
        return {r["rec_id"]: verdict_fn(r) for r in batch}
    monkeypatch.setattr(validate, "_review_batch", fake_review)
    monkeypatch.setattr(matching, "llm_available", lambda: True)


def test_validator_removes_confident_mismatches(monkeypatch):
    _seed_board(n_good=2, n_bad=1)
    _stub_reviews(monkeypatch, lambda r: (
        {"v": "remove", "why": "facility ops role, no analytics", "conf": "hi"}
        if "Facility" in r["title"] else
        {"v": "keep", "why": "", "conf": "hi"}))
    out = validate.validate_new_recommendations()
    assert (out.reviewed, out.removed, out.kept) == (3, 1, 2)
    titles = [r["title"] for r in db.board_rows()]
    assert all("Facility" not in t for t in titles) and len(titles) == 2
    with db.connect() as conn:
        reason = conn.execute(
            """SELECT filtered_reason FROM posting
               WHERE filtered_reason IS NOT NULL""").fetchone()[0]
    assert reason == "failed final review: facility ops role, no analytics"
    # kept rows carry the audit stamp
    for r in db.board_rows():
        assert r["validated_at"]


def test_low_confidence_removal_is_a_keep(monkeypatch):
    _seed_board(n_good=1, n_bad=1)
    _stub_reviews(monkeypatch, lambda r: (
        {"v": "remove", "why": "maybe wrong", "conf": "lo"}
        if "Facility" in r["title"] else {"v": "keep", "why": "", "conf": "hi"}))
    out = validate.validate_new_recommendations()
    assert out.removed == 0 and out.kept == 2 and out.flagged == 1
    flagged = [r for r in db.board_rows() if "Facility" in r["title"]][0]
    assert flagged["validation_note"].startswith("reviewer unsure")


def test_removal_ratio_guard_discards_a_haywire_pass(monkeypatch):
    _seed_board(n_good=8, n_bad=2)
    _stub_reviews(monkeypatch, lambda r: {"v": "remove", "why": "x", "conf": "hi"})
    out = validate.validate_new_recommendations()
    assert out.aborted is True and out.removed == 0
    assert len(db.board_rows()) == 10          # nothing deleted
    # nothing marked validated either — the pass never happened
    assert all(not r["validated_at"] for r in db.board_rows())


def test_llm_failure_changes_nothing(monkeypatch):
    _seed_board(n_good=2, n_bad=1)
    monkeypatch.setattr(validate, "_review_batch", lambda *a, **k: {})
    monkeypatch.setattr(matching, "llm_available", lambda: True)
    out = validate.validate_new_recommendations()
    assert out.reviewed == 0 and out.pending == 3
    assert len(db.board_rows()) == 3
    assert all(not r["validated_at"] for r in db.board_rows())


def test_validation_is_incremental(monkeypatch):
    _seed_board(n_good=2, n_bad=0)
    _stub_reviews(monkeypatch, lambda r: {"v": "keep", "why": "", "conf": "hi"})
    assert validate.validate_new_recommendations().reviewed == 2
    # second run: nothing left to validate
    assert validate.validate_new_recommendations().reviewed == 0
    # a re-score (in a later run — bump the clock) resets the validation stamp
    rec = db.board_rows()[0]
    db.save_recommendation(rec["posting_id"], 80, 82.0, "rescored", "None",
                           llm_used=True)
    with db.connect() as conn:
        conn.execute(
            "UPDATE recommendation SET scored_at = '9999-01-01T00:00:00+00:00' "
            "WHERE posting_id=?", (rec["posting_id"],))
    assert validate.validate_new_recommendations().reviewed == 1


def test_user_decided_rows_are_never_validated(monkeypatch):
    _seed_board(n_good=1, n_bad=1)
    with db.connect() as conn:
        conn.execute("UPDATE recommendation SET status='Marked'")
    _stub_reviews(monkeypatch, lambda r: {"v": "remove", "why": "x", "conf": "hi"})
    out = validate.validate_new_recommendations()
    assert out.reviewed == 0
    assert len(db.board_rows()) == 2


def test_long_tail_is_exempt_from_validation(monkeypatch):
    """Regression for 2026-07-12: the score-all board keeps a deliberate
    sub-shortlist long tail (fit 20-44). Sending it to the validator made the
    validator vote remove on 665/719 rows, trip the ratio guard every run, and
    re-burn the whole review budget forever. Sub-band rows must never reach
    the validator — and must never be deleted or stamped by it."""
    _fresh_db()
    cid = db.upsert_company("TailCo")
    for i in range(20):                                   # the long tail
        pid = _posting(cid, f"t{i}", f"Indirect Tax Manager {i}")
        db.save_recommendation(pid, 30, 22.0, "not analytics", "None",
                               llm_used=True)
    good = _posting(cid, "g0", "Junior BI Analyst")
    db.save_recommendation(good, 85, 82.0, "fits", "None", llm_used=True)

    # band filter applies at the query level
    rows = db.recs_to_validate()
    assert [r["title"] for r in rows] == ["Junior BI Analyst"]

    # a haywire reviewer can therefore no longer see (or nuke) the tail,
    # and reviewing 1 good rec doesn't trip the >=8 ratio guard
    _stub_reviews(monkeypatch, lambda r: {"v": "keep", "why": "", "conf": "hi"})
    out = validate.validate_new_recommendations()
    assert (out.reviewed, out.kept, out.aborted) == (1, 1, False)
    board = db.board_rows()
    assert len(board) == 21                               # tail untouched
    stamped = [r["title"] for r in board if r["validated_at"]]
    assert stamped == ["Junior BI Analyst"]               # tail never stamped


def test_validation_band_respects_company_scope():
    """Band filter and company scoping combine (param ordering regression)."""
    _fresh_db()
    c1 = db.upsert_company("ScopeA")
    c2 = db.upsert_company("ScopeB")
    p1 = _posting(c1, "a1", "BI Analyst A")
    db.save_recommendation(p1, 80, 75.0, "fits", "None", llm_used=True)
    p2 = _posting(c2, "b1", "BI Analyst B")
    db.save_recommendation(p2, 80, 75.0, "fits", "None", llm_used=True)
    p3 = _posting(c2, "b2", "Tax Clerk B")
    db.save_recommendation(p3, 30, 21.0, "weak", "None", llm_used=True)

    rows = db.recs_to_validate(company_ids=[c2])
    assert [r["title"] for r in rows] == ["BI Analyst B"]


# =============================================================================
# 6. Gate caps: the new 'trade' function category
# =============================================================================
def test_trade_gate_cap():
    caps = matching._gate_caps({"rt": "ok", "sen": "ok", "loc": "ok",
                                "lad": "de", "lrq": "none", "fn": "trade"})
    assert (12, "function: trade") in caps


def test_deterministic_occupation_cap():
    caps = matching._deterministic_caps(
        {"title": "Kfz-Mechatroniker (w/m/x)", "location": "Hamburg, Germany",
         "description": "Wartung und Instandsetzung von Fahrzeugen."})
    assert any(v == 12 and "occupation" in why for v, why in caps)


# =============================================================================
# 7. Dashboard stats: run-vs-database separation (2026-07-11)
# =============================================================================
def test_stats_composition_partitions_open_postings():
    """composition segments must sum to the open-posting total."""
    _fresh_db()
    cid = db.upsert_company("CompCo")
    p1 = _posting(cid, "c1", "Junior BI Analyst")                    # board
    db.save_recommendation(p1, 80, 82.0, "r", "None", llm_used=True)
    p2 = _posting(cid, "c2", "Reporting Analyst")                    # below cut
    db.save_recommendation(p2, 50, 30.0, "r", "None", llm_used=True)
    _posting(cid, "c3", "Data Analyst")                              # queued
    p4 = _posting(cid, "c4", "Kfz-Mechatroniker (w/m/x)")            # filtered
    db.mark_posting_filtered(p4, "off-profile occupation")
    p5 = _posting(cid, "c5", "Insights Analyst")                     # archived
    db.save_recommendation(p5, 70, 60.0, "r", "None", llm_used=True)
    db.set_status(db.board_rows(statuses=["New"], db_path=None)[  # the p5 rec
        [r["posting_id"] for r in db.board_rows(statuses=["New"])].index(p5)
    ]["rec_id"], "Skipped")

    from jobscout import web
    c = web._composition()
    # Skipped/Rejected recs on open postings are ARCHIVE, not board — they
    # must land in their own segment (they were miscounted as board rows).
    assert c == {"open": 5, "filtered": 1, "queued": 1,
                 "below_shortlist": 1, "archived": 1, "board": 1}
    assert sum(v for k, v in c.items() if k != "open") == c["open"]


def test_stats_last_run_survives_restart():
    """last_run comes from meta, not process memory."""
    import json
    _fresh_db()
    db.meta_set("last_run_at", "2026-07-11T09:00:00+00:00")
    db.meta_set("last_run_summary", json.dumps({"scored": 5, "floor_rejected": 1}))
    from jobscout import web
    lr = web._last_run()
    assert lr["at"].startswith("2026-07-11")
    assert lr["summary"]["floor_rejected"] == 1

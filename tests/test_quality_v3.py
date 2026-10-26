"""Quality overhaul (2026-07-09): hard-requirement enforcement, skip-pattern
learning, gated LLM scoring, honest fallbacks, and the Marked status.

Run:  JOBSCOUT_DB=/tmp/test.db python -m pytest tests/ -q
"""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["JOBSCOUT_DB"] = str(Path(tempfile.gettempdir()) / "jobscout_pytest.db")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobscout import config, db, feedback, matching, pipeline  # noqa: E402


def _fresh_db():
    p = Path(os.environ["JOBSCOUT_DB"])
    if p.exists():
        p.unlink()
    db.init_db()


# =============================================================================
# Required languages (beyond EN/DE)
# =============================================================================
def test_required_languages_hard_requirements():
    rl = matching.required_languages
    assert rl("Data Analyst", "You are fluent in French.") == ["french"]
    assert rl("Analyst", "Dutch native speaker required") == ["dutch"]
    assert rl("Analyst", "Verhandlungssicheres Französisch") == ["french"]
    assert rl("Analyst", "Sehr gute Niederländischkenntnisse") == ["dutch"]
    assert rl("Analyst", "Proficiency in Swedish (C1) is essential") == ["swedish"]
    # title grammar
    assert rl("Operations Associate (French speaking)", "") == ["french"]
    assert rl("Customer Care Agent - French market", "") == ["french"]
    assert rl("Specialist, Premium Support (French/English)", "") == ["french"]
    assert rl("Integration Engineer (French fluency)", "") == ["french"]


def test_required_languages_pass_cases():
    rl = matching.required_languages
    # satisfiable alternatives that include EN/DE
    assert rl("Graduate Opportunities - French, German or Spanish",
              "Fluency in French, German or Spanish") == []
    assert rl("Analyst", "fluent in English or French") == []
    # softened mentions are not requirements
    assert rl("Analyst", "French is a plus") == []
    assert rl("Analyst", "Französisch von Vorteil") == []
    assert rl("Analyst", "additional languages such as Italian are appreciated") == []
    # no cue at all
    assert rl("Analyst", "our clients include French retailers") == []
    assert rl("Analyst", "fluent English and German required") == []
    assert rl("Junior BI Analyst", "SQL and Power BI") == []


def test_hard_filter_drops_language_requirements():
    assert "french" in matching.hard_filter(
        "Data Analyst", "Paris", "You must be fluent in French and English.")
    assert matching.hard_filter(
        "Data Analyst", "Paris", "English required; French is a plus. " * 10 +
        "We use SQL and Tableau for reporting and analytics every day with "
        "your stakeholders and you will join the analytics team.") is None


# =============================================================================
# Years-of-experience extraction
# =============================================================================
def test_required_years_plain_and_spelled():
    ry = matching.required_years
    assert ry("You have 5 years of professional experience in analytics") == 5
    assert ry("7 years' experience in BI") == 7
    assert ry("five years of relevant experience") == 5
    assert ry("mit 6 Jahren einschlägiger Berufserfahrung") == 6
    assert ry("fünf Jahre Berufserfahrung") == 5
    assert ry("experience of 8+ years") == 8
    # softened plain statements do NOT hard-count
    assert ry("ideally 5 years of experience") is None
    assert ry("5 years of experience preferred") is None
    # ranges count their minimum, not the upper bound
    assert ry("5-7 years of relevant experience") == 5
    # strong phrasings ignore softeners
    assert ry("minimum of 5 years, ideally more") == 5
    assert ry("2 years of experience") == 2


def test_hard_filter_five_years_deep_in_description():
    long_desc = ("About us. " * 200 +
                 "Requirements: 6 years of professional experience in data.")
    assert "6" in matching.hard_filter("Data Analyst", "Vienna", long_desc)


# =============================================================================
# Seniority + location
# =============================================================================
def test_expert_and_architect_are_senior():
    hf = matching.hard_filter
    assert hf("Data Modeling Expertin", "Berlin, Germany") == "senior-level title"
    assert hf("Experte für Controlling und Reporting Systeme", "Wetzlar, Germany") \
        == "senior-level title"
    assert hf("Data & AI Governance Architect", "Amsterdam") == "senior-level title"
    # 'expertise' in a description must not trip the title rule
    assert hf("Junior Data Analyst", "Vienna, Austria",
              "You bring expertise in SQL and reporting dashboards for our "
              "stakeholders and analytics team members every single day.") is None


def test_location_us_namesakes_and_spain():
    ls = matching.location_status
    assert ls("Cambridge, MA") == "outside"
    assert ls("Vienna, VA") == "outside"
    assert ls("Cambridge") == "target"
    assert ls("Valencia") == "outside"
    assert ls("Barcelona Gran Vía") == "target"
    assert ls("Bangui, Central African Republic") == "outside"
    assert ls("Amsterdam, NH, NL") == "target"   # NH = Noord-Holland here
    assert ls("Sindelfingen, DE") == "target"


# =============================================================================
# Fixed-term detection
# =============================================================================
def test_fixed_term_signals():
    ft = matching.is_fixed_term
    assert ft("Demand Generation Analyst - 15 month fixed term cover")
    assert ft("Logistics Analyst - 11 Months Contract")
    assert ft("Junior Professional IT-Service (befristet auf 12 Monate)")
    assert ft("Sales & Communication Associate (Limited Full-time)")
    assert ft("Analyst", "This is a 12-month contract covering parental leave.")
    assert not ft("Data Analyst", "3 months notice period; permanent contract")
    assert not ft("Junior BI Analyst", "unbefristetes Arbeitsverhältnis")


# =============================================================================
# Skip-pattern learning (feedback module)
# =============================================================================
def test_categorize_titles_from_real_skips():
    cat = feedback.categorize_title
    assert cat("(Junior) Produktmanager (m/w/d) Mobilfunk") == "product_management"
    assert cat("Associate Technology Consultant - Graduate Program") == "consulting"
    assert cat("Deployment Strategist - UK Government") == "consulting"
    assert cat("Lakebase Sales Specialist") == "sales_retail"
    assert cat("Junior Controller (m/w/d)") == "finance_accounting"
    assert cat("Junior Policy Analyst") == "policy_program"
    assert cat("Operations Supervisor, Quick Commerce") == "ops_logistics"
    assert cat("HR Analyst – Develop & Perform") == "hr_people"
    assert cat("Field Marketing Specialist") == "marketing_comms"
    assert cat("Specialist, Premium Support (French/English)") == "customer_support"
    assert cat("Applikationsmanager*in") == "it_admin"
    assert cat("CDD Risk Analyst") == "legal_risk_compliance"
    assert cat("VIE - Data Operations & Governance Analyst") == "data_governance"
    assert cat("EMEIA Sales Finance - Analytics Engineer") == "engineering"
    assert cat("Program Manager, Planning") == "project_process_mgmt"
    # on-target roles stay uncategorized
    assert cat("Junior BI Analyst") is None
    assert cat("Market Intelligence & Data Analyst (f/m/x)") is None


def test_skip_profile_and_prompt_section():
    skips = feedback.skip_profile([
        "(Junior) Produktmanager (m/w/d) A", "(Junior) Produktmanager (m/w/d) B",
        "Junior Produktmanager (m/w/d) C", "Product Owner D",
        "Junior Controller (m/w/d)", "Finance Analyst", "Treasury Analyst",
        "Junior BI Analyst",   # uncategorized — never becomes a pattern
    ])
    assert skips["product_management"] == 4
    assert skips["finance_accounting"] == 3
    section = feedback.prompt_section(skips)
    assert "product-manager" in section
    assert "4x" in section
    # applied evidence weakens a pattern
    applies = feedback.applied_profile(["Product Owner X", "Produktmanager Y"])
    avoided = feedback.avoided_categories(skips, applies)
    assert "product_management" not in avoided     # 2 applies vs 4 skips
    assert "finance_accounting" in avoided


def test_prefilter_penalty_guarded():
    skips = {"product_management": 9, "ops_logistics": 9}
    pen, cat = feedback.prefilter_penalty("(Junior) Produktmanager (m/w/d)", skips)
    assert pen < 0 and cat == "product_management"
    # analytics-core titles are protected from the blanket penalty
    pen2, _ = feedback.prefilter_penalty("Transport Performance Analyst", skips)
    assert pen2 == 0.0
    pen3, _ = feedback.prefilter_penalty("Junior BI Analyst", skips)
    assert pen3 == 0.0
    # below the evidence threshold nothing fires
    pen4, _ = feedback.prefilter_penalty("Produktmanager", {"product_management": 3})
    assert pen4 == 0.0


def test_prefilter_ranks_known_junk_down():
    def pre(t, loc="Vienna, Austria", d=""):
        return matching.prefilter(t, loc, d)[0]
    good = pre("Junior BI Analyst", d="SQL, Power BI, reporting dashboards")
    assert good >= 90
    assert pre("(Junior) Produktmanager (m/w/d)") < config.PREFILTER_MIN_SCORE + 10
    assert pre("Analytics Engineer") < config.PREFILTER_MIN_SCORE
    assert pre("Junior Policy Analyst") < good - 30
    assert pre("Business Analyst Finance – Shared Services") < good - 30
    # consulting is mild: down-ranked, not buried
    c = pre("Associate Technology Consultant")
    assert config.PREFILTER_MIN_SCORE - 10 < c < good - 25


# =============================================================================
# Honest scoring: heuristic cap + gate clamps
# =============================================================================
def test_heuristic_is_capped_and_labeled():
    res = matching.heuristic_score(
        "Junior BI Analyst", "Acme", "Vienna",
        "SQL Power BI reporting dashboard analytics kpi stakeholder", 100.0,
        {"title": "on-target (BI/analytics)", "location": "preferred region"})
    assert res.fit_score <= matching.HEURISTIC_MAX_FIT
    assert not res.llm_used
    assert "not yet LLM-verified" in res.rationale


def _fake_llm(monkeypatch, verdicts):
    """Stub the API: score_batch sees a canned JSON verdict array."""
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(matching, "_llm_call",
                        lambda *a, **k: json.dumps(verdicts))


def _job(jid=1, **kw):
    base = dict(id=jid, title="Junior BI Analyst", company="Acme",
                location="Vienna, Austria", posted="2026-07-01",
                description="SQL and Power BI reporting dashboards for "
                            "commercial stakeholders across the region.",
                pre_score=80.0, signals={})
    base.update(kw)
    return base


def test_gate_clamp_from_model_report(monkeypatch):
    # The model admits 'senior' in the gates but still returns 90 — clamp to 30.
    _fake_llm(monkeypatch, [{
        "id": 1, "g": {"rt": "ok", "sen": "senior", "loc": "ok", "lad": "en",
                       "lrq": "none", "fn": "core", "emp": "perm"},
        "s": 90, "why": "great fit", "watch": "None"}])
    out = matching.score_batch("cv", "prefs", [_job()])
    assert out[1].fit_score == 30.0
    assert "capped" in out[1].watch_outs


def test_gate_clamp_function_category(monkeypatch):
    _fake_llm(monkeypatch, [{
        "id": 1, "g": {"rt": "ok", "sen": "ok", "loc": "ok", "lad": "en",
                       "lrq": "none", "fn": "pm", "emp": "perm"},
        "s": 85, "why": "", "watch": "None"}])
    out = matching.score_batch("cv", "prefs", [_job(title="Product Manager")])
    assert out[1].fit_score == 40.0


def test_deterministic_clamp_beats_sloppy_model(monkeypatch):
    # Model reports all-ok and 88, but the description demands 6 years and
    # fluent French — our own extractors cap it regardless.
    _fake_llm(monkeypatch, [{
        "id": 1, "g": {"rt": "ok", "sen": "ok", "loc": "ok", "lad": "en",
                       "lrq": "none", "fn": "core", "emp": "perm"},
        "s": 88, "why": "", "watch": "None"}])
    out = matching.score_batch("cv", "prefs", [_job(
        description="Minimum of 6 years of experience. Fluent French required.")])
    assert out[1].fit_score <= 30.0


def test_clean_pass_keeps_model_score(monkeypatch):
    _fake_llm(monkeypatch, [{
        "id": 1, "g": {"rt": "ok", "sen": "ok", "loc": "ok", "lad": "en",
                       "lrq": "none", "fn": "core", "emp": "perm"},
        "s": 88, "why": "on-target", "watch": "None"}])
    out = matching.score_batch("cv", "prefs", [_job()])
    assert out[1].fit_score == 88.0
    assert out[1].llm_used


def test_llm_failure_leaves_posting_queued(monkeypatch):
    """With a key configured, a failed batch must NOT produce heuristic scores;
    the posting stays unscored so the next run retries it."""
    _fresh_db()
    cid = db.upsert_company("Acme")
    from datetime import date
    db.upsert_posting(cid, "Q1", "Junior BI Analyst", "Vienna, Austria",
                      date.today().isoformat(), "u",
                      "SQL, Power BI, reporting dashboards for stakeholders")
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(matching, "score_batch", lambda *a, **k: {})
    summary = pipeline.RunSummary()
    pipeline._match_new(summary, verbose=False)
    assert summary.scored == 0
    assert db.get_postings_to_score(), "posting must remain in the queue"
    assert any("queued for next run" in n for n in summary.notes)


def test_keyless_mode_still_scores_heuristically(monkeypatch):
    _fresh_db()
    cid = db.upsert_company("Acme")
    from datetime import date
    db.upsert_posting(cid, "Q2", "Junior BI Analyst", "Vienna, Austria",
                      date.today().isoformat(), "u",
                      "SQL, Power BI, reporting dashboards for stakeholders")
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    summary = pipeline.RunSummary()
    pipeline._match_new(summary, verbose=False)
    assert summary.scored == 1
    row = db.board_rows()[0]
    assert not row["llm_used"]
    assert row["fit_score"] <= matching.HEURISTIC_MAX_FIT


# =============================================================================
# Description trimming
# =============================================================================
def test_trim_description_keeps_requirements_and_facts():
    desc = ("About the role. " * 300 +
            " Anforderungen: mindestens 3 Jahre Erfahrung mit SQL. " +
            "Fließendes Niederländisch erforderlich. " + "More text. " * 100)
    out = matching.trim_description(desc, budget=2000)
    assert len(out) < 3200
    assert "Anforderungen" in out
    assert "Niederländisch" in out
    # short descriptions come through whole (whitespace-collapsed)
    assert matching.trim_description("a\n\n  b") == "a b"


def test_payload_carries_extractor_facts():
    p = matching._job_payload(
        7, "Data Analyst", "Acme", "Paris", "2026-07-01",
        "Fluent French required. Minimum of 6 years of experience. "
        "Fixed-term contract.")
    assert "languages_required=french" in p["facts"]
    assert "years_required=6" in p["facts"]
    assert "fixed_term=yes" in p["facts"]


# =============================================================================
# Marked status — end to end
# =============================================================================
def test_marked_status_full_flow():
    _fresh_db()
    assert config.STATUS_MARKED in config.ALL_STATUSES
    cid = db.upsert_company("Acme")
    from datetime import date
    today = date.today().isoformat()
    pid, _ = db.upsert_posting(cid, "M1", "Junior BI Analyst", "Vienna, Austria",
                               today, "u", "SQL reporting")
    db.save_recommendation(pid, 80, 85, "fits", "none", llm_used=True)
    rec = db.board_rows()[0]
    db.set_status(rec["rec_id"], "Marked")

    # visible under its own status filter + in stats
    assert db.board_rows(statuses=["Marked"])[0]["status"] == "Marked"
    assert db.stats()["marked"] == 1

    # refilter must never touch a Marked rec, even if rules would drop it now
    removed = pipeline.refilter_existing()
    assert removed == 0
    assert db.board_rows(statuses=["Marked"]), "Marked rec survived refilter"

    # a Marked rec stays visible after the posting closes (like Applied)
    db.close_missing_postings(cid, seen_job_ids=[])
    rows = db.board_rows(statuses=["Marked"])
    assert rows and not rows[0]["is_open"]


def test_marked_api_endpoints():
    from fastapi.testclient import TestClient
    from jobscout import web
    _fresh_db()
    cid = db.upsert_company("Acme")
    from datetime import date
    pid, _ = db.upsert_posting(cid, "M2", "Junior Data Analyst", "Vienna, Austria",
                               date.today().isoformat(), "u", "d")
    db.save_recommendation(pid, 70, 72, "r", "n", llm_used=True)
    rec_id = db.board_rows()[0]["rec_id"]
    client = TestClient(web.app)
    r = client.post(f"/api/rec/{rec_id}/status", json={"status": "Marked"})
    assert r.status_code == 200
    assert client.get("/api/board?status=Marked").json()["count"] == 1
    # default board view includes Marked
    assert client.get("/api/board").json()["count"] == 1
    assert client.get("/api/stats").json()["marked"] == 1
    r = client.post(f"/api/rec/{rec_id}/status", json={"status": "Bogus"})
    assert r.status_code == 400


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))

"""Core guarantees: dedupe, decision persistence, closing, and prefilter logic.

Run:  JOBSCOUT_DB=/tmp/test.db python -m pytest tests/ -q
or:   JOBSCOUT_DB=/tmp/test.db python tests/test_core.py
"""
import os
import sys
import tempfile
from pathlib import Path

# Use an isolated temp DB regardless of how this is invoked.
os.environ["JOBSCOUT_DB"] = str(Path(tempfile.gettempdir()) / "jobscout_pytest.db")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobscout import db, matching  # noqa: E402


def _fresh_db():
    p = Path(os.environ["JOBSCOUT_DB"])
    if p.exists():
        p.unlink()
    db.init_db()


def test_dedupe_and_first_seen_preserved():
    _fresh_db()
    cid = db.upsert_company("Acme", category="Test", locations="Vienna")
    pid1, new1 = db.upsert_posting(cid, "JOB-1", "Data Analyst", "Vienna", "2026-06-01",
                                   "http://x", "desc")
    pid2, new2 = db.upsert_posting(cid, "JOB-1", "Data Analyst (updated)", "Vienna",
                                   "2026-06-01", "http://x", "desc")
    assert new1 is True and new2 is False
    assert pid1 == pid2
    s = db.stats()
    assert s["postings_total"] == 1, "same job_id must not create a duplicate"


def test_status_persists_across_rescore():
    _fresh_db()
    cid = db.upsert_company("Acme")
    pid, _ = db.upsert_posting(cid, "JOB-2", "BI Analyst", "Dublin", "2026-06-02", "u", "d")
    db.save_recommendation(pid, 80, 88, "fits", "none", llm_used=False)
    rec = db.board_rows()[0]
    db.set_status(rec["rec_id"], "Applied")
    db.set_notes(rec["rec_id"], "sent CV 2026-06-10")
    # Pipeline runs again and re-scores the same posting:
    db.save_recommendation(pid, 82, 90, "fits better", "none", llm_used=True)
    rec2 = db.board_rows()[0]
    assert rec2["status"] == "Applied", "user's decision must survive a re-score"
    assert rec2["notes"] == "sent CV 2026-06-10", "notes must survive a re-score"
    assert rec2["fit_score"] == 90, "score should refresh"


def test_vanished_postings_marked_closed():
    _fresh_db()
    cid = db.upsert_company("Acme")
    db.upsert_posting(cid, "A", "Role A", "Vienna", "2026-06-01", "u", "d")
    db.upsert_posting(cid, "B", "Role B", "Vienna", "2026-06-01", "u", "d")
    closed = db.close_missing_postings(cid, seen_job_ids=["A"])  # B disappeared
    assert closed == 1
    assert db.stats()["postings_open"] == 1


def test_prefilter_targets_bi_not_engineering():
    on, _ = matching.prefilter("Business Analyst", "Vienna, Austria", "SQL Power BI dashboards")
    eng, _ = matching.prefilter("Software Engineer, Analytics Platform", "Dublin",
                                "microservices kubernetes")
    ds, _ = matching.prefilter("Senior Data Scientist", "Munich", "pytorch deep learning")
    tax, _ = matching.prefilter("Payroll Tax Analyst", "Dublin", "tax filings")
    assert on > 70, f"BA role should score high, got {on}"
    assert eng < 40, f"software engineer should be filtered, got {eng}"
    assert ds < 40, f"data scientist should be down-ranked, got {ds}"
    assert tax < on, "off-domain analyst should rank below a true BI analyst"


def test_prefilter_rejects_sales_roles_naming_bi_tools():
    """Regression: a sales/other-function title must NOT score on-target just
    because it name-drops a BI tool (the strong-vs-soft-negative bug)."""
    ae, sig = matching.prefilter("Tableau Account Executive", "Dublin, Ireland",
                                 "Sell Tableau licences to enterprise customers; carry a quota.")
    am, _ = matching.prefilter("Account Manager, Analytics", "London, United Kingdom",
                               "Own a book of accounts")
    rec, _ = matching.prefilter("Technical Recruiter", "Berlin, Germany", "Hire engineers")
    ba, _ = matching.prefilter("Business Analyst", "Dublin, Ireland", "SQL Power BI dashboards")
    assert ae < 45, f"'Account Executive' (sales) must be down-ranked, got {ae}"
    assert am < 45, f"'Account Manager' (sales) must be down-ranked, got {am}"
    assert rec < 45, f"recruiter must be down-ranked, got {rec}"
    assert ba > ae and ba > am, "a real BI analyst must outrank sales roles"
    assert "other function" in sig.get("title", ""), sig


def test_location_preference_applied():
    eu, _ = matching.prefilter("Data Analyst", "Dublin, Ireland", "sql reporting")
    us, _ = matching.prefilter("Data Analyst", "Bengaluru, India", "sql reporting")
    assert eu > us, "preferred-region role should outrank a far one with same title"


def test_location_targets_exactly_the_profile_countries():
    """The stored profile targets AT, DE, UK, IE, SE, CH, ES(Barcelona), FR, NL.
    Everything else — including the rest of Spain and non-target EU countries —
    is out of scope."""
    for loc in ["Vienna, Austria", "Berlin, Germany", "London, United Kingdom",
                "Dublin, Ireland", "Stockholm, Sweden", "Zurich, Switzerland",
                "Barcelona, Spain", "Paris, France", "Amsterdam, Netherlands",
                "Munich", "Rijswijk, Netherlands"]:
        assert matching.location_status(loc) == "target", loc
    for loc in ["Madrid, Spain", "Milano, Italy", "Copenhagen, Denmark",
                "Brussels, Belgium", "Lisbon, Portugal", "Warsaw, Poland",
                "Austin, Texas", "Bengaluru, India", "US-CA-Menlo Park",
                "New York, US", "Singapore"]:
        assert matching.location_status(loc) == "outside", loc
    # empty/ambiguous stays undecided for the LLM
    assert matching.location_status("") == "unknown"


def test_hard_filter_excludes_unwanted_role_types():
    hf = matching.hard_filter
    assert hf("Werkstudent Business Intelligence", "Berlin, Germany") is not None
    assert hf("Praktikant Data & AI (w/m/x)", "Munich") is not None
    assert hf("Internship Group Risk Reporting", "Vienna, Austria") is not None
    assert hf("Trainee Data Analytics", "Vienna, Austria") is not None
    assert hf("Master Thesis: Analytics", "Graz, Austria") is not None
    assert hf("Senior Data Analyst", "Vienna, Austria") is not None
    assert hf("Head of Business Intelligence", "Berlin") is not None
    assert hf("Teamleitung Controlling", "Wien") is not None
    # ... while the wanted ones pass
    assert hf("Junior Business Analyst", "Vienna, Austria") is None
    assert hf("Graduate Data Analyst", "Dublin, Ireland") is None
    assert hf("Business Intelligence Analyst", "Barcelona, Spain") is None
    # 'intern' must not fire inside 'International'
    assert hf("International Business Analyst", "London, United Kingdom") is None


def test_hard_filter_excludes_by_years_and_language():
    hf = matching.hard_filter
    assert hf("Business Analyst", "Vienna, Austria",
              "We require at least 7 years of experience.") is not None
    assert hf("Business Analyst", "Vienna, Austria",
              "Sie bringen mindestens 5 Jahren Berufserfahrung mit.") is not None
    ok_desc = ("You bring 2+ years of experience with SQL. You will build "
               "dashboards for stakeholders and work with our team on reporting "
               "and analysis of our KPIs across the business every day.")
    assert hf("Business Analyst", "Vienna, Austria", ok_desc) is None
    # French ad -> excluded; German ad -> kept
    fr = ("Nous recherchons un analyste pour rejoindre notre équipe. Vous avez "
          "une expérience avec les données et vous êtes motivé. Le poste est "
          "basé dans nos bureaux avec une entreprise dynamique. Vos missions: "
          "analyser les données pour aider nos équipes et nos clients.")
    de = ("Wir suchen einen Analysten für unser Team. Deine Aufgaben: Erstellung "
          "von Dashboards und Auswertungen mit Power BI sowie SQL. Du bringst "
          "erste Erfahrung in der Datenanalyse mit und arbeitest gerne mit "
          "unseren Stakeholdern zusammen. Wir bieten flexible Arbeitszeiten.")
    assert hf("Analyste de données", "Paris, France", fr) is not None
    assert hf("Datenanalyst (m/w/d)", "Berlin, Germany", de) is None


def test_required_years_parsing():
    assert matching.required_years("at least 5 years of experience") == 5
    assert matching.required_years("minimum of 6 years") == 6
    assert matching.required_years("3+ years") == 3
    assert matching.required_years("mindestens 5 Jahren Berufserfahrung") == 5
    assert matching.required_years("5-7 years of relevant experience") == 5
    assert matching.required_years("a 401k plan and 30+ days PTO") is None
    assert matching.required_years("") is None


def test_refilter_existing_cleans_stale_board_but_keeps_decisions():
    """When the rules tighten, undecided (New) recommendations that now fail the
    hard filters must vanish; Applied/Skipped/Rejected history must survive."""
    from datetime import date, timedelta

    from jobscout import pipeline
    _fresh_db()
    recent = (date.today() - timedelta(days=3)).isoformat()  # inside age ceiling
    cid = db.upsert_company("Acme")
    p1, _ = db.upsert_posting(cid, "W1", "Werkstudent BI", "Berlin, Germany",
                              recent, "u", "d")
    p2, _ = db.upsert_posting(cid, "A1", "Junior Data Analyst", "Vienna, Austria",
                              recent, "u", "d")
    p3, _ = db.upsert_posting(cid, "S1", "Senior Data Analyst", "Vienna, Austria",
                              recent, "u", "d")
    for pid in (p1, p2, p3):
        db.save_recommendation(pid, 70, 75, "r", "n", llm_used=False)
    # the user already applied to the senior one — that decision is history
    rec3 = [r for r in db.board_rows() if r["posting_id"] == p3][0]
    db.set_status(rec3["rec_id"], "Applied")
    removed = pipeline.refilter_existing()
    assert removed == 1, "only the undecided Werkstudent rec should be removed"
    rows = db.board_rows()
    titles = {r["title"] for r in rows}
    assert "Werkstudent BI" not in titles
    assert "Junior Data Analyst" in titles
    assert "Senior Data Analyst" in titles, "an Applied role must never be dropped"


def test_partner_and_age_are_not_seniority_signals():
    """Regressions from the 2026-07-03 drop audit: 'Business Partner' is a
    function (not a rank), and age requirements are not experience."""
    hf = matching.hard_filter
    assert hf("HR Business Partner", "Berlin, Germany") is None
    assert hf("Finance Business Partner", "Vienna, Austria") is None
    assert hf("Managing Partner", "Berlin, Germany") is not None
    assert matching.required_years("Sie sind mindestens 24 Jahre alt") is None
    assert matching.required_years("must be at least 21 years of age") is None
    assert matching.required_years("at least 5 years of experience") == 5


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")

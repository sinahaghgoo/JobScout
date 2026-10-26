"""Regression tests for the 2026-07-09 full-pipeline audit.

Each test pins a bug found while verifying the first complete production run
(36k postings, 110 companies):

  * "Barcelona, CT, Spain" (Catalonia) was dropped as Connecticut — 8 real
    target-region postings lost.
  * Moscow / Kyiv / Congo / Almaty locations passed as 'unknown' and wasted
    LLM slots (the loc gate caught them, but only after spending budget).
  * "..., Sr" at the end of a title dodged the senior-title filter.
  * "Software Development Manager" ranked pre=71 (no NEG_HARD hit).
  * Signal-free titles ("Data Center Technician") outranked real candidates
    on geography alone.
  * Browser-tier LLM extraction stored application deadlines / invalid dates
    ("2027-01-25", "2026-13-07") as posted_date — 66 rows.
  * The LLM "mgr" gate capped junior DACH IC roles at 40, below the 45
    surface line, hiding them forever.
"""
import os
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

os.environ["JOBSCOUT_DB"] = str(Path(tempfile.gettempdir()) / "jobscout_audit_pytest.db")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobscout import browser, db, matching, pipeline  # noqa: E402

TODAY = date.today().isoformat()


def _fresh_db():
    p = Path(os.environ["JOBSCOUT_DB"])
    if p.exists():
        p.unlink()
    db.init_db()


# =============================================================================
# location_status
# =============================================================================
def test_barcelona_catalonia_ct_is_target():
    # Catalonia's 'CT' must not be read as Connecticut.
    assert matching.location_status("Barcelona, CT, Spain") == "target"
    assert matching.location_status("Barcelona, Catalonia, ESP") == "target"
    assert matching.location_status("Barcelona, Cataluña, España") == "target"


def test_barcelona_without_spain_marker_stays_conservative():
    # Barcelona, Venezuela exists — without a Spain/Catalonia marker the
    # string must not ride the city name into the target set.
    assert matching.location_status("Barcelona, Venezuela") == "outside"


def test_us_state_codes_still_win_over_european_namesakes():
    assert matching.location_status("Vienna, VIRGINIA, United States") == "outside"
    assert matching.location_status("Cambridge, MA") == "outside"
    assert matching.location_status("New Berlin, Wisconsin, United States of America") == "outside"


def test_cis_and_row_locations_are_outside():
    for loc in ("Moscow (City)", "Kyiv", "Republic of the Congo",
                "Almaty, Kazakhstan", "Tashkent", "Baku, AZE", "Tunis",
                "Ho Chi Minh City", "Ciudad de Panama", "Tirana, Albania",
                "St. Petersburg, Russia", "Erbil", "Beirut, Lebanon"):
        assert matching.location_status(loc) == "outside", loc


def test_swiss_and_target_locations_unharmed_by_new_codes():
    # Geneva's canton code is GE, Bavaria's is BY — neither may be treated as
    # Georgia/Belarus. Bern must not fall to Belgium's BE either.
    assert matching.location_status("Geneva, GE, Switzerland") == "target"
    assert matching.location_status("Munich, BY, Germany") == "target"
    assert matching.location_status("Bern, Switzerland") == "target"
    assert matching.location_status("Gdańsk, Poland") == "outside"


# =============================================================================
# hard filters: senior titles + Spanish internships
# =============================================================================
def test_trailing_sr_is_senior():
    assert matching.hard_filter("Business Insights Specialist, Sr", "Reston", "") \
        == "senior-level title"
    assert matching.hard_filter("Sr. Data Analyst", "Berlin", "") == "senior-level title"
    # "Sri Lanka" / "SRE" style tokens must not fire the Sr rule; the SRE
    # variants are engineering roles for the prefilter, not senior titles.
    assert matching.hard_filter("Analyst, Sri Lanka Market", "London, UK", "") is None


def test_spanish_internship_title_excluded():
    assert matching.hard_filter("BECARIO CONTROL Y REPORTING", "Madrid", "") \
        == "internship/student/thesis role"
    assert matching.hard_filter("Becaria - Data Analytics", "Barcelona, Spain", "") \
        == "internship/student/thesis role"


# =============================================================================
# prefilter: engineering-management + signal-free titles
# =============================================================================
def test_software_development_manager_is_hard_negative():
    pre, sig = matching.prefilter("Software Development Manager, Lambda Experience",
                                  "Dublin, IRL", "")
    assert sig.get("title") == "off-target (engineering/DS)"
    assert pre < 38  # below the LLM cut


def test_signal_free_title_ranks_below_real_candidates():
    tech, sig_t = matching.prefilter("Data Center Technician", "Dublin, IRL", "")
    assert sig_t.get("no_signal") is True
    analyst, _ = matching.prefilter("Junior BI Analyst", "Dublin, IRL",
                                    "SQL and Power BI dashboards, reporting.")
    assert analyst > tech
    # and the penalty must NOT hit postings with any real signal
    _, sig_a = matching.prefilter("Reporting Specialist", "Nowhere", "")
    assert "no_signal" not in sig_a


# =============================================================================
# browser tier: extracted dates must be real and in the past
# =============================================================================
def test_valid_past_date_rejects_deadlines_and_invalid():
    assert browser._valid_past_date("2026-13-07") is None          # no month 13
    future = (date.today() + timedelta(days=30)).isoformat()
    assert browser._valid_past_date(future) is None                # deadline
    assert browser._valid_past_date("garbage") is None
    assert browser._valid_past_date(None) is None
    past = (date.today() - timedelta(days=5)).isoformat()
    assert browser._valid_past_date(past) == past
    assert browser._valid_past_date(TODAY) == TODAY                # today is fine


def test_to_rawjobs_strips_future_dates():
    future = (date.today() + timedelta(days=10)).isoformat()
    jobs = browser._to_rawjobs(
        [{"title": "Data Analyst", "location": "Vienna", "url": "",
          "posted_date": future, "description": ""}], "https://x.example")
    assert jobs[0].posted_date is None
    assert jobs[0].date_confidence == "low"


def test_db_migration_sanitizes_stored_bad_dates():
    _fresh_db()
    cid = db.upsert_company("BadDates Inc")
    db.upsert_posting(cid, "j1", "Role A", "Vienna", "2026-13-07", "u", "d")
    db.upsert_posting(cid, "j2", "Role B", "Vienna",
                      (date.today() + timedelta(days=200)).isoformat(), "u", "d")
    db.upsert_posting(cid, "j3", "Role C", "Vienna", TODAY, "u", "d")
    db.init_db()  # sanitation runs at every init
    with db.connect() as conn:
        rows = {r["job_id"]: r["posted_date"]
                for r in conn.execute("SELECT job_id, posted_date FROM posting")}
    assert rows["j1"] is None and rows["j2"] is None
    assert rows["j3"] == TODAY


# =============================================================================
# LLM gate: manager-in-title cap
# =============================================================================
def test_mgr_gate_caps_at_50_not_40():
    caps = matching._gate_caps({"sen": "mgr"})
    assert caps and caps[0][0] == 50
    # senior and y5 stay at 30
    assert matching._gate_caps({"sen": "senior"})[0][0] == 30
    assert matching._gate_caps({"sen": "y5"})[0][0] == 30


def test_rules_prompt_documents_junior_manager_exemption():
    assert "cap 50" in matching._RULES
    assert "junior/graduate-marked" in matching._RULES.lower() or \
        "(Junior)" in matching._RULES


# =============================================================================
# pipeline: relaxed location rules release wrongly-filtered postings
# =============================================================================
def test_refilter_releases_recovered_locations():
    _fresh_db()
    cid = db.upsert_company("ReleaseCo")
    pid, _ = db.upsert_posting(cid, "r1", "Junior BI Analyst",
                               "Barcelona, CT, Spain", TODAY, "u",
                               "SQL dashboards.")
    db.mark_posting_filtered(pid, "outside target countries")
    pid2, _ = db.upsert_posting(cid, "r2", "Junior BI Analyst",
                                "Warsaw, Poland", TODAY, "u", "SQL.")
    db.mark_posting_filtered(pid2, "outside target countries")
    pipeline.refilter_existing()
    with db.connect() as conn:
        r1 = conn.execute("SELECT filtered_reason FROM posting WHERE id=?",
                          (pid,)).fetchone()[0]
        r2 = conn.execute("SELECT filtered_reason FROM posting WHERE id=?",
                          (pid2,)).fetchone()[0]
    assert r1 is None                       # Barcelona released
    assert r2 == "outside target countries"  # Warsaw stays dropped


def test_mgr_requeue_migration_is_one_shot():
    _fresh_db()
    cid = db.upsert_company("RequeueCo")
    pid, _ = db.upsert_posting(cid, "m1", "(Junior) BI Manager", "Köln", TODAY,
                               "u", "Tableau dashboards.")
    db.save_recommendation(posting_id=pid, prefilter_score=90, fit_score=40.0,
                           rationale="Genuine junior BI role, but Manager title caps.",
                           watch_outs="Manager title for IC role caps score.",
                           llm_used=True)
    with db.connect() as conn:  # make the migration re-runnable in this test
        conn.execute("DELETE FROM meta WHERE key='mig_2026_07_09_mgr_requeue'")
    db.init_db()
    with db.connect() as conn:
        n = conn.execute("SELECT COUNT(*) FROM recommendation").fetchone()[0]
    assert n == 0  # rec deleted -> posting re-enters the scoring queue

    # second pass must NOT delete fresh 40-scores (meta key set)
    db.save_recommendation(posting_id=pid, prefilter_score=90, fit_score=40.0,
                           rationale="Manager title caps.", watch_outs="Manager title",
                           llm_used=True)
    db.init_db()
    with db.connect() as conn:
        n = conn.execute("SELECT COUNT(*) FROM recommendation").fetchone()[0]
    assert n == 1

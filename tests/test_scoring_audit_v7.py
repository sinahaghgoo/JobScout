"""Scoring-stage audit 2026-07-12 — offline regression suite (no API calls).

Locks in the audit fixes:
  1. Gate-cap parser is fail-open on junk, fail-closed only on RECOGNIZED
     failure values (a sloppy model can no longer nuke a perfect job by
     writing lad:"english" instead of "en").
  2. Deterministic caps respect the dashboard hard-filter configuration
     (a disabled rule must not keep capping through the back door) and the
     configured max_years parameter.
  3. Empty/near-empty descriptions cap at 75 with an explicit watch note.
  4. location_status understands mixed multi-city strings (foreign CITY next
     to a target city = sibling offices -> target; foreign REGION qualifying
     a city name -> outside).
  5. trim_description keeps the TAIL when no requirements header is found.
  6. Spelled-number years coverage (twenty / elf / zwanzig ...).
  7. Queue-janitor window is configurable (SCORE_QUEUE_MAX_DAYS).
"""
import json
import os
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

os.environ["JOBSCOUT_DB"] = str(Path(tempfile.gettempdir())
                                / "jobscout_audit_v7_pytest.db")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobscout import config, db, matching, pipeline, validate  # noqa: E402

TODAY = date.today().isoformat()
GOOD_DESC = ("You will build SQL queries and Power BI dashboards, own the "
             "reporting for commercial stakeholders and present insights.")


def _fresh_db():
    p = Path(os.environ["JOBSCOUT_DB"])
    if p.exists():
        p.unlink()
    db.init_db()
    matching.invalidate_filter_cache()


def _caps(gates):
    return matching._gate_caps(gates)


def _cap_values(gates):
    return sorted(c for c, _ in _caps(gates))


# =============================================================================
# 1. Gate-cap parser: fail-open on junk, cap on recognized failures
# =============================================================================
def test_junk_gate_values_never_cap():
    """Unrecognized model output must not destroy a good job's score."""
    junk = {"rt": "full-time", "sen": "mid", "loc": "target", "lad": "english",
            "lrq": "n/a", "fn": "analytics", "emp": "perm"}
    assert _caps(junk) == []
    assert _caps({"rt": "permanent employee", "sen": "0-4 years",
                  "lad": "de/en", "lrq": "not required", "fn": "core"}) == []
    assert _caps({"lad": "both", "lrq": "german and english"}) == []
    assert _caps("not-a-dict") == []


def test_recognized_failures_still_cap():
    assert (20, "role type: internship") in _caps({"rt": "internship"})
    assert (20, "role type: working student") in _caps({"rt": "working student"})
    assert (30, "seniority") in _caps({"sen": "y5"})
    assert (30, "seniority") in _caps({"sen": "senior"})
    assert (30, "seniority") in _caps({"sen": "y7"})
    assert (30, "seniority") in _caps({"sen": "senior title, 6+ years"})
    assert (50, "manager-level title") in _caps({"sen": "mgr"})
    assert (50, "manager-level title") in _caps({"sen": "manager"})
    assert (25, "outside target region") in _caps({"loc": "out"})
    assert (25, "outside target region") in _caps({"loc": "outside"})
    assert (25, "ad not in English/German") in _caps({"lad": "other"})
    assert (25, "ad not in English/German") in _caps({"lad": "fr"})
    assert (25, "ad not in English/German") in _caps({"lad": "french"})
    assert (25, "requires dutch") in _caps({"lrq": "dutch"})
    assert (25, "requires french") in _caps({"lrq": "fluent french"})
    assert (25, "requires dutch") in _caps({"lrq": "german + dutch"})


def test_negated_and_satisfiable_values_pass():
    assert _caps({"rt": "not an internship"}) == []
    assert _caps({"sen": "under 5 years"}) == []
    assert _caps({"lrq": "german or french"}) == []          # satisfiable
    assert _caps({"lrq": "english or spanish"}) == []        # satisfiable
    assert _caps({"lrq": "french is a plus"}) == []          # softened


def test_graduate_schemes_never_rt_capped():
    """The profile TARGETS graduate roles; a model calling a graduate scheme
    'trainee-like' must not nuke it (live-bench failure S3, 2026-07-12)."""
    assert _caps({"rt": "graduate programme"}) == []
    assert _caps({"rt": "graduate scheme (trainee-like)"}) == []
    assert (20, "role type: trainee") in _caps({"rt": "trainee"})


def test_fn_aliases_bind_caps():
    assert _cap_values({"fn": "engineering"}) == [35]
    assert _cap_values({"fn": "product management"}) == [40]
    assert _cap_values({"fn": "consulting"}) == [45]
    assert _cap_values({"fn": "finance"}) == [45]
    assert _cap_values({"fn": "trade"}) == [12]
    assert _cap_values({"fn": "trades"}) == [12]
    assert _cap_values({"fn": "governance"}) == [45]
    assert _cap_values({"fn": "data governance"}) == [45]
    # unknown / on-target labels never cap
    assert _caps({"fn": "bi"}) == []
    assert _caps({"fn": "adj"}) == []


# =============================================================================
# 2. Deterministic caps respect the dashboard filter configuration
# =============================================================================
def _det(job, **kw):
    return matching._deterministic_caps(job, matching.filter_config(refresh=True))


def test_det_caps_default_config():
    _fresh_db()
    job = {"title": "Senior Data Analyst", "location": "Vienna, Austria",
           "description": GOOD_DESC}
    caps = _det(job)
    assert (30, "senior-level title") in caps


def test_det_caps_respect_disabled_filter():
    """User disables the senior filter on the dashboard -> no back-door cap."""
    _fresh_db()
    db.save_hard_filter("senior_title", True, "Senior-level titles",
                        enabled=False, params_json="{}")
    matching.invalidate_filter_cache()
    job = {"title": "Senior Data Analyst", "location": "Vienna, Austria",
           "description": GOOD_DESC}
    assert all(why != "senior-level title" for _v, why in _det(job))
    # cleanup for other tests
    _fresh_db()


def test_det_caps_use_configured_max_years():
    _fresh_db()
    db.save_hard_filter("years_required", True, "Experience ceiling",
                        enabled=True, params_json=json.dumps({"max_years": 6}))
    matching.invalidate_filter_cache()
    base = {"title": "Data Analyst", "location": "Vienna, Austria"}
    seven = dict(base, description=GOOD_DESC + " At least 7 years of "
                                               "professional experience.")
    six = dict(base, description=GOOD_DESC + " At least 6 years of "
                                             "professional experience.")
    assert any("requires 7+" in why for _v, why in _det(seven))
    assert not any("requires" in why for _v, why in _det(six))
    _fresh_db()


def test_no_description_caps_at_75():
    _fresh_db()
    job = {"title": "Junior BI Analyst", "location": "Vienna, Austria",
           "description": ""}
    caps = _det(job)
    assert (matching.NO_DESC_CAP, "no description — judged from title only") in caps
    ok = {"title": "Junior BI Analyst", "location": "Vienna, Austria",
          "description": GOOD_DESC}
    assert all(v != matching.NO_DESC_CAP for v, _w in _det(ok))


def test_no_description_cap_applies_in_score_batch(monkeypatch):
    _fresh_db()
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(matching, "_llm_call", lambda *a, **k: json.dumps([{
        "id": 1, "g": {"rt": "ok", "sen": "ok", "loc": "ok", "lad": "en",
                       "lrq": "none", "fn": "core", "emp": "perm"},
        "s": 92, "why": "looks perfect", "watch": "None"}]))
    job = dict(id=1, title="Junior BI Analyst", company="Acme",
               location="Vienna, Austria", posted=TODAY, description="",
               pre_score=80.0, signals={})
    out = matching.score_batch("cv", "prefs", [job])
    assert out[1].fit_score == matching.NO_DESC_CAP
    assert "no description" in out[1].watch_outs


# =============================================================================
# 3. location_status: mixed multi-city strings
# =============================================================================
def test_multicity_foreign_city_beside_target_city_is_target():
    # Real Philips posting lost on the 2026-07 queue:
    assert matching.location_status(
        "Stockholm, Eindhoven, Amsterdam, Madrid, Farnborough") == "target"
    assert matching.location_status("Berlin, Warsaw") == "target"
    assert matching.location_status("Munich, Berlin, Warsaw, Vienna") == "target"


def test_region_qualified_city_stays_outside():
    assert matching.location_status("London, Ontario") == "outside"
    assert matching.location_status("Barcelona, Venezuela") == "outside"
    assert matching.location_status("Warsaw, Poland") == "outside"
    assert matching.location_status("Milan, Italy") == "outside"
    assert matching.location_status("Bangalore, India") == "outside"


def test_us_signals_still_win():
    assert matching.location_status("Vienna, VA") == "outside"
    assert matching.location_status("Cambridge, MA") == "outside"
    assert matching.location_status("Paris, New York") == "outside"
    assert matching.location_status("Londonderry, NH, United States") == "outside"


def test_retail_assistant_is_store_staff():
    _fresh_db()
    assert matching.hard_filter(
        "RETAIL ASSISTANT (M/F/D) BEFRISTET - FO MONTABAUR",
        "Montabaur, RP, DE") == "retail/store role"
    assert matching.hard_filter(
        "Retail Assistant (m/f/d) 28-35h/Woche", "Berlin, BE, DE") \
        == "retail/store role"
    # corporate roles about the retail BUSINESS still survive
    assert matching.hard_filter("Retail Media Analyst", "Berlin, DE",
                                GOOD_DESC) is None


def test_plain_cases_unchanged():
    assert matching.location_status("Madrid") == "outside"
    assert matching.location_status("Toronto") == "outside"
    assert matching.location_status("Wien, Österreich") == "target"
    assert matching.location_status("Barcelona, Spain") == "target"
    assert matching.location_status("Barcelona, CT, Spain") == "target"
    assert matching.location_status("") == "unknown"
    assert matching.location_status("Hybrid") == "unknown"


# =============================================================================
# 4. trim_description keeps the tail when no requirements header exists
# =============================================================================
def test_trim_keeps_tail_without_req_header():
    filler = "We are a purpose-driven global organisation. " * 200
    tail = "Must hold a certification in underwater basket weaving."
    d = filler + tail
    out = matching.trim_description(d, budget=1000)
    assert tail[-30:] in out
    assert "…[…]…" in out


def test_trim_still_prefers_req_section():
    d = ("Intro text about the company. " * 100
         + "Requirements: 3 years of experience with SQL and Power BI. "
         + "Closing boilerplate. " * 50)
    out = matching.trim_description(d, budget=1200)
    assert "Requirements: 3 years" in out


# =============================================================================
# 5. Spelled-number years
# =============================================================================
def test_spelled_years_extended():
    assert matching.required_years(
        "You bring twenty years of leadership experience.") == 20
    assert matching.required_years(
        "Sie bringen elf Jahre Berufserfahrung mit.") == 11
    assert matching.required_years(
        "fourteen years of relevant experience required") == 14


# =============================================================================
# 6. Configurable queue janitor
# =============================================================================
def test_janitor_window_is_configurable(monkeypatch):
    _fresh_db()
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(matching, "llm_available", lambda: True)
    monkeypatch.setattr(matching, "score_batch", lambda *a, **k: {})
    monkeypatch.setattr(config, "SCORE_QUEUE_MAX_DAYS", 3)
    cid = db.upsert_company("JanitorCo")
    fresh, _ = db.upsert_posting(cid, "f1", "Business Analyst",
                                 "Vienna, Austria", TODAY, "u", GOOD_DESC)
    stale, _ = db.upsert_posting(cid, "s1", "Data Analyst", "Vienna, Austria",
                                 TODAY, "u", GOOD_DESC)
    old = (datetime.now(timezone.utc) - timedelta(days=4)).isoformat(
        timespec="seconds")
    with db.connect() as conn:
        conn.execute("UPDATE posting SET first_seen=? WHERE id=?", (old, stale))
    pipeline._match_new(pipeline.RunSummary(), verbose=False)
    with db.connect() as conn:
        f = conn.execute("SELECT filtered_reason FROM posting WHERE id=?",
                         (fresh,)).fetchone()[0]
        s = conn.execute("SELECT filtered_reason FROM posting WHERE id=?",
                         (stale,)).fetchone()[0]
    assert f is None
    assert s == "not scored within 3 days (LLM budget/errors)"
    assert matching.filter_key_for_reason(s) is None   # janitor ≠ hard filter


# =============================================================================
# 7. Prompt / validator contracts (regression tripwires)
# =============================================================================
def test_prompt_contains_edge_rules():
    for needle in ("EDGE RULES", "no description", "requirements text beats",
                   "never above 75"):
        assert needle in matching._RULES, needle


def test_validator_keeps_borderline_rows_rule():
    assert "BORDERLINE" in validate._REVIEW_RULES
    assert "45-55" in validate._REVIEW_RULES


def test_prompt_examples_parse_as_json():
    """The few-shot examples the model imitates must themselves be valid."""
    import re as _re
    for m in _re.finditer(r"OUT: (\{.*?\})\nIN|OUT: (\{.*?\})\n\nOUTPUT",
                          matching._RULES, _re.S):
        raw = (m.group(1) or m.group(2)).replace("\\\n", "")
        obj = json.loads(raw)
        assert {"id", "g", "s", "why", "watch"} <= set(obj)
        assert isinstance(obj["s"], (int, float))
        assert matching._gate_caps(obj["g"]) is not None

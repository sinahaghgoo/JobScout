"""Hard-filter management (dashboard) + run history.

Covers:
  * defaults: with no stored overrides, hard_filter() behaves exactly as the
    hard-coded legacy pipeline (all 10 built-ins on, config-driven params)
  * built-in toggles via the API actually change filtering and persist
  * built-in parameter edits (age ceiling, years ceiling) take effect
  * extra terms extend a built-in pattern (word-boundary, additive only)
  * custom filters: create / edit / rename / disable / delete, word vs
    contains semantics, reason strings, real effect in the pipeline's
    filter stage
  * validation: duplicate names, empty terms, bad fields/ranges, built-in
    delete -> clean 400s and nothing persisted
  * filter_key_for_reason: hard-filter reasons map back; pipeline verdicts
    (duplicates, LLM floor, final review, queue aging) map to None
  * rev-gated re-judging: refilter_existing releases/relabels excluded
    postings exactly once after a config change; verdict rows untouched
  * stats: /api/filters per-filter counts + other-exclusion grouping
  * run history: record_run + /api/runs, seeding from meta, _finish()
"""
import json
import os
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

os.environ["JOBSCOUT_DB"] = str(Path(tempfile.gettempdir()) / "jobscout_hf_pytest.db")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from jobscout import config, db, matching, pipeline, web  # noqa: E402

client = TestClient(web.app)
TODAY = date.today().isoformat()
DESC = ("You will build SQL queries and Power BI dashboards, own the "
        "reporting for commercial stakeholders and present insights.")


def _fresh_db():
    p = Path(os.environ["JOBSCOUT_DB"])
    if p.exists():
        p.unlink()
    db.init_db()
    matching.invalidate_filter_cache()


@pytest.fixture(autouse=True)
def _clean_state():
    _fresh_db()
    yield
    matching.invalidate_filter_cache()


_job_seq = 0


def _seed_posting(title, location="Berlin", desc=DESC, posted=TODAY,
                  company="TestCo GmbH"):
    global _job_seq
    _job_seq += 1
    cid = db.upsert_company(company)
    db.set_company_ats(cid, "greenhouse", "testco", "detected", "test")
    pid, _new = db.upsert_posting(
        company_id=cid, job_id=f"job-{_job_seq}", title=title,
        location=location, posted_date=posted, url="https://x.example/j",
        description=desc)
    return pid


def _reason_of(pid) -> "str | None":
    with db.connect() as conn:
        return conn.execute("SELECT filtered_reason FROM posting WHERE id=?",
                            (pid,)).fetchone()[0]


# --- defaults --------------------------------------------------------------------
def test_defaults_match_legacy_behavior():
    assert matching.hard_filter("Senior Data Analyst", "Berlin") == "senior-level title"
    assert matching.hard_filter("Data Analyst Intern", "Berlin") == "internship/student/thesis role"
    assert matching.hard_filter("Data Analyst", "New York") == "outside target countries"
    assert matching.hard_filter("Kfz-Mechatroniker", "München") == "off-profile occupation"
    assert matching.hard_filter("Data Analyst", "Berlin", DESC) is None
    r = client.get("/api/filters").json()
    assert len(r["filters"]) == len(matching.BUILTIN_HARD_FILTERS) == 10
    assert all(f["enabled"] for f in r["filters"])
    assert r["stats"]["active"] == 10
    assert r["stats"]["inactive"] == 0
    assert r["stats"]["custom"] == 0
    assert r["changed_since_run"] is False
    # effective params surface the config defaults
    age = next(f for f in r["filters"] if f["key"] == "age")
    assert age["params"]["max_days"] == config.MAX_POSTING_AGE_DAYS


def test_trainee_default_follows_config_until_overridden(monkeypatch):
    assert matching.hard_filter("Trainee Data Analytics", "Berlin") == "trainee programme"
    monkeypatch.setattr(config, "INCLUDE_TRAINEE", True)
    matching.invalidate_filter_cache()
    assert matching.hard_filter("Trainee Data Analytics", "Berlin") is None
    # an explicit dashboard override beats the env default
    assert client.post("/api/filters/trainee", json={"enabled": True}).status_code == 200
    assert matching.hard_filter("Trainee Data Analytics", "Berlin") == "trainee programme"


# --- built-in toggles + params ----------------------------------------------------
def test_builtin_toggle_changes_filtering_and_persists():
    assert client.post("/api/filters/senior_title",
                       json={"enabled": False}).status_code == 200
    assert matching.hard_filter("Senior Data Analyst", "Berlin") is None
    row = db.get_hard_filter("senior_title")           # fresh connection
    assert row is not None and row["enabled"] == 0
    r = client.get("/api/filters").json()
    senior = next(f for f in r["filters"] if f["key"] == "senior_title")
    assert senior["enabled"] is False
    assert r["stats"]["inactive"] == 1 and r["stats"]["active"] == 9
    assert r["changed_since_run"] is True
    client.post("/api/filters/senior_title", json={"enabled": True})
    assert matching.hard_filter("Senior Data Analyst", "Berlin") == "senior-level title"


def test_builtin_param_edits_take_effect():
    old40 = (date.today() - timedelta(days=40)).isoformat()
    assert matching.hard_filter("Data Analyst", "Berlin", posted_date=old40) \
        == f"older than {config.MAX_POSTING_AGE_DAYS} days"
    assert client.post("/api/filters/age", json={"max_days": 60}).status_code == 200
    assert matching.hard_filter("Data Analyst", "Berlin", posted_date=old40) is None
    assert client.post("/api/filters/age", json={"max_days": 10}).status_code == 200
    old15 = (date.today() - timedelta(days=15)).isoformat()
    assert matching.hard_filter("Data Analyst", "Berlin", posted_date=old15) \
        == "older than 10 days"

    desc = DESC + " You bring 3 years of professional experience in analytics."
    assert matching.hard_filter("Data Analyst", "Berlin", desc) is None
    assert client.post("/api/filters/years_required",
                       json={"max_years": 2}).status_code == 200
    assert matching.hard_filter("Data Analyst", "Berlin", desc) \
        == "requires 3+ years' experience"


def test_builtin_extra_terms_are_additive_word_matches():
    assert matching.hard_filter("Bereichsvorstandsassistenz Analytics", "Berlin") is None
    assert client.post("/api/filters/senior_title",
                       json={"extra_terms": ["Bereichsvorstandsassistenz"]}).status_code == 200
    assert matching.hard_filter("Bereichsvorstandsassistenz Analytics", "Berlin") \
        == "senior-level title"
    # word-boundary: a longer word containing the term must NOT hit
    assert matching.hard_filter("XBereichsvorstandsassistenzY Analytics", "Berlin") is None
    # the built-in pattern is untouched
    assert matching.hard_filter("Senior Data Analyst", "Berlin") == "senior-level title"
    # clearing extra terms restores the original behavior
    assert client.post("/api/filters/senior_title",
                       json={"extra_terms": []}).status_code == 200
    assert matching.hard_filter("Bereichsvorstandsassistenz Analytics", "Berlin") is None


# --- custom filters ----------------------------------------------------------------
def test_custom_filter_lifecycle_word_and_contains():
    r = client.post("/api/filters", json={"name": "No consulting",
                                          "terms": ["consultant", "consulting"],
                                          "field": "title", "match": "word"})
    assert r.status_code == 200
    key = r.json()["key"]
    assert matching.hard_filter("Strategy Consultant", "Berlin") \
        == "custom filter: No consulting"
    assert matching.hard_filter("Consultingservices Analyst", "Berlin") is None  # word mode
    # switch to contains
    assert client.post(f"/api/filters/{key}", json={"match": "contains"}).status_code == 200
    assert matching.hard_filter("Consultingservices Analyst", "Berlin") \
        == "custom filter: No consulting"
    # disable -> passes again
    assert client.post(f"/api/filters/{key}", json={"enabled": False}).status_code == 200
    assert matching.hard_filter("Strategy Consultant", "Berlin") is None
    # re-enable + delete -> gone for good
    client.post(f"/api/filters/{key}", json={"enabled": True})
    assert client.delete(f"/api/filters/{key}").status_code == 200
    assert matching.hard_filter("Strategy Consultant", "Berlin") is None
    assert db.get_hard_filter(key) is None


def test_custom_filter_fields_location_and_description():
    client.post("/api/filters", json={"name": "No Ruhrpott",
                                      "terms": ["Gelsenkirchen"], "field": "location"})
    assert matching.hard_filter("Data Analyst", "Gelsenkirchen, Germany") \
        == "custom filter: No Ruhrpott"
    assert matching.hard_filter("Data Analyst Gelsenkirchen", "Berlin") is None
    client.post("/api/filters", json={"name": "No clearance",
                                      "terms": ["security clearance"],
                                      "field": "description"})
    assert matching.hard_filter("Data Analyst", "Berlin",
                                DESC + " Requires security clearance.") \
        == "custom filter: No clearance"


def test_custom_filter_applies_in_pipeline_filter_stage():
    pid = _seed_posting("Management Consultant")
    client.post("/api/filters", json={"name": "No consulting", "terms": ["consultant"]})
    summary = pipeline.RunSummary()
    eligible = pipeline._filter_new_postings(summary, verbose=False)
    assert all(p["id"] != pid for p in eligible)
    assert _reason_of(pid) == "custom filter: No consulting"
    assert summary.filtered_out >= 1


def test_custom_filter_description_persists_and_updates():
    r = client.post("/api/filters", json={"name": "X", "terms": ["foo"],
                                          "description": "My reason"})
    key = r.json()["key"]

    def what():
        return next(x for x in client.get("/api/filters").json()["filters"]
                    if x["key"] == key)["what"]
    assert what() == "My reason"
    client.post(f"/api/filters/{key}", json={"description": "Updated reason"})
    assert what() == "Updated reason"
    client.post(f"/api/filters/{key}", json={"description": ""})   # clearable
    assert what() == ""
    assert client.post(f"/api/filters/{key}",
                       json={"description": "x" * 201}).status_code == 400


def test_builtin_filters_expose_rule_details():
    d = client.get("/api/filters").json()
    senior = next(f for f in d["filters"] if f["key"] == "senior_title")
    assert senior["preview"], "table needs term chips instead of 'built-in pattern'"
    terms = [t for g in senior["details"] for t in g["terms"]]
    assert "senior" in terms and "head of" in terms
    langs = next(f for f in d["filters"] if f["key"] == "other_languages")
    lang_terms = [t for g in langs["details"] for t in g["terms"]]
    assert "French" in lang_terms          # derived from the live language map
    early = next(f for f in d["filters"] if f["key"] == "early_career")
    early_terms = [t for g in early["details"] for t in g["terms"]]
    assert "werkstudent" in early_terms and "thesis" in early_terms


def test_custom_filter_end_to_end_pipeline_run(monkeypatch):
    """Full pipeline.run(): a custom filter created through the API excludes
    its posting before scoring (the LLM never sees it); disabling the filter
    releases the posting on the next run, where it IS scored and boards."""
    from jobscout import validate
    from jobscout.connectors.base import RawJob
    db.save_profile("MSc data science, BI focus", "Business/Data Analyst roles")
    cid = db.upsert_company("E2ECo")
    db.set_company_ats(cid, "greenhouse", "e2eco", "detected")
    board = [
        RawJob("j-ok", "Junior BI Analyst", "Vienna, Austria", TODAY, "http://x/1",
               "Build Power BI dashboards, SQL, KPI reporting."),
        RawJob("j-cons", "Strategy Consultant", "Vienna, Austria", TODAY, "http://x/2",
               "Consulting engagements, client workshops, stakeholder decks."),
    ]
    monkeypatch.setattr(pipeline, "_fetch_one", lambda c: list(board))
    monkeypatch.setattr(matching, "llm_available", lambda: True)
    scored_titles = []

    def fake_score_batch(cv, prefs, jobs, tracker=None, extra_rules=""):
        out = {}
        for j in jobs:
            scored_titles.append(j["title"])
            out[j["id"]] = matching.MatchResult(
                prefilter_score=j["pre_score"], fit_score=77.0,
                rationale="stub", watch_outs="None", llm_used=True)
        return out
    monkeypatch.setattr(matching, "score_batch", fake_score_batch)
    monkeypatch.setattr(validate, "_review_batch",
                        lambda system_blocks, batch, tracker: {
                            r["rec_id"]: {"v": "keep", "why": "", "conf": "hi"}
                            for r in batch})

    r = client.post("/api/filters", json={
        "name": "No consulting", "description": "Skip consulting gigs",
        "terms": ["consultant", "consulting"], "field": "title"})
    assert r.status_code == 200
    key = r.json()["key"]

    pipeline.run(verbose=False)
    assert [x["title"] for x in db.board_rows(statuses=["New"])] == ["Junior BI Analyst"]
    assert "Strategy Consultant" not in scored_titles     # never reached the LLM
    with db.connect() as conn:
        reason = conn.execute("SELECT filtered_reason FROM posting "
                              "WHERE job_id='j-cons'").fetchone()[0]
    assert reason == "custom filter: No consulting"

    client.post(f"/api/filters/{key}", json={"enabled": False})
    pipeline.run(verbose=False)                            # rev-gate releases it
    titles = sorted(x["title"] for x in db.board_rows(statuses=["New"]))
    assert titles == ["Junior BI Analyst", "Strategy Consultant"]
    assert "Strategy Consultant" in scored_titles


def test_custom_filter_rename_updates_stored_reasons():
    r = client.post("/api/filters", json={"name": "Old name", "terms": ["consultant"]})
    key = r.json()["key"]
    pid = _seed_posting("Junior Consultant")
    db.mark_posting_filtered(pid, "custom filter: Old name")
    assert client.post(f"/api/filters/{key}", json={"name": "New name"}).status_code == 200
    assert _reason_of(pid) == "custom filter: New name"
    d = client.get("/api/filters").json()
    f = next(f for f in d["filters"] if f["key"] == key)
    assert f["name"] == "New name" and f["excluded_now"] == 1


# --- categories ---------------------------------------------------------------------
VEHICLES = "Vehicles, workshop & manufacturing trades"


def test_builtin_group_terms_add_change_remove():
    """Add / change / clear user terms INSIDE one category of a built-in
    filter — the user's 'add one for Vehicles, workshop & manufacturing
    trades' case."""
    assert matching.hard_filter("Fahrzeugaufbereiter", "Berlin") is None
    r = client.post("/api/filters/off_profile",
                    json={"extra_groups": {VEHICLES: ["Fahrzeugaufbereiter"]}})
    assert r.status_code == 200
    assert matching.hard_filter("Fahrzeugaufbereiter (m/w/d)", "Berlin") \
        == "off-profile occupation"
    # word-boundary + persisted per category
    assert matching.hard_filter("XFahrzeugaufbereiterY", "Berlin") is None
    f = next(x for x in client.get("/api/filters").json()["filters"]
             if x["key"] == "off_profile")
    assert f["extra_groups"] == {VEHICLES: ["Fahrzeugaufbereiter"]}
    # change the term
    client.post("/api/filters/off_profile",
                json={"extra_groups": {VEHICLES: ["Hufschmied"]}})
    assert matching.hard_filter("Fahrzeugaufbereiter", "Berlin") is None
    assert matching.hard_filter("Hufschmied", "Berlin") == "off-profile occupation"
    # clear the category -> pure built-in behavior again
    client.post("/api/filters/off_profile", json={"extra_groups": {}})
    assert matching.hard_filter("Hufschmied", "Berlin") is None
    f = next(x for x in client.get("/api/filters").json()["filters"]
             if x["key"] == "off_profile")
    assert f["extra_groups"] == {}


def test_flat_and_grouped_extra_terms_coexist():
    client.post("/api/filters/senior_title", json={"extra_terms": ["Obermeister"]})
    client.post("/api/filters/senior_title",
                json={"extra_groups": {"Rank keywords": ["Erzmagier"]}})
    assert matching.hard_filter("Obermeister Analytics", "Berlin") == "senior-level title"
    assert matching.hard_filter("Erzmagier Analytics", "Berlin") == "senior-level title"
    f = next(x for x in client.get("/api/filters").json()["filters"]
             if x["key"] == "senior_title")
    assert f["extra_terms"] == ["Obermeister"]
    assert f["extra_groups"] == {"Rank keywords": ["Erzmagier"]}
    # built-in words still untouched
    assert matching.hard_filter("Senior Data Analyst", "Berlin") == "senior-level title"


def test_group_terms_apply_in_pipeline_and_release_on_clear():
    pid = _seed_posting("Kaminkehrer")
    client.post("/api/filters/off_profile",
                json={"extra_groups": {"Construction, installation & crafts":
                                       ["Kaminkehrer"]}})
    pipeline._filter_new_postings(pipeline.RunSummary(), verbose=False)
    assert _reason_of(pid) == "off-profile occupation"
    client.post("/api/filters/off_profile", json={"extra_groups": {}})
    pipeline.refilter_existing(verbose=False)          # rev-gate re-judges
    assert _reason_of(pid) is None


def test_extra_groups_validation():
    # filters without extra-term support reject groups
    assert client.post("/api/filters/language",
                       json={"extra_groups": {"X": ["y"]}}).status_code == 400
    assert client.post("/api/filters/years_required",
                       json={"extra_groups": {"X": ["y"]}}).status_code == 400
    # too many categories / oversized labels reject cleanly
    many = {f"cat {i}": ["term"] for i in range(21)}
    assert client.post("/api/filters/off_profile",
                       json={"extra_groups": many}).status_code == 400
    assert client.post("/api/filters/off_profile",
                       json={"extra_groups": {"x" * 81: ["y"]}}).status_code == 400
    # nothing persisted by the failed writes
    f = next(x for x in client.get("/api/filters").json()["filters"]
             if x["key"] == "off_profile")
    assert f["extra_groups"] == {}


def test_addable_flags_protect_informational_groups():
    d = client.get("/api/filters").json()
    loc = next(f for f in d["filters"] if f["key"] == "location")
    flags = {g["label"]: g["addable"] for g in loc["details"]}
    assert flags["Dropped — clearly elsewhere (curated lists)"] is True
    assert flags["Kept — target region"] is False       # adding here would invert meaning
    assert flags["Kept for the LLM to judge"] is False
    occ = next(f for f in d["filters"] if f["key"] == "off_profile")
    occ_flags = {g["label"]: g["addable"] for g in occ["details"]}
    assert occ_flags[VEHICLES] is True
    assert occ_flags["Guard rail"] is False
    senior = next(f for f in d["filters"] if f["key"] == "senior_title")
    s_flags = {g["label"]: g["addable"] for g in senior["details"]}
    assert s_flags["Judged by the LLM instead"] is False


def test_custom_filter_category_assign_change_clear_and_sort():
    r1 = client.post("/api/filters", json={"name": "No forklift certs",
                                           "terms": ["forklift certificate"],
                                           "category": VEHICLES})
    assert r1.status_code == 200
    k1 = r1.json()["key"]
    k2 = client.post("/api/filters", json={"name": "Alpha rule", "terms": ["alphaterm"],
                                           "category": "Aviation"}).json()["key"]
    k3 = client.post("/api/filters", json={"name": "Uncat rule",
                                           "terms": ["uncatterm"]}).json()["key"]

    def customs():
        return [f for f in client.get("/api/filters").json()["filters"]
                if not f["builtin"]]
    by_key = {f["key"]: f for f in customs()}
    assert by_key[k1]["category"] == VEHICLES
    assert by_key[k3]["category"] == ""
    # categorized A→Z first, uncategorized last
    assert [f["key"] for f in customs()] == [k2, k1, k3]
    # change + clear
    client.post(f"/api/filters/{k1}", json={"category": "Own bucket"})
    assert {f["key"]: f for f in customs()}[k1]["category"] == "Own bucket"
    client.post(f"/api/filters/{k1}", json={"category": ""})
    assert {f["key"]: f for f in customs()}[k1]["category"] == ""
    # validation + category survives an unrelated edit
    assert client.post(f"/api/filters/{k2}",
                       json={"category": "x" * 61}).status_code == 400
    client.post(f"/api/filters/{k2}", json={"terms": ["betaterm"]})
    assert {f["key"]: f for f in customs()}[k2]["category"] == "Aviation"
    # the category is organizational only — filtering still works
    matching.invalidate_filter_cache()
    assert matching.hard_filter("Betaterm Analyst", "Berlin") == "custom filter: Alpha rule"


# --- validation --------------------------------------------------------------------
def test_validation_rejects_bad_configs_cleanly():
    ok = client.post("/api/filters", json={"name": "No consulting", "terms": ["x"]})
    assert ok.status_code == 200
    # duplicate names (case-insensitive), including built-in names
    assert client.post("/api/filters",
                       json={"name": "no CONSULTING", "terms": ["y"]}).status_code == 400
    assert client.post("/api/filters",
                       json={"name": "Posting age limit", "terms": ["y"]}).status_code == 400
    # empty name / empty terms / bad field / bad match
    assert client.post("/api/filters", json={"name": "  ", "terms": ["y"]}).status_code == 400
    assert client.post("/api/filters", json={"name": "E", "terms": ["  "]}).status_code == 400
    assert client.post("/api/filters",
                       json={"name": "F", "terms": ["y"], "field": "company"}).status_code == 400
    assert client.post("/api/filters",
                       json={"name": "G", "terms": ["y"], "match": "regex"}).status_code == 400
    # out-of-range params, wrong param for the filter
    assert client.post("/api/filters/age", json={"max_days": 0}).status_code == 400
    assert client.post("/api/filters/age", json={"max_days": 999}).status_code == 400
    assert client.post("/api/filters/age", json={"max_years": 3}).status_code == 400
    assert client.post("/api/filters/language",
                       json={"extra_terms": ["x"]}).status_code == 400
    # built-ins cannot be deleted; unknown keys 404
    assert client.delete("/api/filters/senior_title").status_code == 400
    assert client.delete("/api/filters/custom-doesnotexist").status_code == 404
    assert client.post("/api/filters/nope", json={"enabled": False}).status_code == 404
    # exactly one custom filter exists; nothing invalid persisted
    d = client.get("/api/filters").json()
    assert d["stats"]["custom"] == 1
    age = next(f for f in d["filters"] if f["key"] == "age")
    assert age["params"]["max_days"] == config.MAX_POSTING_AGE_DAYS


# --- reason mapping ------------------------------------------------------------------
def test_filter_key_for_reason_mapping():
    m = matching.filter_key_for_reason
    assert m("older than 28 days") == "age"
    assert m("internship/student/thesis role") == "early_career"
    assert m("trainee programme") == "trainee"
    assert m("retail/store role") == "retail_store"
    assert m("off-profile occupation") == "off_profile"
    assert m("senior-level title") == "senior_title"
    assert m("outside target countries") == "location"
    assert m("requires 7+ years' experience") == "years_required"
    assert m("requires french, spanish") == "other_languages"
    assert m("not in English or German") == "language"
    assert m("custom filter: No consulting") == "custom:No consulting"
    # pipeline verdicts are NOT hard filters
    assert m("duplicate posting (same as #12)") is None
    assert m("scored 12/100 by the LLM — thin overlap") is None
    assert m("failed final review: senior scope") is None
    assert m("not scored within 10 days (LLM budget/errors)") is None
    assert m(None) is None and m("") is None


# --- rev-gated re-judging -------------------------------------------------------------
def test_config_change_rejudges_excluded_postings_once():
    pid = _seed_posting("Senior BI Analyst")
    pid2 = _seed_posting("Data Analyst")
    pipeline._filter_new_postings(pipeline.RunSummary(), verbose=False)
    assert _reason_of(pid) == "senior-level title"
    assert _reason_of(pid2) is None                     # eligible, queued
    db.mark_posting_filtered(pid2, "duplicate posting (same as #1)")

    client.post("/api/filters/senior_title", json={"enabled": False})
    assert client.get("/api/filters").json()["changed_since_run"] is True
    pipeline.refilter_existing(verbose=False)
    assert _reason_of(pid) is None                      # released to the queue
    assert _reason_of(pid2) == "duplicate posting (same as #1)"  # untouched
    assert client.get("/api/filters").json()["changed_since_run"] is False
    assert db.meta_get("hard_filters_rev") == db.meta_get("hard_filters_rev_applied")


def test_config_change_relabels_with_current_rules():
    old40 = (date.today() - timedelta(days=40)).isoformat()
    pid = _seed_posting("Senior BI Analyst", posted=old40)
    pipeline._filter_new_postings(pipeline.RunSummary(), verbose=False)
    assert _reason_of(pid) == f"older than {config.MAX_POSTING_AGE_DAYS} days"
    client.post("/api/filters/age", json={"max_days": 60})
    pipeline.refilter_existing(verbose=False)
    assert _reason_of(pid) == "senior-level title"      # next rule takes over


# --- stats ---------------------------------------------------------------------------
def test_filter_stats_breakdown_and_other_exclusions():
    client.post("/api/filters", json={"name": "No consulting", "terms": ["consultant"]})
    p1 = _seed_posting("Senior Analyst A")
    p2 = _seed_posting("Senior Analyst B")
    p3 = _seed_posting("Junior Consultant")
    p4 = _seed_posting("Data Analyst C")
    db.mark_posting_filtered(p1, "senior-level title")
    db.mark_posting_filtered(p2, "senior-level title")
    db.mark_posting_filtered(p3, "custom filter: No consulting")
    db.mark_posting_filtered(p4, "duplicate posting (same as #1)")
    d = client.get("/api/filters").json()
    senior = next(f for f in d["filters"] if f["key"] == "senior_title")
    custom = next(f for f in d["filters"] if f["name"] == "No consulting")
    assert senior["excluded_now"] == 2
    assert custom["excluded_now"] == 1
    assert d["stats"]["excluded_total"] == 3            # verdicts not counted
    other = {o["label"]: o["count"] for o in d["stats"]["other_exclusions"]}
    assert other.get("Duplicate postings") == 1


# --- run history -----------------------------------------------------------------------
def test_run_history_recording_and_api_order():
    db.record_run("2026-07-11T10:00:00+00:00", json.dumps({"postings_seen": 7, "notes": []}))
    db.record_run("2026-07-11T11:00:00+00:00", json.dumps({"postings_seen": 9, "notes": []}))
    r = client.get("/api/runs").json()
    assert r["count"] == 2
    assert [x["summary"]["postings_seen"] for x in r["runs"]] == [9, 7]  # newest first


def test_finish_records_history_and_stats_count():
    s = pipeline.RunSummary()
    s.postings_seen = 42
    pipeline._finish(s)
    r = client.get("/api/runs").json()
    assert r["count"] == 1 and r["runs"][0]["summary"]["postings_seen"] == 42
    assert client.get("/api/stats").json()["runs_recorded"] == 1


def test_run_history_seeded_from_last_run_meta():
    db.meta_set("last_run_at", "2026-07-10T09:00:00+00:00")
    db.meta_set("last_run_summary", json.dumps({"postings_seen": 3, "notes": []}))
    with db.connect() as conn:                 # simulate pre-upgrade state
        conn.execute("DELETE FROM run_history")
    db.init_db()                               # migration seeds one row
    r = client.get("/api/runs").json()
    assert r["count"] == 1
    assert r["runs"][0]["summary"]["postings_seen"] == 3
    db.init_db()                               # idempotent — no duplicate
    assert client.get("/api/runs").json()["count"] == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

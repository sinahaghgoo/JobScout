"""Tests for the production matching stack: language detection, the UN
connector, SuccessFactors detail enrichment, Workday location fallback,
SmartRecruiters URL synthesis, and the Tier-3 extraction cache."""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["JOBSCOUT_DB"] = str(Path(tempfile.gettempdir()) / "jobscout_pytest_v2.db")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobscout import db  # noqa: E402
from jobscout.lang import detect_language, is_supported_language  # noqa: E402


def _fresh_db():
    p = Path(os.environ["JOBSCOUT_DB"])
    if p.exists():
        p.unlink()
    db.init_db()


# --- language detection -------------------------------------------------------
def test_language_detection_en_de_pass_others_fail():
    de = ("Wir suchen dich! Deine Aufgaben: Erstellung von Dashboards und "
          "Auswertungen mit Power BI. Dein Profil: Du hast Erfahrung mit SQL "
          "und arbeitest gerne im Team. Wir bieten flexible Arbeitszeiten "
          "sowie ein tolles Umfeld für deine Entwicklung.")
    en = ("We are looking for an analyst to join our team. You will work with "
          "stakeholders to turn data into decisions, build dashboards and "
          "reports. Experience with SQL and Power BI is a plus for this role.")
    fr = ("Rejoignez notre équipe pour analyser les données clients. Vous êtes "
          "diplômé et vous avez une expérience dans les études de marché. Nous "
          "vous proposons un poste au sein de notre entreprise à Paris.")
    es = ("Buscamos un analista de datos para nuestro equipo. Experiencia con "
          "SQL y herramientas de visualización. Ofrecemos trabajo híbrido para "
          "que puedas desarrollar tu carrera con nosotros en una empresa líder.")
    nl = ("Voor ons team zoeken wij een analist. Je hebt ervaring met SQL en je "
          "werkt graag binnen een team aan dashboards. Wij bieden jou een "
          "uitdagende functie met veel ruimte voor jouw ontwikkeling.")
    assert detect_language(de) == "de"
    assert detect_language(en) == "en"
    assert detect_language(fr) == "other"
    assert detect_language(es) == "other"
    assert detect_language(nl) == "other"
    # short titles alone stay permissive
    assert is_supported_language("Junior BI Analyst (f/m/x)", "")
    # description dominates a misleading English title
    assert not is_supported_language("Data Analyst", fr)
    assert is_supported_language("Data Analyst", de)


# --- UN connector --------------------------------------------------------------
def test_un_connector_filters_and_maps(monkeypatch):
    from jobscout.connectors import un as un_mod
    board = [
        {"jobId": 1, "postingTitle": "STATISTICS ASSISTANT, G5", "jobLevel": "G-5",
         "dutyStation": [{"description": "VIENNA"}],
         "dept": {"name": "United Nations Office on Drugs and Crime"},
         "startDate": "2026-06-30T04:00:00.000Z", "endDate": "2026-07-30T00:00:00Z",
         "jobDescription": "<p>Assist with statistics.</p>", "totalCount": 3},
        {"jobId": 2, "postingTitle": "ECONOMIC AFFAIRS OFFICER, P4", "jobLevel": "P-4",
         "dutyStation": [{"description": "GENEVA"}],
         "dept": {"name": "Economic Commission for Europe"},
         "startDate": "2026-06-20T04:00:00.000Z", "jobDescription": "x", "totalCount": 3},
        {"jobId": 3, "postingTitle": "HR OFFICER, P3", "jobLevel": "P-3",
         "dutyStation": [{"description": "NAIROBI"}],
         "dept": {"name": "United Nations Office at Nairobi"},
         "startDate": "2026-06-10T04:00:00.000Z", "jobDescription": "y", "totalCount": 3},
    ]
    monkeypatch.setattr(un_mod, "_fetch_board", lambda: board)
    conn = un_mod.UnConnector()
    unodc = conn.fetch("dept=Office on Drugs and Crime")
    assert len(unodc) == 1
    j = unodc[0]
    assert j.job_id == "1" and j.location == "Vienna"
    assert j.posted_date == "2026-06-30"
    assert j.url == "https://careers.un.org/jobSearchDescription/1?language=en"
    assert j.extra["level"] == "G-5"
    unece = conn.fetch("dept=Economic Commission for Europe")
    assert [x.job_id for x in unece] == ["2"]
    everything = conn.fetch("all")
    assert len(everything) == 3


def test_un_token_parsing():
    from jobscout.connectors.un import _parse_filters
    assert _parse_filters("dept=Drugs and Crime") == [("dept", "drugs and crime")]
    assert _parse_filters("dept=A;station=VIENNA") == [("dept", "a"), ("station", "vienna")]
    assert _parse_filters("all") == []


# --- SuccessFactors detail enrichment ------------------------------------------
def test_successfactors_detail_parses_description_and_date(monkeypatch):
    from jobscout.connectors import successfactors as sf
    page = """
    <html><body>
      <span itemprop="datePosted" content="2026-06-27T02:11:00.000Z">Jun 27, 2026</span>
      <div class="job"><span class="jobdescription"><p>We help the world run
      better.</p><p>You will build <b>dashboards</b> with SQL and analyse KPIs
      for stakeholders across the business — a genuinely long description body
      so the length check passes without any trouble at all.</p></span></div>
    </body></html>"""

    class FakeResp:
        text = page

    monkeypatch.setattr(sf, "http_get", lambda *a, **k: FakeResp())
    conn = sf.SuccessFactorsConnector()
    d = conn.fetch_detail("jobs.example.com", "/job/x/1/")
    assert "dashboards" in d["description"]
    assert d["posted_date"] == "2026-06-27"


def test_successfactors_detail_missing_page_is_safe(monkeypatch):
    from jobscout.connectors import successfactors as sf
    from jobscout.connectors.base import FetchError

    def boom(*a, **k):
        raise FetchError("404")

    monkeypatch.setattr(sf, "http_get", boom)
    conn = sf.SuccessFactorsConnector()
    assert conn.fetch_detail("jobs.example.com", "/job/x/1/") == {}


# --- Workday location fallback ---------------------------------------------------
def test_workday_bullet_location_fallback():
    from jobscout.connectors.workday import _bullet_location
    assert _bullet_location(["Douai", "P - Human Resources"]) == "Douai"
    assert _bullet_location(["R-12345", "Copenhagen"]) == "Copenhagen"
    assert _bullet_location(["JR0301498"]) == ""
    assert _bullet_location(None) == ""


# --- SmartRecruiters URL synthesis ----------------------------------------------
def test_smartrecruiters_url_synthesized_when_list_has_none(monkeypatch):
    from jobscout.connectors import smartrecruiters as sr
    listing = {"totalFound": 1, "content": [
        {"id": "744000", "name": "Insights Analyst",
         "releasedDate": "2026-06-20T09:00:00.000Z",
         "location": {"fullLocation": "Vienna, Austria"}}]}

    class FakeResp:
        def __init__(self, d):
            self._d = d

        def json(self):
            return self._d

    monkeypatch.setattr(sr, "http_get", lambda url, *a, **k: FakeResp(listing))
    jobs = sr.SmartRecruitersConnector().fetch("AcmeCorp")
    assert jobs[0].url == "https://jobs.smartrecruiters.com/AcmeCorp/744000"


# --- Tier-3 extraction cache ------------------------------------------------------
def test_extraction_cache_roundtrip():
    _fresh_db()
    db.extraction_cache_put("https://x.example/jobs", "hash1",
                            json.dumps([{"title": "Analyst", "location": "Vienna",
                                         "url": "https://x.example/jobs/1",
                                         "posted_date": "", "description": ""}]))
    row = db.extraction_cache_get("https://x.example/jobs")
    assert row["content_hash"] == "hash1"
    jobs = json.loads(row["jobs_json"])
    assert jobs[0]["title"] == "Analyst"
    # upsert replaces
    db.extraction_cache_put("https://x.example/jobs", "hash2", "[]")
    assert db.extraction_cache_get("https://x.example/jobs")["content_hash"] == "hash2"


# --- meta spend counters -----------------------------------------------------------
def test_meta_add_float_accumulates():
    _fresh_db()
    assert db.meta_add_float("llm_spend_usd_total", 0.5) == 0.5
    assert abs(db.meta_add_float("llm_spend_usd_total", 0.25) - 0.75) < 1e-9
    assert db.meta_get("llm_spend_usd_total").startswith("0.75")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        import inspect
        if "monkeypatch" in inspect.signature(fn).parameters:
            continue
        fn()
        print(f"  PASS  {fn.__name__}")


# --- Uber connector -------------------------------------------------------------
# 2026-09-01: Uber retired www.uber.com/api/loadSearchJobsResults (404
# ERR_MISSING_HANDLER — it broke every run for weeks) and moved the whole
# careers site to jobs.uber.com with a paged GET API. These pin the new shape.
def _uber_job(jid, title, city, country, date="2026-08-20"):
    return {"Id": jid, "Title": title, "DisplayDate": date + "T09:00:00Z",
            "Description": "<p>Do PM things</p>",
            "AdditionalDescription1": "<p>Benefits</p>",
            "Locations": [{"City": city, "Country": country, "Address": f"{city}, {country}"}],
            "Urls": [{"Culture": "en-us", "Url": f"/en/jobs/{jid}/", "IsDefault": True}],
            "Teams": ["Product"], "ContractType": "Full time"}


def test_uber_connector_parses_api_results(monkeypatch):
    from jobscout.connectors import uber as uber_mod
    pages = {
        1: {"jobs": [_uber_job(159910, "Product Manager II, Attribution",
                               "Amsterdam", "Netherlands", "2026-06-02")],
            "totalPages": 2, "totalJobs": 2},
        2: {"jobs": [_uber_job(159911, "Data Analyst", "London", "United Kingdom")],
            "totalPages": 2, "totalJobs": 2},
    }
    seen = []

    def fake_get(page):
        seen.append(page)
        return pages.get(page, {"jobs": [], "totalPages": 2})
    monkeypatch.setattr(uber_mod, "_get", fake_get)
    jobs = uber_mod.UberConnector().fetch("uber")
    assert seen == [1, 2], "walks every page, then stops at totalPages"
    assert len(jobs) == 2
    j = jobs[0]
    assert j.job_id == "159910"
    assert j.url == "https://jobs.uber.com/en/jobs/159910/"
    assert j.posted_date == "2026-06-02" and j.date_confidence == "high"
    assert j.location == "Amsterdam, Netherlands"
    assert "PM things" in j.description and "Benefits" in j.description


def test_uber_connector_keeps_only_target_countries(monkeypatch):
    """The API has no country filter and its city filter is unusable (it
    matches ANY location of a job, and half of Uber's European offices return
    nothing), so the whole board is walked and filtered here."""
    from jobscout.connectors import uber as uber_mod
    page = {"jobs": [
        _uber_job(1, "Analyst", "Amsterdam", "Netherlands"),
        _uber_job(2, "Analyst", "San Francisco", "United States"),
        _uber_job(3, "Analyst", "Sao Paulo", "Brazil"),
        _uber_job(4, "Analyst", "Vienna", "Austria"),
    ], "totalPages": 1, "totalJobs": 4}
    monkeypatch.setattr(uber_mod, "_get", lambda page_no: page)
    ids = [j.job_id for j in uber_mod.UberConnector().fetch("uber")]
    assert ids == ["1", "4"]


def test_uber_connector_keeps_multi_office_roles(monkeypatch):
    """A London+New York role is a London role for our purposes."""
    from jobscout.connectors import uber as uber_mod
    j = _uber_job(7, "BI Analyst", "New York", "United States")
    j["Locations"].append({"City": "London", "Country": "United Kingdom"})
    monkeypatch.setattr(uber_mod, "_get",
                        lambda p: {"jobs": [j], "totalPages": 1})
    out = uber_mod.UberConnector().fetch("uber")
    assert len(out) == 1 and "London, United Kingdom" in out[0].location


def test_uber_token_still_accepts_the_old_city_iso3_pairs(monkeypatch):
    """The DB may hold "Amsterdam,NLD;London,GBR" from the old connector — the
    ISO3 codes still select countries, so no migration is needed."""
    from jobscout.connectors import uber as uber_mod
    page = {"jobs": [
        _uber_job(1, "Analyst", "Amsterdam", "Netherlands"),
        _uber_job(2, "Analyst", "Berlin", "Germany"),
    ], "totalPages": 1}
    monkeypatch.setattr(uber_mod, "_get", lambda p: page)
    ids = [j.job_id for j in uber_mod.UberConnector().fetch("Amsterdam,NLD;London,GBR")]
    assert ids == ["1"], "Germany is not in the token's country set"


def test_uber_connector_cannot_loop_forever(monkeypatch):
    """A broken/absent totalPages must not spin: the page budget stops it."""
    from jobscout.connectors import uber as uber_mod
    calls = {"n": 0}

    def endless(page):
        calls["n"] += 1
        return {"jobs": [_uber_job(1000 + page, "Analyst", "Berlin", "Germany")]}
    monkeypatch.setattr(uber_mod, "_get", endless)
    jobs = uber_mod.UberConnector().fetch("uber")
    assert calls["n"] <= uber_mod._MAX_PAGES and len(jobs) <= uber_mod._MAX


def test_browser_extraction_rejects_invented_urls_and_junk_titles():
    from jobscout.browser import _to_rawjobs
    data = [
        {"title": "Data Analyst", "location": "Vienna",
         "url": "https://x.example/jobs/123", "posted_date": "", "description": ""},
        {"title": "Data Analyst 2", "location": "Vienna",
         "url": "https://x.example/invented/999", "posted_date": "", "description": ""},
        {"title": "Search Jobs", "location": "", "url": "", "posted_date": "", "description": ""},
        {"title": "Cookie Settings", "location": "", "url": "", "posted_date": "", "description": ""},
    ]
    allowed = {"https://x.example/jobs/123"}
    jobs = _to_rawjobs(data, "https://x.example", allowed_urls=allowed)
    titles = [j.title for j in jobs]
    assert "Search Jobs" not in titles and "Cookie Settings" not in titles
    by_title = {j.title: j for j in jobs}
    assert by_title["Data Analyst"].url == "https://x.example/jobs/123"
    assert by_title["Data Analyst 2"].url == "", "an invented URL must be discarded"

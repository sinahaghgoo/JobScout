"""Offline parsing tests for the connectors added beyond the original three
(Ashby, SmartRecruiters, d.vinci, eRecruiter) plus detection key-mapping.

These monkeypatch the per-connector http_get so correctness is pinned against
captured-shape payloads without hitting the network.

Run:  python -m pytest tests/test_new_connectors.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobscout.connectors import ashby, smartrecruiters, dvinci, erecruiter, amazon  # noqa: E402


class FakeResp:
    def __init__(self, json_data=None, content=b""):
        self._json = json_data
        self.content = content

    def json(self):
        return self._json


def test_ashby_parse(monkeypatch):
    payload = {"jobs": [
        {"id": "abc-1", "title": " Data Analyst ", "location": "Vienna, Austria",
         "publishedAt": "2026-06-10T12:00:00.000+00:00", "isListed": True,
         "jobUrl": "https://jobs.ashbyhq.com/x/abc-1", "descriptionPlain": "SQL and dashboards"},
        {"id": "abc-2", "title": "Hidden", "isListed": False, "location": "X"},
        {"id": "abc-3", "title": "Remote BI", "isRemote": True, "isListed": True,
         "publishedAt": "2026-05-01", "descriptionHtml": "<p>Power BI</p>"},
    ]}
    monkeypatch.setattr(ashby, "http_get", lambda *a, **k: FakeResp(payload))
    jobs = ashby.AshbyConnector().fetch("acme")
    assert len(jobs) == 2, "isListed:false must be dropped"
    a = jobs[0]
    assert a.job_id == "abc-1" and a.title == "Data Analyst"
    assert a.location == "Vienna, Austria" and a.posted_date == "2026-06-10"
    assert "SQL" in a.description and a.url.endswith("abc-1")
    assert jobs[1].location == "Remote"  # filled from isRemote when location blank


def test_smartrecruiters_list_and_detail(monkeypatch):
    listing = {"totalFound": 1, "content": [
        {"id": "9001", "name": "Insights Analyst",
         "releasedDate": "2026-06-20T09:00:00.000Z",
         "location": {"city": "Vienna", "country": "at", "fullLocation": "Vienna, Austria"},
         "postingUrl": "https://careers.x.com/9001"}]}
    detail = {"postingUrl": "https://careers.x.com/9001",
              "jobAd": {"sections": {
                  "jobDescription": {"title": "Job", "text": "<p>Build <b>dashboards</b></p>"},
                  "qualifications": {"title": "Reqs", "text": "SQL, Power BI"}}}}
    monkeypatch.setattr(smartrecruiters, "http_get",
                        lambda url, *a, **k: FakeResp(detail if "/9001" in url else listing))
    conn = smartrecruiters.SmartRecruitersConnector()
    jobs = conn.fetch("AcmeCorp")
    assert len(jobs) == 1
    j = jobs[0]
    assert j.job_id == "9001" and j.title == "Insights Analyst"
    assert j.location == "Vienna, Austria" and j.posted_date == "2026-06-20"
    assert j.detail_ref == "9001" and j.description == ""  # description deferred to detail
    d = conn.fetch_detail("AcmeCorp", "9001")
    assert "dashboards" in d["description"] and "SQL" in d["description"]
    assert d["url"].endswith("9001")


def test_dvinci_parse(monkeypatch):
    payload = [{
        "id": 27538, "position": "(Junior) SAP Developer*in",
        "jobPublicationURL": "https://jobs.x.at/de/jobs/27538/sap",
        "tasks": "<ul><li>Build reports</li></ul>", "profile": "<p>SQL skills</p>",
        "startDate": None,
        "jobOpening": {"location": "Wien", "createdDate": "2026-04-17T11:41:33.097Z",
                       "locations": [{"name": "Wien", "country": {"name": "Austria"}}],
                       "reference": "2026-0380"}}]
    monkeypatch.setattr(dvinci, "http_get", lambda *a, **k: FakeResp(payload))
    jobs = dvinci.DvinciConnector().fetch("jobs.x.at")
    assert len(jobs) == 1
    j = jobs[0]
    assert j.job_id == "27538" and j.title == "(Junior) SAP Developer*in"
    assert j.location == "Wien" and j.posted_date == "2026-04-17"
    assert "Build reports" in j.description and "SQL skills" in j.description
    assert j.url.endswith("/27538/sap")


def test_erecruiter_parse(monkeypatch):
    xml = b"""<?xml version="1.0" encoding="utf-8"?><jobs>
      <job><id>3256</id><name>SAP HCM Consultant (w/m/d)</name>
        <subTitle>Vollzeit</subTitle><location>Wien</location>
        <description><![CDATA[<p>SAP and reporting</p>]]></description>
        <requirements><![CDATA[ABAP, SQL]]></requirements>
        <url>https://www.brz-jobs.at/job/3256</url>
        <dateOfCreation>2026-03-25T00:00:00</dateOfCreation></job>
      <job><id></id><name>Broken (no id)</name></job>
    </jobs>"""
    monkeypatch.setattr(erecruiter, "http_get", lambda *a, **k: FakeResp(content=xml))
    jobs = erecruiter.ERecruiterConnector().fetch("www.brz-jobs.at")
    assert len(jobs) == 1, "rows without id/title must be skipped"
    j = jobs[0]
    assert j.job_id == "3256" and j.title.startswith("SAP HCM Consultant")
    assert j.location == "Wien" and j.posted_date == "2026-03-25"
    assert "reporting" in j.description and "ABAP" in j.description
    assert j.url.endswith("/3256")


def test_amazon_parse_and_date_and_region(monkeypatch):
    payload = {"hits": 2, "jobs": [
        {"id": "111", "title": " Data Analyst ", "normalized_location": "Dublin, Leinster, IRL",
         "posted_date": "June 20, 2026", "job_path": "/en/jobs/111/data-analyst",
         "description": "<p>Build dashboards</p>", "basic_qualifications": "SQL",
         "preferred_qualifications": "Power BI"},
        {"id": "222", "title": "SDE II", "normalized_location": "Berlin, DEU",
         "posted_date": "2026-06-01", "job_path": "/en/jobs/222/sde", "description": "code"},
    ]}
    monkeypatch.setattr(amazon, "http_get", lambda *a, **k: FakeResp(payload))
    jobs = amazon.AmazonConnector().fetch("DEU,IRL")
    assert len(jobs) == 2
    a = jobs[0]
    assert a.job_id == "111" and a.title == "Data Analyst"
    assert a.location == "Dublin, Leinster, IRL" and a.posted_date == "2026-06-20"
    assert a.url == "https://www.amazon.jobs/en/jobs/111/data-analyst"
    assert "Build dashboards" in a.description and "SQL" in a.description and "Power BI" in a.description
    assert jobs[1].posted_date == "2026-06-01"  # already-ISO dates pass through
    # region codes parse; junk/empty falls back to the default West-Europe set
    assert amazon.AmazonConnector._codes("DEU,IRL") == ["DEU", "IRL"]
    assert len(amazon.AmazonConnector._codes("amazon")) > 5
    assert amazon._parse_date("June 20, 2026") == "2026-06-20"
    assert amazon._parse_date(None) is None


def test_detect_key_mapping_handles_umlauts():
    from jobscout import detect
    # umlaut name must map to a curated token (the old [a-z0-9] norm dropped 'Ö')
    assert detect._key("Österreichische Post") in detect.KNOWN_TOKENS
    assert detect.KNOWN_TOKENS[detect._key("Österreichische Post")][0] == "successfactors"
    # slash / parenthetical names map correctly too
    assert detect._key("Volkswagen / CARIAD") in detect.KNOWN_TOKENS
    assert detect.KNOWN_TOKENS[detect._key("ESA - European Space Agency (HQ)")][0] == "successfactors"
    # the ÖBB browser-tier key resolves under KNOWN_BROWSER, umlaut intact
    assert detect._key("ÖBB Austrian Railways") in detect.KNOWN_BROWSER


def test_every_target_company_is_categorised():
    """Every seeded target company must be curated into exactly one tier — feed
    (KNOWN_TOKENS), browser (KNOWN_BROWSER), or residual (KNOWN_UNSUPPORTED) — so
    detection never has to guess (and can't be tripped by a demo board)."""
    import csv
    from jobscout import detect, config
    seed = config.DATA_DIR / "companies_seed.csv"
    if not seed.exists():
        return  # seed not generated in this environment; skip
    names = [r["name"] for r in csv.DictReader(open(seed, encoding="utf-8"))]
    for n in names:
        k = detect._key(n)
        buckets = (k in detect.KNOWN_TOKENS, k in detect.KNOWN_BROWSER,
                   k in detect.KNOWN_UNSUPPORTED)
        assert sum(buckets) == 1, \
            f"{n!r} must be in exactly one tier, got {buckets}"


def test_is_west_europe_filter():
    from jobscout.connectors.base import is_west_europe
    assert is_west_europe("Vienna, Austria")
    assert is_west_europe("Munich, DE") and is_west_europe("Cork, IRL")
    assert is_west_europe("Remote - Europe") and is_west_europe("EMEA")
    # must NOT match US / non-EU places that share names or look similar
    assert not is_west_europe("Vienna, VA")        # Virginia, USA
    assert not is_west_europe("Cairo, EG")
    assert not is_west_europe("Seattle, WA, USA")
    assert not is_west_europe("Singapore") and not is_west_europe("")


def test_smartrecruiters_subfilter_token():
    # OECD hosts the IEA/FATF; the "|kw" suffix narrows to those bodies.
    slug, kws = smartrecruiters._split_token("OECD|iea,energy agency")
    assert slug == "OECD" and kws == ["iea", "energy agency"]
    slug2, kws2 = smartrecruiters._split_token("BoschGroup")
    assert slug2 == "BoschGroup" and kws2 == []


def test_sf_rmk_parse_and_region_filter():
    from jobscout.connectors.sf_rmk import SfRmkConnector
    rss = """<?xml version='1.0'?><rss xmlns:g='http://base.google.com/ns/1.0'><channel>
      <item><title>BI Analyst</title><description><![CDATA[<p>SQL & Power BI</p>]]></description>
        <link>https://jobs.x.com/job/1/</link><g:id>1</g:id><g:location>Munich, Germany</g:location></item>
      <item><title>Retail Clerk</title><description>x</description><link>https://jobs.x.com/job/2/</link>
        <g:id>2</g:id><g:location>Austin, TX, United States</g:location></item>
    </channel></rss>"""
    eu = SfRmkConnector._parse(rss, west_europe_only=True)
    allj = SfRmkConnector._parse(rss, west_europe_only=False)
    assert [j.job_id for j in eu] == ["1"]          # only the Munich role kept
    assert len(allj) == 2                            # unfiltered keeps both
    assert eu[0].title == "BI Analyst" and "SQL" in eu[0].description
    assert eu[0].posted_date is None and eu[0].date_confidence == "low"


def test_apple_microsoft_token_parsing():
    from jobscout.connectors.apple import AppleConnector
    from jobscout.connectors.microsoft import MicrosoftConnector
    assert AppleConnector._locations("apple") == AppleConnector._locations("")
    assert "postLocation-DEU" in AppleConnector._locations("DEU,GBR")
    assert MicrosoftConnector._locations("Germany;Ireland") == ["Germany", "Ireland"]
    assert len(MicrosoftConnector._locations("microsoft")) > 5


def test_registry_has_new_connectors():
    from jobscout.connectors.registry import REGISTRY
    for ats in ("apple", "microsoft", "sf_rmk", "oracle"):
        assert ats in REGISTRY, f"{ats} missing from registry"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))

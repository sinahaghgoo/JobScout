"""Offline tests for Tier-2 (resolver) and Tier-3 (browser+LLM) plumbing, plus
the Oracle connector. Network/LLM are not hit — only the parsing/normalisation
logic is pinned.

Run:  python -m pytest tests/test_resolver_browser.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobscout import resolver, browser  # noqa: E402
from jobscout.connectors import oracle  # noqa: E402


class FakeResp:
    def __init__(self, json_data):
        self._json = json_data

    def json(self):
        return self._json


# --- Tier-2 resolver: signature extraction ----------------------------------
def test_resolver_extracts_workday_token():
    for url, want in [
        ("https://shell.wd3.myworkdayjobs.com/en-US/ShellCareers", "shell|wd3|ShellCareers"),
        ("https://roche.wd3.myworkdayjobs.com/roche-ext", "roche|wd3|roche-ext"),
        ("https://nvidia.wd5.myworkdayjobs.com/wday/cxs/nvidia/NVIDIAExternalCareerSite/jobs",
         "nvidia|wd5|NVIDIAExternalCareerSite"),
    ]:
        cands = resolver.candidate_feed(url)
        assert ("workday", want) in cands, (url, cands)


def test_resolver_extracts_other_feed_tokens():
    assert ("oracle", "jpmc.fa.oraclecloud.com|CX_1001") in resolver.candidate_feed(
        "https://jpmc.fa.oraclecloud.com/hcmUI/CandidateExperience/en/sites/CX_1001/job/123")
    assert ("greenhouse", "stripe") in resolver.candidate_feed("https://boards.greenhouse.io/stripe")
    assert ("lever", "palantir") in resolver.candidate_feed("https://jobs.lever.co/palantir/abc")
    assert ("ashby", "openai") in resolver.candidate_feed("https://jobs.ashbyhq.com/openai")
    assert ("smartrecruiters", "BoschGroup") in resolver.candidate_feed(
        "https://api.smartrecruiters.com/v1/companies/BoschGroup/postings")
    assert ("personio", "anyline-gmbh|de") in resolver.candidate_feed(
        "https://anyline-gmbh.jobs.personio.de/xml")


def test_resolver_classifies_browser_platforms():
    assert resolver.candidate_browser("see https://oebb.csod.com/ux/ats")[0] == "cornerstone"
    assert resolver.candidate_browser("https://globalcareers-atlassian.icims.com/jobs")[0] == "icims"
    assert resolver.candidate_browser("https://bloomberg.avature.net/careers")[0] == "avature"
    assert resolver.candidate_browser("https://career2.successfactors.eu/career?company=x")[0] == "successfactors-sap"
    assert resolver.candidate_browser("nothing here") is None


def test_resolver_workday_site_skips_locale():
    assert resolver._workday_site("/en-US/ShellCareers") == "ShellCareers"
    assert resolver._workday_site("/Careers") == "Careers"


# --- Oracle connector --------------------------------------------------------
def test_oracle_connector_parse(monkeypatch):
    payload = {"items": [{"TotalJobsCount": 1, "requisitionList": [
        {"Id": "210662854", "Title": "  Data Analyst ", "PostedDate": "2026-06-26",
         "PrimaryLocation": "Vienna, Austria", "secondaryLocations": [{"Name": "Dublin, Ireland"}],
         "ShortDescriptionStr": "<p>Build dashboards</p>",
         "ExternalResponsibilitiesStr": "SQL & Power BI"}]}]}
    monkeypatch.setattr(oracle, "http_get", lambda *a, **k: FakeResp(payload))
    jobs = oracle.OracleConnector().fetch("jpmc.fa.oraclecloud.com|CX_1001")
    assert len(jobs) == 1
    j = jobs[0]
    assert j.job_id == "210662854" and j.title == "Data Analyst"
    assert j.location == "Vienna, Austria, Dublin, Ireland" and j.posted_date == "2026-06-26"
    assert "dashboards" in j.description and "SQL" in j.description
    assert j.url.endswith("/sites/CX_1001/job/210662854/")
    assert j.detail_ref == "210662854"


def test_oracle_token_validation():
    from jobscout.connectors.base import FetchError
    try:
        oracle.OracleConnector()._parts("badtoken")
        raise AssertionError("expected FetchError")
    except FetchError:
        pass


# --- Tier-3 browser cleaner --------------------------------------------------
def test_browser_clean_preserves_links_and_drops_scripts():
    html = '<html><script>evil()</script><a href="/job/9">Data Analyst</a><style>x{}</style><p>Vienna</p></html>'
    out = browser.clean_html_for_llm(html)
    assert "Data Analyst" in out and "(url: /job/9)" in out
    assert "evil" not in out and "x{}" not in out


def test_browser_abs_url():
    assert browser._abs_url("/job/9", "https://x.com/careers/") == "https://x.com/job/9"
    assert browser._abs_url("https://y.com/j/1", "https://x.com") == "https://y.com/j/1"
    assert browser._abs_url("", "https://x.com") == ""


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))

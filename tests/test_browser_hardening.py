"""Regression tests for the 2026-07-04 browser-tier hardening pass.

Covers the defects found while verifying all browser-tier extractions against
live pages: entity-mangled hrefs, session-token churn, Taleo paren-URLs,
unstable job_ids, multi-URL tokens, and the new connectors' token parsing.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jobscout import browser  # noqa: E402
from jobscout.connectors import beesite, ibm, jibe, verbund  # noqa: E402
from jobscout.connectors.base import RawJob  # noqa: E402


# --- clean_html_for_llm / guard round-trip ------------------------------------

def test_clean_keeps_query_param_urls_intact():
    html = ('<a href="https://x.com/index.php?ac=jobad&amp;id=133002">'
            'Data Analyst</a>')
    cleaned = browser.clean_html_for_llm(html)
    assert "(url: https://x.com/index.php?ac=jobad&id=133002)" in cleaned


def test_clean_keeps_entities_in_titles():
    html = '<a href="https://x.com/j/1">R&amp;D Engineer</a>'
    cleaned = browser.clean_html_for_llm(html)
    assert "R&D Engineer" in cleaned


def test_page_hrefs_match_cleaned_urls():
    """What the model copies from the cleaned text must survive the guard —
    Taleo hrefs carry raw spaces + parens ('job=2026/0271 (266824)')."""
    import re
    html = ('<a href="/careersection/ex/jobdetail.ftl?job=2026/0271 (266824)'
            '&amp;tz=GMT%2B02%3A00">Team Lead</a>')
    base = "https://iaea.taleo.net/careersection/ex/jobsearch.ftl?lang=en"
    cleaned = browser.clean_html_for_llm(html)
    shown = re.search(r"\(url: ([^)]+)\)", cleaned).group(1)
    jobs = browser._to_rawjobs(
        [{"title": "Team Lead", "url": shown}], base,
        allowed_urls=browser._page_hrefs(html, base))
    assert jobs and jobs[0].url.startswith("https://iaea.taleo.net/")
    assert "%28266824%29" in jobs[0].url


def test_norm_href_encodes_spaces_and_parens_once():
    u = "/x.ftl?job=2026/0271 (266824)&tz=GMT%2B02%3A00"
    n = browser._norm_href(u)
    assert " " not in n and "(" not in n and ")" not in n
    assert "%2B" in n and "%252B" not in n  # no double-encoding


# --- session params / job ids ---------------------------------------------------

def test_strip_session_params():
    u = ("https://career2.successfactors.eu/career?company=ctbtoprepa"
         "&career_ns=job_listing&_s.crb=SECRETTOKEN%3d")
    s = browser._strip_session_params(u)
    assert "_s.crb" not in s
    assert "company=ctbtoprepa" in s and "career_ns=job_listing" in s


def test_to_rawjobs_stable_ids_and_junk_filter():
    base = "https://jobs.example.com/list"
    data = [
        {"title": "Data Analyst", "url": "https://jobs.example.com/j?id=1&_s.crb=A"},
        {"title": "BI Analyst", "url": "https://jobs.example.com/j?id=2&_s.crb=B"},
        {"title": "Login", "url": "https://jobs.example.com/login"},
    ]
    allowed = {"https://jobs.example.com/j?id=1&_s.crb=A",
               "https://jobs.example.com/j?id=2&_s.crb=B",
               "https://jobs.example.com/login"}
    jobs = browser._to_rawjobs(data, base, allowed_urls=allowed)
    assert [j.title for j in jobs] == ["Data Analyst", "BI Analyst"]
    assert all("_s.crb" not in j.url for j in jobs)
    assert jobs[0].job_id != jobs[1].job_id


def test_to_rawjobs_url_jid_collision_falls_back_to_hash():
    """Portals that link every posting to the same URL must not collapse
    distinct jobs into one."""
    base = "https://jobs.example.com/list"
    data = [
        {"title": "Analyst A", "location": "Vienna", "url": "https://jobs.example.com/search"},
        {"title": "Analyst B", "location": "Berlin", "url": "https://jobs.example.com/search"},
    ]
    jobs = browser._to_rawjobs(data, base, allowed_urls={"https://jobs.example.com/search"})
    assert len(jobs) == 2
    assert jobs[0].job_id != jobs[1].job_id


# --- multi-URL tokens -----------------------------------------------------------

def test_fetch_via_browser_merges_multi_url_tokens(monkeypatch):
    def fake_fetch_page(url):
        n = "1" if "jobOffset" not in url else "2"
        html = "".join(f'<a href="https://x.com/job/{n}{i}">Job {n}{i} title here</a>'
                       for i in range(4))
        return url, html + " filler " * 300
    canned = {
        "http": None,
    }
    def fake_cached_or_extract(key, final_url, cleaned, raw):
        n = "2" if "jobOffset" in key else "1"
        return [RawJob(job_id=f"{n}{i}", title=f"Job {n}{i}", url=f"https://x.com/job/{n}{i}")
                for i in range(4)] + [RawJob(job_id="shared", title="Shared")]
    monkeypatch.setattr(browser, "fetch_page", fake_fetch_page)
    monkeypatch.setattr(browser, "_cached_or_extract", fake_cached_or_extract)
    jobs = browser.fetch_via_browser(
        "https://x.com/SearchJobs + https://x.com/SearchJobs?jobOffset=12",
        allow_render=False)
    ids = [j.job_id for j in jobs]
    assert len(ids) == len(set(ids)) == 9  # 4 + 4 unique + 1 shared deduped
    assert "shared" in ids


# --- wait hints -----------------------------------------------------------------

def test_wait_hint_lookup_and_presence():
    assert browser._wait_hint("https://apply.lufthansagroup.careers/index.php?x=1")
    assert browser._wait_hint("https://unknown.example.com/") is None
    hint = "a[href*='ac=jobad']"
    assert browser._hint_present('<a href="/index.php?ac=jobad&id=1">x</a>', hint)
    assert not browser._hint_present("<div>no jobs yet</div>", hint)


# --- new connector token parsing -------------------------------------------------

def test_jibe_token_and_dates():
    assert jibe._split_token("https://jobs.booking.com/") == ("jobs.booking.com", "")
    assert jibe._split_token("careers.se.com?location=Germany") == \
        ("careers.se.com", "location=Germany")
    assert jibe._posted("2026-07-03T14:44:00+0000") == "2026-07-03"
    assert jibe._posted("July 3, 2026") == "2026-07-03"
    assert jibe._posted("") is None


def test_beesite_token_and_country_harvest():
    host, codes = beesite._split_token(
        "mercedes-benz-beesite-production-gjb.app.beesite.de|AT,DE")
    assert host.endswith("beesite.de") and codes == ["AT", "DE"]
    sr = {"SearchResultItems": [
        {"MatchedObjectDescriptor": {"PositionLocation": [
            {"CountryCode": "DE", "Country": "329"},
            {"CountryCode": "AT", "Country": "320"}]}}]}
    assert beesite._harvest_country_ids(sr, ["AT", "DE", "NL"]) == ["320", "329"]


def test_ibm_codes_and_jid():
    assert ibm._codes("AT, de,GB") == ["AT", "DE", "GB"]
    assert ibm._codes("ibm") == []
    assert ibm._jid("https://careers.ibm.com/careers/JobDetail?jobId=121647",
                    "fallback") == "121647"
    assert ibm._jid("nonsense", "abcdef0123456789deadbeefff") == \
        "abcdef0123456789deadbeef"


def test_verbund_module_id():
    assert verbund._module_id("verbund") == verbund._MODULE_ID
    guid = "12345678-abcd-ef01-2345-6789abcdef01"
    assert verbund._module_id(guid) == guid


# --- posting-age hard filter ------------------------------------------------

def test_posting_age_hard_filter():
    from datetime import date, timedelta
    from jobscout import config, matching

    today = date.today()
    old = (today - timedelta(days=config.MAX_POSTING_AGE_DAYS + 5)).isoformat()
    fresh = (today - timedelta(days=3)).isoformat()

    # dated postings: old dropped, fresh kept
    assert matching.hard_filter("Data Analyst", "Vienna, Austria",
                                posted_date=old) == \
        f"older than {config.MAX_POSTING_AGE_DAYS} days"
    assert matching.hard_filter("Data Analyst", "Vienna, Austria",
                                posted_date=fresh) is None

    # undated postings age from first_seen (ISO timestamp form)
    assert matching.hard_filter("Data Analyst", "Vienna, Austria",
                                first_seen=old + "T09:00:00+00:00") == \
        f"older than {config.MAX_POSTING_AGE_DAYS} days"
    assert matching.hard_filter("Data Analyst", "Vienna, Austria",
                                first_seen=fresh + "T09:00:00+00:00") is None

    # a real posted_date wins over first_seen; no info at all -> kept
    assert matching.hard_filter("Data Analyst", "Vienna, Austria",
                                posted_date=fresh,
                                first_seen=old + "T09:00:00+00:00") is None
    assert matching.hard_filter("Data Analyst", "Vienna, Austria") is None
    assert matching.posting_age_days("garbage", None) is None

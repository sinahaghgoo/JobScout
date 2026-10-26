"""Offline parsing tests for the Workday + SuccessFactors connectors.

These pin the parsing/normalisation logic against captured real-world markup, so
correctness is guaranteed without hitting the network. (A live smoke test lives
in scripts/smoke_connectors.py.)

Run:  python tests/test_connectors.py
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobscout.connectors.base import FetchError, html_to_text, norm_space  # noqa: E402
from jobscout.connectors.successfactors import SuccessFactorsConnector, _location_from_slug  # noqa: E402
from jobscout.connectors.workday import WorkdayConnector, parse_posted_on  # noqa: E402


def test_norm_space_strips_nbsp_and_zero_width():
    # DACH ATS feeds (Personio) embed NBSP/narrow-NBSP in titles & locations.
    assert norm_space("Field Specialist |\xa0Automotive") == "Field Specialist | Automotive"
    assert norm_space("Anyline HQ |\xa0Vienna") == "Anyline HQ | Vienna"
    assert norm_space("a b​c   d") == "a bc d"   # narrow-NBSP->space, ZWSP->dropped
    assert norm_space(None) == "" and norm_space("  ") == ""
    # html_to_text also folds NBSP so descriptions match cleanly on keywords.
    assert html_to_text("<p>Power\xa0BI &amp; SQL</p>") == "Power BI & SQL"


def test_workday_relative_dates():
    today = date(2026, 6, 23)
    assert parse_posted_on("Posted Today", today) == ("2026-06-23", "high")
    assert parse_posted_on("Posted Yesterday", today) == ("2026-06-22", "high")
    assert parse_posted_on("Posted 5 Days Ago", today) == ("2026-06-18", "high")
    d, conf = parse_posted_on("Posted 30+ Days Ago", today)
    assert d == "2026-05-24" and conf == "high"
    assert parse_posted_on("", today) == (None, "low")
    assert parse_posted_on("whenever", today) == (None, "low")


def test_workday_token_coords():
    wc = WorkdayConnector()
    assert wc._coords("nvidia|wd5|NVIDIAExternalCareerSite") == ("nvidia", "wd5", "NVIDIAExternalCareerSite")
    for bad in ("nvidia", "nvidia|wd5", "a|b|c|d", "|wd5|site"):
        try:
            wc._coords(bad)
            raise AssertionError(f"expected FetchError for {bad!r}")
        except FetchError:
            pass


def test_successfactors_parse_tile():
    # Minimal but faithful CSB tile markup (responsive title repeats; one job).
    html = """
    <ul>
    <li class="job-tile job-id-1364596957 job-row-index-1 linkhoverX"
        data-url="/raiffeisencapitalmgmt/job/Wien-Business-Analystin-wmx-1190/1364596957/">
      <div class="tiletitle"><span class="sr-only">Title</span>
        <span class="section-title title">
          <a class="jobTitle-link fontX" href="/x"> Business Analyst:in (w/m/x) </a>
        </span></div>
      <div class="tiletitle"><span class="sr-only">Title</span>
        <span class="section-title title">
          <a class="jobTitle-link fontX" href="/x"> Business Analyst:in (w/m/x) </a></span></div>
    </li>
    <li class="job-tile job-id-1358684157 job-row-index-2 linkhoverX"
        data-url="/job/Linz-Data-Analyst-2000/1358684157/">
      <span class="section-title title"><a class="jobTitle-link" href="/y">Data Analyst</a></span>
    </li>
    </ul>"""
    jobs = SuccessFactorsConnector._parse("jobs.rbinternational.com", html)
    assert len(jobs) == 2, f"expected 2 unique jobs, got {len(jobs)}"
    a, b = jobs
    assert a.job_id == "1364596957"
    assert a.title == "Business Analyst:in (w/m/x)"
    assert a.location == "Wien"
    assert a.url == "https://jobs.rbinternational.com/raiffeisencapitalmgmt/job/Wien-Business-Analystin-wmx-1190/1364596957/"
    assert a.posted_date is None and a.date_confidence == "low"
    assert b.job_id == "1358684157" and b.location == "Linz"


def test_successfactors_location_from_slug():
    assert _location_from_slug("/job/Wien-Studentenjob-1190/1364550457/") == "Wien"
    assert _location_from_slug("/site/job/Vancouver-SAP-iXp/1396999033/") == "Vancouver"
    assert _location_from_slug("/nope") == ""


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} connector tests passed")

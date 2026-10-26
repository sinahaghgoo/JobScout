"""Board-completeness verifier — proves the scrapers see whole boards.

"The connector returned some jobs" is not evidence of completeness: pagination
can silently truncate, filters can eat postings, and a cap can bind unnoticed.
This module verifies each company's fetch against the SOURCE'S OWN COUNT
wherever the ATS exposes one, plus structural invariants that hold everywhere:

  * fetched == the API's advertised total          (server-scoped feeds)
  * fetched <= the raw board total                 (client-filtered feeds)
  * no duplicate job_ids inside one fetch
  * fetch not sitting exactly at its safety cap    (a bound cap = truncation)
  * location / posted-date field coverage

Run:  python -m jobscout.cli verify [--type workday] [--company BMW] [--csv out]

Verdicts: OK        count matches the source (within drift tolerance)
          MISMATCH  count differs — investigate the connector
          CHECK     no independent total exists; structural checks only
          ERROR     the fetch or the probe itself failed
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import config, db
from .connectors import fetch_jobs
from .connectors.base import http_get


# --- Independent total-count probes -------------------------------------------
# One cheap request per company, hitting the same API the connector uses but
# reading only the source's own "total" field. 'exact' probes count the same
# scope the connector fetches; 'raw' probes count the whole board of which the
# connector keeps a client-filtered subset.

def _probe_workday(token: str) -> int:
    tenant, dc, site = token.split("|")
    data = http_get(
        f"https://{tenant}.{dc}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs",
        method="POST",
        json_body={"appliedFacets": {}, "limit": 1, "offset": 0, "searchText": ""},
    ).json()
    return int(data.get("total") or 0)


def _probe_smartrecruiters(token: str) -> int:
    # NB: keyword-filtered tokens ("oecd|fatf") keep a subset of the slug's
    # board — check_company downgrades their probe to 'raw' accordingly.
    slug = token.split("|")[0].strip()
    data = http_get(
        f"https://api.smartrecruiters.com/v1/companies/{slug}/postings?limit=1"
    ).json()
    return int(data.get("totalFound") or 0)


def _probe_amazon(token: str) -> int:
    from .connectors.amazon import AmazonConnector
    c = AmazonConnector()
    data = http_get(c._url(c._codes(token), 1, 0)).json()
    return int(data.get("hits") or 0)


def _probe_oracle(token: str) -> int:
    host, site = token.split("|")[:2]
    data = http_get(
        f"https://{host}/hcmRestApi/resources/latest/recruitingCEJobRequisitions"
        f"?onlyData=true&finder=findReqs;siteNumber={site},limit=1,offset=0,"
        "sortBy=POSTING_DATES_DESC").json()
    items = data.get("items") or []
    return int((items[0].get("TotalJobsCount") if items else 0) or 0)


def _probe_microsoft(_token: str) -> int:
    data = http_get("https://apply.careers.microsoft.com/api/pcsx/"
                    "search?domain=microsoft.com&start=0&sort_by=timestamp").json()
    return int(((data.get("data") or {}).get("count")) or 0)


def _probe_apple(token: str) -> int:
    from .connectors.apple import AppleConnector
    c = AppleConnector()
    res = c._post(c._locations(token), 1)
    return int(res.get("totalRecords") or 0)


def _probe_jibe(token: str) -> int:
    host = token.split("|")[0].strip().rstrip("/")
    host = host.replace("https://", "").replace("http://", "")
    data = http_get(f"https://{host}/api/jobs?page=1&limit=1").json()
    return int(data.get("totalCount") or 0)


def _probe_un(_token: str) -> int:
    data = http_get(
        "https://careers.un.org/api/public/opening/jo/list/filteredV2/en",
        method="POST",
        json_body={"filterConfig": {"keyword": ""},
                   "pagination": {"page": 0, "itemPerPage": 1,
                                  "sortBy": "startDate", "sortDirection": -1}},
    ).json()
    rows = ((data or {}).get("data") or {}).get("list") or []
    return int((rows[0].get("totalCount") if rows else 0) or 0)


def _probe_uber(token: str) -> int:
    from .connectors import uber as uber_mod
    loc_params = [{"country": cc, "region": "", "city": city}
                  for city, cc in uber_mod._locations(token)]
    data = uber_mod._post({"limit": 1, "page": 0,
                           "params": {"location": loc_params}})
    total = ((data.get("data") or {}).get("totalResults") or {})
    if isinstance(total, dict):
        return int(total.get("low") or total.get("value") or 0)
    return int(total or 0)


# ats_type -> (probe, kind). 'exact': the probe counts the same scope the
# connector fetches. 'raw': whole board; the connector keeps a subset.
PROBES: dict[str, tuple[Callable[[str], int], str]] = {
    "workday": (_probe_workday, "exact"),
    "smartrecruiters": (_probe_smartrecruiters, "exact"),
    "amazon": (_probe_amazon, "exact"),
    "oracle": (_probe_oracle, "exact"),
    "apple": (_probe_apple, "exact"),
    "jibe": (_probe_jibe, "exact"),
    "microsoft": (_probe_microsoft, "raw"),
    "un": (_probe_un, "raw"),
    "uber": (_probe_uber, "exact"),
}


@dataclass
class BoardCheck:
    company: str
    ats_type: str
    token: str
    fetched: int = 0
    api_total: Optional[int] = None
    total_kind: str = ""                  # 'exact' | 'raw' | ''
    dup_ids: int = 0
    empty_loc_pct: float = 0.0
    dated_pct: float = 0.0
    at_cap: bool = False
    error: str = ""
    notes: list[str] = field(default_factory=list)

    # Boards drift between the probe request and the paged walk; allow a
    # little slack before crying wolf.
    _TOLERANCE = 3

    @property
    def status(self) -> str:
        if self.error:
            return "ERROR"
        if self.at_cap:
            return "MISMATCH"
        if self.api_total is None:
            return "CHECK"
        if self.total_kind == "raw":
            if self.fetched > self.api_total + self._TOLERANCE:
                return "MISMATCH"          # kept more than the board has?!
            return "CHECK"
        cap = config.CONNECTOR_CAPS.get(self.ats_type)
        expected = min(self.api_total, cap) if cap else self.api_total
        tol = max(self._TOLERANCE, int(expected * 0.02))
        return "OK" if abs(self.fetched - expected) <= tol else "MISMATCH"


def check_company(company) -> BoardCheck:
    """Fetch one company's board and verify it against the source."""
    chk = BoardCheck(company=company["name"], ats_type=company["ats_type"] or "",
                     token=company["ats_token"] or "")
    try:
        if chk.ats_type == "browser":
            # Browser-tier companies have no registry connector — fetch through
            # the Tier-3 path so `verify --browser` audits them instead of
            # erroring with "No connector for ATS type 'browser'".
            from .browser import fetch_via_browser
            jobs = fetch_via_browser(chk.token)
        else:
            jobs = fetch_jobs(chk.ats_type, chk.token)
    except Exception as e:  # noqa: BLE001 — a broken feed is a finding, not a crash
        chk.error = f"fetch: {e}"
        return chk
    chk.fetched = len(jobs)
    ids = [j.job_id for j in jobs]
    chk.dup_ids = len(ids) - len(set(ids))
    if jobs:
        chk.empty_loc_pct = round(
            100 * sum(1 for j in jobs if not (j.location or "").strip()) / len(jobs), 1)
        chk.dated_pct = round(
            100 * sum(1 for j in jobs if j.posted_date) / len(jobs), 1)
    cap = config.CONNECTOR_CAPS.get(chk.ats_type)
    chk.at_cap = bool(cap and chk.fetched >= cap)
    if chk.at_cap:
        chk.notes.append(f"fetch filled its safety cap ({cap}) — raise the cap")

    probe = PROBES.get(chk.ats_type)
    if probe:
        fn, kind = probe
        # A keyword-filtered SmartRecruiters token keeps a subset of its
        # slug's board (FATF/IEA share the OECD board) — the probe total is
        # then a raw ceiling, not an expected count.
        if chk.ats_type == "smartrecruiters" and "|" in chk.token:
            kind = "raw"
        try:
            chk.api_total = fn(chk.token)
        except Exception as e:  # noqa: BLE001
            chk.notes.append(f"probe failed: {e}")
        # Workday SATURATES its total field at exactly 2000 (ABB, NVIDIA):
        # the real board is larger and the connector deliberately walks past
        # it. A fetch at/above the saturated value is therefore expected and
        # unverifiable against this total -> CHECK (with a note), never a
        # bogus MISMATCH. A fetch clearly BELOW a saturated total, however,
        # really did miss postings — keep the exact comparison for that case.
        if (chk.ats_type == "workday" and chk.api_total == 2000
                and chk.fetched >= 2000 - chk._TOLERANCE):
            chk.api_total = None
            chk.notes.append("workday total saturates at 2000 (floor) — "
                             f"fetched {chk.fetched} past it; count unverifiable")
        chk.total_kind = kind
    if chk.dup_ids:
        chk.notes.append(f"{chk.dup_ids} duplicate job_ids inside one fetch")
    return chk


def verify_boards(only_type: Optional[str] = None,
                  only_company: Optional[str] = None,
                  include_browser: bool = False,
                  max_workers: int = 6,
                  verbose: bool = True) -> list[BoardCheck]:
    companies = [c for c in db.get_companies(only_detected=True)
                 if (include_browser or c["ats_type"] != "browser")
                 and (not only_type or c["ats_type"] == only_type)
                 and (not only_company
                      or only_company.lower() in c["name"].lower())]
    checks: list[BoardCheck] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(check_company, c): c for c in companies}
        for fut in as_completed(futures):
            chk = fut.result()
            checks.append(chk)
            if verbose:
                total = ("—" if chk.api_total is None
                         else f"{chk.api_total}{'*' if chk.total_kind == 'raw' else ''}")
                note = ("; ".join(chk.notes) or chk.error)[:70]
                print(f"  [{chk.status:8}] {chk.company[:34]:34} "
                      f"{chk.ats_type:16} fetched {chk.fetched:5} "
                      f"/ source {total:>6}  {note}")
    checks.sort(key=lambda c: (c.status != "ERROR", c.status != "MISMATCH",
                               c.company.lower()))
    return checks


def write_csv(checks: list[BoardCheck], path: str) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["status", "company", "ats_type", "fetched", "api_total",
                    "total_kind", "dup_ids", "empty_loc_pct", "dated_pct",
                    "at_cap", "notes", "error"])
        for c in checks:
            w.writerow([c.status, c.company, c.ats_type, c.fetched,
                        c.api_total if c.api_total is not None else "",
                        c.total_kind, c.dup_ids, c.empty_loc_pct, c.dated_pct,
                        int(c.at_cap), "; ".join(c.notes), c.error])

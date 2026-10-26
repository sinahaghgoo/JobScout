"""Apple connector (jobs.apple.com public search API).

Landing: GET  https://jobs.apple.com/en-us/search           -> sets session cookies
List:    POST https://jobs.apple.com/api/v1/search           (JSON body) newest-first
         body {query, page, locale, sort:"newest", filters:{locations:[...]},
               format:{longDate,mediumDate}}
         -> {res:{searchResults:[{id, positionId, postingTitle,
              transformedPostingTitle, locations:[{name,countryName,countryID,
              postLocationId,level}], postingDate "Mon D, YYYY", postDateInGMT,
              jobSummary, team:{teamName}, reqId, type, standardWeeklyHours}],
              totalRecords}}

No CSRF token and no auth are needed for the *search* endpoint — the apply/profile
endpoints on gcsservices.* require a bearer token, but search is anonymous; we just
need the cookies the landing page sets. (Verified live, June 2026.)

Apple publishes ~6.4k roles globally, dominated by US/India retail, so newest-first
*without* a location filter surfaces almost no Western-Europe roles. The token
therefore carries Apple's own location keys (`postLocation-*`) for the West-Europe
markets; with them the newest-first feed is ~360 EU roles.

Apple's location keys are opaque, NOT a clean ISO mapping. The verified West-Europe
set (June 2026) mixes two forms Apple uses for a country rollup:
  * `postLocation-<ISO3>`   resolves for: DEU, IRL, GBR, NLD, AUT, DNK, SWE, POL
  * `postLocation-<ISO3>C`  resolves for: FRA(C), ITA(C), ESP(C), CHE(C), BEL(C)
(An unknown key is silently ignored by Apple and falls back to a default US page,
so only verified-good keys are baked into the default below.)

Token format:
  * "apple"  or empty                       -> default West-Europe location set
  * "<code>,<code>,..."  comma/pipe list of Apple location keys, either bare ISO3
    (e.g. "DEU,GBR") which we expand to "postLocation-DEU", or full
    "postLocation-..." keys passed through as-is.
"""
from __future__ import annotations

from datetime import datetime

from .. import config
from .base import Connector, RawJob, html_to_text, norm_space

# Verified West-Europe Apple location keys (June 2026). Each returns roles whose
# dominant country matches the request (unknown keys fall back to a US default and
# are deliberately excluded).
_DEFAULT_LOCATIONS = [
    "postLocation-DEU",   # Germany
    "postLocation-IRL",   # Ireland
    "postLocation-GBR",   # United Kingdom
    "postLocation-NLD",   # Netherlands
    "postLocation-AUT",   # Austria
    "postLocation-DNK",   # Denmark
    "postLocation-SWE",   # Sweden
    "postLocation-POL",   # Poland
    "postLocation-FRAC",  # France (country rollup)
    "postLocation-ITAC",  # Italy
    "postLocation-ESPC",  # Spain
    "postLocation-CHEC",  # Switzerland
    "postLocation-BELC",  # Belgium
]

_LANDING = "https://jobs.apple.com/en-us/search"
_SEARCH = "https://jobs.apple.com/api/v1/search"
_PAGE_SIZE = 20  # Apple returns 20 results per page
_FORMAT = {"longDate": "MMMM D, YYYY", "mediumDate": "MMM D, YYYY"}

# Apple uses a max page cap; pull the newest config.AMAZON_MAX_JOBS-equivalent.
try:
    _MAX_JOBS = config.APPLE_MAX_JOBS  # type: ignore[attr-defined]
except AttributeError:  # not configured centrally -> sensible default like Amazon's
    _MAX_JOBS = 200


def _parse_date(s: str | None) -> str | None:
    """'Jun 27, 2026' / 'June 27, 2026' -> '2026-06-27'."""
    if not s:
        return None
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _location(job: dict) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for loc in job.get("locations") or []:
        if not isinstance(loc, dict):
            continue
        name = norm_space(loc.get("name") or loc.get("city") or loc.get("countryName"))
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return ", ".join(out)


def _summary(job: dict) -> str:
    # The list endpoint already carries a usable summary; no detail fetch needed.
    return html_to_text(job.get("jobSummary") or "")


class AppleConnector(Connector):
    ats_type = "apple"

    def __init__(self) -> None:
        # Apple's search 200s only with the cookies its landing page sets and with
        # browser-ish headers; keep a private session so we don't disturb base._session.
        import requests

        self._sess = requests.Session()
        self._sess.headers.update({
            "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/124.0.0.0 Safari/537.36"),
        })
        self._seeded = False

    # -- token -> Apple location keys ----------------------------------------
    @staticmethod
    def _locations(token: str) -> list[str]:
        raw = (token or "").strip()
        if not raw or raw.lower() == "apple":
            return list(_DEFAULT_LOCATIONS)
        keys: list[str] = []
        for part in raw.replace("|", ",").split(","):
            p = part.strip()
            if not p:
                continue
            if p.startswith("postLocation-"):
                keys.append(p)
            elif len(p) == 3 and p.isalpha():
                keys.append(f"postLocation-{p.upper()}")
            else:
                keys.append(p)  # pass through anything else verbatim
        return keys or list(_DEFAULT_LOCATIONS)

    def _seed(self) -> None:
        if self._seeded:
            return
        try:
            self._sess.get(_LANDING, timeout=(8, config.HTTP_TIMEOUT))
        except Exception:  # noqa: BLE001 - cookies are best-effort
            pass
        self._seeded = True

    def _post(self, locations: list[str], page: int) -> dict:
        self._seed()
        body = {
            "query": "",
            "page": page,
            "locale": "en-us",
            "sort": "newest",
            "filters": {"locations": locations} if locations else {},
            "format": _FORMAT,
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Origin": "https://jobs.apple.com",
            "Referer": _LANDING,
            "X-Requested-With": "XMLHttpRequest",
        }
        resp = self._sess.post(_SEARCH, json=body, headers=headers,
                               timeout=(8, config.HTTP_TIMEOUT))
        resp.raise_for_status()
        return resp.json().get("res", {}) or {}

    def fetch(self, token: str) -> list[RawJob]:
        locations = self._locations(token)
        jobs: list[RawJob] = []
        seen: set[str] = set()
        page = 1
        total: int | None = None
        while len(jobs) < _MAX_JOBS:
            res = self._post(locations, page)
            batch = res.get("searchResults") or []
            if total is None:
                total = res.get("totalRecords") or 0
            if not batch:
                break
            for j in batch:
                pid = str(j.get("positionId") or j.get("id") or "").strip()
                if not pid or pid in seen:
                    continue
                seen.add(pid)
                slug = j.get("transformedPostingTitle") or ""
                url = f"https://jobs.apple.com/en-us/details/{pid}"
                if slug:
                    url += f"/{slug}"
                posted = _parse_date(j.get("postingDate"))
                if not posted and j.get("postDateInGMT"):
                    posted = str(j["postDateInGMT"])[:10]
                team = j.get("team") if isinstance(j.get("team"), dict) else {}
                jobs.append(RawJob(
                    job_id=pid,
                    title=norm_space(j.get("postingTitle")),
                    location=_location(j),
                    posted_date=posted,
                    date_confidence="high" if posted else "low",
                    url=url,
                    description=_summary(j),
                    extra={"team": (team or {}).get("teamName"),
                           "reqId": j.get("reqId"),
                           "type": j.get("type")},
                ))
            page += 1
            if total and page * _PAGE_SIZE > total:
                break
        return jobs[:_MAX_JOBS]

    def probe(self, token: str) -> bool:
        try:
            res = self._post(self._locations(token), 1)
            return bool(res.get("searchResults") or res.get("totalRecords"))
        except Exception:  # noqa: BLE001
            return False

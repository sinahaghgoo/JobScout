"""Uber connector (jobs.uber.com search API).

**2026-09-01 — Uber moved its careers site.** `www.uber.com/api/loadSearchJobsResults`
now answers 404 `ERR_MISSING_HANDLER`; the whole careers app lives on its own
host, `jobs.uber.com`, behind a plain GET API:

    GET https://jobs.uber.com/api/jobs/search/?page=N
    -> {"jobs": [{Id, Title, Description, DisplayDate, Locations, Urls,
                  Teams, ContractType, WorkPattern, ExperienceLevel, Remote,
                  Reference, Salary, Summary}, ...],
        "totalPages": 67, "totalJobs": 665, "page": 1, "pageSize": 10}

Design notes learned from probing the live API:

* **The whole board is walked, then filtered by country here.** `?location=City`
  exists but is unusable: it matches a job if ANY of its locations matches, so
  "Amsterdam" returns UK and German roles, and half of Uber's actual European
  offices (Vienna, Dublin, Zurich, Barcelona, Munich, Limerick) return 0 — the
  city spellings the filter expects are not the city names in the data. The
  board is small (~670 jobs, 10 per page), so one full walk is both cheaper to
  reason about and immune to Uber renaming a city.
* `pageSize`/`limit` are ignored — the server fixes the page at 10.
* `DisplayDate` is a real publish date (values span months), not a refresh
  stamp, so it is trusted as `posted_date`.
* Uber's edge rejects plain `requests` clients (403), so this connector needs
  curl_cffi browser impersonation, as it always has.

Token format (unchanged, so no DB migration is needed):
  * "uber" or ""            -> the profile's target countries
  * "Amsterdam,NLD;..."     -> ';'-separated "City,ISO3" pairs; only the ISO3
                               codes are used (see the city-filter note above)
  * "NLD;GBR;DEU"           -> bare ISO3 codes
"""
from __future__ import annotations

import time

from .. import config
from .base import Connector, FetchError, RawJob, html_to_text, norm_space

_API = "https://jobs.uber.com/api/jobs/search/"
_HOST = "https://jobs.uber.com"
_JOB_URL = _HOST + "/en/jobs/{jid}/"
_REFERER = _HOST + "/en/jobs/"

# ISO3 -> the country name Uber's data uses. Only the profile's target
# countries need entries; anything unmapped is compared case-insensitively
# against the raw code, so an unknown code simply matches nothing.
_ISO3_COUNTRY = {
    "AUT": "austria", "DEU": "germany", "GBR": "united kingdom",
    "IRL": "ireland", "SWE": "sweden", "CHE": "switzerland",
    "ESP": "spain", "FRA": "france", "NLD": "netherlands",
}
_DEFAULT_COUNTRIES = tuple(_ISO3_COUNTRY)

# Hard ceiling on pages walked, so a broken totalPages can never loop forever.
# It is a safety valve, NOT a cap: at 10 jobs/page this covers a 5,000-job
# board where the real one is ~670. It must stay far above any plausible board,
# because _MAX (the CONNECTOR_CAPS truncation signal) counts jobs AFTER the
# country filter and so cannot detect a page-budget stop.
_MAX_PAGES = 500

try:
    _MAX = config.UBER_MAX_JOBS  # type: ignore[attr-defined]
except AttributeError:
    _MAX = 600


def _countries(token: str) -> set:
    """Target country names (lowercase) from the connector token."""
    raw = (token or "").strip()
    codes = []
    if raw and raw.lower() != "uber":
        for part in raw.split(";"):
            part = part.strip()
            if not part:
                continue
            # "City,ISO3" -> ISO3; a bare "ISO3" -> itself. Anything that is
            # not a 3-letter code is ignored, so a token the old connector
            # would have fallen back on ("Amsterdam", a URL) still falls back
            # to the default countries instead of silently matching nothing.
            _city, _, cc = part.rpartition(",")
            code = (cc or part).strip().upper()
            if len(code) == 3 and code.isalpha():
                codes.append(code)
    codes = codes or list(_DEFAULT_COUNTRIES)
    return {_ISO3_COUNTRY.get(c, c.lower()) for c in codes if c}


def _get(page: int) -> dict:
    try:
        from curl_cffi import requests as creq
    except ImportError as e:
        raise FetchError("the Uber connector needs curl_cffi "
                         "(pip install curl_cffi)") from e
    last: Exception | None = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            time.sleep(config.REQUEST_DELAY_SECONDS)
            r = creq.get(f"{_API}?page={page}", impersonate="chrome",
                         headers={"Referer": _REFERER,
                                  "Accept": "application/json"},
                         timeout=config.HTTP_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            last = FetchError(f"HTTP {r.status_code} from the Uber jobs API")
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(1.0 * attempt)
    raise FetchError(f"Uber API failed after {config.MAX_RETRIES} tries ({last})")


def _location_str(job: dict) -> str:
    parts = []
    for loc in (job.get("Locations") or []):
        if not isinstance(loc, dict):
            continue
        city = norm_space(loc.get("City") or "")
        country = norm_space(loc.get("Country") or loc.get("CountryCode") or "")
        s = ", ".join(x for x in (city, country) if x) \
            or norm_space(loc.get("Address") or "")
        if s and s not in parts:
            parts.append(s)
    return "; ".join(parts)


def _in_target(job: dict, wanted: set) -> bool:
    """A job counts when ANY of its locations is in a target country — Uber
    posts genuinely multi-office roles, and a London+Amsterdam role is a
    London role for our purposes."""
    for loc in (job.get("Locations") or []):
        if not isinstance(loc, dict):
            continue
        name = norm_space(loc.get("Country") or "").lower()
        code = norm_space(loc.get("CountryCode") or "").lower()
        if name in wanted or code in wanted:
            return True
        if code and _ISO3_COUNTRY.get(code.upper(), "") in wanted:
            return True
    return False


def _job_url(job: dict, jid: str) -> str:
    urls = job.get("Urls") or []
    best = ""
    for u in urls:
        if not isinstance(u, dict):
            continue
        path = norm_space(u.get("Url") or "")
        if not path:
            continue
        if u.get("IsDefault") or not best:
            best = path
        if u.get("IsDefault"):
            break
    if not best:
        return _JOB_URL.format(jid=jid)
    return best if best.startswith("http") else _HOST + best


def _description(job: dict) -> str:
    chunks = [job.get("Description") or "", job.get("AdditionalDescription1") or ""]
    return html_to_text("\n".join(c for c in chunks if c))


class UberConnector(Connector):
    ats_type = "uber"
    supports_detail = False  # the search API already returns full descriptions

    def fetch(self, token: str) -> list[RawJob]:
        wanted = _countries(token)
        jobs: list[RawJob] = []
        seen: set = set()
        page = 1
        total_pages = None
        while page <= _MAX_PAGES and len(jobs) < _MAX:
            data = _get(page)
            results = data.get("jobs") or []
            if total_pages is None:
                try:
                    total_pages = int(data.get("totalPages") or 0)
                except (TypeError, ValueError):
                    total_pages = 0
            if not results:
                break
            for j in results:
                if not isinstance(j, dict):
                    continue
                jid = str(j.get("Id") or j.get("Reference") or "").strip()
                title = norm_space(j.get("Title"))
                if not jid or not title or jid in seen:
                    continue
                seen.add(jid)
                if not _in_target(j, wanted):
                    continue
                posted = (str(j.get("DisplayDate") or "")[:10]) or None
                jobs.append(RawJob(
                    job_id=jid,
                    title=title,
                    location=_location_str(j),
                    posted_date=posted,
                    date_confidence="high" if posted else "low",
                    url=_job_url(j, jid),
                    description=_description(j),
                    extra={"teams": j.get("Teams"),
                           "contract_type": j.get("ContractType"),
                           "experience_level": j.get("ExperienceLevel"),
                           "remote": j.get("Remote")},
                ))
            if total_pages and page >= total_pages:
                break
            page += 1
        return jobs[:_MAX]

    def probe(self, token: str) -> bool:
        try:
            return bool((_get(1).get("jobs")))
        except Exception:  # noqa: BLE001
            return False

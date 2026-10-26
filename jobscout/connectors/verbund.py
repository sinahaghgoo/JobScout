"""VERBUND connector (Umbraco joblist API).

VERBUND's job portal (jobs.verbund.com) is employer-branding content; the
actual vacancy list lives on the German page /de/jobportal/offene-stellen,
which fills itself from a first-party JSON API (no auth, no cookies):

    GET https://jobs.verbund.com/api/joblist/GetJobs
        ?moduleId=<CMS module GUID>&culture=de-AT&currentPage=N
    -> {"data": {"pages": P, "resultsCount": N,
                 "jobs": [{title, locations[], country, region, type,
                           link (verbund.onlyfy.jobs/job/<slug>)}, ...]}}

Postings link to onlyfy (Prescreen) job pages whose slug is the stable
job_id. The API carries no posting dates (date_confidence stays "low") and
no descriptions; relevant postings are enriched from the browser tier's
detail pass if ever needed.

Token format:
  * "verbund" or "" -> the known module GUID (below)
  * a bare GUID     -> overrides the module id (CMS redeploys can change it;
                       grab the new one from the XHR on offene-stellen)
"""
from __future__ import annotations

import re
import time

from .. import config
from .base import Connector, FetchError, RawJob, norm_space

_API = "https://jobs.verbund.com/api/joblist/GetJobs"
_MODULE_ID = "8da98ed3-9377-4a6c-9701-5e37189111b6"
_GUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_MAX_PAGES = 30


def _module_id(token: str) -> str:
    raw = (token or "").strip()
    return raw if _GUID.match(raw) else _MODULE_ID


def _get(params: dict) -> dict:
    import requests
    last: Exception | None = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            time.sleep(config.REQUEST_DELAY_SECONDS)
            r = requests.get(_API, params=params,
                             headers={"User-Agent": config.BROWSER_USER_AGENT,
                                      "Accept": "application/json"},
                             timeout=config.HTTP_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            last = FetchError(f"HTTP {r.status_code} from VERBUND joblist API")
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(1.0 * attempt)
    raise FetchError(f"VERBUND API failed after {config.MAX_RETRIES} tries ({last})")


class VerbundConnector(Connector):
    ats_type = "verbund"
    supports_detail = False  # API has no descriptions; onlyfy pages are JS-only

    def fetch(self, token: str) -> list[RawJob]:
        mid = _module_id(token)
        jobs: list[RawJob] = []
        seen: set[str] = set()
        page, total_pages = 1, 1
        while page <= total_pages and page <= _MAX_PAGES:
            data = (_get({"moduleId": mid, "culture": "de-AT",
                          "currentPage": page}).get("data")) or {}
            total_pages = int(data.get("pages") or 1)
            batch = data.get("jobs") or []
            if not batch:
                break
            for j in batch:
                url = (j.get("link") or "").strip()
                jid = url.rstrip("/").split("/")[-1] if url else ""
                title = norm_space(j.get("title"))
                if not jid or not title or jid in seen:
                    continue
                seen.add(jid)
                loc = "; ".join([norm_space(x) for x in (j.get("locations") or [])
                                 if norm_space(x)])
                country = norm_space(j.get("country") or "")
                if country and country not in loc:
                    loc = f"{loc}, {country}" if loc else country
                jobs.append(RawJob(
                    job_id=jid,
                    title=title,
                    location=loc,
                    posted_date=None,
                    date_confidence="low",
                    url=url,
                    description="",
                    extra={"type": j.get("type"), "region": j.get("region")},
                ))
            page += 1
        return jobs

    def probe(self, token: str) -> bool:
        try:
            data = (_get({"moduleId": _module_id(token), "culture": "de-AT",
                          "currentPage": 1}).get("data")) or {}
            return bool(data.get("jobs"))
        except Exception:  # noqa: BLE001
            return False

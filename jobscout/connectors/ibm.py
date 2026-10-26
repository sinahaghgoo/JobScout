"""IBM connector (careers search API).

IBM's careers search (www.ibm.com/careers/search) is a client-rendered SPA
over an Elasticsearch-style first-party API:

    POST https://www-api.ibm.com/search/api/v2
         headers: {"Content-Type": "application/json",
                   "Origin"/"Referer": ibm.com}
         body: {"appId": "careers", "scopes": ["careers2"],
                "query": {...ES bool query...}, "size": N, "from": offset}
    -> {"hits": {"total": {"value": N},
                 "hits": [{"_id", "_source": {title, url (JobDetail?jobId=),
                           description, field_keyword_19 ("City, CC"),
                           field_keyword_08 (category),
                           field_keyword_18 (level)}}]}}

Locations live in `field_keyword_19` as "City, CC"; filtering by country is a
bool-should of `wildcard` clauses ("*, DE"). The API carries no posting dates
(date_confidence stays "low").

Token format:
  * "" or "ibm"        -> whole board (~900 postings)
  * "AT,DE,GB,IE,..."  -> server-side country filter (ISO-3166 alpha-2)
"""
from __future__ import annotations

import re
import time
from urllib.parse import parse_qs, urlsplit

from .. import config
from .base import Connector, FetchError, RawJob, html_to_text, norm_space

_API = "https://www-api.ibm.com/search/api/v2"
_PAGE_SIZE = 100
try:
    _MAX_JOBS = config.IBM_MAX_JOBS  # type: ignore[attr-defined]
except AttributeError:
    _MAX_JOBS = 1500
_CC = re.compile(r"^[A-Z]{2}$")


def _codes(token: str) -> list[str]:
    raw = (token or "").strip()
    if not raw or raw.lower() == "ibm":
        return []
    return [c.strip().upper() for c in raw.split(",")
            if _CC.match(c.strip().upper())]


def _post(body: dict) -> dict:
    import requests
    headers = {"User-Agent": config.BROWSER_USER_AGENT,
               "Content-Type": "application/json",
               "Origin": "https://www.ibm.com",
               "Referer": "https://www.ibm.com/"}
    last: Exception | None = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            time.sleep(config.REQUEST_DELAY_SECONDS)
            r = requests.post(_API, json=body, headers=headers,
                              timeout=config.HTTP_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            last = FetchError(f"HTTP {r.status_code} from IBM search API")
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(1.0 * attempt)
    raise FetchError(f"IBM search API failed after {config.MAX_RETRIES} tries ({last})")


def _query(codes: list[str]) -> dict:
    if not codes:
        return {"bool": {"must": []}}
    return {"bool": {"must": [{"bool": {"should": [
        {"wildcard": {"field_keyword_19": f"*, {c}"}} for c in codes]}}]}}


def _jid(url: str, fallback: str) -> str:
    try:
        q = parse_qs(urlsplit(url).query)
        if q.get("jobId"):
            return q["jobId"][0]
    except ValueError:
        pass
    return fallback[:24]


class IbmConnector(Connector):
    ats_type = "ibm"
    supports_detail = False  # the search API already returns full descriptions

    def fetch(self, token: str) -> list[RawJob]:
        codes = _codes(token)
        jobs: list[RawJob] = []
        seen: set[str] = set()
        offset = 0
        while offset < _MAX_JOBS:
            data = _post({"appId": "careers", "scopes": ["careers2"],
                          "query": _query(codes),
                          "size": _PAGE_SIZE, "from": offset,
                          "_source": ["title", "url", "description",
                                      "field_keyword_19", "field_keyword_08",
                                      "field_keyword_18", "language"]})
            hits = ((data.get("hits") or {}).get("hits")) or []
            if not hits:
                break
            for h in hits:
                src = h.get("_source") or {}
                title = norm_space(src.get("title"))
                url = (src.get("url") or "").strip()
                jid = _jid(url, str(h.get("_id") or ""))
                if not title or not jid or jid in seen:
                    continue
                seen.add(jid)
                jobs.append(RawJob(
                    job_id=jid,
                    title=title,
                    location=norm_space(src.get("field_keyword_19") or ""),
                    posted_date=None,
                    date_confidence="low",   # API exposes no dates
                    url=url,
                    description=html_to_text(src.get("description") or ""),
                    extra={"category": src.get("field_keyword_08"),
                           "level": src.get("field_keyword_18")},
                ))
            if len(hits) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE
        return jobs

    def probe(self, token: str) -> bool:
        try:
            data = _post({"appId": "careers", "scopes": ["careers2"],
                          "query": {"bool": {"must": []}}, "size": 1, "from": 0})
            return bool(((data.get("hits") or {}).get("hits")))
        except Exception:  # noqa: BLE001
            return False

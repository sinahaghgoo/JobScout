"""Jibe (iCIMS Career Sites) connector.

Several curated companies host their careers site on Jibe's white-label
platform (Booking.com, Schneider Electric). The rendered SPA shows 10 jobs a
page, but the underlying first-party JSON API is public and complete:

    GET https://<careers-host>/api/jobs?page=N&limit=100
        [&sortBy=posted_date&descending=true]
    -> {"totalCount": N, "jobs": [{"data": {slug, req_id, title,
         posted_date, city, country_code, full_location, description,
         employment_type, categories[], meta_data.canonical_url, ...}}]}

Full descriptions and authoritative posted dates come straight from the API;
each job's canonical page URL is in meta_data.canonical_url.

The platform sits behind Akamai, which 403s plain `requests` AND headless
Chromium, but accepts curl_cffi browser impersonation (hard dependency for
this connector only — same pattern as the Uber connector).

Token format:
  * "jobs.booking.com"                    -> fetch every posting
  * "careers.se.com?location=Germany"     -> extra query appended to the API
    call (any filter the site's own search UI supports: location, keywords,
    categories, ...)
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

from .. import config
from .base import Connector, FetchError, RawJob, html_to_text, norm_space

_PAGE_SIZE = 100
_MAX_PAGES = 40  # safety valve (largest curated board is ~2.9k postings)


def _split_token(token: str) -> tuple[str, str]:
    raw = (token or "").strip().rstrip("/")
    raw = raw.replace("https://", "").replace("http://", "")
    host, _, extra = raw.partition("?")
    return host.strip("/"), extra


def _get(host: str, page: int, extra: str) -> dict:
    try:
        from curl_cffi import requests as creq
    except ImportError as e:
        raise FetchError("the Jibe connector needs curl_cffi "
                         "(pip install curl_cffi)") from e
    url = (f"https://{host}/api/jobs?page={page}&limit={_PAGE_SIZE}"
           f"&sortBy=posted_date&descending=true")
    if extra:
        url += f"&{extra}"
    last: Exception | None = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            time.sleep(config.REQUEST_DELAY_SECONDS)
            r = creq.get(url, impersonate="chrome", timeout=config.HTTP_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            last = FetchError(f"HTTP {r.status_code} from {host} jobs API")
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(1.0 * attempt)
    raise FetchError(f"Jibe API {host} failed after {config.MAX_RETRIES} tries ({last})")


def _posted(raw: str) -> Optional[str]:
    """Jibe serves ISO stamps on some sites and 'July 3, 2026' on others."""
    raw = (raw or "").strip()
    if not raw:
        return None
    if len(raw) >= 10 and raw[4] == "-":
        return raw[:10]
    try:
        return datetime.strptime(raw, "%B %d, %Y").strftime("%Y-%m-%d")
    except ValueError:
        return None


class JibeConnector(Connector):
    ats_type = "jibe"
    supports_detail = False  # the list API already returns full descriptions

    def fetch(self, token: str) -> list[RawJob]:
        host, extra = _split_token(token)
        if not host:
            raise FetchError("jibe token must be the careers host, "
                             "e.g. 'jobs.booking.com'")
        jobs: list[RawJob] = []
        seen: set[str] = set()
        page = 1
        while page <= _MAX_PAGES:
            data = _get(host, page, extra)
            batch = data.get("jobs") or []
            if not batch:
                break
            for wrap in batch:
                j = (wrap or {}).get("data") or {}
                jid = str(j.get("slug") or j.get("req_id") or "").strip()
                title = norm_space(j.get("title"))
                if not jid or not title or jid in seen:
                    continue
                seen.add(jid)
                loc = norm_space(j.get("full_location") or "")
                if not loc:
                    loc = ", ".join(x for x in
                                    (norm_space(j.get("city") or ""),
                                     norm_space(j.get("country") or
                                                j.get("country_code") or ""))
                                    if x)
                url = ((j.get("meta_data") or {}).get("canonical_url")
                       or f"https://{host}/jobs/{jid}")
                posted = _posted(str(j.get("posted_date") or
                                     j.get("create_date") or ""))
                jobs.append(RawJob(
                    job_id=jid,
                    title=title,
                    location=loc,
                    posted_date=posted,
                    date_confidence="high" if posted else "low",
                    url=url,
                    description=html_to_text(j.get("description") or ""),
                    extra={"category": j.get("category"),
                           "employment_type": j.get("employment_type"),
                           "brand": j.get("brand")},
                ))
            total = int(data.get("totalCount") or 0)
            if len(batch) < _PAGE_SIZE or (total and len(seen) >= total):
                break
            page += 1
        return jobs

    def probe(self, token: str) -> bool:
        host, extra = _split_token(token)
        if not host:
            return False
        try:
            return bool(_get(host, 1, extra).get("jobs"))
        except Exception:  # noqa: BLE001
            return False

"""SAP SuccessFactors connector (Career Site Builder).

SuccessFactors has no clean public JSON feed, so the list is read from the
platform-standard tile endpoint, which server-renders date-ordered job tiles
with a stable numeric job id:

    GET {host}/tile-search-results/?q=&sortColumn=referencedate&sortDirection=desc&startrow=N

The tiles carry no posting date or description, but on most CSB sites the job
*detail* page server-renders both — the body in a `<span class="jobdescription">`
block and the date in an `itemprop="datePosted"` microdata attribute. So the
connector returns title + stable id + url + a location hint from the list, and
`fetch_detail` (called by the pipeline only for prefilter survivors) enriches
the description and, where present, the authoritative posting date.

Token format:  "host"  e.g.  "jobs.rbinternational.com"
   (optionally "host|<query-string>" to override the default search query)
"""
from __future__ import annotations

import html
import re
from urllib.parse import unquote, urljoin

from .. import config
from .base import (BROWSER_HEADERS, Connector, FetchError, RawJob, html_to_text,
                   http_get)

_JOB_ID = re.compile(r"job-id-(\d+)")
_DATA_URL = re.compile(r'data-url="([^"]+)"')
_TITLE = re.compile(r'<a[^>]*class="[^"]*jobTitle-link[^"]*"[^>]*>(.*?)</a>', re.I | re.S)
_DEFAULT_QS = "q=&sortColumn=referencedate&sortDirection=desc"


def _location_from_slug(data_url: str) -> str:
    """CSB job slugs usually lead with the city, e.g. /job/Wien-Studentenjob.../id/."""
    m = re.search(r"/job/([^/]+)/", unquote(data_url))
    if not m:
        return ""
    first = m.group(1).split("-")[0].strip()
    return first if first and not first.isdigit() else ""


_DESC_SPAN = re.compile(
    r'<span[^>]*class="[^"]*jobdescription[^"]*"[^>]*>(.*?)</span>\s*(?:</div>|<div)',
    re.I | re.S)
_DESC_SPAN_LOOSE = re.compile(
    r'<span[^>]*class="[^"]*jobdescription[^"]*"[^>]*>(.*?)</span>', re.I | re.S)
_DATE_POSTED = re.compile(
    r'itemprop="datePosted"[^>]*content="(\d{4}-\d{2}-\d{2})', re.I)
_DATE_POSTED_TEXT = re.compile(
    r'itemprop="datePosted"[^>]*>\s*([A-Za-z]{3}\s+\d{1,2},\s+\d{4}|\d{4}-\d{2}-\d{2})', re.I)
_LOC_META = re.compile(
    r'itemprop="address"[^>]*>.*?itemprop="addressLocality"[^>]*content="([^"]+)"',
    re.I | re.S)


def _parse_us_date(s: str) -> str | None:
    """'Jun 27, 2026' or '2026-06-27' -> '2026-06-27'."""
    s = s.strip()
    if re.match(r"\d{4}-\d{2}-\d{2}$", s):
        return s
    from datetime import datetime
    try:
        return datetime.strptime(s, "%b %d, %Y").strftime("%Y-%m-%d")
    except ValueError:
        return None


class SuccessFactorsConnector(Connector):
    ats_type = "successfactors"
    supports_detail = True  # job pages server-render description (+ often a date)

    @staticmethod
    def _split_host_qs(token: str) -> tuple[str, str]:
        if "|" in token:
            host, qs = token.split("|", 1)
            return host, qs
        return token, _DEFAULT_QS

    @staticmethod
    def _parse(host: str, markup: str) -> list[RawJob]:
        out: list[RawJob] = []
        seen: set[str] = set()
        # Each job is one <li class="job-tile ... job-id-N" data-url="...">.
        for chunk in markup.split('<li class="job-tile')[1:]:
            mid = _JOB_ID.search(chunk)
            murl = _DATA_URL.search(chunk)
            if not mid or not murl:
                continue
            jid = mid.group(1)
            if jid in seen:
                continue
            mt = _TITLE.search(chunk)
            title = html_to_text(mt.group(1)) if mt else ""
            if not title:
                continue
            seen.add(jid)
            data_url = html.unescape(murl.group(1))  # raw HTML may carry &amp; etc.
            out.append(RawJob(
                job_id=jid,
                title=title,
                location=_location_from_slug(data_url),
                posted_date=None,
                date_confidence="low",
                url=urljoin(f"https://{host}", data_url),
                description="",
                detail_ref=data_url,   # job-page path; fetch_detail enriches from it
            ))
        return out

    def fetch_detail(self, token: str, detail_ref: str) -> dict:
        """Enrich one job from its (server-rendered) CSB job page."""
        host, _ = self._split_host_qs(token)
        url = urljoin(f"https://{host}", detail_ref)
        try:
            resp = http_get(url, accept="text/html", headers=BROWSER_HEADERS)
        except FetchError:
            return {}
        markup = resp.text
        out: dict = {}
        m = _DESC_SPAN.search(markup) or _DESC_SPAN_LOOSE.search(markup)
        if m:
            text = html_to_text(m.group(1))
            if len(text) > 120:          # a real body, not a styling stub
                out["description"] = text
        md = _DATE_POSTED.search(markup)
        if md:
            out["posted_date"] = md.group(1)
        else:
            md = _DATE_POSTED_TEXT.search(markup)
            if md:
                parsed = _parse_us_date(md.group(1))
                if parsed:
                    out["posted_date"] = parsed
        ml = _LOC_META.search(markup)
        if ml:
            out["location"] = html.unescape(ml.group(1)).strip()
        return out

    def probe(self, token: str) -> bool:
        """Cheap detection check: parse just the first page of tiles."""
        host, qs = self._split_host_qs(token)
        try:
            resp = http_get(f"https://{host}/tile-search-results/?{qs}&startrow=0",
                            accept="text/html")
            return len(self._parse(host, resp.text)) > 0
        except Exception:  # noqa: BLE001
            return False

    def fetch(self, token: str) -> list[RawJob]:
        host, qs = self._split_host_qs(token)
        jobs: list[RawJob] = []
        seen: set[str] = set()
        startrow = 0
        step = config.SF_PAGE_SIZE
        while startrow < config.SF_MAX_JOBS:
            url = f"https://{host}/tile-search-results/?{qs}&startrow={startrow}"
            try:
                resp = http_get(url, accept="text/html")
            except FetchError:
                if startrow == 0:
                    raise
                break
            parsed = self._parse(host, resp.text)
            if startrow == 0 and parsed:
                # CSB sites page by their OWN size (10 on some hosts, 25 on
                # others). Step by what the first page actually returned —
                # a larger step would silently skip the rows in between.
                step = max(1, len(parsed))
            page = [j for j in parsed if j.job_id not in seen]
            if not page:  # no new ids -> end of results (or no pagination support)
                break
            for j in page:
                seen.add(j.job_id)
            jobs.extend(page)
            startrow += step
        return jobs

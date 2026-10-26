"""Workday connector (CxS public API).

List:   POST {tenant}.{dc}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs
        body {searchText, limit, offset, appliedFacets}; newest first.
Detail: GET  {tenant}.{dc}.myworkdayjobs.com/wday/cxs/{tenant}/{site}{externalPath}
        -> jobPostingInfo with the authoritative startDate + full description.

The list returns a *relative* "Posted N Days Ago" string, so we approximate the
date there and replace it with the exact `startDate` during detail enrichment
(done only for roles that survive the prefilter).

Token format:  "tenant|dc|site"   e.g.  "nvidia|wd5|NVIDIAExternalCareerSite"
"""
from __future__ import annotations

import re
from datetime import date, timedelta

from .. import config
from .base import Connector, FetchError, RawJob, html_to_text, http_get

_DAYS = re.compile(r"(\d+)\+?\s*day", re.I)
_MONTHS = re.compile(r"(\d+)\+?\s*month", re.I)
# Requisition-id-looking bullet fields ("R-12345", "JR0301498", "12345BR", ...)
_REQ_ID = re.compile(r"^[A-Z]{0,4}[\s_-]?\d{3,}[A-Z]{0,4}$", re.I)


def _bullet_location(bullets: list | None) -> str:
    """Some tenants (Renault, LEGO) omit locationsText and carry the city in
    bulletFields instead. Take the first bullet that doesn't look like a
    requisition id or a department code ("P - Human Resources")."""
    for b in bullets or []:
        s = str(b or "").strip()
        if not s or _REQ_ID.match(s):
            continue
        if re.match(r"^[A-Z] - ", s):        # "P - Human Resources" dept codes
            continue
        if len(s) <= 60:
            return s
    return ""


def parse_posted_on(text: str | None, today: date | None = None) -> tuple[str | None, str]:
    """Workday's relative date -> (YYYY-MM-DD or None, confidence)."""
    today = today or date.today()
    if not text:
        return None, "low"
    t = text.lower()
    if "today" in t or "just posted" in t:
        return today.isoformat(), "high"
    if "yesterday" in t:
        return (today - timedelta(days=1)).isoformat(), "high"
    m = _DAYS.search(t)
    if m:
        return (today - timedelta(days=int(m.group(1)))).isoformat(), "high"
    m = _MONTHS.search(t)
    if m:
        return (today - timedelta(days=30 * int(m.group(1)))).isoformat(), "low"
    return None, "low"


class WorkdayConnector(Connector):
    ats_type = "workday"
    supports_detail = True

    @staticmethod
    def _coords(token: str) -> tuple[str, str, str]:
        parts = token.split("|")
        if len(parts) != 3 or not all(parts):
            raise FetchError(f"workday token must be 'tenant|dc|site', got {token!r}")
        return parts[0], parts[1], parts[2]

    @staticmethod
    def _host(tenant: str, dc: str) -> str:
        return f"https://{tenant}.{dc}.myworkdayjobs.com"

    def fetch(self, token: str) -> list[RawJob]:
        tenant, dc, site = self._coords(token)
        base = self._host(tenant, dc)
        cxs = f"{base}/wday/cxs/{tenant}/{site}/jobs"
        jobs: list[RawJob] = []
        seen: set[str] = set()
        offset, total, total_read = 0, None, False
        while offset < config.WORKDAY_MAX_JOBS:
            body = {"appliedFacets": {}, "limit": config.WORKDAY_PAGE_SIZE,
                    "offset": offset, "searchText": ""}
            data = http_get(cxs, method="POST", json_body=body).json()
            page = data.get("jobPostings", [])
            # `total` is read ONCE. total=None means "unknown — walk until an
            # empty page", and `total_read` is what keeps that decision made:
            # re-testing `total is None` on every page let a later page's total
            # overwrite it, which is how ABB's board collapsed 1971 -> 40 jobs
            # on 2026-09-01. That tenant answers total=2000 on page 1 and
            # total=0 on every page after it (while still serving 20 real
            # postings a page), so page 2 set total=0 and `offset >= total`
            # ended the walk two pages in — reported as a healthy "ok" fetch.
            if not total_read:
                total_read = True
                total = data.get("total", 0)
                # Big tenants (ABB, NVIDIA — 2026-07) SATURATE the reported
                # total at exactly 2000 while serving real postings beyond it.
                # A saturated total is a floor, not a count. A total of 0
                # alongside a full page of postings is simply wrong. Both mean
                # the same thing here: walk until an empty page (or the cap).
                if total == 2000 or (not total and page):
                    total = None
            if not page:
                break
            for j in page:
                ext = j.get("externalPath", "")
                if not ext or ext in seen:   # pages can shift mid-walk
                    continue
                seen.add(ext)
                posted, conf = parse_posted_on(j.get("postedOn"))
                jobs.append(RawJob(
                    job_id=ext,                              # stable + unique
                    title=(j.get("title") or "").strip(),
                    location=(j.get("locationsText") or "").strip()
                             or _bullet_location(j.get("bulletFields")),
                    posted_date=posted,
                    date_confidence=conf,
                    url=f"{base}/{site}{ext}",
                    description="",                          # filled by fetch_detail
                    detail_ref=ext,
                    extra={"reqId": (j.get("bulletFields") or [None])[0]},
                ))
            offset += config.WORKDAY_PAGE_SIZE
            if total is not None and offset >= total:
                break
        return jobs

    def probe(self, token: str) -> bool:
        """Cheap detection check: one posting is enough to confirm the board."""
        try:
            tenant, dc, site = self._coords(token)
            cxs = f"{self._host(tenant, dc)}/wday/cxs/{tenant}/{site}/jobs"
            data = http_get(cxs, method="POST",
                            json_body={"appliedFacets": {}, "limit": 1, "offset": 0,
                                       "searchText": ""}).json()
            return bool(data.get("total") or data.get("jobPostings"))
        except Exception:  # noqa: BLE001
            return False

    def fetch_detail(self, token: str, detail_ref: str) -> dict:
        tenant, dc, site = self._coords(token)
        url = f"{self._host(tenant, dc)}/wday/cxs/{tenant}/{site}{detail_ref}"
        info = http_get(url).json().get("jobPostingInfo", {})
        out: dict = {}
        if info.get("jobDescription"):
            out["description"] = html_to_text(info["jobDescription"])
        if info.get("startDate"):
            out["posted_date"] = str(info["startDate"])[:10]
        if info.get("externalUrl"):
            out["url"] = info["externalUrl"]
        if info.get("location"):
            # The list endpoint shows only "2 Locations" for multi-site
            # postings; the detail carries the real names. Join primary +
            # additional so the location filter and the LLM can see whether
            # ANY listed site is in the target region.
            locs = [str(info["location"]).strip()]
            for al in info.get("additionalLocations") or []:
                s = str(al or "").strip()
                if s and s not in locs:
                    locs.append(s)
            out["location"] = ", ".join(locs[:6])
        return out

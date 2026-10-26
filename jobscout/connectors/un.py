"""United Nations Secretariat connector (careers.un.org public API).

The new UN careers portal (Inspira front end) is an Angular SPA, but it feeds
from a public JSON API that needs no auth:

    POST https://careers.un.org/api/public/opening/jo/list/filteredV2/en
         {"filterConfig": {"keyword": ""},
          "pagination": {"page": N, "itemPerPage": 100,
                         "sortBy": "startDate", "sortDirection": -1}}
    -> {"data": {"list": [{jobId, postingTitle, jobLevel, dutyStation[],
                           dept{name}, startDate, endDate, jobDescription}],
                 ...}}   with totalCount on each row.

One fetch returns the WHOLE Secretariat board (a few hundred openings) with
full descriptions and authoritative posting dates. Several of the target
organizations (UNODC, UNOV, UNOG, UNECE, UNCTAD, ICJ Registry) are Secretariat
entities that recruit through this portal, so the connector filters the shared
board client-side per company.

Token format (filters are ';'-separated, matched case-insensitively):
    "dept=Drugs and Crime"                 -> dept.name contains the phrase
    "dept=Office at Vienna;dept=UNOV"      -> any filter may match
    "station=VIENNA"                       -> a duty station matches
    "all"                                  -> the whole board, unfiltered

The shared board is cached per process for a few minutes so six UN companies
in one run cost one page-walk, not six.
"""
from __future__ import annotations

import time

from .. import config
from .base import Connector, FetchError, RawJob, html_to_text, http_get, norm_space

_API = "https://careers.un.org/api/public/opening/jo/list/filteredV2/en"
_JOB_URL = "https://careers.un.org/jobSearchDescription/{jid}?language=en"
_PAGE_SIZE = 100

# Whole-board cache: (fetched_at, jobs). Valid for one pipeline run.
_cache: tuple[float, list[dict]] | None = None
_CACHE_TTL_SECONDS = 600


def _fetch_board() -> list[dict]:
    """Fetch every opening on the shared Secretariat board (cached)."""
    global _cache
    now = time.time()
    if _cache and now - _cache[0] < _CACHE_TTL_SECONDS:
        return _cache[1]
    rows: list[dict] = []
    page, total = 0, None
    while True:
        body = {"filterConfig": {"keyword": ""},
                "pagination": {"page": page, "itemPerPage": _PAGE_SIZE,
                               "sortBy": "startDate", "sortDirection": -1}}
        payload = http_get(_API, method="POST", json_body=body).json()
        data = (payload or {}).get("data") or {}
        batch = data.get("list") or []
        if not batch:
            break
        rows.extend(batch)
        if total is None:
            total = batch[0].get("totalCount") or 0
        page += 1
        if len(rows) >= total or page > 30:
            break
    if not rows:
        raise FetchError("careers.un.org returned no openings (API change?)")
    _cache = (now, rows)
    return rows


def _parse_filters(token: str) -> list[tuple[str, str]]:
    """'dept=X;station=Y' -> [('dept','x'), ('station','y')]."""
    out: list[tuple[str, str]] = []
    for part in (token or "").split(";"):
        part = part.strip()
        if not part or part.lower() == "all":
            continue
        kind, _, needle = part.partition("=")
        if kind.strip().lower() in ("dept", "station") and needle.strip():
            out.append((kind.strip().lower(), needle.strip().lower()))
    return out


def _matches(row: dict, filters: list[tuple[str, str]]) -> bool:
    if not filters:
        return True
    dept = ((row.get("dept") or {}).get("name") or "").lower()
    stations = " ".join(
        (s.get("description") or "") for s in (row.get("dutyStation") or [])).lower()
    for kind, needle in filters:
        if kind == "dept" and needle in dept:
            return True
        if kind == "station" and needle in stations:
            return True
    return False


def _to_job(row: dict) -> RawJob:
    jid = str(row.get("jobId") or "").strip()
    stations = ", ".join(
        norm_space((s.get("description") or "").title())
        for s in (row.get("dutyStation") or []) if s.get("description"))
    posted = (str(row.get("startDate") or "")[:10]) or None
    level = norm_space(row.get("jobLevel") or "")
    title = norm_space(row.get("postingTitle") or row.get("jobTitle") or "")
    return RawJob(
        job_id=jid,
        title=title,
        location=stations,
        posted_date=posted,
        date_confidence="high" if posted else "low",
        url=_JOB_URL.format(jid=jid),
        description=html_to_text(row.get("jobDescription") or ""),
        extra={"level": level or None,
               "dept": (row.get("dept") or {}).get("name"),
               "closes": (str(row.get("endDate") or "")[:10]) or None},
    )


class UnConnector(Connector):
    ats_type = "un"
    supports_detail = False  # the board feed already carries full descriptions

    def fetch(self, token: str) -> list[RawJob]:
        filters = _parse_filters(token)
        return [_to_job(r) for r in _fetch_board() if _matches(r, filters)]

    def probe(self, token: str) -> bool:
        try:
            return len(_fetch_board()) > 0
        except Exception:  # noqa: BLE001
            return False

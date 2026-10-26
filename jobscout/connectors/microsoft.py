"""Microsoft connector (careers "pcsx" search API, Eightfold-backed).

Microsoft decommissioned the old `gcsservices.careers.microsoft.com` search
API in mid-2026 (host cert no longer matches; endpoint 404s). The careers SPA
(jobs.careers.microsoft.com -> apply.careers.microsoft.com) now runs on
Eightfold's "pcsx" API, which is public and anonymous:

    List:   GET https://apply.careers.microsoft.com/api/pcsx/search
                ?domain=microsoft.com&start=<offset>&sort_by=timestamp
            -> {"data": {"count": N, "positions": [{id, name, locations[],
                postedTs (unix), positionUrl, department,
                workLocationOption}]}}
            Page size is server-fixed at 10; `start` paginates.

    Detail: GET https://apply.careers.microsoft.com/api/pcsx/position_details
                ?position_id=<id>&domain=microsoft.com&hl=en
            -> {"data": {jobDescription, location, efcustomText*}}

The board is ~1.3k roles, so the connector walks the whole list newest-first
and keeps postings whose location matches the token's location filters
(client-side substring match — the API's `location` param is a geo-radius
search that misses country-wide roles). Descriptions come from the detail
endpoint, which the pipeline calls only for shortlisted roles.

Token format:
  * "microsoft" or empty                 -> default West-Europe country set
  * "Germany;Ireland;United Kingdom;..." -> ';'-separated location substrings
"""
from __future__ import annotations

from datetime import datetime, timezone

from .. import config
from .base import Connector, FetchError, RawJob, html_to_text, http_get, norm_space

_BASE = "https://apply.careers.microsoft.com/api/pcsx"
_PAGE_SIZE = 10          # server-fixed; larger `num` values are ignored
_MAX_PAGES = 160         # whole board is ~1.3k roles -> ~133 pages

_DEFAULT_LOCATIONS = [
    "Germany", "Ireland", "United Kingdom", "Netherlands", "France", "Spain",
    "Italy", "Switzerland", "Austria", "Poland", "Czech Republic", "Romania",
    "Portugal", "Sweden", "Denmark", "Finland", "Norway", "Belgium",
]

try:
    _MAX_JOBS = config.MICROSOFT_MAX_JOBS  # type: ignore[attr-defined]
except AttributeError:
    _MAX_JOBS = 200


def _job_url(jid: str) -> str:
    return f"https://apply.careers.microsoft.com/careers/job/{jid}?domain=microsoft.com"


def _posted(ts) -> str | None:
    try:
        ts = int(ts)
    except (TypeError, ValueError):
        return None
    if ts <= 0:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


class MicrosoftConnector(Connector):
    ats_type = "microsoft"
    supports_detail = True   # descriptions live on the position_details endpoint

    @staticmethod
    def _locations(token: str) -> list[str]:
        raw = (token or "").strip()
        if not raw or raw.lower() == "microsoft":
            return list(_DEFAULT_LOCATIONS)
        parts = [p.strip() for p in raw.replace("|", ";").split(";") if p.strip()]
        return parts or list(_DEFAULT_LOCATIONS)

    def fetch(self, token: str) -> list[RawJob]:
        wanted = [w.lower() for w in self._locations(token)]
        jobs: list[RawJob] = []
        seen: set[str] = set()
        start, total = 0, None
        pages_failed = 0
        for _ in range(_MAX_PAGES):
            if len(jobs) >= _MAX_JOBS or (total is not None and start >= total):
                break
            try:
                payload = http_get(
                    f"{_BASE}/search?domain=microsoft.com&start={start}"
                    f"&sort_by=timestamp").json()
            except FetchError:
                pages_failed += 1
                if start == 0:
                    # First page unreachable — surface loudly instead of a
                    # silent "0 jobs" that looks like an empty board.
                    raise FetchError(
                        "Microsoft careers API unreachable (pcsx search failed)")
                break
            data = (payload.get("data") or {})
            batch = data.get("positions") or []
            if total is None:
                total = int(data.get("count") or 0)
            if not batch:
                break
            for pos in batch:
                jid = str(pos.get("id") or "").strip()
                title = norm_space(pos.get("name"))
                if not jid or not title or jid in seen:
                    continue
                locs = [norm_space(str(x)) for x in (pos.get("locations") or [])
                        if norm_space(str(x))]
                loc_str = "; ".join(locs)
                if not any(w in loc_str.lower() for w in wanted):
                    continue
                seen.add(jid)
                posted = _posted(pos.get("postedTs") or pos.get("creationTs"))
                jobs.append(RawJob(
                    job_id=jid,
                    title=title,
                    location=loc_str,
                    posted_date=posted,
                    date_confidence="high" if posted else "low",
                    url=_job_url(jid),
                    description="",            # enriched via fetch_detail
                    detail_ref=jid,
                    extra={"department": pos.get("department"),
                           "workLocationOption": pos.get("workLocationOption")},
                ))
            start += _PAGE_SIZE
        return jobs[:_MAX_JOBS]

    def fetch_detail(self, token: str, detail_ref: str) -> dict:
        """Full description for one role (pipeline calls this only for the few
        postings that survive prefiltering)."""
        try:
            payload = http_get(
                f"{_BASE}/position_details?position_id={detail_ref}"
                "&domain=microsoft.com&hl=en").json()
        except FetchError:
            return {}
        data = (payload.get("data") or {})
        out: dict = {}
        desc = html_to_text(data.get("jobDescription") or "")
        if desc:
            out["description"] = desc
        loc = norm_space(str(data.get("location") or ""))
        if loc:
            out["location"] = loc
        return out

    def probe(self, token: str) -> bool:
        try:
            payload = http_get(
                f"{_BASE}/search?domain=microsoft.com&start=0").json()
            return bool(((payload.get("data") or {}).get("positions")))
        except Exception:  # noqa: BLE001
            return False

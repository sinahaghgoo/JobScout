"""d.vinci connector (used by several DACH employers, e.g. Wiener Stadtwerke).

GET {host}/jobPublication/list.json
JSON array; each item carries position (title), jobPublicationURL, and a nested
`jobOpening` object with location, createdDate and salary. The HTML body lives in
`introduction` / `tasks` / `profile` / `weOffer` fields. No auth.

Token format:  "host"  e.g.  "jobs.wienerstadtwerke.at"
"""
from __future__ import annotations

from .base import Connector, RawJob, html_to_text, http_get


def _loc(opening: dict) -> str:
    if not isinstance(opening, dict):
        return ""
    locs = opening.get("locations")
    if isinstance(locs, list) and locs:
        names = [l.get("name") for l in locs if isinstance(l, dict) and l.get("name")]
        if names:
            return ", ".join(dict.fromkeys(names))
    return str(opening.get("location") or "").strip()


class DvinciConnector(Connector):
    ats_type = "dvinci"
    LIST = "https://{host}/jobPublication/list.json"

    def fetch(self, token: str) -> list[RawJob]:
        host = token.split("|", 1)[0]
        resp = http_get(self.LIST.format(host=host))
        data = resp.json()
        items = data if isinstance(data, list) else (
            data.get("jobPublications") or data.get("data") or [])
        jobs: list[RawJob] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            opening = it.get("jobOpening") or {}
            parts = [html_to_text(it.get(k) or "")
                     for k in ("introduction", "tasks", "profile", "weOffer")]
            desc = "\n\n".join(p for p in parts if p)
            created = opening.get("createdDate") or it.get("startDate") or ""
            jobs.append(
                RawJob(
                    job_id=str(it.get("id")),
                    title=(it.get("position") or it.get("pageTitle") or "").strip(),
                    location=_loc(opening),
                    posted_date=created[:10] if created else None,
                    date_confidence="high" if created else "low",
                    url=it.get("jobPublicationURL") or "",
                    description=desc,
                    extra={"reference": opening.get("reference"),
                           "department": opening.get("department")},
                )
            )
        return jobs

    def probe(self, token: str) -> bool:
        try:
            return len(self.fetch(token)) > 0
        except Exception:  # noqa: BLE001
            return False

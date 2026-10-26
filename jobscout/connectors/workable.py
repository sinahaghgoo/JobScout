"""Workable connector (public job-board widget API).

GET apply.workable.com/api/v1/widget/accounts/{token}?details=true
JSON: {jobs:[{shortcode, title, location:{city,country,region}, created_at,
              url, description, ...}]} — no auth, for accounts that publish a
public board. Returns an empty list for private boards.

Token format:  "{subdomain}"  (the company's apply.workable.com subdomain)
"""
from __future__ import annotations

from .base import Connector, RawJob, html_to_text, http_get


def _location(loc) -> str:
    if isinstance(loc, dict):
        parts = [loc.get("city"), loc.get("region"), loc.get("country")]
        out = ", ".join(p for p in parts if p)
        if loc.get("workplace_type") == "remote" or loc.get("telecommuting"):
            out = (out + " (Remote)").strip() if out else "Remote"
        return out
    return str(loc or "").strip()


class WorkableConnector(Connector):
    ats_type = "workable"
    BASE = "https://apply.workable.com/api/v1/widget/accounts/{token}?details=true"

    def fetch(self, token: str) -> list[RawJob]:
        resp = http_get(self.BASE.format(token=token))
        data = resp.json()
        jobs: list[RawJob] = []
        for j in data.get("jobs", []):
            created = j.get("created_at") or j.get("published_on") or ""
            jobs.append(
                RawJob(
                    job_id=str(j.get("shortcode") or j.get("id")),
                    title=(j.get("title") or "").strip(),
                    location=_location(j.get("location")),
                    posted_date=created[:10] if created else None,
                    url=j.get("url") or j.get("application_url", ""),
                    description=html_to_text(j.get("description", "")),
                    extra={"department": j.get("department"),
                           "employment_type": j.get("employment_type")},
                )
            )
        return jobs

    def probe(self, token: str) -> bool:
        try:
            data = http_get(self.BASE.format(token=token)).json()
            return bool(data.get("jobs"))
        except Exception:  # noqa: BLE001
            return False

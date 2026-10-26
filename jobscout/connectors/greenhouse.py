"""Greenhouse connector.

GET boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true
Returns all jobs with full descriptions, offices and updated_at — no auth.
"""
from __future__ import annotations

from .base import Connector, RawJob, html_to_text, http_get


class GreenhouseConnector(Connector):
    ats_type = "greenhouse"
    BASE = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true"

    def fetch(self, token: str) -> list[RawJob]:
        resp = http_get(self.BASE.format(token=token))
        data = resp.json()
        jobs: list[RawJob] = []
        for j in data.get("jobs", []):
            loc = (j.get("location") or {}).get("name", "") or ""
            # Prefer the first-published date; fall back to last update.
            posted = j.get("first_published") or j.get("updated_at") or ""
            posted = posted[:10] if posted else None
            jobs.append(
                RawJob(
                    job_id=str(j.get("id")),
                    title=(j.get("title") or "").strip(),
                    location=loc.strip(),
                    posted_date=posted,
                    url=j.get("absolute_url", ""),
                    description=html_to_text(j.get("content", "")),
                    extra={"departments": [d.get("name") for d in j.get("departments", [])]},
                )
            )
        return jobs

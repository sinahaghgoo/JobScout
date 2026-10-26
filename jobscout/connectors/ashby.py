"""Ashby connector.

GET api.ashbyhq.com/posting-api/job-board/{token}
One call returns the whole public board as JSON — no auth. Each posting carries
title, a plain-text location, publishedAt, the public jobUrl and both an HTML and
a plain-text description.
"""
from __future__ import annotations

from .base import Connector, RawJob, html_to_text, http_get


class AshbyConnector(Connector):
    ats_type = "ashby"
    BASE = "https://api.ashbyhq.com/posting-api/job-board/{token}?includeCompensation=true"

    def fetch(self, token: str) -> list[RawJob]:
        resp = http_get(self.BASE.format(token=token))
        data = resp.json()
        jobs: list[RawJob] = []
        for j in data.get("jobs", []):
            # Only surface roles that are actually listed on the public board.
            if j.get("isListed") is False:
                continue
            posted = j.get("publishedAt") or ""
            desc = j.get("descriptionPlain") or html_to_text(j.get("descriptionHtml", ""))
            loc = j.get("location") or ""
            if not loc and j.get("isRemote"):
                loc = "Remote"
            jobs.append(
                RawJob(
                    job_id=str(j.get("id")),
                    title=(j.get("title") or "").strip(),
                    location=str(loc).strip(),
                    posted_date=posted[:10] if posted else None,
                    url=j.get("jobUrl") or j.get("applyUrl", ""),
                    description=desc,
                    extra={
                        "department": j.get("department"),
                        "team": j.get("team"),
                        "employmentType": j.get("employmentType"),
                    },
                )
            )
        return jobs

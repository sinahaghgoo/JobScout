"""Recruitee connector.

GET {token}.recruitee.com/api/offers/
JSON: {offers:[{id, title, location, city, country, careers_url, created_at,
                description, department, ...}]} — no auth. Popular with EU SMEs.
"""
from __future__ import annotations

from .base import Connector, RawJob, html_to_text, http_get


class RecruiteeConnector(Connector):
    ats_type = "recruitee"
    BASE = "https://{token}.recruitee.com/api/offers/"

    def fetch(self, token: str) -> list[RawJob]:
        resp = http_get(self.BASE.format(token=token))
        data = resp.json()
        jobs: list[RawJob] = []
        for o in data.get("offers", []):
            # status 'published' only; Recruitee also returns internal/draft sometimes
            if o.get("status") and o.get("status") != "published":
                continue
            loc = o.get("location") or ", ".join(
                p for p in (o.get("city"), o.get("country")) if p)
            created = o.get("created_at") or ""
            jobs.append(
                RawJob(
                    job_id=str(o.get("id")),
                    title=(o.get("title") or "").strip(),
                    location=str(loc).strip(),
                    posted_date=created[:10] if created else None,
                    url=o.get("careers_url") or o.get("careers_apply_url", ""),
                    description=html_to_text(o.get("description", "")),
                    extra={"department": o.get("department"),
                           "employment_type": o.get("employment_type_code")},
                )
            )
        return jobs

"""SmartRecruiters connector (public Posting API).

List:   GET api.smartrecruiters.com/v1/companies/{token}/postings?limit=100&offset=N
        -> {content:[{id, name, releasedDate, location{city,country,fullLocation},
                      ref, ...}], totalFound}   (published postings, no auth)
Detail: GET api.smartrecruiters.com/v1/companies/{token}/postings/{id}
        -> {postingUrl, jobAd.sections.{companyDescription,jobDescription,
            qualifications,additionalInformation}.text}

The list endpoint has everything except the long description, so we fetch detail
only for the few roles that survive the prefilter (supports_detail = True).

Token format:  "{companyIdentifier}"  e.g.  "Visa", "BoschGroup"
Optional sub-filter: "{companyIdentifier}|kw1,kw2" keeps only postings whose
title/department/location contains one of the keywords — used where several
bodies share one SmartRecruiters account (e.g. OECD also hosts the IEA and FATF:
"OECD|iea,energy agency" / "OECD|fatf,financial action").
"""
from __future__ import annotations

from .. import config
from .base import Connector, RawJob, html_to_text, http_get


def _split_token(token: str) -> tuple[str, list[str]]:
    """'OECD|iea,energy agency' -> ('OECD', ['iea', 'energy agency'])."""
    slug, _, flt = (token or "").partition("|")
    keywords = [k.strip().lower() for k in flt.split(",") if k.strip()]
    return slug.strip(), keywords


def _location(loc: dict | None) -> str:
    if not loc:
        return ""
    if loc.get("fullLocation"):
        return str(loc["fullLocation"]).strip()
    parts = [loc.get("city"), loc.get("region"), loc.get("country")]
    out = ", ".join(p for p in parts if p)
    if loc.get("remote"):
        out = (out + " (Remote)").strip() if out else "Remote"
    return out


class SmartRecruitersConnector(Connector):
    ats_type = "smartrecruiters"
    supports_detail = True
    LIST = "https://api.smartrecruiters.com/v1/companies/{token}/postings?limit={limit}&offset={offset}"
    DETAIL = "https://api.smartrecruiters.com/v1/companies/{token}/postings/{pid}"

    def fetch(self, token: str) -> list[RawJob]:
        slug, keywords = _split_token(token)
        jobs: list[RawJob] = []
        offset, total = 0, None
        page_size = 100
        while offset < config.SR_MAX_JOBS:
            url = self.LIST.format(token=slug, limit=page_size, offset=offset)
            data = http_get(url).json()
            if total is None:
                total = data.get("totalFound", 0)
            content = data.get("content", [])
            if not content:
                break
            for j in content:
                rel = j.get("releasedDate") or ""
                dep = (j.get("department") or {}).get("label") \
                    if isinstance(j.get("department"), dict) else j.get("department")
                title = (j.get("name") or "").strip()
                location = _location(j.get("location"))
                if keywords:
                    hay = f"{title} {dep or ''} {location}".lower()
                    if not any(k in hay for k in keywords):
                        continue
                # The list endpoint carries no postingUrl — synthesize the public
                # job page (stable, works for every published posting); detail
                # enrichment replaces it with the canonical URL if one is given.
                jid = str(j.get("id"))
                url = (j.get("postingUrl") or j.get("applyUrl")
                       or f"https://jobs.smartrecruiters.com/{slug}/{jid}")
                jobs.append(
                    RawJob(
                        job_id=jid,
                        title=title,
                        location=location,
                        posted_date=rel[:10] if rel else None,
                        url=url,
                        description="",  # filled by fetch_detail for survivors
                        detail_ref=jid,
                        extra={"department": dep},
                    )
                )
            offset += page_size
            if total is not None and offset >= total:
                break
        return jobs

    def fetch_detail(self, token: str, detail_ref: str) -> dict:
        slug, _ = _split_token(token)
        url = self.DETAIL.format(token=slug, pid=detail_ref)
        data = http_get(url).json()
        out: dict = {}
        sections = ((data.get("jobAd") or {}).get("sections") or {})
        chunks = []
        for key in ("jobDescription", "qualifications", "additionalInformation"):
            sec = sections.get(key) or {}
            text = html_to_text(sec.get("text", ""))
            if text:
                label = sec.get("title") or key
                chunks.append(f"{label}\n{text}")
        if chunks:
            out["description"] = "\n\n".join(chunks)
        if data.get("postingUrl"):
            out["url"] = data["postingUrl"]
        loc = _location(data.get("location"))
        if loc:
            out["location"] = loc
        return out

    def probe(self, token: str) -> bool:
        try:
            slug, _ = _split_token(token)
            url = self.LIST.format(token=slug, limit=1, offset=0)
            data = http_get(url).json()
            return bool(data.get("content"))
        except Exception:  # noqa: BLE001
            return False

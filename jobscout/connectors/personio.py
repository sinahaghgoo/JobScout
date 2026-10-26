"""Personio connector.

GET {token}.jobs.personio.de/xml?language=en   (some tenants use .com)
XML feed: title, office, department, employment type, description, createdAt.
Dominant ATS across the DACH region — no auth.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

from .base import Connector, FetchError, RawJob, html_to_text, http_get, norm_space


class PersonioConnector(Connector):
    ats_type = "personio"
    # token may carry an explicit tld as "token|com"; default is .de
    DOMAINS = ("de", "com")

    def _candidates(self, token: str) -> list[tuple[str, str, str]]:
        """[(url, base, tld)] to try, honoring an explicit 'token|tld'."""
        if "|" in token:
            base, tld = token.split("|", 1)
            return [(f"https://{base}.jobs.personio.{tld}/xml?language=en", base, tld)]
        return [(f"https://{token}.jobs.personio.{tld}/xml?language=en", token, tld)
                for tld in self.DOMAINS]

    def fetch(self, token: str) -> list[RawJob]:
        last_err: Exception | None = None
        for url, base, tld in self._candidates(token):
            try:
                resp = http_get(url, accept="application/xml")
                jobs = self._parse(resp.content)
                # The XML carries no per-job public URL; synthesize the board
                # link HERE, where we know which tld actually served the feed
                # (a bare token may resolve on .com — links must not assume .de).
                for j in jobs:
                    if not j.url and j.job_id:
                        j.url = (f"https://{base}.jobs.personio.{tld}"
                                 f"/job/{j.job_id}?language=en")
                return jobs
            except FetchError as e:
                last_err = e
                continue
        raise FetchError(f"personio: no feed for '{token}' ({last_err})")

    @staticmethod
    def _parse(content: bytes) -> list[RawJob]:
        root = ET.fromstring(content)
        jobs: list[RawJob] = []
        for pos in root.iter("position"):
            def txt(tag: str) -> str:
                el = pos.find(tag)
                return norm_space(el.text) if el is not None and el.text else ""

            descriptions = []
            jd = pos.find("jobDescriptions")
            if jd is not None:
                for d in jd.findall("jobDescription"):
                    name = (d.findtext("name") or "").strip()
                    value = html_to_text(d.findtext("value") or "")
                    if value:
                        descriptions.append((f"{name}\n" if name else "") + value)

            created = txt("createdAt")
            jobs.append(
                RawJob(
                    job_id=txt("id"),
                    title=txt("name"),
                    location=txt("office"),
                    posted_date=created[:10] if created else None,
                    url="",  # Personio XML has no per-job public URL; filled by caller if known
                    description="\n\n".join(descriptions),
                    extra={
                        "department": txt("department"),
                        "employmentType": txt("employmentType"),
                        "schedule": txt("schedule"),
                        "seniority": txt("seniority"),
                    },
                )
            )
        return jobs

"""eRecruiter connector (common in Austria, e.g. BRZ Bundesrechenzentrum).

GET {host}/Syndication/Xml
XML feed: <jobs><job><id/><name/><location/><description/><requirements/>
<url/><dateOfCreation/>...</job></jobs>. No auth.

Token format:  "host"  e.g.  "www.brz-jobs.at"
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

from .base import Connector, RawJob, html_to_text, http_get


class ERecruiterConnector(Connector):
    ats_type = "erecruiter"
    SYND = "https://{host}/Syndication/Xml"

    @staticmethod
    def _date(job: ET.Element) -> str | None:
        for tag in ("dateOfCreation", "dateSinceOnlineCorrected", "dateSinceOnline",
                    "publishedOn", "onlineDate"):
            v = job.findtext(tag)
            if v and v.strip():
                return v.strip()[:10]
        return None

    def fetch(self, token: str) -> list[RawJob]:
        host = token.split("|", 1)[0]
        resp = http_get(self.SYND.format(host=host), accept="application/xml")
        root = ET.fromstring(resp.content)
        jobs: list[RawJob] = []
        for job in root.iter("job"):
            jid = (job.findtext("id") or "").strip()
            title = (job.findtext("name") or "").strip()
            if not jid or not title:
                continue
            desc_parts = [html_to_text(job.findtext(t) or "")
                          for t in ("description", "requirements", "additionalText")]
            sub = (job.findtext("subTitle") or "").strip()
            desc = "\n\n".join(p for p in desc_parts if p)
            date = self._date(job)
            jobs.append(
                RawJob(
                    job_id=jid,
                    title=title,
                    location=(job.findtext("location") or job.findtext("region") or "").strip(),
                    posted_date=date,
                    date_confidence="high" if date else "low",
                    url=(job.findtext("url") or job.findtext("applyUrl") or "").strip(),
                    description=(sub + "\n\n" + desc).strip() if sub else desc,
                    extra={"careerLevel": job.findtext("careerLevel"),
                           "occupationType": job.findtext("occupationType")},
                )
            )
        return jobs

    def probe(self, token: str) -> bool:
        try:
            return len(self.fetch(token)) > 0
        except Exception:  # noqa: BLE001
            return False

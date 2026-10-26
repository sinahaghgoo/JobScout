"""Amazon connector (amazon.jobs public search API).

GET www.amazon.jobs/en/search.json?result_limit=100&offset=N&sort=recent
    &normalized_country_code[]=DEU&normalized_country_code[]=IRL&...
Returns {jobs:[{id, id_icims, title, normalized_location, city, country_code,
                posted_date "Month DD, YYYY", job_path, description,
                basic_qualifications, preferred_qualifications, ...}], hits} — no auth.

Amazon publishes ~10k roles globally, so the token carries the set of country
codes to restrict to (the candidate's West-Europe / EU region); without it the
newest-first page would be almost all US/India.

Token format:  comma-separated ISO-3166 alpha-3 codes, e.g. "DEU,IRL,LUX,AUT".
A bare "amazon" (or empty) falls back to the default West-Europe / EU set.
"""
from __future__ import annotations

from datetime import datetime
from urllib.parse import quote

from .. import config
from .base import Connector, RawJob, html_to_text, http_get, norm_space

# West Europe (EU or UK) ISO-3166 alpha-3 codes — the default region filter.
_DEFAULT_CODES = [
    "AUT", "DEU", "CHE", "IRL", "GBR", "NLD", "BEL", "LUX", "FRA", "ESP",
    "PRT", "ITA", "DNK", "SWE", "NOR", "FIN", "POL", "CZE", "HUN", "ROU", "GRC",
]


def _parse_date(s: str | None) -> str | None:
    """'June 26, 2026' -> '2026-06-26'."""
    if not s:
        return None
    for fmt in ("%B %d, %Y", "%Y-%m-%d", "%b %d, %Y"):
        try:
            return datetime.strptime(s.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return s[:10] if len(s) >= 10 and s[4] == "-" else None


class AmazonConnector(Connector):
    ats_type = "amazon"
    BASE = "https://www.amazon.jobs/en/search.json"

    @staticmethod
    def _codes(token: str) -> list[str]:
        parts = [p.strip().upper() for p in (token or "").replace("|", ",").split(",")]
        codes = [p for p in parts if len(p) == 3 and p.isalpha()]
        return codes or _DEFAULT_CODES

    def _url(self, codes: list[str], limit: int, offset: int) -> str:
        cc = "".join(f"&normalized_country_code[]={quote(c)}" for c in codes)
        return f"{self.BASE}?result_limit={limit}&offset={offset}&sort=recent{cc}"

    def fetch(self, token: str) -> list[RawJob]:
        codes = self._codes(token)
        jobs: list[RawJob] = []
        seen: set[str] = set()
        offset, page = 0, 100
        while offset < config.AMAZON_MAX_JOBS:
            data = http_get(self._url(codes, page, offset)).json()
            batch = data.get("jobs") or []
            if not batch:
                break
            for j in batch:
                jid = str(j.get("id") or j.get("id_icims") or "").strip()
                if not jid or jid in seen:
                    continue
                seen.add(jid)
                desc = html_to_text(j.get("description") or "")
                for extra in ("basic_qualifications", "preferred_qualifications"):
                    t = html_to_text(j.get(extra) or "")
                    if t:
                        desc += f"\n\n{extra.split('_')[0].title()} qualifications\n{t}"
                path = j.get("job_path") or ""
                posted = _parse_date(j.get("posted_date"))
                jobs.append(RawJob(
                    job_id=jid,
                    title=norm_space(j.get("title")),
                    location=norm_space(j.get("normalized_location") or j.get("location")),
                    posted_date=posted,
                    date_confidence="high" if posted else "low",
                    url=("https://www.amazon.jobs" + path) if path.startswith("/") else path,
                    description=desc.strip(),
                    extra={"team": (j.get("team") or {}).get("label")
                           if isinstance(j.get("team"), dict) else None,
                           "is_intern": j.get("is_intern")},
                ))
            offset += page
            if offset >= int(data.get("hits") or 0):
                break
        return jobs

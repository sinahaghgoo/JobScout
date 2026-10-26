"""Oracle Cloud Recruiting (ORC / "Fusion") connector — public Candidate-Experience API.

Used by many large enterprises (JPMorgan, …) behind a `*.fa.oraclecloud.com` host.

List:   GET {host}/hcmRestApi/resources/latest/recruitingCEJobRequisitions
            ?onlyData=true&expand=requisitionList.secondaryLocations
            &finder=findReqs;siteNumber={site},limit={n},offset={m},sortBy=POSTING_DATES_DESC
        -> {items:[{requisitionList:[{Id,Title,PostedDate,PrimaryLocation,
            secondaryLocations,ShortDescriptionStr,ExternalQualificationsStr,
            ExternalResponsibilitiesStr,...}], TotalJobsCount}]}
Detail: GET {host}/hcmRestApi/resources/latest/recruitingCEJobRequisitionDetails
            ?expand=all&onlyData=true&finder=ById;Id="{id}",siteNumber={site}
        -> ExternalDescriptionStr (full HTML body)

No auth. Token format:  "host|siteNumber"  e.g. "jpmc.fa.oraclecloud.com|CX_1001".
"""
from __future__ import annotations

from .. import config
from .base import Connector, FetchError, RawJob, html_to_text, http_get, norm_space


def _location(item: dict) -> str:
    parts = [norm_space(item.get("PrimaryLocation"))]
    for sl in item.get("secondaryLocations") or []:
        if isinstance(sl, dict) and sl.get("Name"):
            parts.append(norm_space(sl["Name"]))
    seen, out = set(), []
    for p in parts:
        if p and p not in seen:
            seen.add(p); out.append(p)
    return ", ".join(out)


def _list_description(item: dict) -> str:
    chunks = []
    for key, label in (("ShortDescriptionStr", ""), ("ExternalResponsibilitiesStr", "Responsibilities"),
                       ("ExternalQualificationsStr", "Qualifications")):
        t = html_to_text(item.get(key) or "")
        if t:
            chunks.append((f"{label}\n" if label else "") + t)
    return "\n\n".join(chunks)


class OracleConnector(Connector):
    ats_type = "oracle"
    supports_detail = True

    @staticmethod
    def _parts(token: str) -> tuple[str, str]:
        bits = token.split("|")
        if len(bits) != 2 or not all(bits):
            raise FetchError(f"oracle token must be 'host|siteNumber', got {token!r}")
        return bits[0], bits[1]

    def _list_url(self, host: str, site: str, limit: int, offset: int) -> str:
        # Build the string manually so the finder's ';' and ',' stay literal.
        return (f"https://{host}/hcmRestApi/resources/latest/recruitingCEJobRequisitions"
                f"?onlyData=true&expand=requisitionList.secondaryLocations"
                f"&finder=findReqs;siteNumber={site},limit={limit},offset={offset},"
                f"sortBy=POSTING_DATES_DESC")

    def fetch(self, token: str) -> list[RawJob]:
        host, site = self._parts(token)
        jobs: list[RawJob] = []
        offset, page = 0, 50
        while offset < config.ORACLE_MAX_JOBS:
            data = http_get(self._list_url(host, site, page, offset)).json()
            items = data.get("items") or []
            rl = (items[0].get("requisitionList") if items else []) or []
            if not rl:
                break
            for it in rl:
                jid = str(it.get("Id") or "").strip()
                if not jid:
                    continue
                posted = (it.get("PostedDate") or "")[:10] or None
                jobs.append(RawJob(
                    job_id=jid,
                    title=norm_space(it.get("Title")),
                    location=_location(it),
                    posted_date=posted,
                    date_confidence="high" if posted else "low",
                    url=f"https://{host}/hcmUI/CandidateExperience/en/sites/{site}/job/{jid}/",
                    description=_list_description(it),
                    detail_ref=jid,
                    extra={"department": it.get("Department"), "jobFamily": it.get("JobFamily")},
                ))
            total = (items[0].get("TotalJobsCount") if items else 0) or 0
            offset += page
            if total and offset >= total:
                break
        return jobs

    def fetch_detail(self, token: str, detail_ref: str) -> dict:
        host, site = self._parts(token)
        url = (f"https://{host}/hcmRestApi/resources/latest/recruitingCEJobRequisitionDetails"
               f'?expand=all&onlyData=true&finder=ById;Id="{detail_ref}",siteNumber={site}')
        data = http_get(url).json()
        it = (data.get("items") or [{}])[0]
        out: dict = {}
        body = html_to_text(it.get("ExternalDescriptionStr") or "")
        quals = html_to_text(it.get("ExternalQualificationsStr") or "")
        full = "\n\n".join(c for c in (body, quals) if c)
        if full:
            out["description"] = full
        return out

    def probe(self, token: str) -> bool:
        try:
            host, site = self._parts(token)
            data = http_get(self._list_url(host, site, 1, 0)).json()
            items = data.get("items") or []
            return bool(items and (items[0].get("requisitionList") or items[0].get("TotalJobsCount")))
        except Exception:  # noqa: BLE001
            return False

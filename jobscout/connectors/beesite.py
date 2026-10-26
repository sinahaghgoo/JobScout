"""BeeSite Global Job Board (milch & zucker) connector.

Modern BeeSite portals (Mercedes-Benz, Porsche) render their job search from
a public JSON API on the vendor's cloud:

    GET https://<client>-beesite-production-gjb.app.beesite.de/search
        ?data=<url-encoded JSON>
    JSON: {"LanguageCode": "EN",
           "SearchParameters": {"FirstItem": 1, "CountItem": 500,
             "Sort": [{"Criterion": "PublicationStartDate",
                       "Direction": "DESC"}],
             "MatchedObjectDescriptor": [...fields...]},
           "SearchCriteria": [{"CriterionName": "PositionLocation.CountryCode",
                               "CriterionValue": ["DE", "AT", ...]}]}
    -> {"SearchResult": {"SearchResultCountAll": N,
                         "SearchResultItems": [{"MatchedObjectDescriptor":
        {ID, PositionID, PositionTitle, PositionURI (absolute),
         PositionLocation[], PublicationStartDate, CareerLevel, ...}}]}}

PositionURI links straight to the portal's job ad; PublicationStartDate is
authoritative. (The *legacy* BeeSite UI — apply.lufthansagroup.careers —
has no such API and stays on the browser tier.)

Token format:
  * "<gjb-host>"                     -> all postings, newest first
  * "<gjb-host>|AT,DE,GB"            -> server-side country filter (ISO-3166)
"""
from __future__ import annotations

import json
import time
from urllib.parse import quote

from .. import config
from .base import Connector, FetchError, RawJob, html_to_text, norm_space

_PAGE_SIZE = 500
_MAX_PAGES = 8

_FIELDS = ["ID", "PositionID", "PositionTitle", "PositionURI",
           "PositionLocation.CityName", "PositionLocation.CountryCode",
           "PositionLocation.Country", "PublicationStartDate",
           "CareerLevel.Name", "JobCategory.Name",
           "PositionOfferingType.Name"]


def _split_token(token: str) -> tuple[str, list[str]]:
    raw = (token or "").strip()
    host, _, ccs = raw.partition("|")
    codes = [c.strip().upper() for c in ccs.split(",") if c.strip()]
    return host.strip().strip("/"), codes


def _get(host: str, first: int, country_ids: list[str]) -> dict:
    import requests
    # Sort on the unique numeric item ID: PublicationStartDate has massive
    # same-day ties, which makes page boundaries unstable (items repeat or
    # vanish between pages). ID DESC is stable AND roughly newest-first.
    body: dict = {"LanguageCode": "EN",
                  "SearchParameters": {
                      "FirstItem": first, "CountItem": _PAGE_SIZE,
                      "Sort": [{"Criterion": "ID", "Direction": "DESC"}],
                      "MatchedObjectDescriptor": _FIELDS}}
    if country_ids:
        # The API ignores ISO codes; it filters on internal numeric Country
        # ids (harvested per host from an unfiltered page — see fetch()).
        body["SearchCriteria"] = [{
            "CriterionName": "PositionLocation.Country",
            "CriterionValue": country_ids}]
    url = f"https://{host}/search?data=" + quote(json.dumps(body))
    last: Exception | None = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            time.sleep(config.REQUEST_DELAY_SECONDS)
            r = requests.get(url, headers={"User-Agent": config.BROWSER_USER_AGENT,
                                           "Accept": "application/json"},
                             timeout=config.HTTP_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            last = FetchError(f"HTTP {r.status_code} from BeeSite GJB {host}")
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(1.0 * attempt)
    raise FetchError(f"BeeSite GJB {host} failed after "
                     f"{config.MAX_RETRIES} tries ({last})")


def _location(d: dict) -> str:
    locs = d.get("PositionLocation") or []
    if isinstance(locs, dict):
        locs = [locs]
    parts = []
    for loc in locs:
        city = norm_space(str(loc.get("CityName") or ""))
        # "Country" is an internal numeric id on these hosts — the ISO
        # CountryCode is the human-usable value.
        country = norm_space(str(loc.get("CountryCode") or ""))
        s = ", ".join(x for x in (city, country) if x)
        if s and s not in parts:
            parts.append(s)
    return "; ".join(parts)


def _harvest_country_ids(sr: dict, wanted: list[str]) -> list[str]:
    """Map ISO codes -> the host's internal numeric Country ids, using the
    codes/ids that ride along on an unfiltered result page."""
    found: dict[str, str] = {}
    for it in sr.get("SearchResultItems") or []:
        d = (it or {}).get("MatchedObjectDescriptor") or {}
        locs = d.get("PositionLocation") or []
        if isinstance(locs, dict):
            locs = [locs]
        for loc in locs:
            cc = str(loc.get("CountryCode") or "").upper()
            cid = str(loc.get("Country") or "")
            if cc and cid.isdigit():
                found[cc] = cid
    return [found[cc] for cc in wanted if cc in found]


class BeesiteConnector(Connector):
    ats_type = "beesite"
    supports_detail = False  # list API carries no full description

    def fetch(self, token: str) -> list[RawJob]:
        host, codes = _split_token(token)
        if not host:
            raise FetchError("beesite token must be the GJB API host, e.g. "
                             "'mercedes-benz-beesite-production-gjb.app.beesite.de'")
        jobs: list[RawJob] = []
        seen: set[str] = set()
        first = 1
        country_ids: list[str] = []
        for page_no in range(_MAX_PAGES):
            sr = (_get(host, first, country_ids).get("SearchResult")) or {}
            items = sr.get("SearchResultItems") or []
            if page_no == 0 and codes:
                # First page ran unfiltered: learn this host's numeric ids for
                # the wanted countries, then restart filtered (server-side).
                country_ids = _harvest_country_ids(sr, codes)
                if country_ids:
                    first = 1
                    continue
            if not items:
                break
            for it in items:
                d = (it or {}).get("MatchedObjectDescriptor") or {}
                jid = str(d.get("PositionID") or d.get("ID") or "").strip()
                title = norm_space(str(d.get("PositionTitle") or ""))
                if not jid or not title:
                    continue
                if jid in seen:
                    # Multi-location ads repeat per city under one PositionID —
                    # merge the extra location instead of dropping it, so a
                    # Berlin+Madrid ad isn't mistaken for Madrid-only.
                    loc = _location(d)
                    for j in jobs:
                        if j.job_id == jid:
                            if loc and loc not in j.location:
                                j.location = (f"{j.location}; {loc}"
                                              if j.location else loc)
                            break
                    continue
                seen.add(jid)
                posted = (str(d.get("PublicationStartDate") or "")[:10]) or None
                extra_bits = {}
                for k, out in (("CareerLevel", "career_level"),
                               ("JobCategory", "category"),
                               ("PositionOfferingType", "offering_type")):
                    v = d.get(k)
                    if isinstance(v, list) and v:
                        v = v[0]
                    if isinstance(v, dict):
                        v = v.get("Name")
                    if v:
                        extra_bits[out] = v
                jobs.append(RawJob(
                    job_id=jid,
                    title=title,
                    location=_location(d),
                    posted_date=posted,
                    date_confidence="high" if posted else "low",
                    url=str(d.get("PositionURI") or "").strip(),
                    description=html_to_text(str(
                        d.get("PositionFormattedDescription", {}).get("Content", "")
                        if isinstance(d.get("PositionFormattedDescription"), dict)
                        else "")),
                    extra=extra_bits,
                ))
            total = int(sr.get("SearchResultCountAll") or 0)
            if len(items) < _PAGE_SIZE or (total and len(seen) >= total):
                break
            first += _PAGE_SIZE
        return jobs

    def probe(self, token: str) -> bool:
        host, codes = _split_token(token)
        if not host:
            return False
        try:
            return "SearchResult" in _get(host, 1, codes)
        except Exception:  # noqa: BLE001
            return False

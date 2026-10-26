"""Shared connector plumbing: a normalized job record, an HTTP helper with
retries/throttling, and the Connector base class.

Every connector turns one company's ATS feed into a list of `RawJob` with the
same fields, so everything downstream is ATS-agnostic.
"""
from __future__ import annotations

import html
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

from .. import config

_TAG_RE = re.compile(r"<[^>]+>")
# Horizontal whitespace incl. NBSP (\xa0), the various Unicode spaces, the narrow
# no-break space DACH ATS feeds embed in titles, and the ideographic space ->
# collapse to a single ASCII space. (Non-raw string so the escapes are real chars.)
_WS_RE = re.compile("[ \t\xa0  -   　]+")
# Zero-width characters (zero-width space/joiner/non-joiner, BOM) -> drop entirely.
_ZW_RE = re.compile("[​‌‍﻿]")
_NL_RE = re.compile(r"\n{3,}")


def norm_space(raw: Optional[str]) -> str:
    """Normalize plain text: unescape entities, drop zero-width characters, and
    fold every kind of horizontal whitespace (including the NBSP that DACH ATS
    feeds embed in titles/locations) into a single ASCII space."""
    if not raw:
        return ""
    text = html.unescape(raw)
    text = _ZW_RE.sub("", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def html_to_text(raw: Optional[str]) -> str:
    """Cheap HTML -> readable text. Good enough for matching + previews."""
    if not raw:
        return ""
    text = html.unescape(raw)
    text = re.sub(r"<\s*(br|/p|/div|/li)\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<\s*li[^>]*>", "• ", text, flags=re.I)
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    text = _ZW_RE.sub("", text)
    text = _WS_RE.sub(" ", text)
    text = _NL_RE.sub("\n\n", text)
    return text.strip()


# --- West-Europe location filter (shared by big global boards) ---------------
# The product targets "everything in West Europe (EU or UK)". For platforms that
# return a whole global board in one shot (SAP RMK, big Workday/Oracle boards),
# connectors can pre-filter to this region so the relevant roles aren't crowded
# out of the per-run cap by thousands of US/APAC/retail postings. Matching is on
# country names + ISO-3166 alpha-2/alpha-3 codes + region words, NOT city names
# (cities collide with US places — Vienna VA, Manchester NH, …); these feeds carry
# a country in the location string, so the country signal is enough and safe.
_WE_COUNTRY_WORDS = {
    "austria", "germany", "deutschland", "switzerland", "schweiz", "suisse",
    "ireland", "united kingdom", "great britain", "england", "scotland", "wales",
    "northern ireland", "u.k.", "netherlands", "nederland", "holland", "belgium",
    "belgië", "belgique", "luxembourg", "france", "spain", "españa", "portugal",
    "italy", "italia", "denmark", "danmark", "sweden", "sverige", "norway",
    "norge", "finland", "suomi", "poland", "polska", "czech", "czechia",
    "hungary", "romania", "greece", "iceland", "liechtenstein",
    # region catch-alls
    "europe", "european union", "emea", "remote - eu", "remote eu",
    "remote, europe", "remote - europe", "pan-eu", "eu remote",
}
_WE_CODES = {
    "AT", "DE", "CH", "IE", "GB", "UK", "NL", "BE", "LU", "FR", "ES", "PT",
    "IT", "DK", "SE", "NO", "FI", "PL", "CZ", "HU", "RO", "GR", "IS", "LI",
    "AUT", "DEU", "CHE", "IRL", "GBR", "NLD", "BEL", "LUX", "FRA", "ESP",
    "PRT", "ITA", "DNK", "SWE", "NOR", "FIN", "POL", "CZE", "HUN", "ROU", "GRC",
}
_TOKEN_SPLIT_RE = re.compile(r"[^A-Za-z]+")


def is_west_europe(location: Optional[str]) -> bool:
    """True if a feed location string clearly sits in West Europe (EU or UK).

    Conservative: matches country names / region words, plus ISO country codes
    that appear as a WHOLE, UPPER-CASE token (e.g. "Munich, DE" / "Cork, IRL").
    Requiring a whole upper-case token avoids two false-positive traps: substrings
    inside longer words ("Cai**ro**" → RO) and lowercase English words ("at", "is",
    "no", "it"). City names are deliberately not matched (they collide with US
    places like Vienna VA); these feeds carry a country, so that's enough."""
    if not location:
        return False
    low = location.lower()
    if any(w in low for w in _WE_COUNTRY_WORDS):
        return True
    for tok in _TOKEN_SPLIT_RE.split(location):
        if tok and tok.isupper() and tok in _WE_CODES:
            return True
    return False


@dataclass
class RawJob:
    job_id: str
    title: str
    location: str = ""
    posted_date: Optional[str] = None  # YYYY-MM-DD
    url: str = ""
    description: str = ""
    detail_ref: Optional[str] = None  # opaque ref for a follow-up detail fetch
    date_confidence: str = "high"     # "high" (from feed) | "low" (approx/unknown)
    extra: dict = field(default_factory=dict)


class FetchError(RuntimeError):
    pass


_session = requests.Session()
_session.headers.update({"User-Agent": config.USER_AGENT})

# Some career sites (SAP CSB job pages behind Akamai, org portals) only serve
# real content to browser-like clients. Connectors pass these for such hosts.
BROWSER_HEADERS = {
    "User-Agent": config.BROWSER_USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
}


def http_get(url: str, *, accept: str = "application/json", method: str = "GET",
             json_body: Optional[dict] = None,
             headers: Optional[dict] = None) -> requests.Response:
    """GET/POST with polite delay, retries and clear errors."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            time.sleep(config.REQUEST_DELAY_SECONDS)
            hdrs = {"Accept": accept, **(headers or {})}
            if method == "POST":
                resp = _session.post(url, json=json_body, timeout=config.HTTP_TIMEOUT,
                                     headers=hdrs)
            else:
                resp = _session.get(url, timeout=config.HTTP_TIMEOUT,
                                    headers=hdrs)
            if resp.status_code == 429:
                # rate limited (Personio does this): back off and retry
                time.sleep(2.0 * attempt)
                last_exc = FetchError(f"429 rate limited at {url}")
                continue
            resp.raise_for_status()
            return resp
        except requests.HTTPError as e:
            # 404 = not on this ATS; don't waste retries
            if e.response is not None and e.response.status_code == 404:
                raise FetchError(f"404 not found: {url}") from e
            last_exc = e
            time.sleep(1.0 * attempt)
        except requests.RequestException as e:
            last_exc = e
            time.sleep(1.0 * attempt)
    raise FetchError(f"GET failed after {config.MAX_RETRIES} tries: {url} ({last_exc})")


class Connector:
    """Base class. Subclasses set `ats_type` and implement `fetch`.

    Some platforms (Workday) only return title/location/relative-date in their
    list endpoint; the full description and authoritative date live on a per-job
    detail endpoint. Such connectors set `supports_detail = True` and implement
    `fetch_detail`, which the pipeline calls only for the few roles that survive
    the prefilter — keeping cost bounded regardless of board size.
    """

    ats_type: str = "base"
    supports_detail: bool = False

    def fetch(self, token: str) -> list[RawJob]:  # pragma: no cover - interface
        raise NotImplementedError

    def fetch_detail(self, token: str, detail_ref: str) -> dict:
        """Return any of {description, posted_date, url, location} for one job."""
        return {}

    def probe(self, token: str) -> bool:
        """Cheap check used by detection: does this token resolve on this ATS?"""
        try:
            jobs = self.fetch(token)
            return jobs is not None
        except FetchError:
            return False

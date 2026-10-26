"""Tier-2: the careers-page resolver.

Most large employers don't expose a slug-guessable feed — they sit on Workday,
SuccessFactors, Oracle, SmartRecruiters, etc. behind a vanity careers domain. But
the ATS + tenant token is almost always discoverable from the careers page itself
(a redirect to `*.myworkdayjobs.com`, a `successfactors`/`oraclecloud` URL in the
HTML, an embedded board slug, …).

`resolve()` takes a company (optionally a known careers URL), fetches a few
candidate pages, extracts the ATS signature + token, and — crucially — **probe-
verifies every feed token against the real connector before returning it**, so it
can never hand the pipeline a broken token. Platforms that have no server feed
(Cornerstone, iCIMS, Avature, SAP-hosted SuccessFactors, …) are returned with
`tier="browser"` so Tier-3 can render them.

Pure HTTP — no browser needed for the resolution itself.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import requests

from . import config
from .connectors.registry import probe as _probe

_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}

# --- signature regexes (case-insensitive) ------------------------------------
_RE_WORKDAY = re.compile(r"https?://([a-z0-9][a-z0-9-]*)\.(wd\d+)\.myworkdayjobs\.com"
                         r"(?:/wday/cxs/[a-z0-9-]+)?((?:/[A-Za-z0-9_%-]+)*)", re.I)
_RE_ORACLE = re.compile(r"https?://([a-z0-9.-]*oraclecloud\.com)"
                        r"[^\"'\\s]*?/sites/([A-Za-z0-9_]+)", re.I)
_RE_ORACLE_HOST = re.compile(r"https?://([a-z0-9-]+\.fa\.[a-z0-9]*oraclecloud\.com)", re.I)
_RE_SF_SAP = re.compile(r"https?://(career\d*\.successfactors\.[a-z.]+|[a-z0-9-]+\.sapsf\.[a-z.]+)", re.I)
_RE_GREENHOUSE = re.compile(r"(?:boards|job-boards)\.greenhouse\.io/(?:embed/job_board\?for=)?([a-z0-9_]+)", re.I)
_RE_LEVER = re.compile(r"jobs\.lever\.co/([a-z0-9-]+)", re.I)
_RE_ASHBY = re.compile(r"jobs\.ashbyhq\.com/([a-z0-9-]+)", re.I)
_RE_SR = re.compile(r"(?:careers|jobs)\.smartrecruiters\.com/([A-Za-z0-9-]+)"
                    r"|api\.smartrecruiters\.com/v1/companies/([A-Za-z0-9-]+)", re.I)
_RE_PERSONIO = re.compile(r"([a-z0-9-]+)\.jobs\.personio\.(de|com)", re.I)
_RE_RECRUITEE = re.compile(r"([a-z0-9-]+)\.recruitee\.com", re.I)
_RE_WORKABLE = re.compile(r"apply\.workable\.com/([a-z0-9-]+)|([a-z0-9-]+)\.workable\.com", re.I)
# browser-tier platforms (no clean server feed)
_RE_CSOD = re.compile(r"https?://([a-z0-9-]+\.csod\.com)", re.I)
_RE_ICIMS = re.compile(r"https?://([a-z0-9-]+\.icims\.com)", re.I)
_RE_AVATURE = re.compile(r"https?://([a-z0-9-]+\.avature\.net)", re.I)
_RE_EIGHTFOLD = re.compile(r"https?://([a-z0-9-]+\.eightfold\.ai)", re.I)
_RE_SOFTGARDEN = re.compile(r"([a-z0-9-]+)\.softgarden\.(io|de)", re.I)

_LOCALE = re.compile(r"^[a-z]{2}([-_][A-Za-z]{2})?$")
_SUFFIXES = {"ag", "gmbh", "se", "inc", "group", "international", "holding", "plc",
             "nv", "sa", "the", "co", "kg", "ab", "as", "oyj", "spa", "list"}


@dataclass
class Resolution:
    ats_type: Optional[str]
    token: Optional[str]
    tier: str            # "feed" | "browser"
    source_url: str = ""
    note: str = ""


def _workday_site(path: str) -> Optional[str]:
    """Pick the Workday site name from the path after the host, skipping locales."""
    for seg in [s for s in path.split("/") if s]:
        if _LOCALE.match(seg) or seg.lower() in ("wday", "cxs", "jobs"):
            continue
        return seg
    return None


def candidate_feed(text: str) -> list[tuple[str, str]]:
    """Every (ats_type, token) feed candidate found in a URL/HTML blob."""
    out: list[tuple[str, str]] = []

    def add(a, t):
        if t and (a, t) not in out:
            out.append((a, t))

    for m in _RE_WORKDAY.finditer(text):
        site = _workday_site(m.group(3) or "")
        if site:
            add("workday", f"{m.group(1)}|{m.group(2)}|{site}")
    for m in _RE_ORACLE.finditer(text):
        add("oracle", f"{m.group(1)}|{m.group(2)}")
    for m in _RE_GREENHOUSE.finditer(text):
        add("greenhouse", m.group(1))
    for m in _RE_LEVER.finditer(text):
        add("lever", m.group(1))
    for m in _RE_ASHBY.finditer(text):
        add("ashby", m.group(1))
    for m in _RE_SR.finditer(text):
        add("smartrecruiters", m.group(1) or m.group(2))
    for m in _RE_PERSONIO.finditer(text):
        add("personio", f"{m.group(1)}|{m.group(2).lower()}")
    for m in _RE_RECRUITEE.finditer(text):
        add("recruitee", m.group(1))
    for m in _RE_WORKABLE.finditer(text):
        add("workable", m.group(1) or m.group(2))
    return out


def candidate_browser(text: str) -> Optional[tuple[str, str]]:
    """First (platform, host) needing the browser tier, if any."""
    for rx, name in ((_RE_CSOD, "cornerstone"), (_RE_ICIMS, "icims"),
                     (_RE_AVATURE, "avature"), (_RE_EIGHTFOLD, "eightfold"),
                     (_RE_SF_SAP, "successfactors-sap"), (_RE_SOFTGARDEN, "softgarden")):
        m = rx.search(text)
        if m:
            return name, m.group(1)
    return None


def _slug(name: str) -> str:
    n = re.split(r"[(/|]", name.lower())[0]
    n = re.sub(r"[&.,'’]", " ", n)
    words = [w for w in re.sub(r"[^a-z0-9 -]", " ", n).split() if w and w not in _SUFFIXES]
    return "".join(words) or re.sub(r"[^a-z0-9]", "", name.lower())


def candidate_urls(name: str, careers_url: Optional[str]) -> list[str]:
    urls: list[str] = []
    if careers_url:
        urls.append(careers_url if careers_url.startswith("http") else "https://" + careers_url)
    s = _slug(name)
    if s:
        for host in (f"careers.{s}.com", f"jobs.{s}.com", f"{s}.com/careers",
                     f"{s}.com/en/careers", f"www.{s}.com/careers", f"{s}.com"):
            urls.append("https://" + host)
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u); out.append(u)
    return out


def _fetch(url: str) -> Optional[tuple[str, str]]:
    """Return (final_url, html) or None."""
    try:
        # short connect timeout so dead domain guesses fail fast
        r = requests.get(url, headers=_UA, timeout=(6, config.HTTP_TIMEOUT), allow_redirects=True)
        if r.status_code < 400 and r.text:
            return str(r.url), r.text
    except requests.RequestException:
        return None
    return None


def resolve(name: str, careers_url: Optional[str] = None, *, guess: bool = True,
            verify: bool = True) -> Optional[Resolution]:
    """Resolve a company to a connector (ats, token) by reading its careers page.

    Feed tokens are probe-verified before being returned. Browser-tier platforms
    are returned without verification (Tier-3 handles them)."""
    urls = candidate_urls(name, careers_url) if guess else (
        [careers_url] if careers_url else [])
    browser_hit: Optional[Resolution] = None
    for url in urls:
        got = _fetch(url)
        if not got:
            continue
        final_url, html = got
        blob = final_url + "\n" + html
        for ats, token in candidate_feed(blob):
            if not verify or _probe(ats, token):
                return Resolution(ats, token, "feed", final_url, note="resolved+verified")
        # SuccessFactors custom-domain CSB: the feed token is the careers host
        # itself (not regex-guessable), so probe it directly. SAP-hosted shared
        # hosts (career2.successfactors.eu, *.sapsf.eu) fail this and fall to browser.
        if "successfactors" in blob.lower() and not _RE_SF_SAP.search(final_url):
            host = urlparse(final_url).netloc
            if host and ((not verify) or _probe("successfactors", host)):
                return Resolution("successfactors", host, "feed", final_url, "resolved+verified")
        if browser_hit is None:
            b = candidate_browser(blob)
            if b:
                browser_hit = Resolution(None, final_url, "browser", final_url,
                                         note=f"renders client-side ({b[0]})")
    return browser_hit

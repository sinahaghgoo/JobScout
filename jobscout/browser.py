"""Tier-3: careers pages with no structured feed — fetch, render, LLM-extract.

The last-resort tier for career sites that render client-side or sit on
platforms without a public feed (Cornerstone, iCIMS, Avature, SAP-hosted
SuccessFactors, bespoke SPAs). Three independently useful pieces:

  fetch_page(url)          -> (final_url, html)   HTTP first (with optional
                              curl_cffi browser impersonation for bot-walled
                              sites), else None
  render_html(url)         -> rendered HTML via headless Chromium (Playwright;
                              optional dependency)
  extract_jobs(html, url)  -> list[RawJob]  (Claude reads the page like a human)

`fetch_via_browser` ties them together and adds a CONTENT-HASH CACHE: the
cleaned page text is hashed, and when it hasn't changed since the last run the
cached extraction is reused — zero LLM cost for unchanged pages, which is the
common case day to day.

Extraction runs on the cheap model (config.EXTRACT_MODEL): reading postings
off a page is mechanical; the expensive judgment model is reserved for
matching. Playwright is optional — without it, server-rendered pages still
work over plain HTTP and SPAs simply stay empty until it's installed
(`pip install playwright && playwright install chromium`).
"""
from __future__ import annotations

import hashlib
import html as _htmlmod
import json
import os
import re
import sys
from typing import Optional

from . import config, db
from .connectors.base import RawJob, html_to_text, norm_space

_SCRIPT = re.compile(r"<(script|style|noscript|svg)\b[^>]*>.*?</\1>", re.I | re.S)
# Boilerplate blocks that never contain the job list but often push it past the
# text budget (cookie walls, mega-menus, footers).
_BOILERPLATE = re.compile(r"<(header|nav|footer)\b[^>]*>.*?</\1>", re.I | re.S)
_ANCHOR = re.compile(r'<a\b[^>]*\bhref=["\']([^"\']+)["\'][^>]*>(.*?)</a>', re.I | re.S)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\xa0]+")
_NL = re.compile(r"\n\s*\n\s*\n+")

_HTTP_HEADERS = {
    "User-Agent": config.BROWSER_USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
}


def browser_available() -> bool:
    try:
        import playwright  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _curl_cffi_get(url: str) -> Optional[tuple[str, str]]:
    """Fetch with browser TLS impersonation — beats simple bot walls (Avature,
    Cloudflare-fronted portals). Optional dependency; None if unavailable."""
    try:
        from curl_cffi import requests as creq
    except Exception:  # noqa: BLE001
        return None
    try:
        r = creq.get(url, impersonate="chrome", timeout=config.HTTP_TIMEOUT,
                     allow_redirects=True)
        if r.status_code < 400 and r.text:
            return str(r.url), r.text
    except Exception:  # noqa: BLE001
        return None
    return None


def fetch_page(url: str) -> Optional[tuple[str, str]]:
    """Plain HTTP GET of a careers page — (final_url, html) or None.

    Works for server-rendered pages (many org/Taleo/bespoke portals) with no
    browser at all. Bot-walled sites get a second try with curl_cffi browser
    impersonation. Client-rendered SPAs return a shell here and fall through
    to the Playwright renderer in `fetch_via_browser`."""
    import requests
    try:
        r = requests.get(url, headers=_HTTP_HEADERS,
                         timeout=(8, config.HTTP_TIMEOUT), allow_redirects=True)
        if r.status_code < 400 and r.text:
            return str(r.url), r.text
    except requests.RequestException:
        pass
    return _curl_cffi_get(url)


# Hosts whose job list arrives via a late XHR *after* the DOM looks complete
# (nav-heavy shells defeat the generic "link-sparse" heuristic below). Value =
# a CSS selector that only exists once real postings have landed; the renderer
# waits for it (bounded) before snapshotting. Selector present = instant, so
# healthy pages pay nothing.
_WAIT_HINTS: dict[str, str] = {
    "apply.lufthansagroup.careers": "a[href*='ac=jobad']",   # BeeSite portal
    "jobs.porsche.com": "a[href*='ac=jobad']",               # BeeSite portal
    "oebb.csod.com": "a[href*='/requisition/']",             # Cornerstone
    "jobs.booking.com": "a[href*='/jobs/']",                 # Jibe/iCIMS SPA
    "careers.nike.com": "a[href*='/job/']",                  # bespoke SPA
    "www.ibm.com": "a[href*='JobDetail?jobId=']",            # Avature-backed SPA
}


def _wait_hint(url: str) -> Optional[str]:
    from urllib.parse import urlparse
    return _WAIT_HINTS.get(urlparse(url).netloc.lower())


_HINT_HREF = re.compile(r"href\*='([^']+)'")


def _hint_present(content: str, hint: str) -> bool:
    """Cheap textual check that the hinted job-link pattern is in the HTML."""
    m = _HINT_HREF.search(hint)
    return (m.group(1) in content) if m else True


def render_html(url: str, *, wait_ms: int = 3000, timeout_ms: int = 35000) -> str:
    """Render a page with headless Chromium and return its HTML after JS runs.

    Dismisses common cookie banners and scrolls once so lazy-loaded job lists
    populate before the snapshot."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Tier-3 needs Playwright. Install it with:\n"
            "    pip install playwright && playwright install chromium"
        ) from e
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_context(
                user_agent=config.BROWSER_USER_AGENT,
                locale="en-US",
            ).new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            try:
                page.wait_for_load_state("networkidle", timeout=12000)
            except Exception:  # noqa: BLE001 — busy pages never go idle; fine
                pass
            # Best-effort cookie-banner dismissal (OneTrust and generic,
            # EN/DE/FR). Some walls (EU institutions, Avature) render the
            # accept control as an <a>, not a <button>.
            for sel in ("#onetrust-accept-btn-handler",
                        "button:has-text('Accept all')",
                        "button:has-text('Accept All')",
                        "button:has-text('Alle akzeptieren')",
                        "button:has-text('I agree')",
                        "button:has-text('J’accepte')",
                        "button:has-text('Tout accepter')",
                        "button:has-text('Agree to cookies')",
                        "a:has-text('Accept all cookies')",
                        "a:has-text('I accept')",
                        "button:has-text('I accept')"):
                try:
                    page.click(sel, timeout=800)
                    break
                except Exception:  # noqa: BLE001
                    continue
            hint = _wait_hint(url)
            if hint:  # job list loads via late XHR on this host — wait for it
                try:
                    page.wait_for_selector(hint, timeout=15000)
                except Exception:  # noqa: BLE001 — fall back to timed waits
                    pass
            page.wait_for_timeout(wait_ms)
            try:  # one scroll for lazy lists
                page.mouse.wheel(0, 4000)
                page.wait_for_timeout(1200)
            except Exception:  # noqa: BLE001
                pass
            content = page.content()
            # Slow SPAs may still be loading their result list; if the page
            # looks link-sparse — or is a Taleo search whose job rows
            # (jobdetail.ftl links) haven't landed yet — give it one more beat.
            def _incomplete(c: str) -> bool:
                if clean_html_for_llm(c).count("(url:") < 12:
                    return True
                if hint and not _hint_present(c, hint):
                    return True
                return "jobsearch.ftl" in c and "jobdetail.ftl" not in c
            if _incomplete(content):
                page.wait_for_timeout(7000)
                content = page.content()
            if _incomplete(content):
                # Some portals (legacy SAP SuccessFactors "career?company=" UI)
                # only list jobs after the search form is submitted. Submit an
                # empty search — on such pages that means "show everything".
                for sel in ('button:has-text("Search")',
                            'input[type="submit"]',
                            'a:has-text("Search for Openings")'):
                    try:
                        page.click(sel, timeout=1200)
                        page.wait_for_timeout(6000)
                        content = page.content()
                        break
                    except Exception:  # noqa: BLE001
                        continue
            # Taleo careersection lists are stateful (no page-2 URL) and some
            # boards sort by *closing* date, so new postings start on later
            # pages. Walk 'Next' a couple of times and concatenate snapshots —
            # the extractor dedupes, and the content-hash cache still applies.
            if "jobsearch.ftl" in url and "jobdetail.ftl" in content:
                for _ in range(2):
                    try:
                        before = len(re.findall(r"jobdetail\.ftl\?job=", content))
                        page.click("a:has-text('Next')", timeout=1500)
                        page.wait_for_timeout(4000)
                        extra = page.content()
                        if re.findall(r"jobdetail\.ftl\?job=", extra) and \
                                len(re.findall(r"jobdetail\.ftl\?job=", content + extra)) > before:
                            content += extra
                        else:
                            break
                    except Exception:  # noqa: BLE001
                        break
            return content
        finally:
            browser.close()


def clean_html_for_llm(html: str, *, limit: int = 30000) -> str:
    """Strip a page down to the text + links an LLM needs to spot job postings."""
    text = _SCRIPT.sub(" ", html or "")
    text = _BOILERPLATE.sub(" ", text)
    # keep link targets inline (in parens) so the model can return real apply
    # URLs. Must run before tag-stripping, and use parens — angle brackets would
    # be eaten by the tag stripper below. Hrefs are normalized here (entity
    # unescape + percent-encode spaces/parens) so the model can copy a URL
    # that exactly matches the guard set in `_to_rawjobs` — otherwise
    # query-param and Taleo-style URLs silently lose their apply links.
    text = _ANCHOR.sub(
        lambda m: f" {_TAG.sub('', m.group(2))} "
                  f"(url: {_norm_href(m.group(1))}) ", text)
    text = _TAG.sub(" ", text)
    # Unescape entities instead of blanking them: keeps "R&amp;D" as "R&D" in
    # titles and leaves the just-inlined hrefs intact.
    text = _htmlmod.unescape(text)
    text = _WS.sub(" ", text)
    text = _NL.sub("\n\n", text)
    return text.strip()[:limit]


def _looks_extractable(cleaned: str) -> bool:
    """Does this cleaned page plausibly hold a job listing? Avoids spending an
    LLM call on an empty SPA shell or a bare search form."""
    if len(cleaned) < 800:
        return False
    return cleaned.count("(url:") >= 3


_SYS = (
    "You extract job postings from a careers page's text. Return ONLY a JSON array "
    "(no prose). Each element: {\"title\": str, \"location\": str, \"url\": str, "
    "\"posted_date\": \"YYYY-MM-DD or empty\", \"description\": \"short, if present\"}. "
    "Include ONLY real individual job postings actually listed on this page (not nav "
    "links, categories, or marketing). Use the (url: ...) shown next to each title; "
    "if a posting has no URL use \"\". If there are no real postings, return []."
)


def _abs_url(u: str, base: str) -> str:
    if not u:
        return ""
    if u.startswith("http"):
        return u
    from urllib.parse import urljoin
    return urljoin(base, u)


def _norm_href(u: str) -> str:
    """Entity-unescape + percent-encode the characters that break the
    '(url: ...)' inline wrapper and URL comparison (spaces, parentheses —
    Taleo puts 'job=2026/0271 (266824)' style refs in hrefs). '%' stays safe
    so already-encoded URLs are not double-encoded."""
    from urllib.parse import quote
    return quote(_htmlmod.unescape(u), safe=":/?&=%+#,;@!$*-._~'")


def _page_hrefs(html: str, base_url: str) -> set[str]:
    """Every absolute link target actually present on the page (hrefs are
    normalized exactly like `clean_html_for_llm` shows them to the model)."""
    out: set[str] = set()
    for m in _ANCHOR.finditer(html or ""):
        u = _abs_url(_norm_href(m.group(1).strip()), base_url)
        if u.startswith("http"):
            out.add(u.rstrip("/"))
    return out


# One shared ceiling for how many postings a single extraction may return.
# Used by BOTH the fresh-extraction path and the cache-rematerialization path —
# they must agree, or a cached page would silently lose postings on re-runs.
MAX_EXTRACT_JOBS = 100

# Output budget for one extraction. Must comfortably fit MAX_EXTRACT_JOBS
# objects; see the note at the messages.create() call.
EXTRACT_MAX_TOKENS = int(os.environ.get("JOBSCOUT_EXTRACT_MAX_TOKENS", "16000"))


def _parse_job_array(raw: str, *, truncated: bool = False,
                     source: str = "") -> list:
    """Parse the model's JSON array, salvaging a truncated one.

    A board with many postings can exhaust the output budget mid-array. The old
    code required a closing bracket (`re.search(r"\[.*\]")`) and returned []
    when it was missing — so a 70-job board became an empty employer with no
    error anywhere. Recover every complete object instead, and say so."""
    start = raw.find("[")
    if start < 0:
        return []
    end = raw.rfind("]")
    if end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass                       # malformed: fall through and salvage
    # Salvage: keep whole `{...}` objects, ignoring braces inside strings.
    body, depth, obj_start, objs = raw[start + 1:], 0, None, []
    in_str = esc = False
    for i, ch in enumerate(body):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                objs.append(body[obj_start:i + 1])
                obj_start = None
    out = []
    for o in objs:
        try:
            out.append(json.loads(o))
        except json.JSONDecodeError:
            continue
    if out:
        sys.stderr.write(
            f"[jobscout] extraction output was {'truncated' if truncated else 'malformed'}"
            f" for {source or 'a careers page'}; recovered {len(out)} postings"
            " (raise JOBSCOUT_EXTRACT_MAX_TOKENS if this repeats)\n")
    return out


def extract_jobs(html: str, base_url: str = "", *, max_jobs: int = MAX_EXTRACT_JOBS,
                 clean_limit: int = 30000) -> list[RawJob]:
    """Use the cheap model to pull structured postings out of careers HTML.

    `clean_limit` must match the budget the caller used to hash/inspect the
    page (multi-page Taleo snapshots pass 90000) — otherwise postings past the
    default 30k-char trim would be collected by the renderer but never shown
    to the extractor."""
    from . import matching
    if not config.ANTHROPIC_API_KEY:
        raise matching.LLMUnavailable(
            "no Anthropic API key configured (key.txt / ANTHROPIC_API_KEY) — "
            "browser-tier companies need one to read their careers pages")
    if matching.llm_fatal_reason:
        # A permanent API failure was already diagnosed this run. Every
        # browser-tier company would hit the same wall; fail immediately with
        # the real reason instead of 26 identical stack traces.
        raise matching.LLMUnavailable(
            f"page extraction needs the Anthropic API — "
            f"{matching.llm_fatal_reason}")
    import anthropic
    cleaned = clean_html_for_llm(html, limit=clean_limit)
    if not cleaned:
        return []
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    try:
        resp = client.messages.create(
            model=config.EXTRACT_MODEL,
            # ~100-160 output tokens per posting × MAX_EXTRACT_JOBS(100). The
            # old 8000 was below what a full board needs, so big boards came
            # back truncated — and truncation used to read as "no jobs".
            max_tokens=EXTRACT_MAX_TOKENS,
            system=_SYS,
            messages=[{"role": "user", "content":
                       f"Careers page: {base_url}\n\nPAGE TEXT:\n{cleaned}\n\n"
                       "Return the JSON array."}],
        )
    except Exception as e:  # noqa: BLE001 — classify before it escapes
        # Extraction runs on EXTRACT_MODEL, not the scorer's model: name the
        # one that actually failed, and only let an ACCOUNT-level failure
        # (no credit, dead key) stop scoring too. A problem with the cheap
        # extraction model must not shut down Sonnet.
        reason = matching.classify_fatal(e, model=config.EXTRACT_MODEL)
        if reason:
            if matching.account_level_fatal(reason):
                matching.note_fatal(reason)
            raise matching.LLMUnavailable(
                f"page extraction needs the Anthropic API — {reason}") from e
        raise
    raw = "".join(getattr(b, "text", "") for b in resp.content)
    data = _parse_job_array(raw, truncated=getattr(resp, "stop_reason", "")
                            == "max_tokens", source=base_url)
    if not data:
        return []
    # Anti-hallucination guard: a returned URL is only trusted if that exact
    # link exists on the page. The model must copy hrefs, never invent them
    # (observed: plausible-but-404 '/en/jobs/<id>/' patterns).
    return _to_rawjobs(data, base_url, max_jobs, allowed_urls=_page_hrefs(html, base_url))


# Session/CSRF query params that some portals (legacy SuccessFactors) embed in
# job links. They differ on every visit, so they must not reach the stored URL
# or the job_id — otherwise every run re-keys the same posting (dupes + churn)
# and saved links expire with the session.
_SESSION_PARAMS = {"_s.crb", "jsessionid", "phpsessid"}
_JSESSIONID_PATH = re.compile(r";jsessionid=[^?#]*", re.I)


def _strip_session_params(url: str) -> str:
    if not url:
        return url
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
    url = _JSESSIONID_PATH.sub("", url)
    try:
        parts = urlsplit(url)
        q = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if k.lower() not in _SESSION_PARAMS]
        return urlunsplit((parts.scheme, parts.netloc, parts.path,
                           urlencode(q), parts.fragment))
    except ValueError:
        return url


def _valid_past_date(s: Optional[str]) -> Optional[str]:
    """A real, not-in-the-future YYYY-MM-DD — or None.

    LLM extraction sometimes returns application DEADLINES (future) or
    day/month-swapped strings ("2026-13-07") as posted_date; 66 such rows
    reached the DB on 2026-07-09. A calendar-valid check plus a today-ceiling
    kills both classes."""
    from datetime import date
    if not s:
        return None
    try:
        d = date.fromisoformat(str(s)[:10])
    except ValueError:
        return None
    return d.isoformat() if d <= date.today() else None


# Nav/UI artifacts that must never become "postings".
_JUNK_TITLE = re.compile(
    r"^(?:search(?:\s+jobs?)?|jobs?|careers?|vacancies|open\s+(?:positions|vacancies)"
    r"|login|log\s?in|sign\s?in|register|home|about(?:\s+us)?|faq|help|contact(?:\s+us)?"
    r"|apply(?:\s+now)?|read\s+more|learn\s+more|view\s+(?:all|more|job)s?|show\s+more"
    r"|next|previous|back|filter|sort|menu|cookie.*|privacy.*|imprint|impressum"
    r"|datenschutz|deutsch|english|français|save|share|print)$", re.I)


def _to_rawjobs(data: list, base_url: str, max_jobs: int = MAX_EXTRACT_JOBS,
                allowed_urls: Optional[set[str]] = None) -> list[RawJob]:
    # First pass: normalize fields, validate URLs against the page, and strip
    # session tokens so stored links/IDs are stable across runs.
    rows: list[dict] = []
    for d in data[:max_jobs]:
        if not isinstance(d, dict):
            continue
        title = norm_space(d.get("title"))
        if not title or _JUNK_TITLE.match(title):
            continue
        # A trailing raw ')' can only be residue of the "(url: ...)" wrapper —
        # real parens in hrefs are percent-encoded by _norm_href.
        url = _abs_url((d.get("url") or "").strip().rstrip(")"), base_url)
        if url and allowed_urls is not None and url.rstrip("/") not in allowed_urls:
            url = ""  # not a link that exists on the page -> don't trust it
        url = _strip_session_params(url)
        rows.append({"title": title, "url": url,
                     "location": norm_space(d.get("location")),
                     "posted": (d.get("posted_date") or "").strip() or None,
                     "description": html_to_text(d.get("description") or "")})

    # A URL-derived job_id is only trustworthy if it's unique on this page —
    # portals that link every posting to the same search/session URL would
    # otherwise collapse distinct jobs into one. Non-unique or missing URL
    # falls back to a title+location hash (which also collapses true dupes).
    def _url_jid(u: str) -> str:
        return u.rstrip("/").split("/")[-1] if u else ""

    from collections import Counter
    counts = Counter(_url_jid(r["url"]) for r in rows if r["url"])

    jobs: list[RawJob] = []
    seen: set[str] = set()
    for r in rows:
        jid = _url_jid(r["url"])
        if not jid or counts[jid] > 1:
            jid = hashlib.md5(
                f"{r['title']}|{r['location']}".encode()).hexdigest()[:16]
        if jid in seen:
            continue
        seen.add(jid)
        jobs.append(RawJob(
            job_id=jid,
            title=r["title"],
            location=r["location"],
            posted_date=_valid_past_date(r["posted"]),
            date_confidence="low",          # LLM-extracted, not from a dated feed
            url=r["url"],
            description=r["description"],
            extra={"source": "browser+llm"},
        ))
    return jobs


def _jobs_to_json(jobs: list[RawJob]) -> str:
    return json.dumps([{
        "title": j.title, "location": j.location, "url": j.url,
        "posted_date": j.posted_date or "", "description": j.description,
    } for j in jobs], ensure_ascii=False)


def _cached_or_extract(cache_key: str, final_url: str, cleaned: str,
                       raw_html: str, clean_limit: int = 30000) -> list[RawJob]:
    """Extract postings, reusing the cached result when the page text is
    unchanged since the last run (zero LLM cost for unchanged pages)."""
    content_hash = hashlib.sha1(cleaned.encode("utf-8", "ignore")).hexdigest()
    cached = db.extraction_cache_get(cache_key)
    if cached and cached["content_hash"] == content_hash:
        try:
            # Same job ceiling as a fresh extraction — a cached page must
            # rematerialize ALL of its postings, not a smaller default.
            return _to_rawjobs(json.loads(cached["jobs_json"]), final_url,
                               MAX_EXTRACT_JOBS)
        except (json.JSONDecodeError, TypeError):
            pass  # corrupt cache — fall through to re-extract
    jobs = extract_jobs(raw_html, base_url=final_url, clean_limit=clean_limit)
    if jobs:
        db.extraction_cache_put(cache_key, content_hash, _jobs_to_json(jobs))
    # An EMPTY extraction is not cached. The caller only reaches this code for
    # a page that already looked extractable, so "0 postings" is far more often
    # a failed extraction than an empty board — and caching it by content hash
    # would freeze that failure in place until the page text happens to change.
    # (Live 2026-09-01: WHO's rendered Taleo board carried 70 real jobs; the
    # model's JSON came back truncated, the [] was cached, and WHO would have
    # reported an empty board on every future run.) Retrying costs one cheap
    # call per unchanged-but-empty page per run; a silently empty employer
    # costs the user the jobs.
    return jobs


def fetch_via_browser(url: str, *, allow_render: bool = True) -> list[RawJob]:
    """Tier-3 fetch — robust and cheap:

      1. HTTP first (requests, then curl_cffi impersonation) — server-rendered
         pages need no browser at all.
      2. If plain HTTP yields a shell — or yields a page the extractor finds
         NO jobs on (login shells and landing pages can look link-rich) — and
         Playwright is installed, render the page with headless Chromium.
      3. LLM-extract the postings, but only when the page content actually
         changed since the last run (per-source content-hash cache); otherwise
         the cached extraction is reused at zero cost.

    A token may be several URLs joined with ' + ' (page-2 offsets, split
    lists): each is fetched independently and the postings are merged,
    deduped by job_id.

    Returns [] (never raises on unreachable pages), so a browser-tier company
    simply stays empty until run on a machine with Playwright."""
    if " + " in url:
        merged: list[RawJob] = []
        seen: set[str] = set()
        for part in (p.strip() for p in url.split(" + ")):
            if not part:
                continue
            for j in fetch_via_browser(part, allow_render=allow_render):
                if j.job_id in seen:
                    continue
                seen.add(j.job_id)
                merged.append(j)
        return merged

    got = fetch_page(url)
    if got:
        final_url, html = got
        cleaned = clean_html_for_llm(html)
        if _looks_extractable(cleaned):
            jobs = _cached_or_extract(f"{url}#http", final_url, cleaned, html)
            if jobs:
                return jobs
            # fall through: the HTTP page carried no postings (SPA shell,
            # landing page, login wall) — a rendered pass may reveal them.

    if allow_render and browser_available():
        try:
            rendered = render_html(url)
        except Exception:  # noqa: BLE001 — a crashed render must not kill a run
            rendered = ""
        if rendered:
            # Multi-page Taleo snapshots concatenate several lists — give the
            # cleaner enough budget that page 2-3 postings survive the trim.
            # The SAME limit is passed to the extractor, so what the walk
            # collected is what the model actually reads.
            limit = 90000 if "jobsearch.ftl" in url else 30000
            cleaned = clean_html_for_llm(rendered, limit=limit)
            if _looks_extractable(cleaned):
                return _cached_or_extract(f"{url}#render", url, cleaned, rendered,
                                          clean_limit=limit)
    return []

"""Dead-link pruning — a recommendation may only point at a job page that
still exists.

Recommendations persist between runs by design (a New rec waits for the
user's decision), but postings die whenever a company closes them: days after
the last pipeline run the board can hold links that answer "page not found".
Feeds close vanished postings during a fetch, but that only happens when a
run happens — and it cannot see a posting that is still listed while its URL
is broken, nor browser-tier boards whose extraction missed a removal.

This module verifies the actual links:

    recheck_recommendations()  every visible New/Marked rec (run start +
                               the dashboard's "Check links" + CLI linkcheck)
    check_candidates()         scoring survivors, right before LLM spend —
                               a dead posting must never be scored or surfaced

Verdicts are three-valued: ALIVE / DEAD / UNKNOWN. Only a *certain* DEAD —
an authoritative ATS-API 404, a page 404/410, or an unambiguous not-found
page — removes anything. Bot walls (401/403/429), timeouts, server errors
are UNKNOWN and keep the posting: a false "dead" silently deletes a real
opportunity, while a false "alive" merely waits for the next feed fetch to
close it. Where the ATS is a JavaScript app whose job pages answer 200 for
every URL (Workday, Oracle, SmartRecruiters, Ashby, Greenhouse embeds), the
check asks the platform's own API instead of the HTML page.

A DEAD posting is archived exactly like the other non-hard-filter verdicts:
its New recommendation is deleted, the posting is closed and keeps
`filtered_reason = "job page gone (…)"` so it can never re-enter the queue
if a stale feed re-lists it. Marked/Applied/… recs are never deleted — the
posting just closes underneath them (the board shows them as closed).
"""
from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional
from urllib.parse import urlsplit

import requests

from . import config, db

ALIVE = "alive"
DEAD = "dead"
UNKNOWN = "unknown"

# Sentinel for the pipeline's enrichment loop: "the detail endpoint said this
# job no longer exists" (returned instead of a detail dict).
GONE: dict = {"_gone": True}


@dataclass
class CheckResult:
    verdict: str                 # ALIVE | DEAD | UNKNOWN
    detail: str = ""             # "HTTP 404", "API 404", "not-found page", …


@dataclass
class PruneOutcome:
    checked: int = 0
    dead: int = 0
    unknown: int = 0
    skipped_fresh: int = 0       # inside the recheck TTL — not re-verified
    recs_removed: int = 0        # New recs deleted from the board
    postings_closed: int = 0
    notes: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)   # "Company — Title (detail)"
    by_type: dict = field(default_factory=dict)        # ats -> {alive,dead,unknown}


# --- HTTP plumbing -------------------------------------------------------------
# Job pages are made for browsers: check them with a browser UA. One session,
# per-host serialization with a politeness delay (hosts run in parallel).
_session = requests.Session()
_session.headers.update({"User-Agent": config.BROWSER_USER_AGENT,
                         "Accept-Language": "en-US,en;q=0.9,de;q=0.8"})
_MAX_BODY = 200_000          # read at most this much of a page (signatures sit early)


class _Gone(Exception):
    """The source answered with a definitive 'this job does not exist'."""
    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


def _fetch(url: str, *, accept: str = "text/html", method: str = "GET",
           json_body: Optional[dict] = None) -> tuple[int, str, str]:
    """One polite request. Returns (status, final_url, body_head).

    Raises `_Gone` on 404/410 (certain), `requests.RequestException` on
    network trouble (the caller maps that to UNKNOWN — fail-open)."""
    hdrs = {"Accept": accept}
    if method == "POST":
        resp = _session.post(url, json=json_body, timeout=config.LINKCHECK_TIMEOUT,
                             headers=hdrs, allow_redirects=True, stream=True)
    else:
        resp = _session.get(url, timeout=config.LINKCHECK_TIMEOUT,
                            headers=hdrs, allow_redirects=True, stream=True)
    try:
        if resp.status_code in (404, 410):
            raise _Gone(f"HTTP {resp.status_code}")
        body = b""
        for chunk in resp.iter_content(chunk_size=32_768):
            body += chunk
            if len(body) >= _MAX_BODY:
                break
        text = body.decode(resp.encoding or "utf-8", errors="replace")
        return resp.status_code, str(resp.url), text
    finally:
        resp.close()


# --- Not-found page signatures ---------------------------------------------------
# Deliberately tight, whole phrases only: a signature match deletes a
# recommendation, so "probably dead" is not good enough. EN + DE (the two
# languages of this board). Checked lowercase against the first ~200 KB.
_DEAD_SIGNATURES = (
    "job you are looking for is no longer",
    "job you're looking for is no longer",
    "position you are looking for is no longer",
    "job you are looking for could not be found",
    "position is no longer available",
    "position has been filled",
    "position is no longer open",
    "job is no longer available",
    "job is no longer active",
    "job is no longer open",
    "posting is no longer available",
    "posting is no longer active",
    "vacancy is no longer available",
    "vacancy has been closed",
    "job posting has expired",
    "job posting is expired",
    "posting has expired",
    "job has expired",
    "job posting has closed",
    "no longer accepting applications",
    "job could not be found",
    "job posting was not found",
    "job not found",
    "posting not found",
    "page you were looking for doesn't exist",
    "page you requested was not found",
    "requisition is no longer open",
    # German
    "stellenanzeige wurde nicht gefunden",
    "job nicht mehr verfügbar",
    "stelle ist nicht mehr verfügbar",
    "stellenangebot ist nicht mehr verfügbar",
    "stellenanzeige ist nicht mehr online",
    "stellenanzeige ist nicht mehr verfügbar",
    "stellenausschreibung ist nicht mehr verfügbar",
    "diese stelle wurde bereits besetzt",
    "stelle wurde bereits besetzt",
    "angebot ist nicht mehr verfügbar",
    "diese seite existiert nicht",
    "seite wurde nicht gefunden",
)

# A dead job URL redirected to an explicit tombstone page — the slug alone is
# the verdict, whatever the host (Mercedes: group.mercedes-benz.com/karriere/
# jobsuche/nicht-mehr-verfuegbar/).
_DEAD_URL_SLUGS = ("nicht-mehr-verfuegbar", "job-not-found", "jobnotfound",
                   "no-longer-available", "job-expired", "position-closed",
                   "expired-job")

_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
_LONG_NUM_RE = re.compile(r"\d{4,}")
# Board/search landing paths a dead job page redirects to (same host).
_BOARD_ROOTS = re.compile(r"^/?(|search|jobs|careers|career|en|de|joblist|"
                          r"go/[^/]+|search/?.*|.*/search/?)$", re.I)


def _page_signature_dead(text: str) -> Optional[str]:
    low = text.lower()
    for sig in _DEAD_SIGNATURES:
        if sig in low:
            return f"page says: “{sig[:48]}”"
    return None


def _job_marker(path: str) -> Optional[str]:
    """The job's identifying token inside a URL path (uuid or a long number)."""
    m = _UUID_RE.search(path)
    if m:
        return m.group(0).lower()
    nums = _LONG_NUM_RE.findall(path)
    return max(nums, key=len) if nums else None


def _redirected_off_job(orig_url: str, final_url: str) -> bool:
    """True when a dead job URL was redirected to its board's landing page.

    Conservative on purpose: same host only (cross-host redirects are usually
    re-platformings, not removals), the original path must carry a job marker
    (uuid / long id) that the final path lost, and the final path must look
    like a board root ("/", "/search", "/jobs", …) — a localized or slugged
    variant of the same posting keeps its marker and stays ALIVE."""
    try:
        o, f = urlsplit(orig_url), urlsplit(final_url)
    except ValueError:
        return False
    if not f.netloc or o.netloc.lower() != f.netloc.lower():
        return False
    marker = _job_marker(o.path)
    if not marker or marker in f.path.lower() or marker in (f.query or "").lower():
        return False
    return bool(_BOARD_ROOTS.match(f.path.strip()))


# --- Per-platform checkers -------------------------------------------------------
# Each takes the posting row (url, job_id, detail_ref, ats_type, ats_token)
# and returns a CheckResult. API-backed checks are authoritative; the generic
# page check is the fallback for everything else.

def _fetch_impersonated(url: str) -> Optional[tuple[int, str, str]]:
    """Bot-walled hosts (Akamai on jobs.mercedes-benz.com & co) 403 plain
    clients but serve browser-TLS fingerprints. Optional curl_cffi retry —
    None when the library is missing or the request fails (stay UNKNOWN)."""
    try:
        from curl_cffi import requests as creq
    except ImportError:
        return None
    try:
        r = creq.get(url, impersonate="chrome", timeout=config.LINKCHECK_TIMEOUT,
                     allow_redirects=True)
        return int(r.status_code), str(getattr(r, "url", url) or url), \
            (r.text or "")[:_MAX_BODY]
    except Exception:  # noqa: BLE001 — optional path, never certain enough to kill
        return None


def _verdict_from_page(url: str, status: int, final_url: str,
                       text: str) -> CheckResult:
    if status in (404, 410):
        return CheckResult(DEAD, f"HTTP {status}")
    if status != 200:
        # 401/403/405/429/5xx…: can't tell — never delete on a bot wall.
        return CheckResult(UNKNOWN, f"HTTP {status}")
    sig = _page_signature_dead(text)
    if sig:
        return CheckResult(DEAD, sig)
    low_final = (final_url or "").lower()
    if any(s in low_final for s in _DEAD_URL_SLUGS):
        return CheckResult(DEAD, "redirected to a 'job no longer available' page")
    if _redirected_off_job(url, final_url):
        return CheckResult(DEAD, "redirected to the board's landing page")
    return CheckResult(ALIVE, "page ok")


def _check_generic(row) -> CheckResult:
    url = (row["url"] or "").strip()
    if not url.lower().startswith(("http://", "https://")):
        return CheckResult(UNKNOWN, "no usable URL")
    try:
        status, final_url, text = _fetch(url)
    except _Gone as e:
        return CheckResult(DEAD, e.detail)
    except requests.RequestException as e:
        return CheckResult(UNKNOWN, type(e).__name__)
    res = _verdict_from_page(url, status, final_url, text)
    if res.verdict == UNKNOWN and status in (401, 403, 405, 429):
        # One browser-impersonated retry gets past TLS-fingerprint walls
        # (Mercedes/beesite career pages) — same conservative rules after.
        imp = _fetch_impersonated(url)
        if imp is not None:
            return _verdict_from_page(url, *imp)
    return res


def _check_greenhouse(row) -> CheckResult:
    """Greenhouse job URLs are often company-hosted embeds (stripe.com/jobs/…)
    that answer 200 for any gh_jid — ask the boards API instead."""
    token, jid = (row["ats_token"] or "").strip(), (row["job_id"] or "").strip()
    if not token or not jid.isdigit():
        return _check_generic(row)
    try:
        status, _, _ = _fetch(
            f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs/{jid}",
            accept="application/json")
    except _Gone:
        return CheckResult(DEAD, "Greenhouse API: posting gone")
    except requests.RequestException as e:
        return CheckResult(UNKNOWN, type(e).__name__)
    return CheckResult(ALIVE, "API ok") if status == 200 \
        else CheckResult(UNKNOWN, f"API HTTP {status}")


def _check_lever(row) -> CheckResult:
    """The hosted page 404s for removed postings — and keeps working for
    companies that disabled the public list API (Mistral), where a posting-API
    check would false-positive."""
    return _check_generic(row)


# Workday needs a per-tenant session warm-up: some tenants (Rolls-Royce)
# answer 403 "permission denied" to cookieless CXS calls. One list POST per
# tenant+site sets the CALYPSO/PLAY session cookies on the module session.
_workday_lock = threading.Lock()
_workday_warm: dict[str, bool] = {}
# The requisition suffix inside an externalPath ("…_JR6153861-1", "…_R2863913").
_WD_REQ_RE = re.compile(r"_([A-Za-z]{0,5}\d{4,}[\w-]*)$")


def _workday_list_url(tenant: str, dc: str, site: str) -> str:
    return f"https://{tenant}.{dc}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs"


def _workday_warmup(tenant: str, dc: str, site: str) -> bool:
    """True when the tenant's CXS list endpoint answers (cookies now set)."""
    key = f"{tenant}|{site}"
    with _workday_lock:
        if key in _workday_warm:
            return _workday_warm[key]
    ok = False
    try:
        status, _, _ = _fetch(_workday_list_url(tenant, dc, site),
                              accept="application/json", method="POST",
                              json_body={"appliedFacets": {}, "limit": 1,
                                         "offset": 0, "searchText": ""})
        ok = status == 200
    except (_Gone, requests.RequestException):
        ok = False
    with _workday_lock:
        _workday_warm[key] = ok
    return ok


def _workday_still_listed(tenant: str, dc: str, site: str, ext: str) -> Optional[bool]:
    """Search the tenant for the requisition id — the second, independent
    signal before a 403 may count as dead. True/False = listed/absent;
    None = could not tell (no req id in the path, or the search failed)."""
    m = _WD_REQ_RE.search(ext)
    if not m:
        return None
    try:
        status, _, text = _fetch(_workday_list_url(tenant, dc, site),
                                 accept="application/json", method="POST",
                                 json_body={"appliedFacets": {}, "limit": 20,
                                            "offset": 0, "searchText": m.group(1)})
    except (_Gone, requests.RequestException):
        return None
    if status != 200:
        return None
    import json as _json
    try:
        postings = (_json.loads(text) or {}).get("jobPostings") or []
    except ValueError:
        return None
    return any((p.get("externalPath") or "") == ext for p in postings)


def _check_workday(row) -> CheckResult:
    """Workday job pages are an SPA (HTTP 200 for anything); the CXS detail
    endpoint is the truth. Removed postings answer 404 on most tenants — but
    some (Rolls-Royce) answer 403 "permission denied" (errorCode S22), the
    same body their SPA renders as "page not found". A 403 counts as DEAD
    only under the strictest reading: the tenant's list endpoint just worked
    on this very session (so it's not a bot wall) AND the requisition id is
    absent from the tenant's own search. Anything less stays UNKNOWN."""
    token = (row["ats_token"] or "").strip()
    parts = token.split("|")
    ext = (row["detail_ref"] or "").strip()
    if len(parts) != 3 or not all(parts):
        return CheckResult(UNKNOWN, "bad workday token")
    tenant, dc, site = parts
    if not ext:
        # Older rows may lack detail_ref; the list URL is {base}/{site}{ext}.
        path = urlsplit(row["url"] or "").path
        i = path.find(f"/{site}/")
        if i < 0:
            return CheckResult(UNKNOWN, "no workday detail path")
        ext = path[i + len(site) + 1:]
    warm = _workday_warmup(tenant, dc, site)
    try:
        status, _, text = _fetch(
            f"https://{tenant}.{dc}.myworkdayjobs.com/wday/cxs/{tenant}/{site}{ext}",
            accept="application/json")
    except _Gone:
        return CheckResult(DEAD, "Workday API: posting gone")
    except requests.RequestException as e:
        return CheckResult(UNKNOWN, type(e).__name__)
    if status == 200 and '"jobPostingInfo"' in text:
        return CheckResult(ALIVE, "API ok")
    if status == 403 and warm and "permission denied" in text.lower():
        listed = _workday_still_listed(tenant, dc, site, ext)
        if listed is False:
            return CheckResult(
                DEAD, "Workday API: posting no longer public "
                      "(denied + absent from the tenant's search)")
        if listed is True:
            return CheckResult(UNKNOWN, "detail denied but requisition still listed")
    return CheckResult(UNKNOWN, f"API HTTP {status}")


def _check_smartrecruiters(row) -> CheckResult:
    """The public postings API (same one the connector lists from) 404s for
    removed postings; the HTML page just redirects to the company board."""
    slug = (row["ats_token"] or "").split("|")[0].strip()
    jid = (row["job_id"] or "").strip()
    if not slug or not jid:
        return _check_generic(row)
    try:
        status, _, _ = _fetch(
            f"https://api.smartrecruiters.com/v1/companies/{slug}/postings/{jid}",
            accept="application/json")
    except _Gone:
        return CheckResult(DEAD, "SmartRecruiters API: posting gone")
    except requests.RequestException as e:
        return CheckResult(UNKNOWN, type(e).__name__)
    return CheckResult(ALIVE, "API ok") if status == 200 \
        else CheckResult(UNKNOWN, f"API HTTP {status}")


def _check_oracle(row) -> CheckResult:
    """Oracle CandidateExperience is an SPA; the requisition finder returns an
    empty item list for a requisition that no longer exists on the site."""
    parts = (row["ats_token"] or "").split("|")
    jid = (row["job_id"] or "").strip()
    if len(parts) < 2 or not jid:
        return _check_generic(row)
    host, site = parts[0].strip(), parts[1].strip()
    url = (f"https://{host}/hcmRestApi/resources/latest/"
           f'recruitingCEJobRequisitionDetails?onlyData=true&'
           f'finder=ById;Id="{jid}",siteNumber={site}')
    try:
        status, _, text = _fetch(url, accept="application/json")
    except _Gone:
        return CheckResult(DEAD, "Oracle API: requisition gone")
    except requests.RequestException as e:
        return CheckResult(UNKNOWN, type(e).__name__)
    if status != 200:
        return CheckResult(UNKNOWN, f"API HTTP {status}")
    try:
        import json as _json
        items = (_json.loads(text) or {}).get("items")
    except ValueError:
        return CheckResult(UNKNOWN, "API returned non-JSON")
    if items is None:
        return CheckResult(UNKNOWN, "API answer has no items field")
    return CheckResult(ALIVE, "API ok") if items \
        else CheckResult(DEAD, "Oracle API: requisition gone")


# One Ashby org board serves every posting in a single request — cache it for
# the duration of one prune pass instead of refetching per posting.
_ashby_lock = threading.Lock()
_ashby_boards: dict[str, Optional[set]] = {}


def _ashby_ids(org: str) -> Optional[set]:
    with _ashby_lock:
        if org in _ashby_boards:
            return _ashby_boards[org]
    ids: Optional[set] = None
    try:
        status, _, text = _fetch(
            f"https://api.ashbyhq.com/posting-api/job-board/{org}",
            accept="application/json")
        if status == 200:
            import json as _json
            jobs = (_json.loads(text) or {}).get("jobs")
            if isinstance(jobs, list):
                ids = {str(j.get("id")) for j in jobs}
    except (requests.RequestException, ValueError, _Gone):
        ids = None                      # org feed unreachable — fail open
    with _ashby_lock:
        _ashby_boards[org] = ids
    return ids


def _check_ashby(row) -> CheckResult:
    """Ashby job pages are an SPA; membership in the org's posting-API board
    (the exact feed the connector fetches) is the source of truth."""
    org = (row["ats_token"] or "").strip()
    jid = (row["job_id"] or "").strip()
    if not org or not jid:
        return _check_generic(row)
    ids = _ashby_ids(org)
    if ids is None:
        return CheckResult(UNKNOWN, "Ashby board unreachable")
    return CheckResult(ALIVE, "on the org board") if jid in ids \
        else CheckResult(DEAD, "Ashby API: not on the org board")


def _check_amazon(row) -> CheckResult:
    """amazon.jobs serves a JSON twin of every job page; a removed job returns
    an error payload (the HTML page itself answers 200 with a shell)."""
    jid = (row["job_id"] or "").strip()
    if not jid.isdigit():
        return _check_generic(row)
    try:
        status, _, text = _fetch(f"https://www.amazon.jobs/en/jobs/{jid}.json",
                                 accept="application/json")
    except _Gone:
        return CheckResult(DEAD, "amazon.jobs API: job gone")
    except requests.RequestException as e:
        return CheckResult(UNKNOWN, type(e).__name__)
    if status != 200:
        return CheckResult(UNKNOWN, f"API HTTP {status}")
    low = text[:2000].lower()
    if '"error"' in low and '"job"' not in low:
        return CheckResult(DEAD, "amazon.jobs API: job gone")
    return CheckResult(ALIVE, "API ok")


def _check_apple(row) -> CheckResult:
    """jobs.apple.com details pages are an SPA; the jobDetails API answers
    404 (jobsite.general.serviceError) once a role is gone."""
    jid = (row["job_id"] or "").strip()
    if not jid:
        return _check_generic(row)
    try:
        status, _, _ = _fetch(
            f"https://jobs.apple.com/api/v1/jobDetails/{jid}?languageCd=en-us",
            accept="application/json")
    except _Gone:
        return CheckResult(DEAD, "Apple API: role gone")
    except requests.RequestException as e:
        return CheckResult(UNKNOWN, type(e).__name__)
    return CheckResult(ALIVE, "API ok") if status == 200 \
        else CheckResult(UNKNOWN, f"API HTTP {status}")


CHECKERS: dict[str, Callable] = {
    "greenhouse": _check_greenhouse,
    "lever": _check_lever,
    "workday": _check_workday,
    "smartrecruiters": _check_smartrecruiters,
    "oracle": _check_oracle,
    "ashby": _check_ashby,
    "amazon": _check_amazon,
    "apple": _check_apple,
    # Everything else (personio, successfactors career sites, sf_rmk, beesite,
    # jibe, dvinci, erecruiter, verbund, recruitee, workable, uber, un,
    # microsoft, ibm, browser tier) uses the generic page check: real 404s and
    # explicit not-found pages are caught (live-verified on successfactors,
    # jibe, dvinci, verbund, beesite-via-impersonation, browser tier); pure
    # SPAs that answer 200 for dead jobs (uber, un, microsoft, ibm) stay
    # UNKNOWN/ALIVE here and are closed by feed truth during the next fetch
    # instead — fail-open by design: a link that still opens a page is never
    # deleted on a guess.
}


def check_posting(row) -> CheckResult:
    """Verdict for one posting row (needs url/job_id/detail_ref/ats_type/ats_token)."""
    checker = CHECKERS.get((row["ats_type"] or "").strip(), _check_generic)
    try:
        return checker(row)
    except _Gone as e:                       # a checker let one bubble through
        return CheckResult(DEAD, e.detail)
    except Exception as e:  # noqa: BLE001 — a checker bug must never kill a run
        return CheckResult(UNKNOWN, f"checker error: {type(e).__name__}")


# --- Concurrency: parallel across hosts, serial + delayed within one host -------
def _host_of(row) -> str:
    try:
        return urlsplit(row["url"] or "").netloc.lower() or f"#none-{row['id']}"
    except ValueError:
        return f"#bad-{row['id']}"


def _check_many(rows, progress=None, label_current: bool = True) -> dict[int, CheckResult]:
    """Check many postings politely; returns {posting_id: CheckResult}."""
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(_host_of(r), []).append(r)
    results: dict[int, CheckResult] = {}
    lock = threading.Lock()

    def _one_host(host_rows) -> None:
        import time as _time
        for r in host_rows:
            res = check_posting(r)
            with lock:
                results[r["id"]] = res
            if progress is not None:
                cur = f"{r['company_name']} — {r['title']}" if label_current else ""
                progress.step(1, current=cur[:110])
            _time.sleep(config.LINKCHECK_DELAY)

    workers = max(1, min(config.LINKCHECK_WORKERS, len(groups)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_one_host, g) for g in groups.values()]
        for fut in as_completed(futures):
            fut.result()                     # surface unexpected errors
    return results


# --- Archival ---------------------------------------------------------------------
def dead_reason(detail: str) -> str:
    today = datetime.now(timezone.utc).date().isoformat()
    detail = (detail or "link dead").strip()
    return f"job page gone ({detail}; checked {today})"


def _fresh_cutoff(ttl_hours: Optional[float]) -> str:
    ttl = config.LINKCHECK_TTL_HOURS if ttl_hours is None else ttl_hours
    return (datetime.now(timezone.utc) - timedelta(hours=ttl)) \
        .isoformat(timespec="seconds")


# --- Entry point 1: the visible board ----------------------------------------------
def recheck_recommendations(*, ttl_hours: Optional[float] = None,
                            force: bool = False, progress=None,
                            verbose: bool = False, db_path=None) -> PruneOutcome:
    """Verify every New/Marked recommendation whose posting is still open.

    DEAD  → New rec deleted + posting archived ("job page gone …", closed);
            a Marked rec is kept, only its posting closes underneath it.
    ALIVE → link_checked_at stamped (skipped until the TTL passes).
    UNKNOWN → untouched, so the next pass retries.
    """
    out = PruneOutcome()
    rows = db.postings_for_linkcheck(db_path=db_path)
    if not force:
        cutoff = _fresh_cutoff(ttl_hours)
        fresh = [r for r in rows if (r["link_checked_at"] or "") >= cutoff]
        out.skipped_fresh = len(fresh)
        rows = [r for r in rows if (r["link_checked_at"] or "") < cutoff]
    if not rows:
        return out
    if progress is not None:
        progress.phase("linkcheck",
                       f"Verifying {len(rows)} recommendation links", total=len(rows))
    _ashby_boards.clear()                    # fresh feed truth for this pass
    with _workday_lock:
        _workday_warm.clear()
    results = _check_many(rows, progress=progress)
    alive_ids: list[int] = []
    for r in rows:
        res = results.get(r["id"]) or CheckResult(UNKNOWN, "not checked")
        out.checked += 1
        t = out.by_type.setdefault((r["ats_type"] or "?"),
                                   {ALIVE: 0, DEAD: 0, UNKNOWN: 0})
        t[res.verdict] += 1
        if res.verdict == DEAD:
            out.dead += 1
            removed_rec = db.archive_dead_link(r["id"], dead_reason(res.detail),
                                               db_path=db_path)
            out.postings_closed += 1
            out.recs_removed += int(removed_rec)
            out.removed.append(f"{r['company_name']} — {r['title']} ({res.detail})")
            if verbose:
                print(f"  ✗ dead: {r['company_name']} — {r['title'][:60]} "
                      f"[{res.detail}]")
        elif res.verdict == ALIVE:
            alive_ids.append(r["id"])
        else:
            out.unknown += 1
    db.touch_link_checked(alive_ids, db_path=db_path)
    if out.dead:
        out.notes.append(f"link check removed {out.dead} dead postings "
                         f"({out.recs_removed} recommendations) from the board")
    return out


# --- Entry point 2: scoring survivors, before any LLM spend -------------------------
def check_candidates(rows, progress=None, db_path=None) -> tuple[int, dict[int, str]]:
    """Verify hard-filter survivors right before they are scored.

    Returns (checked, {posting_id: reason}) — the DEAD ones (already archived
    here). Skips postings checked within the TTL, and postings whose
    description is about to be detail-enriched anyway — for those, the
    enrichment fetch IS the liveness probe (the pipeline maps its 404 to GONE
    via `gone_result`)."""
    from .connectors import get_connector
    todo = []
    cutoff = _fresh_cutoff(None)
    for p in rows:
        if (p["link_checked_at"] or "") >= cutoff:
            continue
        conn = get_connector(p["ats_type"] or "")
        will_enrich = (conn is not None and getattr(conn, "supports_detail", False)
                       and p["detail_ref"] and not (p["description"] or ""))
        if will_enrich:
            continue
        todo.append(p)
    if not todo:
        return 0, {}
    if progress is not None:
        progress.phase("linkcheck",
                       f"Verifying {len(todo)} job links before scoring",
                       total=len(todo))
    dead: dict[int, str] = {}
    results = _check_many(todo, progress=progress)
    alive_ids: list[int] = []
    for p in todo:
        res = results.get(p["id"]) or CheckResult(UNKNOWN, "not checked")
        if res.verdict == DEAD:
            reason = dead_reason(res.detail)
            db.archive_dead_link(p["id"], reason, db_path=db_path)
            dead[p["id"]] = reason
        elif res.verdict == ALIVE:
            alive_ids.append(p["id"])
    db.touch_link_checked(alive_ids, db_path=db_path)
    return len(todo), dead


def gone_result(exc: Exception) -> bool:
    """True when a connector's detail fetch failed because the job is GONE
    (HTTP 404 underneath), as opposed to network trouble. Used by the
    pipeline's enrichment loop to turn a swallowed error into a verdict."""
    cause = exc.__cause__
    resp = getattr(cause, "response", None)
    if resp is not None and getattr(resp, "status_code", None) in (404, 410):
        return True
    return "404 not found" in str(exc).lower()

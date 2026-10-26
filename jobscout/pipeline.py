"""The pipeline: a simple, one-directional flow.

    detect (once) -> fetch -> dedupe & store -> hard filters -> prefilter
                  -> enrich -> LLM score -> recommend

Each run is incremental: only postings never seen before are filtered and
scored, so cost and time stay near zero as feeds refresh. Fetching runs
companies in parallel (requests within one company stay sequential and
throttled), and every fetch outcome is recorded per company so the Coverage
view shows real health, not guesses.
"""
from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from . import config, db, feedback, matching
from .connectors import fetch_detail, fetch_jobs
from .detect import detect


# --- Live run progress ---------------------------------------------------------
class Progress:
    """Thread-safe live progress of the current pipeline run.

    The pipeline thread updates it at every meaningful step (each company
    fetched, each LLM batch scored, ...); the web API serves `snapshot()` to
    the dashboard's poll loop, which renders it as a progress bar. Costs
    nothing when nobody looks at it.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._d = {
                "phase": None,        # machine name: fetch | score | ... | done
                "label": "",          # human line: "Fetching 84 feed companies"
                "done": 0,            # units finished within this phase
                "total": None,        # units in this phase (None = indeterminate)
                "current": "",        # what is being worked on right now
                "counters": {},       # rolling run totals (new postings, scored…)
                "events": [],         # last few "BMW: 796 open, 407 new" lines
                "updated_at": None,
            }

    def _touch(self) -> None:
        self._d["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def phase(self, phase: str, label: str, total: int | None = None) -> None:
        with self._lock:
            self._d.update(phase=phase, label=label, done=0, total=total, current="")
            self._touch()

    def step(self, n: int = 1, current: str | None = None) -> None:
        with self._lock:
            self._d["done"] += n
            if current is not None:
                self._d["current"] = current[:120]
            self._touch()

    def add(self, **counters: float) -> None:
        """Increment rolling counters (postings_new=…, scored=…)."""
        with self._lock:
            c = self._d["counters"]
            for k, v in counters.items():
                c[k] = round(c.get(k, 0) + v, 4)
            self._touch()

    def set(self, **counters: float) -> None:
        """Set counters to absolute values (llm_cost=…)."""
        with self._lock:
            self._d["counters"].update({k: round(v, 4) for k, v in counters.items()})
            self._touch()

    def event(self, msg: str) -> None:
        with self._lock:
            ev = self._d["events"]
            ev.append(msg[:160])
            del ev[:-6]
            self._touch()

    def snapshot(self) -> dict:
        with self._lock:
            d = dict(self._d)
            d["counters"] = dict(self._d["counters"])
            d["events"] = list(self._d["events"])
            return d


PROGRESS = Progress()


def _days_ago_iso(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()


@dataclass
class RunSummary:
    companies_fetched: int = 0
    fetch_errors: int = 0
    postings_seen: int = 0
    postings_new: int = 0
    postings_closed: int = 0
    filtered_out: int = 0
    links_checked: int = 0      # job-page URLs verified by the dead-link prune
    links_dead: int = 0         # verified-dead postings archived off the board
    prefiltered_in: int = 0
    enriched: int = 0
    scored: int = 0
    floor_rejected: int = 0     # LLM-scored below REC_MIN_SCORE — no board row
    surfaced: int = 0
    validated: int = 0          # recs reviewed by the final validation gate
    validation_removed: int = 0
    llm_used: int = 0
    llm_errors: int = 0
    llm_cost_usd: float = 0.0
    # Set when the Anthropic API is permanently unusable (no credit, bad key,
    # forbidden model). Non-empty means: nothing was scored and the board
    # could not grow — the dashboard shows this instead of a green summary.
    llm_fatal: str = ""
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in (
            "companies_fetched", "fetch_errors", "postings_seen", "postings_new",
            "postings_closed", "filtered_out", "links_checked", "links_dead",
            "prefiltered_in", "enriched",
            "scored", "floor_rejected", "surfaced", "validated",
            "validation_removed", "llm_used", "llm_errors")}
        d["llm_cost_usd"] = round(self.llm_cost_usd, 4)
        d["llm_fatal"] = self.llm_fatal
        d["notes"] = self.notes[:12]
        return d


# --- One-time detection ------------------------------------------------------
def detect_all(force: bool = False, limit: int | None = None, verbose: bool = True,
               known_only: bool = False, use_resolver: bool = True) -> dict:
    companies = db.get_companies()
    pending = [c for c in companies if force or c["detect_status"] != "detected"]
    if pending:
        PROGRESS.phase("detect", f"Detecting ATS for {len(pending)} companies",
                       total=len(pending))
    result = {"detected": 0, "unsupported": 0, "skipped": 0}
    n = 0
    for c in companies:
        if not force and c["detect_status"] == "detected":
            result["skipped"] += 1
            continue
        if limit is not None and n >= limit:
            break
        n += 1
        PROGRESS.step(0, current=c["name"])
        d = detect(c["name"], known_only=known_only, use_resolver=use_resolver)
        PROGRESS.step(1)
        db.set_company_ats(c["id"], d.ats_type, d.token, d.status, d.note)
        if verbose:
            tag = f"{d.ats_type}:{d.token}" if d.status == "detected" else f"— ({d.note})"
            print(f"  [{d.status:11}] {c['name']:42} {tag}")
        result[d.status] = result.get(d.status, 0) + 1
    return result


def apply_curated_detection() -> int:
    """Re-apply the curated detection maps to already-detected companies.

    Cheap and offline. Keeps the DB in line with the curated maps when they
    improve (e.g. a company moves from the browser tier to a real feed).
    KNOWN_UNSUPPORTED also wins over a stale auto-probe: "Meta" once probed
    as recruitee/meta — an unrelated (scam) tenant at facebookdata.recruitee
    .com whose posting reached the queue. Curation overrides the guess and
    closes any postings the bogus source left behind."""
    from .detect import KNOWN_BROWSER, KNOWN_TOKENS, KNOWN_UNSUPPORTED, _key
    changed = 0
    for c in db.get_companies():
        key = _key(c["name"])
        if key in KNOWN_TOKENS:
            ats, token = KNOWN_TOKENS[key]
            if c["ats_type"] != ats or c["ats_token"] != token:
                db.set_company_ats(c["id"], ats, token, "detected", "curated")
                changed += 1
        elif key in KNOWN_BROWSER:
            url = KNOWN_BROWSER[key]
            if c["ats_type"] != "browser" or c["ats_token"] != url:
                db.set_company_ats(c["id"], "browser", url, "detected", "curated-browser")
                changed += 1
        elif key in KNOWN_UNSUPPORTED and c["detect_status"] == "detected":
            db.set_company_ats(c["id"], None, None, "unsupported",
                               KNOWN_UNSUPPORTED[key])
            db.close_missing_postings(c["id"], seen_job_ids=[])
            changed += 1
    return changed


# --- Company selection (partial runs) -----------------------------------------
class UnknownCompanyError(ValueError):
    """A requested company name matched nothing (or was ambiguous)."""


def resolve_company_selection(names, companies=None) -> list:
    """Map user-supplied company names to company rows.

    Matching is case-insensitive: exact name first, then unique substring
    ("bmw" -> "BMW", "apple" -> "Apple"). An unknown or ambiguous name raises
    UnknownCompanyError listing the offenders — a partial run over the wrong
    set must never start silently."""
    companies = companies if companies is not None else db.get_companies()
    by_exact = {(c["name"] or "").casefold(): c for c in companies}
    picked: dict[int, object] = {}
    problems: list[str] = []
    for raw in names or []:
        want = " ".join(str(raw).split()).casefold()
        if not want:
            continue
        row = by_exact.get(want)
        if row is None:
            subs = [c for c in companies if want in (c["name"] or "").casefold()]
            if len(subs) == 1:
                row = subs[0]
            elif len(subs) > 1:
                problems.append(f"'{raw}' is ambiguous ({', '.join(c['name'] for c in subs[:4])}"
                                + ("…" if len(subs) > 4 else "") + ")")
                continue
        if row is None:
            problems.append(f"'{raw}' matches no company")
            continue
        picked[row["id"]] = row
    if problems:
        raise UnknownCompanyError("; ".join(problems))
    return list(picked.values())


# --- Fetch + store -----------------------------------------------------------
def _fetch_one(company) -> list:
    """Fetch one company's jobs (runs on a worker thread; no DB access)."""
    if company["ats_type"] == "browser":
        from . import browser as _browser
        return _browser.fetch_via_browser(company["ats_token"])
    return fetch_jobs(company["ats_type"], company["ats_token"])


def _fetch_and_store(summary: RunSummary, verbose: bool, companies,
                     max_workers: int | None = None,
                     phase_label: str | None = None) -> None:
    """Fetch companies in parallel; store results serially (SQLite writer)."""
    if not companies:
        return
    PROGRESS.phase("fetch",
                   phase_label or f"Fetching {len(companies)} company feeds",
                   total=len(companies))
    llm_blocked: list[str] = []
    went_quiet: list[tuple] = []
    with ThreadPoolExecutor(max_workers=max_workers or config.FETCH_WORKERS) as ex:
        futures = {ex.submit(_fetch_one, c): c for c in companies}
        for fut in as_completed(futures):
            c = futures[fut]
            try:
                jobs = fut.result()
            except Exception as e:  # noqa: BLE001 — one bad feed must never kill a run
                summary.fetch_errors += 1
                if isinstance(e, matching.LLMUnavailable):
                    # Browser-tier pages are read by Claude, so an API outage
                    # is NOT a scraping failure. Name the real cause once,
                    # not 26 times, and keep the company list readable.
                    llm_blocked.append(c["name"])
                    short = "needs the Anthropic API (unavailable this run)"
                else:
                    short = _short_error(e)
                    summary.notes.append(f"fetch failed for {c['name']}: {short}")
                # Keep the last known job count: zeroing it here blinds the
                # "board went from N to 0" check on the NEXT run — which is
                # exactly when a company that moved ATS needs catching.
                db.set_fetch_health(c["id"], f"error: {short}",
                                    c["last_fetch_jobs"] or 0)
                PROGRESS.step(1, current=c["name"])
                PROGRESS.event(f"✗ {c['name']}: {short[:80]}")
                PROGRESS.add(fetch_errors=1)
                if verbose:
                    print(f"  ! {c['name']}: {short}")
                continue
            summary.companies_fetched += 1
            seen_ids = []
            new_here = 0
            for j in jobs:
                seen_ids.append(j.job_id)
                _pid, is_new = db.upsert_posting(
                    company_id=c["id"], job_id=j.job_id, title=j.title,
                    location=j.location, posted_date=j.posted_date, url=j.url,
                    description=j.description, detail_ref=j.detail_ref,
                    date_confidence=j.date_confidence)
                summary.postings_seen += 1
                if is_new:
                    summary.postings_new += 1
                    new_here += 1
            PROGRESS.step(1, current=c["name"])
            PROGRESS.add(postings_new=new_here)
            PROGRESS.event(f"{c['name']}: {len(jobs)} open, {new_here} new")
            # Only a fetch that actually returned jobs may close vanished
            # postings — an empty result (SPA without Playwright, temporary
            # block) must not wipe a previously healthy board.
            if jobs:
                summary.postings_closed += db.close_missing_postings(c["id"], seen_ids)
                cap = config.CONNECTOR_CAPS.get(c["ats_type"] or "")
                if cap and len(jobs) >= cap:
                    # A fetch that fills its safety cap probably truncated the
                    # board — surface it instead of missing postings silently.
                    db.set_fetch_health(
                        c["id"], f"ok (at cap {cap} — board may be larger; "
                                 f"raise JOBSCOUT_*_MAX)", len(jobs))
                    summary.notes.append(f"{c['name']}: fetch hit cap {cap}")
                else:
                    db.set_fetch_health(c["id"], "ok", len(jobs))
            else:
                db.set_fetch_health(c["id"], "empty", 0)
                # A board that HAD jobs and now returns none is the quiet way
                # a company disappears: the fetch "succeeds", the health says
                # "empty", and nobody notices the employer moved ATS. (Mistral
                # kept 200-OK-with-an-empty-list on its abandoned Lever board
                # for weeks while 172 postings sat on Ashby.) Feed truth still
                # refuses to close anything on an empty fetch — this only
                # raises a flag.
                had = c["last_fetch_jobs"] if "last_fetch_jobs" in c.keys() else None
                if had:
                    went_quiet.append((c["name"], int(had)))
            if verbose:
                print(f"  + {c['name']:42} {len(jobs):4d} open, {new_here:3d} new")
    for name, had in sorted(went_quiet):
        summary.notes.append(
            f"{name}: 0 jobs, but {had} last run — its careers board may have "
            "moved (check the ATS token)")
        PROGRESS.event(f"⚠ {name}: board went from {had} to 0")
    if llm_blocked:
        summary.llm_fatal = summary.llm_fatal or matching.llm_fatal_reason
        why = (matching.llm_fatal_reason
               or ("no Anthropic API key is configured"
                   if not config.ANTHROPIC_API_KEY else "an API error"))
        summary.notes.append(
            f"{len(llm_blocked)} browser-tier companies could not be read — "
            f"their careers pages are read by Claude and {why}: "
            + ", ".join(sorted(llm_blocked)[:8])
            + (" …" if len(llm_blocked) > 8 else ""))


# --- Short, readable error text ----------------------------------------------
def _short_error(e: BaseException, limit: int = 140) -> str:
    """One readable line for a failed fetch.

    Raw SDK/HTTP exceptions carry hundreds of characters of JSON payload and
    request ids; they used to be written verbatim into the company health
    column and the run notes, where they pushed everything else out of view."""
    text = " ".join(str(e).split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


# --- Dead-link prune ------------------------------------------------------------
def _prune_dead_links(summary: RunSummary, verbose: bool,
                      force: bool = False) -> None:
    """Board hygiene, FIRST thing in a run — before the hard filters, the
    queue, or any scoring is touched: verify that the job pages behind the
    current New/Marked recommendations still exist, and archive the ones
    that are gone (the user must never be shown a "page not found" link).
    Verified-alive links are stamped and skipped until the TTL passes, so
    daily runs stay cheap. Fail-open: only certain 404s/not-found pages
    remove anything; bot walls and network trouble change nothing."""
    if not config.LINKCHECK_ENABLED:
        return
    from . import linkcheck
    out = linkcheck.recheck_recommendations(force=force, progress=PROGRESS,
                                            verbose=verbose)
    summary.links_checked += out.checked
    summary.links_dead += out.dead
    summary.notes.extend(out.notes)
    if out.dead:
        PROGRESS.add(links_dead=out.dead)
        PROGRESS.event(f"link check: {out.dead} dead job pages "
                       "removed from the board")
    if verbose and (out.checked or out.skipped_fresh):
        print(f"  link check: {out.checked} verified, {out.dead} dead, "
              f"{out.unknown} unknown, {out.skipped_fresh} fresh (skipped)")


# --- Hard filters ------------------------------------------------------------
def _filter_new_postings(summary: RunSummary, verbose: bool,
                         company_ids=None) -> list:
    """Apply hard filters + duplicate suppression to unscored postings."""
    rows = db.get_postings_to_score(company_ids=company_ids)
    PROGRESS.phase("filter", f"Applying hard filters to {len(rows)} new postings",
                   total=len(rows) or None)
    fcfg = matching.filter_config(refresh=True)   # one snapshot for this pass
    eligible = []
    dropped: list[tuple[int, str]] = []
    for i, p in enumerate(rows):
        reason = matching.hard_filter(p["title"], p["location"] or "",
                                      p["description"] or "",
                                      posted_date=p["posted_date"],
                                      first_seen=p["first_seen"], cfg=fcfg)
        if reason:
            dropped.append((p["id"], reason))
        else:
            eligible.append(p)
        if i % 500 == 499:               # cheap loop — tick in coarse strides
            PROGRESS.step(500)

    # Duplicate suppression: the same vacancy often exists under several
    # job_ids (reposts, per-slot requisitions). One fingerprint = one
    # recommendation, ever. Existing recommendations claim their fingerprint
    # first; among new twins the earliest-seen posting wins.
    taken: dict[str, int] = {}
    for r in db.recommended_open_postings():
        fp = matching.posting_fingerprint(r["company_id"], r["title"],
                                          r["location"] or "")
        taken.setdefault(fp, r["posting_id"])
    uniq = []
    dup_count = 0
    for p in sorted(eligible, key=lambda p: ((p["first_seen"] or ""), p["id"])):
        fp = matching.posting_fingerprint(p["company_id"], p["title"],
                                          p["location"] or "")
        keeper = taken.get(fp)
        if keeper is not None:
            dropped.append((p["id"], f"duplicate posting (same as #{keeper})"))
            dup_count += 1
        else:
            taken[fp] = p["id"]
            uniq.append(p)

    db.mark_postings_filtered(dropped)   # one transaction, not one per posting
    PROGRESS.step(len(rows) % 500)
    PROGRESS.add(filtered_out=len(dropped))
    summary.filtered_out += len(dropped)
    if verbose and dropped:
        print(f"  hard filters dropped {len(dropped)} postings "
              f"({dup_count} duplicates)")
    return uniq


def refilter_existing(verbose: bool = False) -> int:
    """Re-apply hard filters to undecided recommendations.

    Runs at the start of every pipeline run. When the rules or the preferences
    tighten, stale 'New' recommendations (e.g. internships surfaced by an older
    version) vanish from the board; Marked/Applied/Skipped/Rejected history is
    never touched."""
    PROGRESS.phase("refilter", "Re-checking existing recommendations against "
                               "the current rules")
    # The dashboard can change the hard-filter configuration between runs
    # (toggle a rule, edit its parameters, add/edit/delete custom filters).
    # When it did, every open posting we once excluded gets ONE fresh verdict
    # under the current rules: cleared verdicts re-enter the normal queue,
    # changed verdicts are relabeled in place. Non-filter verdicts
    # (duplicates, LLM floor scores, final review, queue aging) are never
    # touched — a filter edit must not resurrect an LLM judgment.
    rev = db.meta_get("hard_filters_rev", "0") or "0"
    if rev != (db.meta_get("hard_filters_rev_applied", "0") or "0"):
        fcfg = matching.filter_config(refresh=True)   # one snapshot
        changed = []
        rechecked = 0
        for p in db.open_filtered_postings():
            if matching.filter_key_for_reason(p["filtered_reason"]) is None:
                continue
            rechecked += 1
            new_reason = matching.hard_filter(
                p["title"], p["location"] or "", p["description"] or "",
                posted_date=p["posted_date"], first_seen=p["first_seen"],
                cfg=fcfg)
            if new_reason != p["filtered_reason"]:
                changed.append((p["id"], new_reason))
        if changed:
            db.mark_postings_filtered(changed)
        db.meta_set("hard_filters_rev_applied", rev)
        if verbose and rechecked:
            freed = sum(1 for _pid, r in changed if r is None)
            print(f"  filter config changed — re-judged {rechecked} excluded "
                  f"postings: {freed} released, {len(changed) - freed} relabeled")

    # Rules can also RELAX: postings once dropped as "outside target
    # countries" get a fresh location check, because that list is curated and
    # has had false positives ("Barcelona, CT, Spain" read as Connecticut).
    # Released postings re-enter the normal queue, where the full hard-filter
    # set judges them again this same run.
    released = 0
    for r in db.postings_filtered_as("outside target countries"):
        if matching.location_status(r["location"] or "") != "outside":
            db.mark_posting_filtered(r["id"], None)
            released += 1
    if verbose and released:
        print(f"  re-filter released {released} postings back into the queue "
              "(location rules relaxed)")

    removed = 0
    fcfg = matching.filter_config()
    for r in db.open_recommendations():
        reason = matching.hard_filter(r["title"], r["location"] or "",
                                      r["description"] or "",
                                      posted_date=r["posted_date"],
                                      first_seen=r["first_seen"], cfg=fcfg)
        # The recommendation floor also applies retroactively: an LLM-judged
        # New rec sitting below REC_MIN_SCORE should never have existed (the
        # user manually skipped 131 such rows before this rule). Real LLM
        # verdicts only — provisional heuristic scores stay for re-scoring.
        if not reason and r["llm_used"] \
                and (r["fit_score"] or 0) < config.REC_MIN_SCORE:
            reason = (f"scored {(r['fit_score'] or 0):.0f}/100 by the LLM — "
                      "below the recommendation floor")
        if reason:
            db.delete_recommendation(r["rec_id"])
            db.mark_posting_filtered(r["posting_id"], reason)
            removed += 1

    # Duplicate recommendations already on the board: keep ONE per
    # fingerprint. A user-decided rec always outranks a New one; among New
    # twins the higher fit (then the older rec) survives. Decided rows are
    # never deleted here, even if they duplicate each other.
    groups: dict[str, list] = {}
    for r in db.recommended_open_postings():
        fp = matching.posting_fingerprint(r["company_id"], r["title"],
                                          r["location"] or "")
        groups.setdefault(fp, []).append(r)
    for fp, rows in groups.items():
        if len(rows) < 2:
            continue
        decided = [r for r in rows if r["status"] != config.STATUS_NEW]
        news = [r for r in rows if r["status"] == config.STATUS_NEW]
        if decided:
            losers = news                      # a decision claims the vacancy
        else:
            news.sort(key=lambda r: (-(r["fit_score"] or 0), r["rec_id"]))
            losers = news[1:]
        for r in losers:
            db.delete_recommendation(r["rec_id"])
            db.mark_posting_filtered(
                r["posting_id"], "duplicate posting (same vacancy on the board)")
            removed += 1

    if verbose and removed:
        print(f"  re-filter removed {removed} stale recommendations from the board")
    return removed


# --- Match -------------------------------------------------------------------
def _match_new(summary: RunSummary, verbose: bool,
               tracker: matching.SpendTracker | None = None,
               company_ids=None) -> None:
    profile = db.get_profile()
    cv = profile["cv_text"] if profile else ""
    prefs = profile["preferences_note"] if profile else ""

    # Learned decision profiles: the user's Skipped/Applied history feeds both
    # the prefilter (rank known-rejected patterns down) and the LLM prompt.
    skip_cats, applied_cats = feedback.load_profiles(db)
    skip_rules = feedback.prompt_section(skip_cats, applied_cats)

    # 1) hard filters, then prefilter ranking. The ranking is ORDERING ONLY:
    #    every survivor gets scored (hard filters decide eligibility, Claude
    #    decides fit — nothing is denied a verdict by a keyword heuristic).
    #    Sorting best-first means a budget stop leaves the most promising
    #    postings already scored; the rest stay queued for the next run.
    scored_pool = []
    for p in _filter_new_postings(summary, verbose, company_ids=company_ids):
        pre, sig = matching.prefilter(p["title"], p["location"] or "",
                                      p["description"] or "",
                                      skip_cats, applied_cats)
        scored_pool.append((pre, sig, p))
    scored_pool.sort(key=lambda x: x[0], reverse=True)

    # 2) optional throttle (JOBSCOUT_PREFILTER_KEEP > 0): top-N per pass,
    #    fair-shared per company so one huge board can't eat every slot.
    #    Off by default — all survivors go through.
    if config.PREFILTER_KEEP > 0:
        candidates = []
        per_company: dict = {}
        for pre, sig, p in scored_pool:
            cid = p["company_id"]
            if per_company.get(cid, 0) >= config.PER_COMPANY_LLM_CAP:
                continue
            candidates.append((pre, sig, p))
            per_company[cid] = per_company.get(cid, 0) + 1
            if len(candidates) >= config.PREFILTER_KEEP:
                break
    else:
        candidates = scored_pool
    summary.prefiltered_in += len(candidates)   # += : run_tiered scores twice

    # 2b) dead-link gate: no posting may be scored (= LLM spend + a board row)
    #     while its job page is verifiably gone. Feed truth already closed
    #     postings that vanished from their boards during the fetch; this
    #     catches the rest — stale queue rows from failed fetches, partial
    #     browser boards, and listed jobs whose URL is broken. Postings the
    #     enrichment step below will fetch anyway are verified BY that fetch
    #     (its 404 maps to GONE), so nothing is requested twice.
    if config.LINKCHECK_ENABLED and candidates:
        from . import linkcheck
        n_checked, dead_links = linkcheck.check_candidates(
            [p for _pre, _sig, p in candidates], progress=PROGRESS)
        summary.links_checked += n_checked
        if dead_links:
            candidates = [c for c in candidates if c[2]["id"] not in dead_links]
            summary.links_dead += len(dead_links)
            PROGRESS.add(links_dead=len(dead_links))
            if verbose:
                print(f"  link check: {len(dead_links)} queued postings' job "
                      "pages are gone — archived unscored")

    # 3) enrich survivors that lack a description (Workday, SuccessFactors,
    #    SmartRecruiters detail endpoints). Detail pages are fetched with one
    #    worker PER COMPANY (a company's own requests stay sequential and
    #    throttled, exactly like the fetch phase) — with every survivor now
    #    scored, a serial loop over thousands of details would dominate the
    #    run's wall time.
    by_company: dict = {}
    for _pre, _sig, p in candidates:
        if p["ats_type"] and p["detail_ref"] and not (p["description"] or ""):
            by_company.setdefault(p["company_id"], []).append(p)
    details: dict = {}
    if by_company:
        n_details = sum(len(v) for v in by_company.values())
        PROGRESS.phase("prepare",
                       f"Fetching {n_details} missing descriptions "
                       f"({len(by_company)} companies in parallel)",
                       total=n_details)

        def _company_details(plist) -> dict:
            from . import linkcheck
            out: dict = {}
            for p in plist:
                try:
                    out[p["id"]] = fetch_detail(p["ats_type"], p["ats_token"],
                                                p["detail_ref"])
                except Exception as e:  # noqa: BLE001 — enrichment must never break a run
                    # A 404 here is a verdict, not an error: the ATS's own
                    # detail endpoint says this job no longer exists.
                    out[p["id"]] = linkcheck.GONE if linkcheck.gone_result(e) else {}
                PROGRESS.step(1, current=f"{p['company_name']} — {p['title']}")
            return out

        with ThreadPoolExecutor(
                max_workers=min(config.FETCH_WORKERS, len(by_company))) as ex:
            for fut in as_completed([ex.submit(_company_details, pl)
                                     for pl in by_company.values()]):
                details.update(fut.result())

    if candidates:
        PROGRESS.phase("prepare",
                       f"Preparing {len(candidates)} candidates for scoring",
                       total=len(candidates))
    jobs_for_llm = []
    for pre, sig, p in candidates:
        PROGRESS.step(1, current=f"{p['company_name']} — {p['title']}")
        title, company = p["title"], p["company_name"]
        location = p["location"] or ""
        posted = p["posted_date"] or ""
        desc = p["description"] or ""

        if not desc:
            d = details.get(p["id"]) or {}
            if d.get("_gone"):
                # The job page is gone (detail endpoint 404) — archive it
                # unscored instead of judging a corpse on its title alone.
                from . import linkcheck
                db.archive_dead_link(
                    p["id"], linkcheck.dead_reason("detail endpoint: job gone"))
                summary.links_dead += 1
                PROGRESS.add(links_dead=1)
                if verbose:
                    print(f"    gone  {p['company_name']} — {p['title'][:50]} "
                          "(job page removed — archived)")
                continue
            if d:
                desc = d.get("description") or desc
                posted = d.get("posted_date") or posted
                location = d.get("location") or location
                db.update_posting_enrichment(
                    p["id"], description=d.get("description"),
                    posted_date=d.get("posted_date"), url=d.get("url"),
                    location=d.get("location"))
                summary.enriched += 1
                # The enriched description may reveal a disqualifier the title
                # hid (required years, non-EN/DE language, internship body,
                # a real posting date past the age ceiling).
                reason = matching.hard_filter(title, location, desc,
                                              posted_date=posted or None,
                                              first_seen=p["first_seen"])
                if reason:
                    db.mark_posting_filtered(p["id"], reason)
                    summary.filtered_out += 1
                    continue
        jobs_for_llm.append({
            "id": p["id"], "title": title, "company": company,
            "location": location, "posted": posted, "description": desc,
            "pre_score": pre, "signals": sig,
        })

    # 4) score in batches — Sonnet with prompt caching; budget-capped.
    # When the LLM is configured but a batch fails (or the budget runs out),
    # those postings are NOT heuristic-scored: a keyword guess must never wear
    # the same badge as a real judgment. They stay queued and get scored on
    # the next run. The heuristic only runs in keyless mode (and is capped).
    if tracker is None:
        tracker = matching.SpendTracker()
    run_spend_before = tracker.spent
    llm_on = matching.llm_available()
    llm_pending = 0
    batch_size = max(1, config.LLM_BATCH_SIZE)
    api_down = llm_on and bool(matching.llm_fatal_reason)
    if jobs_for_llm:
        if api_down:
            PROGRESS.phase("score",
                           f"Anthropic API unavailable — {len(jobs_for_llm)} "
                           "postings stay queued",
                           total=len(jobs_for_llm))
        else:
            scorer = "Claude" if llm_on else "the keyless heuristic"
            PROGRESS.phase("score",
                           f"Scoring {len(jobs_for_llm)} postings with {scorer}",
                           total=len(jobs_for_llm))
    for i in range(0, len(jobs_for_llm), batch_size):
        batch = jobs_for_llm[i:i + batch_size]
        PROGRESS.step(0, current=f"{batch[0]['company']} — {batch[0]['title']}"
                      + (f" (+{len(batch) - 1} more)" if len(batch) > 1 else ""))
        results: dict = {}
        if llm_on and not tracker.exhausted:
            results = matching.score_batch(cv, prefs, batch, tracker,
                                           extra_rules=skip_rules)
        elif llm_on and tracker.exhausted and i > 0:
            if not any("budget cap" in n for n in summary.notes):
                summary.notes.append(
                    f"LLM budget cap (${tracker.cap:.2f}) reached — remaining "
                    "postings stay queued for the next run")
        for job in batch:
            res = results.get(job["id"])
            if res is None:
                if llm_on:
                    llm_pending += 1   # scored next run — no fake scores
                    continue
                res = matching.heuristic_score(
                    job["title"], job["company"], job["location"],
                    job["description"], job["pre_score"], job["signals"])
            # Recommendation floor: a real LLM judgment far below the
            # shortlist band means "clearly off-profile" — archive the posting
            # with its verdict instead of writing a board row nobody wants.
            if res.llm_used and res.fit_score < config.REC_MIN_SCORE:
                why = (res.rationale or "").strip()
                db.mark_posting_filtered(
                    job["id"], f"scored {res.fit_score:.0f}/100 by the LLM"
                               + (f" — {why[:160]}" if why else ""))
                summary.floor_rejected += 1
                summary.llm_used += 1
                PROGRESS.add(floor_rejected=1)
                if verbose:
                    print(f"    {res.fit_score:5.1f} [LLM] {job['company']} — "
                          f"{job['title'][:50]}  (below floor — archived)")
                continue
            db.save_recommendation(
                posting_id=job["id"], prefilter_score=job["pre_score"],
                fit_score=res.fit_score, rationale=res.rationale,
                watch_outs=res.watch_outs, llm_used=res.llm_used)
            summary.scored += 1
            if res.llm_used:
                summary.llm_used += 1
            if res.fit_score >= config.SHORTLIST_MIN_SCORE:
                summary.surfaced += 1
            if res.fit_score >= config.SHORTLIST_MIN_SCORE:
                PROGRESS.event(f"★ {res.fit_score:.0f} — {job['company']}: "
                               f"{job['title'][:70]}")
            if verbose:
                flag = "LLM" if res.llm_used else "heur"
                print(f"    {res.fit_score:5.1f} [{flag}] {job['company']} — "
                      f"{job['title'][:50]}")
        PROGRESS.step(len(batch))
        PROGRESS.add(scored=len([j for j in batch if results.get(j["id"])])
                     if llm_on else len(batch))
        PROGRESS.set(llm_cost=tracker.spent)
    if llm_pending:
        if matching.llm_fatal_reason:
            summary.notes.append(
                f"{llm_pending} postings could not be scored — "
                f"{matching.llm_fatal_reason}. They stay queued; re-run once "
                "the API works and they will be judged then.")
        else:
            summary.notes.append(
                f"{llm_pending} postings not scored (LLM error/budget) — "
                "queued for next run")

    # Janitor: every survivor is entitled to a verdict, so the only way a
    # posting sits unscored is repeated budget stops or LLM errors. Ten days
    # of retries is plenty — retire older leftovers so the queue can't grow
    # without bound. Scoped to the same companies as this pass: a partial run
    # ("Apple only") must never age out other companies' queued postings
    # before a full run has had the chance to score them.
    PROGRESS.phase("cleanup", "Retiring postings that aged out of the queue")
    max_q_days = config.SCORE_QUEUE_MAX_DAYS
    if matching.llm_fatal_reason:
        # "Ten days of retries is plenty" assumes retries HAPPENED. With the
        # API down nothing was retried, so retiring the queue would destroy
        # the very backlog the next working run has to score. (Live: the
        # 2026-09-01 run threw away 354 postings this way while the account
        # had no credit.) Leave the queue alone and say so.
        summary.notes.append(
            "queue ageing skipped — nothing could be scored this run, so "
            "queued postings keep their retries")
        aged = []
    else:
        aged = [(p["id"], f"not scored within {max_q_days} days (LLM budget/errors)")
                for p in db.get_postings_to_score(company_ids=company_ids)
                if (p["first_seen"] or "")[:10] < _days_ago_iso(max_q_days)]
    if aged:
        db.mark_postings_filtered(aged)
        summary.notes.append(f"{len(aged)} old postings aged out of the queue")

    summary.llm_errors = matching.llm_error_count
    phase_spend = tracker.spent - run_spend_before
    summary.llm_cost_usd += phase_spend
    if phase_spend:
        total = db.meta_add_float("llm_spend_usd_total", phase_spend)
        db.meta_set("llm_spend_usd_last_run", f"{summary.llm_cost_usd:.4f}")
        if verbose:
            print(f"  LLM cost this phase ≈ ${phase_spend:.3f} "
                  f"(cumulative ≈ ${total:.2f})")


def _validate_board(summary: RunSummary, verbose: bool,
                    tracker: matching.SpendTracker,
                    company_ids=None) -> None:
    """Final LLM quality gate over the board (see validate.py)."""
    from . import validate
    spend_before = tracker.spent
    outcome = validate.validate_new_recommendations(
        tracker=tracker, verbose=verbose, progress=PROGRESS,
        company_ids=company_ids)
    summary.validated += outcome.reviewed
    summary.validation_removed += outcome.removed
    summary.notes.extend(outcome.notes)
    phase_spend = tracker.spent - spend_before
    summary.llm_cost_usd += phase_spend
    if phase_spend:
        db.meta_add_float("llm_spend_usd_total", phase_spend)
        db.meta_set("llm_spend_usd_last_run", f"{summary.llm_cost_usd:.4f}")
    if outcome.removed:
        PROGRESS.add(validation_removed=outcome.removed)
        PROGRESS.event(f"final review removed {outcome.removed} of "
                       f"{outcome.reviewed} reviewed")
    if verbose and outcome.reviewed:
        print(f"  final review: {outcome.reviewed} reviewed, "
              f"{outcome.removed} removed, {outcome.flagged} flagged")


def _llm_preflight(summary: RunSummary, verbose: bool) -> None:
    """Check ONCE, before any work, that Claude can actually be called.

    Without this a dead key (no credit, revoked, wrong model) is discovered
    only batch by batch: the run spends its whole wall time retrying, scores
    nothing, and still reports "complete" — which is exactly how the board
    silently stopped growing. A single one-token probe turns that into an
    explicit, up-front verdict that every later phase can act on.

    Keyless mode is untouched: with no key configured the run uses the
    heuristic scorer exactly as before. JOBSCOUT_PREFLIGHT=0 disables the
    probe (the test suite sets it, so a run under pytest never touches the
    network); a permanent failure is then still caught by the first real
    call."""
    matching.reset_llm_state()
    if not config.ANTHROPIC_API_KEY:
        return
    if os.environ.get("JOBSCOUT_PREFLIGHT", "1").strip().lower() in (
            "0", "false", "no", "off"):
        return
    PROGRESS.phase("starting", "Checking the Anthropic API…")
    reason = matching.preflight()
    if reason:
        summary.llm_fatal = reason
        summary.notes.insert(
            0, f"Anthropic API unavailable — {reason}. Nothing was scored and "
               "no new roles could reach the board; feeds were still refreshed, "
               "so a re-run once the API works will score the backlog.")
        PROGRESS.event(f"✗ Anthropic API unavailable — {reason}")
        if verbose:
            print(f"  ! Anthropic API unavailable — {reason}")


def _finish(summary: RunSummary) -> RunSummary:
    # A permanent API failure discovered mid-run (not caught by the preflight)
    # still has to lead the summary — it explains an otherwise inexplicable
    # "0 scored, 0 surfaced" result.
    summary.llm_fatal = summary.llm_fatal or matching.llm_fatal_reason
    if summary.llm_fatal and not any(
            n.startswith("Anthropic API unavailable") for n in summary.notes):
        # The outage can be found by the preflight, by the scorer, or by the
        # browser tier — whichever it was, the reason leads the summary.
        summary.notes.insert(
            0, f"Anthropic API unavailable — {summary.llm_fatal}. "
               "Scoring stopped; the queued postings stay for the next run.")
    # run_tiered scores twice (feed tier, then browser tier), so per-phase
    # notes arrive in duplicate. The summary is read by a human and capped at
    # 12 lines — a repeat costs one of those lines and adds nothing.
    seen: set = set()
    summary.notes = [n for n in summary.notes
                     if not (n in seen or seen.add(n))]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = json.dumps(summary.as_dict())
    db.meta_set("last_run_at", now)
    db.meta_set("last_run_summary", payload)
    db.record_run(now, payload)          # feeds the dashboard's Run History
    PROGRESS.phase("done", "Run complete")
    return summary


def _select_companies(detected, only, summary: RunSummary):
    """Apply an optional company selection to the detected list.

    Returns (companies_to_fetch, company_ids_scope). Scope is None for a full
    run — every downstream phase then behaves exactly as before."""
    if not only:
        return detected, None
    chosen = resolve_company_selection(only)          # raises on unknown names
    chosen_ids = {c["id"] for c in chosen}
    fetchable = [c for c in detected if c["id"] in chosen_ids]
    # Selected companies that are not fetchable (unsupported/undetected) must
    # be surfaced, not silently skipped.
    fetchable_ids = {c["id"] for c in fetchable}
    for c in chosen:
        if c["id"] not in fetchable_ids:
            summary.notes.append(
                f"{c['name']}: selected but not fetchable "
                f"({c['detect_status']}{' / ' + c['ats_type'] if c['ats_type'] else ''})")
    summary.notes.insert(0, "partial run: " + ", ".join(
        sorted(c["name"] for c in chosen)))
    return fetchable, sorted(chosen_ids)


def run(detect_first: bool = False, verbose: bool = True,
        skip_browser: bool = False, only=None) -> RunSummary:
    """One full pipeline pass. `only` = optional list of company names — the
    run then fetches, filters, scores and validates JUST those companies
    (global board hygiene still applies)."""
    db.init_db()
    PROGRESS.reset()
    summary = RunSummary()
    _llm_preflight(summary, verbose)
    if detect_first:
        if verbose:
            print("Detecting ATS for pending companies...")
        detect_all(verbose=verbose)
    apply_curated_detection()
    _prune_dead_links(summary, verbose)   # before the hard filters see anything
    refilter_existing(verbose)
    detected = db.get_companies(only_detected=True)
    detected, scope_ids = _select_companies(detected, only, summary)
    if skip_browser:
        detected = [c for c in detected if c["ats_type"] != "browser"]
    if verbose:
        print(f"Fetching {len(detected)} feeds...")
    _fetch_and_store(summary, verbose, detected,
                     phase_label=f"Fetching {len(detected)} company feeds")
    if verbose:
        print("Matching new postings...")
    tracker = matching.SpendTracker()
    _match_new(summary, verbose, tracker, company_ids=scope_ids)
    _validate_board(summary, verbose, tracker, company_ids=scope_ids)
    return _finish(summary)


def run_tiered(detect_first: bool = True, verbose: bool = True,
               only=None) -> RunSummary:
    """Production run: detect, then fetch+score the fast FEED tier first (so the
    core board is complete + scored quickly), then the slower BROWSER tier and
    score again. Safe to interrupt after the feed phase — that board is persisted.

    `only` = optional list of company names for a partial run (see `run`).
    """
    db.init_db()
    PROGRESS.reset()
    summary = RunSummary()
    _llm_preflight(summary, verbose)
    if detect_first:
        if verbose:
            print("Detecting ATS (curated, fast)...")
        detect_all(verbose=verbose, known_only=True)
    apply_curated_detection()
    _prune_dead_links(summary, verbose)   # before the hard filters see anything
    refilter_existing(verbose)
    detected = db.get_companies(only_detected=True)
    detected, scope_ids = _select_companies(detected, only, summary)
    feeds = [c for c in detected if c["ats_type"] != "browser"]
    browsers = [c for c in detected if c["ats_type"] == "browser"]

    tracker = matching.SpendTracker()   # one budget across both phases
    if verbose:
        print(f"\n[Tier 1-2] Fetching {len(feeds)} feed companies...")
    _fetch_and_store(summary, verbose, feeds,
                     phase_label=f"Tier 1-2 · Fetching {len(feeds)} feed companies")
    if verbose:
        print("[Tier 1-2] Scoring feed postings...")
    _match_new(summary, verbose, tracker, company_ids=scope_ids)

    if browsers or not only:
        if verbose:
            print(f"\n[Tier 3] Fetching {len(browsers)} browser-tier companies "
                  "(HTTP-first; SPAs need Playwright)...")
        # Fewer workers here: each SPA may launch its own headless Chromium.
        _fetch_and_store(summary, verbose, browsers, max_workers=3,
                         phase_label=f"Tier 3 · Fetching {len(browsers)} browser-tier "
                                     "companies (may be slow)")
        if verbose:
            print("[Tier 3] Scoring any new browser-tier postings...")
        _match_new(summary, verbose, tracker, company_ids=scope_ids)
    _validate_board(summary, verbose, tracker, company_ids=scope_ids)
    return _finish(summary)

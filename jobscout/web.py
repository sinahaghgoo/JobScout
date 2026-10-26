"""JobScout dashboard — a small local FastAPI web app.

A polished single page that watches your target companies and shows only the
roles that fit your CV and goals. Replaces the POC Streamlit dashboard.

Run:
    python -m jobscout.web            # then open http://127.0.0.1:8000
    python -m jobscout.web --port 8080 --no-browser
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, db, matching, profile as profile_mod
from . import pipeline

STATIC_DIR = Path(__file__).resolve().parent / "web_static"

app = FastAPI(title="JobScout", docs_url=None, redoc_url=None)

# --- Background pipeline run state -------------------------------------------
_run_lock = threading.Lock()
_run_state: dict = {"running": False, "started_at": None, "finished_at": None,
                    "summary": None, "error": None}

# --- Idle auto-shutdown --------------------------------------------------------
# The dashboard is a local tool. The open page sends a heartbeat every 25 s
# (and every real interaction counts too, via middleware); when every tab has
# been closed for WEB_IDLE_MINUTES — and no pipeline run is active — the
# server exits on its own, so the port never stays occupied behind a closed
# tab. 0 disables the watchdog.
WEB_IDLE_MINUTES = float(os.environ.get("JOBSCOUT_WEB_IDLE_MINUTES", "10"))
_last_activity = time.time()
_server = None                     # uvicorn.Server, set in main()


@app.middleware("http")
async def _touch_activity(request, call_next):
    global _last_activity
    _last_activity = time.time()
    return await call_next(request)


@app.get("/api/ping")
def api_ping() -> dict:
    """Identifies a running JobScout instance (used by the port probe)."""
    return {"app": "jobscout", "ok": True}


@app.post("/api/heartbeat")
def api_heartbeat() -> dict:
    return {"ok": True}            # the middleware already logged the activity


def _idle_watchdog(idle_seconds: float, poll_seconds: float = 15.0) -> None:
    """Exit the server once no tab has spoken for `idle_seconds`.

    A running pipeline counts as activity (a run kicked off before closing
    the tab always completes; idleness starts from its end), so work is
    never interrupted and the DB is never touched mid-write."""
    global _last_activity
    while True:
        time.sleep(poll_seconds)
        if _run_state["running"]:
            _last_activity = time.time()
            continue
        if time.time() - _last_activity > idle_seconds:
            print("JobScout dashboard idle — no open tab for "
                  f"{idle_seconds / 60:.0f} min; shutting down (port freed).")
            if _server is not None:
                _server.should_exit = True
            return


def _row_to_dict(r) -> dict:
    return {k: r[k] for k in r.keys()}


def _board_payload(r) -> dict:
    wo = (r["watch_outs"] or "").strip()
    if wo.lower() in ("none", "none.", ""):
        wo = ""
    desc = (r["description"] or "").strip()
    if len(desc) > 1800:
        desc = desc[:1800].rsplit(" ", 1)[0] + " …"
    return {
        "rec_id": r["rec_id"],
        "title": r["title"],
        "company": r["company"],
        "category": r["category"],
        "location": r["location"] or "",
        "posted_date": r["posted_date"] or "",
        "date_confidence": r["date_confidence"] or "high",
        "fit_score": None if r["fit_score"] is None else round(r["fit_score"], 1),
        "prefilter_score": None if r["prefilter_score"] is None else round(r["prefilter_score"], 1),
        "rationale": r["rationale"] or "",
        "watch_outs": wo,
        "status": r["status"],
        "notes": r["notes"] or "",
        "llm_used": bool(r["llm_used"]),
        "validated": bool(r["validated_at"]),
        "validation_note": r["validation_note"] or "",
        "url": r["url"] or "",
        "interview_pct": r["interview_pct"],
        "description": desc,
        "is_open": bool(r["is_open"]),
        "first_seen": (r["rec_first_seen"] or "")[:10],
    }


# --- API ---------------------------------------------------------------------
def _composition() -> dict:
    """Where every open posting stands right now — the segments sum to
    postings_open, so the dashboard can draw them as one bar."""
    with db.connect() as conn:
        def one(sql, *a):
            return conn.execute(sql, a).fetchone()[0]
        open_total = one("SELECT COUNT(*) FROM posting WHERE is_open=1")
        filtered = one("SELECT COUNT(*) FROM posting WHERE is_open=1 "
                       "AND filtered_reason IS NOT NULL")
        queued = one("""SELECT COUNT(*) FROM posting p
                        WHERE p.is_open=1 AND p.filtered_reason IS NULL
                          AND NOT EXISTS (SELECT 1 FROM recommendation r
                                          WHERE r.posting_id = p.id)""")
        below = one("""SELECT COUNT(*) FROM recommendation r
                       JOIN posting p ON p.id = r.posting_id
                       WHERE p.is_open=1 AND r.status='New'
                         AND r.fit_score < ?""", config.SHORTLIST_MIN_SCORE)
        # Skipped/Rejected recs on open postings live in the ARCHIVE, not on
        # the board — counting them as "board" overstated that segment.
        archived = one("""SELECT COUNT(*) FROM recommendation r
                          JOIN posting p ON p.id = r.posting_id
                          WHERE p.is_open=1
                            AND r.status IN ('Skipped','Rejected')""")
    return {"open": open_total, "filtered": filtered, "queued": queued,
            "below_shortlist": below, "archived": archived,
            "board": max(0, open_total - filtered - queued - below - archived)}


def _last_run() -> dict:
    """The most recent finished run — read from meta so it survives restarts."""
    import json as _json
    raw = db.meta_get("last_run_summary")
    summary = None
    if raw:
        try:
            summary = _json.loads(raw)
        except ValueError:
            summary = None
    return {"at": db.meta_get("last_run_at"), "summary": summary}


@app.get("/api/stats")
def api_stats() -> dict:
    db.init_db()
    s = db.stats()
    s["surfaced"] = _count_surfaced()
    s["composition"] = _composition()
    s["last_run"] = _last_run()
    with db.connect() as conn:
        s["companies_with_jobs"] = conn.execute(
            """SELECT COUNT(DISTINCT company_id) FROM posting
               WHERE is_open=1""").fetchone()[0]
    # tier breakdown across the target list
    tiers = {"feed": 0, "browser": 0, "residual": 0}
    for c in db.get_companies():
        tiers[_tier_of(c["ats_type"], c["detect_status"])] += 1
    s["companies_feed"] = tiers["feed"]
    s["companies_browser"] = tiers["browser"]
    s["companies_residual"] = tiers["residual"]
    s["llm_enabled"] = matching.llm_available()
    s["model"] = config.ANTHROPIC_MODEL if matching.llm_available() else "heuristic"
    # A permanent API failure (no credit, revoked key, forbidden model) is the
    # one condition that quietly empties the board: nothing can be scored, so
    # no new roles arrive while the age filter keeps retiring old ones. Report
    # it explicitly — live if this process saw it, otherwise from the last run
    # — so the dashboard can say so instead of showing a green "run complete".
    fatal = matching.llm_fatal_reason
    if not fatal:
        lr = s.get("last_run") or {}
        fatal = ((lr.get("summary") or {}).get("llm_fatal") or "")
    s["llm_fatal"] = fatal
    s["shortlist_min"] = config.SHORTLIST_MIN_SCORE
    s["llm_spend_total"] = float(db.meta_get("llm_spend_usd_total", "0") or 0)
    s["llm_spend_last_run"] = float(db.meta_get("llm_spend_usd_last_run", "0") or 0)
    s["last_run_at"] = db.meta_get("last_run_at")
    s["runs_recorded"] = db.count_runs()
    with _run_lock:
        s["run"] = {k: _run_state[k] for k in ("running", "started_at", "finished_at",
                                               "summary", "error")}
    s["has_profile"] = db.get_profile() is not None
    return s


def _count_surfaced() -> int:
    rows = db.board_rows(statuses=["New", "Marked", "Applied"])
    return sum(1 for r in rows if (r["fit_score"] or 0) >= config.SHORTLIST_MIN_SCORE)


@app.get("/api/board")
def api_board(status: str = "New,Marked,Applied", min_fit: float = 0.0,
              company: str = "", sort: str = "fit") -> dict:
    statuses = [s for s in status.split(",") if s] or None
    rows = [_board_payload(r) for r in db.board_rows(statuses=statuses)]
    comp_filter = {c for c in company.split(",") if c}
    rows = [r for r in rows
            if (r["fit_score"] is None or r["fit_score"] >= min_fit)
            and (not comp_filter or r["company"] in comp_filter)]
    if sort == "posted":
        rows.sort(key=lambda r: (r["posted_date"] or ""), reverse=True)
    elif sort == "company":
        rows.sort(key=lambda r: r["company"].lower())
    else:  # fit
        rows.sort(key=lambda r: (r["fit_score"] if r["fit_score"] is not None else -1),
                  reverse=True)
    return {"rows": rows, "count": len(rows)}


@app.get("/api/archive")
def api_archive() -> dict:
    rows = [_board_payload(r) for r in db.board_rows(statuses=["Skipped", "Rejected"])]
    return {"rows": rows, "count": len(rows)}


class StatusBody(BaseModel):
    status: str


class NotesBody(BaseModel):
    notes: str


@app.post("/api/rec/{rec_id}/status")
def api_set_status(rec_id: int, body: StatusBody) -> dict:
    if body.status not in config.ALL_STATUSES:
        raise HTTPException(400, f"invalid status: {body.status}")
    db.set_status(rec_id, body.status)
    return {"ok": True}


@app.post("/api/rec/{rec_id}/notes")
def api_set_notes(rec_id: int, body: NotesBody) -> dict:
    db.set_notes(rec_id, body.notes)
    return {"ok": True}


def _tier_of(ats_type: Optional[str], detect_status: str) -> str:
    """feed = live structured feed; browser = Tier-3 (needs local browser run);
    residual = no reachable feed (anti-bot / login / apply-by-email)."""
    if detect_status == "detected" and ats_type and ats_type != "browser":
        return "feed"
    if ats_type == "browser":
        return "browser"
    return "residual"


@app.get("/api/companies")
def api_companies() -> dict:
    comps = db.get_companies()
    # per-company posting counts: open, eligible (passed hard filters), and how
    # many made the board — so Coverage shows real signal, not just raw volume.
    with db.connect() as conn:
        open_counts = {r["company_id"]: r["n"] for r in conn.execute(
            "SELECT company_id, COUNT(*) n FROM posting WHERE is_open=1 GROUP BY company_id")}
        eligible_counts = {r["company_id"]: r["n"] for r in conn.execute(
            """SELECT company_id, COUNT(*) n FROM posting
               WHERE is_open=1 AND filtered_reason IS NULL GROUP BY company_id""")}
        surfaced_counts = {r["company_id"]: r["n"] for r in conn.execute(
            """SELECT p.company_id, COUNT(*) n
               FROM recommendation r JOIN posting p ON p.id = r.posting_id
               WHERE p.is_open=1 AND r.fit_score >= ?
               GROUP BY p.company_id""", (config.SHORTLIST_MIN_SCORE,))}
    out = []
    by_ats: dict[str, int] = {}
    tiers = {"feed": 0, "browser": 0, "residual": 0}
    empty = 0
    for c in comps:
        d = _row_to_dict(c)
        tier = _tier_of(d["ats_type"], d["detect_status"])
        tiers[tier] += 1
        jobs = open_counts.get(d["id"], 0)
        if jobs == 0:
            empty += 1
        item = {
            "name": d["name"], "category": d["category"],
            "locations": d.get("locations") or "",
            "ats_type": d["ats_type"], "ats_token": d["ats_token"],
            "detect_status": d["detect_status"], "detect_note": d.get("detect_note") or "",
            "tier": tier, "open_postings": jobs,
            "eligible_postings": eligible_counts.get(d["id"], 0),
            "surfaced": surfaced_counts.get(d["id"], 0),
            "last_fetch_at": (d.get("last_fetch_at") or "")[:16].replace("T", " "),
            "last_fetch_status": d.get("last_fetch_status") or "",
        }
        out.append(item)
        if tier == "feed":
            by_ats[d["ats_type"]] = by_ats.get(d["ats_type"], 0) + 1
    tier_rank = {"feed": 0, "browser": 1, "residual": 2}
    out.sort(key=lambda x: (tier_rank[x["tier"]], -x["open_postings"], x["category"] or "", x["name"]))
    return {"companies": out, "total": len(out),
            "detected": tiers["feed"] + tiers["browser"],
            "tiers": tiers, "empty": empty,
            "with_jobs": len(out) - empty,
            "filtered_reasons": db.filtered_stats(),
            "by_ats": dict(sorted(by_ats.items(), key=lambda kv: -kv[1]))}


# --- Hard filters (dashboard-managed) -----------------------------------------
_ALLOWED_FIELDS = set(matching.CUSTOM_FIELDS)
_ALLOWED_MATCH = {"word", "contains"}
_FLOOR_REASON = re.compile(r"^scored \d+(\.\d+)?/100 by the LLM")


def _filters_changed() -> None:
    """Every filter write lands here: bump the revision the pipeline checks
    at run start, and drop matching's cache so the change is live at once."""
    db.bump_filters_rev()
    matching.invalidate_filter_cache()


def _clean_terms(terms, what: str = "terms", max_terms: int = 40) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for t in terms or []:
        t = " ".join(str(t).split())
        if not t:
            continue
        if len(t) > 80:
            raise HTTPException(400, f"{what}: '{t[:24]}…' is longer than 80 characters")
        if t.lower() in seen:
            continue                      # silently drop exact duplicates
        seen.add(t.lower())
        out.append(t)
    if len(out) > max_terms:
        raise HTTPException(400, f"{what}: at most {max_terms} entries")
    return out


def _clean_name(raw: Optional[str]) -> str:
    name = " ".join((raw or "").split())
    if not name:
        raise HTTPException(400, "Give the filter a name")
    if len(name) > 60:
        raise HTTPException(400, "Name: at most 60 characters")
    return name


def _ensure_unique_name(name: str, exclude_key: Optional[str] = None) -> None:
    taken = {f["name"].lower(): f["key"] for f in matching.BUILTIN_HARD_FILTERS}
    for r in db.get_hard_filters():
        if not r["builtin"] and r["name"]:
            taken.setdefault(r["name"].lower(), r["key"])
    owner = taken.get(name.lower())
    if owner is not None and owner != exclude_key:
        raise HTTPException(400, f"A filter named '{name}' already exists")


def _other_exclusion_label(reason: str) -> str:
    """Group the non-hard-filter verdicts under a few stable labels."""
    if reason.startswith("duplicate posting"):
        return "Duplicate postings"
    if reason.startswith("job page gone"):
        return "Dead links (job page gone)"
    if reason.startswith("failed final review"):
        return "Removed by the final review"
    if _FLOOR_REASON.match(reason):
        return "Below the recommendation floor (LLM verdict)"
    if reason.startswith("not scored within"):
        return "Aged out of the queue unscored"
    return reason[:80]


def _row_params(row) -> dict:
    try:
        p = json.loads(row["params"] or "{}")
        return p if isinstance(p, dict) else {}
    except ValueError:
        return {}


def _filter_list() -> tuple[list[dict], dict]:
    """Registry + stored overrides + custom rows, with per-filter counts of
    currently excluded open postings. Returns (filters, other_exclusions)."""
    rows = {r["key"]: r for r in db.get_hard_filters()}
    counts: dict[str, int] = {}
    other: dict[str, int] = {}
    for reason, n in db.filtered_stats().items():
        k = matching.filter_key_for_reason(reason)
        if k is None:
            lbl = _other_exclusion_label(reason)
            other[lbl] = other.get(lbl, 0) + n
        else:
            counts[k] = counts.get(k, 0) + n
    out = []
    for spec in matching.BUILTIN_HARD_FILTERS:
        key = spec["key"]
        row = rows.get(key)
        params = _row_params(row) if row else {}
        enabled = (bool(row["enabled"]) if row and row["enabled"] is not None
                   else matching.default_filter_enabled(key))
        eff = dict(matching.default_filter_params(key))
        eff.update({k: v for k, v in params.items() if k != "extra_terms"})
        out.append({
            "key": key, "name": spec["name"], "what": spec["what"],
            "builtin": True, "enabled": enabled,
            "editable": list(spec["editable"]),
            "extra_field": spec["extra_field"],
            "preview": list(spec.get("preview") or []),
            "details": list(spec.get("details") or []),
            "params": eff,
            "extra_terms": params.get("extra_terms") or [],
            "extra_groups": params.get("extra_groups") or {},
            "excluded_now": counts.get(key, 0),
        })
    customs = []
    for key, row in rows.items():
        if row["builtin"]:
            continue
        params = _row_params(row)
        name = row["name"] or key
        customs.append({
            "key": key, "name": name,
            "what": params.get("description") or "",
            "category": params.get("category") or "", "builtin": False,
            "enabled": True if row["enabled"] is None else bool(row["enabled"]),
            "editable": [], "extra_field": None,
            "preview": [], "details": [],
            "params": {"field": params.get("field") or "title",
                       "match": params.get("match") or "word",
                       "terms": params.get("terms") or []},
            "extra_terms": [], "extra_groups": {},
            "excluded_now": counts.get("custom:" + name, 0),
        })
    # categorized filters cluster together (A→Z); uncategorized go last
    customs.sort(key=lambda f: ((f["category"] or "\uffff").lower(),
                                f["name"].lower()))
    out.extend(customs)
    return out, other


@app.get("/api/filters")
def api_filters() -> dict:
    db.init_db()          # first hit after an upgrade must create the table
    filters, other = _filter_list()
    active = sum(1 for f in filters if f["enabled"])
    excluded = sum(f["excluded_now"] for f in filters)
    return {
        "filters": filters,
        "fields": sorted(_ALLOWED_FIELDS),
        "stats": {
            "active": active,
            "inactive": len(filters) - active,
            "custom": sum(1 for f in filters if not f["builtin"]),
            "excluded_total": excluded,
            "other_exclusions": [{"label": k, "count": v}
                                 for k, v in sorted(other.items(),
                                                    key=lambda kv: -kv[1])],
        },
        "changed_since_run": (db.meta_get("hard_filters_rev", "0") or "0")
                             != (db.meta_get("hard_filters_rev_applied", "0") or "0"),
    }


class FilterBody(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    enabled: Optional[bool] = None
    field: Optional[str] = None
    match: Optional[str] = None
    terms: Optional[list[str]] = None
    max_days: Optional[int] = None
    max_years: Optional[int] = None
    extra_terms: Optional[list[str]] = None
    extra_groups: Optional[dict[str, list[str]]] = None


def _clean_category(raw: str) -> str:
    cat = " ".join((raw or "").split())
    if len(cat) > 60:
        raise HTTPException(400, "Category: at most 60 characters")
    return cat


def _clean_extra_groups(raw: dict) -> dict:
    """Per-category user terms for a built-in filter: {label: [terms]}.
    Empty categories vanish; labels are trimmed; terms get the usual checks."""
    if len(raw) > 20:
        raise HTTPException(400, "extra terms: at most 20 categories")
    groups: dict[str, list[str]] = {}
    for label, terms in raw.items():
        lbl = " ".join(str(label).split())
        if not lbl:
            continue
        if len(lbl) > 80:
            raise HTTPException(400, f"category '{lbl[:24]}…': name too long")
        cleaned = _clean_terms(terms, what=f"extra terms ({lbl})")
        if cleaned:
            groups[lbl] = cleaned
    return groups


def _clean_description(raw: str) -> str:
    desc = " ".join((raw or "").split())
    if len(desc) > 200:
        raise HTTPException(400, "Description: at most 200 characters")
    return desc


@app.post("/api/filters")
def api_create_filter(body: FilterBody) -> dict:
    """Create a custom hard filter. It participates in filtering exactly like
    the built-in rules, starting with the next pipeline run."""
    db.init_db()
    name = _clean_name(body.name)
    _ensure_unique_name(name)
    fieldv = body.field or "title"
    if fieldv not in _ALLOWED_FIELDS:
        raise HTTPException(400, f"field must be one of: {', '.join(sorted(_ALLOWED_FIELDS))}")
    match = body.match or "word"
    if match not in _ALLOWED_MATCH:
        raise HTTPException(400, "match must be 'word' or 'contains'")
    terms = _clean_terms(body.terms)
    if not terms:
        raise HTTPException(400, "Add at least one term to match")
    params = {"field": fieldv, "match": match, "terms": terms}
    desc = _clean_description(body.description)
    if desc:
        params["description"] = desc
    cat = _clean_category(body.category)
    if cat:
        params["category"] = cat
    key = "custom-" + uuid4().hex[:10]
    db.save_hard_filter(key, builtin=False, name=name,
                        enabled=True if body.enabled is None else bool(body.enabled),
                        params_json=json.dumps(params))
    _filters_changed()
    return {"ok": True, "key": key}


@app.post("/api/filters/{key}")
def api_update_filter(key: str, body: FilterBody) -> dict:
    db.init_db()
    spec = matching.builtin_spec(key)
    row = db.get_hard_filter(key)
    if spec is None and row is None:
        raise HTTPException(404, f"no filter '{key}'")

    if spec is not None:                          # built-in: enabled + params
        params = _row_params(row) if row else {}
        if body.max_days is not None:
            if "max_days" not in spec["editable"]:
                raise HTTPException(400, f"'{spec['name']}' has no day ceiling")
            if not 1 <= int(body.max_days) <= 365:
                raise HTTPException(400, "max_days must be between 1 and 365")
            params["max_days"] = int(body.max_days)
        if body.max_years is not None:
            if "max_years" not in spec["editable"]:
                raise HTTPException(400, f"'{spec['name']}' has no years ceiling")
            if not 0 <= int(body.max_years) <= 30:
                raise HTTPException(400, "max_years must be between 0 and 30")
            params["max_years"] = int(body.max_years)
        if body.extra_terms is not None:
            if not spec["extra_field"]:
                raise HTTPException(400, f"'{spec['name']}' does not take extra terms")
            extra = _clean_terms(body.extra_terms, what="extra terms")
            if extra:
                params["extra_terms"] = extra
            else:
                params.pop("extra_terms", None)
        if body.extra_groups is not None:
            if not spec["extra_field"]:
                raise HTTPException(400, f"'{spec['name']}' does not take extra terms")
            groups = _clean_extra_groups(body.extra_groups)
            if groups:
                params["extra_groups"] = groups
            else:
                params.pop("extra_groups", None)
        enabled = row["enabled"] if row else None
        if body.enabled is not None:
            enabled = bool(body.enabled)
        db.save_hard_filter(key, builtin=True, name="",
                            enabled=None if enabled is None else bool(enabled),
                            params_json=json.dumps(params))
    else:                                          # custom filter
        if row["builtin"]:
            raise HTTPException(404, f"no filter '{key}'")
        params = _row_params(row)
        name = row["name"]
        if body.name is not None:
            new_name = _clean_name(body.name)
            if new_name != name:
                _ensure_unique_name(new_name, exclude_key=key)
                db.rename_filter_reasons(
                    matching.CUSTOM_REASON_PREFIX + name,
                    matching.CUSTOM_REASON_PREFIX + new_name)
                name = new_name
        if body.description is not None:
            desc = _clean_description(body.description)
            if desc:
                params["description"] = desc
            else:
                params.pop("description", None)
        if body.category is not None:
            cat = _clean_category(body.category)
            if cat:
                params["category"] = cat
            else:
                params.pop("category", None)
        if body.field is not None:
            if body.field not in _ALLOWED_FIELDS:
                raise HTTPException(400, f"field must be one of: {', '.join(sorted(_ALLOWED_FIELDS))}")
            params["field"] = body.field
        if body.match is not None:
            if body.match not in _ALLOWED_MATCH:
                raise HTTPException(400, "match must be 'word' or 'contains'")
            params["match"] = body.match
        if body.terms is not None:
            terms = _clean_terms(body.terms)
            if not terms:
                raise HTTPException(400, "Add at least one term to match")
            params["terms"] = terms
        enabled = True if row["enabled"] is None else bool(row["enabled"])
        if body.enabled is not None:
            enabled = bool(body.enabled)
        db.save_hard_filter(key, builtin=False, name=name, enabled=enabled,
                            params_json=json.dumps(params))
    _filters_changed()
    return {"ok": True}


@app.delete("/api/filters/{key}")
def api_delete_filter(key: str) -> dict:
    db.init_db()
    if matching.builtin_spec(key) is not None:
        raise HTTPException(400, "Built-in filters can be disabled, not deleted")
    row = db.get_hard_filter(key)
    if row is None or row["builtin"]:
        raise HTTPException(404, f"no custom filter '{key}'")
    db.delete_hard_filter(key)
    _filters_changed()
    return {"ok": True}


# --- Run history ----------------------------------------------------------------
@app.get("/api/runs")
def api_runs(limit: int = 60) -> dict:
    db.init_db()
    runs = []
    for r in db.get_runs(limit=min(max(int(limit), 1), 200)):
        try:
            sm = json.loads(r["summary"])
        except ValueError:
            continue
        runs.append({"at": r["at"], "summary": sm})
    return {"runs": runs, "count": db.count_runs()}


@app.get("/api/profile")
def api_get_profile() -> dict:
    p = db.get_profile()
    cv = (p["cv_text"] if p else "") or ""
    return {
        "preferences": (p["preferences_note"] if p else "") or "",
        "cv_preview": cv[:6000],
        "cv_chars": len(cv),
        "llm_enabled": matching.llm_available(),
        "model": config.ANTHROPIC_MODEL,
        "extract_model": config.EXTRACT_MODEL,
        "shortlist_min": config.SHORTLIST_MIN_SCORE,
        "budget_per_run": config.MAX_RUN_LLM_USD,
        "spend_total": float(db.meta_get("llm_spend_usd_total", "0") or 0),
        "spend_last_run": float(db.meta_get("llm_spend_usd_last_run", "0") or 0),
    }


class PrefsBody(BaseModel):
    preferences: str


@app.post("/api/profile")
def api_save_profile(body: PrefsBody) -> dict:
    p = db.get_profile()
    cv = (p["cv_text"] if p else "") or ""
    db.save_profile(cv, body.preferences)
    try:
        config.PREFERENCES_PATH.write_text(body.preferences, encoding="utf-8")
    except OSError:
        pass
    return {"ok": True}


def _do_run(only: Optional[list[str]] = None) -> None:
    try:
        matching.reset_llm_state()
        summary = pipeline.run_tiered(detect_first=True, verbose=False, only=only)
        with _run_lock:
            _run_state["summary"] = summary.as_dict()
            _run_state["error"] = None
    except Exception as e:  # noqa: BLE001
        with _run_lock:
            _run_state["error"] = f"{type(e).__name__}: {e}"
    finally:
        with _run_lock:
            _run_state["running"] = False
            _run_state["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunBody(BaseModel):
    # Optional selection: run the pipeline for these companies only.
    companies: list[str] = []


@app.post("/api/run")
def api_run(body: Optional[RunBody] = None) -> dict:
    only = [c for c in (body.companies if body else []) if c and c.strip()] or None
    if only:
        # Validate BEFORE claiming the run slot — a typo must return a clear
        # 400, not a background run that dies immediately.
        try:
            chosen = pipeline.resolve_company_selection(only)
        except pipeline.UnknownCompanyError as e:
            raise HTTPException(400, str(e))
        only = [c["name"] for c in chosen]
    with _run_lock:
        if _run_state["running"]:
            return {"ok": False, "running": True}
        _run_state.update(running=True, started_at=datetime.now(timezone.utc)
                          .isoformat(timespec="seconds"), finished_at=None,
                          summary=None, error=None)
    # Reset progress BEFORE the thread starts so the first poll can never see
    # the previous run's final ("done") snapshot.
    pipeline.PROGRESS.reset()
    pipeline.PROGRESS.phase("starting", "Starting run…"
                            + (f" ({len(only)} selected companies)" if only else ""))
    threading.Thread(target=_do_run, args=(only,), daemon=True).start()
    return {"ok": True, "running": True, "companies": only or []}


def _do_linkcheck() -> None:
    from . import linkcheck
    try:
        out = linkcheck.recheck_recommendations(force=True,
                                                progress=pipeline.PROGRESS)
        summary = {"links_checked": out.checked, "links_dead": out.dead,
                   "links_unknown": out.unknown,
                   "recs_removed": out.recs_removed,
                   "notes": (out.notes or ["every checked link is reachable"])
                   + out.removed[:12]}
        with _run_lock:
            _run_state["summary"] = summary
            _run_state["error"] = None
    except Exception as e:  # noqa: BLE001
        with _run_lock:
            _run_state["error"] = f"{type(e).__name__}: {e}"
    finally:
        pipeline.PROGRESS.phase("done", "Link check complete")
        with _run_lock:
            _run_state["running"] = False
            _run_state["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")


@app.post("/api/linkcheck")
def api_linkcheck() -> dict:
    """Verify the links behind the board's New/Marked recommendations, in the
    background (shares the run slot + progress bar with pipeline runs)."""
    db.init_db()
    with _run_lock:
        if _run_state["running"]:
            return {"ok": False, "running": True}
        _run_state.update(running=True, started_at=datetime.now(timezone.utc)
                          .isoformat(timespec="seconds"), finished_at=None,
                          summary=None, error=None)
    pipeline.PROGRESS.reset()
    pipeline.PROGRESS.phase("linkcheck", "Checking recommendation links…")
    threading.Thread(target=_do_linkcheck, daemon=True).start()
    return {"ok": True, "running": True}


@app.get("/api/run/status")
def api_run_status() -> dict:
    with _run_lock:
        out = dict(_run_state)
    # Live phase/step detail maintained by the pipeline thread — this is what
    # drives the dashboard's progress bar while a run is active.
    out["progress"] = pipeline.PROGRESS.snapshot()
    return out


# --- Static frontend ---------------------------------------------------------
@app.get("/")
def index() -> FileResponse:
    # no-cache: the dashboard is one self-contained file that changes with the
    # code — a browser must always revalidate instead of serving a stale copy.
    return FileResponse(STATIC_DIR / "index.html",
                        headers={"Cache-Control": "no-cache, must-revalidate"})


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _existing_instance(url: str) -> bool:
    """True when a JobScout dashboard already answers on this address."""
    import urllib.request
    try:
        with urllib.request.urlopen(f"{url}/api/ping", timeout=1.5) as r:
            return b"jobscout" in r.read(200).lower()
    except Exception:  # noqa: BLE001 — connection refused = port is free
        return False


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="jobscout.web", description="JobScout dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--idle-minutes", type=float, default=None,
                        help="auto-exit after this many minutes without an "
                             "open dashboard tab (default: "
                             "JOBSCOUT_WEB_IDLE_MINUTES or 10; 0 = never)")
    args = parser.parse_args(argv)

    url = f"http://{args.host}:{args.port}"
    # A second start must not crash with "address already in use": if the
    # port is held by a live JobScout, just open a tab on it and leave.
    if _existing_instance(url):
        print(f"JobScout dashboard already running → {url} (reusing it)")
        if not args.no_browser:
            webbrowser.open(url)
        return

    db.init_db()
    print(f"JobScout dashboard → {url}")
    if not matching.llm_available():
        print("  (No Anthropic key found — using the heuristic scorer. Add key.txt "
              "or set ANTHROPIC_API_KEY for Claude Sonnet scoring.)")
    idle_min = WEB_IDLE_MINUTES if args.idle_minutes is None else args.idle_minutes
    if idle_min > 0:
        print(f"  auto-exits after {idle_min:g} min without an open tab "
              "(--idle-minutes 0 to disable)")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    import uvicorn
    global _server
    _server = uvicorn.Server(uvicorn.Config(app, host=args.host,
                                            port=args.port,
                                            log_level="warning"))
    global _last_activity
    _last_activity = time.time()
    if idle_min > 0:
        threading.Thread(target=_idle_watchdog, args=(idle_min * 60.0,),
                         daemon=True).start()
    _server.run()


if __name__ == "__main__":
    main()

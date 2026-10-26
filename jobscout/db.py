"""SQLite storage + dedupe layer.

Four records run the whole app, straight from the blueprint:

    Profile        — your CV and preferences
    Company        — a target employer (name, ats_type, ats_token, locations)
    Posting        — a job pulled from a feed (job_id, title, posted_date, ...)
    Recommendation — a scored posting + your decision (fit_score, status, notes)

Because Posting.job_id is a *stable* identifier from the source feed, your
decisions on each Recommendation persist correctly every time the pipeline runs.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

from . import config


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


SCHEMA = """
CREATE TABLE IF NOT EXISTS profile (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    cv_text         TEXT NOT NULL DEFAULT '',
    preferences_note TEXT NOT NULL DEFAULT '',
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS company (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL UNIQUE,
    category      TEXT,
    locations     TEXT,            -- preferred locations from your list
    interview_pct REAL,            -- your own est. screen probability
    ats_type      TEXT,            -- greenhouse | lever | personio | unknown | none
    ats_token     TEXT,            -- the company's identifier on that ATS
    detect_status TEXT DEFAULT 'pending',   -- pending | detected | unsupported
    detect_note   TEXT,            -- why: 'curated' / 'probed' / the unsupported platform
    last_checked  TEXT
);

CREATE TABLE IF NOT EXISTS posting (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id      INTEGER NOT NULL REFERENCES company(id),
    job_id          TEXT NOT NULL,    -- stable id from the source feed
    title           TEXT NOT NULL,
    location        TEXT,
    posted_date     TEXT,             -- YYYY-MM-DD (authoritative, from the feed)
    date_confidence TEXT DEFAULT 'high',  -- 'high' | 'low' (approx/unknown, e.g. SuccessFactors)
    url             TEXT,
    description     TEXT,
    detail_ref      TEXT,             -- opaque ref for a follow-up detail fetch (Workday)
    first_seen      TEXT NOT NULL,
    last_seen       TEXT NOT NULL,
    is_open         INTEGER NOT NULL DEFAULT 1,
    UNIQUE (company_id, job_id)
);

CREATE TABLE IF NOT EXISTS recommendation (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    posting_id      INTEGER NOT NULL UNIQUE REFERENCES posting(id),
    prefilter_score REAL,
    fit_score       REAL,
    rationale       TEXT,
    watch_outs      TEXT,
    llm_used        INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'New',
    notes           TEXT NOT NULL DEFAULT '',
    first_seen      TEXT NOT NULL,
    scored_at       TEXT
);

-- Small key/value store: cumulative LLM spend, last-run summary, etc.
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- Tier-3 extraction cache: skip the LLM when a careers page hasn't changed.
CREATE TABLE IF NOT EXISTS extraction_cache (
    url          TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    jobs_json    TEXT NOT NULL,
    extracted_at TEXT NOT NULL
);

-- Hard-filter configuration, managed from the dashboard. Built-in rules live
-- in matching.BUILTIN_HARD_FILTERS; a row here overrides one (enabled/params,
-- keyed by the registry key). Rows with builtin=0 are user-created keyword
-- filters. No row = the built-in default applies.
CREATE TABLE IF NOT EXISTS hard_filter (
    key        TEXT PRIMARY KEY,
    builtin    INTEGER NOT NULL DEFAULT 0,
    name       TEXT NOT NULL DEFAULT '',   -- display name (custom filters)
    enabled    INTEGER,                    -- NULL = built-in default
    params     TEXT NOT NULL DEFAULT '{}', -- JSON: terms/field/match or max_days/…
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Per-run statistics, one row per finished pipeline run (Run History view).
CREATE TABLE IF NOT EXISTS run_history (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    at      TEXT NOT NULL,
    summary TEXT NOT NULL                  -- RunSummary.as_dict() as JSON
);

CREATE INDEX IF NOT EXISTS idx_posting_company ON posting(company_id);
CREATE INDEX IF NOT EXISTS idx_posting_open ON posting(is_open);
CREATE INDEX IF NOT EXISTS idx_rec_status ON recommendation(status);
"""


@contextmanager
def connect(db_path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    # Resolve JOBSCOUT_DB at call time so a test (or the caller) can point the DB
    # somewhere else even after config was first imported.
    path = Path(db_path or os.environ.get("JOBSCOUT_DB") or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    # timeout: the background pipeline thread may be writing while the UI reads;
    # wait rather than raising 'database is locked'. WAL allows concurrent reads.
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.OperationalError:
        pass
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Optional[Path] = None) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the first schema version (idempotent)."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(posting)")}
    if "detail_ref" not in cols:
        conn.execute("ALTER TABLE posting ADD COLUMN detail_ref TEXT")
    if "date_confidence" not in cols:
        conn.execute("ALTER TABLE posting ADD COLUMN date_confidence TEXT DEFAULT 'high'")
    if "filtered_reason" not in cols:
        # Why a posting was excluded by the hard filters (NULL = eligible).
        conn.execute("ALTER TABLE posting ADD COLUMN filtered_reason TEXT")
    if "link_checked_at" not in cols:
        # When the job page URL was last verified reachable (linkcheck.py);
        # NULL = never. Gates how often the dead-link prune re-checks a URL.
        conn.execute("ALTER TABLE posting ADD COLUMN link_checked_at TEXT")
    rcols = {r["name"] for r in conn.execute("PRAGMA table_info(recommendation)")}
    if "validated_at" not in rcols:
        # Final LLM validation gate: when this rec was last reviewed, and what
        # the reviewer said. NULL = not yet validated (or re-scored since).
        conn.execute("ALTER TABLE recommendation ADD COLUMN validated_at TEXT")
        conn.execute("ALTER TABLE recommendation ADD COLUMN validation_note TEXT")
    ccols = {r["name"] for r in conn.execute("PRAGMA table_info(company)")}
    if "detect_note" not in ccols:
        conn.execute("ALTER TABLE company ADD COLUMN detect_note TEXT")
    if "last_fetch_at" not in ccols:
        # Per-company fetch health, shown on the Coverage view.
        conn.execute("ALTER TABLE company ADD COLUMN last_fetch_at TEXT")
        conn.execute("ALTER TABLE company ADD COLUMN last_fetch_status TEXT")
        conn.execute("ALTER TABLE company ADD COLUMN last_fetch_jobs INTEGER")

    # Data sanitation (idempotent, runs at every init): a posted_date must be a
    # real calendar date and must not lie in the future. LLM extraction on the
    # browser tier occasionally returned application DEADLINES ("2027-01-25")
    # or day/month-swapped strings ("2026-13-07"); both would let a posting
    # dodge the age window forever. date(x) IS NULL catches the invalid ones.
    # The future check runs on LOCAL time with one day of slack: date('now')
    # alone is UTC, which stripped legitimate "posted today" dates in the
    # local-past-midnight window (and boards in timezones ahead of the user
    # can honestly be one day ahead).
    conn.execute(
        """UPDATE posting SET posted_date=NULL, date_confidence='low'
           WHERE posted_date IS NOT NULL
             AND (date(posted_date) IS NULL
                  OR posted_date > date('now','localtime','+1 day'))""")

    # Seed run_history with the last recorded run so the Run History view has
    # data right after the upgrade (earlier runs were never persisted per-run).
    if not conn.execute("SELECT 1 FROM run_history LIMIT 1").fetchone():
        last_at = conn.execute(
            "SELECT value FROM meta WHERE key='last_run_at'").fetchone()
        last_sm = conn.execute(
            "SELECT value FROM meta WHERE key='last_run_summary'").fetchone()
        if last_at and last_sm and last_at["value"] and last_sm["value"]:
            conn.execute("INSERT INTO run_history (at, summary) VALUES (?, ?)",
                         (last_at["value"], last_sm["value"]))

    # One-off (meta-keyed): re-queue the New recommendations that the pre-fix
    # "mgr" gate froze at exactly 40 — deleting the rec puts the posting back
    # into the scoring queue, where the corrected gate judges it afresh.
    if not conn.execute(
            "SELECT 1 FROM meta WHERE key='mig_2026_07_09_mgr_requeue'").fetchone():
        conn.execute(
            """DELETE FROM recommendation WHERE status='New' AND llm_used=1
               AND fit_score BETWEEN 39.5 AND 40.5
               AND (watch_outs LIKE '%Manager title%'
                    OR rationale LIKE '%Manager title%')""")
        conn.execute("INSERT INTO meta (key, value) "
                     "VALUES ('mig_2026_07_09_mgr_requeue','done')")


# --- Profile -----------------------------------------------------------------
def save_profile(cv_text: str, preferences_note: str, db_path=None) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """INSERT INTO profile (id, cv_text, preferences_note, updated_at)
               VALUES (1, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   cv_text=excluded.cv_text,
                   preferences_note=excluded.preferences_note,
                   updated_at=excluded.updated_at""",
            (cv_text, preferences_note, _now()),
        )


def get_profile(db_path=None) -> Optional[sqlite3.Row]:
    with connect(db_path) as conn:
        return conn.execute("SELECT * FROM profile WHERE id = 1").fetchone()


# --- Company -----------------------------------------------------------------
def upsert_company(name, category=None, locations=None, interview_pct=None, db_path=None) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO company (name, category, locations, interview_pct)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(name) DO UPDATE SET
                   category=excluded.category,
                   locations=excluded.locations,
                   interview_pct=excluded.interview_pct
               RETURNING id""",
            (name, category, locations, interview_pct),
        )
        return cur.fetchone()[0]


def set_company_ats(company_id, ats_type, ats_token, detect_status, detect_note=None,
                    db_path=None) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """UPDATE company SET ats_type=?, ats_token=?, detect_status=?,
                   detect_note=?, last_checked=? WHERE id=?""",
            (ats_type, ats_token, detect_status, detect_note, _now(), company_id),
        )


def get_companies(only_detected=False, db_path=None) -> list[sqlite3.Row]:
    q = "SELECT * FROM company"
    if only_detected:
        q += " WHERE detect_status='detected' AND ats_type IS NOT NULL"
    q += " ORDER BY name"
    with connect(db_path) as conn:
        return conn.execute(q).fetchall()


def set_fetch_health(company_id, status: str, jobs: int, db_path=None) -> None:
    """Record the outcome of the latest fetch ('ok' | an error string)."""
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE company SET last_fetch_at=?, last_fetch_status=?, last_fetch_jobs=? WHERE id=?",
            (_now(), status[:300], jobs, company_id))


# --- Posting (dedupe lives here) --------------------------------------------
def upsert_posting(company_id, job_id, title, location, posted_date, url, description,
                   detail_ref=None, date_confidence="high", db_path=None) -> tuple[int, bool]:
    """Insert or refresh a posting. Returns (posting_id, is_new).

    Dedupe key is (company_id, job_id). On a repeat sighting we keep the
    original first_seen and posted_date, and just bump last_seen / is_open.
    """
    now = _now()
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT id FROM posting WHERE company_id=? AND job_id=?",
            (company_id, str(job_id)),
        ).fetchone()
        if row:
            # Refresh mutable fields, but never let a WORSE feed value clobber
            # data we already have:
            #   * location/url: empty list values (Workday, SuccessFactors send
            #     none) must not blank what detail-enrichment stored;
            #   * description: LONGER wins — Oracle's list endpoint sends a
            #     short summary while enrichment stores the full ad, so a plain
            #     overwrite would shrink enriched postings on every re-sighting
            #     (while genuinely updated feed descriptions still refresh).
            conn.execute(
                """UPDATE posting SET last_seen=?, is_open=1, title=?,
                       location=COALESCE(NULLIF(?,''), location),
                       url=COALESCE(NULLIF(?,''), url),
                       description=CASE
                           WHEN length(?) > length(COALESCE(description,''))
                           THEN ? ELSE description END
                   WHERE id=?""",
                (now, title, location or "", url or "",
                 description or "", description or "", row["id"]),
            )
            return row["id"], False
        cur = conn.execute(
            """INSERT INTO posting
               (company_id, job_id, title, location, posted_date, date_confidence,
                url, description, detail_ref, first_seen, last_seen, is_open)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,1) RETURNING id""",
            (company_id, str(job_id), title, location, posted_date, date_confidence,
             url, description, detail_ref, now, now),
        )
        return cur.fetchone()[0], True


def update_posting_enrichment(posting_id, description=None, posted_date=None,
                              url=None, location=None, db_path=None) -> None:
    """Apply detail-fetch results to a posting (only the fields provided)."""
    sets, params = [], []
    for col, val in (("description", description), ("posted_date", posted_date),
                     ("url", url), ("location", location)):
        if val:
            sets.append(f"{col}=?")
            params.append(val)
    if posted_date:
        sets.append("date_confidence='high'")
    if not sets:
        return
    params.append(posting_id)
    with connect(db_path) as conn:
        conn.execute(f"UPDATE posting SET {', '.join(sets)} WHERE id=?", params)


def close_missing_postings(company_id, seen_job_ids: Iterable[str], db_path=None) -> int:
    """Mark postings that vanished from a company's feed as closed."""
    seen = {str(j) for j in seen_job_ids}
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, job_id FROM posting WHERE company_id=? AND is_open=1",
            (company_id,),
        ).fetchall()
        closed = 0
        for r in rows:
            if r["job_id"] not in seen:
                conn.execute("UPDATE posting SET is_open=0 WHERE id=?", (r["id"],))
                closed += 1
        return closed


def get_postings_to_score(db_path=None,
                          company_ids: Optional[Iterable[int]] = None) -> list[sqlite3.Row]:
    """Open, unfiltered postings that have no recommendation yet.

    `company_ids` scopes the queue to selected companies (partial pipeline
    runs); None = every company."""
    q = """SELECT p.*, c.name AS company_name, c.locations AS company_locations,
                  c.ats_type AS ats_type, c.ats_token AS ats_token
           FROM posting p
           JOIN company c ON c.id = p.company_id
           LEFT JOIN recommendation r ON r.posting_id = p.id
           WHERE r.id IS NULL AND p.is_open = 1 AND p.filtered_reason IS NULL"""
    params: tuple = ()
    if company_ids is not None:
        ids = [int(i) for i in company_ids]
        if not ids:
            return []
        q += " AND p.company_id IN (%s)" % ",".join("?" * len(ids))
        params = tuple(ids)
    q += " ORDER BY p.posted_date DESC"
    with connect(db_path) as conn:
        return conn.execute(q, params).fetchall()


def mark_posting_filtered(posting_id, reason: Optional[str], db_path=None) -> None:
    with connect(db_path) as conn:
        conn.execute("UPDATE posting SET filtered_reason=? WHERE id=?",
                     (reason, posting_id))


def mark_postings_filtered(pairs: list[tuple[int, str]], db_path=None) -> None:
    """Bulk version of mark_posting_filtered — one transaction, not thousands."""
    if not pairs:
        return
    with connect(db_path) as conn:
        conn.executemany("UPDATE posting SET filtered_reason=? WHERE id=?",
                         [(reason, pid) for pid, reason in pairs])


def postings_for_linkcheck(db_path=None) -> list[sqlite3.Row]:
    """Every undecided/bookmarked recommendation whose posting is still open —
    the links the board is actually showing, for the dead-link prune.
    Applied/Skipped/Rejected are history: their link no longer matters."""
    with connect(db_path) as conn:
        return conn.execute(
            """SELECT p.id, p.url, p.job_id, p.detail_ref, p.link_checked_at,
                      p.title, r.id AS rec_id, r.status,
                      c.name AS company_name, c.ats_type, c.ats_token
               FROM recommendation r
               JOIN posting p ON p.id = r.posting_id
               JOIN company c ON c.id = p.company_id
               WHERE r.status IN ('New', 'Marked') AND p.is_open = 1"""
        ).fetchall()


def archive_dead_link(posting_id, reason: str, db_path=None) -> bool:
    """A verified-dead job page: close the posting and stamp the reason so a
    stale feed re-sighting can never put it back in the queue. Deletes the
    posting's New recommendation (returns True if one existed) — user-decided
    recs (Marked/Applied/…) are kept; the posting just closes underneath them."""
    now = _now()
    with connect(db_path) as conn:
        conn.execute(
            """UPDATE posting SET is_open=0, filtered_reason=?, link_checked_at=?
               WHERE id=?""", (reason, now, posting_id))
        cur = conn.execute(
            "DELETE FROM recommendation WHERE posting_id=? AND status='New'",
            (posting_id,))
        return cur.rowcount > 0


def touch_link_checked(posting_ids: Iterable[int], db_path=None) -> None:
    """Stamp postings whose link was just verified ALIVE (one transaction)."""
    ids = [int(i) for i in posting_ids]
    if not ids:
        return
    now = _now()
    with connect(db_path) as conn:
        conn.executemany("UPDATE posting SET link_checked_at=? WHERE id=?",
                         [(now, i) for i in ids])


def postings_filtered_as(reason: str, db_path=None) -> list[sqlite3.Row]:
    """Open postings dropped with exactly this filter reason (for re-checks
    when the rules RELAX — e.g. a location list bug gets fixed)."""
    with connect(db_path) as conn:
        return conn.execute(
            """SELECT id, title, location, description FROM posting
               WHERE is_open=1 AND filtered_reason=?""", (reason,)).fetchall()


def filtered_stats(db_path=None) -> dict:
    """Counts of open postings dropped by each hard-filter reason."""
    with connect(db_path) as conn:
        rows = conn.execute(
            """SELECT filtered_reason, COUNT(*) AS n FROM posting
               WHERE is_open=1 AND filtered_reason IS NOT NULL
               GROUP BY filtered_reason ORDER BY n DESC""").fetchall()
        return {r["filtered_reason"]: r["n"] for r in rows}


def open_filtered_postings(db_path=None) -> list[sqlite3.Row]:
    """Every open posting currently excluded — with everything needed to
    re-judge it when the hard-filter configuration changes."""
    with connect(db_path) as conn:
        return conn.execute(
            """SELECT id, title, location, description, posted_date, first_seen,
                      filtered_reason
               FROM posting WHERE is_open=1 AND filtered_reason IS NOT NULL"""
        ).fetchall()


# --- Hard-filter configuration --------------------------------------------------
def get_hard_filters(db_path=None) -> list[sqlite3.Row]:
    with connect(db_path) as conn:
        return conn.execute(
            "SELECT * FROM hard_filter ORDER BY builtin DESC, created_at, key"
        ).fetchall()


def get_hard_filter(key: str, db_path=None) -> Optional[sqlite3.Row]:
    with connect(db_path) as conn:
        return conn.execute(
            "SELECT * FROM hard_filter WHERE key=?", (key,)).fetchone()


def save_hard_filter(key: str, builtin: bool, name: str,
                     enabled: Optional[bool], params_json: str,
                     db_path=None) -> None:
    """Insert or update one filter row. `enabled=None` means "the built-in
    default decides" (only meaningful for builtin rows)."""
    now = _now()
    with connect(db_path) as conn:
        conn.execute(
            """INSERT INTO hard_filter (key, builtin, name, enabled, params,
                                        created_at, updated_at)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(key) DO UPDATE SET
                 name=excluded.name, enabled=excluded.enabled,
                 params=excluded.params, updated_at=excluded.updated_at""",
            (key, int(builtin), name or "",
             None if enabled is None else int(bool(enabled)),
             params_json or "{}", now, now))


def delete_hard_filter(key: str, db_path=None) -> None:
    with connect(db_path) as conn:
        conn.execute("DELETE FROM hard_filter WHERE key=?", (key,))


def rename_filter_reasons(old_reason: str, new_reason: str, db_path=None) -> int:
    """Keep stored exclusion reasons in step when a custom filter is renamed."""
    with connect(db_path) as conn:
        cur = conn.execute(
            "UPDATE posting SET filtered_reason=? WHERE filtered_reason=?",
            (new_reason, old_reason))
        return cur.rowcount


def bump_filters_rev(db_path=None) -> int:
    """Mark the hard-filter configuration as changed. The pipeline compares
    this against 'hard_filters_rev_applied' at run start and re-judges every
    excluded open posting once when they differ."""
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT value FROM meta WHERE key='hard_filters_rev'").fetchone()
        rev = (int(row["value"]) if row and str(row["value"]).isdigit() else 0) + 1
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('hard_filters_rev', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(rev),))
        return rev


# --- Run history -----------------------------------------------------------------
def record_run(at: str, summary_json: str, db_path=None) -> None:
    """Append one finished run; the table stays bounded (400 ≈ a year of use)."""
    with connect(db_path) as conn:
        conn.execute("INSERT INTO run_history (at, summary) VALUES (?, ?)",
                     (at, summary_json))
        conn.execute("""DELETE FROM run_history WHERE id NOT IN
                        (SELECT id FROM run_history ORDER BY id DESC LIMIT 400)""")


def get_runs(limit: int = 60, db_path=None) -> list[sqlite3.Row]:
    with connect(db_path) as conn:
        return conn.execute(
            "SELECT id, at, summary FROM run_history ORDER BY id DESC LIMIT ?",
            (int(limit),)).fetchall()


def count_runs(db_path=None) -> int:
    with connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM run_history").fetchone()[0]


def open_recommendations(db_path=None) -> list[sqlite3.Row]:
    """Status-New recommendations joined with their posting, for re-filtering."""
    with connect(db_path) as conn:
        return conn.execute(
            """SELECT r.id AS rec_id, r.status, r.fit_score, r.llm_used,
                      p.id AS posting_id, p.title,
                      p.location, p.description, p.is_open,
                      p.posted_date, p.first_seen
               FROM recommendation r JOIN posting p ON p.id = r.posting_id
               WHERE r.status = 'New'""").fetchall()


def recs_to_validate(revalidate_all: bool = False, db_path=None,
                     company_ids: Optional[Iterable[int]] = None) -> list[sqlite3.Row]:
    """New recommendations awaiting the final LLM validation pass.

    A rec needs (re-)validation when it has never been validated or has been
    re-scored since its last validation. Only undecided rows on open postings
    at/above VALIDATION_MIN_SCORE qualify — the board's deliberate low-score
    long tail is exempt (already labelled weak by its own score), and user
    decisions are never second-guessed. `company_ids` scopes the review to
    selected companies (partial pipeline runs)."""
    q = """SELECT r.id AS rec_id, r.fit_score, r.rationale, r.watch_outs,
                  r.llm_used, r.validated_at, r.scored_at,
                  p.id AS posting_id, p.title, p.location, p.description,
                  p.posted_date, c.name AS company
           FROM recommendation r
           JOIN posting p ON p.id = r.posting_id
           JOIN company c ON c.id = p.company_id
           WHERE r.status = 'New' AND p.is_open = 1
             AND COALESCE(r.fit_score, 0) >= ?"""
    params: tuple = (config.VALIDATION_MIN_SCORE,)
    if not revalidate_all:
        q += """ AND (r.validated_at IS NULL
                      OR (r.scored_at IS NOT NULL AND r.scored_at > r.validated_at))"""
    if company_ids is not None:
        ids = [int(i) for i in company_ids]
        if not ids:
            return []
        q += " AND p.company_id IN (%s)" % ",".join("?" * len(ids))
        params += tuple(ids)
    q += " ORDER BY r.id"
    with connect(db_path) as conn:
        return conn.execute(q, params).fetchall()


def set_validation(rec_id, note: Optional[str], db_path=None) -> None:
    """Record that the final validation gate reviewed (and kept) this rec."""
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE recommendation SET validated_at=?, validation_note=? WHERE id=?",
            (_now(), note, rec_id))


def recommended_open_postings(db_path=None) -> list[sqlite3.Row]:
    """Every recommendation (any status) whose posting is still open — the
    reference set for duplicate suppression."""
    with connect(db_path) as conn:
        return conn.execute(
            """SELECT r.id AS rec_id, r.status, r.fit_score,
                      p.id AS posting_id, p.company_id, p.title, p.location,
                      p.first_seen
               FROM recommendation r JOIN posting p ON p.id = r.posting_id
               WHERE p.is_open = 1""").fetchall()


def reset_except(keep_statuses: Iterable[str], drop_rec_ids: Iterable[int] = (),
                 db_path=None) -> dict:
    """Wipe the board for a from-scratch rebuild, keeping only decisions.

    Deletes every recommendation outside `keep_statuses` (plus any explicitly
    listed rec ids, e.g. duplicates among the kept ones), then every posting
    left without a recommendation. Companies, profile, meta (spend history)
    and the extraction cache survive. VACUUM reclaims the space.
    """
    keep = tuple(keep_statuses)
    drop = tuple(int(i) for i in drop_rec_ids)
    with connect(db_path) as conn:
        stats = {}
        if drop:
            conn.execute(
                "DELETE FROM recommendation WHERE id IN (%s)"
                % ",".join("?" * len(drop)), drop)
            stats["dropped_kept_duplicates"] = len(drop)
        cur = conn.execute(
            "DELETE FROM recommendation WHERE status NOT IN (%s)"
            % ",".join("?" * len(keep)), keep)
        stats["recommendations_deleted"] = cur.rowcount
        cur = conn.execute(
            """DELETE FROM posting WHERE id NOT IN
               (SELECT posting_id FROM recommendation)""")
        stats["postings_deleted"] = cur.rowcount
        stats["recommendations_kept"] = conn.execute(
            "SELECT COUNT(*) FROM recommendation").fetchone()[0]
        stats["postings_kept"] = conn.execute(
            "SELECT COUNT(*) FROM posting").fetchone()[0]
        conn.execute("UPDATE company SET last_fetch_at=NULL, "
                     "last_fetch_status=NULL, last_fetch_jobs=NULL")
        conn.commit()
        conn.execute("VACUUM")
        return stats


def delete_recommendation(rec_id, db_path=None) -> None:
    with connect(db_path) as conn:
        conn.execute("DELETE FROM recommendation WHERE id=?", (rec_id,))


# --- Meta (key/value) ----------------------------------------------------------
def meta_get(key: str, default: Optional[str] = None, db_path=None) -> Optional[str]:
    with connect(db_path) as conn:
        row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default


def meta_set(key: str, value: str, db_path=None) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))


def meta_add_float(key: str, delta: float, db_path=None) -> float:
    """Atomically add to a float meta value; returns the new total."""
    with connect(db_path) as conn:
        row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        total = (float(row["value"]) if row else 0.0) + delta
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, f"{total:.6f}"))
        return total


# --- Tier-3 extraction cache ----------------------------------------------------
def extraction_cache_get(url: str, db_path=None) -> Optional[sqlite3.Row]:
    with connect(db_path) as conn:
        return conn.execute(
            "SELECT * FROM extraction_cache WHERE url=?", (url,)).fetchone()


def extraction_cache_put(url: str, content_hash: str, jobs_json: str, db_path=None) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """INSERT INTO extraction_cache (url, content_hash, jobs_json, extracted_at)
               VALUES (?,?,?,?)
               ON CONFLICT(url) DO UPDATE SET
                 content_hash=excluded.content_hash,
                 jobs_json=excluded.jobs_json,
                 extracted_at=excluded.extracted_at""",
            (url, content_hash, jobs_json, _now()))


# --- Recommendation ----------------------------------------------------------
def save_recommendation(posting_id, prefilter_score, fit_score, rationale, watch_outs,
                        llm_used, db_path=None) -> None:
    """Create a recommendation. Status/notes are user-owned, so if one already
    exists we never clobber the decision — we only refresh the score fields."""
    now = _now()
    with connect(db_path) as conn:
        conn.execute(
            """INSERT INTO recommendation
                 (posting_id, prefilter_score, fit_score, rationale, watch_outs,
                  llm_used, status, notes, first_seen, scored_at)
               VALUES (?,?,?,?,?,?, 'New', '', ?, ?)
               ON CONFLICT(posting_id) DO UPDATE SET
                  prefilter_score=excluded.prefilter_score,
                  fit_score=excluded.fit_score,
                  rationale=excluded.rationale,
                  watch_outs=excluded.watch_outs,
                  llm_used=excluded.llm_used,
                  scored_at=excluded.scored_at""",
            (posting_id, prefilter_score, fit_score, rationale, watch_outs,
             int(llm_used), now, now),
        )


def set_status(rec_id, status, db_path=None) -> None:
    with connect(db_path) as conn:
        conn.execute("UPDATE recommendation SET status=? WHERE id=?", (status, rec_id))


def set_notes(rec_id, notes, db_path=None) -> None:
    with connect(db_path) as conn:
        conn.execute("UPDATE recommendation SET notes=? WHERE id=?", (notes, rec_id))


def board_rows(statuses: Optional[Iterable[str]] = None, db_path=None) -> list[sqlite3.Row]:
    """Full denormalized rows for the dashboard.

    Closed postings (vanished from their feed) stay visible only where the user
    already acted (Applied/Skipped/Rejected history); undecided 'New' rows for
    closed postings are hidden — recommending an unavailable job helps nobody.
    """
    q = """SELECT r.id AS rec_id, r.fit_score, r.prefilter_score, r.rationale,
                  r.watch_outs, r.status, r.notes, r.llm_used,
                  r.validated_at, r.validation_note, r.first_seen AS rec_first_seen,
                  p.id AS posting_id, p.title, p.location, p.posted_date, p.date_confidence,
                  p.url, p.description, p.is_open,
                  c.name AS company, c.category, c.interview_pct
           FROM recommendation r
           JOIN posting p ON p.id = r.posting_id
           JOIN company c ON c.id = p.company_id
           WHERE (p.is_open = 1 OR r.status != 'New')"""
    params: tuple = ()
    if statuses:
        statuses = list(statuses)
        q += " AND r.status IN (%s)" % ",".join("?" * len(statuses))
        params = tuple(statuses)
    q += " ORDER BY r.fit_score DESC, p.posted_date DESC"
    with connect(db_path) as conn:
        return conn.execute(q, params).fetchall()


def stats(db_path=None) -> dict:
    with connect(db_path) as conn:
        def one(sql, *a):
            return conn.execute(sql, a).fetchone()[0]
        return {
            "companies": one("SELECT COUNT(*) FROM company"),
            "companies_detected": one("SELECT COUNT(*) FROM company WHERE detect_status='detected'"),
            "postings_open": one("SELECT COUNT(*) FROM posting WHERE is_open=1"),
            "postings_total": one("SELECT COUNT(*) FROM posting"),
            "postings_filtered": one(
                "SELECT COUNT(*) FROM posting WHERE is_open=1 AND filtered_reason IS NOT NULL"),
            "recommendations": one("SELECT COUNT(*) FROM recommendation"),
            "applied": one("SELECT COUNT(*) FROM recommendation WHERE status='Applied'"),
            "marked": one("SELECT COUNT(*) FROM recommendation WHERE status='Marked'"),
            "skipped": one("SELECT COUNT(*) FROM recommendation WHERE status='Skipped'"),
            "rejected": one("SELECT COUNT(*) FROM recommendation WHERE status='Rejected'"),
            "new": one("SELECT COUNT(*) FROM recommendation "
                       "WHERE status='New' AND posting_id IN "
                       "(SELECT id FROM posting WHERE is_open=1)"),
        }

"""Central configuration and paths for JobScout.

Nothing here is prescriptive — tune freely. Paths are resolved relative to the
project root so the tool runs from anywhere.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROFILE_DIR = DATA_DIR / "profile"
DB_PATH = Path(os.environ.get("JOBSCOUT_DB", DATA_DIR / "jobscout.db"))
CV_TXT_PATH = PROFILE_DIR / "cv.txt"
PREFERENCES_PATH = PROFILE_DIR / "preferences.txt"
COMPANIES_SEED_CSV = DATA_DIR / "companies_seed.csv"
CAREERS_SEED_CSV = DATA_DIR / "careers_seed.csv"

for _d in (DATA_DIR, PROFILE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- HTTP --------------------------------------------------------------------
USER_AGENT = "JobScout/1.0 (+personal job search; respectful, low-volume)"
BROWSER_USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")
HTTP_TIMEOUT = 25  # seconds
# Per-request politeness delay. Companies are fetched in parallel but each
# company's own requests stay sequential, so a small delay is enough.
REQUEST_DELAY_SECONDS = float(os.environ.get("JOBSCOUT_REQUEST_DELAY", "0.25"))
MAX_RETRIES = 3
# Companies fetched concurrently during a run (per-host requests stay serial).
FETCH_WORKERS = int(os.environ.get("JOBSCOUT_FETCH_WORKERS", "8"))

# --- Dead-link pruning (linkcheck.py) -----------------------------------------
# Every run verifies that the job pages behind the board's New/Marked
# recommendations still exist, and that scoring survivors are alive before
# any LLM spend. Only certain verdicts (API/HTTP 404, an unambiguous
# not-found page) remove anything; bot walls and network trouble fail open.
LINKCHECK_ENABLED = os.environ.get("JOBSCOUT_LINKCHECK", "1").lower() \
    not in ("0", "false", "no")
# Re-verify a URL at most this often (hours). Manual "Check links" runs and
# `cli linkcheck --force` ignore the TTL.
LINKCHECK_TTL_HOURS = float(os.environ.get("JOBSCOUT_LINKCHECK_TTL", "20"))
# Hosts checked in parallel; one host's requests stay serial + delayed.
LINKCHECK_WORKERS = int(os.environ.get("JOBSCOUT_LINKCHECK_WORKERS", "10"))
LINKCHECK_TIMEOUT = float(os.environ.get("JOBSCOUT_LINKCHECK_TIMEOUT", "12"))
LINKCHECK_DELAY = float(os.environ.get("JOBSCOUT_LINKCHECK_DELAY", "0.3"))

# Hard-filter ceiling on posting age: anything older never reaches the board,
# and existing New recommendations past the ceiling are swept on every run.
# Postings without a published date age from the day JobScout first saw them.
MAX_POSTING_AGE_DAYS = int(os.environ.get("JOBSCOUT_MAX_AGE_DAYS", "28"))

# --- Matching ----------------------------------------------------------------
# Anthropic key resolution order:
#   1. ANTHROPIC_API_KEY env var
#   2. a key.txt file (project root, its parent, or JOBSCOUT_KEY_FILE)
# If no key is found, JobScout falls back to a local heuristic scorer so the
# loop still runs end to end.
def _load_api_key() -> str:
    env = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if env:
        return env
    candidates = []
    if os.environ.get("JOBSCOUT_KEY_FILE"):
        candidates.append(Path(os.environ["JOBSCOUT_KEY_FILE"]))
    candidates += [
        PROJECT_ROOT / "key.txt",
        PROJECT_ROOT.parent / "key.txt",
        DATA_DIR / "key.txt",
    ]
    for p in candidates:
        try:
            if p.is_file():
                text = p.read_text(encoding="utf-8").strip()
                # tolerate "ANTHROPIC_API_KEY=sk-..." or a bare key
                if "=" in text and text.split("=", 1)[0].strip().upper().endswith("KEY"):
                    text = text.split("=", 1)[1].strip()
                if text.startswith("sk-"):
                    return text
        except OSError:
            continue
    return ""


ANTHROPIC_API_KEY = _load_api_key()

# Scoring model — the final filtering stage. Claude Sonnet judges each surviving
# posting against the CV + preferences. (Sonnet 5 intro pricing: $2/MTok in,
# $10/MTok out, cache reads $0.20/MTok — with batching + prompt caching a full
# initial build costs well under $1, and incremental daily runs are cents.)
ANTHROPIC_MODEL = os.environ.get("JOBSCOUT_MODEL", "claude-sonnet-5")
# Extraction model — Tier-3 browser extraction is mechanical (read a page, list
# the postings), so it stays on Haiku; quality-critical *judgment* uses Sonnet.
EXTRACT_MODEL = os.environ.get("JOBSCOUT_EXTRACT_MODEL", "claude-haiku-4-5-20251001")

# $/MTok (input, output, cache-read, cache-write-5m) per model family, used for
# live spend tracking. Matched by substring against the model id.
MODEL_PRICES_PER_MTOK: dict[str, tuple[float, float, float, float]] = {
    "sonnet": (2.0, 10.0, 0.20, 2.50),
    "haiku": (1.0, 5.0, 0.10, 1.25),
    "opus": (5.0, 25.0, 0.50, 6.25),
}
# Hard per-run budget for LLM scoring. When spend hits this cap, remaining
# postings stay QUEUED for the next run (never fake-scored). This is a
# runaway brake, not a target — a run only spends what its postings cost.
# Raised 10 -> 30 (user choice 2026-07-13) so the one-time ~4.2k-posting
# backlog (~$19-26) completes in a single dashboard-triggered run; daily
# incremental runs cost cents and never approach the cap.
MAX_RUN_LLM_USD = float(os.environ.get("JOBSCOUT_MAX_RUN_USD", "30.00"))
# Jobs per LLM request. Batching amortizes the cached CV/preferences prefix and
# the per-request overhead; 8 keeps outputs comfortably inside max_tokens.
LLM_BATCH_SIZE = int(os.environ.get("JOBSCOUT_LLM_BATCH", "8"))
# Description budget (characters) sent to the LLM per job. Descriptions above
# this are smart-trimmed (head + requirements section + key facts) rather than
# cut blindly — hard requirements usually live at the END of an ad.
LLM_DESC_CHARS = int(os.environ.get("JOBSCOUT_LLM_DESC", "3500"))

# EVERY posting that survives the hard filters is LLM-scored (user rule,
# 2026-07-11: hard filters decide eligibility, Claude decides fit — nothing
# expires unscored because of a keyword ranking). The prefilter now only
# ORDERS the queue so the most promising postings are scored first; if the
# budget cap stops a run mid-way, the best of the day are already done and
# the rest stay queued. Set JOBSCOUT_PREFILTER_KEEP > 0 to re-enable a
# top-N-per-pass throttle (emergency cost brake); 0 = no limit.
PREFILTER_KEEP = int(os.environ.get("JOBSCOUT_PREFILTER_KEEP", "0"))
# Ordering anchor (0-100), NOT an eligibility gate: signal-free titles are
# capped just below it so they always sort behind real candidates.
PREFILTER_MIN_SCORE = float(os.environ.get("JOBSCOUT_PREFILTER_MIN", "38"))
# Minimum final fit score to surface on the board (lower = see more).
SHORTLIST_MIN_SCORE = float(os.environ.get("JOBSCOUT_SHORTLIST_MIN", "45"))
# Floor for CREATING a recommendation at all. When the LLM judges a posting
# below this, no board row is written — the posting is archived with its score
# as the filter reason. Calibration: clear wrong-occupation roles score 5-15;
# the lowest recommendation the user ever engaged with sat at exactly 20
# (ESA, Marked). 131 of the user's first 152 manual skips were sub-30 rows
# that should never have existed. Applies to real LLM judgments only.
REC_MIN_SCORE = float(os.environ.get("JOBSCOUT_REC_MIN", "20"))
# Final LLM validation gate: after scoring, a second independent Claude pass
# reviews every remaining New recommendation against the profile and removes
# clear mismatches (conservative: uncertain = keep). Runs incrementally —
# each rec is validated once, and again only if it gets re-scored.
VALIDATION_ENABLED = os.environ.get("JOBSCOUT_VALIDATE", "1").lower() not in ("0", "false", "no")
VALIDATE_BATCH_SIZE = int(os.environ.get("JOBSCOUT_VALIDATE_BATCH", "10"))
# Only recs at/above this fit score get the validation pass. The board
# deliberately keeps a low-score long tail (score-all rule: every hard-filter
# survivor >= REC_MIN_SCORE gets a row), and those rows are already labelled
# weak by their own score — sending them to the validator just makes it vote
# "remove" on most of the board, trip the removal-ratio guard, and burn the
# same ~70 LLM calls again every run (observed 2026-07-10 and 2026-07-12).
# Validation is for rows a human would actually be shown prominently.
VALIDATION_MIN_SCORE = float(os.environ.get("JOBSCOUT_VALIDATE_MIN", "45"))
# Safety valve: if the validator wants to remove more than this share of the
# recs it reviewed in one run, its verdicts are discarded (model gone haywire
# beats a wiped board). Post-floor boards should see single-digit removal %.
VALIDATION_MAX_REMOVAL_RATIO = float(os.environ.get("JOBSCOUT_VALIDATE_MAX_REMOVAL", "0.5"))
# Per-company fairness cap — only meaningful in throttled mode
# (PREFILTER_KEEP > 0), where slots are scarce and one huge board could eat
# them all. With the default score-everything flow it is not applied.
PER_COMPANY_LLM_CAP = int(os.environ.get("JOBSCOUT_PER_COMPANY_CAP", "12"))
# Queue janitor: postings that stay unscored this many days (budget stops,
# LLM errors) are retired so the queue can't grow without bound. Raise it
# when a large backlog must survive several budget-capped runs.
SCORE_QUEUE_MAX_DAYS = int(os.environ.get("JOBSCOUT_QUEUE_MAX_DAYS", "10"))

# Early-career roles the profile explicitly does NOT want. Trainee programmes are
# excluded by default per the preferences; flip to include them.
INCLUDE_TRAINEE = os.environ.get("JOBSCOUT_INCLUDE_TRAINEE", "").lower() in ("1", "true", "yes")

# --- Paginated / large-board connectors --------------------------------------
# These platforms sort newest-first (or return the whole board). The caps are
# SAFETY VALVES against runaway pagination, not sampling targets — they sit
# well above every observed board size, because a cap that binds means missed
# postings (BMW/Andritz/Red Bull were silently truncated at the old 300-400
# caps until 2026-07-09). The pipeline warns when a fetch lands on a cap.
# Workday lists newest-first. Live totals on 2026-07-09: ABB and NVIDIA
# report 2000+ (the API SATURATES its total field at 2000 — the connector
# treats that as "keep walking"), Salesforce 1477. The old 1200 cap truncated
# ABB inside the 28-day window; 3000 gives saturated mega-tenants headroom.
WORKDAY_MAX_JOBS = int(os.environ.get("JOBSCOUT_WORKDAY_MAX", "3000"))
WORKDAY_PAGE_SIZE = 20
# SuccessFactors CSB tiles are referencedate-DESC; SAP alone serves 800+ and
# hit the old cap, so raise it well past the biggest observed board.
SF_MAX_JOBS = int(os.environ.get("JOBSCOUT_SF_MAX", "1500"))
SF_PAGE_SIZE = 25
# SAP Recruiting-Marketing (jobs2web) RSS — whole board in one file, region-filtered.
SF_RMK_MAX = int(os.environ.get("JOBSCOUT_SF_RMK_MAX", "1200"))
# SmartRecruiters sort order is not guaranteed, so the cap must clear the
# whole board: Bosch publishes ~4.7k postings, Red Bull ~1.2k.
SR_MAX_JOBS = int(os.environ.get("JOBSCOUT_SR_MAX", "6000"))
# Amazon (amazon.jobs) is region-filtered + newest-first. The 9-target-country
# board hit 2,857 on 2026-07-10 — it outgrew the 1500 cap within a day of that
# cap being set. 4000 leaves real headroom; hard filters make the excess cheap.
AMAZON_MAX_JOBS = int(os.environ.get("JOBSCOUT_AMAZON_MAX", "4000"))
# Oracle Cloud Recruiting (ORC) paginates newest-first; the eeho host served
# 2,117 live postings on 2026-07-10 — already past the 1500 cap set a day
# earlier. 3000 buys real headroom.
ORACLE_MAX_JOBS = int(os.environ.get("JOBSCOUT_ORACLE_MAX", "3000"))
# Apple / Microsoft careers APIs (region-filtered, newest-first).
APPLE_MAX_JOBS = int(os.environ.get("JOBSCOUT_APPLE_MAX", "500"))
MICROSOFT_MAX_JOBS = int(os.environ.get("JOBSCOUT_MICROSOFT_MAX", "500"))
# Uber (city-filtered search API) and IBM (country-filtered search API).
UBER_MAX_JOBS = int(os.environ.get("JOBSCOUT_UBER_MAX", "600"))
IBM_MAX_JOBS = int(os.environ.get("JOBSCOUT_IBM_MAX", "1500"))

# ats_type -> its cap, used by the pipeline to flag saturated (possibly
# truncated) fetches in the Coverage view. EVERY capped/paginated connector
# must appear here — a missing entry means a bound cap truncates silently.
# (jibe/un/beesite walk pages up to a fixed page budget; the values below are
# those budgets' effective job ceilings.)
CONNECTOR_CAPS: dict[str, int] = {
    "workday": WORKDAY_MAX_JOBS,
    "successfactors": SF_MAX_JOBS,
    "sf_rmk": SF_RMK_MAX,
    "smartrecruiters": SR_MAX_JOBS,
    "amazon": AMAZON_MAX_JOBS,
    "oracle": ORACLE_MAX_JOBS,
    "apple": APPLE_MAX_JOBS,
    "microsoft": MICROSOFT_MAX_JOBS,
    "uber": UBER_MAX_JOBS,
    "ibm": IBM_MAX_JOBS,
    "jibe": 4000,       # _MAX_PAGES(40) × page size 100
    "un": 3100,         # page budget (>30 × 100) + first page
    "beesite": 4000,    # _MAX_PAGES(8) × page size 500
}

# --- Status values -----------------------------------------------------------
STATUS_NEW = "New"
STATUS_MARKED = "Marked"        # user bookmark: "interesting, decide later"
STATUS_APPLIED = "Applied"
STATUS_SKIPPED = "Skipped"
STATUS_REJECTED = "Rejected"
ALL_STATUSES = (STATUS_NEW, STATUS_MARKED, STATUS_APPLIED, STATUS_SKIPPED,
                STATUS_REJECTED)
# Statuses that represent a user decision — the pipeline must never delete or
# re-score these, and they stay visible even after the posting closes.
USER_DECIDED_STATUSES = (STATUS_MARKED, STATUS_APPLIED, STATUS_SKIPPED,
                         STATUS_REJECTED)

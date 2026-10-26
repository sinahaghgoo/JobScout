# JobScout

An automated, CV-driven job recommender. You give it a CV and a list of target
companies; it gives you a ranked, up-to-date shortlist of roles that actually
fit — plus a local dashboard to manage them.

```
company list → ATS detection → feed / browser extraction → dedupe & store
             → hard filters → prefilter → Claude scoring
             → final Claude review (validation gate) → dashboard
```

The core design decision: **read structured ATS feeds instead of scraping
HTML.** Each connector pulls a company's open roles as clean JSON/XML/RSS —
title, location, posting date, description and a **stable job ID** — so dates
are reliable and your decisions (applied, skipped, marked) survive across runs.


---

## Features

- **22 ATS connectors** — Greenhouse, Lever, Ashby, Personio, SmartRecruiters,
  Recruitee, Workable, Workday, SuccessFactors, SAP Recruiting-Marketing,
  Oracle Cloud Recruiting, d.vinci, eRecruiter, BeeSite, Jibe, plus dedicated
  connectors for Amazon, Apple, Microsoft, IBM, Uber, Verbund and the
  **UN careers API**. All use public, no-auth endpoints.
- **3-tier routing** — every company goes to the cheapest surface that works:
  1. live ATS feeds,
  2. a careers-page resolver that finds feed tokens in page HTML,
  3. a browser tier (plain HTTP → `curl_cffi` impersonation → Playwright) with
     LLM extraction and a content-hash cache, so unchanged pages cost nothing.
- **Hard filters before any LLM spend** — drops internships / working-student /
  thesis roles, senior titles and 5+ years requirements, out-of-scope
  locations and languages, each with a recorded reason. Manageable (and
  extendable with custom keyword filters) from the dashboard.
- **Claude as the judge** — survivors are scored in batches against your CV and
  preferences with an explicit rubric. Prompt caching and batching keep daily
  runs at cents; a per-run budget cap prevents runaway spend. A second,
  independent Claude pass reviews every recommendation and removes only clear
  mismatches, each with an auditable reason.
- **Learns from you** — your skip history feeds back into scoring and
  validation.
- **Dead-link pruning** — job links on the board are re-checked every run;
  only unambiguous 404s are removed.
- **Local dashboard** (FastAPI + single-file frontend) — search, status chips,
  fit-score slider, date filters, archive/restore, coverage view (per-company
  fetch health), run history, hard-filter manager, live spend tracking and a
  Run button with progress.
- **Incremental & safe** — only new postings are filtered and scored; vanished
  postings close automatically, but a failed or empty fetch never wipes a
  healthy board, and user decisions are never overwritten.
- **Works without an API key** -

## Requirements

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/) (optional, but
  needed for LLM scoring and validation)
- Optional: Playwright + Chromium for client-rendered career sites

## Installation

```bash
git clone https://github.com/<your-username>/jobscout.git
cd jobscout
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# optional — enables the browser tier for JS-heavy career sites
pip install playwright && playwright install chromium
```

### API key

Either set an environment variable:

```bash
cp .env.example .env      # then fill in ANTHROPIC_API_KEY
```

or put the key in a `key.txt` file in the project root (or its parent folder).
Both `.env` and `key.txt` are git-ignored — never commit them.

## Usage

### 1. One-time setup

```bash
python -m jobscout.cli setup --cv path/to/your_cv.docx \
    --excel path/to/companies.xlsx --yellow --detect
```

- `--cv` accepts `.docx` or `.txt`. The extracted text is stored in
  `data/profile/cv.txt` (see `cv.example.txt` for the format).
- `--excel` points to your target-company workbook (column B = company name,
  column C = locations, column D = optional priority). `--yellow` loads only
  rows highlighted yellow. A CSV snapshot of the loaded list is written to
  `data/companies_seed.csv` for easy viewing.
- `--detect` classifies which ATS each company uses.

Then edit `data/profile/preferences.txt` (see `preferences.example.txt`) to
describe which roles you want and which you don't.

### 2. Run the pipeline

```bash
python -m jobscout.cli run --tiered                # full run: fetch, filter, score, validate
python -m jobscout.cli run --only "BMW,Apple"      # only selected companies
```

### 3. Open the dashboard

```bash
python -m jobscout.web                 # http://127.0.0.1:8000
python -m jobscout.web --port 8080 --no-browser
```

## CLI reference

| Command | What it does |
| --- | --- |
| `setup --cv <path> --excel <path> [--yellow] [--detect]` | Load CV and companies |
| `detect [--force] [--known-only]` | Detect each company's ATS |
| `run [--tiered \| --skip-browser] [--only "A,B"]` | The main fetch → score loop |
| `rescore [--llm] [--limit N] [--budget USD]` | Re-apply filters (and optionally re-score) after editing preferences |
| `validate [--all] [--budget USD]` | Re-run only the final Claude review |
| `linkcheck [--force]` | Verify job links on the board |
| `verify [--type workday] [--company X] [--csv out.csv]` | Scraper completeness audit |
| `reset [--keep Applied,Marked] [--yes]` | Wipe the board, keep chosen decisions |
| `stats` | Database counts |
| `show [--status New] [--limit 20]` | Print the current shortlist |

All commands are run as `python -m jobscout.cli <command>`.

## Configuration

Everything is tunable via environment variables — see
[`.env.example`](.env.example) for the full list. The most useful ones:

| Variable | Default | Purpose |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | – | Claude API key |
| `JOBSCOUT_MODEL` | `claude-sonnet-5` | Scoring model |
| `JOBSCOUT_EXTRACT_MODEL` | `claude-haiku-4-5-20251001` | Browser-tier extraction model |
| `JOBSCOUT_MAX_RUN_USD` | `30.00` | Hard LLM budget cap per run |
| `JOBSCOUT_SHORTLIST_MIN` | `45` | Minimum fit score shown on the board |
| `JOBSCOUT_VALIDATE` | `1` | Enable the final Claude review |
| `JOBSCOUT_DB` | `data/jobscout.db` | SQLite database path |

> The scoring prompt in `jobscout/matching.py` is tuned for analytics / BI
> roles in Europe. Adjust the rubric there if you target a different profile.

## Scheduling

`scripts/run_daily.sh` runs one tiered pass. On macOS, edit the paths in
`scripts/com.jobscout.daily.plist` and load it:

```bash
cp scripts/com.jobscout.daily.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.jobscout.daily.plist
```

On Linux, a cron entry pointing at `run_daily.sh` does the same.

## Tests

```bash
pip install pytest
pytest
```

## Project layout

```
jobscout/
  cli.py            command line
  web.py            FastAPI app · web_static/index.html (single-file frontend)
  pipeline.py       fetch → filter → enrich → score orchestration
  matching.py       hard filters · prefilter · Claude batch scorer
  validate.py       final Claude review of recommendations
  feedback.py       learned skip history
  linkcheck.py      dead-link pruning
  lang.py           EN/DE language detection (no dependencies)
  detect.py         company → connector routing
  resolver.py       careers-page resolver (finds feed tokens in HTML)
  browser.py        browser tier: HTTP / curl_cffi / Playwright + LLM extraction
  db.py             SQLite storage, dedupe, spend tracking, caches
  connectors/       one module per ATS platform
data/
  companies_seed.csv · careers_seed.csv     target companies + resolver hints
  profile/          cv.txt + preferences.txt (local only, git-ignored)
scripts/            daily runner, launchd plist, benchmarks, smoke tests
tests/              pytest suite
```

## Known limitations

- **Microsoft** needs outbound access to `gcsservices.careers.microsoft.com`
  (blocked on some networks — the connector reports a fetch error instead of a
  silent 0).
- Some employers (e.g. Schneider Electric, Tesla, TotalEnergies) use aggressive
  bot protection; extraction is best-effort and may stay empty from datacenter
  IPs.
- A few organisations have no public job surface at all (e.g. Google, Meta,
  TikTok, OPEC) and are listed as residual.

## Responsible use

JobScout only reads public career feeds, sends a low request volume with
polite delays, and identifies itself via its User-Agent. Please keep it that
way and respect each site's terms of service.

#!/usr/bin/env python3
"""Live benchmark of the Claude scoring stage (audit 2026-07-12).

Runs the PRODUCTION path — matching.score_batch with the real system prompt
(CV + preferences + learned skip history from the DB), real batching, real
clamps — against a graded case matrix:

  * 40 synthetic cases spanning perfect -> irrelevant, plus edge cases
    (missing/short/long descriptions, vague & quirky titles, conflicting
    requirements, multilingual asks, fixed-term, remote-EU, visa notes);
  * real queued postings with hand-set expected bands;
  * a consistency probe (one batch scored 3x, then once shuffled);
  * a validator-agreement probe (planted mistakes must be removed,
    borderline-by-design rows must be kept).

Usage:
    JOBSCOUT_DB=/path/to/jobscout.db python3 scripts/bench_scoring.py [--out r.json]

READ-ONLY: never writes to the DB. Spends real API money (~$0.60 full pass).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobscout import config, db, feedback, matching, validate  # noqa: E402

V = "Vienna, Austria"
PERM = " Unbefristete Vollzeitstelle."
BI_CORE = ("You will write SQL queries against our warehouse, build and "
           "maintain Power BI dashboards, define KPIs with commercial "
           "stakeholders and turn findings into clear recommendations. "
           "Tools: SQL, Power BI, Excel, a bit of Tableau. ")

# --- Synthetic case matrix ----------------------------------------------------
# id, title, location, description, expected (lo, hi), optional checks:
#   gates: {gate: value-or-set}  | watch_has: substring | no_cap: True
CASES: list[dict] = [
    # ---- perfect matches -------------------------------------------------
    dict(id=1, title="Junior BI Analyst (m/w/d)", loc=V,
         desc=BI_CORE + "0-2 years of experience welcome; mentoring provided."
         + PERM, expect=(80, 100), gates={"fn": {"core"}}, no_cap=True),
    dict(id=2, title="Data Analyst (m/w/d)", loc="München, Deutschland",
         desc="Du erstellst Auswertungen und Dashboards in Power BI, "
              "schreibst SQL-Abfragen, entwickelst KPI-Berichte für die "
              "Fachbereiche und präsentierst Ergebnisse den Stakeholdern. "
              "Auch für Berufseinsteiger:innen geeignet." + PERM,
         expect=(78, 100), gates={"lad": {"de", "en"}}, no_cap=True),
    dict(id=3, title="Graduate Business Intelligence Analyst",
         loc="London, United Kingdom",
         desc=BI_CORE + "Two-year graduate development programme with a "
              "permanent contract from day one.", expect=(75, 100)),
    # ---- strong ----------------------------------------------------------
    dict(id=4, title="Business Analyst", loc="Amsterdam, Netherlands",
         desc="Gather reporting requirements from operations teams, analyse "
              "process data in SQL, own weekly management dashboards in Qlik "
              "and drive improvements with stakeholders.",
         expect=(60, 90)),
    dict(id=5, title="Commercial Analytics Analyst", loc="Stockholm, Sweden",
         desc="Pricing and promotion analytics for the Nordic business: SQL, "
              "Excel models, Tableau dashboards. 2-4 years of analytics "
              "experience expected.", expect=(65, 95)),
    dict(id=6, title="Insights Analyst", loc="Dublin, Ireland",
         desc="Turn campaign and customer data into insights decks, maintain "
              "reporting dashboards, partner with marketing stakeholders. "
              "SQL and strong Excel required.", expect=(60, 90)),
    # ---- acceptable --------------------------------------------------------
    dict(id=7, title="Reporting Specialist", loc="Zurich, Switzerland",
         desc="Produce monthly management reports in Excel and SAP BW, "
              "improve reporting templates, limited SQL. German and English "
              "environment.", expect=(50, 80)),
    dict(id=8, title="Operations Analyst", loc="Berlin, Germany",
         desc="Track operational KPIs, prepare weekly performance reviews, "
              "coordinate between warehouse teams and management, maintain "
              "dashboards. Some Excel analysis, lots of alignment meetings.",
         expect=(25, 65)),
    # ---- borderline by design (function caps) ------------------------------
    dict(id=9, title="Junior Controller (m/w/d)", loc="Munich, Germany",
         desc="Monatsreporting, Budgetplanung, Forecasts in Excel und SAP, "
              "Unterstützung bei Abschlüssen und Ad-hoc-Analysen.",
         expect=(25, 45), gates={"fn": {"fin"}}),
    dict(id=10, title="Graduate Technology Consultant - Data & Analytics",
         loc=V, desc="Client-facing consulting: requirements workshops, "
              "PowerPoint studies, some dashboard prototyping in Power BI, "
              "travel up to 60%.", expect=(25, 50),
         gates={"fn": {"consult", "core", "adj"}}),
    dict(id=11, title="BI Manager", loc="Hamburg, Germany",
         desc="Own the BI roadmap-execution for one business unit as an "
              "individual contributor: build Power BI models yourself, no "
              "direct reports.", expect=(30, 55)),
    # ---- poor (wrong function, white collar) --------------------------------
    dict(id=12, title="Data Engineer", loc="Berlin, Germany",
         desc="Build ELT pipelines with dbt, Airflow and Spark on AWS; "
              "own data models and infrastructure as code.",
         expect=(15, 35), gates={"fn": {"eng"}}),
    dict(id=13, title="Data Scientist", loc="Paris, France",
         desc="Develop ML models in Python (scikit-learn, PyTorch), run "
              "experiments, publish internal research.", expect=(10, 35),
         gates={"fn": {"eng"}}),
    dict(id=14, title="Product Manager - Analytics Platform",
         loc="Amsterdam, Netherlands",
         desc="Own the roadmap for our internal analytics platform, write "
              "specs, prioritise the backlog, run discovery with users.",
         expect=(15, 40), gates={"fn": {"pm"}}),
    dict(id=15, title="Customer Support Analyst", loc=V,
         desc="Handle inbound tickets, resolve user issues, escalate bugs, "
              "keep the knowledge base current.", expect=(5, 30),
         gates={"fn": {"support"}}),
    # ---- completely irrelevant ---------------------------------------------
    dict(id=16, title="Kfz-Mechatroniker (m/w/d)", loc="Munich, Germany",
         desc="Wartung und Instandsetzung von Fahrzeugen, Fehlerdiagnose, "
              "Reparatur von Bremsanlagen.", expect=(0, 15),
         gates={"fn": {"trade"}}),
    dict(id=17, title="Trial Lawyer", loc=V,
         desc="Represent clients in commercial litigation before Austrian "
              "courts; draft pleadings; 2nd state exam required.",
         expect=(0, 15)),
    dict(id=18, title="Warehouse Operative", loc="Dublin, Ireland",
         desc="Pick and pack customer orders, operate pallet trucks, "
              "shift work including weekends.", expect=(0, 15)),
    # ---- hard gates that reach the LLM --------------------------------------
    dict(id=19, title="Data Analyst - French Market", loc="Paris, France",
         desc=BI_CORE + "Fluent French is required for daily client contact.",
         expect=(0, 25), gates={"lrq": {"french", "fr"}}),
    dict(id=20, title="Senior Data Analyst (m/w/d)", loc=V,
         desc=BI_CORE + "Mindestens 6 Jahre einschlägige Berufserfahrung.",
         expect=(0, 30)),
    dict(id=21, title="Data Analyst", loc=V,
         desc=BI_CORE + "Minimum 5 years of professional experience in "
              "analytics required.", expect=(0, 30)),
    dict(id=22, title="Business Analyst", loc="New York, NY",
         desc=BI_CORE, expect=(0, 25), gates={"loc": {"out"}}),
    dict(id=23, title="Analista de Datos", loc="Barcelona, Spain",
         desc="Buscamos analista de datos para nuestro equipo: consultas "
              "SQL, cuadros de mando en Power BI, informes para las áreas "
              "de negocio. Experiencia de 1-2 años. Contrato indefinido.",
         expect=(0, 25)),
    dict(id=24, title="Praktikum Data Analytics", loc="Berlin, Germany",
         desc="6-monatiges Pflichtpraktikum im Analytics-Team: SQL-Abfragen, "
              "Dashboard-Pflege, Ad-hoc-Auswertungen.", expect=(0, 20),
         gates={"rt": None}),          # any non-ok value acceptable
    dict(id=25, title="Data Analyst", loc="Madrid, Spain",
         desc=BI_CORE, expect=(0, 25), gates={"loc": {"out"}}),
    # ---- edge cases -----------------------------------------------------------
    dict(id=26, title="Junior BI Analyst", loc=V, desc="",
         expect=(30, 75), watch_has="no description"),
    dict(id=27, title="Data Analyst", loc=V, desc="Analyze data.",
         expect=(20, 75)),
    dict(id=28, title="Business Intelligence Analyst", loc=V,
         desc=("Join our team. " * 150)
              + "About the role: build Power BI dashboards and SQL reports "
                "for commercial teams. "
              + ("We value diversity and offer great benefits. " * 120)
              + "Your profile: 2-4 Jahre Erfahrung mit SQL und Power BI, "
                "sehr gutes Deutsch und Englisch, keine Reisebereitschaft "
                "nötig. Unbefristeter Vertrag.",
         expect=(60, 95)),
    dict(id=29, title="Analyst Coordination & Steering", loc=V,
         desc="Coordinate steering meetings, align stakeholders, prepare "
              "management presentations, maintain the action-item tracker, "
              "occasionally pull figures from existing reports.",
         expect=(25, 60)),
    dict(id=30, title="Zahlen-Detektiv (m/w/d) - Data & Reporting",
         loc="Cologne, Germany",
         desc="Hinter dem Titel steckt klassische BI-Arbeit: SQL, Power BI, "
              "Berichtswesen für den Vertrieb, KPI-Definition mit den "
              "Fachabteilungen.", expect=(60, 95)),
    dict(id=31, title="Junior Data Analyst", loc="Berlin, Germany",
         desc=BI_CORE + "Requirement: mindestens 5 Jahre Berufserfahrung im "
              "Analytics-Umfeld.", expect=(0, 30)),
    dict(id=32, title="Business Data Analyst", loc="Frankfurt, Germany",
         desc="We are looking for a Business Data Analyst. Du arbeitest mit "
              "SQL und Power BI, erstellst Reports für internationale "
              "Stakeholder. The role is part of our hybrid Frankfurt hub. "
              "Fließendes Englisch, Deutsch von Vorteil.",
         expect=(60, 95), gates={"lad": {"en", "de"}}, no_cap=True),
    dict(id=33, title="Junior Data Analyst", loc="Amsterdam, Netherlands",
         desc=BI_CORE + "You must be fluent in English or French.",
         expect=(55, 95), gates={"lrq": {"none"}}),
    dict(id=34, title="Junior Data Analyst", loc="Amsterdam, Netherlands",
         desc=BI_CORE + "French is a plus but absolutely not required.",
         expect=(55, 95), gates={"lrq": {"none"}}),
    dict(id=35, title="Data Analyst", loc="Remote - Europe",
         desc=BI_CORE + "Fully remote within the EU.", expect=(55, 95)),
    dict(id=36, title="Data Analyst", loc=V,
         desc=BI_CORE + "This is a 12-month fixed-term contract (parental "
              "leave cover).", expect=(45, 90), watch_has="fix"),
    dict(id=37, title="Data Analyst", loc="London, United Kingdom",
         desc=BI_CORE + "You must already hold the right to work in the UK; "
              "we are unable to sponsor visas.", expect=(45, 90),
         watch_has="visa"),
    dict(id=38, title="Credit Risk Analyst", loc="Frankfurt, Germany",
         desc="Assess counterparty credit risk, run rating models, prepare "
              "credit committee papers. Some SQL for data pulls.",
         expect=(15, 45)),
    dict(id=39, title="Data Governance Analyst", loc=V,
         desc="Maintain the data catalogue, define data-quality rules, "
              "steward master data, align data owners.", expect=(20, 45),
         gates={"fn": {"gov"}}),
    dict(id=40, title="ESG Data Analyst", loc="Berlin, Germany",
         desc="Collect and validate ESG metrics, build sustainability "
              "dashboards in Power BI, support CSRD reporting with SQL "
              "analyses.", expect=(55, 90)),
    dict(id=41, title="Business Intelligence Engineer",
         loc="London, United Kingdom",
         desc="Own reporting and dashboards for the retail analytics team: "
              "write complex SQL, build QuickSight and Power BI dashboards, "
              "define KPIs with business stakeholders, automate recurring "
              "reports. Light scripting for data pulls; no production "
              "pipeline ownership.", expect=(55, 95),
         gates={"fn": {"core", "adj"}}),
]

# --- Real queued postings (fetched from the DB by id) ---------------------------
REAL_EXPECT: dict[int, dict] = {
    36588: dict(expect=(15, 45), note="Amazon BIE, but THIS ad is production-"
                "pipeline engineering — eng cap is correct (verified by "
                "reading the ad)"),
    36743: dict(expect=(20, 50), note="Commercial Finance Analyst -> fin-ish"),
    37684: dict(expect=(15, 45), note="BMW Kreditwürdigkeitsprüfung -> credit"),
    57573: dict(expect=(0, 30), note="Novartis Vilvoorde = Belgium -> out"),
    52495: dict(expect=(10, 40), note="Palantir SRE Ops Analyst -> eng"),
    36550: dict(expect=(5, 35), note="Amazon Data Scientist -> eng"),
    37132: dict(expect=(15, 50), note="Apple 15-month FTC sales finance",
                watch_has=None),
    60474: dict(expect=(15, 45), note="Schneider VIE governance analyst"),
    36167: dict(expect=(0, 20), note="Adidas store Sales Consultant temp"),
    38223: dict(expect=(10, 40), note="Booking Group Data PM -> pm"),
}


def _real_jobs() -> list[dict]:
    out = []
    with db.connect() as conn:
        for pid, exp in REAL_EXPECT.items():
            r = conn.execute(
                """SELECT p.id, p.title, p.location, p.posted_date,
                          p.description, c.name company
                   FROM posting p JOIN company c ON c.id=p.company_id
                   WHERE p.id=?""", (pid,)).fetchone()
            if r is None:
                continue
            out.append(dict(id=r["id"], title=r["title"], company=r["company"],
                            location=r["location"] or "", posted=r["posted_date"] or "",
                            description=r["description"] or "", pre_score=50.0,
                            signals={}, _exp=exp))
    return out


def _to_job(c: dict) -> dict:
    return dict(id=c["id"], title=c["title"], company="BenchCo",
                location=c["loc"], posted="2026-07-10",
                description=c["desc"], pre_score=50.0, signals={})


def _grade(case: dict, res) -> list[str]:
    fails = []
    lo, hi = case["expect"]
    if not (lo <= res.fit_score <= hi):
        fails.append(f"score {res.fit_score:.0f} outside [{lo},{hi}]")
    g = (res.signals or {}).get("_gates") or {}
    for gate, allowed in (case.get("gates") or {}).items():
        got = str(g.get(gate, "")).lower()
        if allowed is None:                 # any non-ok value required
            if got in matching._GATE_PASS:
                fails.append(f"gate {gate} expected non-ok, got {got!r}")
        elif got not in allowed:
            fails.append(f"gate {gate}={got!r} not in {sorted(allowed)}")
    want = case.get("watch_has")
    if want and want.lower() not in (res.watch_outs or "").lower():
        fails.append(f"watch missing {want!r}: {res.watch_outs[:60]!r}")
    if case.get("no_cap") and "capped at" in (res.watch_outs or ""):
        fails.append(f"unexpected cap: {res.watch_outs[:80]!r}")
    return fails


def _score(cv, prefs, jobs, tracker, skip_rules):
    """score_batch, keeping the model's raw gates for grading."""
    raw_results = {}
    orig_parse = matching._parse_llm_array

    def capture(text):
        data = orig_parse(text)
        if data:
            for item in data:
                if isinstance(item, dict) and "id" in item:
                    raw_results[item.get("id")] = item.get("g") or {}
        return data

    matching._parse_llm_array = capture
    try:
        out = matching.score_batch(cv, prefs, jobs, tracker,
                                   extra_rules=skip_rules)
    finally:
        matching._parse_llm_array = orig_parse
    for jid, res in out.items():
        res.signals["_gates"] = raw_results.get(jid) or raw_results.get(
            int(jid) if str(jid).isdigit() else jid, {})
    return out


def _load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"cases": {}, "consistency_runs": [], "validator": {},
            "spent": 0.0, "calls": 0, "unscored": []}


def _save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=1, ensure_ascii=False))


def _profile_bits():
    profile = db.get_profile()
    cv = profile["cv_text"] if profile else ""
    prefs = profile["preferences_note"] if profile else ""
    skip_cats, applied_cats = feedback.load_profiles(db)
    return cv, prefs, feedback.prompt_section(skip_cats, applied_cats)


PROBE_IDS = [1, 5, 9, 11, 19, 29, 38, 40]      # consistency probe, all bands


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="bench_state.json")
    ap.add_argument("--out", default="bench_results.json")
    ap.add_argument("--slice", help="A:B — score matrix jobs [A:B) and exit")
    ap.add_argument("--consistency-rep", type=int,
                    help="0..2 identical rep, 3 = shuffled rep")
    ap.add_argument("--validator", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--rerun", nargs="*", type=int,
                    help="synthetic case ids to re-score (overwrites)")
    args = ap.parse_args()

    state_path = Path(args.state)
    state = _load_state(state_path)

    if not args.report and not matching.llm_available():
        print("no ANTHROPIC_API_KEY — aborting"); return 2
    cv, prefs, skip_rules = _profile_bits()
    if not args.report and (not cv or not prefs):
        print("no CV/preferences in DB — aborting"); return 2
    tracker = matching.SpendTracker(cap_usd=6.0)
    t0 = time.time()

    def _persist_cost():
        state["spent"] = round(state.get("spent", 0.0) + tracker.spent, 4)
        state["calls"] = state.get("calls", 0) + tracker.calls
        _save_state(state_path, state)

    # ---- stage: score one slice of the matrix ------------------------------
    if args.slice or args.rerun:
        if args.rerun:
            todo = [_to_job(c) for c in CASES if c["id"] in args.rerun]
        else:
            a, b = (int(x) for x in args.slice.split(":"))
            todo = ([_to_job(c) for c in CASES] + _real_jobs())[a:b]
        for i in range(0, len(todo), config.LLM_BATCH_SIZE):
            batch = todo[i:i + config.LLM_BATCH_SIZE]
            out = _score(cv, prefs, batch, tracker, skip_rules)
            for job in batch:
                res = out.get(job["id"])
                tag = "S" if any(c["id"] == job["id"] for c in CASES) else "R"
                if res is None:
                    state["unscored"].append(job["id"]); continue
                state["cases"][f"{tag}{job['id']}"] = {
                    "id": job["id"], "title": job["title"],
                    "score": res.fit_score,
                    "gates": (res.signals or {}).get("_gates"),
                    "why": res.rationale, "watch": res.watch_outs,
                    "exp_note": (job.get("_exp") or {}).get("note"),
                    "real_expect": (job.get("_exp") or {}).get("expect"),
                }
        _persist_cost()
        print(f"slice done | ${tracker.spent:.3f} this call | "
              f"{time.time() - t0:.0f}s", flush=True)
        return 0

    # ---- stage: one consistency rep ------------------------------------------
    if args.consistency_rep is not None:
        probe = [_to_job(c) for c in CASES if c["id"] in PROBE_IDS]
        if args.consistency_rep == 3:
            random.Random(42).shuffle(probe)
        out = _score(cv, prefs, probe, tracker, skip_rules)
        state["consistency_runs"].append(
            {str(j): out[j].fit_score for j in out})
        _persist_cost()
        print(f"consistency rep {args.consistency_rep} done | "
              f"${tracker.spent:.3f}", flush=True)
        return 0

    # ---- stage: validator probe -------------------------------------------------
    if args.validator:
        sys_blocks = validate._system_blocks(cv, prefs, skip_rules)
        planted = [
            dict(rec_id=9001, title="Data Analyst - French Market",
                 company="BenchCo", location="Paris, France", fit_score=55.0,
                 description=BI_CORE + "Fluent French required for daily "
                 "client calls.", rationale="Core analyst role.",
                 watch_outs="None", _want="remove"),
            dict(rec_id=9002, title="Praktikum Business Intelligence",
                 company="BenchCo", location="Berlin, Germany", fit_score=60.0,
                 description="6-monatiges Praktikum im BI-Team, SQL und "
                 "Power BI.", rationale="BI stack matches.",
                 watch_outs="None", _want="remove"),
            dict(rec_id=9003, title="Elektriker (m/w/d)", company="BenchCo",
                 location=V, fit_score=48.0,
                 description="Elektroinstallationen im Gewerbebau, "
                 "Schaltschrankverdrahtung.", rationale="Vienna location.",
                 watch_outs="None", _want="remove"),
        ]
        keepers = [
            dict(rec_id=9101, title="Junior BI Analyst", company="BenchCo",
                 location=V, fit_score=88.0, description=BI_CORE + PERM,
                 rationale="Direct stack match.", watch_outs="None",
                 _want="keep"),
            dict(rec_id=9102, title="Junior Controller (m/w/d)",
                 company="BenchCo", location="Munich, Germany", fit_score=45.0,
                 description="Monatsreporting, Budgetplanung, Forecasts in "
                 "Excel und SAP.", rationale="Finance function with "
                 "reporting overlap.", watch_outs="capped at 45 — function: fin",
                 _want="keep"),
            dict(rec_id=9103, title="Graduate Technology Consultant - Data",
                 company="BenchCo", location=V, fit_score=45.0,
                 description="Consulting with Power BI prototyping and "
                 "dashboarding for clients.", rationale="Consulting-adjacent "
                 "analytics.", watch_outs="capped at 45 — function: consult",
                 _want="keep"),
            dict(rec_id=9104, title="Operations Analyst", company="BenchCo",
                 location="Berlin, Germany", fit_score=55.0,
                 description="KPI tracking, weekly performance dashboards, "
                 "SQL pulls, stakeholder reviews.", rationale="Analytics-"
                 "flavoured ops role.", watch_outs="None", _want="keep"),
            dict(rec_id=9105, title="ESG Data Analyst", company="BenchCo",
                 location="Berlin, Germany", fit_score=70.0,
                 description="ESG metrics collection, Power BI dashboards, "
                 "CSRD reporting with SQL.", rationale="Adjacent analytics.",
                 watch_outs="None", _want="keep"),
            dict(rec_id=9106, title="Data Analyst", company="BenchCo",
                 location="Remote - Europe", fit_score=72.0,
                 description=BI_CORE + "Fully remote within the EU.",
                 rationale="Remote-EU core analyst.", watch_outs="None",
                 _want="keep"),
            dict(rec_id=9107, title="Business Analyst", company="BenchCo",
                 location="Amsterdam, Netherlands", fit_score=68.0,
                 description="Requirements gathering, SQL analysis, Qlik "
                 "dashboards for operations.", rationale="Solid BA/BI mix.",
                 watch_outs="None", _want="keep"),
        ]
        probe = planted + keepers
        verdicts = validate._review_batch(sys_blocks, probe, tracker)
        for r in probe:
            v = verdicts.get(r["rec_id"], {"v": "missing", "conf": "-"})
            state["validator"][str(r["rec_id"])] = {
                "title": r["title"], "want": r["_want"], "got": v}
        _persist_cost()
        print(f"validator probe done | ${tracker.spent:.3f}", flush=True)
        return 0

    # ---- stage: offline grading & report -----------------------------------------
    if not args.report:
        print("nothing to do (use --slice / --consistency-rep / "
              "--validator / --report)")
        return 2

    failures: list[str] = []
    graded = {}
    for key, rec in state["cases"].items():
        fails = []
        case = next((c for c in CASES if c["id"] == rec["id"]), None) \
            if key.startswith("S") else None

        class _R:                                    # grade from stored fields
            fit_score = rec["score"]
            watch_outs = rec["watch"] or ""
            rationale = rec["why"] or ""
            signals = {"_gates": rec.get("gates") or {}}
        if case:
            fails = _grade(case, _R)
        elif rec.get("real_expect"):
            lo, hi = rec["real_expect"]
            if not (lo <= rec["score"] <= hi):
                fails.append(f"score {rec['score']:.0f} outside [{lo},{hi}] "
                             f"({rec.get('exp_note')})")
        graded[key] = dict(rec, fails=fails)
        for f in fails:
            failures.append(f"[{key} {rec['title'][:44]}] {f}")

    spread = {}
    runs = state.get("consistency_runs") or []
    for jid in PROBE_IDS:
        vals = [r[str(jid)] for r in runs if str(jid) in r]
        if vals:
            spread[jid] = {"scores": vals,
                           "spread": round(max(vals) - min(vals), 1)}
    worst = max((v["spread"] for v in spread.values()), default=0.0)
    line_flips = sum(
        1 for v in spread.values()
        if (min(v["scores"]) < config.SHORTLIST_MIN_SCORE <= max(v["scores"]))
        or (min(v["scores"]) < config.REC_MIN_SCORE <= max(v["scores"])))
    if runs:
        if worst > 15:
            failures.append(f"consistency: worst spread {worst} > 15")
        if line_flips > 1:
            failures.append(f"consistency: {line_flips} threshold flips")

    removed_planted, keeper_removed = 0, 0
    for rid, v in (state.get("validator") or {}).items():
        got = v["got"]
        eff = "remove" if (got.get("v") == "remove"
                           and got.get("conf") == "hi") else "keep"
        if v["want"] == "remove" and eff == "remove":
            removed_planted += 1
        if v["want"] == "keep" and eff == "remove":
            keeper_removed += 1
            failures.append(f"validator removed keeper {v['title']!r} "
                            f"({got.get('why', '')!r})")
    if state.get("validator") and removed_planted < 2:
        failures.append(f"validator caught only {removed_planted}/3 "
                        "planted mistakes")

    n_scored = len(graded)
    results = {
        "cases": graded, "consistency": spread,
        "validator": state.get("validator"),
        "cost": {"usd_spent": state.get("spent"), "calls": state.get("calls"),
                 "jobs_scored": n_scored,
                 "usd_per_job": round(state.get("spent", 0) / max(
                     state.get("calls", 1), 1), 5)},
        "unscored": state.get("unscored"), "failures": failures,
    }
    Path(args.out).write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"\n==== BENCH SUMMARY ====\n{n_scored} cases | "
          f"${state.get('spent', 0):.3f} total | worst spread {worst} | "
          f"validator {removed_planted}/3 planted removed, "
          f"{keeper_removed} keepers lost | {len(failures)} failure(s)")
    for f in failures:
        print("  FAIL:", f)
    if state.get("unscored"):
        print("  UNSCORED:", state["unscored"])
    print(f"results -> {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

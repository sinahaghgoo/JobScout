"""JobScout command line.

    python -m jobscout.cli setup --cv <path> --excel <path> --yellow  # one-time
    python -m jobscout.cli detect [--force] [--limit N]     # classify ATS
    python -m jobscout.cli run [--tiered]                   # the daily loop
    python -m jobscout.cli rescore [--llm]                  # re-apply filters / re-score
    python -m jobscout.cli stats
    python -m jobscout.cli show [--status New] [--limit 20]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from . import companies as companies_mod
from . import config, db, matching, profile
from . import pipeline


def _cmd_setup(args) -> None:
    db.init_db()
    print("Loading profile (CV + preferences)...")
    cv_src = Path(args.cv) if args.cv else None
    profile.init_profile(cv_source=cv_src)
    p = db.get_profile()
    print(f"  CV: {len(p['cv_text'])} chars | preferences: {len(p['preferences_note'])} chars")
    if args.excel:
        if args.yellow:
            n = companies_mod.seed_yellow_companies(Path(args.excel))
            print(f"  Seeded {n} YELLOW-highlighted target companies from {Path(args.excel).name}")
        else:
            n = companies_mod.seed_companies(Path(args.excel))
            print(f"  Seeded {n} companies from {Path(args.excel).name}")
    if args.detect:
        print("Detecting ATS (one-time)...")
        res = pipeline.detect_all(limit=args.limit)
        print(f"  -> {res}")
    print("Setup complete. Next:")
    print("  python -m jobscout.cli run      # fetch + score the daily loop")
    print("  python -m jobscout.web          # open the dashboard")


def _cmd_detect(args) -> None:
    db.init_db()
    res = pipeline.detect_all(force=args.force, limit=args.limit, known_only=args.known_only,
                              use_resolver=not args.no_resolver)
    print(f"Detection: {res}")


def _cmd_run(args) -> None:
    only = [s.strip() for s in (args.only or "").split(",") if s.strip()] or None
    try:
        if args.tiered:
            summary = pipeline.run_tiered(detect_first=True, only=only)
        else:
            summary = pipeline.run(detect_first=args.detect,
                                   skip_browser=args.skip_browser, only=only)
    except pipeline.UnknownCompanyError as e:
        known = ", ".join(c["name"] for c in db.get_companies())
        print(f"Cannot start: {e}\nKnown companies: {known}")
        raise SystemExit(2)
    print("\nRun summary:")
    for k, v in summary.as_dict().items():
        print(f"  {k:18} {v}")
    if not config.ANTHROPIC_API_KEY:
        print("\n(No ANTHROPIC_API_KEY set — used the heuristic scorer. "
              "Set the key for Claude-written rationales.)")


def _cmd_rescore(args) -> None:
    """Re-apply hard filters to the board; optionally re-score survivors with
    the LLM (useful after changing preferences or upgrading the model)."""
    db.init_db()
    removed = pipeline.refilter_existing(verbose=True)
    print(f"Hard filters removed {removed} stale recommendations.")
    if not args.llm:
        return
    if not matching.llm_available():
        print("No Anthropic key found — skipping LLM re-score.")
        return
    from . import feedback
    profile_row = db.get_profile()
    cv = profile_row["cv_text"] if profile_row else ""
    prefs = profile_row["preferences_note"] if profile_row else ""
    skip_cats, applied_cats = feedback.load_profiles(db)
    skip_rules = feedback.prompt_section(skip_cats, applied_cats)
    rows = [r for r in db.board_rows(statuses=["New"]) if r["is_open"]]
    if args.heuristic_only:
        rows = [r for r in rows if not r["llm_used"]]
    if args.limit:
        rows = rows[: args.limit]
    print(f"Re-scoring {len(rows)} New recommendations with {config.ANTHROPIC_MODEL} ...")
    tracker = matching.SpendTracker(cap_usd=args.budget)
    batch_size = max(1, config.LLM_BATCH_SIZE)
    done = 0
    for i in range(0, len(rows), batch_size):
        if tracker.exhausted:
            print(f"  budget cap (${tracker.cap:.2f}) reached — stopping at {done}")
            break
        batch = [{
            "id": r["posting_id"], "title": r["title"], "company": r["company"],
            "location": r["location"] or "", "posted": r["posted_date"] or "",
            "description": r["description"] or "",
            "pre_score": r["prefilter_score"] or 0, "signals": {},
        } for r in rows[i:i + batch_size]]
        results = matching.score_batch(cv, prefs, batch, tracker,
                                       extra_rules=skip_rules)
        for job in batch:
            res = results.get(job["id"])
            if res is None:
                continue
            if res.fit_score < config.REC_MIN_SCORE:
                # The floor applies on re-score too: a rec the LLM now judges
                # clearly off-profile leaves the board instead of lingering.
                rec = next((r for r in rows if r["posting_id"] == job["id"]), None)
                if rec is not None:
                    db.delete_recommendation(rec["rec_id"])
                why = (res.rationale or "").strip()
                db.mark_posting_filtered(
                    job["id"], f"scored {res.fit_score:.0f}/100 by the LLM"
                               + (f" — {why[:160]}" if why else ""))
            else:
                db.save_recommendation(
                    posting_id=job["id"], prefilter_score=job["pre_score"],
                    fit_score=res.fit_score, rationale=res.rationale,
                    watch_outs=res.watch_outs, llm_used=True)
            done += 1
        print(f"  {done}/{len(rows)} re-scored (${tracker.spent:.3f})")
    if tracker.spent:
        total = db.meta_add_float("llm_spend_usd_total", tracker.spent)
        db.meta_set("llm_spend_usd_last_run", f"{tracker.spent:.4f}")
        print(f"Re-score cost ≈ ${tracker.spent:.3f} (cumulative ≈ ${total:.2f})")
    _run_validation(budget=args.budget)


def _run_validation(revalidate_all: bool = False, budget=None) -> None:
    """Run the final LLM validation gate and print its outcome."""
    from . import validate
    if not matching.llm_available():
        print("No Anthropic key found — skipping final validation.")
        return
    tracker = matching.SpendTracker(cap_usd=budget)
    outcome = validate.validate_new_recommendations(
        tracker=tracker, verbose=True, revalidate_all=revalidate_all)
    print(f"Final review: {outcome.reviewed} reviewed, {outcome.removed} removed, "
          f"{outcome.flagged} flagged, {outcome.pending} pending")
    for n in outcome.notes:
        print(f"  note: {n}")
    if tracker.spent:
        total = db.meta_add_float("llm_spend_usd_total", tracker.spent)
        print(f"Validation cost ≈ ${tracker.spent:.3f} (cumulative ≈ ${total:.2f})")


def _cmd_validate(args) -> None:
    """Run only the final validation gate over the current board."""
    db.init_db()
    _run_validation(revalidate_all=args.all, budget=args.budget)


def _cmd_reset(args) -> None:
    """Wipe the board for a from-scratch rebuild, keeping only chosen
    decisions (default: Applied + Marked). What the deleted Skipped rows
    taught the recommender is frozen into a baseline first, so the learned
    skip patterns survive the wipe. Dry-run unless --yes."""
    from . import feedback, matching as m
    db.init_db()
    keep = [s.strip() for s in (args.keep or "Applied,Marked").split(",") if s.strip()]
    bad = [s for s in keep if s not in config.ALL_STATUSES]
    if bad:
        print(f"Unknown status(es): {', '.join(bad)}. "
              f"Valid: {', '.join(config.ALL_STATUSES)}")
        return

    # duplicates among the kept rows (same company + title + location under
    # different job_ids): keep the earliest recommendation of each vacancy
    kept_rows = [r for r in db.recommended_open_postings() if r["status"] in keep]
    by_fp: dict = {}
    dup_ids: list[int] = []
    for r in sorted(kept_rows, key=lambda r: r["rec_id"]):
        fp = m.posting_fingerprint(r["company_id"], r["title"], r["location"] or "")
        if fp in by_fp:
            dup_ids.append(r["rec_id"])
        else:
            by_fp[fp] = r["rec_id"]

    s = db.stats()
    frozen = [st for st in ("Skipped", "Applied") if st not in keep]
    print(f"Reset plan (DB: {os.environ.get('JOBSCOUT_DB') or config.DB_PATH}):")
    print(f"  keep      : {', '.join(keep)} "
          f"({len(kept_rows) - len(dup_ids)} recommendations, "
          f"{len(dup_ids)} duplicate(s) among them collapsed)")
    print(f"  delete    : every other recommendation "
          f"({s['recommendations'] - len(kept_rows)}) and every posting "
          f"without a kept recommendation (~{s['postings_total']})")
    print(f"  preserved : companies, profile, spend history, extraction cache"
          + (f", learned patterns from {', '.join(frozen)} (frozen to baseline)"
             if frozen else ""))
    if not args.yes:
        print("\nDry run — nothing changed. Re-run with --yes to execute.")
        return

    if frozen:
        feedback.persist_baselines(db, freeze_statuses=frozen)
    stats = db.reset_except(keep, drop_rec_ids=dup_ids)
    print("\nDone:")
    for k, v in stats.items():
        print(f"  {k:26} {v}")
    print("Next: python -m jobscout.cli run --tiered   (rebuilds from scratch)")


def _cmd_linkcheck(args) -> None:
    """Verify the links behind the board's New/Marked recommendations and
    archive the dead ones — without running the whole pipeline."""
    from . import linkcheck
    db.init_db()
    rows = db.postings_for_linkcheck()
    print(f"Checking {len(rows)} recommendation links "
          f"({'ignoring' if args.force else 'respecting'} the "
          f"{config.LINKCHECK_TTL_HOURS:g}h recheck TTL)...")
    out = linkcheck.recheck_recommendations(force=args.force, verbose=True)
    print(f"\n{out.checked} checked, {out.dead} dead, {out.unknown} unknown, "
          f"{out.skipped_fresh} skipped (checked recently)")
    if out.by_type:
        print(f"  {'platform':17} {'alive':>6} {'dead':>5} {'unknown':>8}")
        for t in sorted(out.by_type):
            v = out.by_type[t]
            print(f"  {t:17} {v['alive']:6} {v['dead']:5} {v['unknown']:8}")
    if out.recs_removed or out.postings_closed:
        print(f"Removed {out.recs_removed} recommendations; closed "
              f"{out.postings_closed} postings (archived as 'job page gone').")
    for line in out.removed:
        print(f"  - {line}")
    if not out.dead:
        print("Every checked link is reachable.")


def _cmd_verify(args) -> None:
    """Verify scraper completeness against each source's own counts."""
    from . import verify
    db.init_db()
    print("Verifying boards against their sources "
          "(* = raw board total; connector keeps a filtered subset)...")
    checks = verify.verify_boards(only_type=args.type, only_company=args.company,
                                  include_browser=args.browser)
    bad = [c for c in checks if c.status in ("MISMATCH", "ERROR")]
    ok = sum(1 for c in checks if c.status == "OK")
    chk = sum(1 for c in checks if c.status == "CHECK")
    print(f"\n{len(checks)} boards: {ok} OK, {chk} CHECK (no independent total), "
          f"{len(bad)} needing attention")
    for c in bad:
        print(f"  ! {c.company} [{c.status}] fetched={c.fetched} "
              f"total={c.api_total} {('; '.join(c.notes) or c.error)[:100]}")
    if args.csv:
        verify.write_csv(checks, args.csv)
        print(f"Full report: {args.csv}")


def _cmd_stats(args) -> None:
    for k, v in db.stats().items():
        print(f"  {k:20} {v}")
    spend = db.meta_get("llm_spend_usd_total")
    if spend:
        print(f"  {'llm_spend_total':20} ${float(spend):.2f}")
    reasons = db.filtered_stats()
    if reasons:
        print("  filtered out (open postings):")
        for k, v in reasons.items():
            print(f"    {v:6}  {k}")


def _cmd_show(args) -> None:
    statuses = [args.status] if args.status else None
    rows = db.board_rows(statuses=statuses)
    rows = [r for r in rows if r["fit_score"] is None or r["fit_score"] >= config.SHORTLIST_MIN_SCORE]
    for r in rows[: args.limit]:
        print(f"\n[{r['fit_score']:.0f}] {r['title']} — {r['company']} ({r['location']})")
        print(f"   posted {r['posted_date']} | status {r['status']} | {r['url']}")
        print(f"   why: {r['rationale']}")
        if r["watch_outs"] and r["watch_outs"].lower() not in ("none", "none."):
            print(f"   watch: {r['watch_outs']}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="jobscout", description="CV-driven job recommender (POC)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup", help="one-time: load CV + companies, optionally detect")
    s.add_argument("--cv", help="path to your CV (.docx or .txt)")
    s.add_argument("--excel", help="path to the companies workbook (.xlsx)")
    s.add_argument("--yellow", action="store_true",
                   help="seed only the yellow-highlighted target shortlist")
    s.add_argument("--detect", action="store_true", help="also run ATS detection")
    s.add_argument("--limit", type=int, default=None)
    s.set_defaults(func=_cmd_setup)

    d = sub.add_parser("detect", help="classify which ATS each company uses (once)")
    d.add_argument("--force", action="store_true")
    d.add_argument("--limit", type=int, default=None)
    d.add_argument("--known-only", action="store_true",
                   help="use only the curated KNOWN_TOKENS map (fast, no slug-guessing)")
    d.add_argument("--no-resolver", action="store_true",
                   help="skip the Tier-2 careers-page resolver (slug auto-detect only)")
    d.set_defaults(func=_cmd_detect)

    r = sub.add_parser("run", help="fetch + dedupe + match (the daily loop)")
    r.add_argument("--detect", action="store_true", help="detect pending companies first")
    r.add_argument("--tiered", action="store_true",
                   help="full tiered build: feeds first (scored), then browser tier")
    r.add_argument("--skip-browser", action="store_true",
                   help="fetch feed tiers only (skip the slower browser tier)")
    r.add_argument("--only", default="",
                   help="run the pipeline for these companies only "
                        "(comma-separated names, e.g. --only 'BMW,Apple'); "
                        "default: all companies")
    r.set_defaults(func=_cmd_run)

    rs = sub.add_parser("rescore", help="re-apply hard filters to the board; "
                                        "--llm also re-scores survivors with Claude")
    rs.add_argument("--llm", action="store_true", help="re-score surviving New "
                                                       "recommendations with the LLM")
    rs.add_argument("--heuristic-only", action="store_true",
                    help="only re-score recommendations that never got a real "
                         "LLM judgment (llm_used=0)")
    rs.add_argument("--limit", type=int, default=None, help="max postings to re-score")
    rs.add_argument("--budget", type=float, default=None,
                    help=f"USD cap for the re-score (default {config.MAX_RUN_LLM_USD})")
    rs.set_defaults(func=_cmd_rescore)

    va = sub.add_parser("validate", help="run only the final LLM review over "
                                         "the current board")
    va.add_argument("--all", action="store_true",
                    help="re-validate every New rec (default: only unvalidated)")
    va.add_argument("--budget", type=float, default=None,
                    help=f"USD cap (default {config.MAX_RUN_LLM_USD})")
    va.set_defaults(func=_cmd_validate)

    lc = sub.add_parser("linkcheck", help="verify the job links behind the "
                                          "board's recommendations; archive dead ones")
    lc.add_argument("--force", action="store_true",
                    help="re-check every link (ignore the recheck TTL)")
    lc.set_defaults(func=_cmd_linkcheck)

    vf = sub.add_parser("verify", help="verify scraper completeness against "
                                       "each source's own job counts")
    vf.add_argument("--type", help="only this ats_type (e.g. workday)")
    vf.add_argument("--company", help="only companies whose name contains this")
    vf.add_argument("--browser", action="store_true",
                    help="include browser-tier companies (slow)")
    vf.add_argument("--csv", help="write the full report to this CSV path")
    vf.set_defaults(func=_cmd_verify)

    st = sub.add_parser("stats", help="show DB counts")
    st.set_defaults(func=_cmd_stats)

    rst = sub.add_parser("reset", help="wipe the board (keep chosen decisions) "
                                       "for a from-scratch rebuild")
    rst.add_argument("--keep", default="Applied,Marked",
                     help="comma-separated statuses to keep (default: Applied,Marked)")
    rst.add_argument("--yes", action="store_true",
                     help="actually execute (default is a dry run)")
    rst.set_defaults(func=_cmd_reset)

    sh = sub.add_parser("show", help="print the current shortlist")
    sh.add_argument("--status", choices=list(config.ALL_STATUSES))
    sh.add_argument("--limit", type=int, default=20)
    sh.set_defaults(func=_cmd_show)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

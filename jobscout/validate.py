"""The final quality gate — an independent Claude review of the finished board.

After scraping, filtering, dedup and scoring have all run, one question
remains: "would a careful human assistant actually show this list to the
candidate?" This module asks exactly that. A second, independent LLM pass
reviews every remaining New recommendation in the shortlist band
(fit >= VALIDATION_MIN_SCORE — the board's deliberate low-score long tail is
exempt, being already labelled weak by its own score) against the candidate's
profile, preferences and observed decision history, and removes the rows that
are clearly wrong for them — the false positives that survive keyword filters and
even scoring (a well-scored posting can still be a mismatch the moment a human
reads it in context).

Design principles, in order:

  1. ASYMMETRIC CAUTION. A wrong removal costs the candidate a real
     opportunity; a wrong keep costs them three seconds. The reviewer may only
     remove what is CLEARLY incompatible, must keep anything plausible, and a
     low-confidence removal is treated as a keep by the calling code — the
     model's uncertainty is never allowed to delete a job.
  2. EVERY DECISION IS RECORDED. Removed postings carry the reviewer's reason
     in `posting.filtered_reason` ("failed final review: …"); kept recs carry
     `validated_at` (+ a note when the reviewer flagged something), so every
     verdict is auditable and reproducible.
  3. INCREMENTAL AND BOUNDED. Each rec is validated once and re-validated only
     after a re-score. Batches share the run's SpendTracker budget. An LLM
     failure changes nothing (affected recs stay unvalidated and are retried
     next run), and a runaway pass that tries to remove more than
     VALIDATION_MAX_REMOVAL_RATIO of the board is discarded wholesale.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from . import config, db, feedback, matching

_REVIEW_RULES = """You are the FINAL QUALITY GATE of JobScout, a personal job \
recommender. Earlier stages already filtered and scored these postings; they \
are about to be shown to the candidate. Your ONLY task is to catch clear \
mistakes — postings a careful human assistant would be embarrassed to show \
this candidate because they are plainly wrong for them.

For each posting, verdict "remove" or "keep":

REMOVE only when the posting is CLEARLY incompatible with the profile:
- a different occupation altogether: skilled trade, manufacturing/warehouse \
floor, driver, medical care, legal practice, tax administration, hospitality, \
in-store retail, security guard, cabin crew, teaching …
- a hard disqualifier in the text: daily-work language the candidate does not \
speak (anything beyond English/German, unless an or-alternative allows EN/DE), \
5+ years of experience explicitly required, senior leadership scope \
(department/team ownership), internship/apprenticeship/thesis role, location \
outside the target region (see preferences; Spain = Barcelona ONLY)
- a dead-centre example of a pattern this candidate has skipped again and \
again (see LEARNED SKIP HISTORY) with no analytics substance to redeem it.

KEEP everything else, explicitly including:
- imperfect but plausible matches (adjacent functions with real analytics \
content, mild seniority stretch, unusual titles)
- analytics-adjacent work: reporting, insights, market/business intelligence, \
pricing, forecasting, commercial/product/customer/operations analytics, \
data-heavy project roles at international organisations
- scorer-capped BORDERLINE rows (fit 45-55, e.g. finance/consulting/\
governance functions): the scorer deliberately surfaces those as borderline. \
Being a non-analytics function is NOT grounds for removal on its own — remove \
only on the hard disqualifiers above or zero analytics substance.
- anything you are UNSURE about. When in doubt, ALWAYS keep — a wrong removal \
destroys a real opportunity, a wrong keep costs seconds.

RULES:
1. Judge the ROLE from title + description. The fit score and the previous \
stage's rationale are context, not evidence; do not remove merely because the \
score is low.
2. Skip history means heightened scrutiny, not auto-removal. The candidate \
still applies to SOME roles in skipped categories (see ENGAGED list).
3. Never invent facts. If the description does not state a disqualifier, it \
does not exist.
4. Identical input must yield identical verdicts. Be decisive and terse.

OUTPUT — a JSON array ONLY, one object per posting, same order:
[{"id":<id>,"v":"keep"|"remove","why":"<concrete reason, <=15 words>",\
"conf":"hi"|"lo"}]
"conf" is YOUR certainty in the verdict. A "remove" you cannot back with a \
quoted disqualifier or an obvious occupation mismatch must be conf "lo".
No text before or after the JSON array. No markdown fences."""


@dataclass
class ValidationOutcome:
    reviewed: int = 0
    removed: int = 0
    kept: int = 0
    flagged: int = 0          # kept, but with a reviewer note
    pending: int = 0          # not reviewed (LLM failure / budget)
    aborted: bool = False     # removal-ratio guard tripped — nothing applied
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"reviewed": self.reviewed, "removed": self.removed,
                "kept": self.kept, "flagged": self.flagged,
                "pending": self.pending, "aborted": self.aborted,
                "notes": self.notes[:8]}


def _engaged_titles(limit: int = 15) -> list[str]:
    """Titles the user actually pursued (Applied/Marked) — positive anchors."""
    rows = db.board_rows(statuses=["Applied", "Marked"])
    out, seen = [], set()
    for r in rows:
        t = (r["title"] or "").strip()
        key = t.lower()
        if t and key not in seen:
            seen.add(key)
            out.append(t)
        if len(out) >= limit:
            break
    return out


def _system_blocks(cv: str, prefs: str, skip_rules: str) -> list[dict]:
    profile = (f"CANDIDATE CV:\n{(cv or '')[:6000]}\n\n"
               f"PREFERENCES NOTE (authoritative — overrides the CV where they "
               f"conflict):\n{(prefs or '')[:2000]}")
    engaged = _engaged_titles()
    if engaged:
        profile += ("\n\nENGAGED — roles this candidate applied to or "
                    "bookmarked (their revealed taste; keep lookalikes):\n- "
                    + "\n- ".join(engaged))
    if skip_rules:
        profile += f"\n\n{skip_rules[:3000]}"
    return [
        {"type": "text", "text": _REVIEW_RULES},
        {"type": "text", "text": profile, "cache_control": {"type": "ephemeral"}},
    ]


def _payload(r) -> dict:
    p = {
        "id": r["rec_id"],
        "title": (r["title"] or "")[:200],
        "company": (r["company"] or "")[:80],
        "location": (r["location"] or "")[:120],
        "fit_score": r["fit_score"],
        "description": matching.trim_description(r["description"] or "", 1800),
    }
    why = (r["rationale"] or "").strip()
    if why:
        p["scorer_said"] = re.sub(r"\s+", " ", why)[:200]
    watch = (r["watch_outs"] or "").strip()
    if watch and watch.lower() not in ("none", "none."):
        p["watch"] = re.sub(r"\s+", " ", watch)[:150]
    return p


def _review_batch(system_blocks: list[dict], batch: list,
                  tracker) -> dict[int, dict]:
    """One validator call → {rec_id: {"v":…, "why":…, "conf":…}}."""
    payload = [_payload(r) for r in batch]
    user_text = ("POSTINGS ABOUT TO BE SHOWN TO THE CANDIDATE:\n"
                 + json.dumps(payload, ensure_ascii=False)
                 + "\n\nReturn the JSON array of verdicts now.")
    text = matching._llm_call(system_blocks, user_text,
                              max_tokens=120 * len(batch) + 300, tracker=tracker)
    if not text:
        return {}
    data = matching._parse_llm_array(text)
    if not data:
        return {}
    valid_ids = {r["rec_id"] for r in batch}
    out: dict[int, dict] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            rid = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        if rid not in valid_ids:
            continue
        v = str(item.get("v", "")).strip().lower()
        if v not in ("keep", "remove"):
            continue
        out[rid] = {
            "v": v,
            "why": str(item.get("why", "")).strip()[:200],
            "conf": str(item.get("conf", "lo")).strip().lower(),
        }
    return out


def validate_new_recommendations(tracker=None, verbose: bool = False,
                                 revalidate_all: bool = False,
                                 progress=None,
                                 company_ids=None) -> ValidationOutcome:
    """Run the final review over the current board. Returns the outcome.

    Verdicts are collected for ALL pending recs first and applied only after
    the removal-ratio guard passes, so a haywire model can never wipe a board
    batch by batch. `company_ids` scopes the review (partial pipeline runs)."""
    out = ValidationOutcome()
    rows = db.recs_to_validate(revalidate_all=revalidate_all,
                               company_ids=company_ids)
    if not rows:
        return out
    if not (config.VALIDATION_ENABLED and matching.llm_available()):
        out.pending = len(rows)
        if config.VALIDATION_ENABLED and not matching.llm_available():
            out.notes.append("validation skipped: no API key")
        return out
    if matching.llm_fatal_reason:
        # A permanent API failure was already diagnosed. Every batch would
        # come back empty; nothing would be removed (the gate only acts on an
        # explicit "remove"), but the run would still walk every batch.
        out.pending = len(rows)
        out.notes.append(
            f"validation skipped: {matching.llm_fatal_reason}")
        return out
    if tracker is None:
        tracker = matching.SpendTracker()

    profile = db.get_profile()
    cv = profile["cv_text"] if profile else ""
    prefs = profile["preferences_note"] if profile else ""
    skip_cats, applied_cats = feedback.load_profiles(db)
    system_blocks = _system_blocks(
        cv, prefs, feedback.prompt_section(skip_cats, applied_cats))

    if progress is not None:
        progress.phase("validate",
                       f"Final review of {len(rows)} recommendations",
                       total=len(rows))

    verdicts: dict[int, dict] = {}
    batch_size = max(1, config.VALIDATE_BATCH_SIZE)
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        if tracker.exhausted:
            out.notes.append(
                f"validation stopped at budget cap (${tracker.cap:.2f}) — "
                f"{len(rows) - i} recs stay queued for the next run")
            break
        if progress is not None:
            progress.step(0, current=f"{batch[0]['company']} — {batch[0]['title']}")
        verdicts.update(_review_batch(system_blocks, batch, tracker))
        if progress is not None:
            progress.step(len(batch))
            progress.set(llm_cost=tracker.spent)

    reviewed = [r for r in rows if r["rec_id"] in verdicts]
    out.reviewed = len(reviewed)
    out.pending = len(rows) - len(reviewed)

    # A "remove" only counts when the reviewer is confident; hesitation keeps.
    removals = [r for r in reviewed
                if verdicts[r["rec_id"]]["v"] == "remove"
                and verdicts[r["rec_id"]]["conf"] == "hi"]

    # Safety valve: the reviewer is a janitor, not a judge — if it tries to
    # clear out most of the board, distrust the whole pass.
    if (len(reviewed) >= 8
            and len(removals) > config.VALIDATION_MAX_REMOVAL_RATIO * len(reviewed)):
        out.aborted = True
        out.notes.append(
            f"validation DISCARDED: wanted to remove {len(removals)}/"
            f"{len(reviewed)} — above the {config.VALIDATION_MAX_REMOVAL_RATIO:.0%} "
            "safety ratio; board left untouched")
        return out

    removal_ids = {r["rec_id"] for r in removals}
    for r in reviewed:
        v = verdicts[r["rec_id"]]
        if r["rec_id"] in removal_ids:
            db.delete_recommendation(r["rec_id"])
            db.mark_posting_filtered(
                r["posting_id"], f"failed final review: {v['why'] or 'clear mismatch'}")
            out.removed += 1
            if verbose:
                print(f"    ✗ removed [{(r['fit_score'] or 0):.0f}] "
                      f"{r['company']} — {r['title'][:60]}  ({v['why']})")
        else:
            note = None
            if v["v"] == "remove":              # low-confidence remove -> keep
                note = f"reviewer unsure: {v['why']}" if v["why"] else "reviewer unsure"
                out.flagged += 1
            db.set_validation(r["rec_id"], note)
            out.kept += 1
    if out.removed:
        out.notes.append(f"final review removed {out.removed} of "
                         f"{out.reviewed} reviewed recommendations")
    return out

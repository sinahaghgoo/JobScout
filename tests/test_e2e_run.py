"""Capstone integration test: one pipeline.run() through every stage.

fetch → store → hard filters → prefilter → LLM scoring (stubbed) → floor →
final validation gate (stubbed) → board. Proves the stages interlock: garbage
dies at the right layer, good roles survive every layer, and each casualty
carries an auditable reason.
"""
import os
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

os.environ["JOBSCOUT_DB"] = str(Path(tempfile.gettempdir()) / "jobscout_pytest_e2e.db")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobscout import config, db, matching, pipeline, validate  # noqa: E402
from jobscout.connectors.base import RawJob  # noqa: E402

TODAY = date.today().isoformat()
OLD = (date.today() - timedelta(days=60)).isoformat()

BOARD = [
    # survives everything -> board
    RawJob("j-good", "Junior BI Analyst", "Vienna, Austria", TODAY, "http://x/1",
           "Build Power BI dashboards, SQL, KPI reporting. 0-2 years."),
    # scores well but the validator spots a hard mismatch -> removed by review
    RawJob("j-sneaky", "Business Data Coordinator", "Berlin, Germany", TODAY,
           "http://x/2", "Coordinate paper archive logistics; occasional Excel."),
    # LLM judges clearly off-profile -> floor archives it, no board row
    RawJob("j-lowfit", "Court Records Assistant", "The Hague, Netherlands", TODAY,
           "http://x/3", "Maintain court case files and hearing schedules. Excel."),
    # hard filters: occupation / age / seniority / internship / location
    RawJob("j-mech", "Kfz-Mechatroniker (w/m/x)", "Hamburg, Germany", TODAY,
           "http://x/4", "Wartung und Instandsetzung."),
    RawJob("j-old", "Reporting Analyst", "Vienna, Austria", OLD, "http://x/5",
           "SQL reporting."),
    RawJob("j-senior", "Werkstattleiter (w/m/x)", "Berlin, Germany", TODAY,
           "http://x/6", "Leitung der Werkstatt."),
    RawJob("j-intern", "Praktikum Data Analytics", "Munich, Germany", TODAY,
           "http://x/7", "Praktikum im Analytics Team."),
    RawJob("j-usa", "Data Analyst", "Austin, Texas, USA", TODAY, "http://x/8",
           "SQL dashboards."),
]

_SCORES = {"Junior BI Analyst": 88.0, "Business Data Coordinator": 52.0,
           "Court Records Assistant": 12.0}


def test_full_run_every_stage(monkeypatch):
    p = Path(os.environ["JOBSCOUT_DB"])
    if p.exists():
        p.unlink()
    db.init_db()
    db.save_profile("MSc data science, BI/analytics focus", "Business/BI/Data Analyst roles")
    cid = db.upsert_company("E2ECo")
    db.set_company_ats(cid, "greenhouse", "e2eco", "detected")

    monkeypatch.setattr(pipeline, "_fetch_one", lambda c: list(BOARD))
    monkeypatch.setattr(matching, "llm_available", lambda: True)

    def fake_score_batch(cv, prefs, jobs, tracker=None, extra_rules=""):
        out = {}
        for j in jobs:
            fit = _SCORES.get(j["title"])
            if fit is not None:
                out[j["id"]] = matching.MatchResult(
                    prefilter_score=j["pre_score"], fit_score=fit,
                    rationale="stub", watch_outs="None", llm_used=True)
        return out
    monkeypatch.setattr(matching, "score_batch", fake_score_batch)

    def fake_review(system_blocks, batch, tracker):
        return {r["rec_id"]: (
            {"v": "remove", "why": "archive logistics, not analytics", "conf": "hi"}
            if r["title"] == "Business Data Coordinator"
            else {"v": "keep", "why": "", "conf": "hi"}) for r in batch}
    monkeypatch.setattr(validate, "_review_batch", fake_review)

    summary = pipeline.run(verbose=False)

    # --- the board: exactly the one genuinely good role, validated
    rows = db.board_rows(statuses=["New"])
    assert [r["title"] for r in rows] == ["Junior BI Analyst"]
    assert rows[0]["validated_at"]

    # --- every casualty died at the right layer, with the right reason
    with db.connect() as conn:
        reasons = {r["job_id"]: r["filtered_reason"] for r in conn.execute(
            "SELECT job_id, filtered_reason FROM posting")}
    assert reasons["j-mech"] == "off-profile occupation"
    assert reasons["j-old"].startswith("older than")
    assert reasons["j-senior"] == "senior-level title"
    assert reasons["j-intern"] == "internship/student/thesis role"
    assert reasons["j-usa"] == "outside target countries"
    assert reasons["j-lowfit"].startswith("scored 12/100 by the LLM")
    assert reasons["j-sneaky"].startswith("failed final review:")
    assert reasons["j-good"] is None

    # --- the summary tells the same story
    assert summary.postings_new == len(BOARD)
    assert summary.scored == 2                  # good + sneaky got board rows
    assert summary.floor_rejected == 1          # lowfit archived pre-board
    assert summary.validated == 2
    assert summary.validation_removed == 1
    d = summary.as_dict()
    assert d["validation_removed"] == 1 and d["floor_rejected"] == 1

"""What happens when the Anthropic API is permanently unusable.

The failure this pins down, in the user's words: "I click Run pipeline and no
jobs appear on the board." The account had run out of credit, so every scoring
call 400'd. The run still walked every batch, still reported "complete", and
the dashboard still showed a green summary — while `refilter_existing` kept
retiring aged-out recommendations, so the board drained to zero with nothing
anywhere saying why.

The contract now:
  * a permanent failure (no credit / bad key / forbidden model) is diagnosed
    ONCE and short-circuits every later call — no hundreds of retries;
  * it is never mistaken for a scraping problem (browser-tier companies are
    read by Claude, so they fail with the API's reason, aggregated to one note);
  * postings stay QUEUED — a broken key must never fall back to heuristic
    scores wearing an LLM badge;
  * the reason reaches the run summary, the DB, and /api/stats, so the
    dashboard can say it instead of "Run complete".
"""
import os
import sys
import tempfile
from datetime import date
from pathlib import Path

os.environ["JOBSCOUT_DB"] = str(Path(tempfile.gettempdir()) / "jobscout_pytest_outage.db")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from jobscout import browser, config, db, matching, pipeline, validate, web  # noqa: E402
from jobscout.connectors.base import RawJob  # noqa: E402

TODAY = date.today().isoformat()

CREDIT_MSG = ("Error code: 400 - {'type': 'error', 'error': {'type': "
              "'invalid_request_error', 'message': 'Your credit balance is too "
              "low to access the Anthropic API. Please go to Plans & Billing to "
              "upgrade or purchase credits.'}, 'request_id': 'req_011Cec'}")


import anthropic  # noqa: E402


class _Status(anthropic.APIStatusError):
    """A real APIStatusError without the httpx.Response the SDK wants.

    Subclassing matters: `_llm_call` branches on the SDK's own exception
    types, so a plain Exception would exercise a different path than a live
    API failure does.
    """

    def __init__(self, msg, status_code=400):
        Exception.__init__(self, msg)
        self.status_code = status_code
        self.message = msg


def _fresh_db():
    p = Path(os.environ["JOBSCOUT_DB"])
    if p.exists():
        p.unlink()
    db.init_db()


# --- 1. classification --------------------------------------------------------
def test_classify_fatal_recognises_an_empty_balance():
    assert "out of credit" in matching.classify_fatal(_Status(CREDIT_MSG))


@pytest.mark.parametrize("exc,expect", [
    (_Status("invalid x-api-key", 401), "invalid or revoked"),
    (_Status("permission_error: not allowed", 403), "not permitted"),
])
def test_classify_fatal_recognises_key_problems(exc, expect):
    assert expect in matching.classify_fatal(exc)


@pytest.mark.parametrize("exc", [
    _Status("overloaded_error", 529),
    _Status("internal server error", 500),
    _Status("rate limited", 429),
    ValueError("could not parse the model's JSON"),
    _Status("temperature: unexpected parameter", 400),
])
def test_transient_failures_are_not_fatal(exc):
    """Only conditions a retry can never fix may open the circuit — a 400 that
    is merely a bad parameter must stay retryable/local."""
    assert matching.classify_fatal(exc) == ""


# --- 2. the circuit breaker ---------------------------------------------------
def test_one_fatal_error_stops_every_later_call(monkeypatch):
    """284 identical errors in one run is the symptom this removes."""
    calls = {"n": 0}

    class _Client:
        class messages:
            @staticmethod
            def create(**kw):
                calls["n"] += 1
                raise _Status(CREDIT_MSG)

    monkeypatch.setattr(matching, "_get_client", lambda: _Client())
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-test")

    assert matching._llm_call([], "x", max_tokens=10, tracker=None) is None
    first = calls["n"]
    assert first == 1, "a permanent 400 must not burn the 3-attempt retry loop"
    assert matching.llm_fatal_reason

    for _ in range(5):
        assert matching._llm_call([], "x", max_tokens=10, tracker=None) is None
    assert calls["n"] == first, "the circuit stayed open — no further API calls"


def test_a_non_sdk_exception_is_still_classified(monkeypatch):
    """The SDK is not the only thing that can raise — a wrapped/re-raised
    billing error must open the circuit just the same."""
    class _Client:
        class messages:
            @staticmethod
            def create(**kw):
                raise RuntimeError(CREDIT_MSG)

    monkeypatch.setattr(matching, "_get_client", lambda: _Client())
    assert matching._llm_call([], "x", max_tokens=10, tracker=None) is None
    assert "out of credit" in matching.llm_fatal_reason


def test_a_transient_error_still_retries(monkeypatch):
    calls = {"n": 0}

    class _Client:
        class messages:
            @staticmethod
            def create(**kw):
                calls["n"] += 1
                raise _Status("internal server error", 500)

    monkeypatch.setattr(matching, "_get_client", lambda: _Client())
    monkeypatch.setattr(matching.time, "sleep", lambda *_: None)
    assert matching._llm_call([], "x", max_tokens=10, tracker=None) is None
    assert calls["n"] == 3 and matching.llm_fatal_reason == ""


def test_llm_available_stays_true_so_a_dead_key_never_fakes_scores():
    """`llm_available` means "LLM mode", not "API is up" — otherwise a broken
    key would silently drop the run into keyword scoring."""
    matching.note_fatal("out of credit")
    if config.ANTHROPIC_API_KEY:
        assert matching.llm_available() is True
    assert matching.llm_healthy() is False


class _ProbeClient:
    """Mimics the SDK's client closely enough for preflight: `with_options`
    returns a configured client, and the probe raises `exc`."""

    def __init__(self, exc):
        self.exc = exc
        self.options = None

    def with_options(self, **kw):
        self.options = kw
        return self

    @property
    def messages(self):
        outer = self

        class _M:
            @staticmethod
            def create(**kw):
                raise outer.exc
        return _M


def test_preflight_arms_the_breaker(monkeypatch):
    client = _ProbeClient(_Status(CREDIT_MSG))
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(matching, "_get_client", lambda: client)
    assert "out of credit" in matching.preflight()
    assert matching.llm_fatal_reason


def test_preflight_probe_cannot_hang_the_run(monkeypatch):
    """The SDK default is a 600 s read timeout with 2 retries — a half-open
    proxy would stall the whole run on "Checking the Anthropic API…"."""
    client = _ProbeClient(_Status("gateway timeout", 504))
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(matching, "_get_client", lambda: client)
    matching.preflight(timeout=7.5)
    assert client.options == {"timeout": 7.5, "max_retries": 0}


def test_preflight_ignores_a_flaky_network(monkeypatch):
    client = _ProbeClient(_Status("gateway timeout", 504))
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(matching, "_get_client", lambda: client)
    assert matching.preflight() == ""
    assert matching.llm_fatal_reason == ""


# --- what must NOT be called fatal (review findings, 2026-09-01) -------------
def test_a_rate_limit_mentioning_billing_is_not_fatal():
    """Anthropic returns 429 for spend limits, and its message says "Plans &
    Billing". Reading that as "out of credit" would abort a funded run."""
    exc = _Status("This request would exceed your organization's spend limit. "
                  "Go to Plans & Billing to raise it.", 429)
    assert matching.classify_fatal(exc) == ""


def test_a_proxy_401_or_403_is_not_a_dead_key():
    """Captive portals and TLS-inspecting proxies answer 401/403 with HTML.
    Blaming key.txt for that is both wrong and run-ending."""
    assert matching.classify_fatal(
        _Status("<html>403 Forbidden</html><hr>nginx", 403)) == ""
    assert matching.classify_fatal(
        _Status("<html>Proxy Authentication Required</html>", 401)) == ""
    # a real Anthropic 401 still is
    assert "invalid or revoked" in matching.classify_fatal(
        _Status('{"type":"error","error":{"type":"authentication_error",'
                '"message":"invalid x-api-key"}}', 401))


def test_an_identity_linked_key_is_diagnosed_precisely():
    """The exact 400 the user's replacement key returned."""
    exc = _Status("anthropic-workspace-id is required when authenticating "
                  "with an identity-linked API key", 400)
    r = matching.classify_fatal(exc)
    assert "identity-linked" in r and "workspace API key" in r


def test_a_model_error_names_the_model_that_failed():
    """Page extraction runs on the cheap model; blaming the scorer's model
    sends the user looking in the wrong place."""
    exc = _Status("model: claude-haiku-4-5 not found", 404)
    exc.__class__ = type("NotFoundError", (type(exc),), {})
    r = matching.classify_fatal(exc, model="claude-haiku-4-5")
    assert "claude-haiku-4-5" in r and config.ANTHROPIC_MODEL not in r
    assert not matching.account_level_fatal(r), "one model != the whole account"


def test_repeated_unknown_failures_still_stop_the_run(monkeypatch):
    """Backstop for an error shape classify_fatal has never seen — the failure
    mode this whole change set exists to prevent must not depend on the
    classifier recognising the message."""
    calls = {"n": 0}

    class _Client:
        class messages:
            @staticmethod
            def create(**kw):
                calls["n"] += 1
                raise _Status("something entirely new went wrong", 400)

    monkeypatch.setattr(matching, "_get_client", lambda: _Client())
    monkeypatch.setattr(matching.time, "sleep", lambda *_: None)
    for _ in range(matching.CONSECUTIVE_FAIL_LIMIT):
        matching._llm_call([], "x", max_tokens=10, tracker=None)
    assert matching.llm_fatal_reason.startswith(
        f"{matching.CONSECUTIVE_FAIL_LIMIT} Anthropic API calls failed in a row")
    before = calls["n"]
    matching._llm_call([], "x", max_tokens=10, tracker=None)
    assert calls["n"] == before, "the circuit is open"


def test_a_success_resets_the_failure_streak(monkeypatch):
    """Intermittent failures on a working API must never trip the backstop."""
    state = {"i": 0}

    class _Resp:
        content = []
        usage = None

    class _Client:
        class messages:
            @staticmethod
            def create(**kw):
                state["i"] += 1
                if state["i"] % 2:
                    raise _Status("transient weirdness", 400)
                return _Resp()

    monkeypatch.setattr(matching, "_get_client", lambda: _Client())
    monkeypatch.setattr(matching.time, "sleep", lambda *_: None)
    for _ in range(matching.CONSECUTIVE_FAIL_LIMIT * 3):
        matching._llm_call([], "x", max_tokens=10, tracker=None)
    assert matching.llm_fatal_reason == ""


# --- 3. browser tier ----------------------------------------------------------
def test_browser_extraction_reports_the_api_not_the_page(monkeypatch):
    """Browser-tier pages are READ by Claude. An API outage there is not a
    scraping failure and must not be labelled as one."""
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-test")
    matching.note_fatal("the Anthropic account is out of credit")
    with pytest.raises(matching.LLMUnavailable) as e:
        browser.extract_jobs("<html><a href='/jobs/1'>Data Analyst</a></html>",
                             "https://x.example")
    assert "out of credit" in str(e.value)


def test_browser_extraction_arms_the_breaker_itself(monkeypatch):
    """The browser tier usually runs before scoring, so it is where the outage
    is discovered — it has to open the circuit for the rest of the run."""
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-test")

    class _Anthropic:
        def __init__(self, **kw):
            pass

        class messages:
            @staticmethod
            def create(**kw):
                raise _Status(CREDIT_MSG)

    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", _Anthropic)
    html = "<html><body><a href='/jobs/1'>Data Analyst — Vienna</a></body></html>"
    with pytest.raises(matching.LLMUnavailable):
        browser.extract_jobs(html, "https://x.example")
    assert "out of credit" in matching.llm_fatal_reason


# --- 4. the run summary -------------------------------------------------------
def test_fetch_errors_are_short_and_llm_failures_collapse_to_one_note():
    """Raw SDK errors used to be written verbatim into the company health
    column and once per company into the notes."""
    _fresh_db()
    summary = pipeline.RunSummary()
    matching.note_fatal("the Anthropic account is out of credit")
    for i in range(3):
        cid = db.upsert_company(f"BrowserCo{i}")
        db.set_company_ats(cid, "browser", f"https://x{i}.example", "detected")
    feed_id = db.upsert_company("FeedCo")
    db.set_company_ats(feed_id, "greenhouse", "feedco", "detected")
    companies = db.get_companies(only_detected=True)

    def _fetch(c):
        if c["ats_type"] == "browser":
            raise matching.LLMUnavailable("page extraction needs the Anthropic API")
        raise RuntimeError("boom " + "x" * 500)

    orig = pipeline._fetch_one
    pipeline._fetch_one = _fetch
    try:
        pipeline._fetch_and_store(summary, False, companies)
    finally:
        pipeline._fetch_one = orig

    assert summary.fetch_errors == 4
    assert summary.llm_fatal, "the run knows scoring is impossible"
    llm_notes = [n for n in summary.notes if "browser-tier companies" in n]
    assert len(llm_notes) == 1 and "3 browser-tier" in llm_notes[0]
    assert not any("fetch failed for BrowserCo" in n for n in summary.notes)
    feed_note = [n for n in summary.notes if "fetch failed for FeedCo" in n]
    assert len(feed_note) == 1 and len(feed_note[0]) < 220, "error text is trimmed"
    row = [c for c in db.get_companies() if c["id"] == feed_id][0]
    assert len(row["last_fetch_status"]) < 200


def test_short_error_trims_a_wall_of_json():
    long = pipeline._short_error(RuntimeError(CREDIT_MSG * 4))
    assert len(long) <= 140 and long.endswith("…")
    assert pipeline._short_error(RuntimeError("nope")) == "nope"


def test_finish_deduplicates_repeated_phase_notes():
    """run_tiered runs the scoring phase twice, so its notes arrive twice —
    and the summary only shows 12."""
    _fresh_db()
    s = pipeline.RunSummary()
    s.notes = ["957 postings could not be scored", "queue ageing skipped",
               "957 postings could not be scored", "queue ageing skipped"]
    pipeline._finish(s)
    assert s.notes == ["957 postings could not be scored", "queue ageing skipped"]


def test_summary_carries_the_reason_to_the_dashboard():
    s = pipeline.RunSummary()
    s.llm_fatal = "the Anthropic account is out of credit"
    assert s.as_dict()["llm_fatal"] == s.llm_fatal


def test_finish_puts_the_reason_first_even_if_found_mid_run():
    _fresh_db()
    s = pipeline.RunSummary()
    s.notes.append("some other note")
    matching.note_fatal("the Anthropic API key is invalid or revoked")
    pipeline._finish(s)
    assert s.llm_fatal
    assert s.notes[0].startswith("Anthropic API unavailable")
    stored = db.meta_get("last_run_summary")
    assert "llm_fatal" in stored and "invalid or revoked" in stored


# --- 5. the whole run ---------------------------------------------------------
JOBS = [
    RawJob("o-1", "Junior BI Analyst", "Vienna, Austria", TODAY,
           "https://x.example/1", "SQL dashboards, Power BI reporting."),
    RawJob("o-2", "Reporting Analyst", "Dublin, Ireland", TODAY,
           "https://x.example/2", "KPI reporting, Excel, SQL."),
]


def _seed_run_db():
    _fresh_db()
    db.save_profile("MSc data science, BI focus", "Analyst roles")
    cid = db.upsert_company("OutageCo")
    db.set_company_ats(cid, "greenhouse", "outageco", "detected")
    return cid


def test_a_dead_api_leaves_the_queue_intact_and_says_so(monkeypatch):
    """The end-to-end contract: no board rows, no fake scores, no silent
    success — and the postings are still there for the next run."""
    _seed_run_db()
    monkeypatch.setattr(pipeline, "_fetch_one", lambda c: list(JOBS))
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(config, "LINKCHECK_ENABLED", False)
    monkeypatch.setattr(matching, "llm_available", lambda: True)

    calls = {"n": 0}

    def dead_score(cv, prefs, jobs, tracker=None, extra_rules=""):
        calls["n"] += 1
        matching.note_fatal("the Anthropic account is out of credit")
        return {}
    monkeypatch.setattr(matching, "score_batch", dead_score)

    summary = pipeline.run(verbose=False)

    assert db.board_rows(statuses=["New"]) == [], "no board rows were invented"
    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM recommendation").fetchone()[0] == 0
        queued = conn.execute(
            "SELECT COUNT(*) FROM posting WHERE is_open=1 AND filtered_reason IS NULL"
        ).fetchone()[0]
    assert queued == 2, "both postings stay queued for the next run"
    assert summary.scored == 0 and summary.surfaced == 0
    assert "out of credit" in summary.llm_fatal
    assert summary.notes[0].startswith("Anthropic API unavailable")
    assert any("could not be scored" in n for n in summary.notes)
    assert calls["n"] == 1, "one batch discovered it; the rest short-circuited"


def test_a_dead_api_does_not_age_out_the_backlog(monkeypatch):
    """The queue janitor retires postings that went unscored for 10 days —
    which is right when the retries happened and wrong when the API was down.
    Live on 2026-09-01 it discarded 354 postings during an outage."""
    _seed_run_db()
    cid = db.get_companies()[0]["id"]
    old = (date.today() - __import__("datetime").timedelta(days=40)).isoformat()
    pid, _ = db.upsert_posting(company_id=cid, job_id="stale-1",
                               title="Reporting Analyst",
                               location="Vienna, Austria", posted_date=TODAY,
                               url="https://x.example/s1", description="SQL KPIs")
    with db.connect() as conn:               # backdate first_seen past the limit
        conn.execute("UPDATE posting SET first_seen=? WHERE id=?", (old, pid))
        conn.commit()

    monkeypatch.setattr(pipeline, "_fetch_one", lambda c: [])
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(config, "LINKCHECK_ENABLED", False)
    monkeypatch.setattr(matching, "llm_available", lambda: True)
    monkeypatch.setattr(matching, "score_batch",
                        lambda *a, **k: matching.note_fatal(
                            "the Anthropic account is out of credit") or {})

    summary = pipeline.run(verbose=False)
    with db.connect() as conn:
        row = conn.execute("SELECT filtered_reason FROM posting WHERE id=?",
                           (pid,)).fetchone()
    assert row["filtered_reason"] is None, "the backlog survived the outage"
    assert any("queue ageing skipped" in n for n in summary.notes)


def test_a_healthy_run_still_ages_the_queue(monkeypatch):
    """The guard must not disable the janitor in normal operation."""
    _seed_run_db()
    cid = db.get_companies()[0]["id"]
    old = (date.today() - __import__("datetime").timedelta(days=40)).isoformat()
    pid, _ = db.upsert_posting(company_id=cid, job_id="stale-2",
                               title="Reporting Analyst",
                               location="Vienna, Austria", posted_date=TODAY,
                               url="https://x.example/s2", description="SQL KPIs")
    with db.connect() as conn:
        conn.execute("UPDATE posting SET first_seen=? WHERE id=?", (old, pid))
        conn.commit()
    monkeypatch.setattr(pipeline, "_fetch_one", lambda c: [])
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(config, "LINKCHECK_ENABLED", False)
    monkeypatch.setattr(matching, "llm_available", lambda: True)
    monkeypatch.setattr(matching, "score_batch", lambda *a, **k: {})   # no fatal
    summary = pipeline.run(verbose=False)
    with db.connect() as conn:
        row = conn.execute("SELECT filtered_reason FROM posting WHERE id=?",
                           (pid,)).fetchone()
    assert (row["filtered_reason"] or "").startswith("not scored within")
    assert any("aged out of the queue" in n for n in summary.notes)


def test_validation_is_skipped_not_silently_destructive(monkeypatch):
    """The validator only ever acts on an explicit 'remove'. With a dead API
    it must review nothing rather than walk every batch."""
    _seed_run_db()
    cid = db.get_companies()[0]["id"]
    pid, _ = db.upsert_posting(company_id=cid, job_id="v-1", title="BI Analyst",
                               location="Vienna, Austria", posted_date=TODAY,
                               url="https://x.example/v1", description="SQL")
    db.save_recommendation(posting_id=pid, prefilter_score=50, fit_score=80.0,
                           rationale="r", watch_outs="", llm_used=True)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(config, "VALIDATION_ENABLED", True)
    monkeypatch.setattr(matching, "llm_available", lambda: True)
    monkeypatch.setattr(validate, "_review_batch",
                        lambda *a, **k: pytest.fail("must not call the API"))
    matching.note_fatal("the Anthropic account is out of credit")

    out = validate.validate_new_recommendations()
    assert out.removed == 0 and out.pending == 1
    assert any("out of credit" in n for n in out.notes)
    assert len(db.board_rows(statuses=["New"])) == 1


def test_keyless_mode_still_scores_heuristically(monkeypatch):
    """Regression guard: the outage handling must not disturb the supported
    no-API-key mode, where the heuristic scorer is the intended path."""
    _seed_run_db()
    monkeypatch.setattr(pipeline, "_fetch_one", lambda c: list(JOBS))
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(config, "LINKCHECK_ENABLED", False)
    summary = pipeline.run(verbose=False)
    assert summary.llm_fatal == ""
    assert summary.scored > 0
    assert all(not r["llm_used"] for r in db.board_rows(statuses=["New"]))


# --- 6. the dashboard ---------------------------------------------------------
def test_api_stats_exposes_the_reason(monkeypatch):
    _fresh_db()
    matching.note_fatal("the Anthropic account is out of credit")
    assert "out of credit" in web.api_stats()["llm_fatal"]


def test_api_stats_reads_the_reason_back_from_the_last_run():
    """After a dashboard restart the live flag is gone; the banner must still
    appear, from what the last run recorded."""
    _fresh_db()
    s = pipeline.RunSummary()
    s.llm_fatal = "the Anthropic API key is invalid or revoked"
    pipeline._finish(s)
    matching.reset_llm_state()
    assert "invalid or revoked" in web.api_stats()["llm_fatal"]


def test_api_stats_is_clean_when_the_api_is_fine():
    _fresh_db()
    pipeline._finish(pipeline.RunSummary())
    assert web.api_stats()["llm_fatal"] == ""

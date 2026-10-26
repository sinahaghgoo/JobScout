"""Learning from the user's own decisions on the board.

Every Skipped recommendation is feedback. This module turns that feedback into
*generalizable* signals — never per-job hacks:

  1. A function-category TAXONOMY classifies any job title into the functional
     family it belongs to (product management, consulting, sales, finance, ...).
  2. `skip_profile()` counts how often the user skipped each category, and
     `applied_profile()` how often they applied — categories the user keeps
     rejecting become penalties, categories they engage with stay open.
  3. `prompt_section()` renders the learned patterns into a short block for the
     LLM system prompt, so the scorer knows what this user rejects in practice.
  4. `prefilter_penalty()` gives the cheap ranking stage the same knowledge.

Analysis of the first 229 manual skips (2026-07) showed dense clusters:
product-manager spam, consulting/deployment-strategist roles, sales/retail,
finance & controlling, policy/programme roles at international organisations,
operations/logistics, HR, marketing, customer support, IT administration,
legal/risk/compliance, data-governance roles dressed as data roles, and
engineering titles that name-drop "analytics". The taxonomy below encodes those
clusters as reusable categories; the counts stay dynamic so future decisions
keep tuning the system without code changes.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

# --- Function-category taxonomy ------------------------------------------------
# Order matters: the FIRST matching category wins. More specific categories
# (engineering, product management) sit above broader ones (sales, ops).
# Patterns match against a lowercased title padded with spaces.
CATEGORY_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("engineering", re.compile(
        r"software (?:engineer|developer)|(?:back|front)[- ]?end|full[- ]?stack"
        r"|\bdata engineer|analytics engineer|platform engineer|cloud engineer"
        r"|infrastructure engineer|security engineer|systems? engineer"
        r"|qa engineer|test engineer|software tester|\bdevops\b|\bsre\b"
        r"|site reliability|machine learning|\bml engineer|\bmlops\b|ml ops"
        r"|deep learning|data scientist|research (?:scientist|engineer)"
        r"|ai engineer|entwickler\b|softwareentwickler|data modeling"
        r"|network engineer|hardware|physical design|field design"
        r"|reliability engineer|integration engineer|performance engineer"
        r"|\bengineer(?:in)?\b", re.I)),
    ("product_management", re.compile(
        r"produ[ck]t[- ]?manager|product owner|produ[ck]t[- ]?own|\bcpo\b"
        r"|product lead|technical product|produktmanagement", re.I)),
    ("consulting", re.compile(
        r"consultant|consulting|advisory|adviser|advisor|presales|pre-sales"
        r"|solution(?:s)? engineer|solution(?:s)? consultant|value engineer"
        r"|deployment strateg|client value|professional services", re.I)),
    ("sales_retail", re.compile(
        r"\bsales\b|account executive|account manager|business development"
        r"|\bbdr\b|\bsdr\b|partner(?:ships)? manager|partnermanager|vertrieb"
        r"|verkauf|store manager|store associate|sales assistant|merchandis"
        r"|kundenberater|shop |retail\b|\bexpert\b(?=.*(?:store|retail))"
        r"|inside sales|field sales|go[- ]to[- ]market|deal desk|deal strateg"
        r"|deal pricing|briefing partner", re.I)),
    ("finance_accounting", re.compile(
        r"account(?:ant|ing)|controller\b|controlling|treasury|\btax\b|steuer"
        r"|payroll|\baudit|buchhalt|rechnungswesen|billing|invoic"
        r"|baufinanzierung|kredit|credit (?:risk|analyst|officer)|rating"
        r"|\bfinance\b|financial (?:analyst|controller|market)|budget"
        r"|versicherung|insurance|private equity|einkauf|einkäufer|procurement"
        r"|sourcing|auftragsrechnung|cash manage", re.I)),
    ("policy_program", re.compile(
        r"policy (?:analyst|officer|advis)|economic affairs|programme? (?:officer"
        r"|coordinator|assistant|associate)|donor relations|monitoring"
        r"(?:,| and| &) evaluation|humanitarian|protection officer|political affairs"
        r"|public affairs|legal officer|human rights|international cooperation"
        r"|reform management|liaison|advocacy|resettlement|peace|governance officer", re.I)),
    ("ops_logistics", re.compile(
        r"operations? (?:manager|supervisor|associate|coordinator|specialist"
        r"|expert|lead)|area manager|shift manager|supply chain|logistics"
        r"|warehouse|inventory|fulfil?lment|delivery station|transporte?"
        r"(?:ation)? (?:manager|supervisor|coordinator)|process improvement"
        r"|process management|lean |produktion|production planner"
        r"|quick commerce|facility|facilities", re.I)),
    ("hr_people", re.compile(
        r"\bhr\b|human resources|people (?:operations|partner|analytics"
        r"|generalist)|recruit(?:er|ing|ment)|talent|learning (?:&|and)"
        r" development|\bl&d\b", re.I)),
    ("marketing_comms", re.compile(
        r"marketing|campaign|communications?\b|content (?:manager|specialist"
        r"|creator)|social media|\bbrand\b|public relations|\bpr manager"
        r"|grafik|graphic design|\bdesigner?\b|\bux\b|\bui\b|media (?:manager"
        r"|planner)|kommunikation", re.I)),
    ("customer_support", re.compile(
        r"customer (?:service|care|support|success)|technical support"
        r"|support (?:engineer|specialist|agent|analyst)|service desk"
        r"|helpdesk|call center|kundenservice|kundenbetreu|servicecenter"
        r"|service center|premium support|customer experience agent", re.I)),
    ("it_admin", re.compile(
        r"application manager|applikationsmanager|system administrator"
        r"|\bsysadmin\b|it[- ]support|it[- ]service|incident manager"
        r"|itsm|service manager|windows|linux admin|network admin"
        r"|application management|it consultant|erp (?:specialist|manager)"
        r"|sap (?:basis|success ?factors|administrator)|workday (?:functional"
        r"|specialist|analyst)", re.I)),
    ("legal_risk_compliance", re.compile(
        r"\blegal\b|counsel|compliance|sanctions|\brisk\b|\baml\b|anti[- ]money"
        r"|fraud|\bkyc\b|due diligence|regulatory affairs|qc analyst"
        r"|quality (?:control|assurance)|infosec|information security"
        r"|security analyst|penetration|datenschutz|privacy", re.I)),
    ("data_governance", re.compile(
        r"data governance|data steward|master data|data quality|data management"
        r"|data operations|records management|document management|data protection"
        r"|governance architect|metadata", re.I)),
    ("project_process_mgmt", re.compile(
        r"project manager|projektleit|projektmanag|program manager|programm-?"
        r"manager|\bpmo\b|scrum master|agile coach|prozessmanag|process manager"
        r"|transformation manager|change manager|delivery manager", re.I)),
]

# What each category may score at most when the LLM judges function fit.
# Informed by the user's preferences AND their observed skip behaviour;
# consulting sits higher because the user occasionally applies to
# consulting-flavoured analyst/graduate roles.
CATEGORY_CAPS: dict[str, int] = {
    "engineering": 35,
    "product_management": 40,
    "consulting": 45,
    "sales_retail": 35,
    "finance_accounting": 45,
    "policy_program": 40,
    "ops_logistics": 40,
    "hr_people": 40,
    "marketing_comms": 40,
    "customer_support": 30,
    "it_admin": 35,
    "legal_risk_compliance": 35,
    "data_governance": 45,
    "project_process_mgmt": 40,
}

# Titles that signal the role IS analytics-core even when a category pattern
# also matches (e.g. "Transport Performance Analyst" hits ops_logistics but is
# a data role the user applied to). If one of these matches, the category
# penalty must not fire blindly — the LLM decides with full context instead.
_ANALYTICS_CORE = re.compile(
    r"(?:business intelligence|\bbi\b|data|analytics?|reporting|insights?"
    r"|market intelligence|performance|pricing|commercial)[\s-]*"
    r"(?:analyst|analytics|developer\b)|analyst(?:in)?\b"
    r"|business intelligence engineer|\bbi engineer\b", re.I)


def categorize_title(title: str) -> Optional[str]:
    """First matching function category for a title, or None."""
    t = f" {(title or '').lower()} "
    for name, rx in CATEGORY_PATTERNS:
        if rx.search(t):
            return name
    return None


def is_analytics_core_title(title: str) -> bool:
    return bool(_ANALYTICS_CORE.search(title or ""))


# --- Decision profiles ----------------------------------------------------------
def _profile(titles: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in titles:
        cat = categorize_title(t)
        if cat:
            counts[cat] = counts.get(cat, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def skip_profile(skipped_titles: Iterable[str]) -> dict[str, int]:
    """How many manually-skipped roles fall into each function category."""
    return _profile(skipped_titles)


def applied_profile(applied_titles: Iterable[str]) -> dict[str, int]:
    return _profile(applied_titles)


_SKIP_BASELINE_KEY = "skip_profile_baseline"
_APPLIED_BASELINE_KEY = "applied_profile_baseline"


def _merge(into: dict[str, int], other: dict[str, int]) -> dict[str, int]:
    for k, v in other.items():
        into[k] = into.get(k, 0) + int(v)
    return into


def load_profiles(db_module) -> tuple[dict[str, int], dict[str, int]]:
    """Read Skipped/Applied titles from the DB and build both profiles.

    A board reset deletes Skipped rows but must not delete what they taught
    us — `persist_baselines()` freezes the counts into meta first, and this
    loader folds any stored baseline back in."""
    import json
    skips = skip_profile(r["title"] for r in db_module.board_rows(statuses=["Skipped"]))
    applies = applied_profile(r["title"] for r in db_module.board_rows(statuses=["Applied"]))
    for key, target in ((_SKIP_BASELINE_KEY, skips), (_APPLIED_BASELINE_KEY, applies)):
        raw = db_module.meta_get(key)
        if raw:
            try:
                _merge(target, json.loads(raw))
            except (ValueError, TypeError):
                pass
    return (dict(sorted(skips.items(), key=lambda kv: -kv[1])),
            dict(sorted(applies.items(), key=lambda kv: -kv[1])))


def persist_baselines(db_module, freeze_statuses) -> dict[str, dict[str, int]]:
    """Fold the decision rows that are ABOUT TO BE DELETED into the stored
    baselines. Only statuses in `freeze_statuses` are frozen — rows that
    survive a reset keep contributing live, so nothing is ever counted twice.
    Call exactly once, immediately before deleting the rows."""
    import json
    out: dict[str, dict[str, int]] = {}
    for status, key, build in (("Skipped", _SKIP_BASELINE_KEY, skip_profile),
                               ("Applied", _APPLIED_BASELINE_KEY, applied_profile)):
        if status not in freeze_statuses:
            continue
        live = build(r["title"] for r in db_module.board_rows(statuses=[status]))
        raw = db_module.meta_get(key)
        baseline = {}
        if raw:
            try:
                baseline = json.loads(raw)
            except (ValueError, TypeError):
                baseline = {}
        merged = _merge(dict(baseline), live)
        db_module.meta_set(key, json.dumps(merged))
        out[status] = merged
    return out


# --- Consumers -------------------------------------------------------------------
_MIN_SKIPS_FOR_SIGNAL = 3          # a pattern needs repetition before it counts
_MIN_SKIPS_FOR_PREFILTER = 5       # the cheap stage needs stronger evidence


def avoided_categories(skips: dict[str, int],
                       applies: Optional[dict[str, int]] = None) -> dict[str, int]:
    """Categories the user demonstrably avoids: repeatedly skipped and rarely
    (less than a third as often) applied to."""
    applies = applies or {}
    out = {}
    for cat, n in skips.items():
        if n >= _MIN_SKIPS_FOR_SIGNAL and applies.get(cat, 0) * 3 <= n:
            out[cat] = n
    return out


_CAT_LABELS = {
    "engineering": "software/data/ML engineering roles",
    "product_management": "product-manager / product-owner roles",
    "consulting": "consulting / presales / deployment-strategist roles",
    "sales_retail": "sales, business-development and retail/store roles",
    "finance_accounting": "finance, accounting, controlling, treasury, credit and procurement roles",
    "policy_program": "policy / programme-officer / international-affairs roles",
    "ops_logistics": "operations, logistics, supply-chain and warehouse roles",
    "hr_people": "HR / recruiting / people roles",
    "marketing_comms": "marketing, communications, content and design roles",
    "customer_support": "customer-service / support roles",
    "it_admin": "IT administration / application-management roles",
    "legal_risk_compliance": "legal, risk, compliance and QA/QC roles",
    "data_governance": "data-governance / master-data / data-quality roles",
    "project_process_mgmt": "project / programme / process-management roles",
}


def prompt_section(skips: dict[str, int],
                   applies: Optional[dict[str, int]] = None) -> str:
    """A short system-prompt block describing the user's observed skip patterns.

    Empty string when there is not enough history yet."""
    avoided = avoided_categories(skips, applies)
    if not avoided:
        return ""
    lines = [f"- {_CAT_LABELS.get(c, c)} (skipped {n}x)"
             for c, n in sorted(avoided.items(), key=lambda kv: -kv[1])[:12]]
    return (
        "LEARNED SKIP HISTORY — this candidate has manually rejected every "
        "recommendation matching these patterns. Score similar roles <= 40 "
        "(<= 30 when the match is obvious) unless the day-to-day work is "
        "unmistakably core BI/data analytics:\n" + "\n".join(lines))


def prefilter_penalty(title: str, skips: Optional[dict[str, int]],
                      applies: Optional[dict[str, int]] = None) -> tuple[float, Optional[str]]:
    """Extra prefilter penalty for titles matching a heavily-skipped category.

    Conservative: needs >= 5 skips in the category, and never fires on
    analytics-core titles (the LLM judges those with full context)."""
    if not skips:
        return 0.0, None
    cat = categorize_title(title)
    if not cat or is_analytics_core_title(title):
        return 0.0, None
    n = avoided_categories(skips, applies).get(cat, 0)
    if n >= _MIN_SKIPS_FOR_PREFILTER:
        return -12.0, cat
    return 0.0, None

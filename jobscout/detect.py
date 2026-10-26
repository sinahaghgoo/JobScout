"""One-time ATS classification.

For each company we figure out *once* which ATS it uses and what its token is,
then store it. From then on, fetching that company's jobs is a single, fast,
structured request.

Three sources, in order:
  1. KNOWN_TOKENS  — a curated map of verified (ats_type, token) overrides,
     keyed by the company's exact name. Every entry here was confirmed live
     against the platform's public feed (June 2026).
  2. KNOWN_UNSUPPORTED — companies researched and confirmed to run a career
     platform with no usable public structured feed (Cornerstone, Avature,
     onlyfy, iCIMS, bespoke portals, SAP-hosted client-rendered SuccessFactors,
     and the big-tech proprietary systems). Marked unsupported without probing,
     which also prevents same-name demo boards from being mis-assigned.
  3. Candidate probing — slug variants of the company name tried against each
     slug-guessable connector; the first that returns >=1 job wins.

The curated maps always take precedence over slug-probing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from . import config
from .connectors import DETECTION_ORDER
from .connectors.registry import probe as _registry_probe


def _key(name: str) -> str:
    """Stable lookup key: case-folded, whitespace-collapsed (keeps umlauts)."""
    return " ".join((name or "").split()).casefold()


def _load_careers_seed() -> dict[str, str]:
    """Curated `company -> careers_url` hints for the Tier-2 resolver."""
    seed: dict[str, str] = {}
    try:
        import csv
        if config.CAREERS_SEED_CSV.is_file():
            with open(config.CAREERS_SEED_CSV, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    nm = (row.get("name") or "").strip()
                    url = (row.get("careers_url") or "").strip()
                    if nm and not nm.startswith("#") and url:
                        seed[_key(nm)] = url
    except Exception:  # noqa: BLE001
        pass
    return seed


CAREERS_SEED: dict[str, str] = _load_careers_seed()


# --- Curated, verified overrides --------------------------------------------
# Keyed by the company's exact name as it appears in the company list.
# token may be "tenant|dc|site" (Workday), a career host (SuccessFactors / sf_rmk
# / d.vinci / eRecruiter), "host|site" (Oracle), or a slug. Every token below was
# verified live (probe()==True against the real feed), June 2026.
_RAW_KNOWN_TOKENS: dict[str, tuple[str, str]] = {
    # ==== Austrian Tier-1 & 2 ================================================
    "Erste Group / Erste Digital": ("successfactors", "erstegroup-careers.com"),
    "Mondi": ("successfactors", "careers.mondigroup.com"),
    "Andritz": ("successfactors", "careers.andritz.com"),
    "Red Bull": ("smartrecruiters", "redbull"),
    "Österreichische Post": ("successfactors", "karriere.post.at"),
    "Wiener Stadtwerke / Wien Energie": ("dvinci", "jobs.wienerstadtwerke.at"),
    "BRZ Bundesrechenzentrum": ("erecruiter", "www.brz-jobs.at"),
    # (non-yellow but kept: harmless if not seeded)
    "Bitpanda": ("greenhouse", "bitpanda"),
    "GoStudent": ("greenhouse", "gostudent"),
    "Frequentis": ("smartrecruiters", "frequentis"),
    "Anyline": ("personio", "anyline-gmbh"),
    "Tricentis": ("workday", "tricentis|wd1|Tricentis_Careers"),
    # ==== Big Tech (US-HQ EU offices) ========================================
    "NVIDIA": ("workday", "nvidia|wd5|NVIDIAExternalCareerSite"),
    "Salesforce": ("workday", "salesforce|wd12|External_Career_Site"),
    "Adobe": ("workday", "adobe|wd5|external_experienced"),
    "Palantir": ("lever", "palantir"),
    "OpenAI": ("ashby", "openai"),
    "Anthropic": ("greenhouse", "anthropic"),
    "Snowflake": ("ashby", "snowflake"),
    "Stripe": ("greenhouse", "stripe"),
    "ServiceNow": ("smartrecruiters", "servicenow"),
    "Google DeepMind": ("greenhouse", "deepmind"),
    # Target countries only (profile: AT/DE/CH/IE/UK/NL/FR/SE + Barcelona).
    # The old 21-country EU set burned the fetch cap on BEL/ITA/POL/... roles
    # the location hard-filter dropped anyway (live 2026-07-09: 2,861 hits vs
    # 2,400 in the 9 target countries — and a 600 cap).
    "Amazon / AWS": ("amazon", "AUT,DEU,CHE,IRL,GBR,NLD,FRA,SWE,ESP"),
    "Apple": ("apple", "apple"),                       # region-filtered to West Europe
    "Microsoft": ("microsoft", "microsoft"),           # built-to-spec; verify on a normal network
    "Oracle": ("oracle", "eeho.fa.us2.oraclecloud.com|CX_45001"),
    # ==== European Tech ======================================================
    "SAP": ("successfactors", "jobs.sap.com"),
    "Celonis": ("greenhouse", "celonis"),
    "Databricks": ("greenhouse", "databricks"),
    # 2026-09-01: Mistral left Lever for Ashby. The old Lever board still
    # answers 200 with an EMPTY list, so the fetch looked "successful"
    # and simply reported 0 jobs for weeks — 172 postings invisible.
    "Mistral AI": ("ashby", "mistral.ai"),
    "ASML": ("workday", "asml|wd3|ASMLEXT1"),
    "Adyen": ("greenhouse", "adyen"),
    "Atos / Eviden": ("successfactors", "jobs.atos.net"),
    "Check24": ("smartrecruiters", "CHECK24"),
    # ==== Pharma & Life Sciences =============================================
    "Roche": ("workday", "roche|wd3|roche-ext"),
    "Novartis": ("workday", "novartis|wd3|Novartis_Careers"),
    "Novo Nordisk": ("successfactors", "careers.novonordisk.com"),
    # ==== Industrial & Energy ================================================
    "Bosch": ("smartrecruiters", "BoschGroup"),
    "ABB": ("workday", "abb|wd3|External_Career_Page"),
    "Shell": ("workday", "shell|wd3|ShellCareers"),
    "BP": ("workday", "bpinternational|wd3|bpCareers"),
    "Equinor": ("workday", "equinor|wd3|EQNR"),
    "Eni": ("oracle", "fa-evkm-saasfaprod1.fa.ocs.oraclecloud.com|CX_1004"),
    "Logitech": ("workday", "logitech|wd5|Logitech"),
    "Rolls-Royce": ("workday", "rollsroyce|wd3|professional"),
    "Repsol": ("workday", "repsol|wd3|Repsol"),
    "Philips": ("workday", "philips|wd3|jobs-and-careers"),
    "Renault": ("workday", "alliancewd|wd3|renault-group-careers"),
    # ==== Consumer Retail & Luxury ===========================================
    "LVMH": ("smartrecruiters", "LVMH2"),
    "Adidas": ("sf_rmk", "jobs.adidas-group.com"),
    "LEGO Group": ("workday", "lego|wd103|LEGO_External"),
    "Carlsberg": ("successfactors", "careers.carlsberg.com"),
    # ==== Automotive =========================================================
    "BMW": ("sf_rmk", "jobs.bmwgroup.com"),
    "Volkswagen / CARIAD": ("successfactors", "jobs.volkswagen-group.com"),
    "Audi": ("successfactors", "jobs.volkswagen-group.com"),
    # ==== Telecom & Logistics ================================================
    "Maersk": ("workday", "maersk|wd3|Maersk_Careers"),
    "Air France-KLM": ("successfactors", "vacancy.klm.com"),
    # ==== Media & Travel =====================================================
    "Sky": ("workday", "sky|wd3|Sky_Careers"),
    "Airbnb": ("greenhouse", "airbnb"),
    # ==== International Organizations =========================================
    "UNHCR - UN Refugee Agency": ("workday", "unhcr|wd3|External"),
    "WEF - World Economic Forum": ("workday", "weforum|wd3|Forum_Careers"),
    "IOM - International Organization for Migration":
        ("oracle", "fa-evlj-saasfaprod1.fa.ocs.oraclecloud.com|CX_1001"),
    "WMO - World Meteorological Organization":
        ("oracle", "estm.fa.em2.oraclecloud.com|CX_5001"),
    "OECD - Org for Economic Co-operation and Development": ("smartrecruiters", "OECD"),
    "IEA - International Energy Agency": ("smartrecruiters", "OECD|iea,energy agency"),
    "FATF - Financial Action Task Force": ("smartrecruiters", "OECD|fatf,financial action"),
    "ESA - European Space Agency (HQ)": ("successfactors", "jobs.esa.int"),
    "ICRC - International Committee of the Red Cross": ("successfactors", "careers.icrc.org"),
    "UNESCO": ("successfactors", "careers.unesco.org"),
    "OFID - OPEC Fund for International Development": ("successfactors", "careers.opecfund.org"),
    "ITU - International Telecommunication Union": ("successfactors", "jobs.itu.int"),
    "UNIDO - UN Industrial Development Org": ("successfactors", "careers.unido.org"),
    "ILO - International Labour Organization": ("sf_rmk", "jobs.ilo.org"),
    # UN Secretariat entities recruit through the careers.un.org portal, which
    # exposes a public JSON API (see connectors/un.py) — a real feed with full
    # descriptions and authoritative dates, filtered per entity client-side.
    "UNODC - UN Office on Drugs and Crime": ("un", "dept=Office on Drugs and Crime"),
    "UNOV - UN Office at Vienna": ("un", "dept=Office at Vienna"),
    "UNOG - UN Office at Geneva": ("un", "dept=Office at Geneva"),
    "UNECE - UN Economic Commission for Europe": ("un", "dept=Economic Commission for Europe"),
    "UNCTAD - UN Conference on Trade and Development": ("un", "dept=Trade and Development"),
    # ICJ Registry vacancies are published on the same UN portal when open.
    "ICJ - International Court of Justice": ("un", "dept=International Court of Justice"),
    # Sanofi runs a public Workday board (verified live).
    "Sanofi": ("workday", "sanofi|wd3|SanofiCareers"),
    # Uber's SPA serves stale HTML to non-JS clients (dead job links); its own
    # JSON search API is exact — routed as a feed (see connectors/uber.py).
    "Uber": ("uber", "uber"),
    # VERBUND's job list fills from a first-party Umbraco JSON API; the branded
    # portal page itself never exposes postings to a scraper (consent-gated
    # widget) — routed as a feed (see connectors/verbund.py).
    "Verbund": ("verbund", "verbund"),
    # Jibe (iCIMS Career Sites) white-label hosts: the SPA shows 10 jobs/page
    # behind Akamai, but the first-party /api/jobs JSON is complete, with real
    # posted dates + descriptions (see connectors/jibe.py).
    "Booking.com": ("jibe", "jobs.booking.com"),
    "Schneider Electric": ("jibe", "careers.se.com"),
    # Modern BeeSite portals expose the vendor's Global Job Board JSON API —
    # full boards with real publication dates, country-filtered server-side to
    # the profile's target countries (see connectors/beesite.py).
    "Mercedes-Benz": ("beesite", "mercedes-benz-beesite-production-gjb.app.beesite.de|AT,DE,GB,IE,SE,CH,ES,FR,NL"),
    "Porsche": ("beesite", "porsche-beesite-production-gjb.app.beesite.de|AT,DE,GB,IE,SE,CH,ES,FR,NL"),
    # careers.nike.com is a skin over Workday (apply links betray tenant nke @
    # wd1); the CXS feed gives the full board with real posted dates.
    "Nike EMEA": ("workday", "nike|wd1|nke"),
    # IBM careers search rides an Elasticsearch-style first-party API with
    # server-side country filtering (see connectors/ibm.py).
    "IBM": ("ibm", "AT,DE,GB,IE,SE,CH,ES,FR,NL"),
    # WTO retired erecruitment.wto.org (NXDOMAIN, mid-2026) and moved to
    # Workday — wto.org's vacancies page links the External career site.
    "WTO - World Trade Organization": ("workday", "wto|wd103|External"),
}

# --- Curated browser-tier (Tier-3) -------------------------------------------
# Researched client-rendered / bespoke career sites with NO server-reachable feed.
# Value = the best public job-listing URL. Detection marks these ats_type="browser";
# the pipeline renders + LLM-extracts them (HTTP-first, Playwright for SPAs). They
# populate when run with the browser tier available (e.g. on the user's Mac).
_RAW_KNOWN_BROWSER: dict[str, str] = {
    # ---- Industrial / Auto / Pharma (SAP-shared SF, Avature, bespoke, auth-WD)
    # Avature portal: /Home is a cookie-walled landing page; /SearchJobs
    # server-renders the actual list (20/page) even without a browser.
    "TotalEnergies": "https://jobs.totalenergies.com/en_US/careers/SearchJobs"
                     " + https://jobs.totalenergies.com/en_US/careers/SearchJobs?jobOffset=20"
                     " + https://jobs.totalenergies.com/en_US/careers/SearchJobs?jobOffset=40",
    "Tesla": "https://www.tesla.com/careers/search/",
    # ---- Consumer / Telecom / Media
    # jobRecordsPerPage: Avature honours it here (50/page vs 20) — two offset
    # pages cover the 100 most recent of a ~1000-role global board.
    "L'Oréal": "https://careers.loreal.com/en_US/jobs/SearchJobs?jobRecordsPerPage=50"
               " + https://careers.loreal.com/en_US/jobs/SearchJobs?jobRecordsPerPage=50&jobOffset=50",
    # Lufthansa Group runs a BeeSite portal; the rendered search-result page
    # lists every group company's ads with real publication dates. Austrian
    # Airlines is isolated via the portal's full-text keyword filter.
    "Lufthansa Group": "https://apply.lufthansagroup.careers/index.php?ac=search_result&language=2",
    "Austrian Airlines (Lufthansa Group)": "https://apply.lufthansagroup.careers/index.php?ac=search_result&language=2&search_criterion_keyword=Austrian+Airlines",
    # ---- Austrian
    "A1 Telekom Austria": "https://jobs.a1.com/job-search/",
    "ÖBB Austrian Railways": "https://oebb.csod.com/ux/ats/careersite/4/home?c=oebb",
    # ---- Big Tech (Avature / Akamai-gated; best-effort on the user's machine)
    # Bloomberg's global board is 400+ roles at 12/page; the portal's keyword
    # path filter isolates the profile-relevant London office (both pages).
    "Bloomberg": "https://bloomberg.avature.net/careers/SearchJobs/London"
                 " + https://bloomberg.avature.net/careers/SearchJobs/London?jobOffset=12",
    # ---- International Organizations: Oracle Taleo (CSRF-walled to servers)
    "IAEA - International Atomic Energy Agency": "https://iaea.taleo.net/careersection/ex/jobsearch.ftl?lang=en",
    "WHO - World Health Organization": "https://careers.who.int/careersection/ex/jobsearch.ftl?lang=en",
    "WIPO - World Intellectual Property Organization": "https://wipo.taleo.net/careersection/wp_2/jobsearch.ftl?lang=en",
    "OSCE - Org for Security and Co-operation in Europe": "https://vacancies.osce.org/",
    # ---- International Organizations: SAP-shared SF & bespoke portals
    "CTBTO - Comprehensive Nuclear-Test-Ban Treaty Org": "https://jobs.ctbto.org/",
    # careers.icc-cpi.int was retired mid-2026 (NXDOMAIN); the live portal is
    # the legacy-SF career?company= UI (same flow the renderer handles for
    # CTBTO: empty-search submit lists everything).
    "ICC - International Criminal Court": "https://career5.successfactors.eu/career?company=1657261P",
    "ECB - European Central Bank": "https://talent.ecb.europa.eu/",
    "EDA - European Defence Agency": "https://www.eda.europa.eu/careers",
    "EBA - European Banking Authority": "https://www.eba.europa.eu/about-us/careers/vacancies",
    "ESMA - European Securities and Markets Authority": "https://esmacareers.adequasys.com/",
    "OPCW - Org for the Prohibition of Chemical Weapons": "https://jobs.opcw.org/",
    # The standalone TalentLink search honours ?page= (the widget embedded on
    # www.ifrc.org/jobs does not); the list is closing-date-sorted, so later
    # pages hold the newest postings.
    "IFRC - International Federation of Red Cross":
        "https://careers.ifrc.org/lumesse_jobsearch.html"
        " + https://careers.ifrc.org/lumesse_jobsearch.html?page=2"
        " + https://careers.ifrc.org/lumesse_jobsearch.html?page=3",
    "Europol": "https://www.europol.europa.eu/work-with-us/careers/open-vacancies",
    # /careers/ is a landing page; the FacetWP-driven list lives one level down.
    "ICC - International Chamber of Commerce": "https://iccwbo.org/careers/career-opportunities/",
    # ---- EU institutions (EPSO restructured mid-2026: /open-vacancies is now a
    # hub page; actual lists are server-rendered on per-type pages. The two
    # EPSO-fed companies split coverage: permanent officials vs temporary
    # agents across institutions. apply4ep lists into /search when calls open.)
    "European Commission": "https://eu-careers.europa.eu/en/permanent-vacancies",
    "European Parliament": "https://apply4ep.gestmax.eu/search/index/lang/en_US",
    "Council of the EU": "https://eu-careers.europa.eu/en/temporary-agents-other-institutions-vacancies",
}

# Genuinely out of reach under any feed/browser architecture from a server:
# anti-bot proprietary stacks that defeat headless rendering, or apply-by-email.
_RAW_KNOWN_UNSUPPORTED: dict[str, str] = {
    "Google": "Proprietary anti-bot (google.com/about/careers); defeats headless rendering",
    "Meta": "Proprietary anti-bot (metacareers.com); defeats headless rendering",
    "TikTok / ByteDance": "Anti-bot signed requests (a-bogus); careers.tiktok.com API blocks non-JS clients",
    "OPEC - Org of the Petroleum Exporting Countries": "Apply-by-email only (recruitment@opec.org); no vacancy portal",
}

KNOWN_TOKENS: dict[str, tuple[str, str]] = {
    _key(k): v for k, v in _RAW_KNOWN_TOKENS.items()}
KNOWN_BROWSER: dict[str, str] = {
    _key(k): v for k, v in _RAW_KNOWN_BROWSER.items()}
KNOWN_UNSUPPORTED: dict[str, str] = {
    _key(k): v for k, v in _RAW_KNOWN_UNSUPPORTED.items()}

_SUFFIXES = {
    "ag", "gmbh", "se", "inc", "group", "international", "holding", "plc",
    "nv", "sa", "the", "co", "kg", "ab", "as", "oyj", "spa", "list",
}


def candidate_tokens(name: str) -> list[str]:
    """Generate plausible ATS slug tokens from a company name."""
    n = name.lower()
    n = n.split("(")[0]                     # drop parentheticals
    n = re.split(r"[/|]", n)[0]             # take part before a slash
    n = re.sub(r"[&.,'’]", " ", n)
    n = re.sub(r"[^a-z0-9 -]", " ", n)
    words = [w for w in n.split() if w]
    core = [w for w in words if w not in _SUFFIXES] or words

    out: list[str] = []

    def add(s: str) -> None:
        s = s.strip("-")
        if s and s not in out:
            out.append(s)

    add("".join(core))
    if core:
        add(core[0])
    if len(core) >= 2:
        add(core[0] + core[1])
        add(core[0] + "-" + core[1])
    add("-".join(core))
    add("".join(words))
    # CamelCase variants help SmartRecruiters (e.g. "BoschGroup").
    add("".join(w.capitalize() for w in core))
    return out[:7]


@dataclass
class Detection:
    ats_type: Optional[str]
    token: Optional[str]
    status: str  # detected | unsupported
    tried: int = 0
    note: str = ""


def _probe(ats_type: str, token: str) -> bool:
    # Connector.probe is cheap (one page for paginated connectors).
    try:
        return _registry_probe(ats_type, token)
    except Exception:  # noqa: BLE001 - detection must never crash a run
        return False


def detect(name: str, verbose: bool = False, known_only: bool = False,
           use_resolver: bool = True, careers_url: Optional[str] = None) -> Detection:
    """Classify a company's ATS, tier by tier:

      1. curated KNOWN_TOKENS (verified)            -> feed
      2. slug auto-detect (Tier-1)                  -> feed
      3. careers-page resolver (Tier-2/3)           -> feed, or browser
      4. researched KNOWN_UNSUPPORTED / no feed     -> unsupported
    """
    key = _key(name)

    # 1) curated feed override (always preferred) — verified live tokens.
    # Curated tokens are authoritative: if the probe fails we still return the
    # curated token (most failures are transient network blips), rather than
    # falling through to slug-guessing, which could mis-assign a same-named demo
    # board to this company. A truly-dead token just yields 0 jobs at fetch time.
    if key in KNOWN_TOKENS:
        ats, token = KNOWN_TOKENS[key]
        if ats == "microsoft" or _probe(ats, token):
            return Detection(ats, token, "detected", note="curated")
        return Detection(ats, token, "detected", note="curated (probe failed — trusting curation)")

    # 2) curated browser-tier (researched client-rendered; Tier-3 renders + extracts)
    if key in KNOWN_BROWSER:
        return Detection("browser", KNOWN_BROWSER[key], "detected", note="curated-browser")

    # known_only mode skips slug-guessing + resolver entirely (fast, collision-free)
    if known_only:
        if key in KNOWN_UNSUPPORTED:
            return Detection(None, None, "unsupported", note=KNOWN_UNSUPPORTED[key])
        return Detection(None, None, "unsupported", note="not in curated map")

    # 2) slug auto-detect (Tier-1)
    tried = 0
    for ats in DETECTION_ORDER:
        for token in candidate_tokens(name):
            tried += 1
            if verbose:
                print(f"    probe {ats}:{token}")
            if _probe(ats, token):
                return Detection(ats, token, "detected", tried, note="probed")

    # 3) careers-page resolver (Tier-2 feed / Tier-3 browser)
    if use_resolver:
        try:
            from . import resolver  # lazy: keeps resolver/requests optional
            res = resolver.resolve(name, careers_url=careers_url or CAREERS_SEED.get(key),
                                   guess=True)
        except Exception:  # noqa: BLE001 - resolution must never crash detection
            res = None
        if res and res.tier == "feed":
            return Detection(res.ats_type, res.token, "detected", tried,
                             note=f"resolved:{res.ats_type}")
        if res and res.tier == "browser":
            return Detection("browser", res.token, "detected", tried, note=res.note)

    # 4) researched as having no usable public feed, or nothing found
    if key in KNOWN_UNSUPPORTED:
        return Detection(None, None, "unsupported", tried, note=KNOWN_UNSUPPORTED[key])
    return Detection(None, None, "unsupported", tried, note="no feed found")

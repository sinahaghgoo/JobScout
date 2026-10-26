"""The matching engine — three stages, for quality and cost.

1. HARD FILTERS drop postings the profile can never want, with a recorded
   reason: internships / working-student / trainee roles, clearly senior
   roles (title or 5+ years required), roles requiring a language the
   candidate does not speak, locations outside the target countries, and
   postings that are neither English nor German. These never reach the LLM,
   never appear on the board, and cost nothing.

2. A cheap PREFILTER orders what's left (title family, skills, location,
   seniority signals, learned skip patterns). Ordering only — EVERY survivor
   is scored; ranking just makes sure a budget stop leaves the most
   promising postings already judged.

3. Claude Sonnet SCORES the survivors in small batches against the CV and
   the preferences note (sent once per run via prompt caching). The model
   must evaluate a fixed set of HARD GATES per posting before scoring;
   deterministic post-checks clamp any score that contradicts a failed gate,
   so a hard requirement can never be out-reasoned. If no ANTHROPIC_API_KEY
   is set, a keyless heuristic scorer stands in (clearly labeled, capped).
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

from . import config, feedback
from .lang import is_supported_language

# =============================================================================
# Stage 1 — hard filters
# =============================================================================

# --- Role types the profile explicitly excludes (word-boundary regexes; NB:
# plain "intern" must not hit "internal"/"international") -----------------
_EARLY_CAREER_EXCLUDE = re.compile(
    # NB: no \b before praktik/werkstudent/…arbeit — German compounds
    # ("Pflichtpraktikum", "Werkstudententätigkeit", "Studienabschlussarbeit")
    # hide the stem mid-word.
    r"\bintern(?:ship|s)?\b|praktik(?:um|ant|a\b)|werkstudent|\bworking\s+student\b"
    r"|\bstudent(?:ische)?\s+(?:assistant|worker|hilfskraft|job)|\bstudentenjob"
    r"|\bapprentice|\blehrling|\blehrstelle|\blehre\s+(?:zum|zur|als)\b"
    r"|\bausbildung|\bazubi\b"
    r"|\bduale?s?\s+studium|\bdual\s+study|\bferial|\bferienjob"
    r"|\bthesis\b|abschlussarbeit|masterarbeit|bachelorarbeit|diplomarbeit"
    r"|\bmasterand|\bbachelorand|\bdoktorand|\bphd\b|\bdoctoral|\bpostdoc"
    r"|\bvolunteer|\bvolont[aä]r|\bvolontariat|\bstagiaire\b|\balternan[ct]"
    r"|\bsummer\s+(?:student|job|intern)|\bco-op\b|\bgap\s+year\b"
    r"|\bbecari[oa]",   # Spanish internship titles ("BECARIO CONTROL Y REPORTING")
    re.I)
_TRAINEE = re.compile(r"\btrainee\b|\btraineeship\b|\btrainee-programm", re.I)

# --- In-store retail roles: never wanted, hard-excluded (user request
# 2026-07-09 after Adidas/Nike store staff flooded the archive). Targets
# STORE-STAFF titles specifically — corporate roles about the retail
# *business* ("Retail Media Analyst", "...im Retailgeschäft") must survive,
# hence word-level patterns and the "app store" guard.
_RETAIL_STORE = re.compile(
    r"sales\s+assistant|sales\s+associate|store\s+associate|retail\s+associate"
    r"|retail\s+assistant"      # Adidas/Nike floor staff (13 slipped 2026-07)
    r"|shop\s+assistant|stockroom|\bcashier|kassier"
    r"|(?<!app\s)store\s+manager|floor\s+manager"
    r"|store\s+(?:lead(?:er)?|supervisor|coach|colleague|crew|team\s+member)"
    r"|\bflagship\b|visual\s+merchandis|\bmerchandiser\b"
    r"|factory\s+outlet|\boutlet\b|retail\s+store|in-?store\b"
    r"|verk[äa]ufer|einzelhandel|filial(?:leit|mitarbeit)"
    r"|\bathlete\b|\bgenius\b"    # Nike / Apple store-staff titles
    r"|\baushilfe\b|\bminijob|geringf[üu]gig|saisonkraft|\bseasonal\b",
    re.I)

# --- Occupations with zero overlap with a business/data-analytics profile.
# These reached the LLM queue (and the board, at fit 5-15) purely on geography:
# BMW Kfz-Mechatroniker, Porsche Karosseriebauer, ICC Trial Lawyer, Check24
# Finanzwirt, Andritz Maschinenbautechniker, Schneider Servicetechniker … all
# manually skipped by the user (2026-07). Word-level patterns; German compounds
# are matched mid-word on purpose (\w*techniker, \w*fahrer, \w*monteur).
# `hard_filter` never applies this to analytics-core titles (extra guard), so
# a hypothetical "Analyst Fahrerassistenzsysteme" would still survive.
_OFF_PROFILE_OCCUPATION = re.compile(
    # vehicles / workshop / manufacturing trades
    r"\bkfz\b|karosserie|lackierer|mechatronik|mechaniker(?:in)?\b|\bmechanic\b"
    r"|zerspanung|schwei[ßs]s?er|schlosser|elektriker|elektroniker(?:in)?\b"
    r"|elektroinstall|metallbauer|werkzeugmacher|\w*monteur(?:in)?\b"
    r"|\w*techniker(?:in)?\b|\btechnician\b|\belectrician\b|\bwelder\b"
    r"|\bmachinist\b|toolmaker|\bfitter\b|\bassembler\b"
    r"|facharbeiter|produktionsmitarbeiter|fertigungsmitarbeiter"
    r"|montagemitarbeiter|produktionshelfer|maschinenbediener"
    r"|maschinenf[üu]hrer|anlagenf[üu]hrer|anlagenbediener"
    r"|(?:machine|production|process|plant)\s+operat(?:or|ive)"
    r"|production\s+(?:worker|associate)"
    # construction / installation / crafts
    r"|tischler|schreiner|zimmerer|zimmermann|dachdecker|maurer|betonbauer"
    r"|trockenbau|installateur|klempner|spengler|sanit[äa]r|heizungsbau"
    r"|k[äa]lteanlagen|\bmaler\b|\bplumber\b|\bcarpenter\b|ger[üu]stbauer"
    r"|g[äa]rtner|landschaftsbau|\b(?:forst|land)wirt\b"
    # logistics floor / drivers
    r"|staplerfahrer|gabelstapler|\bforklift|lagermitarbeiter|lagerist"
    r"|kommissionier|warehouse\s+(?:operative|associate|worker|assistant|team)"
    r"|order\s+picker|\w*fahrer(?:in)?\b|\bdriver\b|chauffeur|\bcourier\b"
    r"|zusteller|postbote"
    # medical / care / education / social
    r"|krankenpfleg|pflegefachkraft|pflegekraft|pflegehelfer|altenpfleg"
    r"|\bnurse\b|nursing|physician|\barzt\b|[äa]rztin|zahnarzt|zahnmedizin"
    r"|apotheker|pharmacist|physiotherap|ergotherap|logop[äa]d|hebamme"
    r"|toxicolog|laborant(?:in)?\b|radiolog(?:e|in|ist)\b|caregiver"
    r"|erzieher(?:in)?\b|kindergartenp[äa]dagog|kinderbetreu|sozialarbeiter"
    r"|sozialp[äa]dagog|\blehrkraft\b"
    # legal practice / notarial / tax-clerical professions
    r"|\blawyer\b|attorney|barrister|solicitor|\bjurist(?:in)?\b"
    r"|rechtsanw[äa]lt\w*|syndikus|\bnotar\b|paralegal|\bcounsel\b"
    r"|finanzwirt|steuerberater|steuerfachangestellte|steuerfachwirt"
    r"|steuerassistent|bilanzbuchhalter|lohnbuchhalter|finanzbuchhalter"
    r"|buchhalter(?:in)?\b"
    # hospitality / food / front-of-house
    r"|\bkoch\b|k[öo]chin|k[üu]chenhilfe|kellner|servicekraft|\bbarista\b"
    r"|bartender|barkeeper|housekeeping|room\s+attendant|\bwaiter\b|waitress"
    r"|sous[- ]chef|chef\s+de\s+(?:cuisine|partie|rang)|commis\s+de"
    r"|hotelfach|restaurantfach|rezeptionist|receptionist|front\s+desk"
    r"|\bconcierge\b|b[äa]cker(?:in)?\b|konditor|metzger|fleischer(?:in)?\b"
    # physical security / cleaning / facilities
    r"|security\s+guard|sicherheitsmitarbeiter|sicherheitsdienst|\bwachmann\b"
    r"|objektschutz|werkschutz|reinigungskraft|reinigungsmitarbeiter"
    r"|geb[äa]udereinig|\bcleaner\b|cleaning\s+operative|\bjanitor\b"
    r"|hausmeister|caretaker"
    # beauty / personal services / cabin crew
    r"|friseur|fris[öo]r|hairdresser|kosmetiker|\bbarber\b"
    r"|flugbegleiter|cabin\s+crew|flight\s+attendant|first\s+officer|copilot",
    re.I)

# --- Senior titles (5+ years territory). "Manager"/"Partner" stay LLM-judged:
# in DACH they often label mid-level individual-contributor roles ("HR Business
# Partner" is a function, not a rank; the user applied to an "Engagement
# Manager" role). "Expert"/"Architect" ARE senior ranks in practice — every
# expert/architect posting the user saw was manually skipped. -----------------
_SENIOR_TITLE = re.compile(
    # \bsr\.? must also hit at end-of-title ("Business Insights Specialist, Sr")
    r"\bsenior\b|\bsr\.?(?=[\s,/)]|$)|\blead\b|\bprincipal\b|\bdistinguished\b|\bfellow\b"
    r"|\bhead\s+of\b|\bhead\b,|,\s*head\b|\bdirector\b|\bdirectrice\b|\bdirektor"
    r"|\bvp\b|\bvice\s+president|\bchief\b|\bmanaging\s+partner\b"
    r"|\bstaff\s+(?:software\s+|data\s+|ml\s+|ai\s+)?(?:engineer|scientist|developer)"
    # -leiter/-leitung ANYWHERE in a compound (Werkstattleiter, Filialleiter,
    # Bauleiter, Projektleitung… all leadership). Lookbehinds spare the
    # non-leadership -leiter nouns: Begleiter/Begleitung (companion),
    # Halbleiter (semiconductor — "Ingenieur für Leistungshalbleiter"!),
    # Wellenleiter (waveguide), Blitzableiter/Ableiter (arrester),
    # Rohr-/Freileitung (pipe / overhead line).
    r"|(?<!beg)(?<!halb)(?<!wellen)(?<!ab)(?<!rohr)(?<!frei)leit(?:er(?:in)?|ung)\b"
    r"|\bgesch[aä]ftsf[uü]hr|\bvorstand\b"
    r"|\bexpert(?:e|in)?\b|\barchitect\b|\barchitekt(?:in)?\b",
    re.I)

# --- "requires N+ years" in EN/DE ---------------------------------------------
# Strong, unambiguous phrasings ("at least/minimum/N+") always count. Plain
# statements ("5 years of professional experience") count unless softened
# ("ideally", "preferred", "a plus", "von Vorteil" ...) in the same breath.
_SPELLED_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "twenty": 20,
    "ein": 1, "zwei": 2, "drei": 3, "vier": 4, "fünf": 5, "fuenf": 5,
    "sechs": 6, "sieben": 7, "acht": 8, "neun": 9, "zehn": 10, "elf": 11,
    "zwölf": 12, "zwoelf": 12, "fünfzehn": 15, "fuenfzehn": 15, "zwanzig": 20,
}
_SPELLED_RE = "|".join(_SPELLED_NUM)
_YEARS_STRONG = [
    re.compile(r"(?:at\s+least|minimum(?:\s+of)?|min\.?)\s+(\d{1,2})\s*\+?\s*years?", re.I),
    re.compile(r"(\d{1,2})\s*\+\s*(?:years?|yrs?|jahren?)", re.I),
    re.compile(r"(\d{1,2})\s+or\s+more\s+years?", re.I),
    re.compile(r"(?:mindestens|mind\.?)\s+(\d{1,2})\s*(?:\+\s*)?jahren?", re.I),
    re.compile(r"(\d{1,2})\s*(?:-|–|bis)\s*\d{1,2}\s+(?:years?|jahren?)\s+(?:of\s+)?"
               r"(?:relevant\s+|professional\s+|work\s+|berufs)?(?:experience|erfahrung)", re.I),
]
# Plain requirement statements — the dominant phrasing in real ads, previously
# missed entirely (root cause of several 5+-years roles reaching the board).
_YEARS_PLAIN = [
    # (?<![-–\d]) — never grab the upper bound of a "5-7 years" range; the
    # dedicated range pattern above already counts its minimum.
    re.compile(r"(?<![-–\d])(\d{1,2})\s*\+?\s*years?['’]?\s+(?:of\s+)?(?:\w+[- ]){0,3}?"
               r"(?:experience|track record|berufserfahrung)", re.I),
    re.compile(r"(?<![-–\d])(\d{1,2})\s*\+?\s*jahren?\s+(?:\w+[- ]){0,3}?"
               r"(?:berufserfahrung|erfahrung|praxis)", re.I),
    re.compile(r"experience\s+of\s+(\d{1,2})\s*\+?\s*years?", re.I),
    re.compile(rf"\b({_SPELLED_RE})\s+years?['’]?\s+(?:of\s+)?(?:\w+[- ]){{0,3}}?experience", re.I),
    re.compile(rf"\b({_SPELLED_RE})\s+jahren?\s+(?:\w+[- ]){{0,3}}?(?:berufserfahrung|erfahrung)", re.I),
]
_YEARS_SOFTENER = re.compile(
    r"prefer|ideal|a\s+plus|nice[- ]to[- ]have|desirable|advantage|bonus"
    r"|von\s+vorteil|wünschenswert|w[üu]nschenswert|not\s+required|or\s+equivalent",
    re.I)
MAX_YEARS_REQUIRED = 4  # 5+ years required -> excluded per the profile

# "mindestens 24 Jahre alt" / "at least 21 years of age" are AGE requirements
# (drivers, on-premise reps), not experience — never count them.
_AGE_NOT_EXPERIENCE = re.compile(r"^\s*(?:alt\b|old\b|of\s+age\b)", re.I)


def _years_value(raw: str) -> int:
    raw = raw.lower()
    return _SPELLED_NUM.get(raw, int(raw) if raw.isdigit() else 0)


def required_years(description: str) -> Optional[int]:
    """The strongest explicit years-of-experience requirement, if any."""
    if not description:
        return None
    best: Optional[int] = None

    def consider(m: re.Match, softenable: bool) -> None:
        nonlocal best
        if _AGE_NOT_EXPERIENCE.match(description[m.end():m.end() + 12]):
            return
        if softenable:
            ctx = description[max(0, m.start() - 60):m.end() + 60]
            if _YEARS_SOFTENER.search(ctx):
                return
        n = _years_value(m.group(1))
        if 0 < n < 30:  # ignore "401k", "30+ days" style noise
            best = max(best or 0, n)

    for rx in _YEARS_STRONG:
        for m in rx.finditer(description):
            consider(m, softenable=False)
    for rx in _YEARS_PLAIN:
        for m in rx.finditer(description):
            consider(m, softenable=True)
    return best


# --- Required languages beyond English/German ---------------------------------
# The ad may be written in English yet demand fluent French/Dutch/... — those
# roles are unusable for the candidate and were a major source of bad
# recommendations. Detection is sentence-scoped and conservative:
#   * needs a requirement cue (fluent, native, C1, verhandlungssicher, ...)
#     in the same sentence as the language name;
#   * an "or"-alternative that includes English or German passes
#     ("French, German or Spanish" — satisfiable with German);
#   * softened mentions ("French is a plus") never count.
_OTHER_LANGS: dict[str, re.Pattern] = {
    name: re.compile(rx, re.I) for name, rx in {
        "french": r"\bfrench\b|französisch|\bfrançais\b|\bfrancais\b",
        "dutch": r"\bdutch\b|niederländisch|\bnederlands\b|\bflemish\b",
        "spanish": r"\bspanish\b|spanisch|\bespañol\b|\bespanol\b|\bcastilian\b",
        "italian": r"\bitalian\b|italienisch|\bitaliano\b",
        "swedish": r"\bswedish\b|schwedisch|\bsvenska\b",
        "danish": r"\bdanish\b|dänisch|\bdansk\b",
        "norwegian": r"\bnorwegian\b|norwegisch|\bnorsk\b",
        "finnish": r"\bfinnish\b|finnisch",
        "polish": r"\bpolish\b|polnisch|\bpolski\b",
        "czech": r"\bczech\b|tschechisch",
        "portuguese": r"\bportuguese\b|portugiesisch|\bportuguês\b",
        "russian": r"\brussian\b|russisch",
        "arabic": r"\barabic\b|arabisch",
        "chinese": r"\bchinese\b|\bmandarin\b|chinesisch",
        "japanese": r"\bjapanese\b|japanisch",
        "korean": r"\bkorean\b|koreanisch",
        "turkish": r"\bturkish\b|türkisch",
        "hungarian": r"\bhungarian\b|ungarisch",
        "romanian": r"\bromanian\b|rumänisch",
        "greek": r"\bgreek\b|griechisch",
        "hebrew": r"\bhebrew\b|hebräisch",
    }.items()
}
_LANG_CUE = re.compile(
    r"fluent|fluency|native|proficien|business[- ]level|mother\s+tongue"
    r"|must\s+speak|required|mandatory|essential|\bC1\b|\bC2\b|\bB2\b"
    r"|speaking|speaker|bilingual|written\s+and\s+spoken|verbal\s+and\s+written"
    r"|excellent\s+(?:command|knowledge)|full\s+professional"
    r"|fließend|verhandlungssicher|muttersprach|sehr\s+gute\w*\s+\w*kenntnisse"
    r"|zwingend|erforderlich|vorausgesetzt|voraussetzung",
    re.I)
_LANG_SOFTENER = re.compile(
    r"a\s+plus|nice[- ]to[- ]have|advantage|advantageous|an?\s+asset|preferred"
    r"|desirable|bonus|beneficial|appreciated|von\s+vorteil|wünschenswert"
    r"|ideal(?:erweise)?\b|optional|not\s+required|would\s+be|helpful"
    r"|other\s+languages?|additional\s+language|weitere\s+sprache",
    re.I)
_LANG_ALTERNATIVE = re.compile(r"\bor\b|\boder\b|\beither\b|\bone\s+of\b|\bou\b", re.I)
_EN_DE = re.compile(r"\benglish\b|\bgerman\b|\bdeutsch\b|\benglisch\b", re.I)
_SENTENCE_SPLIT = re.compile(r"[.!?\n;•·|]+")
# Titles carry their own requirement grammar: "(French/English)", "- French
# Market", "French speaker", "with Dutch", "Spanish fluency" all mean the role
# needs that language even without a cue word nearby.
_LANGS_ALT = "|".join(rx.pattern for rx in _OTHER_LANGS.values())
_TITLE_LANG = re.compile(
    rf"\((?:with\s+)?[^)]{{0,40}}?({_LANGS_ALT})[^)]{{0,40}}\)"
    rf"|({_LANGS_ALT})[-\s]+(?:speak|speaker|market|fluency)"
    rf"|with\s+({_LANGS_ALT})\b"
    rf"|\b({_LANGS_ALT})\s*/\s*english", re.I)


def required_languages(title: str, description: str = "") -> list[str]:
    """Languages (beyond EN/DE) the posting requires, e.g. ['french'].

    Empty list when nothing beyond English/German is clearly required."""
    found: set[str] = set()
    t = title or ""
    if _TITLE_LANG.search(t) and not (_EN_DE.search(t) and _LANG_ALTERNATIVE.search(t)):
        for name, rx in _OTHER_LANGS.items():
            if rx.search(t):
                found.add(name)
    text = f"{t}\n{description or ''}"
    for sent in _SENTENCE_SPLIT.split(text):
        if len(sent) > 400:  # not a requirements bullet; too noisy to judge
            continue
        hits = [name for name, rx in _OTHER_LANGS.items() if rx.search(sent)]
        if not hits:
            continue
        if not _LANG_CUE.search(sent):
            continue
        if _LANG_SOFTENER.search(sent):
            continue
        # "German or French" — satisfiable with a language the candidate has.
        if _EN_DE.search(sent) and _LANG_ALTERNATIVE.search(sent):
            continue
        found.update(hits)
    return sorted(found)


# --- Fixed-term / temporary contracts ------------------------------------------
# Not a hard exclusion (the user applied to one temporary OECD role), but a
# consistent negative: 10+ fixed-term roles were manually skipped. Signals a
# prefilter penalty and an explicit LLM watch-out.
# In a TITLE, a bare "11 Months" / "12-month" is a contract length; in a
# DESCRIPTION it may be a notice period or onboarding phase, so there the
# month-count must sit right next to a contract word.
_FIXED_TERM_TITLE = re.compile(
    r"fixed[- ]term|\btemporary\b|\btemp\b(?!late)|\bbefristet\w*|\binterim\b"
    r"|maternity\s+cover|parental\s+leave\s+cover"
    r"|\b\d{1,2}[- ](?:month|monate)s?\b"
    r"|limited\s+(?:full[- ]time|contract|duration)|zeitvertrag|temp\s+until"
    r"|\bftc\b|contrat\s+à\s+durée\s+déterminée",
    re.I)
_FIXED_TERM_DESC = re.compile(
    r"fixed[- ]term|\bbefristet\w*|maternity\s+cover|parental\s+leave\s+cover"
    r"|\b\d{1,2}[- ](?:month|monate)s?\s+(?:contract|cover|assignment|position"
    r"|role|vertrag)|temporary\s+(?:contract|position|role|assignment)"
    r"|limited\s+(?:full[- ]time|contract|duration)|zeitvertrag|temp\s+until"
    r"|contrat\s+à\s+durée\s+déterminée",
    re.I)


def is_fixed_term(title: str, description: str = "") -> bool:
    """True when the posting clearly advertises a temporary / fixed-term role."""
    if _FIXED_TERM_TITLE.search(title or ""):
        return True
    return bool(_FIXED_TERM_DESC.search((description or "")[:2000]))


# --- Target region, straight from data/profile/preferences.txt:
# Austria, Germany, England (UK), Ireland, Sweden, Switzerland,
# Spain (Barcelona only), France, Netherlands. --------------------------------
_TARGET_COUNTRIES = [
    "austria", "österreich", "germany", "deutschland", "united kingdom",
    "great britain", "england", "scotland", "wales", "u.k.", "ireland",
    "sweden", "sverige", "switzerland", "schweiz", "suisse", "svizzera",
    "france", "netherlands", "nederland", "holland",
]
_TARGET_CITIES = [
    # AT
    "vienna", "wien", "linz", "graz", "salzburg", "innsbruck", "klagenfurt",
    # DE
    "berlin", "munich", "münchen", "muenchen", "hamburg", "frankfurt",
    "cologne", "köln", "koeln", "stuttgart", "düsseldorf", "duesseldorf",
    "leipzig", "dresden", "nuremberg", "nürnberg", "hannover", "bremen",
    "bonn", "essen", "dortmund", "karlsruhe", "mannheim", "heidelberg",
    "ingolstadt", "wolfsburg", "walldorf", "böblingen", "erlangen", "jena",
    "regensburg", "ludwigshafen", "leverkusen", "aachen", "münster",
    # UK
    "london", "manchester", "birmingham", "leeds", "bristol", "cambridge",
    "oxford", "edinburgh", "glasgow", "belfast", "reading", "milton keynes",
    # IE
    "dublin", "cork", "galway", "limerick",
    # SE
    "stockholm", "gothenburg", "göteborg", "malmö", "malmo", "lund", "uppsala",
    # CH
    "zurich", "zürich", "zuerich", "geneva", "genève", "genf", "basel", "bern",
    "lausanne", "zug", "lucerne", "luzern", "winterthur", "baden", "kaiseraugst",
    # FR
    "paris", "lyon", "toulouse", "lille", "marseille", "nantes", "bordeaux",
    "grenoble", "nice", "strasbourg", "sophia antipolis", "rueil", "clichy",
    # NL
    "amsterdam", "rotterdam", "utrecht", "eindhoven", "the hague", "den haag",
    "veldhoven", "rijswijk", "leiden", "delft", "groningen", "arnhem", "breda",
    "hilversum", "schiphol", "amstelveen",
    # ES — Barcelona only, per the profile
    "barcelona",
]
_TARGET_CODES = {"AT", "AUT", "DE", "DEU", "GB", "GBR", "UK", "IE", "IRL",
                 "SE", "SWE", "CH", "CHE", "FR", "FRA", "NL", "NLD"}
# Country codes that are definitely OUTSIDE the target region (checked on
# uppercase tokens in location strings, e.g. "Aartselaar, BE"). ES is absent
# on purpose: Spain is Barcelona-only, so a bare "ES" needs the city check.
_ELSEWHERE_CODES = {
    "BE", "BEL", "IT", "ITA", "PL", "POL", "PT", "PRT", "DK", "DNK", "NO",
    "NOR", "FI", "FIN", "CZ", "CZE", "HU", "HUN", "RO", "ROU", "SK", "SVK",
    "SI", "SVN", "HR", "HRV", "BG", "BGR", "GR", "GRC", "LT", "LTU", "LV",
    "LVA", "EE", "EST", "LU", "LUX", "RS", "SRB", "UA", "UKR", "TR", "TUR",
    "US", "USA", "CA", "CAN", "MX", "MEX", "BR", "BRA", "AR", "ARG", "CL",
    "CHL", "CO", "COL", "PE", "PER", "IN", "IND", "CN", "CHN", "JP", "JPN",
    "KR", "KOR", "TW", "TWN", "HK", "HKG", "SG", "SGP", "MY", "MYS", "TH",
    "THA", "ID", "IDN", "PH", "PHL", "VN", "VNM", "AU", "AUS", "NZ", "NZL",
    "AE", "ARE", "SA", "SAU", "QA", "QAT", "IL", "ISR", "EG", "EGY", "ZA",
    "ZAF", "NG", "NGA", "KE", "KEN", "MA", "MAR",
    # CIS / Caucasus / Central Asia (Novartis Moscow & ICRC Kyiv postings
    # reached the LLM queue before 2026-07-09). NB: no "GE" (Geneva canton),
    # no "AM" and no "BY" (Bavaria) — 3-letter codes only where 2-letter
    # would collide with target-region strings.
    "RU", "RUS", "KZ", "KAZ", "AZE", "GEO", "ARM", "UZB", "TJK", "KGZ", "TKM",
    # Africa / Middle East / South & SE Asia / Latin America / Balkans
    "TUN", "DZA", "SEN", "GHA", "ETH", "TZA", "UGA", "CMR", "CIV", "COD",
    "COG", "AGO", "MOZ", "ZMB", "ZWE", "NAM", "SDN", "SOM", "YEM", "KWT",
    "BHR", "OMN", "IRQ", "IRN", "LBN", "JOR", "PAK", "BGD", "LKA", "NPL",
    "MMR", "KHM", "LAO", "MNG", "CUB", "DOM", "HTI", "ECU", "BOL", "PRY",
    "VEN", "URY", "CRI", "GTM", "PAN", "HND", "NIC", "SLV", "ALB", "MKD",
    "BIH", "MDA", "MNE",
}
_REMOTE_EU = re.compile(
    r"remote[\s,–-]*(?:in\s+)?(?:eu|europe|emea)|(?:eu|europe|emea)[\s,–-]*remote", re.I)

# Signals that a role clearly sits OUTSIDE the target region.
_US_STATE_RE = re.compile(
    r"(?:^|[,\s(])u\.?s\.?a?\.?(?:[),\s\-]|$)"      # standalone US / USA token
    r"|\bus-[a-z]{2}\b|\bunited states\b"
    r"|\b(?:alabama|alaska|arizona|arkansas|california|colorado|connecticut|"
    r"delaware|florida|georgia|hawaii|idaho|illinois|indiana|iowa|kansas|"
    r"kentucky|louisiana|maine|maryland|massachusetts|michigan|minnesota|"
    r"mississippi|missouri|montana|nebraska|nevada|new hampshire|new jersey|"
    r"new mexico|new york|north carolina|north dakota|ohio|oklahoma|oregon|"
    r"pennsylvania|rhode island|south carolina|south dakota|tennessee|texas|"
    r"utah|vermont|virginia|washington|west virginia|wisconsin|wyoming)\b", re.I)
# "Cambridge, MA" — a US state code after a comma (case-sensitive; runs only
# after the target-country checks, so "Amsterdam, NH, NL" never gets here).
_US_STATE_CODE_RE = re.compile(
    r",\s*(?:AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|"
    r"MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|"
    r"VT|VA|WA|WV|WI|WY)\b")
_ELSEWHERE = [
    # non-target Europe
    "italy", "italia", "portugal", "denmark", "danmark", "norway", "norge",
    "finland", "suomi", "poland", "polska", "czech", "czechia", "hungary",
    "romania", "greece", "belgium", "belgique", "belgië", "luxembourg",
    "slovakia", "slovenia", "croatia", "serbia", "bulgaria", "lithuania",
    "latvia", "estonia", "ukraine", "turkey", "türkiye", "malta", "cyprus",
    "iceland", "bosnia", "albania", "macedonia", "montenegro", "kosovo",
    "moldova", "belarus", "minsk",
    # cities that would otherwise slip through (feeds often send a bare city)
    "madrid", "milan", "milano", "rome", "roma", "lisbon", "lisboa", "porto",
    "copenhagen", "københavn", "oslo", "helsinki", "warsaw", "warszawa",
    "krakow", "kraków", "gdansk", "gdańsk", "wroclaw", "wrocław", "poznan",
    "prague", "praha", "brno", "ostrava", "bratislava", "budapest", "debrecen",
    "szeged", "nyíregyháza", "nyiregyhaza", "bucharest", "bucuresti",
    "bucurești", "cluj", "brasov", "brașov", "iasi", "iași", "timisoara",
    "timișoara", "sofia", "belgrade", "zagreb", "ljubljana", "vilnius", "riga",
    "tallinn", "athens", "thessaloniki", "brussels", "bruxelles", "antwerp",
    "ghent", "istanbul", "ankara",
    # Spain outside Barcelona (bare-city feeds!)
    "valencia", "sevilla", "seville", "bilbao", "zaragoza", "málaga", "malaga",
    "alicante", "murcia", "palma", "mallorca", "las palmas", "tenerife",
    "vigo", "gijón", "gijon", "la coruña", "a coruña", "coruna", "granada",
    "valladolid", "pamplona", "santander", "córdoba", "cordoba",
    "san sebastián", "san sebastian",
    # Canada (city-only strings are common on SuccessFactors boards)
    "pointe-claire", "montréal", "quebec", "québec", "ontario", "ottawa",
    "mississauga", "laval", "calgary", "edmonton", "winnipeg", "waterloo",
    "markham", "burnaby",
    # rest of world
    "india", "bangalore", "bengaluru", "hyderabad", "gurugram", "gurgaon",
    "mumbai", "delhi", "pune", "chennai", "noida", "china", "shanghai",
    "beijing", "shenzhen", "hong kong", "japan", "tokyo", "korea", "seoul",
    "singapore", "taiwan", "taipei", "australia", "sydney", "melbourne",
    "new zealand", "canada", "toronto", "vancouver", "montreal", "mexico",
    "cdmx", "brazil", "brasil", "são paulo", "sao paulo", "argentina",
    "chile", "colombia", "peru", "costa rica", "guatemala", "uruguay",
    "israel", "tel aviv", "uae", "dubai", "abu dhabi", "saudi", "riyadh",
    "qatar", "doha", "egypt", "cairo", "south africa", "johannesburg",
    "nigeria", "kenya", "nairobi", "morocco", "philippines", "manila",
    "indonesia", "jakarta", "malaysia", "kuala lumpur", "vietnam", "thailand",
    "bangkok", "pakistan", "bangladesh", "juba", "kabul", "amman", "geneva il",
    "central african republic", "bangui", "ethiopia", "addis ababa",
    # CIS / Caucasus / Central Asia (observed in the 2026-07-09 queue: Moscow,
    # Kyiv, Almaty, Tashkent, Baku, Yerevan postings all sat at 'unknown')
    "russia", "moscow", "st. petersburg", "saint petersburg", "kyiv", "kiev",
    "kharkiv", "lviv", "odesa", "odessa", "kazakhstan", "almaty", "astana",
    "azerbaijan", "baku", "armenia", "yerevan", "tbilisi", "uzbekistan",
    "tashkent", "kyrgyzstan", "bishkek", "tajikistan", "dushanbe",
    "turkmenistan", "ashgabat",
    # Africa / Middle East (int-org boards list bare duty-station cities)
    "tunisia", "tunis", "algeria", "algiers", "libya", "tripoli", "senegal",
    "dakar", "ghana", "accra", "ivory coast", "côte d'ivoire", "abidjan",
    "cameroon", "yaound", "douala", "tanzania", "dar es salaam", "uganda",
    "kampala", "zambia", "lusaka", "zimbabwe", "harare", "mozambique",
    "maputo", "angola", "luanda", "botswana", "gaborone", "namibia",
    "windhoek", "congo", "kinshasa", "brazzaville", "sudan", "khartoum",
    "somalia", "mogadishu", "yemen", "sana'a", "sanaa", "kuwait", "bahrain",
    "oman", "muscat", "iraq", "baghdad", "erbil", "iran", "tehran", "lebanon",
    "beirut", "casablanca", "rabat",
    # South / SE Asia & Latin America (bare-city strings from global boards)
    "sri lanka", "colombo", "nepal", "kathmandu", "myanmar", "yangon",
    "cambodia", "phnom penh", "laos", "vientiane", "mongolia", "ulaanbaatar",
    "ho chi minh", "hanoi", "panama", "havana", "cuba", "santo domingo",
    "dominican republic", "haiti", "port-au-prince", "ecuador", "quito",
    "bolivia", "la paz", "paraguay", "asunción", "asuncion", "venezuela",
    "caracas", "honduras", "tegucigalpa", "nicaragua", "managua",
    "el salvador",
    # Balkans (bare cities; the countries are already above)
    "tirana", "skopje", "sarajevo", "chisinau", "chișinău", "pristina",
    "podgorica",
    "new york", "san francisco", "seattle", "boston", "chicago", "austin",
    "atlanta", "dallas", "houston", "denver", "phoenix", "menlo park",
    "mountain view", "palo alto", "sunnyvale", "cupertino", "redmond",
]

# _ELSEWHERE entries that name a COUNTRY / STATE / PROVINCE / REGION rather
# than a city. Needed to judge mixed location strings: a region word right
# next to a "target" city usually QUALIFIES that city ("London, Ontario",
# "Barcelona, Venezuela" — outside), while a foreign CITY next to a target
# city means sibling offices ("Stockholm, Eindhoven, Amsterdam, Madrid" — the
# candidate simply picks the target office; that posting is target). A real
# Philips posting was lost to exactly this on the 2026-07 queue.
_ELSEWHERE_REGIONS = frozenset({
    "italy", "italia", "portugal", "denmark", "danmark", "norway", "norge",
    "finland", "suomi", "poland", "polska", "czech", "czechia", "hungary",
    "romania", "greece", "belgium", "belgique", "belgië", "luxembourg",
    "slovakia", "slovenia", "croatia", "serbia", "bulgaria", "lithuania",
    "latvia", "estonia", "ukraine", "turkey", "türkiye", "malta", "cyprus",
    "iceland", "bosnia", "albania", "macedonia", "montenegro", "kosovo",
    "moldova", "belarus",
    "quebec", "québec", "ontario",
    "india", "china", "hong kong", "japan", "korea", "singapore", "taiwan",
    "australia", "new zealand", "canada", "mexico", "brazil", "brasil",
    "argentina", "chile", "colombia", "peru", "costa rica", "guatemala",
    "uruguay", "israel", "uae", "saudi", "qatar", "egypt", "south africa",
    "nigeria", "kenya", "morocco", "philippines", "indonesia", "malaysia",
    "vietnam", "thailand", "pakistan", "bangladesh", "sri lanka", "nepal",
    "myanmar", "cambodia", "laos", "mongolia", "cuba", "dominican republic",
    "haiti", "ecuador", "bolivia", "paraguay", "venezuela", "honduras",
    "nicaragua", "el salvador", "panama", "russia", "kazakhstan",
    "azerbaijan", "armenia", "uzbekistan", "kyrgyzstan", "tajikistan",
    "turkmenistan", "tunisia", "algeria", "libya", "senegal", "ghana",
    "ivory coast", "côte d'ivoire", "cameroon", "tanzania", "uganda",
    "zambia", "zimbabwe", "mozambique", "angola", "botswana", "namibia",
    "congo", "sudan", "somalia", "yemen", "kuwait", "bahrain", "oman",
    "iraq", "iran", "lebanon", "central african republic", "ethiopia",
})

# Boundary-aware matcher for counting DISTINCT target cities in a mixed
# location string (substring checks would count "nice" inside "venice").
_TARGET_CITY_RX = re.compile(
    r"(?<![a-zäöüßéè])(?:"
    + "|".join(sorted((re.escape(c) for c in _TARGET_CITIES),
                      key=len, reverse=True))
    + r")(?![a-zäöüßéè])")


def location_status(location: str) -> str:
    """'target' | 'outside' | 'unknown' for a feed location string.

    Multi-location postings count as target when ANY listed site is a target
    (the candidate can choose that office). Empty/ambiguous strings are
    'unknown' and left for the LLM to judge with full context."""
    loc = (location or "").strip().lower()
    if not loc:
        return "unknown"
    # 1) explicit target countries / country codes — the strongest signal.
    if any(c in loc for c in _TARGET_COUNTRIES):
        return "target"
    for tok in re.split(r"[^A-Za-z]+", location):
        if tok and tok.isupper() and tok in _TARGET_CODES:
            return "target"
    # 1b) "Barcelona, CT, Spain" — Catalonia's 'CT' is not Connecticut. The
    #     profile's one Spanish target must beat the US-state-code heuristic
    #     below (8 real Barcelona postings were dropped this way on the
    #     2026-07-09 run). Requires an explicit Spain/Catalonia marker so
    #     "Barcelona, Venezuela" still lands outside.
    if "barcelona" in loc and any(w in loc for w in
                                  ("spain", "españa", "espana", "catalonia",
                                   "cataluña", "catalunya", "cataluna")):
        return "target"
    # 2) explicit outside signals BEFORE bare city names, so "Cambridge, MA"
    #    or "Vienna, VA" can't ride on their European namesakes.
    if _US_STATE_RE.search(location) or _US_STATE_CODE_RE.search(location):
        return "outside"
    for tok in re.split(r"[^A-Za-z]+", location):
        if tok and tok.isupper() and tok in _ELSEWHERE_CODES:
            return "outside"
    # 2b) curated elsewhere words. Mixed strings need care (multi-location
    #     postings count as target when ANY listed site is a target):
    #     * a foreign REGION word qualifies an adjacent city name
    #       ("London, Ontario" → outside) unless SEVERAL distinct target
    #       cities are listed alongside it;
    #     * foreign CITIES next to a target city are sibling offices
    #       ("Stockholm, Eindhoven, Amsterdam, Madrid" → target).
    hits = [w for w in _ELSEWHERE if w in loc]
    if hits:
        n_target_cities = len(set(_TARGET_CITY_RX.findall(loc)))
        region_hit = any(w in _ELSEWHERE_REGIONS for w in hits)
        if region_hit:
            return "target" if n_target_cities >= 2 else "outside"
        return "target" if n_target_cities >= 1 else "outside"
    # 3) bare target cities (feeds often send just "Wien" or "London").
    if any(c in loc for c in _TARGET_CITIES):
        return "target"
    if "spain" in loc or "españa" in loc:      # Spain outside Barcelona
        return "outside"
    if _REMOTE_EU.search(loc):
        return "target"
    return "unknown"


def posting_age_days(posted_date: Optional[str],
                     first_seen: Optional[str] = None) -> Optional[int]:
    """Days since a posting went up — from its published date when the site
    provides one, else from the day JobScout first saw it. None when neither
    parses (such postings are never age-filtered)."""
    from datetime import date
    for raw in (posted_date, first_seen):
        if not raw:
            continue
        try:
            return (date.today() - date.fromisoformat(str(raw)[:10])).days
        except ValueError:
            continue
    return None


# =============================================================================
# Hard-filter configuration (managed from the dashboard's Hard Filters tab)
# =============================================================================
# The RULES above are code; this layer decides which of them run and with
# which parameters, plus user-created keyword filters. Overrides live in the
# `hard_filter` table (db.py). With no rows there — or no DB at all — the
# defaults below apply and `hard_filter()` behaves exactly as always.

# Each entry documents its rule for the dashboard: `preview` = the short chip
# row in the filter table; `details` = the full "what this looks for" groups
# shown in the Edit dialog. They are curated mirrors of the regexes above —
# when a pattern learns a new word, add it here too.
BUILTIN_HARD_FILTERS: list[dict] = [
    {"key": "years_required", "name": "Experience ceiling",
     "what": "Descriptions that clearly require more years of professional "
             "experience than the ceiling allows.",
     "editable": ("max_years",), "extra_field": None,
     "preview": [], "details": [
         {"label": "Phrases it understands (EN + DE)",
          "terms": ["at least N years", "minimum N years", "N+ years",
                    "N or more years", "N–M years experience",
                    "mindestens N Jahre", "N Jahre Berufserfahrung",
                    "spelled numbers ('five years of experience')"]},
         {"label": "Never counts",
          "terms": ["softened asks ('ideally', 'a plus', 'von Vorteil', "
                    "'or equivalent')", "age requirements ('21 years of age')"]}]},
    {"key": "age", "name": "Posting age limit",
     "what": "Drops postings older than the ceiling (published date when the "
             "feed provides one, else the day JobScout first saw them).",
     "editable": ("max_days",), "extra_field": None,
     "preview": [], "details": []},
    {"key": "early_career", "name": "Internships & student roles",
     "what": "Internships, working-student jobs, thesis projects, "
             "apprenticeships, PhD/postdoc, volunteers and similar "
             "early-career formats (EN + DE patterns, incl. German "
             "compounds like 'Pflichtpraktikum').",
     "editable": (), "extra_field": "title",
     "preview": ["internship", "werkstudent", "praktikum", "thesis",
                 "ausbildung", "phd"],
     "details": [
         {"label": "Internships & working students",
          "terms": ["internship", "intern", "praktikum", "praktikant",
                    "werkstudent", "working student", "studentische hilfskraft",
                    "student assistant", "student worker", "studentenjob",
                    "summer intern", "summer job", "ferialjob", "ferienjob"]},
         {"label": "Apprenticeships & dual study",
          "terms": ["apprentice", "lehrling", "lehrstelle", "lehre zum/zur/als",
                    "ausbildung", "azubi", "duales studium", "dual study"]},
         {"label": "Thesis & academia",
          "terms": ["thesis", "abschlussarbeit", "masterarbeit",
                    "bachelorarbeit", "diplomarbeit", "masterand", "bachelorand",
                    "doktorand", "phd", "doctoral", "postdoc"]},
         {"label": "Other early-career formats",
          "terms": ["volunteer", "volontär", "volontariat", "stagiaire",
                    "alternance", "co-op", "gap year", "becario"]}]},
    {"key": "trainee", "name": "Trainee programmes",
     "what": "Structured trainee / traineeship programmes.",
     "editable": (), "extra_field": "title",
     "preview": ["trainee", "traineeship", "trainee-programm"],
     "details": [
         {"label": "Title words",
          "terms": ["trainee", "traineeship", "trainee-programm"]}]},
    {"key": "retail_store", "name": "In-store retail roles",
     "what": "Store staff: sales associates, cashiers, store managers, "
             "visual merchandisers, seasonal floor jobs. Corporate roles "
             "about the retail business survive.",
     "editable": (), "extra_field": "title",
     "preview": ["sales associate", "cashier", "store manager", "verkäufer",
                 "outlet", "seasonal"],
     "details": [
         {"label": "Store staff",
          "terms": ["sales assistant", "sales associate", "store associate",
                    "retail associate", "retail assistant", "shop assistant",
                    "cashier", "kassier", "stockroom", "verkäufer(in)",
                    "einzelhandel"]},
         {"label": "Store management & floor",
          "terms": ["store manager", "floor manager", "store lead",
                    "store supervisor", "store coach", "store crew",
                    "store team member", "filialleitung", "filialmitarbeiter"]},
         {"label": "Merchandising & outlet",
          "terms": ["visual merchandising", "merchandiser", "flagship",
                    "outlet", "factory outlet", "retail store", "in-store"]},
         {"label": "Brand store titles & casual work",
          "terms": ["athlete (Nike)", "genius (Apple)", "aushilfe", "minijob",
                    "geringfügig", "saisonkraft", "seasonal"]}]},
    {"key": "off_profile", "name": "Off-profile occupations",
     "what": "Trades, drivers, medical/care, legal practice, hospitality, "
             "cleaning, security… — occupations with zero overlap with the "
             "CV. Titles naming analytics itself always survive.",
     "editable": (), "extra_field": "title",
     "preview": ["mechatroniker", "techniker", "fahrer/driver", "nurse/pflege",
                 "rechtsanwalt", "koch/kellner"],
     "details": [
         {"label": "Vehicles, workshop & manufacturing trades",
          "terms": ["kfz", "karosserie", "lackierer", "mechatroniker",
                    "mechaniker", "mechanic", "zerspanung", "schweißer",
                    "schlosser", "elektriker", "elektroniker", "elektroinstall…",
                    "metallbauer", "werkzeugmacher", "…monteur", "…techniker",
                    "technician", "electrician", "welder", "machinist",
                    "toolmaker", "fitter", "assembler"]},
         {"label": "Production floor",
          "terms": ["facharbeiter", "produktionsmitarbeiter",
                    "fertigungsmitarbeiter", "montagemitarbeiter",
                    "produktionshelfer", "maschinenbediener", "maschinenführer",
                    "anlagenführer", "anlagenbediener", "machine operator",
                    "production operator", "process operator", "plant operator",
                    "production worker"]},
         {"label": "Construction, installation & crafts",
          "terms": ["tischler", "schreiner", "zimmerer", "dachdecker", "maurer",
                    "betonbauer", "trockenbau", "installateur", "klempner",
                    "spengler", "sanitär", "heizungsbau", "kälteanlagen",
                    "maler", "plumber", "carpenter", "gerüstbauer", "gärtner",
                    "landschaftsbau", "forstwirt", "landwirt"]},
         {"label": "Drivers & warehouse floor",
          "terms": ["staplerfahrer", "gabelstapler", "forklift",
                    "lagermitarbeiter", "lagerist", "kommissionier…",
                    "warehouse operative/associate/worker", "order picker",
                    "…fahrer", "driver", "chauffeur", "courier", "zusteller",
                    "postbote"]},
         {"label": "Medical, care, education & social",
          "terms": ["krankenpflege…", "pflegefachkraft", "pflegekraft",
                    "pflegehelfer", "altenpflege…", "nurse", "nursing",
                    "physician", "arzt", "ärztin", "zahnarzt", "apotheker",
                    "pharmacist", "physiotherap…", "ergotherap…", "logopäd…",
                    "hebamme", "toxicolog…", "laborant", "radiolog…",
                    "caregiver", "erzieher", "kindergartenpädagog…",
                    "kinderbetreu…", "sozialarbeiter", "sozialpädagog…",
                    "lehrkraft"]},
         {"label": "Legal practice & tax-clerical professions",
          "terms": ["lawyer", "attorney", "barrister", "solicitor", "jurist",
                    "rechtsanwalt", "syndikus", "notar", "paralegal", "counsel",
                    "finanzwirt", "steuerberater", "steuerfachangestellte",
                    "steuerfachwirt", "bilanzbuchhalter", "lohnbuchhalter",
                    "finanzbuchhalter", "buchhalter"]},
         {"label": "Hospitality, food & front-of-house",
          "terms": ["koch", "köchin", "küchenhilfe", "kellner", "servicekraft",
                    "barista", "bartender", "barkeeper", "housekeeping",
                    "room attendant", "waiter", "waitress", "sous-chef",
                    "chef de cuisine/partie/rang", "hotelfach", "restaurantfach",
                    "rezeptionist", "receptionist", "front desk", "concierge",
                    "bäcker", "konditor", "metzger", "fleischer"]},
         {"label": "Security, cleaning & facilities",
          "terms": ["security guard", "sicherheitsmitarbeiter",
                    "sicherheitsdienst", "wachmann", "objektschutz",
                    "werkschutz", "reinigungskraft", "gebäudereinigung",
                    "cleaner", "janitor", "hausmeister", "caretaker"]},
         {"label": "Personal services & cabin crew",
          "terms": ["friseur", "hairdresser", "kosmetiker", "barber",
                    "flugbegleiter", "cabin crew", "flight attendant",
                    "first officer", "copilot"]},
         {"label": "Guard rail",
          "terms": ["titles naming analytics itself (e.g. 'Analyst "
                    "Fahrerassistenzsysteme') always survive"]}]},
    {"key": "senior_title", "name": "Senior-level titles",
     "what": "Senior / Lead / Principal / Head of / Director / VP / C-level "
             "titles, plus German -leiter/-leitung compounds.",
     "editable": (), "extra_field": "title",
     "preview": ["senior", "lead", "principal", "head of", "director",
                 "…leiter/…leitung"],
     "details": [
         {"label": "Rank keywords",
          "terms": ["senior", "sr.", "lead", "principal", "distinguished",
                    "fellow", "staff engineer/scientist/developer", "expert",
                    "architect", "architekt"]},
         {"label": "Leadership titles",
          "terms": ["head of", "director", "direktor", "vp", "vice president",
                    "chief (C-level)", "managing partner", "geschäftsführer",
                    "vorstand"]},
         {"label": "German compounds",
          "terms": ["…leiter / …leitung anywhere in the word (Teamleiter, "
                    "Projektleitung, Werkstattleiter …)",
                    "not: Begleiter, Halbleiter, Wellenleiter, Ableiter, "
                    "Rohrleitung, Freileitung"]},
         {"label": "Judged by the LLM instead",
          "terms": ["manager", "partner (often mid-level IC roles in DACH)"]}]},
    {"key": "location", "name": "Outside target countries",
     "what": "Locations clearly outside the target region. Ambiguous "
             "locations pass and are judged by the LLM with full context.",
     "editable": (), "extra_field": "location",
     "preview": ["outside AT · DE · UK · IE · SE · CH · FR · NL · Barcelona"],
     "details": [
         {"label": "Kept — target region",
          "terms": ["Austria", "Germany", "United Kingdom", "Ireland",
                    "Sweden", "Switzerland", "France", "Netherlands",
                    "Barcelona (Spain)", "remote in EU/Europe/EMEA",
                    "multi-city listings that include a target office "
                    "(e.g. 'Stockholm, Amsterdam, Madrid')"]},
         {"label": "Dropped — clearly elsewhere (curated lists)",
          "terms": ["US states & cities", "Canada", "non-target Europe "
                    "(Italy, Poland, Denmark, Belgium, …)", "Spain outside "
                    "Barcelona", "CIS & Central Asia", "Middle East & Africa",
                    "South & SE Asia", "Latin America", "country codes "
                    "(US, IN, CN, BE, PL, …)"]},
         {"label": "Kept for the LLM to judge",
          "terms": ["empty or ambiguous location strings"]}]},
    {"key": "other_languages", "name": "Requires other languages",
     "what": "Roles requiring fluency in a language beyond English/German. "
             "Softened mentions ('a plus') and satisfiable alternatives "
             "('German or French') never count.",
     "editable": (), "extra_field": None,
     "preview": [n.capitalize() for n in sorted(_OTHER_LANGS)][:6],
     "details": [
         {"label": "Languages detected",
          "terms": [n.capitalize() for n in sorted(_OTHER_LANGS)]},
         {"label": "Requirement cues (same sentence)",
          "terms": ["fluent", "native", "C1/C2/B2", "business-level",
                    "mother tongue", "required", "mandatory",
                    "verhandlungssicher", "muttersprache", "fließend"]},
         {"label": "Never counts",
          "terms": ["'a plus' / 'von Vorteil' / 'nice to have'",
                    "alternatives satisfiable with EN/DE "
                    "('French, German or Spanish')"]}]},
    {"key": "language", "name": "Not in English or German",
     "what": "Postings written in a language the candidate does not apply "
             "in, detected from the title + description text.",
     "editable": (), "extra_field": None,
     "preview": ["posting text must be English or German"],
     "details": [
         {"label": "Accepted posting languages",
          "terms": ["English", "German"]}]},
]
_BUILTIN_BY_KEY = {f["key"]: f for f in BUILTIN_HARD_FILTERS}
CUSTOM_FIELDS = ("title", "location", "description", "title+description")
CUSTOM_REASON_PREFIX = "custom filter: "

# Which detail groups may take user-added terms ("addable"): only groups that
# LIST EXCLUSIONS on a filter that supports extra terms. Informational groups
# ("Kept — target region", guard rails, …) must never take additions — a word
# added to "Kept" would EXCLUDE it, the exact opposite of what it reads as.
_NON_ADDABLE_GROUPS = {
    "Guard rail", "Judged by the LLM instead", "Kept — target region",
    "Kept for the LLM to judge",
}
for _f in BUILTIN_HARD_FILTERS:
    for _g in _f.get("details") or []:
        _g["addable"] = bool(_f.get("extra_field")) \
            and _g["label"] not in _NON_ADDABLE_GROUPS


def builtin_extra_terms(params: dict) -> list[str]:
    """Every user-added term for a built-in filter, flat: the legacy
    `extra_terms` list plus all per-category `extra_groups` values. Categories
    organize the terms for humans — matching treats them all the same."""
    flat = list(params.get("extra_terms") or [])
    for terms in (params.get("extra_groups") or {}).values():
        flat.extend(terms or [])
    return flat


def builtin_spec(key: str) -> Optional[dict]:
    return _BUILTIN_BY_KEY.get(key)


def default_filter_enabled(key: str) -> bool:
    """Built-in default when the user never touched a filter (config-driven,
    so JOBSCOUT_INCLUDE_TRAINEE keeps working until a dashboard override)."""
    if key == "trainee":
        return not config.INCLUDE_TRAINEE
    return True


def default_filter_params(key: str) -> dict:
    if key == "age":
        return {"max_days": config.MAX_POSTING_AGE_DAYS}
    if key == "years_required":
        return {"max_years": MAX_YEARS_REQUIRED}
    return {}


def _word_terms_regex(terms: Sequence[str]) -> Optional[re.Pattern]:
    """One case-insensitive regex hitting any term on word boundaries."""
    parts = [t for t in (str(t).strip() for t in terms or []) if t]
    if not parts:
        return None
    return re.compile("|".join(rf"(?<!\w){re.escape(p)}(?!\w)" for p in parts),
                      re.I)


@dataclass
class CustomFilter:
    key: str
    name: str
    field: str                    # title | location | description | title+description
    match: str                    # word | contains
    terms: list[str]
    enabled: bool
    rx: Optional[re.Pattern] = None

    def hits(self, title: str, location: str, description: str) -> bool:
        text = {"title": title, "location": location,
                "description": description,
                "title+description": f"{title}\n{description}"}.get(self.field,
                                                                    title)
        if not text:
            return False
        if self.match == "contains":
            low = text.lower()
            return any(t.lower() in low for t in self.terms)
        return bool(self.rx and self.rx.search(text))


def _load_filter_config() -> dict:
    """Read the hard_filter table into an evaluated shape. Missing DB or
    table (unit tests, first start) simply means 'defaults'."""
    cfg: dict = {"enabled": {}, "params": {}, "extra_rx": {}, "custom": []}
    try:
        from . import db
        rows = db.get_hard_filters()
    except Exception:  # noqa: BLE001 — no DB is a legal state here
        return cfg
    for r in rows:
        try:
            params = json.loads(r["params"] or "{}")
        except ValueError:
            params = {}
        if not isinstance(params, dict):
            params = {}
        if r["builtin"]:
            key = r["key"]
            if key not in _BUILTIN_BY_KEY:
                continue                          # row from a newer version
            if r["enabled"] is not None:
                cfg["enabled"][key] = bool(r["enabled"])
            cfg["params"][key] = params
            rx = _word_terms_regex(builtin_extra_terms(params))
            if rx is not None:
                cfg["extra_rx"][key] = rx
        else:
            terms = [str(t).strip() for t in (params.get("terms") or [])
                     if str(t).strip()]
            if not terms:
                continue
            cf = CustomFilter(
                key=r["key"], name=r["name"] or r["key"],
                field=params.get("field") or "title",
                match=params.get("match") or "word", terms=terms,
                enabled=True if r["enabled"] is None else bool(r["enabled"]))
            if cf.match != "contains":
                cf.rx = _word_terms_regex(terms)
            cfg["custom"].append(cf)
    return cfg


_filter_cache: dict = {"path": None, "at": 0.0, "cfg": None}
_FILTER_CACHE_TTL = 3.0    # seconds — a dashboard edit lands within one poll


def filter_config(refresh: bool = False) -> dict:
    """The active hard-filter configuration, briefly cached.

    Cache key = resolved DB path (tests point JOBSCOUT_DB at throwaway
    files); the short TTL keeps the per-posting cost at a dict lookup while
    dashboard edits still apply promptly. Writers call
    `invalidate_filter_cache()` for immediate effect."""
    path = str(os.environ.get("JOBSCOUT_DB") or config.DB_PATH)
    now = time.time()
    c = _filter_cache
    if (not refresh and c["cfg"] is not None and c["path"] == path
            and now - c["at"] < _FILTER_CACHE_TTL):
        return c["cfg"]
    cfg = _load_filter_config()
    c.update(path=path, at=now, cfg=cfg)
    return cfg


def invalidate_filter_cache() -> None:
    _filter_cache.update(path=None, at=0.0, cfg=None)


def filter_enabled(key: str, cfg: Optional[dict] = None) -> bool:
    cfg = cfg if cfg is not None else filter_config()
    v = cfg["enabled"].get(key)
    return default_filter_enabled(key) if v is None else v


def filter_param(key: str, name: str, cfg: Optional[dict] = None):
    cfg = cfg if cfg is not None else filter_config()
    v = (cfg["params"].get(key) or {}).get(name)
    return default_filter_params(key).get(name) if v is None else v


def _extra_hit(key: str, text: str, cfg: dict) -> bool:
    """User-added extra terms for a built-in filter (only ever ADD coverage)."""
    rx = cfg["extra_rx"].get(key)
    return bool(rx and text and rx.search(text))


def _title_rule_hit(key: str, rx: re.Pattern, title: str, cfg: dict) -> bool:
    if not filter_enabled(key, cfg):
        return False
    return bool(rx.search(title) or _extra_hit(key, title, cfg))


_YEARS_REASON_RE = re.compile(r"^requires \d{1,2}\+ years")


def filter_key_for_reason(reason: Optional[str]) -> Optional[str]:
    """Map a stored `filtered_reason` back to the filter that produced it.

    Returns a builtin key, 'custom:<name>' for user filters, or None for
    verdicts that are NOT hard filters (duplicates, LLM floor scores, the
    final review, queue aging) — those must never be re-released or counted
    by filter management."""
    r = (reason or "").strip()
    if not r:
        return None
    if r.startswith(CUSTOM_REASON_PREFIX):
        return "custom:" + r[len(CUSTOM_REASON_PREFIX):]
    if r.startswith("older than "):
        return "age"
    fixed = {"internship/student/thesis role": "early_career",
             "trainee programme": "trainee",
             "retail/store role": "retail_store",
             "off-profile occupation": "off_profile",
             "senior-level title": "senior_title",
             "outside target countries": "location",
             "not in English or German": "language"}
    if r in fixed:
        return fixed[r]
    if _YEARS_REASON_RE.match(r):
        return "years_required"
    if r.startswith("requires "):
        return "other_languages"
    return None


def hard_filter(title: str, location: str, description: str = "",
                posted_date: Optional[str] = None,
                first_seen: Optional[str] = None,
                cfg: Optional[dict] = None) -> Optional[str]:
    """Return an exclusion reason, or None when the posting stays eligible.

    Applied to every new posting before any ranking or LLM work, and
    re-applied to the board's New recommendations each run. Reasons are
    stored on the posting so the dashboard can show what was dropped why.
    Which rules run — and with which parameters — is governed by the
    dashboard-managed configuration (see filter_config above); without any
    stored overrides every rule runs with its built-in defaults. Bulk loops
    pass `cfg` (one snapshot) so a dashboard edit mid-run can never split a
    single pass across two rule sets.
    """
    t = title or ""
    cfg = cfg if cfg is not None else filter_config()
    if filter_enabled("age", cfg):
        max_age = int(filter_param("age", "max_days", cfg)
                      or config.MAX_POSTING_AGE_DAYS)
        age = posting_age_days(posted_date, first_seen)
        if age is not None and age > max_age:
            return f"older than {max_age} days"
    if _title_rule_hit("early_career", _EARLY_CAREER_EXCLUDE, t, cfg):
        return "internship/student/thesis role"
    if _title_rule_hit("trainee", _TRAINEE, t, cfg):
        return "trainee programme"
    if _title_rule_hit("retail_store", _RETAIL_STORE, t, cfg):
        return "retail/store role"
    # Occupations with no overlap with the profile (trades, medical, legal,
    # drivers, hospitality…). Guarded: a title that names analytics itself
    # ("Analyst Fahrerassistenzsysteme") is never occupation-excluded — the
    # LLM judges those with full context.
    if (filter_enabled("off_profile", cfg)
            and (_OFF_PROFILE_OCCUPATION.search(t)
                 or _extra_hit("off_profile", t, cfg))
            and not feedback.is_analytics_core_title(t)):
        return "off-profile occupation"
    if _title_rule_hit("senior_title", _SENIOR_TITLE, t, cfg):
        return "senior-level title"
    if filter_enabled("location", cfg):
        if (location_status(location) == "outside"
                or _extra_hit("location", location or "", cfg)):
            return "outside target countries"
    if filter_enabled("years_required", cfg) and description:
        max_years = filter_param("years_required", "max_years", cfg)
        max_years = MAX_YEARS_REQUIRED if max_years is None else int(max_years)
        yrs = required_years(description)
        if yrs is not None and yrs > max_years:
            return f"requires {yrs}+ years' experience"
    if filter_enabled("other_languages", cfg):
        langs = required_languages(t, description or "")
        if langs:
            return f"requires {', '.join(langs)}"
    if (filter_enabled("language", cfg)
            and not is_supported_language(t, description)):
        return "not in English or German"
    for cf in cfg["custom"]:
        if cf.enabled and cf.hits(t, location or "", description or ""):
            return f"{CUSTOM_REASON_PREFIX}{cf.name}"
    return None


# =============================================================================
# Duplicate detection — content fingerprints
# =============================================================================
# ATSs regularly carry the SAME vacancy under several job_ids (repostings,
# per-slot requisitions, consecutive announcement numbers), so the
# (company_id, job_id) dedupe key cannot catch them. A content fingerprint —
# company + normalized title + normalized location — does. Gender tags
# ("(m/w/d)") and reference numbers ("- 878389", "REF290650V") are stripped so
# a repost with a fresh requisition code still collapses onto the original.
_GENDER_TAG = re.compile(
    r"\(\s*(?:[mwfdx]\s*[/|]\s*){1,3}[mwfdx]\s*\)|\(all\s+genders?\)"
    r"|\(m/f/diverse\)|\bw/m/d\b|\bm/w/d\b|\bm/f/d\b|\bf/m/x\b", re.I)
_REF_CODE = re.compile(
    r"\b(?:ref|req|id|vn|job)?[-:#. ]?\d{4,}\b"          # bare/prefixed numbers
    r"|\b(?:ref|req)[a-z]*[-_ ]?\d{3,}[a-z0-9]*\b",      # REF290650V-style codes
    re.I)


def title_fingerprint(title: str) -> str:
    t = _GENDER_TAG.sub(" ", (title or "").lower())
    t = _REF_CODE.sub(" ", t)
    t = re.sub(r"[^a-z0-9äöüßéèáà]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def posting_fingerprint(company_id, title: str, location: str) -> str:
    """Identity of a vacancy as a human sees it. Same company, same cleaned
    title, same location = the same job, whatever the ATS calls it. Different
    locations stay distinct (a role advertised per-city is many real jobs)."""
    loc = re.sub(r"[^a-z0-9]+", " ", (location or "").lower()).strip()
    return f"{company_id}|{title_fingerprint(title)}|{loc}"


# =============================================================================
# Stage 2 — prefilter ranking (cheap, keyword-based)
# =============================================================================
STRONG_POS = [
    "business intelligence", "bi analyst", "bi developer", "bi engineer",
    "business analyst", "data analyst", "data analytics",
    "reporting analyst", "insights analyst", "power bi", "tableau",
    "commercial analyst", "customer analytics", "product analyst",
    "esg analyst", "esg data", "analytics", "data & analytics",
    "pricing analyst", "market intelligence",
    # BI engineering is reporting/dashboards work, not software engineering —
    # the user's very top pick was an Amazon "BI Engineer" role.
    "bi engineer", "business intelligence engineer",
    # German / DACH equivalents
    "datenanalyst", "datenanalystin", "business intelligence analyst",
    "data & ai analyst", "reporting specialist", "bi specialist",
]
MED_POS = ["analyst", "reporting", "insight", "dashboard", "data visualisation",
           "data visualization", "intelligence", "kpi",
           "berichtswesen", "auswertung", "datenanalyse", "data & ai",
           "forecast", "statistik", "statistic", "quantitative",
           "market research", "marktforschung"]
# Hard negatives: engineering / dev / research roles. Off-target even when the
# title also contains "analytics" (e.g. "Software Engineer, Analytics" or
# "Analytics Engineer" — a dbt/data-engineering role the user skips on sight).
NEG_HARD = [
    "data scientist", "machine learning", "ml engineer", "mlops", "ml ops",
    "deep learning", "software engineer", "software developer",
    "software development",   # "Software Development Manager/Engineer" — pure engineering
    "back end",
    "backend", "back-end", "front end", "frontend", "front-end",
    "full stack", "fullstack", "full-stack",
    "data engineer", "analytics engineer", "platform engineer",
    "distributed systems", "devops", "site reliability", " sre",
    "research scientist", "research engineer", "ai engineer",
    "cloud engineer", "infrastructure engineer", "security engineer",
    "systems engineer", "qa engineer", "test engineer", "softwareentwickler",
    "data modeling", "data modelling",
    # QA/testing roles that hide behind "analyst" ("TEST ANALYST" — skipped)
    "test analyst", "qa analyst", "testmanager", "test manager",
    "test automation", "testautomatisierung", "penetration tester",
]
# Soft negatives: other functions. Off-target unless a strong analytics signal.
# Grown from the user's manual skip history (229 decisions analysed 2026-07):
# product management, consulting, sales/retail, HR, marketing, support,
# IT administration, project/process management, data governance.
NEG_SOFT = [
    "account executive", "account manager", "sales development", "recruiter",
    "designer", "ux ", " ui", "accountant", "procurement", "solutions architect",
    "customer success", "marketing manager", "project manager", "vertrieb",
    # partner-titled function roles (HR/recruiting/people) — not analytics
    "hr business partner", "people partner", "recruitment business partner",
    "hr partner", "talent partner", "business partner", "partnermanager",
    "partner manager", "briefing partner",
    # product management (the single most-skipped family)
    "product manager", "produktmanager", "product owner", "produkt owner",
    "technical product", "produktmanagement",
    # consulting-adjacent roles the user consistently skips
    "deployment strategist", "value engineering",
    # sales / retail / business development
    "business development", "business developer", "inside sales", "sales specialist",
    "sales manager", "sales assistant", "store manager", "kundenberater",
    "deal desk", "deal strategist", "deal pricing", "merchandiser",
    # marketing / comms / content
    "campaign manager", "communications", "content manager", "social media",
    "brand manager", "kommunikation",
    # HR / people
    "hr analyst", "hr generalist", "people operations", "recruiting",
    "learning & development",
    # customer support / service
    "customer service", "customer care", "technical support",
    "support engineer", "service desk", "kundenservice", "servicecenter",
    "premium support",
    # IT administration / app management
    "application manager", "applikationsmanager", "system administrator",
    "it support", "software tester", "incident manager",
    # project / process management
    "program manager", "programm manager", "prozessmanager",
    "process manager", "scrum master", "projektleit", "transformation manager",
    "change manager", "delivery manager", "pmo",
    # data governance / stewardship (dressed as data, isn't analytics)
    "data governance", "data steward", "master data", "data quality",
    "data operations", "governance architect", "data management",
]
# Mild negatives (-15, not -35): functions the user mostly skips but has
# occasionally applied to (graduate consultant / solution-consultant roles) —
# down-ranked, not buried, so the LLM can still judge borderline ones.
NEG_MILD = [
    "consultant", "consulting", "presales", "pre-sales", "solution engineer",
    "solutions engineer", "solution consultant", "advisory",
]
# Off-domain analyst roles: contain "analyst" but sit outside BI/data analytics.
# Also grown from the skip history: policy/economics, finance/controlling,
# risk/compliance/legal — all skipped repeatedly even at junior level.
OFF_DOMAIN = [
    "tax", "payroll", "infosec", "information security", "security analyst",
    "credit risk", "fraud", "aml", "anti-money", "audit", "penetration",
    "policy", "economic affairs", "political", "treasury", "budget",
    "controlling", "controller", "accounting", "credit", "compliance",
    "sanctions", "legal", "regulatory", "quality control", "qc analyst",
    "safeguards", "kreditwürdigkeit", "risk", "financ",
    "private equity", "open source", "cash", "support", "sales",
]
# Early-career markers that make a role MORE attractive (interns/students are
# hard-excluded before ranking, so only genuinely wanted markers remain).
JUNIOR_POS = ["junior", "graduate", "entry level", "entry-level", "associate",
              "early career", "absolvent", "berufseinsteiger", "einsteiger",
              "young professional", "jr.", "jr "]
SKILL_KEYWORDS = ["sql", "power bi", "powerbi", "tableau", "excel", "dax", "etl",
                  "reporting", "dashboard", "analytics", "kpi", "stakeholder",
                  "data visualisation", "data visualization", "looker", "qlik",
                  "snowflake", "bigquery", "requirements", "business"]
HEAVY_PROG = ["microservices", "kubernetes", "ci/cd", "terraform", "pytorch",
              "tensorflow", "deep learning", "low-level", "c++", "golang",
              "distributed systems"]


@dataclass
class MatchResult:
    prefilter_score: float
    fit_score: float
    rationale: str
    watch_outs: str
    llm_used: bool = False
    surface: bool = True
    signals: dict = field(default_factory=dict)


def _has(text: str, needles) -> bool:
    return any(n in text for n in needles)


def prefilter(title: str, location: str, description: str,
              skip_cats: Optional[dict] = None,
              applied_cats: Optional[dict] = None) -> tuple[float, dict]:
    """Rank an eligible posting 0-100 for the LLM queue (higher = sooner).

    `skip_cats` / `applied_cats` are the learned decision profiles from
    `feedback.load_profiles()`; when given, titles matching a heavily-skipped
    function category are pushed down before they can eat an LLM slot."""
    t = f" {(title or '').lower()} "
    loc = (location or "").lower()
    desc = (description or "").lower()
    score = 50.0
    sig: dict = {}

    hard_neg = _has(t, NEG_HARD)
    strong = _has(t, STRONG_POS)
    med = _has(t, MED_POS)
    soft_neg = _has(t, NEG_SOFT)
    mild_neg = _has(t, NEG_MILD)
    off_domain = _has(t, OFF_DOMAIN)

    if hard_neg:
        score -= 45
        sig["title"] = "off-target (engineering/DS)"
    elif soft_neg:
        # The role's primary function is something else (sales, recruiting, PM,
        # design, ...). Even when the title name-drops a BI tool — e.g. "Tableau
        # Account Executive" — it is NOT an analytics role, so the
        # other-function signal must win over STRONG_POS.
        score -= 35
        sig["title"] = "off-target (other function)"
    elif strong and off_domain:
        score -= 10  # analyst-shaped but off the BI/data domain (tax/policy/…)
        sig["title"] = "analyst, off-domain"
    elif strong:
        score += 32
        sig["title"] = "on-target (BI/analytics)"
    elif mild_neg:
        score -= 15
        sig["title"] = "off-target (consulting-adjacent)"
    elif med and not off_domain:
        score += 14
        sig["title"] = "adjacent analytics"
    elif off_domain:
        score -= 35
        sig["title"] = "off-target (off-domain analyst)"

    if _has(t, JUNIOR_POS):
        score += 10
        sig["seniority"] = "junior/early-career"

    loc_status = location_status(location)
    if loc_status == "target":
        score += 14
        sig["location"] = "preferred region"
    elif loc_status == "outside":     # normally hard-filtered before ranking
        score -= 30
        sig["location"] = "outside target region"
    elif "remote" in loc or "hybrid" in loc:
        score += 4
        sig["location"] = "remote/hybrid"

    overlap = sorted({k for k in SKILL_KEYWORDS if k in desc})
    score += min(len(overlap) * 2.5, 18)
    if overlap:
        sig["skills"] = overlap[:8]
    if _has(desc, HEAVY_PROG):
        score -= 8
        sig["heavy_programming"] = True
    yrs = required_years(description)
    if yrs is not None and yrs >= 4:
        score -= 8
        sig["years_required"] = yrs
    if is_fixed_term(title, description):
        score -= 10
        sig["fixed_term"] = True

    # No analytics signal at all (no title family, no skill keywords, no
    # junior marker): geography alone must not carry mechanics, lawyers and
    # tax officers up the queue (base 50 + target-location 14 used to beat
    # real candidates). Since 2026-07-11 every survivor is scored eventually,
    # so this cap is pure ORDERING: signal-free postings sort behind anything
    # with a real signal and are scored last (still before the 10-day janitor
    # on any normally-budgeted run).
    if "title" not in sig and not overlap and "seniority" not in sig:
        score = min(score, config.PREFILTER_MIN_SCORE - 1)
        sig["no_signal"] = True

    # Learned skip patterns: repeatedly-rejected function families rank lower.
    penalty, cat = feedback.prefilter_penalty(title, skip_cats, applied_cats)
    if penalty:
        score += penalty
        sig["skip_pattern"] = cat

    score = max(0.0, min(100.0, score))
    return score, sig


# =============================================================================
# Stage 3a — heuristic scorer (fallback / no API key)
# =============================================================================
# Capped below the "worth applying" band: a keyword heuristic must never look
# like a confident recommendation. (Heuristic leftovers once sat at the very
# top of the board with 90+ scores — every one of them was manually skipped.)
HEURISTIC_MAX_FIT = 59.0


def heuristic_score(title, company, location, description, pre_score, sig) -> MatchResult:
    fit = min(round(pre_score * 0.9 + 5, 1), HEURISTIC_MAX_FIT)
    bits = []
    if sig.get("title", "").startswith("on-target"):
        bits.append("title is squarely in BI/analytics")
    elif sig.get("title") == "adjacent analytics":
        bits.append("an analytics-adjacent role")
    skills = sig.get("skills") or []
    if skills:
        bits.append("description mentions " + ", ".join(skills[:4]))
    if sig.get("location") == "preferred region":
        bits.append(f"location ({location}) fits your preferences")
    rationale = ("Provisional keyword match (not yet LLM-verified): "
                 + "; ".join(bits) + ".") if bits else \
        "Provisional keyword match (not yet LLM-verified) with your analytics profile."

    watch = []
    if sig.get("title", "").startswith("off-target"):
        watch.append("title leans toward data science / engineering, which you'd rather avoid")
    if sig.get("location") == "outside target region":
        watch.append(f"location ({location}) is outside your target region")
    yrs = sig.get("years_required")
    if yrs:
        watch.append(f"asks for {yrs} years' experience")
    if sig.get("fixed_term"):
        watch.append("appears to be a fixed-term/temporary contract")
    watch_outs = "; ".join(watch) if watch else "None obvious."

    return MatchResult(pre_score, fit, rationale, watch_outs, llm_used=False,
                       surface=fit >= config.SHORTLIST_MIN_SCORE, signals=sig)


# =============================================================================
# Stage 3b — Claude Sonnet batch scorer (the final filtering stage)
# =============================================================================
# Design for precision AND cost:
#   * The decision rules, CV, preferences and learned skip patterns form the
#     SYSTEM prompt and are cached (cache_control) — every batch after the
#     first reads them at 10% of the input price.
#   * Jobs are sent in batches (LLM_BATCH_SIZE) as compact JSON. Descriptions
#     are whitespace-collapsed and smart-trimmed (head + the requirements
#     section + key requirement sentences), so the model sees what actually
#     decides fit instead of the first 1200 characters.
#   * The model must return a per-posting GATE CHECK before its score;
#     deterministic post-checks clamp any score that contradicts a failed
#     gate. Hard requirements can never be out-reasoned.
#   * Live spend tracking against MAX_RUN_LLM_USD — the run can never overrun
#     the budget; unscored postings stay queued for the next run.

_RULES = """You are the final screening stage of JobScout, a personal job \
recommender. Each request contains several job postings (JSON). Judge every \
posting independently against the candidate profile below.

PROCEDURE — for EVERY posting, in this order:
1. Read the ENTIRE posting text, including the requirements at the end. \
Descriptions may be trimmed; "…[…]…" marks a cut, and a "facts" field carries \
deterministic extractor hints (verify them against the text; the text wins).
2. Fill in the GATE CHECK below, quoting evidence to yourself as you go.
3. Score fit. The FIRST failed gate caps the score — no exceptions, no matter \
how attractive the rest looks.

GATES (with score caps):
- rt (role type): internship / working-student / trainee / apprenticeship / \
thesis / PhD / volunteer -> cap 20. Else ok. NB: graduate schemes / graduate \
development programmes are OK — the profile TARGETS graduate roles; only \
programmes explicitly labelled "trainee" fail this gate.
- sen (seniority): explicit 5+ years required -> "y5", cap 30. Senior / Lead / \
Principal / Staff / Head / Director / VP / Chief / Expert / Architect title or \
people-management responsibility -> "senior", cap 30. Manager-in-title \
individual-contributor role -> "mgr", cap 50 — EXCEPT junior/graduate-marked \
manager titles ("(Junior) BI Manager"): in DACH "Manager" often just means \
employee, so those are "ok". 0-4 years or unstated -> ok.
- loc (location): must be Austria, Germany, UK, Ireland, Sweden, Switzerland, \
France, Netherlands, or Barcelona (Spain) — or remote within Europe. Use \
geography knowledge on bare city names ("Kraków", "Austin", "Pointe-Claire" \
are outside). Anywhere else incl. rest of Spain -> "out", cap 25. Unclear -> "unk".
- lad (ad language): posting written in English or German -> en/de. Anything \
else -> "other", cap 25.
- lrq (required language): if the job REQUIRES a language the candidate does \
not speak (anything beyond English/German; "fluent French", "Dutch native", \
"French-speaking market") -> name it, cap 25. An "or"-alternative satisfiable \
by English/German passes. "Nice to have" passes. Else "none".
- fn (function): what the day-to-day actually is, judged from duties AND \
requirements, not the title:
    core = business/data/BI analysis, reporting, dashboards, insights, \
commercial/pricing/market/product/customer analytics -> no cap. NB: "BI \
Engineer"/"Business Intelligence Engineer" roles centred on SQL, dashboards \
and reporting are core, NOT eng.
    adj  = analytics-heavy adjacent (ESG data, analytics-centred operations \
research) -> no cap
    eng  = software/data/ML/platform engineering, data science, DevOps, QA, \
IT/cloud/security engineering -> cap 35 (even when the title says "analytics")
    pm = product management/ownership -> cap 40 | consult = consulting/presales/\
deployment strategy -> cap 45 | sales = sales/BD/retail -> cap 35 | fin = \
finance/accounting/controlling/treasury/credit/procurement -> cap 45 | policy = \
policy/programme/international-affairs -> cap 40 | ops = operations/logistics/\
supply-chain coordination -> cap 40 | hr -> cap 40 | mkt = marketing/comms/\
content/design -> cap 40 | support = customer service/support -> cap 30 | \
itadm = IT administration/app management -> cap 35 | legal = legal/risk/\
compliance/QC -> cap 35 | gov = data governance/stewardship/master data -> \
cap 45 | projm = project/process/programme management -> cap 40 | trade = \
occupation with NO overlap with business/data analysis: skilled trades, \
manufacturing/warehouse floor, drivers, medical care, legal practice (lawyer/\
counsel), tax administration, hospitality, store staff, security guards, \
cabin crew -> cap 12 | other -> cap 45
- emp (employment type): fixed-term/temporary -> "fixed" (no cap, but MUST \
appear in watch_outs). Else perm/unk.

EDGE RULES:
- Empty or near-empty description: judge from title + company only, set \
unknown gates to ok/unk, score conservatively (never above 75), and write \
"no description" in watch. NEVER invent duties the text does not state.
- Conflicting signals: the explicit requirements text beats the title — a \
"Junior Analyst" ad demanding "5+ years' experience" is sen="y5".
- Mixed English+German ads are normal in DACH -> lad "en" or "de", never "other".
- A UK or Switzerland posting that explicitly refuses visa sponsorship: score \
normally but flag it in watch (candidate is Vienna-based, EU work rights).
- Very long descriptions: the gates hide in the requirements — read to the end.

SCORING (only after all gates pass) — weigh hard evidence, not vibes:
+ direct overlap with the candidate's tools: SQL, Excel, Power BI, Tableau, \
reporting, dashboards, KPI analysis, stakeholder communication
+ junior / graduate / entry designation or 0-2 years asked (mid, 3-4y, is fine)
+ analytics use of Python/R is a plus; a heavy engineering stack is not
+ German-language postings are just as good as English ones
- vague "analyst" roles whose duties are mostly coordination, administration \
or stakeholder management with token data work belong at 45-60, not 70+.

CALIBRATION — identical input must give identical output:
  85-100 on-target analytics role, right level, right region — apply now
  70-84  clearly worth applying; minor gaps or mild seniority stretch
  55-69  plausible but flawed: partly adjacent function, thin analytics core
  45-54  borderline; only if the week is slow
  21-44  fails a gate or wrong white-collar function — the caps above land here
  5-15   different occupation altogether (fn=trade): a mechanic, lawyer, tax
officer, nurse or warehouse role shares NOTHING with this candidate — never
score such a posting above 15, whatever its location or employer

EXAMPLES (format + calibration; abbreviated inputs):
IN: {"id":1,"title":"Senior Data Analyst (m/w/d)","location":"Berlin",\
"description":"...mindestens 6 Jahre Berufserfahrung..."}
OUT: {"id":1,"g":{"rt":"ok","sen":"y5","loc":"ok","lad":"de","lrq":"none",\
"fn":"core","emp":"unk"},"s":25,"why":"Senior title with an explicit 6-year \
requirement — out of reach for a 2026 graduate.","watch":"Requires 6+ years."}
IN: {"id":2,"title":"Junior BI Analyst","location":"Vienna",\
"description":"...SQL, Power BI dashboards for commercial stakeholders, \
0-2 years, unbefristet..."}
OUT: {"id":2,"g":{"rt":"ok","sen":"ok","loc":"ok","lad":"en","lrq":"none",\
"fn":"core","emp":"perm"},"s":90,"why":"Junior BI role on SQL/Power BI \
reporting in Vienna — exactly the profile's target stack and level.",\
"watch":"None"}
IN: {"id":3,"title":"Data Analyst - French Market","location":"Paris",\
"description":"...fluent French required for daily client contact..."}
OUT: {"id":3,"g":{"rt":"ok","sen":"ok","loc":"ok","lad":"en","lrq":"french",\
"fn":"core","emp":"unk"},"s":22,"why":"Core analyst role, but daily work \
requires fluent French, which the candidate does not speak.",\
"watch":"Hard French requirement."}
IN: {"id":4,"title":"Kfz-Mechatroniker (w/m/x)","location":"Hamburg",\
"description":"...Wartung und Instandsetzung von Fahrzeugen..."}
OUT: {"id":4,"g":{"rt":"ok","sen":"ok","loc":"ok","lad":"de","lrq":"none",\
"fn":"trade","emp":"unk"},"s":8,"why":"Vehicle-workshop trade role — no \
overlap whatsoever with a business/data analytics profile.",\
"watch":"Wrong occupation."}
IN: {"id":5,"title":"Junior Controller (m/w/d)","location":"Munich",\
"description":"...Monatsreporting, Budgetplanung, Forecasts in Excel und SAP, \
Unterstützung der Abschlüsse..."}
OUT: {"id":5,"g":{"rt":"ok","sen":"ok","loc":"ok","lad":"de","lrq":"none",\
"fn":"fin","emp":"unk"},"s":42,"why":"Junior controlling role — reporting \
overlap exists, but the core work is finance processes, not BI/data \
analytics.","watch":"Finance function, not analytics."}

OUTPUT: a JSON array ONLY — one object per posting, in the same order:
[{"id":<id>,"g":{"rt":..,"sen":..,"loc":..,"lad":..,"lrq":..,"fn":..,"emp":..},\
"s":<int 0-100>,"why":"<1-2 concrete sentences tying THIS role to the \
candidate's actual background>","watch":"<specific risks/gaps, or 'None'>"}]
Keep "why" under 30 words and "watch" under 15 words — dense, no filler.
No text before or after the JSON array. No markdown fences."""


def _system_blocks(cv: str, prefs: str, extra_rules: str = "") -> list[dict]:
    profile = (f"CANDIDATE CV:\n{(cv or '')[:6000]}\n\n"
               f"PREFERENCES NOTE (authoritative — overrides the CV where they "
               f"conflict):\n{(prefs or '')[:2000]}")
    if extra_rules:
        profile += f"\n\n{extra_rules[:3000]}"
    return [
        {"type": "text", "text": _RULES},
        # One cache breakpoint at the end of the static prefix: rules + profile
        # + learned patterns are identical for every batch in a run, so all
        # follow-up batches hit the cache.
        {"type": "text", "text": profile, "cache_control": {"type": "ephemeral"}},
    ]


# --- Description trimming -------------------------------------------------------
_REQ_SECTION = re.compile(
    r"(?:requirements?|qualifications?|your\s+profile|what\s+you(?:'|’)?ll\s+bring"
    r"|what\s+you\s+bring|who\s+you\s+are|about\s+you|must[- ]haves?"
    r"|minimum\s+qualifications|basic\s+qualifications|what\s+we\s+(?:are\s+)?"
    r"look(?:ing)?\s+for|anforderungen|qualifikation(?:en)?|ihr\s+profil"
    r"|dein\s+profil|das\s+bringst\s+du\s+mit|was\s+du\s+mitbringst"
    r"|wir\s+erwarten|voraussetzungen|dein\s+know[- ]how)", re.I)
_KEY_FACT = re.compile(
    r"[^.\n•;|]{0,160}(?:\byears?\b|\bjahren?\b|fluen|native|proficien"
    r"|muttersprach|verhandlungssicher|fließend|sprachkenntnisse"
    r"|fixed[- ]term|befristet|unbefristet|permanent\s+contract|temporary"
    r"|\bC1\b|\bC2\b|work\s+permit|visa\s+sponsorship)[^.\n•;|]{0,160}", re.I)


def trim_description(desc: str, budget: Optional[int] = None) -> str:
    """Compress a job description for the LLM without losing what decides fit.

    Whitespace is collapsed (job ads are full of blank lines — pure token
    waste). Within the budget we keep the head (role summary + duties), then
    jump to the requirements section, then append any key requirement
    sentences (years, languages, contract type) found anywhere else."""
    d = re.sub(r"\s+", " ", desc or "").strip()
    budget = budget or config.LLM_DESC_CHARS
    if len(d) <= budget:
        return d
    head_len = int(budget * 0.55)
    head = d[:head_len]
    out = head
    m = _REQ_SECTION.search(d, head_len)
    if m:
        out += " …[…]… " + d[m.start():m.start() + (budget - head_len)]
    else:
        # No recognizable requirements header — keep the TAIL instead of
        # silently dropping everything after the head: real ads state years,
        # languages and contract terms near the end far more often than in
        # the middle (audit 2026-07-12).
        out += " …[…]… " + d[-(budget - head_len):]
    facts = []
    seen_len = 0
    for fm in _KEY_FACT.finditer(d):
        s = fm.group(0).strip()
        if s and s not in out and seen_len < 600:
            facts.append(s)
            seen_len += len(s)
    if facts:
        out += " || KEY FACTS: " + " | ".join(facts[:6])
    return out


def _job_payload(job_id, title, company, location, posted, description) -> dict:
    payload = {
        "id": job_id,
        "title": (title or "")[:200],
        "company": (company or "")[:80],
        "location": (location or "")[:120],
        "posted": (posted or "")[:10],
        "description": trim_description(description or ""),
    }
    # Deterministic extractor hints — the model verifies them against the text.
    facts = []
    yrs = required_years(description or "")
    if yrs:
        facts.append(f"years_required={yrs}")
    langs = required_languages(title or "", description or "")
    if langs:
        facts.append(f"languages_required={'/'.join(langs)}")
    if is_fixed_term(title or "", description or ""):
        facts.append("fixed_term=yes")
    ls = location_status(location or "")
    if ls != "unknown":
        facts.append(f"location_check={ls}")
    if facts:
        payload["facts"] = "; ".join(facts)
    return payload


def _parse_llm_array(text: str) -> Optional[list]:
    m = re.search(r"\[.*\]", text, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    return _salvage_objects(text)


def _salvage_objects(text: str) -> Optional[list]:
    """Recover complete verdict objects from truncated/mangled output.

    If max_tokens cuts the array mid-object, every COMPLETE object is still
    perfectly usable — losing one posting to a retry beats losing the batch."""
    out = []
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(text[start:i + 1])
                    if isinstance(obj, dict) and "id" in obj and (
                            "s" in obj or "fit_score" in obj):
                        out.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None
    return out or None


# --- Gate enforcement ------------------------------------------------------------
# The model reports its gate check; these caps make the gates binding even if
# the score contradicts them. Belt and braces: our own deterministic extractors
# cap too, so neither a sloppy model nor a sloppy regex can inflate a score.
#
# PARSING PRINCIPLE (audit 2026-07-12): a cap must fire on RECOGNIZED failure
# values — never on unrecognized junk. The old exact-string checks were
# fail-closed: a model writing lad:"english" (instead of "en") or
# rt:"full-time" (instead of "ok") capped a perfect job at 20-25. Unknown
# values now PASS the model-gate layer; the deterministic extractor caps
# below still catch every real disqualifier they can verify in the text.
_FN_CAPS = {"eng": 35, "pm": 40, "consult": 45, "sales": 35, "fin": 45,
            "policy": 40, "ops": 40, "hr": 40, "mkt": 40, "support": 30,
            "itadm": 35, "legal": 35, "gov": 45, "projm": 40, "trade": 12,
            "other": 45}
# Long-form / synonym function labels the model may emit instead of the
# canonical short keys — mapped so the caps still bind (fail-open reduction).
_FN_ALIASES = {
    "engineering": "eng", "engineer": "eng", "software": "eng",
    "software engineering": "eng", "data engineering": "eng",
    "data science": "eng", "ds": "eng", "ml": "eng", "devops": "eng",
    "qa": "eng", "security": "eng", "it engineering": "eng",
    "product": "pm", "product management": "pm", "product manager": "pm",
    "product owner": "pm", "product_management": "pm",
    "consulting": "consult", "consultant": "consult", "presales": "consult",
    "advisory": "consult",
    "sale": "sales", "business development": "sales", "bd": "sales",
    "retail": "sales", "sales_retail": "sales",
    "finance": "fin", "financial": "fin", "accounting": "fin",
    "controlling": "fin", "treasury": "fin", "procurement": "fin",
    "credit": "fin", "finance_accounting": "fin",
    "marketing": "mkt", "communications": "mkt", "comms": "mkt",
    "content": "mkt", "design": "mkt", "marketing_comms": "mkt",
    "operations": "ops", "logistics": "ops", "supply chain": "ops",
    "supply-chain": "ops", "ops_logistics": "ops",
    "human resources": "hr", "people": "hr", "recruiting": "hr",
    "hr_people": "hr",
    "customer support": "support", "customer service": "support",
    "customer_support": "support",
    "it admin": "itadm", "it administration": "itadm", "it": "itadm",
    "application management": "itadm", "it_admin": "itadm",
    "legal/risk": "legal", "risk": "legal", "compliance": "legal",
    "quality control": "legal", "legal_risk_compliance": "legal",
    "governance": "gov", "data governance": "gov", "master data": "gov",
    "stewardship": "gov", "data_governance": "gov",
    "project management": "projm", "project": "projm", "process": "projm",
    "programme": "projm", "program": "projm", "pmo": "projm",
    "project_process_mgmt": "projm",
    "trades": "trade", "blue collar": "trade", "blue-collar": "trade",
}
# Values that mean "this gate passed" — never cap on these.
_GATE_PASS = {"", "ok", "none", "n/a", "na", "no", "pass", "passed", "fine",
              "yes", "unk", "unknown", "unclear", "unstated", "-"}
# Early-career role-type failures the rt gate may legitimately report.
_RT_FAIL = re.compile(
    r"intern|praktik|werkstudent|working.?student|\bstudent|trainee"
    r"|apprentice|ausbildung|azubi|\blehr|thesis|arbeit\b|phd|doctoral"
    r"|doktorand|postdoc|volunteer|volont|stagiaire|co-?op|dual\s+stud"
    r"|ferial|summer\s+(?:job|intern)|gap.?year|becari|early.?career\s+program",
    re.I)
_SEN_FAIL = re.compile(
    r"\by\s*(?:[5-9]|1\d)\b|\b(?:[5-9]|1\d)\s*\+?\s*(?:y\b|yrs?|years?|jahre)"
    r"|senior|\blead\b|principal|staff|head|director|\bvp\b|vice.?president"
    r"|chief|expert|architect|leitung|leiter", re.I)
_MGR_VAL = re.compile(r"\bmgr\b|\bmanager\b", re.I)
_LOC_FAIL = re.compile(r"^out|outside|elsewhere|abroad|non.?target|\bfail",
                       re.I)
# "not an internship", "under 5 years" — negated/bounded values must PASS.
_NEGATED = re.compile(r"\bnot?\b|\bkein|\bunder\b|\bless\b|\bup\s+to\b"
                      r"|\bbelow\b|\bmax(?:imum)?\b|\bbis\b|<", re.I)
# Language names the gates may report (superset of the hard-filter list, plus
# German exonyms and common 2-letter codes; NO "no"/"in" — too ambiguous).
_GATE_LANG_NAMES = set(_OTHER_LANGS) | {
    "norwegian", "icelandic", "slovak", "slovenian", "croatian", "serbian",
    "bulgarian", "ukrainian", "lithuanian", "latvian", "estonian", "hindi",
    "thai", "vietnamese", "indonesian", "malay", "tagalog", "filipino",
    "swahili", "flemish", "catalan", "basque", "galician", "cantonese",
    "français", "francais", "französisch", "niederländisch", "nederlands",
    "spanisch", "español", "espanol", "italiano", "italienisch", "svenska",
    "schwedisch", "dänisch", "dansk", "norsk", "norwegisch", "suomi",
    "finnisch", "polnisch", "polski", "tschechisch", "portugiesisch",
    "português", "russisch", "arabisch", "chinesisch", "mandarin",
    "japanisch", "koreanisch", "türkisch", "ungarisch", "rumänisch",
    "griechisch", "hebräisch",
    "fr", "es", "it", "nl", "sv", "da", "fi", "pl", "cs", "pt", "ru", "ar",
    "zh", "ja", "ko", "tr", "hu", "ro", "el", "he",
}
_LANG_OK_TOKENS = {"en", "de", "english", "german", "englisch", "deutsch",
                   "none", "ok", "both", "mixed", "bilingual"}
_SOFT_TOKENS = {"plus", "optional", "preferred", "nice", "bonus", "asset",
                "desirable", "vorteil"}


def _gate_tokens(value: str) -> set[str]:
    return {t for t in re.split(r"[^a-zäöüßéèáàãç]+", value.lower()) if t}


def _named_other_language(value: str) -> Optional[str]:
    """First recognized non-EN/DE language a gate value names — unless the
    value itself signals the requirement is soft ('nice to have') or
    satisfiable with EN/DE ('german or french')."""
    toks = _gate_tokens(value)
    named = sorted(toks & _GATE_LANG_NAMES)
    if not named:
        return None
    if toks & _SOFT_TOKENS:
        return None
    if toks & {"or", "oder", "either"} and toks & {"en", "de", "english",
                                                   "german", "englisch",
                                                   "deutsch"}:
        return None
    return named[0]


def _gate_caps(gates: dict) -> list[tuple[float, str]]:
    caps: list[tuple[float, str]] = []
    if not isinstance(gates, dict):
        return caps
    rt = str(gates.get("rt", "ok")).strip().lower()
    if (rt not in _GATE_PASS and "graduat" not in rt   # graduate schemes OK
            and not _NEGATED.search(rt) and _RT_FAIL.search(rt)):
        caps.append((20, f"role type: {rt}"))
    sen = str(gates.get("sen", "ok")).strip().lower()
    if sen not in _GATE_PASS and not _NEGATED.search(sen):
        if sen in ("senior", "y5") or _SEN_FAIL.search(sen):
            caps.append((30, "seniority"))
        elif _MGR_VAL.search(sen):
            # 50, not 40: manager-titled IC roles surface as "borderline"
            # instead of being buried below the shortlist line — the user has
            # applied to such roles before (Engagement Manager), and DACH
            # junior-IC "Manager" titles were being hidden entirely at 40.
            caps.append((50, "manager-level title"))
    loc = str(gates.get("loc", "ok")).strip().lower()
    if loc not in _GATE_PASS and _LOC_FAIL.search(loc):
        caps.append((25, "outside target region"))
    lad = str(gates.get("lad", "en")).strip().lower()
    if lad not in _GATE_PASS and not (_gate_tokens(lad) & _LANG_OK_TOKENS):
        if lad == "other" or _named_other_language(lad):
            caps.append((25, "ad not in English/German"))
    lrq = str(gates.get("lrq", "none")).strip().lower()
    if lrq not in _GATE_PASS and not (_gate_tokens(lrq) <= _LANG_OK_TOKENS):
        named = _named_other_language(lrq)
        if named:
            caps.append((25, f"requires {named}"))
    fn = str(gates.get("fn", "core")).strip().lower()
    fn = _FN_ALIASES.get(fn, fn)
    if fn in _FN_CAPS:
        caps.append((_FN_CAPS[fn], f"function: {fn}"))
    return caps


# Below this many characters a description carries no evidence — the model
# judges from the title alone, so the score is capped into "plausible, go
# read the ad yourself" territory and can never look like a verified match.
NO_DESC_CHARS = 40
NO_DESC_CAP = 75.0


def _deterministic_caps(job: dict,
                        cfg: Optional[dict] = None) -> list[tuple[float, str]]:
    """Caps our own extractors can prove from the posting text.

    Mirrors the hard-filter rules — and therefore RESPECTS the dashboard
    configuration: a rule the user disabled must not keep capping scores
    through the back door (audit 2026-07-12). Parameters (max_years) follow
    the configured value too."""
    caps: list[tuple[float, str]] = []
    cfg = cfg if cfg is not None else filter_config()
    desc = job.get("description") or ""
    title = job.get("title") or ""
    if filter_enabled("years_required", cfg):
        max_years = filter_param("years_required", "max_years", cfg)
        max_years = MAX_YEARS_REQUIRED if max_years is None else int(max_years)
        yrs = required_years(desc)
        if yrs is not None and yrs > max_years:
            caps.append((30, f"requires {yrs}+ years"))
    if filter_enabled("other_languages", cfg):
        langs = required_languages(title, desc)
        if langs:
            caps.append((25, f"requires {', '.join(langs)}"))
    if filter_enabled("senior_title", cfg) and _SENIOR_TITLE.search(title):
        caps.append((30, "senior-level title"))
    if (filter_enabled("off_profile", cfg)
            and _OFF_PROFILE_OCCUPATION.search(title)
            and not feedback.is_analytics_core_title(title)):
        caps.append((12, "off-profile occupation"))
    if (filter_enabled("location", cfg)
            and location_status(job.get("location") or "") == "outside"):
        caps.append((25, "outside target region"))
    if len(desc.strip()) < NO_DESC_CHARS:
        caps.append((NO_DESC_CAP, "no description — judged from title only"))
    return caps


# --- Live spend tracking ------------------------------------------------------
def _price(model: str) -> tuple[float, float, float, float]:
    for key, p in config.MODEL_PRICES_PER_MTOK.items():
        if key in (model or ""):
            return p
    return (3.0, 15.0, 0.30, 3.75)  # conservative default


def usage_cost_usd(usage, model: str) -> float:
    """Cost of one API response, from its usage block."""
    p_in, p_out, p_cache_read, p_cache_write = _price(model)
    inp = getattr(usage, "input_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    cr = getattr(usage, "cache_read_input_tokens", 0) or 0
    cw = getattr(usage, "cache_creation_input_tokens", 0) or 0
    return (inp * p_in + out * p_out + cr * p_cache_read + cw * p_cache_write) / 1e6


class SpendTracker:
    """Accumulates LLM spend for the current run against the budget cap."""

    def __init__(self, cap_usd: float | None = None):
        self.cap = config.MAX_RUN_LLM_USD if cap_usd is None else cap_usd
        self.spent = 0.0
        self.calls = 0

    def add(self, usage, model: str) -> None:
        self.spent += usage_cost_usd(usage, model)
        self.calls += 1

    @property
    def exhausted(self) -> bool:
        return self.spent >= self.cap


# Cache one client per process; track whether the LLM path is healthy so the
# pipeline can report if it silently degraded.
_client = None
_warned = False
llm_error_count = 0

# A PERMANENT API failure — no credit, a revoked/typo'd key, a model this key
# may not use. Retrying cannot help, and a run that keeps trying spends
# minutes producing nothing while the dashboard reports "complete". Once the
# first such failure is seen, this holds a short human reason, every further
# LLM call short-circuits, and the pipeline surfaces the reason at the top of
# the run summary. Reset per run via reset_llm_state().
llm_fatal_reason: str = ""

# Backstop for failures classify_fatal() has never seen (a new error shape, an
# auth mode this app does not speak). If this many calls fail in a row with
# nothing succeeding in between, the API is not going to start working during
# this run — stop, and report the last error verbatim.
CONSECUTIVE_FAIL_LIMIT = int(os.environ.get("JOBSCOUT_LLM_FAIL_LIMIT", "5"))
_consecutive_failures = 0


class LLMUnavailable(RuntimeError):
    """The Anthropic API is permanently unusable for this run."""


# Substrings that mark a permanent, non-retryable condition. Matched
# case-insensitively against the exception text.
_FATAL_MARKERS = (
    "credit balance is too low",
    "credit balance",
    "plans & billing",
    "billing",
    "insufficient_quota",
    "insufficient quota",
    "spend limit",
    "quota exceeded",
    "organization has been disabled",
    "account has been disabled",
    "invalid x-api-key",
    "invalid api key",
    "authentication_error",
    "permission_error",
    "permission denied",
)


# An error body that actually came from Anthropic says so. A bare 401/403 with
# an HTML body is a proxy, a captive portal or a CDN — not a dead key, and
# calling it one both aborts the run and sends the user to edit key.txt for
# nothing.
_ANTHROPIC_AUTH_SHAPE = (
    "authentication_error", "permission_error", "invalid x-api-key",
    "invalid api key", "x-api-key", "anthropic",
)

# Account-level failures stop everything (scoring AND page extraction);
# model-level ones only stop the model that raised them.
_ACCOUNT_LEVEL = ("out of credit", "invalid or revoked", "not permitted",
                  "workspace")


def account_level_fatal(reason: str) -> bool:
    """Does this reason mean the whole account is unusable (vs one model)?"""
    return any(m in (reason or "") for m in _ACCOUNT_LEVEL)


def classify_fatal(exc: BaseException, model: str | None = None) -> str:
    """Short human reason when `exc` means "retrying is pointless", else "".

    Deliberately conservative: only conditions a retry can never fix count as
    fatal, because opening the circuit aborts scoring, page extraction and the
    validation gate for the rest of the run. Rate limits, connection errors,
    5xx and anything that does not look like it came from Anthropic stay
    retryable.

    `model` names the model whose call failed — the browser tier uses a
    different (cheap) model than the scorer, and a message blaming the wrong
    one sends the user hunting in the wrong place.
    """
    name = type(exc).__name__
    text = str(exc)
    low = text.lower()
    status = getattr(exc, "status_code", None)
    model = model or config.ANTHROPIC_MODEL
    anthropic_shaped = any(m in low for m in _ANTHROPIC_AUTH_SHAPE)

    # 429 is NEVER fatal here: Anthropic returns it for ordinary rate limits,
    # and its message can mention billing ("would exceed your organization's
    # spend limit"). _llm_call already backs off and retries those.
    if status not in (429,) and ("credit balance" in low
                                 or "plans & billing" in low):
        return ("the Anthropic account is out of credit — top up at "
                "console.anthropic.com \u2192 Plans & Billing")
    # An identity-linked key needs a workspace header this app does not send.
    if "workspace-id is required" in low or "identity-linked api key" in low:
        return ("this is an identity-linked API key, which needs an "
                "anthropic-workspace-id header — create a workspace API key "
                "in the Console instead")
    if name == "AuthenticationError" or (status == 401 and anthropic_shaped) \
            or "invalid x-api-key" in low or "invalid api key" in low \
            or "authentication_error" in low:
        return ("the Anthropic API key is invalid or revoked — check "
                "key.txt / ANTHROPIC_API_KEY")
    if name == "PermissionDeniedError" or (status == 403 and anthropic_shaped) \
            or "permission_error" in low:
        return f"this Anthropic API key is not permitted to use {model}"
    if name == "NotFoundError" and "model" in low:
        return f"model {model!r} is not available to this API key"
    if status == 400 and any(m in low for m in _FATAL_MARKERS):
        return f"the Anthropic API rejected every request: {text[:160]}"
    return ""


def reset_llm_state() -> None:
    """Clear per-run LLM health so a new run starts from a clean slate."""
    global _warned, llm_error_count, llm_fatal_reason, _consecutive_failures
    _warned = False
    llm_error_count = 0
    llm_fatal_reason = ""
    _consecutive_failures = 0


_fatal_lock = threading.Lock()


def note_fatal(reason: str) -> None:
    """Arm the circuit breaker (first reason wins).

    Called from the browser-tier worker threads as well as the scorer, so the
    check-and-set is locked."""
    global llm_fatal_reason
    if not reason:
        return
    with _fatal_lock:
        if llm_fatal_reason:
            return
        llm_fatal_reason = reason
    sys.stderr.write(f"[jobscout] Anthropic API unavailable: {reason}\n")


def llm_available() -> bool:
    """True when an API key is configured — i.e. this run is in LLM mode.

    Deliberately NOT affected by llm_fatal_reason: a configured-but-broken key
    must keep postings queued for a real judgment, never fall back to the
    keyword heuristic. Only a run with no key at all scores heuristically.
    Use llm_healthy() to ask whether the API is actually reachable."""
    return bool(config.ANTHROPIC_API_KEY)


def llm_healthy() -> bool:
    """True when a key is configured AND no permanent failure was seen."""
    return bool(config.ANTHROPIC_API_KEY) and not llm_fatal_reason


def preflight(timeout: float = 30.0) -> str:
    """One-token credential probe, run once at the start of a pipeline run.

    Returns "" when the API is usable, otherwise the fatal reason (and arms
    the circuit breaker, so the run skips LLM work fast instead of retrying
    it hundreds of times). A merely flaky network leaves the run untouched:
    only permanent conditions are reported.
    """
    if not config.ANTHROPIC_API_KEY:
        return "no Anthropic API key configured (key.txt / ANTHROPIC_API_KEY)"
    if llm_fatal_reason:
        return llm_fatal_reason
    try:
        # The SDK default read timeout is 600 s with 2 retries — a half-open
        # proxy would stall the run for half an hour on "Checking the
        # Anthropic API…". A probe must fail fast or not at all.
        _get_client().with_options(timeout=timeout, max_retries=0) \
            .messages.create(
                model=config.ANTHROPIC_MODEL, max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
            )
        return ""
    except Exception as e:  # noqa: BLE001 — the point is to classify it
        reason = classify_fatal(e)
        if reason:
            note_fatal(reason)
            return reason
        return ""     # transient — let the run try for real


def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


def _llm_call(system_blocks: list[dict], user_text: str, *, max_tokens: int,
              tracker: Optional[SpendTracker]) -> Optional[str]:
    """One Messages call with retry; returns the text or None on failure."""
    global _warned, llm_error_count, _consecutive_failures
    import anthropic
    if llm_fatal_reason:
        # Circuit open: a permanent failure was already diagnosed this run.
        # Making the call again would waste a network round trip per batch and
        # bury the real reason under hundreds of identical errors.
        return None
    last_exc: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            # NB: no temperature parameter — newer models (Sonnet 5+) reject
            # it; the prompt itself demands deterministic, decisive judgments.
            resp = _get_client().messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=max_tokens,
                system=system_blocks,
                messages=[{"role": "user", "content": user_text}],
            )
            if tracker is not None:
                tracker.add(resp.usage, config.ANTHROPIC_MODEL)
            _consecutive_failures = 0
            return "".join(getattr(b, "text", "") for b in resp.content)
        except (anthropic.RateLimitError, anthropic.APIConnectionError) as e:
            last_exc = e
            time.sleep(2.0 * attempt)      # transient — back off and retry
        except anthropic.APIStatusError as e:
            last_exc = e
            if classify_fatal(e):
                break                       # permanent — handled below
            if getattr(e, "status_code", 500) >= 500:
                time.sleep(2.0 * attempt)  # server hiccup — retry
            else:
                break                       # 4xx — don't burn retries
        except Exception as e:  # noqa: BLE001 — SDK/parse issue
            last_exc = e
            break
    _consecutive_failures += 1
    if last_exc is not None:
        note_fatal(classify_fatal(last_exc))
        if not llm_fatal_reason and _consecutive_failures >= CONSECUTIVE_FAIL_LIMIT:
            note_fatal(
                f"{_consecutive_failures} Anthropic API calls failed in a row "
                f"({type(last_exc).__name__}: {str(last_exc)[:180]})")
    llm_error_count += 1
    if not _warned:
        _warned = True
        sys.stderr.write(
            f"[jobscout] LLM error ({type(last_exc).__name__}: {last_exc}); "
            "affected postings stay queued and will be scored on the next run.\n")
    return None


def score_batch(cv: str, prefs: str, jobs: Sequence[dict],
                tracker: Optional[SpendTracker] = None,
                extra_rules: str = "") -> dict[object, MatchResult]:
    """Score a batch of eligible postings with Claude Sonnet.

    `jobs` items need: id, title, company, location, posted, description,
    pre_score, signals. `extra_rules` (e.g. the learned skip-history section)
    joins the cached system prefix. Returns {id: MatchResult} — ONLY for
    postings the LLM actually judged; callers keep the rest queued.
    """
    if not config.ANTHROPIC_API_KEY or not jobs:
        return {}
    payload = [_job_payload(j["id"], j["title"], j["company"], j["location"],
                            j["posted"], j["description"]) for j in jobs]
    user_text = ("POSTINGS:\n" + json.dumps(payload, ensure_ascii=False)
                 + "\n\nReturn the JSON array of verdicts now.")
    text = _llm_call(_system_blocks(cv, prefs, extra_rules), user_text,
                     max_tokens=700 * len(jobs) + 400, tracker=tracker)
    if not text:
        return {}
    data = _parse_llm_array(text)
    if not data:
        return {}
    by_id = {j["id"]: j for j in jobs}
    out: dict[object, MatchResult] = {}
    fcfg = filter_config()          # one snapshot for the whole batch
    for item in data:
        if not isinstance(item, dict):
            continue
        raw_score = item.get("s", item.get("fit_score"))
        if raw_score is None:
            continue
        jid = item.get("id")
        job = by_id.get(jid) or (by_id.get(int(jid)) if str(jid).isdigit() else None)
        if job is None:
            continue
        try:
            fit = max(0.0, min(100.0, float(raw_score)))
        except (TypeError, ValueError):
            continue
        rationale = str(item.get("why", "")).strip()
        watch = str(item.get("watch", item.get("watch_outs", "None"))).strip() or "None"

        # Enforce the gates: model-reported first, then our own extractors.
        caps = _gate_caps(item.get("g") or {}) + _deterministic_caps(job, fcfg)
        if caps:
            cap_val, cap_why = min(caps, key=lambda c: c[0])
            if fit > cap_val:
                fit = cap_val
                note = f"capped at {cap_val:.0f} — {cap_why}"
                watch = note if watch.lower() in ("none", "none.", "") else f"{watch}; {note}"

        out[job["id"]] = MatchResult(
            prefilter_score=job["pre_score"],
            fit_score=round(fit, 1),
            rationale=rationale,
            watch_outs=watch,
            llm_used=True,
            surface=fit >= config.SHORTLIST_MIN_SCORE,
            signals=job.get("signals") or {},
        )
    return out

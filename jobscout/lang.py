"""Lightweight posting-language detection (no dependencies).

The candidate reads English and German only, so postings written in any other
language are dropped before they cost an LLM call or clutter the board. Job ads
are short, formulaic texts — counting distinctive function words is reliable
here and costs microseconds.

`detect_language(text)` returns "en", "de", or "other".
A posting passes when its dominant language is English or German; mixed
EN+DE ads (common in DACH) pass because both scores are high.
"""
from __future__ import annotations

import re

# Distinctive, high-frequency words per language. Words that two languages
# share (EN "team" / DE "Team", DE "als" / NL "als", ES "en" / NL-SV "en", EN
# "do" / PL "do", ...) are deliberately left out of the smaller lists so a hit
# is real evidence. "die" stays German: it is vanishingly rare in English ads.
_LANG_WORDS: dict[str, frozenset[str]] = {
    "en": frozenset((
        "the", "and", "you", "your", "we", "our", "with", "will",
        "are", "this", "that", "have", "role", "skills",
        "experience", "about", "years", "join", "of", "to",
    )),
    "de": frozenset((
        "und", "der", "die", "das", "für", "mit", "wir", "du", "dich", "dein",
        "deine", "ihre", "eine", "einen", "bei", "von", "zu",
        "werden", "sind", "sowie", "oder", "aufgaben", "kenntnisse",
        "erfahrung", "unser", "unsere", "unserem", "arbeiten", "bieten",
        "profil", "bereich", "wenn", "nicht", "auch",
    )),
    "fr": frozenset((
        "le", "la", "les", "des", "vous", "nous", "et", "est", "pour", "dans",
        "une", "du", "au", "aux", "avec", "vos", "votre", "sur", "être",
        "poste", "chez", "afin", "ainsi", "compétences",
        "missions", "expérience", "entreprise", "êtes", "d'une", "d'un",
    )),
    "es": frozenset((
        "el", "los", "las", "para", "con", "una", "es", "que",
        "del", "por", "más", "nuestro", "nuestra", "equipo",
        "experiencia", "trabajo", "empresa", "buscamos", "años", "sobre",
    )),
    "it": frozenset((
        "il", "di", "della", "che", "nel", "sono",
        "più", "nostro", "nostra", "lavoro", "esperienza", "anni",
        "ricerca", "azienda", "competenze", "attività", "presso",
    )),
    "nl": frozenset((
        "het", "een", "van", "voor", "je", "jij", "jouw", "wij", "aan", "bij",
        "worden", "werken", "binnen", "ervaring", "onze", "naar", "zijn",
        "kennis", "niet", "ook", "nederlands", "functie", "wat",
    )),
    "sv": frozenset((
        "och", "att", "för", "med", "som", "det", "är", "av", "på",
        "vi", "ett", "din", "hos", "arbeta", "erfarenhet", "eller",
        "kommer", "inom", "våra", "vår",
    )),
    "pl": frozenset((
        "się", "oraz", "jest", "dla", "pracy", "praca", "doświadczenie",
        "zespołu", "firmy", "umiejętności", "będzie", "których", "zakresie",
    )),
    "cs": frozenset((
        "práce", "zkušenosti", "týmu", "nabízíme", "požadujeme", "znalost",
        "jsme", "které", "vzdělání", "pozice", "hledáme",
    )),
    "pt": frozenset((
        "não", "mais", "nosso", "nossa", "experiência", "trabalho",
        "equipa", "equipe", "função", "conhecimento", "área", "você",
    )),
}

# Unicode letters + apostrophe (for French d'un / d'une).
_WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")

_PASS = ("en", "de")


def _scores(text: str) -> dict[str, int]:
    words = _WORD_RE.findall(text.lower())
    counts = {lang: 0 for lang in _LANG_WORDS}
    for w in words:
        for lang, vocab in _LANG_WORDS.items():
            if w in vocab:
                counts[lang] += 1
    return counts


def detect_language(text: str, *, min_hits: int = 4) -> str:
    """Dominant language of a posting text: 'en', 'de', or 'other'.

    Short/ambiguous texts (fewer than `min_hits` stopword hits for every
    language) default to 'en' — most feed titles are English and a hard drop
    on no evidence would be wrong more often than right.
    """
    if not text:
        return "en"
    counts = _scores(text[:4000])
    best = max(counts, key=counts.get)  # type: ignore[arg-type]
    if counts[best] < min_hits:
        return "en"
    return best if best in _PASS else "other"


def is_supported_language(title: str, description: str = "") -> bool:
    """True when the posting reads as English or German.

    The description (when present) dominates: titles are often English while
    the ad body is not — e.g. an 'Analyst' title on a French-only ad."""
    body = (description or "").strip()
    if len(body) >= 200:
        return detect_language(body) in _PASS
    combined = f"{title or ''}\n{body}".strip()
    return detect_language(combined) in _PASS

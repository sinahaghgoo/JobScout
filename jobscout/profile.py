"""Your profile: CV text + a short free-text preferences note.

Setup is done once. The CV is stored and reusable; the preferences note is a
single editable sentence-or-two that reshapes what the matcher surfaces.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import config, db

# Your words count. This note is treated as part of your profile by the matcher,
# so adjacent roles you would actually take are surfaced and ones you would not
# are filtered out. Edit data/profile/preferences.txt anytime to reshape results.
DEFAULT_PREFERENCES = """\
I am finishing an MSc in Data Science (graduating 2026), but I am targeting roles in
business analysis, business intelligence and data analytics — NOT data science or
heavy software/ML engineering.

Roles that fit me: Business Analyst, BI Analyst / Business Intelligence Analyst,
Data Analyst, Analytics / Reporting Analyst, BI Developer, Insights Analyst,
Product / Commercial / Customer Analytics, ESG Data Analyst.

I want work centred on analytics, reporting, dashboards and turning data into
decisions for stakeholders — using SQL, Excel, Power BI and Tableau. I prefer
roles that are NOT heavy on programming: please down-rank pure Data Scientist,
Machine Learning Engineer, Software / Backend Engineer, Data Engineer, MLOps and
research-heavy positions.

Seniority: entry-level, graduate, junior or mid. Locations: everything in West
Europe (EU or UK). Languages: German (native) and English (fluent).
"""


def cv_text_from_docx(path: Path) -> str:
    """Extract readable text from a .docx CV."""
    from docx import Document  # imported lazily so the dep is optional at runtime

    doc = Document(str(path))
    parts: list[str] = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts).strip()


def load_cv_text(cv_source: Optional[Path] = None) -> str:
    """Return CV text, preferring a provided source, else the cached cv.txt."""
    if cv_source:
        src = Path(cv_source)
        if src.suffix.lower() == ".docx":
            return cv_text_from_docx(src)
        return src.read_text(encoding="utf-8")
    if config.CV_TXT_PATH.exists():
        return config.CV_TXT_PATH.read_text(encoding="utf-8")
    return ""


def init_profile(cv_source: Optional[Path] = None, preferences: Optional[str] = None) -> None:
    """Load the CV + preferences into files and the DB. Idempotent."""
    cv = load_cv_text(cv_source)
    if cv:
        config.CV_TXT_PATH.write_text(cv, encoding="utf-8")

    if preferences is None:
        if config.PREFERENCES_PATH.exists():
            preferences = config.PREFERENCES_PATH.read_text(encoding="utf-8")
        else:
            preferences = DEFAULT_PREFERENCES
    config.PREFERENCES_PATH.write_text(preferences, encoding="utf-8")

    db.save_profile(cv, preferences)

"""Connector registry + helpers to fetch one company's jobs by ATS type."""
from __future__ import annotations

from typing import Optional

from .amazon import AmazonConnector
from .apple import AppleConnector
from .ashby import AshbyConnector
from .base import Connector, RawJob
from .beesite import BeesiteConnector
from .dvinci import DvinciConnector
from .erecruiter import ERecruiterConnector
from .greenhouse import GreenhouseConnector
from .ibm import IbmConnector
from .jibe import JibeConnector
from .lever import LeverConnector
from .microsoft import MicrosoftConnector
from .oracle import OracleConnector
from .personio import PersonioConnector
from .recruitee import RecruiteeConnector
from .sf_rmk import SfRmkConnector
from .smartrecruiters import SmartRecruitersConnector
from .successfactors import SuccessFactorsConnector
from .uber import UberConnector
from .un import UnConnector
from .verbund import VerbundConnector
from .workable import WorkableConnector
from .workday import WorkdayConnector

# ATS type -> connector instance
REGISTRY: dict[str, Connector] = {
    GreenhouseConnector.ats_type: GreenhouseConnector(),
    LeverConnector.ats_type: LeverConnector(),
    PersonioConnector.ats_type: PersonioConnector(),
    AshbyConnector.ats_type: AshbyConnector(),
    SmartRecruitersConnector.ats_type: SmartRecruitersConnector(),
    RecruiteeConnector.ats_type: RecruiteeConnector(),
    WorkableConnector.ats_type: WorkableConnector(),
    WorkdayConnector.ats_type: WorkdayConnector(),
    SuccessFactorsConnector.ats_type: SuccessFactorsConnector(),
    DvinciConnector.ats_type: DvinciConnector(),
    ERecruiterConnector.ats_type: ERecruiterConnector(),
    AmazonConnector.ats_type: AmazonConnector(),
    OracleConnector.ats_type: OracleConnector(),
    AppleConnector.ats_type: AppleConnector(),
    MicrosoftConnector.ats_type: MicrosoftConnector(),
    SfRmkConnector.ats_type: SfRmkConnector(),
    UnConnector.ats_type: UnConnector(),
    UberConnector.ats_type: UberConnector(),
    VerbundConnector.ats_type: VerbundConnector(),
    JibeConnector.ats_type: JibeConnector(),
    BeesiteConnector.ats_type: BeesiteConnector(),
    IbmConnector.ats_type: IbmConnector(),
}

# Order for one-time slug auto-detection: cheapest/most common first.
# These all resolve from a single slug guessed from the company name.
# Workday + SuccessFactors are NOT slug-guessable (they need tenant/data-centre/
# site coordinates or a career host), so they are configured via KNOWN_TOKENS.
DETECTION_ORDER = ["greenhouse", "lever", "ashby", "personio", "recruitee",
                   "smartrecruiters", "workable"]


def get_connector(ats_type: str) -> Optional[Connector]:
    return REGISTRY.get(ats_type)


def fetch_detail(ats_type: str, token: str, detail_ref: str) -> dict:
    conn = get_connector(ats_type)
    if conn is None or not getattr(conn, "supports_detail", False):
        return {}
    return conn.fetch_detail(token, detail_ref)


def probe(ats_type: str, token: str) -> bool:
    conn = get_connector(ats_type)
    return bool(conn and conn.probe(token))


def fetch_jobs(ats_type: str, token: str) -> list[RawJob]:
    conn = get_connector(ats_type)
    if conn is None:
        raise ValueError(f"No connector for ATS type '{ats_type}'")
    # NB: Personio job URLs are synthesized inside the connector itself, which
    # knows which tld (.de/.com) actually served the feed.
    return conn.fetch(token)

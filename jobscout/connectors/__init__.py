"""ATS connectors. Each turns one company's feed into normalized RawJob records.

Add a new ATS by writing one connector module + registering it in registry.py —
the rest of the pipeline never needs to change.
"""
from .base import Connector, FetchError, RawJob, html_to_text  # noqa: F401
from .registry import (  # noqa: F401
    DETECTION_ORDER,
    REGISTRY,
    fetch_detail,
    fetch_jobs,
    get_connector,
)

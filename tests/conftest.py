"""Shared pytest setup.

Two invariants for the whole suite:

1. **No test ever calls the Anthropic API.** Scoring/extraction are always
   monkeypatched, but `pipeline._llm_preflight` would otherwise make a real
   one-token probe (it found `key.txt` in the project root and hit the live
   API — which then armed the circuit breaker and silently changed what the
   rest of the run did). JOBSCOUT_PREFLIGHT=0 turns the probe off.

2. **LLM health is per-test.** `matching.llm_fatal_reason` /
   `llm_error_count` are module globals; one test arming the breaker used to
   change the behaviour of every test that ran after it.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("JOBSCOUT_PREFLIGHT", "0")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_llm_state():
    from jobscout import matching
    matching.reset_llm_state()
    yield
    matching.reset_llm_state()

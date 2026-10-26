"""Live smoke test for all eleven ATS connectors.

Hits each platform with a known-good token (drawn from the POC company list) and
prints one normalized job, so you can confirm connectivity end to end.

Run:  python scripts/smoke_connectors.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobscout.connectors import fetch_detail, fetch_jobs  # noqa: E402
from jobscout.connectors.registry import probe  # noqa: E402

CASES = [
    ("greenhouse", "bitpanda"),
    ("lever", "palantir"),
    ("ashby", "openai"),
    ("personio", "anyline-gmbh"),
    ("smartrecruiters", "frequentis"),
    ("workable", "huggingface"),
    ("workday", "nvidia|wd5|NVIDIAExternalCareerSite"),
    ("successfactors", "jobs.rbinternational.com"),
    ("dvinci", "jobs.wienerstadtwerke.at"),
    ("erecruiter", "www.brz-jobs.at"),
    ("amazon", "AUT,DEU,IRL,GBR"),
    # recruitee has no token in the POC list; add one here if a target uses it.
]


def main() -> int:
    ok = True
    for ats, token in CASES:
        try:
            assert probe(ats, token), "probe returned False"
            jobs = fetch_jobs(ats, token)
            assert jobs, "no jobs returned"
            j = jobs[0]
            extra = ""
            if not j.description and j.detail_ref:
                d = fetch_detail(ats, token, j.detail_ref)
                extra = f" | enriched desc={len(d.get('description', ''))} chars"
            print(f"OK  {ats:15} {len(jobs):4d} jobs | {j.title[:42]!r} "
                  f"@ {j.location[:18]!r} | {j.posted_date or 'date n/a'}{extra}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"FAIL {ats:15} {token!r}: {e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

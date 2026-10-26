"""Lever connector.

GET api.lever.co/v0/postings/{token}?mode=json
Returns postings with categories (team, location, commitment), hosted/apply
URLs and createdAt (epoch ms) — no auth.

Some companies (Mistral AI, 2026-07) disable this public API — it returns []
while their hosted board at jobs.lever.co/{token} stays live and full. For
that case there is an HTML fallback: the hosted board is server-rendered with
a stable structure (div.posting > a.posting-title + h5 + location spans),
so postings are recovered without a headless browser. Descriptions are empty
in fallback mode (titles/locations carry the ranking; the posting URL is real).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from .base import Connector, RawJob, html_to_text, http_get


def _epoch_ms_to_date(ms) -> str | None:
    if not ms:
        return None
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    except (ValueError, TypeError, OSError):
        return None


# One hosted-board posting: the anchor carries the canonical URL (with the
# posting UUID), the h5 the title, and the location span sits nearby.
_HTML_POSTING = re.compile(
    r'<a[^>]+class="[^"]*posting-title[^"]*"[^>]+href="'
    r'(?P<url>https://jobs\.(?:eu\.)?lever\.co/[^/"]+/(?P<id>[0-9a-f-]{36}))"'
    r'.*?<h5[^>]*>(?P<title>[^<]+)</h5>'
    r'(?P<meta>.{0,600}?)</a>',
    re.S | re.I)
_HTML_LOCATION = re.compile(
    r'<span[^>]+class="[^"]*sort-by-location[^"]*"[^>]*>([^<]+)</span>', re.I)


class LeverConnector(Connector):
    ats_type = "lever"
    BASE = "https://api.lever.co/v0/postings/{token}?mode=json"
    BOARD = "https://jobs.lever.co/{token}"

    def _fetch_html_fallback(self, token: str) -> list[RawJob]:
        try:
            html = http_get(self.BOARD.format(token=token),
                            accept="text/html").text
        except Exception:  # noqa: BLE001 — fallback is best-effort
            return []
        jobs: list[RawJob] = []
        seen: set[str] = set()
        for m in _HTML_POSTING.finditer(html):
            jid = m.group("id")
            if jid in seen:
                continue
            seen.add(jid)
            loc = _HTML_LOCATION.search(m.group("meta") or "")
            jobs.append(RawJob(
                job_id=jid,
                title=html_to_text(m.group("title")).strip(),
                location=html_to_text(loc.group(1)).strip() if loc else "",
                posted_date=None,
                date_confidence="low",
                url=m.group("url"),
                description="",
                extra={"source": "lever-html-fallback"},
            ))
        return jobs

    def fetch(self, token: str) -> list[RawJob]:
        resp = http_get(self.BASE.format(token=token))
        data = resp.json()
        jobs: list[RawJob] = []
        for p in data:
            cats = p.get("categories") or {}
            desc = p.get("descriptionPlain") or html_to_text(p.get("description", ""))
            extra_plain = p.get("additionalPlain") or ""
            jobs.append(
                RawJob(
                    job_id=str(p.get("id")),
                    title=(p.get("text") or "").strip(),
                    location=(cats.get("location") or "").strip(),
                    posted_date=_epoch_ms_to_date(p.get("createdAt")),
                    url=p.get("hostedUrl") or p.get("applyUrl", ""),
                    description=(desc + ("\n\n" + extra_plain if extra_plain else "")).strip(),
                    extra={"team": cats.get("team"), "commitment": cats.get("commitment")},
                )
            )
        if not jobs:
            # API opt-out (returns [] while the hosted board is full) — read
            # the server-rendered board instead of reporting an empty company.
            return self._fetch_html_fallback(token)
        return jobs

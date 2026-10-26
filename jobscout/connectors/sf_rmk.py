"""SAP SuccessFactors *Recruiting Marketing* connector (a.k.a. "jobs2web" / RMK).

A large set of SuccessFactors customers front their career site with the SAP
Recruiting-Marketing platform (formerly Jobs2Web). These sites are served from a
dedicated host (e.g. ``jobs.adidas-group.com``, ``jobs.bmwgroup.com``,
``jobs.ilo.org``) and — crucially for us — publish a complete, public
**Google-for-Jobs product feed** as an RSS 2.0 document at::

    GET https://<host>/sitemap_index.xml

Each ``<item>`` is one live posting and carries everything we need:

    <item>
      <title>Social Health Protection Technical Officer - P3 (Cairo, EG)</title>
      <description><![CDATA[ ...full HTML body... ]]></description>
      <link>https://jobs.ilo.org/job/Cairo-...-Officer-P3/1399744233/</link>
      <guid>1399744233</guid>
      <g:id>1399744233</g:id>                      (Google namespace)
      <g:location>Cairo, EG</g:location>
      <g:employer>ilo</g:employer>
      <g:job_function>Egypt, Cairo</g:job_function>
      <g:expiration_date>2026-07-27</g:expiration_date>
    </item>

No auth, no CSRF, no JS — a single GET returns the whole board. This is the
SAP-shared SuccessFactors fallback that *does* work: the ``?company=<X>`` apply
backend on ``career5.successfactors.eu`` is client-rendered and unreachable, but
the RMK front-host feed is plain XML.

Important quirks this connector handles:
  * The feed has **no posting date** (only an *expiration* date). Per the
    project blueprint we therefore emit ``posted_date=None`` flagged
    ``date_confidence="low"`` and let dedup + first_seen drive "what's new".
  * The feed is **not paginated** — it is the entire board in one document
    (Adidas ~1,100 items / ~10 MB). We parse it once and cap at
    ``SF_MAX_JOBS`` to stay bounded, taking the feed's own order (RMK lists the
    most recently published first).
  * ``<description>`` is CDATA-wrapped HTML -> run through ``html_to_text``.

Token format:  the RMK host, e.g.  ``"jobs.adidas-group.com"``.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

from .. import config
from .base import (Connector, FetchError, RawJob, html_to_text, http_get,
                   is_west_europe, norm_space)

# Google product-feed namespace used for the structured <g:*> fields.
_G = "{http://base.google.com/ns/1.0}"


def _text(item: ET.Element, tag: str) -> str:
    el = item.find(tag)
    return (el.text or "") if el is not None else ""


class SfRmkConnector(Connector):
    ats_type = "sf_rmk"
    supports_detail = False  # the feed already carries the full description

    @staticmethod
    def _feed_url(token: str) -> str:
        host = token.split("|", 1)[0].strip().strip("/")
        if not host:
            raise FetchError(f"sf_rmk token must be a host, got {token!r}")
        return f"https://{host}/sitemap_index.xml"

    @staticmethod
    def _west_europe_only(token: str) -> bool:
        """Default: keep only West-Europe roles (these boards are global and
        retail-heavy, so an unfiltered cap would crowd out the EU office roles).
        Append ``|all`` to the token to keep the whole board."""
        return "|all" not in token.lower()

    @classmethod
    def _parse(cls, markup: str, west_europe_only: bool = True) -> list[RawJob]:
        # The feed is a Google-Jobs RSS document; tolerate a stray leading BOM.
        root = ET.fromstring(markup.lstrip("﻿"))
        out: list[RawJob] = []
        seen: set[str] = set()
        cap = config.SF_RMK_MAX
        for item in root.iter("item"):
            jid = (_text(item, f"{_G}id") or _text(item, "guid")).strip()
            title = norm_space(_text(item, "title"))
            if not jid or not title or jid in seen:
                continue
            location = norm_space(_text(item, f"{_G}location"))
            # Whole board in one document → filter to the target region here so
            # the per-run cap captures EU office roles, not global retail.
            if west_europe_only and not is_west_europe(location):
                continue
            seen.add(jid)
            url = (_text(item, "link") or "").strip()
            expires = (_text(item, f"{_G}expiration_date") or "").strip()[:10] or None
            out.append(RawJob(
                job_id=jid,
                title=title,
                location=location,
                posted_date=None,            # RMK feed exposes no posting date
                date_confidence="low",
                url=url,
                description=html_to_text(_text(item, "description")),
                extra={
                    "employer": norm_space(_text(item, f"{_G}employer")) or None,
                    "job_function": norm_space(_text(item, f"{_G}job_function")) or None,
                    "expiration_date": expires,
                },
            ))
            if len(out) >= cap:
                break
        return out

    def fetch(self, token: str) -> list[RawJob]:
        resp = http_get(self._feed_url(token), accept="application/xml")
        return self._parse(resp.text, self._west_europe_only(token))

    def probe(self, token: str) -> bool:
        try:
            resp = http_get(self._feed_url(token), accept="application/xml")
        except FetchError:
            return False
        text = resp.text
        # Must be the RSS product feed (not an HTML maintenance/SPA shell).
        if "<item>" not in text and "<item " not in text:
            return False
        try:
            # validate the feed structure regardless of region
            return len(self._parse(text, west_europe_only=False)) > 0
        except ET.ParseError:
            return False

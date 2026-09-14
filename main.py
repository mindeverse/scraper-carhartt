"""Carhartt WIP scraper - entry point.

Pipeline:
  1. Discover every product URL by walking category pages (?page=N) until the
     store's "page does not exist" page is reached.
  2. Fetch + parse every product detail page (JSON-LD + RSC gallery + HTML).
  3. Diff against existing Supabase rows; embed only what changed (local SigLIP).
  4. Batch-upsert (merge-duplicates) and clean up stale products.
  5. Print a run summary; write failed_products.log on partial failures.

Run:
    python main.py [--category men] [--limit 10] [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

import parser as store_parser
from config import CONFIG
from embeddings import RateLimiter, SiglipEmbedder
from supabase_client import (
    SupabaseClient,
    build_record,
    diff_record,
    needs_embedding,
)

log = logging.getLogger("scraper")

STATS = {
    "discovered": 0,
    "new": 0,
    "updated": 0,
    "unchanged": 0,
    "front_embeddings": 0,
    "back_embeddings": 0,
    "text_embeddings": 0,
    "stale_deleted": 0,
    "errors": 0,
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging() -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)


def build_info_text(record: dict[str, Any], meta: dict[str, Any]) -> str:
    parts = [
        record.get("title"),
        record.get("description"),
        record.get("category"),
        record.get("gender"),
        meta.get("color"),
        record.get("price"),
        record.get("sale"),
        meta.get("material"),
        " ".join(meta.get("care_instructions") or []),
        " ".join(meta.get("details") or []),
    ]
    return " ".join(p for p in parts if p) or ""


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

async def discover_category(
    client: httpx.AsyncClient, limiter: RateLimiter, category_url: str
) -> list[dict[str, Any]]:
    slug = category_url.rstrip("/").split("/")[-1]
    label = store_parser.humanize_category(slug)
    gender_hint = store_parser.gender_from_category(slug)
    found: list[dict[str, Any]] = []
    page = 1
    while page <= CONFIG.max_pages_per_category:
        url = category_url if page == 1 else f"{category_url}?page={page}"
        html = await _fetch_page(client, limiter, url)
        if page > 1 and store_parser.is_not_found_page(html):
            log.info("category %s ended at page %d (not-found)", slug, page)
            break
        cards = store_parser.parse_category_page(html)
        if not cards:
            log.info("category %s page %d empty, stopping", slug, page)
            break
        for card in cards:
            card["category"] = label
            card["gender_hint"] = gender_hint
        found.extend(cards)
        log.info("category %s page %d -> %d products", slug, page, len(cards))
        page += 1
    return found


async def _fetch_page(client: httpx.AsyncClient, limiter: RateLimiter, url: str) -> str:
    await limiter.wait()
    r = await client.get(url)
    r.raise_for_status()
    return r.text


# --------------------------------------------------------------------------- #
# Product processing
# --------------------------------------------------------------------------- #

async def process_product(
    client: httpx.AsyncClient,
    limiter: RateLimiter,
    embedder: SiglipEmbedder,
    entry: dict[str, Any],
    existing: dict[str, dict[str, Any]],
    product_url: str,
    now: str,
    stats: dict[str, int],
    do_embed: bool = True,
) -> Optional[dict[str, Any]]:
    """Fetch, parse, diff and embed one product. Returns the write payload or None."""
    try:
        html = await _fetch_page(client, limiter, product_url)
        rsc = store_parser.extract_rsc_text(html)
        parsed = store_parser.parse_product_page(html, rsc)

        parsed["gender"] = _resolve_gender(parsed, entry)
        parsed["compressed_image_url"] = store_parser.compressed_image_url(parsed["image_url"])

        record, meta = build_record(
            parsed, product_url, entry["categories"], CONFIG.source, CONFIG.brand, CONFIG.country, now
        )

        payload = diff_record(record, existing.get(product_url))
        front_needed, back_needed, info_needed = needs_embedding(record, existing.get(product_url))

        if not payload and not (front_needed or back_needed or info_needed):
            stats["unchanged"] += 1
            return None

        if not do_embed:
            front_needed = back_needed = info_needed = False

        # ---- embeddings (local SigLIP, run in a thread so the loop isn't blocked) ----
        if front_needed:
            vec = await _embed_image(embedder, record["image_url"]) if record["image_url"] else None
            if vec is not None:
                payload["image_embedding"] = vec
                stats["front_embeddings"] += 1
        if back_needed:
            vec = await _embed_image(embedder, record["back_image_url"]) if record["back_image_url"] else None
            payload["back_image_embedding"] = vec
            if vec is not None:
                stats["back_embeddings"] += 1
        elif record.get("back_image_url") is None and existing.get(product_url, {}).get("back_image_url"):
            payload["back_image_embedding"] = None

        if info_needed:
            info_text = build_info_text(record, meta)
            vec = await _embed_text(embedder, info_text) if info_text else None
            if vec is not None:
                payload["info_embedding"] = vec
                stats["text_embeddings"] += 1

        # record embedding flags in metadata so future runs can skip unchanged work
        if front_needed or back_needed:
            merged_meta = json.loads(payload.get("metadata", record["metadata"]))
            flags = dict(merged_meta.get("embedding_flags") or {})
            if front_needed:
                flags["image"] = payload.get("image_embedding") is not None
            if back_needed:
                flags["back"] = payload.get("back_image_embedding") is not None
            merged_meta["embedding_flags"] = flags
            payload["metadata"] = json.dumps(merged_meta, ensure_ascii=False)

        is_new = existing.get(product_url) is None
        stats["new" if is_new else "updated"] += 1
        # Always attach conflict keys for PostgREST on_conflict matching.
        payload["id"] = record["id"]
        payload["source"] = record["source"]
        payload["product_url"] = record["product_url"]
        payload.pop("embedding_version", None)
        return payload
    except Exception as exc:
        stats["errors"] += 1
        log.error("failed product %s: %s", product_url, exc)
        _write_failed(product_url, exc)
        return None


async def _embed_image(embedder: SiglipEmbedder, url: str) -> Optional[list[float]]:
    await asyncio.sleep(CONFIG.embedding_delay)
    return await asyncio.to_thread(embedder.generate_image_embedding, url)


async def _embed_text(embedder: SiglipEmbedder, text: str) -> Optional[list[float]]:
    await asyncio.sleep(CONFIG.embedding_delay)
    return await asyncio.to_thread(embedder.generate_text_embedding, text)


def _resolve_gender(parsed: dict[str, Any], entry: dict[str, Any]) -> Optional[str]:
    g = parsed.get("gender")
    if g in ("men", "women"):
        return g
    if g == "unisex":
        return None
    hint = entry.get("gender_hint")
    if hint in ("men", "women"):
        return hint
    return None


def _write_failed(product_url: str, exc: Exception) -> None:
    try:
        with open("failed_products.log", "a") as fh:
            fh.write(f"{datetime.now(timezone.utc).isoformat()}\t{product_url}\t{exc}\n")
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Carhartt WIP scraper")
    ap.add_argument("--category", help="scrape only this category slug (e.g. accessories-sale)")
    ap.add_argument("--limit", type=int, default=0, help="max products to process")
    ap.add_argument("--dry-run", action="store_true", help="discovery only, no product fetch / embeddings / DB")
    ap.add_argument("--no-embed", action="store_true", help="scrape + write rows but skip embeddings")
    return ap.parse_args()


async def run() -> int:
    args = parse_args()
    setup_logging()
    cfg = CONFIG

    if not args.dry_run:
        cfg.require_supabase()

    headers = {"User-Agent": cfg.user_agent}
    store_limiter = RateLimiter(cfg.request_delay)

    async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=60) as client:
        # ---------- 1. discovery ----------
        categories: dict[str, list[str]] = {}
        cat_gender_hints: dict[str, Optional[str]] = {}
        for category_url in cfg.category_urls:
            if args.category and args.category not in category_url:
                continue
            try:
                cards = await discover_category(client, store_limiter, category_url)
            except httpx.HTTPError as exc:
                log.error("discovery failed for %s: %s", category_url, exc)
                STATS["errors"] += 1
                continue
            slug = category_url.rstrip("/").split("/")[-1]
            cat_gender_hints[slug] = store_parser.gender_from_category(slug)
            for card in cards:
                categories.setdefault(card["product_url"], [])
                if card["category"] not in categories[card["product_url"]]:
                    categories[card["product_url"]].append(card["category"])
        STATS["discovered"] = len(categories)

        entries = [{"product_url": url, "categories": cats} for url, cats in categories.items()]
        log.info("discovered %d unique products", len(entries))

        if args.dry_run:
            print(json.dumps({"dry_run": True, "discovered": len(entries),
                              "categories": list(cat_gender_hints)}, indent=2))
            return 0

        limit = args.limit or cfg.scrape_limit or len(entries)
        entries = entries[:limit]

        # ---------- 2. load existing rows ----------
        supabase = SupabaseClient(cfg)
        existing = await supabase.fetch_existing(cfg.source)

        # ---------- 3. fetch/parse/embed ----------
        embedder = await asyncio.to_thread(SiglipEmbedder)
        sem = asyncio.Semaphore(cfg.fetch_concurrency)
        now = datetime.now(timezone.utc).isoformat()

        async def guarded(entry: dict[str, Any]) -> Optional[dict[str, Any]]:
            async with sem:
                return await process_product(
                    client, store_limiter, embedder, entry,
                    existing, entry["product_url"], now, STATS,
                    do_embed=not args.no_embed,
                )

        results = await asyncio.gather(*(guarded(e) for e in entries))
        payloads = [r for r in results if r]

        # ---------- 4. upsert ----------
        failed = await supabase.upsert_batch(payloads)
        for row in failed:
            _write_failed(row.get("product_url", "?"), RuntimeError("upsert failed after retries"))
            STATS["errors"] += 1

        # ---------- 5. stale cleanup ----------
        seen = set(categories.keys())
        deleted, _ = await supabase.cleanup_stale(existing, seen, cfg.source)
        STATS["stale_deleted"] = len(deleted)

    _print_summary()
    return 0 if STATS["errors"] == 0 else 1


def _print_summary() -> None:
    print("=" * 46)
    print("RUN SUMMARY")
    print("=" * 46)
    print(f"Products discovered          : {STATS['discovered']}")
    print(f"New products added           : {STATS['new']}")
    print(f"Products updated             : {STATS['updated']}")
    print(f"Products unchanged (skipped) : {STATS['unchanged']}")
    print(f"Front embeddings generated   : {STATS['front_embeddings']}")
    print(f"Back embeddings generated    : {STATS['back_embeddings']}")
    print(f"Text embeddings generated    : {STATS['text_embeddings']}")
    print(f"Stale products deleted       : {STATS['stale_deleted']}")
    print(f"Errors / failures            : {STATS['errors']}")
    print("=" * 46)


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(run()))
    except KeyboardInterrupt:
        sys.exit(130)

"""Supabase (PostgREST) client: batch upserts, smart diffing, stale cleanup.

Design:
  * One SELECT at run start loads every existing row for the source into memory.
  * Upserts use PostgREST `on_conflict=(source, product_url)` with
    `Prefer: resolution=merge-duplicates`, so a payload only updates the columns
    it contains (partial update) and new rows are inserted in the same request.
    Batches of 10 keep round trips low; large batches hit Supabase statement timeouts.
  * Failed batches are retried 3x with exponential backoff; remaining failures
    are returned so main.py can log them to failed_products.log.
  * Stale products (not seen this run) get scrape_miss_count incremented in their
    metadata JSON and are deleted after 2 consecutive misses.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

log = logging.getLogger(__name__)

BATCH_SIZE = 10
RETRY_ATTEMPTS = 3

# Columns compared between scraped record and existing DB row.
COMPARE_FIELDS = [
    "title",
    "description",
    "category",
    "gender",
    "price",
    "sale",
    "image_url",
    "back_image_url",
    "additional_images",
    "affiliate_url",
    "size",
    "brand",
    "tags",
    "country",
    "other",
    "compressed_image_url",
]

# Keys inside metadata JSON that are volatile and never trigger an update.
VOLATILE_METADATA_KEYS = {
    "scraped_at",
    "last_seen_at",
    "last_missed_at",
    "scrape_miss_count",
    "embedding_flags",
}


def make_product_id(source: str, product_url: str) -> str:
    """Stable per-product id: hash of source + product_url."""
    return hashlib.sha256(f"{source}:{product_url}".encode("utf-8")).hexdigest()[:32]


class SupabaseClient:
    def __init__(self, config) -> None:
        self.config = config
        self.headers = {
            "apikey": config.supabase_service_role_key,
            "Authorization": f"Bearer {config.supabase_service_role_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Cache-Control": "no-cache, no-store",
            "Pragma": "no-cache",
        }
        self.base = f"{config.supabase_url}/rest/v1/products"
        self._columns: Optional[set[str]] = None

    # ------------------------------------------------------------------ #
    # Schema detection (keeps payloads aligned with the real table)
    # ------------------------------------------------------------------ #
    async def fetch_columns(self) -> set[str]:
        if self._columns is not None:
            return self._columns
        url = f"{self.config.supabase_url}/rest/v1/?apikey={self.config.supabase_service_role_key}"
        try:
            r = await self._request("GET", url)
            data = r.json()
            props = data["definitions"]["products"]["properties"]
            self._columns = set(props.keys())
        except Exception as exc:  # fall back to the documented set
            log.warning("could not detect products schema (%s); using documented columns", exc)
            self._columns = {
                "id", "source", "product_url", "affiliate_url", "image_url",
                "compressed_image_url", "back_image_url", "brand", "title",
                "description", "category", "gender", "price", "sale", "metadata",
                "size", "second_hand", "country", "tags", "additional_images",
                "other", "image_embedding", "back_image_embedding",
                "info_embedding", "embedding_version", "created_at", "last_seen_at",
            }
        log.info("products table exposes %d columns", len(self._columns))
        return self._columns

    async def filter_payload(self, row: dict[str, Any]) -> dict[str, Any]:
        cols = await self.fetch_columns()
        return {k: v for k, v in row.items() if k in cols}

    # ------------------------------------------------------------------ #
    # Fetch existing rows
    # ------------------------------------------------------------------ #
    async def fetch_existing(self, source: str) -> dict[str, dict[str, Any]]:
        cols = await self.fetch_columns()
        select = ["id", "product_url", "metadata"] + [c for c in COMPARE_FIELDS if c in cols]
        if "embedding_version" in cols:
            select.append("embedding_version")
        url = self.base
        params = {"select": ",".join(select), "source": f"eq.{source}"}
        r = await self._request("GET", url, params=params)
        out: dict[str, dict[str, Any]] = {}
        for row in r.json():
            out[row["product_url"]] = row
        log.info("loaded %d existing rows for source=%s", len(out), source)
        return out

    # ------------------------------------------------------------------ #
    # Upsert
    # ------------------------------------------------------------------ #
    async def upsert_batch(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Upsert a batch of product payloads (merge-duplicates).

        Only sends keys present on each row (never pads with None). Drops null
        values except always requiring id/source/product_url for on_conflict.
        Never includes embedding_version. On batch failure after retries, falls
        back to single-row upserts.

        Returns the list of rows that failed after all retries (and single-row fallback).
        """
        failed: list[dict[str, Any]] = []
        cols = await self.fetch_columns()
        # Live Finds products table must not receive embedding_version (PGRST204).
        cols = {c for c in cols if c != "embedding_version"}
        required = ("id", "source", "product_url")

        def _sanitize(row: dict[str, Any]) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for k, v in row.items():
                if k == "embedding_version" or k not in cols:
                    continue
                # Omit nulls for conflict-safe partial updates (except required keys).
                if v is None and k not in required:
                    continue
                out[k] = v
            for k in required:
                if out.get(k) is None and row.get(k) is not None:
                    out[k] = row[k]
            return out

        headers = {
            **self.headers,
            "Prefer": "resolution=merge-duplicates,return=minimal",
        }
        url = f"{self.base}?on_conflict=source,product_url"

        async def _post(payload: list[dict[str, Any]]) -> None:
            r = await self._request("POST", url, json=payload, headers=headers)
            if r.status_code >= 300:
                raise httpx.HTTPStatusError(
                    f"upsert {r.status_code}", request=r.request, response=r
                )

        for i in range(0, len(rows), BATCH_SIZE):
            raw_chunk = rows[i : i + BATCH_SIZE]
            chunk: list[dict[str, Any]] = []
            for row in raw_chunk:
                sanitized = _sanitize(row)
                if all(sanitized.get(k) is not None for k in required):
                    chunk.append(sanitized)
                else:
                    log.error(
                        "upsert row missing id/source/product_url: %s",
                        {k: sanitized.get(k) for k in required},
                    )
                    failed.append(sanitized)
            if not chunk:
                continue

            ok = False
            for attempt in range(RETRY_ATTEMPTS):
                try:
                    await _post(chunk)
                    ok = True
                    break
                except httpx.HTTPError as exc:
                    detail = ""
                    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
                        detail = exc.response.text[:1200]
                    log.warning(
                        "upsert batch %d failed (attempt %d): %s %s",
                        i // BATCH_SIZE, attempt + 1, exc, detail,
                    )
                    if attempt < RETRY_ATTEMPTS - 1:
                        await asyncio.sleep(2 ** (attempt + 1))
            if ok:
                continue

            log.warning(
                "upsert batch %d failed after %d attempts; falling back to single-row upserts",
                i // BATCH_SIZE,
                RETRY_ATTEMPTS,
            )
            for row in chunk:
                row_ok = False
                for attempt in range(RETRY_ATTEMPTS):
                    try:
                        await _post([row])
                        row_ok = True
                        break
                    except httpx.HTTPError as exc:
                        log.warning(
                            "single-row upsert failed (attempt %d): %s",
                            attempt + 1,
                            exc,
                        )
                        if attempt < RETRY_ATTEMPTS - 1:
                            await asyncio.sleep(2 ** (attempt + 1))
                if not row_ok:
                    failed.append(row)
        return failed

    # ------------------------------------------------------------------ #
    # Stale cleanup
    # ------------------------------------------------------------------ #
    async def cleanup_stale(
        self, existing: dict[str, dict[str, Any]], seen: set[str], source: str
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Handle products not seen this run.

        Returns (deleted_ids, updated_metadata_rows) where updated rows carry
        product_url -> new metadata payload for PATCH.
        """
        deleted: list[str] = []
        to_patch: list[dict[str, Any]] = []
        now = datetime.now(timezone.utc).isoformat()
        for product_url, row in existing.items():
            if product_url in seen:
                continue
            metadata = _parse_metadata(row.get("metadata"))
            miss = int(metadata.get("scrape_miss_count", 0) or 0) + 1
            if miss >= 2:
                deleted.append(row["id"])
                log.info("stale delete (missed %dx): %s", miss, product_url)
            else:
                metadata["scrape_miss_count"] = miss
                metadata["last_missed_at"] = now
                metadata["last_seen_at"] = now
                to_patch.append({"product_url": product_url, "metadata": json.dumps(metadata)})
                log.info("stale miss #%d: %s", miss, product_url)
        if deleted:
            await self._delete_rows(deleted)
        await self._patch_metadata(to_patch, source)
        return deleted, to_patch

    async def _delete_rows(self, ids: list[str]) -> None:
        for i in range(0, len(ids), 100):
            chunk = ids[i : i + 100]
            url = f"{self.base}?id=in.({','.join(chunk)})"
            for attempt in range(RETRY_ATTEMPTS):
                try:
                    r = await self._request("DELETE", url)
                    if r.status_code >= 300:
                        raise httpx.HTTPStatusError(
                            f"delete {r.status_code}", request=r.request, response=r
                        )
                    break
                except httpx.HTTPError as exc:
                    log.warning("delete batch failed (attempt %d): %s", attempt + 1, exc)
                    if attempt < RETRY_ATTEMPTS - 1:
                        await asyncio.sleep(2 ** (attempt + 1))

    async def _patch_metadata(self, rows: list[dict[str, Any]], source: str) -> None:
        for row in rows:
            url = f"{self.base}?source=eq.{source}&product_url=eq.{row['product_url']}"
            try:
                await self._request("PATCH", url, json={"metadata": row["metadata"]})
            except httpx.HTTPError as exc:
                log.warning("metadata patch failed for %s: %s", row["product_url"], exc)

    # ------------------------------------------------------------------ #
    # Low-level request with retry
    # ------------------------------------------------------------------ #
    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        headers = kwargs.pop("headers", {})
        merged = {**self.headers, **headers}
        # trust_env=False bypasses any local/system caching proxy so SELECTs always
        # return fresh data (avoids stale reads in dev environments with proxies).
        async with httpx.AsyncClient(timeout=90, trust_env=False) as client:
            r = await client.request(method, url, headers=merged, **kwargs)
        if r.status_code == 401 or r.status_code == 403:
            raise RuntimeError(f"Supabase auth error ({r.status_code}): check service-role key")
        return r


# --------------------------------------------------------------------------- #
# Record assembly + diffing
# --------------------------------------------------------------------------- #

def _parse_metadata(raw: Optional[str]) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def build_record(
    parsed: dict[str, Any],
    product_url: str,
    categories: list[str],
    source: str,
    brand: str,
    country: str,
    now: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Assemble the full DB payload for a (possibly new) product.

    Returns (record, metadata_dict) where metadata_dict is the parsed metadata
    used for the info-embedding text.
    """
    color = parsed.get("color")
    sizes = parsed.get("sizes") or []
    gender = parsed.get("gender") or None
    on_sale = bool(parsed.get("sale"))

    metadata: dict[str, Any] = {
        "sku": parsed.get("sku"),
        "product_key": parsed.get("product_key"),
        "color": color,
        "sizes": sizes,
        "availability": parsed.get("availability"),
        "material": parsed.get("material"),
        "material_care": parsed.get("material_care"),
        "care_instructions": parsed.get("care_instructions"),
        "size_and_fit": parsed.get("size_fit"),
        "details": parsed.get("details_list"),
        "currency": parsed.get("currency"),
        "price_original": parsed.get("price"),
        "price_sale": parsed.get("sale"),
        "on_sale": on_sale,
        "categories": categories,
        "gender_source": parsed.get("gender"),
        "scrape_miss_count": 0,
        "scraped_at": now,
        "last_seen_at": now,
    }

    tags: list[str] = [brand]
    for c in categories:
        tags.append(c)
    if color:
        tags.append(color)
    if gender:
        tags.append(gender)
    if on_sale:
        tags.append("Sale")

    return {
        "id": make_product_id(source, product_url),
        "source": source,
        "product_url": product_url,
        "affiliate_url": None,
        "image_url": parsed.get("image_url"),
        "compressed_image_url": parsed.get("compressed_image_url"),
        "back_image_url": parsed.get("back_image_url"),
        "brand": brand,
        "title": parsed.get("title"),
        "description": parsed.get("description"),
        "category": ", ".join(dict.fromkeys(categories)) or None,
        "gender": gender,
        "price": parsed.get("price"),
        "sale": parsed.get("sale"),
        "metadata": json.dumps(metadata, ensure_ascii=False),
        "size": ", ".join(sizes) if sizes else None,
        "second_hand": False,
        "country": country,
        "tags": tags,
        "additional_images": parsed.get("additional_images"),
        "other": None,
        "image_embedding": None,
        "back_image_embedding": None,
        "info_embedding": None,
        "created_at": now,
    }, metadata


def diff_record(record: dict[str, Any], existing: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Compare a scraped record against the existing DB row.

    Returns the minimal payload of changed NON-embedding columns (or the full
    record for new rows). Embedding regeneration is decided separately by
    `needs_embedding` so the actual embedding work can happen before building the
    final write payload.
    """
    if existing is None:
        return record

    old_metadata = _parse_metadata(existing.get("metadata"))
    new_metadata = _parse_metadata(record["metadata"])

    old_stable = {k: v for k, v in old_metadata.items() if k not in VOLATILE_METADATA_KEYS}
    new_stable = {k: v for k, v in new_metadata.items() if k not in VOLATILE_METADATA_KEYS}

    changed: dict[str, Any] = {}

    def _same(a: Any, b: Any) -> bool:
        if isinstance(a, list) and isinstance(b, list):
            return a == b
        return (a or None) == (b or None)

    for field in COMPARE_FIELDS:
        if not _same(record.get(field), existing.get(field)):
            changed[field] = record.get(field)
    if old_stable != new_stable:
        changed["metadata"] = record["metadata"]

    # reset miss count once a product is seen again
    if (old_metadata.get("scrape_miss_count") or 0) > 0:
        merged = _parse_metadata(changed.get("metadata", record["metadata"]))
        merged["scrape_miss_count"] = 0
        changed["metadata"] = json.dumps(merged, ensure_ascii=False)

    return changed


def needs_embedding(record: dict[str, Any], existing: Optional[dict[str, Any]]) -> tuple[bool, bool, bool]:
    """Decide whether front / back / text embeddings must be (re)generated.

      * front: new row, image_url changed, or no front embedding exists yet.
      * back : new row with back image, back_image_url changed (incl. removal),
               or back image set but embedding missing.
      * info : new row, or any info text field (title, description, category,
               gender, price, sale, metadata) changed.
    """
    if existing is None:
        return (bool(record.get("image_url")), bool(record.get("back_image_url")), True)
    old_metadata = _parse_metadata(existing.get("metadata"))
    emb_flags = old_metadata.get("embedding_flags") or {}
    front_ok = bool(existing.get("embedding_version") == 2 or emb_flags.get("image"))
    back_ok = bool(emb_flags.get("back"))
    front = bool(record.get("image_url")) and (
        not front_ok or existing.get("image_url") != record.get("image_url")
    )
    back = bool(record.get("back_image_url")) and (
        not back_ok or existing.get("back_image_url") != record.get("back_image_url")
    )
    changed = diff_record(record, existing)
    info = bool(set(changed) & {"title", "description", "category", "gender", "price", "sale", "metadata"})
    return front, back, info
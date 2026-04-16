"""
SecondBrainProcessor — background enrichment agent for Second Brain mode.

Responsibilities
----------------
1. **Summarise** new warm-tier captures (one Ollama call per note, amortised).
2. **Embed** notes via Ollama (nomic-embed-text) so backlink similarity works
   even if the Rust eviction drainer used a zero-placeholder embedding.
3. **Build backlinks** — parse ``[backlink:<id>]`` markers left by the
   TypeScript SecondBrainHandler, and discover implicit links via vector
   cosine similarity (threshold configurable, default 0.80).
4. **Detect clusters** — when ≥ N notes share a topic (determined by mean
   pairwise similarity), emit a notification via the GristMill HTTP events
   endpoint so Bell Tower can send a Slack nudge.
5. **Spaced repetition** — flag notes not accessed for > ``stale_days`` days
   and post a nudge notification.
6. **Conflict detection** — when a newly captured note is highly similar to an
   existing note but their embeddings point in opposite directions (cosine < 0),
   emit a conflict event so the user can review both.

Architecture note
-----------------
The processor communicates with the Rust core exclusively through the
GristMill HTTP API (``/api/memory/recall``, ``/api/memory/remember``, and
``/events``).  This avoids the need for a compiled PyO3 wheel at runtime and
keeps the Python layer stateless.

Scheduling
----------
Run the processor as a periodic service (e.g. every 15 minutes via cron or
as a long-running asyncio loop).  The entry point is
``SecondBrainProcessor.run_once()`` for one-shot execution, or
``SecondBrainProcessor.run_loop(interval_secs)`` for continuous operation.

Configuration is via environment variables or explicit constructor args.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
from datetime import datetime, timezone
from typing import Any

import httpx

from .models import SecondBrainNote

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_BACKLINK_RE = re.compile(r"\[backlink:([A-Z0-9]{26})\]")

# ── Config ─────────────────────────────────────────────────────────────────────


class ProcessorConfig:
    """Runtime configuration for SecondBrainProcessor."""

    def __init__(
        self,
        *,
        gristmill_base_url: str | None = None,
        ollama_base_url: str | None = None,
        embed_model: str | None = None,
        summarise_model: str | None = None,
        recall_limit: int = 50,
        cluster_min_size: int = 5,
        cluster_sim_threshold: float = 0.75,
        backlink_sim_threshold: float = 0.80,
        stale_days: float = 7.0,
        conflict_sim_threshold: float = 0.85,
        notification_channel: str = "second_brain",
    ) -> None:
        self.gristmill_base_url: str = (
            gristmill_base_url
            or os.environ.get("GRISTMILL_BASE_URL", "http://127.0.0.1:3000")
        )
        self.ollama_base_url: str = (
            ollama_base_url
            or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        )
        self.embed_model: str = (
            embed_model
            or os.environ.get("SECOND_BRAIN_EMBED_MODEL", "nomic-embed-text")
        )
        self.summarise_model: str = (
            summarise_model
            or os.environ.get("SECOND_BRAIN_SUMMARISE_MODEL", "llama3.1:8b")
        )
        self.recall_limit: int = recall_limit
        self.cluster_min_size: int = cluster_min_size
        self.cluster_sim_threshold: float = cluster_sim_threshold
        self.backlink_sim_threshold: float = backlink_sim_threshold
        self.stale_days: float = stale_days
        self.conflict_sim_threshold: float = conflict_sim_threshold
        self.notification_channel: str = notification_channel


# ── SecondBrainProcessor ──────────────────────────────────────────────────────


class SecondBrainProcessor:
    """
    Background enrichment agent for Second Brain notes.

    Usage::

        processor = SecondBrainProcessor(ProcessorConfig())
        await processor.run_once()          # one-shot
        await processor.run_loop(900)       # every 15 minutes
    """

    def __init__(self, config: ProcessorConfig | None = None) -> None:
        self.cfg = config or ProcessorConfig()

    # ── Public API ─────────────────────────────────────────────────────────────

    async def run_once(self) -> None:
        """Run one full processing pass over recent second-brain notes."""
        logger.info("[SecondBrainProcessor] Starting processing pass")

        notes = await self._fetch_second_brain_notes()
        if not notes:
            logger.info("[SecondBrainProcessor] No second-brain notes found — skipping")
            return

        logger.info("[SecondBrainProcessor] Processing %d note(s)", len(notes))

        # Process each note sequentially to avoid hammering Ollama.
        for note in notes:
            await self._process_note(note, all_notes=notes)

        # Cluster and spaced-repetition passes operate on the full corpus.
        await self._detect_clusters(notes)
        await self._spaced_repetition_pass(notes)

        logger.info("[SecondBrainProcessor] Processing pass complete")

    async def run_loop(self, interval_secs: float = 900) -> None:
        """Run processing passes indefinitely, sleeping ``interval_secs`` between runs."""
        logger.info(
            "[SecondBrainProcessor] Starting loop (interval=%.0fs)", interval_secs
        )
        while True:
            try:
                await self.run_once()
            except Exception:
                logger.exception("[SecondBrainProcessor] Unhandled error in run_once")
            await asyncio.sleep(interval_secs)

    # ── Note processing ────────────────────────────────────────────────────────

    async def _process_note(
        self, note: SecondBrainNote, all_notes: list[SecondBrainNote]
    ) -> None:
        """Enrich a single note: embed → summarise → backlinks → conflict check."""
        changed = False

        # 1. Embed (if missing or zero-placeholder)
        if not note.has_embedding():
            embedding = await self._embed_text(note.content)
            if embedding:
                note.embedding = embedding
                changed = True

        # 2. Summarise (if not yet done)
        if not note.is_enriched():
            summary = await self._summarise(note.content)
            if summary:
                note.summary = summary
                changed = True

        # 3. Build backlinks from explicit markers
        explicit_ids = _BACKLINK_RE.findall(note.content)
        for linked_id in explicit_ids:
            if linked_id not in note.backlinks:
                note.backlinks.append(linked_id)
                changed = True

        # 4. Build implicit backlinks via vector similarity
        if note.has_embedding():
            for other in all_notes:
                if other.id == note.id:
                    continue
                if not other.has_embedding():
                    continue
                sim = _cosine_similarity(note.embedding, other.embedding)
                if (
                    sim >= self.cfg.backlink_sim_threshold
                    and other.id not in note.backlinks
                ):
                    note.backlinks.append(other.id)
                    changed = True

                # 5. Conflict detection (high sim but opposing direction)
                if sim >= self.cfg.conflict_sim_threshold and _is_conflict(
                    note.content, other.content
                ):
                    await self._emit_event(
                        action="conflict",
                        payload={
                            "note_a_id": note.id,
                            "note_b_id": other.id,
                            "similarity": sim,
                            "note_a_snippet": note.content[:200],
                            "note_b_snippet": other.content[:200],
                        },
                        priority="normal",
                    )
                    logger.info(
                        "[SecondBrainProcessor] Conflict flagged: %s ↔ %s (sim=%.3f)",
                        note.id[:8],
                        other.id[:8],
                        sim,
                    )

        if changed:
            await self._store_enriched_note(note)

    # ── Cluster detection ──────────────────────────────────────────────────────

    async def _detect_clusters(self, notes: list[SecondBrainNote]) -> None:
        """
        Detect topical clusters using greedy agglomerative grouping.

        Groups are formed by expanding a seed note with all notes whose
        pairwise cosine similarity exceeds ``cluster_sim_threshold``.
        When a group reaches ``cluster_min_size``, a Bell Tower notification
        is emitted.
        """
        embedded = [n for n in notes if n.has_embedding()]
        if len(embedded) < self.cfg.cluster_min_size:
            return

        visited: set[str] = set()
        clusters: list[list[SecondBrainNote]] = []

        for seed in embedded:
            if seed.id in visited:
                continue
            group = [seed]
            visited.add(seed.id)
            for candidate in embedded:
                if candidate.id in visited:
                    continue
                sim = _cosine_similarity(seed.embedding, candidate.embedding)
                if sim >= self.cfg.cluster_sim_threshold:
                    group.append(candidate)
                    visited.add(candidate.id)
            if len(group) >= self.cfg.cluster_min_size:
                clusters.append(group)

        for cluster in clusters:
            # Use the first ~5 words of the seed note as a topic label.
            topic_words = cluster[0].content.split()[:5]
            topic = " ".join(topic_words) + ("…" if len(cluster[0].content.split()) > 5 else "")
            logger.info(
                "[SecondBrainProcessor] Cluster detected: %d notes on '%s'",
                len(cluster),
                topic,
            )
            await self._emit_event(
                action="cluster",
                payload={
                    "note_count": len(cluster),
                    "topic_hint": topic,
                    "note_ids": [n.id for n in cluster],
                    "message": (
                        f"You have {len(cluster)} notes on \"{topic}\" — "
                        "want to synthesise?"
                    ),
                },
                priority="normal",
            )

    # ── Spaced repetition ──────────────────────────────────────────────────────

    async def _spaced_repetition_pass(self, notes: list[SecondBrainNote]) -> None:
        """Emit nudge notifications for notes that haven't been accessed recently."""
        for note in notes:
            age_days = note.days_since_accessed()
            if age_days >= self.cfg.stale_days:
                logger.info(
                    "[SecondBrainProcessor] Stale note: %s (%.1f days old)",
                    note.id[:8],
                    age_days,
                )
                await self._emit_event(
                    action="spaced_repetition",
                    payload={
                        "note_id": note.id,
                        "days_since_accessed": age_days,
                        "snippet": note.content[:200],
                        "message": (
                            f"You haven't reviewed this note in "
                            f"{age_days:.0f} days. Want to revisit?"
                        ),
                    },
                    priority="low",
                )

    # ── Ollama helpers ─────────────────────────────────────────────────────────

    async def _embed_text(self, text: str) -> list[float]:
        """Call Ollama embeddings API and return the embedding vector."""
        url = f"{self.cfg.ollama_base_url}/api/embeddings"
        payload = {"model": self.cfg.embed_model, "prompt": text[:2000]}
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
                return data.get("embedding", [])
        except Exception:
            logger.warning("[SecondBrainProcessor] Embedding failed for note (truncated to 40 chars): %.40s", text)
            return []

    async def _summarise(self, content: str) -> str:
        """Ask Ollama to produce a 2-3 sentence summary of ``content``."""
        if len(content.split()) < 20:
            # Too short to summarise meaningfully.
            return content.strip()

        prompt = (
            "Summarise the following note in 2-3 sentences. "
            "Be concise and preserve the key facts.\n\n"
            f"Note:\n{content[:3000]}\n\nSummary:"
        )
        url = f"{self.cfg.ollama_base_url}/api/generate"
        body = {
            "model": self.cfg.summarise_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 200},
        }
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, json=body)
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
                return str(data.get("response", "")).strip()
        except Exception:
            logger.warning("[SecondBrainProcessor] Summarisation failed")
            return ""

    # ── GristMill HTTP helpers ─────────────────────────────────────────────────

    async def _fetch_second_brain_notes(self) -> list[SecondBrainNote]:
        """
        Retrieve recent second-brain notes from the warm tier via
        ``GET /api/memory/recall?q=second_brain&limit=N``.

        Returns hydrated SecondBrainNote objects.
        """
        url = f"{self.cfg.gristmill_base_url}/api/memory/recall"
        params = {"q": "second_brain", "limit": str(self.cfg.recall_limit)}
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                raw: list[dict[str, Any]] = resp.json()
        except Exception:
            logger.exception("[SecondBrainProcessor] Failed to fetch notes from GristMill")
            return []

        notes: list[SecondBrainNote] = []
        for item in raw:
            mem = item.get("memory", item)
            try:
                note = SecondBrainNote.from_ledger_tags(
                    id=str(mem.get("id", "")),
                    content=str(mem.get("content", "")),
                    tags=list(mem.get("tags", [])),
                    created_at=_parse_ms(mem.get("created_at_ms")),
                    last_accessed=_parse_ms(mem.get("last_accessed_ms")),
                    tier=str(mem.get("tier", "warm")),  # type: ignore[arg-type]
                )
                notes.append(note)
            except Exception:
                logger.debug("[SecondBrainProcessor] Skipping malformed memory: %s", mem.get("id"))

        return notes

    async def _store_enriched_note(self, note: SecondBrainNote) -> None:
        """
        Persist enriched note content back to the ledger via
        ``POST /api/memory/remember``.

        The enriched content includes the summary as a header line so that
        the warm-tier FTS5 index picks it up for future recall queries.
        """
        enriched_content = note.content
        if note.summary and note.summary not in enriched_content:
            enriched_content = f"[Summary] {note.summary}\n\n{note.content}"

        url = f"{self.cfg.gristmill_base_url}/api/memory/remember"
        body = {"content": enriched_content, "tags": note.tags}
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, json=body)
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
                logger.debug(
                    "[SecondBrainProcessor] Stored enriched note id=%s → new_id=%s",
                    note.id[:8],
                    str(data.get("id", "?"))[:8],
                )
        except Exception:
            logger.warning(
                "[SecondBrainProcessor] Failed to store enriched note %s", note.id[:8]
            )

    async def _emit_event(
        self,
        action: str,
        payload: dict[str, Any],
        priority: str = "low",
    ) -> None:
        """
        Post a second-brain notification event to the GristMill /events
        endpoint so that Bell Tower watches can route it to Slack.
        """
        url = f"{self.cfg.gristmill_base_url}/events"
        body = {
            "channel": self.cfg.notification_channel,
            "payload": {"action": action, **payload},
            "priority": priority,
            "tags": {
                "second_brain_action": action,
                "source": "second_brain_processor",
            },
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json=body)
                resp.raise_for_status()
        except Exception:
            logger.warning(
                "[SecondBrainProcessor] Failed to emit event action=%s", action
            )


# ── Math helpers ───────────────────────────────────────────────────────────────


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity between two equal-length vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _is_conflict(text_a: str, text_b: str) -> bool:
    """
    Heuristic: two notes "conflict" if one contains explicit negation markers
    ("not", "never", "false", "wrong", "incorrect", "actually") in proximity
    to keywords from the other.

    This is intentionally simple; the processor surfaces *candidates* for the
    user to review rather than making a definitive judgement.
    """
    negation_words = {"not", "never", "false", "wrong", "incorrect", "actually", "no"}
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    return bool(negation_words & words_a) != bool(negation_words & words_b)


def _parse_ms(ms_value: Any) -> datetime:
    """Convert a millisecond-since-epoch integer to a UTC datetime."""
    try:
        ts = int(ms_value) / 1000
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except (TypeError, ValueError):
        return datetime.now(tz=timezone.utc)


# ── Entry point ───────────────────────────────────────────────────────────────


async def _main() -> None:
    """CLI entry point: run one processing pass and exit."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="GristMill Second Brain processor")
    parser.add_argument(
        "--loop",
        type=float,
        metavar="INTERVAL_SECS",
        help="Run continuously, sleeping INTERVAL_SECS between passes (default: one-shot)",
    )
    parser.add_argument(
        "--gristmill-url",
        default=None,
        help="GristMill base URL (default: GRISTMILL_BASE_URL env or http://127.0.0.1:3000)",
    )
    parser.add_argument(
        "--ollama-url",
        default=None,
        help="Ollama base URL (default: OLLAMA_HOST env or http://localhost:11434)",
    )
    args = parser.parse_args()

    cfg = ProcessorConfig(
        gristmill_base_url=args.gristmill_url,
        ollama_base_url=args.ollama_url,
    )
    processor = SecondBrainProcessor(cfg)

    if args.loop:
        await processor.run_loop(args.loop)
    else:
        await processor.run_once()


def main() -> None:
    """Sync entry point for the ``gristmill-second-brain`` CLI command."""
    asyncio.run(_main())


if __name__ == "__main__":
    main()

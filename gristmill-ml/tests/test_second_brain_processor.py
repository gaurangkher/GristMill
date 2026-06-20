"""Tests for SecondBrainProcessor's GristMill HTTP client calls.

Regression coverage for a wire-format mismatch: _fetch_second_brain_notes
used to issue a GET request with query-string params
(?q=second_brain&limit=N) against /api/memory/recall, but that route is
only registered as POST and expects a JSON body of
{"query": ..., "limit": ...} (see
gristmill-integrations/src/dashboard/routes/memory.ts). The GET request
always 404'd, so the processor silently fetched zero notes on every run
(the failure is swallowed by a broad except Exception in
_fetch_second_brain_notes) — the entire Second Brain enrichment pipeline
was a no-op end to end.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from gristmill_ml.second_brain.processor import ProcessorConfig, SecondBrainProcessor


def _recall_response(items: list[dict]) -> httpx.Response:
    request = httpx.Request("POST", "http://127.0.0.1:3000/api/memory/recall")
    return httpx.Response(200, json=items, request=request)


@pytest.mark.asyncio
async def test_fetch_notes_posts_query_and_limit_as_json_body() -> None:
    processor = SecondBrainProcessor(ProcessorConfig(recall_limit=25))

    mock_post = AsyncMock(return_value=_recall_response([]))
    with patch.object(httpx.AsyncClient, "post", mock_post):
        await processor._fetch_second_brain_notes()

    mock_post.assert_awaited_once()
    args, kwargs = mock_post.call_args
    assert args[0].endswith("/api/memory/recall")
    assert kwargs["json"] == {"query": "second_brain", "limit": 25}


@pytest.mark.asyncio
async def test_fetch_notes_hydrates_ranked_memory_response() -> None:
    processor = SecondBrainProcessor(ProcessorConfig())

    raw = [
        {
            "memory": {
                "id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
                "content": "Remember to follow up on the Q3 roadmap",
                "tags": ["second_brain", "capture", "slack_ts:1713200000.0001"],
                "created_at_ms": 1_700_000_000_000,
                "last_accessed_ms": 1_700_000_500_000,
                "tier": "warm",
            },
            "score": 0.91,
            "sources": ["keyword"],
        }
    ]

    with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=_recall_response(raw))):
        notes = await processor._fetch_second_brain_notes()

    assert len(notes) == 1
    note = notes[0]
    assert note.id == "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    assert note.source == "slack_dm"
    assert note.slack_ts == "1713200000.0001"


@pytest.mark.asyncio
async def test_fetch_notes_returns_empty_list_on_http_error() -> None:
    processor = SecondBrainProcessor(ProcessorConfig())

    request = httpx.Request("POST", "http://127.0.0.1:3000/api/memory/recall")
    error_response = httpx.Response(404, json={"error": "not found"}, request=request)

    with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=error_response)):
        notes = await processor._fetch_second_brain_notes()

    assert notes == []

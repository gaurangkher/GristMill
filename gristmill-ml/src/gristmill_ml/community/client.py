"""CommunityRepoClient — opt-in client for the GristMill community adapter hub.

The community repository is **opt-in only**.  It is disabled by default and
must be explicitly enabled in ``~/.gristmill/config.yaml``::

    community:
      enabled: true
      endpoint: https://adapters.gristmill.dev   # optional override

All uploads are anonymous: ``BundleManifest.anonymized_id`` (a UUID4) is the
only identifier sent.  No query text, user data, or IP-linked metadata is
transmitted with adapter weights.

Environment-variable overrides:
    ``GRISTMILL_COMMUNITY_URL``   — override the endpoint URL
    ``GRISTMILL_COMMUNITY_TOKEN`` — optional Bearer token for private repos
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "https://adapters.gristmill.dev"
_UPLOAD_TIMEOUT_S = 120
_DOWNLOAD_TIMEOUT_S = 300


# ── Data transfer objects ─────────────────────────────────────────────────────


@dataclass
class AdapterMeta:
    """Summary of a community adapter returned by the list endpoint."""

    adapter_id: str  # server-assigned opaque ID
    anonymized_id: str  # UUID4 from the bundle manifest
    domain: str
    base_model: str
    validation_score: float
    record_count: int
    created_at: str  # ISO 8601
    tags: list
    download_url: str

    @classmethod
    def from_dict(cls, d: dict) -> "AdapterMeta":
        return cls(
            adapter_id=d["adapter_id"],
            anonymized_id=d["anonymized_id"],
            domain=d["domain"],
            base_model=d["base_model"],
            validation_score=float(d.get("validation_score", 0.0)),
            record_count=int(d.get("record_count", 0)),
            created_at=d.get("created_at", ""),
            tags=list(d.get("tags", [])),
            download_url=d["download_url"],
        )


# ── Client ────────────────────────────────────────────────────────────────────


class CommunityRepoClient:
    """HTTP client for the GristMill community adapter repository.

    Args:
        endpoint: Base URL of the community API.  Resolved from
            ``GRISTMILL_COMMUNITY_URL`` env var, then ``config.yaml``, then
            the built-in default.
        token: Optional Bearer token (for private/enterprise repos).
        enabled: If ``False`` every method raises ``RuntimeError`` immediately.
            Pass ``enabled=True`` only after confirming user opt-in.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        token: Optional[str] = None,
        enabled: bool = False,
    ) -> None:
        self._enabled = enabled
        self._endpoint = (
            os.environ.get("GRISTMILL_COMMUNITY_URL") or endpoint or _DEFAULT_ENDPOINT
        ).rstrip("/")
        self._token = os.environ.get("GRISTMILL_COMMUNITY_TOKEN") or token

    # ── Opt-in gate ───────────────────────────────────────────────────────────

    def _check_enabled(self) -> None:
        if not self._enabled:
            raise RuntimeError(
                "Community repository is disabled. "
                "Set community.enabled: true in ~/.gristmill/config.yaml "
                "to opt-in to adapter sharing."
            )

    def _headers(self) -> dict:
        h: dict = {"Accept": "application/json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    # ── API methods ───────────────────────────────────────────────────────────

    def push(self, gmpack_path: Path, *, dry_run: bool = False) -> str:
        """Upload a `.gmpack` bundle to the community repository.

        Args:
            gmpack_path: Path to the bundle produced by ``AdapterBundle.pack()``.
            dry_run: If ``True``, validate the bundle locally but skip the
                network upload (useful for CI / smoke tests).

        Returns:
            The server-assigned ``adapter_id`` string.

        Raises:
            RuntimeError: If community sharing is not enabled.
            OSError: If the bundle file cannot be read.
        """
        self._check_enabled()
        gmpack_path = Path(gmpack_path)
        if not gmpack_path.exists():
            raise OSError(f"Bundle not found: {gmpack_path}")

        from gristmill_ml.export.bundle import AdapterBundle

        manifest = AdapterBundle.read_manifest(gmpack_path)

        if dry_run:
            logger.info(
                "[dry-run] Would push bundle: domain=%s id=%s size=%d bytes",
                manifest.domain,
                manifest.anonymized_id,
                gmpack_path.stat().st_size,
            )
            return f"dry-run-{manifest.anonymized_id}"

        try:
            import urllib.request

            url = f"{self._endpoint}/v1/adapters"
            with gmpack_path.open("rb") as fh:
                data = fh.read()
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    **self._headers(),
                    "Content-Type": "application/octet-stream",
                    "X-GristMill-Domain": manifest.domain,
                    "X-GristMill-Anonymized-ID": manifest.anonymized_id,
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=_UPLOAD_TIMEOUT_S) as resp:
                body = resp.read().decode()
            import json as _json

            adapter_id: str = _json.loads(body).get("adapter_id", "unknown")
            logger.info(
                "Pushed adapter bundle: domain=%s adapter_id=%s",
                manifest.domain,
                adapter_id,
            )
            return adapter_id
        except Exception as exc:
            logger.error("Failed to push adapter bundle: %s", exc)
            raise

    def list_adapters(
        self,
        domain: str,
        *,
        min_score: float = 0.0,
        limit: int = 20,
    ) -> list[AdapterMeta]:
        """List community adapters available for *domain*.

        Args:
            domain: Domain tag to filter by (e.g. ``"code"``).
            min_score: Minimum ``validation_score`` to include.
            limit: Maximum number of results.

        Returns:
            List of :class:`AdapterMeta` sorted by ``validation_score`` desc.
        """
        self._check_enabled()
        try:
            import json as _json
            import urllib.parse
            import urllib.request

            params = urllib.parse.urlencode(
                {
                    "domain": domain,
                    "min_score": min_score,
                    "limit": limit,
                }
            )
            url = f"{self._endpoint}/v1/adapters?{params}"
            req = urllib.request.Request(url, headers=self._headers())
            with urllib.request.urlopen(req, timeout=30) as resp:
                items = _json.loads(resp.read())
            return [AdapterMeta.from_dict(item) for item in items]
        except Exception as exc:
            logger.error("Failed to list community adapters: %s", exc)
            raise

    def pull(
        self,
        domain: str,
        dest_path: Path,
        *,
        min_score: float = 0.7,
        adapter_id: Optional[str] = None,
    ) -> Optional[Path]:
        """Download the best community adapter for *domain*.

        Selects the highest-scored adapter that meets *min_score*.  If
        *adapter_id* is given, that specific adapter is fetched instead.

        Args:
            domain: Domain to fetch an adapter for.
            dest_path: Where to write the downloaded ``.gmpack`` file.
            min_score: Minimum acceptable validation score.
            adapter_id: Optional specific adapter to fetch (skips list step).

        Returns:
            Path to the downloaded ``.gmpack`` file, or ``None`` if no
            suitable adapter is available.
        """
        self._check_enabled()
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import urllib.request

            if adapter_id:
                download_url = f"{self._endpoint}/v1/adapters/{adapter_id}/download"
            else:
                adapters = self.list_adapters(domain, min_score=min_score, limit=1)
                if not adapters:
                    logger.info(
                        "No community adapters found for domain=%s (min_score=%.2f)",
                        domain,
                        min_score,
                    )
                    return None
                download_url = adapters[0].download_url

            logger.info("Downloading community adapter: %s → %s", download_url, dest_path)
            req = urllib.request.Request(download_url, headers=self._headers())
            tmp = dest_path.with_suffix(".gmpack.dl")
            with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT_S) as resp:
                tmp.write_bytes(resp.read())
            tmp.replace(dest_path)
            logger.info("Downloaded community adapter → %s", dest_path)
            return dest_path
        except Exception as exc:
            logger.error("Failed to download community adapter for domain=%s: %s", domain, exc)
            return None

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config_path: Optional[Path] = None) -> "CommunityRepoClient":
        """Build a client from ``~/.gristmill/config.yaml``.

        Reads the ``community`` section.  Returns a *disabled* client if the
        section is absent or ``enabled: false``.
        """
        import yaml  # type: ignore[import]

        if config_path is None:
            config_path = Path.home() / ".gristmill" / "config.yaml"

        cfg: dict = {}
        if config_path.exists():
            try:
                cfg = yaml.safe_load(config_path.read_text()) or {}
            except Exception as exc:
                logger.warning("Could not parse config: %s", exc)

        community = cfg.get("community", {})
        enabled = bool(community.get("enabled", False))
        endpoint = community.get("endpoint", None)
        token = community.get("token", None)

        if enabled:
            logger.info(
                "Community adapter repository: enabled (endpoint=%s)", endpoint or _DEFAULT_ENDPOINT
            )
        else:
            logger.debug("Community adapter repository: disabled (opt-in required)")

        return cls(endpoint=endpoint, token=token, enabled=enabled)

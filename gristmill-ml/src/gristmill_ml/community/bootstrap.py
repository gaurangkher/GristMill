"""ColdStartBootstrapper — accelerate cold start via community adapter seeding.

When a new GristMill user has fewer than ``min_records`` training examples for
a domain, the grinder for that domain has nothing to specialise on and falls
back entirely to the teacher LLM.  The bootstrapper short-circuits this wait:

    1. Check the local training buffer for record count per domain.
    2. If sparse, download the highest-rated community adapter for that domain.
    3. Unpack the bundle into the checkpoint staging directory.
    4. Promote it as an initial adapter (version 0 baseline).

The promoted adapter is *not* the user's personal model — it is a warm start
that the local distillation cycle will refine as real queries accumulate.

The bootstrapper is called by the trainer service on startup (if community
opt-in is enabled) and can also be triggered manually via the dashboard API.
"""

from __future__ import annotations

import logging
import sqlite3
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path("/gristmill/db/training_buffer.sqlite")
_FALLBACK_DB = Path.home() / ".gristmill" / "db" / "training_buffer.sqlite"
_DEFAULT_MIN_RECORDS = 50  # bootstrap if fewer records than this
_DEFAULT_MIN_SCORE = 0.70  # community adapter must meet this threshold


class ColdStartBootstrapper:
    """Seed a domain's grinder with a community adapter when local data is sparse.

    Args:
        checkpoint_root: Root of the checkpoint directory (contains active/,
            staging/, history/).  Defaults to the standard checkpoint path.
        db_path: Path to ``training_buffer.sqlite``.
        min_records: If the domain has fewer training records than this,
            bootstrapping is considered necessary.
        min_score: Minimum community-adapter ``validation_score`` to accept.
        community_client: A :class:`~gristmill_ml.community.client.CommunityRepoClient`
            instance.  If omitted, one is built from ``~/.gristmill/config.yaml``.
    """

    def __init__(
        self,
        checkpoint_root: Optional[Path] = None,
        db_path: Optional[Path] = None,
        min_records: int = _DEFAULT_MIN_RECORDS,
        min_score: float = _DEFAULT_MIN_SCORE,
        community_client=None,
    ) -> None:
        from gristmill_ml.trainer.checkpoint import CheckpointManager

        self._ckpt = CheckpointManager(root=checkpoint_root)
        self._db_path = db_path or (_DEFAULT_DB if _DEFAULT_DB.exists() else _FALLBACK_DB)
        self._min_records = min_records
        self._min_score = min_score

        if community_client is None:
            from gristmill_ml.community.client import CommunityRepoClient

            community_client = CommunityRepoClient.from_config()
        self._client = community_client

    # ── Public API ────────────────────────────────────────────────────────────

    def needs_bootstrap(self, domain: str) -> bool:
        """Return ``True`` if *domain* has fewer than ``min_records`` training examples.

        Also returns ``True`` if the training buffer DB does not yet exist
        (brand-new installation).
        """
        if not self._db_path.exists():
            logger.debug("Training buffer not found — bootstrap needed for domain=%s", domain)
            return True

        try:
            count = self._count_records(domain)
            if count < self._min_records:
                logger.info(
                    "domain=%s has %d training records (< %d) — bootstrap recommended",
                    domain,
                    count,
                    self._min_records,
                )
                return True
            logger.debug("domain=%s has %d records — no bootstrap needed", domain, count)
            return False
        except Exception as exc:
            logger.warning("Could not query training buffer (%s) — assuming bootstrap needed", exc)
            return True

    def bootstrap(self, domain: str, *, force: bool = False) -> bool:
        """Download and stage a community adapter for *domain*.

        Args:
            domain: Domain to bootstrap (e.g. ``"code"``).
            force: If ``True``, bootstrap even when ``needs_bootstrap()`` is
                ``False`` (used for manual re-seeding via the dashboard).

        Returns:
            ``True`` if a community adapter was successfully staged and
            promoted, ``False`` otherwise (not enough data, download failed,
            community disabled, etc.).
        """
        if not force and not self.needs_bootstrap(domain):
            return False

        if not self._client._enabled:
            logger.info("Skipping bootstrap for domain=%s — community repo not enabled", domain)
            return False

        # Skip if there's already a healthy active adapter for this domain
        if not force and self._ckpt.active_adapter_path(domain) is not None:
            logger.debug("domain=%s already has an active adapter — skipping bootstrap", domain)
            return False

        logger.info("Bootstrapping domain=%s from community repository …", domain)
        with tempfile.TemporaryDirectory(prefix="gm_bootstrap_") as tmp:
            tmp_path = Path(tmp)
            gmpack_dest = tmp_path / f"{domain}-bootstrap.gmpack"

            downloaded = self._client.pull(
                domain,
                dest_path=gmpack_dest,
                min_score=self._min_score,
            )
            if downloaded is None:
                logger.info("No suitable community adapter found for domain=%s", domain)
                return False

            # Unpack into a temp adapter dir
            from gristmill_ml.export.bundle import AdapterBundle

            adapter_dest = tmp_path / "adapter"
            try:
                manifest = AdapterBundle.unpack(downloaded, adapter_dest)
            except Exception as exc:
                logger.error("Failed to unpack community bundle for domain=%s: %s", domain, exc)
                return False

            # Stage and promote as version 0 baseline
            self._ckpt.write_staging(adapter_dest, domain=domain)
            version = self._ckpt.promote_staging(
                validation_score=manifest.validation_score,
                record_count=manifest.record_count,
                domain=domain,
            )

        logger.info(
            "Bootstrap complete: domain=%s promoted community adapter as v%d "
            "(validation_score=%.3f, community_id=%s)",
            domain,
            version,
            manifest.validation_score,
            manifest.anonymized_id,
        )
        return True

    def bootstrap_all_sparse(self) -> dict[str, bool]:
        """Bootstrap every known domain that is below the record threshold.

        Returns:
            Mapping of domain → bootstrap success/skip.
        """
        from gristmill_ml.trainer.checkpoint import KNOWN_DOMAINS

        results: dict[str, bool] = {}
        for domain in KNOWN_DOMAINS:
            if domain == "default":
                continue
            results[domain] = self.bootstrap(domain)
        return results

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _count_records(self, domain: str) -> int:
        """Query the training buffer for the number of PENDING + IN_TRAINING records for *domain*."""
        with sqlite3.connect(self._db_path) as conn:
            # The training buffer schema stores domain in a `domain` column.
            # Count both PENDING and IN_TRAINING as "usable" data.
            cur = conn.execute(
                "SELECT COUNT(*) FROM training_buffer "
                "WHERE domain = ? AND status IN ('PENDING', 'IN_TRAINING', 'CONSUMED')",
                (domain,),
            )
            row = cur.fetchone()
            return int(row[0]) if row else 0

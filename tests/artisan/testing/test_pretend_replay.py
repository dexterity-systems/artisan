"""Tests for experimental pretend replay utilities."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from artisan.schemas.orchestration.commit_config import CommitConfig
from artisan.storage.core.table_schemas import EXECUTIONS_SCHEMA
from artisan.testing.pretend_replay import run_pretend_replay


def _write_source_executions(delta_root, rows: int) -> None:
    now = datetime.now(UTC)
    source_df = pl.DataFrame(
        {
            "execution_run_id": [f"source_{i:03d}" for i in range(rows)],
            "execution_spec_id": [f"spec_{i:03d}" for i in range(rows)],
            "step_run_id": [None] * rows,
            "origin_step_number": [0] * rows,
            "operation_name": ["source"] * rows,
            "params": ["{}"] * rows,
            "user_overrides": ["{}"] * rows,
            "timestamp_start": [now] * rows,
            "timestamp_end": [now] * rows,
            "source_worker": [0] * rows,
            "compute_backend": ["local"] * rows,
            "success": [True] * rows,
            "error": [None] * rows,
            "tool_output": [None] * rows,
            "worker_log": [None] * rows,
            "metadata": ["{}"] * rows,
        },
        schema=EXECUTIONS_SCHEMA,
    )
    source_df.write_delta(
        str(delta_root / "orchestration/executions"),
        mode="overwrite",
    )


def test_schema_only_pretend_replay_stages_and_commits(tmp_path):
    """Schema-only replay writes normal staged dirs and commits them."""
    source_delta = tmp_path / "source"
    target_delta = tmp_path / "target"
    staging = tmp_path / "staging"
    _write_source_executions(source_delta, rows=3)

    results = run_pretend_replay(
        source_delta=source_delta,
        delta_root=target_delta,
        staging_root=staging,
        max_units=3,
        commit_config=CommitConfig(initial_chunk_size=2),
    )

    assert results["staged_units"] == 3
    assert results["executions"] == 3
    assert pl.read_delta(str(target_delta / "orchestration/executions")).shape[0] == 3
    assert list(staging.rglob("*.parquet")) == []


def test_exact_pretend_replay_mode_is_scaffolded(tmp_path):
    """Exact replay is explicit scaffold, not silently fake exact mode."""
    source_delta = tmp_path / "source"
    _write_source_executions(source_delta, rows=1)

    with pytest.raises(NotImplementedError, match="scaffolded"):
        run_pretend_replay(
            source_delta=source_delta,
            delta_root=tmp_path / "target",
            staging_root=tmp_path / "staging",
            mode="exact",
        )

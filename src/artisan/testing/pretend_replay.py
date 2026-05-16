"""Generate pretend staged outputs from an existing Delta run.

This module is intentionally experimental. It is for stressing orchestration,
staging, recovery, and commit behavior without launching scientific workers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from artisan.schemas.artifact.metric import MetricArtifact
from artisan.schemas.enums import TablePath
from artisan.schemas.orchestration.commit_config import CommitConfig
from artisan.storage.core.table_schemas import (
    ARTIFACT_INDEX_SCHEMA,
    EXECUTION_EDGES_SCHEMA,
    EXECUTIONS_SCHEMA,
)
from artisan.storage.io.commit import DeltaCommitter
from artisan.utils.path import shard_path

logger = logging.getLogger(__name__)


SUPPORTED_MODES = {"schema-only", "failure-injection"}
SCAFFOLDED_MODES = {"exact", "materialization-check"}


def _stable_id(prefix: str, index: int) -> str:
    """Return a deterministic 32-character hex id."""
    return hashlib.sha256(f"{prefix}:{index}".encode()).hexdigest()[:32]


def _source_execution_count(source_delta: Path) -> int:
    """Count source executions without collecting the full table."""
    executions_path = source_delta / TablePath.EXECUTIONS.value
    if not executions_path.exists():
        msg = f"source executions table not found: {executions_path}"
        raise FileNotFoundError(msg)
    return int(
        pl.scan_delta(str(executions_path))
        .select(pl.len().alias("rows"))
        .collect()
        .item()
    )


def _write_schema_only_unit(
    *,
    staging_root: Path,
    unit_index: int,
    step_number: int,
    operation_name: str,
    mode: str,
    failed: bool,
) -> None:
    """Write one pretend execution-run directory in normal staging format."""
    execution_run_id = _stable_id("pretend-execution", unit_index)
    artifact_id = _stable_id("pretend-artifact", unit_index)
    staging_path = shard_path(
        staging_root,
        execution_run_id,
        step_number=step_number,
        operation_name=operation_name,
    )
    staging_path.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    metadata = {
        "pretend_replay": True,
        "pretend_mode": mode,
        "unit_index": unit_index,
    }

    exec_df = pl.DataFrame(
        {
            "execution_run_id": [execution_run_id],
            "execution_spec_id": [_stable_id("pretend-spec", unit_index)],
            "step_run_id": [None],
            "origin_step_number": [step_number],
            "operation_name": [operation_name],
            "params": [json.dumps({"pretend_unit_index": unit_index})],
            "user_overrides": ["{}"],
            "timestamp_start": [now],
            "timestamp_end": [now],
            "source_worker": [0],
            "compute_backend": ["pretend"],
            "success": [not failed],
            "error": ["pretend failure injection" if failed else None],
            "tool_output": [None],
            "worker_log": [None],
            "metadata": [json.dumps(metadata)],
        },
        schema=EXECUTIONS_SCHEMA,
    )
    exec_df.write_parquet(staging_path / "executions.parquet", compression="zstd")

    if failed:
        return

    metric_df = pl.DataFrame(
        {
            "artifact_id": [artifact_id],
            "origin_step_number": [step_number],
            "content": [json.dumps({"pretend_score": unit_index}).encode()],
            "original_name": [f"pretend_metric_{unit_index}"],
            "extension": [".json"],
            "metadata": [json.dumps(metadata)],
            "external_path": [None],
        },
        schema=MetricArtifact.POLARS_SCHEMA,
    )
    metric_df.write_parquet(staging_path / "metrics.parquet", compression="zstd")

    index_df = pl.DataFrame(
        {
            "artifact_id": [artifact_id],
            "artifact_type": ["metric"],
            "origin_step_number": [step_number],
            "metadata": [json.dumps(metadata)],
        },
        schema=ARTIFACT_INDEX_SCHEMA,
    )
    index_df.write_parquet(staging_path / "index.parquet", compression="zstd")

    edge_df = pl.DataFrame(
        {
            "execution_run_id": [execution_run_id],
            "direction": ["output"],
            "role": ["metric"],
            "artifact_id": [artifact_id],
        },
        schema=EXECUTION_EDGES_SCHEMA,
    )
    edge_df.write_parquet(staging_path / "execution_edges.parquet", compression="zstd")


def run_pretend_replay(
    *,
    source_delta: Path,
    delta_root: Path,
    staging_root: Path,
    mode: str = "schema-only",
    worker_latency_ms: int = 0,
    failure_rate: float = 0.0,
    max_units: int = 0,
    step_number: int = 0,
    operation_name: str = "pretend_replay",
    commit: bool = True,
    commit_config: CommitConfig | None = None,
) -> dict[str, int]:
    """Generate pretend staged units and optionally commit them."""
    if mode in SCAFFOLDED_MODES:
        msg = f"pretend mode {mode!r} is scaffolded but not implemented yet"
        raise NotImplementedError(msg)
    if mode not in SUPPORTED_MODES:
        msg = f"unknown pretend mode {mode!r}; expected one of {sorted(SUPPORTED_MODES)}"
        raise ValueError(msg)
    if failure_rate < 0.0 or failure_rate > 1.0:
        msg = "failure_rate must be between 0.0 and 1.0"
        raise ValueError(msg)

    source_count = _source_execution_count(source_delta)
    unit_count = source_count if max_units <= 0 else min(source_count, max_units)
    rng = random.Random(0)
    logger.info(
        "Pretend replay staging: source=%s mode=%s units=%d target_delta=%s staging=%s",
        source_delta,
        mode,
        unit_count,
        delta_root,
        staging_root,
    )

    for unit_index in range(unit_count):
        failed = mode == "failure-injection" and rng.random() < failure_rate
        _write_schema_only_unit(
            staging_root=staging_root,
            unit_index=unit_index,
            step_number=step_number,
            operation_name=operation_name,
            mode=mode,
            failed=failed,
        )
        if worker_latency_ms > 0:
            time.sleep(worker_latency_ms / 1000)

    if not commit:
        return {"staged_units": unit_count}

    committer = DeltaCommitter(
        delta_root,
        staging_root,
        commit_config=commit_config or CommitConfig(),
    )
    commit_results = committer.commit_all_tables(
        cleanup_staging=True,
        step_number=step_number,
        operation_name=operation_name,
    )
    return {"staged_units": unit_count, **commit_results}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretend-from-delta", required=True, type=Path)
    parser.add_argument("--delta-root", required=True, type=Path)
    parser.add_argument("--staging-root", required=True, type=Path)
    parser.add_argument(
        "--pretend-mode",
        default="schema-only",
        choices=sorted(SUPPORTED_MODES | SCAFFOLDED_MODES),
    )
    parser.add_argument("--pretend-worker-latency-ms", default=0, type=int)
    parser.add_argument("--pretend-failure-rate", default=0.0, type=float)
    parser.add_argument("--pretend-max-units", default=0, type=int)
    parser.add_argument("--step-number", default=0, type=int)
    parser.add_argument("--operation-name", default="pretend_replay")
    parser.add_argument("--no-commit", action="store_true")
    parser.add_argument("--commit-initial-chunk-size", default=250, type=int)
    parser.add_argument("--commit-min-chunk-size", default=1, type=int)
    parser.add_argument("--commit-max-chunk-rows", default=None, type=int)
    parser.add_argument("--commit-max-chunk-bytes", default=None, type=int)
    parser.add_argument("--commit-max-memory-fraction", default=0.25, type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)
    commit_config = CommitConfig(
        initial_chunk_size=args.commit_initial_chunk_size,
        min_chunk_size=args.commit_min_chunk_size,
        max_commit_chunk_rows=args.commit_max_chunk_rows,
        max_commit_chunk_bytes=args.commit_max_chunk_bytes,
        max_commit_memory_fraction=args.commit_max_memory_fraction,
    )
    results = run_pretend_replay(
        source_delta=args.pretend_from_delta,
        delta_root=args.delta_root,
        staging_root=args.staging_root,
        mode=args.pretend_mode,
        worker_latency_ms=args.pretend_worker_latency_ms,
        failure_rate=args.pretend_failure_rate,
        max_units=args.pretend_max_units,
        step_number=args.step_number,
        operation_name=args.operation_name,
        commit=not args.no_commit,
        commit_config=commit_config,
    )
    sys.stdout.write(json.dumps(results, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

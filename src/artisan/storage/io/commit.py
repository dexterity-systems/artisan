"""Commit staged Parquet files to Delta Lake tables.

The orchestrator is the single writer; workers stage Parquet files (see
``staging.py``) and ``DeltaCommitter`` merges them into Delta tables
with content-addressed deduplication, partitioning, and optional
compaction/vacuum.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
from deltalake import DeltaTable, WriterProperties

from artisan.schemas.artifact.registry import ArtifactTypeDef
from artisan.schemas.enums import TablePath
from artisan.schemas.orchestration.commit_config import CommitConfig
from artisan.storage.core.table_schemas import (
    FRAMEWORK_SCHEMAS,
    NON_PARTITIONED_TABLES,
)
from artisan.storage.io.staging import StagingManager

logger = logging.getLogger(__name__)

# Default writer properties for Delta Lake writes
# Using zstd compression for good compression ratio and performance
DEFAULT_WRITER_PROPERTIES = WriterProperties(compression="ZSTD")


def _to_str(table: str) -> str:
    """Coerce a TablePath enum or plain string to its string value."""
    return table.value if hasattr(table, "value") else table


def _table_name_from_path(table_path: str) -> str:
    """Extract table name (last segment) from a table path string."""
    return Path(table_path).name


def _get_commit_order() -> list[str]:
    """Build the table commit order from the current artifact registry.

    Order ensures referential integrity on partial failure: content
    tables first, then index, then provenance edges, then executions.
    """
    artifact_paths = [td.table_path for td in ArtifactTypeDef.get_all().values()]
    framework_paths = [
        TablePath.ARTIFACT_INDEX.value,
        TablePath.ARTIFACT_EDGES.value,
        TablePath.EXECUTION_EDGES.value,
        TablePath.EXECUTIONS.value,
    ]
    return [*artifact_paths, *framework_paths]


class CommitFailureError(RuntimeError):
    """Raised when a staged execution directory cannot be fully committed."""

    def __init__(
        self,
        *,
        execution_run_id: str,
        run_dir: Path,
        table_name: str,
        cause: Exception | str,
        partial_results: dict[str, int] | None = None,
    ) -> None:
        self.execution_run_id = execution_run_id
        self.run_dir = run_dir
        self.table_name = table_name
        self.partial_results = partial_results or {}
        detail = (
            cause
            if isinstance(cause, str)
            else f"{type(cause).__name__}: {cause}"
        )
        super().__init__(
            f"Commit failed for execution_run_id={execution_run_id} "
            f"at table={table_name}: {detail}"
        )


@dataclass(frozen=True)
class ChunkEstimate:
    """Conservative size estimate for a staged commit chunk."""

    rows_by_table: dict[str, int] = field(default_factory=dict)
    bytes_by_table: dict[str, int] = field(default_factory=dict)

    @property
    def total_rows(self) -> int:
        return sum(self.rows_by_table.values())

    @property
    def total_bytes(self) -> int:
        return sum(self.bytes_by_table.values())

    def estimated_memory_bytes(self, multiplier: float) -> int:
        return int(self.total_bytes * multiplier)


class ChunkCommitFailureError(CommitFailureError):
    """Raised when a staged execution-dir chunk cannot be fully committed."""

    def __init__(
        self,
        *,
        chunk_index: int,
        run_dirs: list[Path],
        table_name: str,
        cause: Exception | str,
        partial_results: dict[str, int] | None = None,
        estimate: ChunkEstimate | None = None,
    ) -> None:
        self.chunk_index = chunk_index
        self.run_dirs = run_dirs
        self.execution_run_ids = [path.name for path in run_dirs]
        self.estimate = estimate
        first_dir = run_dirs[0] if run_dirs else Path("<none>")
        first_id = self.execution_run_ids[0] if self.execution_run_ids else "<none>"
        super().__init__(
            execution_run_id=first_id,
            run_dir=first_dir,
            table_name=table_name,
            cause=cause,
            partial_results=partial_results,
        )
        detail = (
            cause
            if isinstance(cause, str)
            else f"{type(cause).__name__}: {cause}"
        )
        ids = _format_execution_ids(self.execution_run_ids)
        estimate_part = ""
        if estimate is not None:
            estimate_part = (
                f", estimated_rows={estimate.total_rows}, "
                f"estimated_bytes={estimate.total_bytes}"
            )
        RuntimeError.__init__(
            self,
            f"Chunk commit failed for chunk={chunk_index}, table={table_name}, "
            f"execution_run_ids={ids}{estimate_part}: {detail}",
        )


def _format_execution_ids(execution_run_ids: list[str], sample_size: int = 3) -> str:
    """Format execution-run IDs for concise progress/error logging."""
    if not execution_run_ids:
        return "[]"
    if len(execution_run_ids) <= sample_size * 2:
        return "[" + ", ".join(execution_run_ids) + "]"
    head = ", ".join(execution_run_ids[:sample_size])
    tail = ", ".join(execution_run_ids[-sample_size:])
    return f"[{head}, ..., {tail}]"


def _available_memory_bytes() -> int | None:
    """Return available system memory without requiring psutil."""
    try:
        import psutil  # type: ignore[import-untyped]

        return int(psutil.virtual_memory().available)
    except Exception:
        pass

    meminfo = Path("/proc/meminfo")
    try:
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                parts = line.split()
                return int(parts[1]) * 1024
    except OSError:
        return None
    return None


class DeltaCommitter:
    """Commit staged Parquet files to Delta Lake tables.

    Attributes:
        delta_base_path (Path): Root directory for Delta Lake tables.
        staging_manager (StagingManager): Manages staged Parquet files.
    """

    def __init__(
        self,
        delta_base_path: Path | str,
        staging_dir: Path | str,
        *,
        commit_config: CommitConfig | None = None,
    ):
        """Initialize with Delta Lake root and staging directories.

        Args:
            delta_base_path: Root directory for Delta Lake tables.
            staging_dir: Root directory containing worker-staged
                Parquet files.
            commit_config: Optional tuning for staged commit chunking.
        """
        self.delta_base_path = Path(delta_base_path)
        self.staging_manager = StagingManager(staging_dir)
        self.commit_config = commit_config or CommitConfig()

    def _table_path(self, table: str) -> Path:
        """Resolve the filesystem path for a Delta table."""
        return self.delta_base_path / table

    def _is_non_partitioned(self, table: str) -> bool:
        """Check if a table should not be partitioned."""
        return any(_to_str(npt) == _to_str(table) for npt in NON_PARTITIONED_TABLES)

    def _has_artifact_id(self, table: str) -> bool:
        """Check if the table supports artifact_id deduplication."""
        return _to_str(table) != _to_str(TablePath.EXECUTION_EDGES)

    def _default_partition_by(self, table: str) -> list[str] | None:
        """Return the default partition columns for a table."""
        if self._is_non_partitioned(table):
            return None
        return ["origin_step_number"]

    def _dedupe_key_column(self, table: str, df: pl.DataFrame) -> str | None:
        """Return the primary dedupe key for a staged table."""
        table_str = _to_str(table)
        if table_str in {
            TablePath.EXECUTIONS.value,
            TablePath.EXECUTION_EDGES.value,
            TablePath.ARTIFACT_EDGES.value,
        } and "execution_run_id" in df.columns:
            return "execution_run_id"
        if "artifact_id" in df.columns and self._has_artifact_id(table):
            return "artifact_id"
        return None

    # -------------------------------------------------------------------------
    # Commit operations
    # -------------------------------------------------------------------------

    def commit_table(
        self,
        table: str,
        deduplicate: bool = True,
        partition_by: list[str] | None = None,
        step_number: int | None = None,
        operation_name: str | None = None,
    ) -> int:
        """Commit staged Parquet data for one table to Delta Lake.

        Args:
            table: Table path (e.g. ``"artifacts/data"`` or a
                ``TablePath`` member).
            deduplicate: Skip rows whose ``artifact_id`` already
                exists in the target table.
            partition_by: Partition columns. Defaults to
                ``["origin_step_number"]`` for partitioned tables.
            step_number: Restrict to staged files from this step
                directory. None commits from all directories.
            operation_name: Human-readable step directory suffix used
                alongside ``step_number``.

        Returns:
            Number of rows written. Zero when nothing was staged or
            all rows were deduplicated.
        """
        table_name = _table_name_from_path(_to_str(table))
        staged_df = self.staging_manager.read_all_staged_for_table(
            table_name,
            step_number=step_number,
            operation_name=operation_name,
        )
        if staged_df is None or staged_df.is_empty():
            return 0
        return self.commit_dataframe(
            staged_df,
            table,
            deduplicate=deduplicate,
            partition_by=partition_by,
        )

    def commit_execution_run_dir(
        self,
        run_dir: Path | str,
        *,
        cleanup_after: bool = True,
    ) -> dict[str, int]:
        """Commit a single staged execution directory to Delta Lake.

        The directory must contain ``executions.parquet`` to be treated
        as a complete execution record. On any failure, the directory is
        preserved so recovery can retry safely.
        """
        run_dir_path = Path(run_dir)
        if not run_dir_path.exists():
            return {}

        execution_run_id = run_dir_path.name
        if not (run_dir_path / "executions.parquet").exists():
            raise CommitFailureError(
                execution_run_id=execution_run_id,
                run_dir=run_dir_path,
                table_name="executions",
                cause="missing required executions.parquet",
            )

        results: dict[str, int] = {}
        for table in _get_commit_order():
            table_name = _table_name_from_path(_to_str(table))
            try:
                staged_df = self.staging_manager.read_staged_table(run_dir_path, table_name)
                if staged_df is None or staged_df.is_empty():
                    continue
                rows_committed = self.commit_dataframe(staged_df, table)
            except CommitFailureError:
                raise
            except Exception as exc:
                raise CommitFailureError(
                    execution_run_id=execution_run_id,
                    run_dir=run_dir_path,
                    table_name=table_name,
                    cause=exc,
                    partial_results=results,
                ) from exc

            if rows_committed > 0:
                results[table_name] = rows_committed

        if cleanup_after:
            self.staging_manager.cleanup_execution_run_dir(run_dir_path)

        return results

    def commit_execution_run_dir_chunk(
        self,
        run_dirs: list[Path | str],
        *,
        cleanup_after: bool = True,
        chunk_index: int = 1,
        estimate: ChunkEstimate | None = None,
    ) -> dict[str, int]:
        """Commit a bounded chunk of staged execution directories.

        All staged dirs in the chunk are preserved on any table failure.
        Cleanup happens only after every table in the chunk has committed.
        """
        run_dir_paths = [Path(path) for path in run_dirs if Path(path).exists()]
        if not run_dir_paths:
            return {}

        if estimate is None:
            estimate = self._estimate_chunk(run_dir_paths)

        self._validate_chunk_run_dirs(run_dir_paths, chunk_index, estimate)

        results: dict[str, int] = {}
        execution_ids = [path.name for path in run_dir_paths]
        logger.info(
            "Committing staged chunk %d: dirs=%d execution_run_ids=%s "
            "estimated_rows=%d estimated_bytes=%d rows_by_table=%s",
            chunk_index,
            len(run_dir_paths),
            _format_execution_ids(execution_ids),
            estimate.total_rows,
            estimate.total_bytes,
            estimate.rows_by_table,
        )

        chunk_start = time.monotonic()
        for table in _get_commit_order():
            table_name = _table_name_from_path(_to_str(table))
            try:
                staged_df = self._read_staged_table_for_run_dirs(
                    run_dir_paths,
                    table_name,
                )
                if staged_df is None or staged_df.is_empty():
                    continue
                table_start = time.monotonic()
                rows_committed = self.commit_dataframe(staged_df, table)
                table_elapsed = time.monotonic() - table_start
            except ChunkCommitFailureError:
                raise
            except Exception as exc:
                logger.exception(
                    "Staged chunk %d failed at table %s for execution_run_ids=%s",
                    chunk_index,
                    table_name,
                    _format_execution_ids(execution_ids),
                )
                raise ChunkCommitFailureError(
                    chunk_index=chunk_index,
                    run_dirs=run_dir_paths,
                    table_name=table_name,
                    cause=exc,
                    partial_results=results,
                    estimate=estimate,
                ) from exc

            if rows_committed > 0:
                results[table_name] = rows_committed
            logger.info(
                "Committed staged chunk %d table=%s staged_rows=%d "
                "committed_rows=%d elapsed=%.2fs",
                chunk_index,
                table_name,
                staged_df.shape[0],
                rows_committed,
                table_elapsed,
            )

        if cleanup_after:
            for run_dir in run_dir_paths:
                self.staging_manager.cleanup_execution_run_dir(run_dir)

        elapsed = time.monotonic() - chunk_start
        throughput = len(run_dir_paths) / elapsed * 60 if elapsed > 0 else 0.0
        logger.info(
            "Finished staged chunk %d: dirs=%d elapsed=%.2fs throughput=%.1f dirs/min",
            chunk_index,
            len(run_dir_paths),
            elapsed,
            throughput,
        )

        return results

    def commit_all_tables(
        self,
        cleanup_staging: bool = True,
        step_number: int | None = None,
        operation_name: str | None = None,
    ) -> dict[str, int]:
        """Commit all staged data across every table to Delta Lake.

        Each table is committed independently; Delta Lake does not
        support multi-table transactions. The commit order (content
        tables, index, edges, executions) minimises referential
        integrity issues on partial failure.

        Args:
            cleanup_staging: Remove staging files after a successful
                commit.
            step_number: Restrict to staged files from this step
                directory. None commits from all directories.
            operation_name: Human-readable step directory suffix used
                alongside ``step_number``.

        Returns:
            Mapping of table name to rows committed. Tables with zero
            rows are omitted.

        Raises:
            CommitFailureError: If any staged execution directory fails
                to commit. Successful sibling directories are preserved
                only according to ``cleanup_staging``.
        """
        results: dict[str, int] = {}
        run_dirs = self.staging_manager.iter_execution_run_dirs(
            step_number=step_number,
            operation_name=operation_name,
        )
        if not run_dirs:
            return results

        if not self.commit_config.enabled:
            for run_dir in run_dirs:
                run_results = self.commit_execution_run_dir(
                    run_dir,
                    cleanup_after=cleanup_staging,
                )
                for table_name, rows_committed in run_results.items():
                    results[table_name] = results.get(table_name, 0) + rows_committed

            if results:
                parts = [f"{name}={count}" for name, count in results.items()]
                if parts:
                    logger.debug(
                        "Step %d commit: %s",
                        step_number or 0,
                        ", ".join(parts),
                    )

            return results

        start = time.monotonic()
        remaining = list(run_dirs)
        chunk_index = 0
        logger.info(
            "Chunked staged commit starting: dirs=%d initial_chunk_size=%d "
            "min_chunk_size=%d cleanup=%s",
            len(remaining),
            self.commit_config.initial_chunk_size,
            self.commit_config.min_chunk_size,
            cleanup_staging,
        )

        while remaining:
            chunk_index += 1
            chunk_run_dirs, estimate = self._select_commit_chunk(
                remaining,
                chunk_index,
            )
            run_results = self.commit_execution_run_dir_chunk(
                chunk_run_dirs,
                cleanup_after=cleanup_staging,
                chunk_index=chunk_index,
                estimate=estimate,
            )
            for table_name, rows_committed in run_results.items():
                results[table_name] = results.get(table_name, 0) + rows_committed
            remaining = remaining[len(chunk_run_dirs) :]

        if results:
            parts = [f"{name}={count}" for name, count in results.items()]
            if parts:
                logger.debug("Step %d commit: %s", step_number or 0, ", ".join(parts))

        elapsed = time.monotonic() - start
        throughput = len(run_dirs) / elapsed * 60 if elapsed > 0 else 0.0
        logger.info(
            "Chunked staged commit finished: dirs=%d chunks=%d elapsed=%.2fs "
            "throughput=%.1f dirs/min rows=%s",
            len(run_dirs),
            chunk_index,
            elapsed,
            throughput,
            results,
        )

        return results

    def recover_staged(self, *, preserve_staging: bool = False) -> dict[str, int]:
        """Commit leftover staging files from a prior crashed run.

        Idempotent: content-addressed deduplication skips rows that
        already exist in Delta.

        Args:
            preserve_staging: Keep staging files after commit instead
                of cleaning them up.

        Returns:
            Mapping of table name to rows committed. Empty dict when
            no leftover staging files are found.
        """
        if not self.staging_manager.staging_dir.exists():
            return {}

        run_dirs = self.staging_manager.iter_execution_run_dirs()
        if not run_dirs:
            return {}

        logger.debug(
            "Staged recovery: found %d leftover staged run directorie(s), committing...",
            len(run_dirs),
        )

        results = self.commit_all_tables(
            cleanup_staging=not preserve_staging,
            step_number=None,
        )

        if results:
            parts = [f"{name}={count}" for name, count in results.items()]
            logger.debug("Staged recovery committed: %s", ", ".join(parts))
        else:
            logger.debug("Staged recovery: no new rows to commit")

        return results

    def commit_batch(self, batch_id: str, cleanup_after: bool = True) -> dict[str, int]:
        """Commit a single staging batch to Delta Lake.

        Args:
            batch_id: Identifies the batch subdirectory to commit.
            cleanup_after: Remove the batch staging directory after a
                successful commit.

        Returns:
            Mapping of table name to rows committed. Tables with zero
            rows are omitted.
        """
        results = {}
        batch_dir = self.staging_manager.staging_dir / batch_id

        if not batch_dir.exists():
            return results

        for table in _get_commit_order():
            table_name = _table_name_from_path(_to_str(table))
            parquet_path = batch_dir / f"{table_name}.parquet"
            if parquet_path.exists():
                staged_df = pl.read_parquet(parquet_path)
                if not staged_df.is_empty():
                    rows = self.commit_dataframe(staged_df, table)
                    if rows > 0:
                        results[table_name] = rows

        if cleanup_after:
            self.staging_manager.cleanup_batch(batch_id)

        return results

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _validate_chunk_run_dirs(
        self,
        run_dirs: list[Path],
        chunk_index: int,
        estimate: ChunkEstimate,
    ) -> None:
        """Ensure every candidate run dir has the required execution record."""
        for run_dir in run_dirs:
            if not (run_dir / "executions.parquet").exists():
                raise ChunkCommitFailureError(
                    chunk_index=chunk_index,
                    run_dirs=run_dirs,
                    table_name="executions",
                    cause=f"missing required executions.parquet in {run_dir}",
                    estimate=estimate,
                )

    def _read_staged_table_for_run_dirs(
        self,
        run_dirs: list[Path],
        table_name: str,
    ) -> pl.DataFrame | None:
        """Read one table for a selected chunk, raising on unreadable files."""
        dfs: list[pl.DataFrame] = []
        for run_dir in run_dirs:
            staged_df = self.staging_manager.read_staged_table(run_dir, table_name)
            if staged_df is not None and not staged_df.is_empty():
                dfs.append(staged_df)
        if not dfs:
            return None
        return pl.concat(dfs, rechunk=True)

    def _select_commit_chunk(
        self,
        remaining_run_dirs: list[Path],
        chunk_index: int,
    ) -> tuple[list[Path], ChunkEstimate]:
        """Choose the largest safe prefix of remaining staged run dirs."""
        config = self.commit_config
        chunk_size = min(config.initial_chunk_size, len(remaining_run_dirs))
        effective_min = min(config.min_chunk_size, len(remaining_run_dirs))
        last_estimate: ChunkEstimate | None = None
        last_violations: list[str] = []

        while chunk_size >= effective_min:
            candidate = remaining_run_dirs[:chunk_size]
            estimate = self._estimate_chunk(candidate)
            violations = self._chunk_limit_violations(estimate)
            if not violations:
                if chunk_size < min(config.initial_chunk_size, len(remaining_run_dirs)):
                    logger.info(
                        "Staged chunk %d shrunk to %d dirs: estimated_rows=%d "
                        "estimated_bytes=%d",
                        chunk_index,
                        chunk_size,
                        estimate.total_rows,
                        estimate.total_bytes,
                    )
                return candidate, estimate

            last_estimate = estimate
            last_violations = violations
            next_size = max(effective_min, chunk_size // 2)
            if next_size == chunk_size:
                break
            logger.info(
                "Staged chunk %d candidate too large; shrinking %d -> %d dirs: %s",
                chunk_index,
                chunk_size,
                next_size,
                "; ".join(violations),
            )
            chunk_size = next_size

        estimate = last_estimate or self._estimate_chunk(
            remaining_run_dirs[:effective_min]
        )
        violations = last_violations or self._chunk_limit_violations(estimate)
        msg = (
            f"minimum chunk size {effective_min} exceeds commit limits: "
            f"{'; '.join(violations) if violations else 'unknown limit'}"
        )
        raise ChunkCommitFailureError(
            chunk_index=chunk_index,
            run_dirs=remaining_run_dirs[:effective_min],
            table_name="probe",
            cause=msg,
            estimate=estimate,
        )

    def _estimate_chunk(self, run_dirs: list[Path]) -> ChunkEstimate:
        """Estimate staged rows and bytes for a chunk before any Delta write."""
        rows_by_table: dict[str, int] = {}
        bytes_by_table: dict[str, int] = {}

        for table in _get_commit_order():
            table_name = _table_name_from_path(_to_str(table))
            row_count = 0
            byte_count = 0
            for run_dir in run_dirs:
                parquet_path = run_dir / f"{table_name}.parquet"
                if not parquet_path.exists():
                    continue
                byte_count += parquet_path.stat().st_size
                row_count += self._parquet_row_count(parquet_path)
            if row_count or byte_count:
                rows_by_table[table_name] = row_count
                bytes_by_table[table_name] = byte_count

        return ChunkEstimate(rows_by_table=rows_by_table, bytes_by_table=bytes_by_table)

    @staticmethod
    def _parquet_row_count(parquet_path: Path) -> int:
        """Count rows through lazy Parquet metadata before loading file data."""
        return int(
            pl.scan_parquet(str(parquet_path))
            .select(pl.len().alias("rows"))
            .collect()
            .item()
        )

    def _chunk_limit_violations(self, estimate: ChunkEstimate) -> list[str]:
        """Return reasons a chunk estimate exceeds configured limits."""
        config = self.commit_config
        violations: list[str] = []

        if (
            config.max_commit_chunk_rows is not None
            and estimate.total_rows > config.max_commit_chunk_rows
        ):
            violations.append(
                f"rows {estimate.total_rows} > {config.max_commit_chunk_rows}"
            )

        if (
            config.max_commit_chunk_bytes is not None
            and estimate.total_bytes > config.max_commit_chunk_bytes
        ):
            violations.append(
                f"bytes {estimate.total_bytes} > {config.max_commit_chunk_bytes}"
            )

        if config.max_commit_memory_fraction is not None:
            available = _available_memory_bytes()
            if available is not None:
                budget = int(available * config.max_commit_memory_fraction)
                estimated = estimate.estimated_memory_bytes(
                    config.parquet_memory_multiplier
                )
                if estimated > budget:
                    violations.append(
                        f"estimated_memory {estimated} > budget {budget}"
                    )
            else:
                logger.debug(
                    "Skipping commit memory-fraction limit; available memory unknown"
                )

        return violations

    def _deduplicate_rows(
        self,
        df: pl.DataFrame,
        table_path: Path,
        key_column: str,
    ) -> pl.DataFrame:
        """Remove rows whose dedupe key already exists in Delta."""
        if not table_path.exists():
            return df

        key_values = (
            df.get_column(key_column)
            .drop_nulls()
            .unique()
            .to_list()
        )
        if not key_values:
            return df

        existing_ids = (
            pl.scan_delta(str(table_path))
            .filter(pl.col(key_column).is_in(key_values))
            .select(key_column)
            .collect()
        )

        if existing_ids.is_empty():
            return df

        return df.join(existing_ids, on=key_column, how="anti")

    def commit_dataframe(
        self,
        df: pl.DataFrame,
        table: str,
        deduplicate: bool = True,
        partition_by: list[str] | None = None,
    ) -> int:
        """Write a single DataFrame directly to a Delta table.

        Args:
            df: Data to write (appended to existing table or creates
                a new one).
            table: Target table path string or ``TablePath`` member.
            deduplicate: Skip rows whose primary key already exists in
                the target table.
            partition_by: Partition columns. Defaults to
                ``["origin_step_number"]`` for partitioned tables.

        Returns:
            Number of rows written. Zero when the DataFrame is empty
            or all rows were deduplicated.
        """
        table_path = self._table_path(_to_str(table))
        staged_df = df
        dedupe_key = self._dedupe_key_column(table, staged_df)
        if deduplicate and dedupe_key is not None:
            staged_df = self._deduplicate_rows(staged_df, table_path, dedupe_key)
            if staged_df.is_empty():
                return 0

        if partition_by is None:
            partition_by = self._default_partition_by(table)

        if table_path.exists():
            staged_df.write_delta(
                str(table_path),
                mode="append",
                delta_write_options={
                    "writer_properties": DEFAULT_WRITER_PROPERTIES,
                    "schema_mode": "merge",
                },
            )
        else:
            write_opts = {"writer_properties": DEFAULT_WRITER_PROPERTIES}
            if partition_by:
                write_opts["partition_by"] = partition_by
            staged_df.write_delta(
                str(table_path),
                mode="overwrite",
                delta_write_options=write_opts,
            )

        return staged_df.shape[0]

    # -------------------------------------------------------------------------
    # Table management
    # -------------------------------------------------------------------------

    def initialize_tables(self) -> None:
        """Create empty Delta tables for all framework and artifact types.

        Skip tables that already exist. Useful for bootstrapping a new
        pipeline database.
        """
        # Initialize framework tables
        for table, schema in FRAMEWORK_SCHEMAS.items():
            table_path = self._table_path(_to_str(table))
            if not table_path.exists():
                empty_df = pl.DataFrame(schema=schema)
                if self._is_non_partitioned(table):
                    partition_by = None
                else:
                    partition_by = ["origin_step_number"]
                write_opts = {"writer_properties": DEFAULT_WRITER_PROPERTIES}
                if partition_by:
                    write_opts["partition_by"] = partition_by
                empty_df.write_delta(
                    str(table_path),
                    mode="overwrite",
                    delta_write_options=write_opts,
                )

        # Initialize artifact content tables from registry
        for type_def in ArtifactTypeDef.get_all().values():
            table_path = self._table_path(type_def.table_path)
            if not table_path.exists():
                empty_df = pl.DataFrame(schema=type_def.polars_schema())
                write_opts = {
                    "writer_properties": DEFAULT_WRITER_PROPERTIES,
                    "partition_by": ["origin_step_number"],
                }
                empty_df.write_delta(
                    str(table_path),
                    mode="overwrite",
                    delta_write_options=write_opts,
                )

    def compact_table(
        self,
        table: str,
        z_order_columns: list[str] | None = None,
        step_number: int | None = None,
    ) -> dict[str, int]:
        """Compact a Delta table, optionally applying Z-ORDER clustering.

        Args:
            table: Table path string or ``TablePath`` member.
            z_order_columns: Columns to cluster by. None performs a
                simple file compaction without ordering.
            step_number: Restrict compaction to this partition. None
                compacts the entire table.

        Returns:
            Dict with ``files_added`` and ``files_removed`` counts.
        """
        table_path = self._table_path(_to_str(table))
        if not table_path.exists():
            return {"files_added": 0, "files_removed": 0}

        dt = DeltaTable(str(table_path))

        partition_filters = None
        if step_number is not None and not self._is_non_partitioned(table):
            partition_filters = [("origin_step_number", "=", str(step_number))]

        if z_order_columns:
            result = dt.optimize.z_order(
                columns=z_order_columns,
                partition_filters=partition_filters,
            )
        else:
            result = dt.optimize.compact(partition_filters=partition_filters)

        return {
            "files_added": result.get("numFilesAdded", 0),
            "files_removed": result.get("numFilesRemoved", 0),
        }

    def compact_all_tables(
        self,
        z_order: bool = True,
        step_number: int | None = None,
    ) -> dict[str, dict[str, int]]:
        """Compact every Delta table, optionally with Z-ORDER clustering.

        Args:
            z_order: Apply Z-ORDER clustering on each table's key
                columns during compaction.
            step_number: Restrict compaction to this partition. None
                compacts all partitions.

        Returns:
            Mapping of table name to compaction statistics. Tables
            with no file changes are omitted.
        """
        results = {}

        # Build Z-ORDER config from registry + framework tables
        zorder_config: dict[str, list[str]] = {}

        # Artifact tables from registry
        for type_def in ArtifactTypeDef.get_all().values():
            zorder_config[type_def.table_path] = ["artifact_id"]

        # Framework tables
        zorder_config[_to_str(TablePath.EXECUTIONS)] = ["execution_spec_id"]
        zorder_config[_to_str(TablePath.ARTIFACT_INDEX)] = ["artifact_id"]
        zorder_config[_to_str(TablePath.ARTIFACT_EDGES)] = [
            "source_artifact_id",
            "target_artifact_id",
        ]
        zorder_config[_to_str(TablePath.EXECUTION_EDGES)] = ["execution_run_id"]
        zorder_config[_to_str(TablePath.STEPS)] = ["step_spec_id"]

        for table, z_order_cols in zorder_config.items():
            table_name = _table_name_from_path(table)
            stats = self.compact_table(
                table,
                z_order_columns=z_order_cols if z_order else None,
                step_number=step_number,
            )
            if stats["files_added"] > 0 or stats["files_removed"] > 0:
                results[table_name] = stats

        return results

    def vacuum_table(self, table: str, retention_hours: int = 168) -> None:
        """Remove stale data files from a Delta table.

        Args:
            table: Table path string or ``TablePath`` member.
            retention_hours: Keep files newer than this threshold.
                Defaults to 168 (7 days).
        """
        table_path = self._table_path(_to_str(table))
        if not table_path.exists():
            return

        dt = DeltaTable(str(table_path))
        dt.vacuum(retention_hours=retention_hours, enforce_retention_duration=False)

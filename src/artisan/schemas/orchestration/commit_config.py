"""Configuration for staged Delta commit behavior."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class CommitConfig(BaseModel):
    """Tuning knobs for committing staged execution-run directories."""

    enabled: bool = Field(
        default=True,
        description="Use chunked staged commits instead of one execution dir at a time.",
    )
    initial_chunk_size: int = Field(
        default=250,
        ge=1,
        description="Preferred number of staged execution dirs per commit chunk.",
    )
    min_chunk_size: int = Field(
        default=1,
        ge=1,
        description="Smallest chunk size to try when a probe exceeds limits.",
    )
    max_commit_chunk_rows: int | None = Field(
        default=None,
        gt=0,
        description="Maximum total staged rows across all tables in one chunk.",
    )
    max_commit_chunk_bytes: int | None = Field(
        default=None,
        gt=0,
        description="Maximum staged Parquet bytes across all tables in one chunk.",
    )
    max_commit_memory_fraction: float | None = Field(
        default=0.25,
        gt=0.0,
        le=1.0,
        description="Maximum fraction of currently available driver RAM to use.",
    )
    parquet_memory_multiplier: float = Field(
        default=4.0,
        gt=0.0,
        description="Multiplier from staged Parquet bytes to in-memory estimate.",
    )

    @model_validator(mode="after")
    def _validate_chunk_bounds(self) -> CommitConfig:
        if self.min_chunk_size > self.initial_chunk_size:
            msg = "min_chunk_size must be <= initial_chunk_size"
            raise ValueError(msg)
        return self

    model_config = {"frozen": True}

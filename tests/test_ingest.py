"""Tests for ingestion helpers."""
from __future__ import annotations

from pipeline import ingest
from pipeline.config import settings


def test_chunk_size_respects_limit():
    num_columns = 7000
    chunk_size = ingest._determine_chunk_size(num_columns)
    max_allowed = max(1, (ingest.POSTGRES_MAX_BIND_PARAMS // num_columns) - 1)
    assert chunk_size <= max_allowed


def test_chunk_size_defaults_to_setting():
    chunk_size = ingest._determine_chunk_size(10)
    assert chunk_size == settings.ingest_chunksize

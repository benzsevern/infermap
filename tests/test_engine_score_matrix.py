"""Tests for MapEngine.return_score_matrix flag."""
from __future__ import annotations

from infermap.engine import MapEngine


def _make_schema(*names: str):
    return [{n: "" for n in names}]


def test_score_matrix_is_none_by_default():
    engine = MapEngine()
    src = _make_schema("fname", "lname")
    tgt = _make_schema("first_name", "last_name")
    result = engine.map(src, tgt)
    assert result.score_matrix is None


def test_score_matrix_populated_when_flag_set():
    engine = MapEngine(return_score_matrix=True)
    src = _make_schema("fname", "lname")
    tgt = _make_schema("first_name", "last_name")
    result = engine.map(src, tgt)
    assert result.score_matrix is not None
    assert set(result.score_matrix.keys()) == {"fname", "lname"}
    for row in result.score_matrix.values():
        assert set(row.keys()) == {"first_name", "last_name"}
        for score in row.values():
            assert 0.0 <= score <= 1.0


def test_score_matrix_matches_assignment():
    """The (src, tgt) pair for each mapping should appear in the score matrix
    with a score equal to the mapping's confidence (modulo rounding)."""
    engine = MapEngine(return_score_matrix=True, min_confidence=0.0)
    src = _make_schema("fname", "email_addr")
    tgt = _make_schema("first_name", "email")
    result = engine.map(src, tgt)
    assert result.score_matrix is not None
    for m in result.mappings:
        sm_score = result.score_matrix[m.source][m.target]
        assert abs(sm_score - m.confidence) < 0.01

"""Tests for the runner orchestration layer."""
from __future__ import annotations

import pytest

from infermap.types import FieldInfo, SchemaInfo
from infermap_bench.cases import Case, Expected
from infermap_bench.runner import (
    FailureBudgetExceededError,
    RunOptions,
    _abort_if_over_budget,
    _schema_to_rows,
    run_cases,
)


def _case(case_id: str, src_names: list[str], tgt_names: list[str],
          mappings: list[tuple[str, str]], tags: list[str] | None = None,
          category: str = "valentine", difficulty: str = "easy") -> Case:
    src_set = {s for s, _ in mappings}
    return Case(
        id=case_id,
        category=category,
        subcategory="test",
        tags=tags or [],
        expected_difficulty=difficulty,
        source_schema=SchemaInfo(fields=[FieldInfo(name=n) for n in src_names]),
        target_schema=SchemaInfo(fields=[FieldInfo(name=n) for n in tgt_names]),
        expected=Expected(
            mappings=[{"source": s, "target": t} for s, t in mappings],
            unmapped_source=[n for n in src_names if n not in src_set],
            unmapped_target=[n for n in tgt_names if n not in {t for _, t in mappings}],
        ),
    )


class TestSchemaToRows:
    """Regression guard: _schema_to_rows must preserve sample_values so that
    PatternTypeScorer and ProfileScorer have real data to work with."""

    def test_preserves_sample_values(self):
        schema = SchemaInfo(fields=[
            FieldInfo(name="email", sample_values=["a@b.co", "c@d.co", "e@f.co"]),
            FieldInfo(name="age", sample_values=["25", "30", "35"]),
        ])
        rows = _schema_to_rows(schema)
        assert len(rows) == 3
        assert rows[0] == {"email": "a@b.co", "age": "25"}
        assert rows[1] == {"email": "c@d.co", "age": "30"}
        assert rows[2] == {"email": "e@f.co", "age": "35"}

    def test_uneven_sample_lengths_padded_with_none(self):
        schema = SchemaInfo(fields=[
            FieldInfo(name="a", sample_values=["x", "y"]),
            FieldInfo(name="b", sample_values=["p", "q", "r"]),
        ])
        rows = _schema_to_rows(schema)
        assert len(rows) == 3
        assert rows[2] == {"a": None, "b": "r"}

    def test_empty_samples_still_produces_name_row(self):
        """Fall-back path: no samples anywhere, but column names must survive."""
        schema = SchemaInfo(fields=[FieldInfo(name="x"), FieldInfo(name="y")])
        rows = _schema_to_rows(schema)
        assert rows == [{"x": None, "y": None}]

    def test_empty_schema_returns_empty(self):
        schema = SchemaInfo(fields=[])
        assert _schema_to_rows(schema) == []

    def test_round_trip_preserves_pattern_signal(self):
        """End-to-end check: after passing through _schema_to_rows and
        InMemoryProvider.extract, the PatternTypeScorer should still detect
        emails. This is the bug-fix regression guard — if anyone reverts
        _schema_to_rows to strip samples, this test fires."""
        from infermap.providers.memory import InMemoryProvider
        from infermap.scorers.pattern_type import classify_field

        schema = SchemaInfo(fields=[FieldInfo(
            name="contact",
            sample_values=["alice@example.com", "bob@example.org", "carol@test.io"],
        )])
        rows = _schema_to_rows(schema)
        round_tripped = InMemoryProvider().extract(rows)
        email_field = round_tripped.fields[0]
        assert classify_field(email_field) == "email"


class TestAbortBudget:
    def test_under_budget_does_not_raise(self):
        _abort_if_over_budget(failed_count=5, total_count=100, budget=0.10)
        _abort_if_over_budget(failed_count=10, total_count=100, budget=0.10)  # exactly 10% — allowed

    def test_over_budget_raises(self):
        with pytest.raises(FailureBudgetExceededError) as exc:
            _abort_if_over_budget(failed_count=15, total_count=100, budget=0.10)
        msg = str(exc.value)
        assert "15" in msg and "100" in msg

    def test_empty_total_does_not_raise(self):
        """Zero cases is a no-op, not a failure."""
        _abort_if_over_budget(failed_count=0, total_count=0, budget=0.10)


class TestRunCases:
    def test_runs_all_cases(self):
        cases = [
            _case("a", ["first_name"], ["first_name"], [("first_name", "first_name")]),
            _case("b", ["email"], ["email"], [("email", "email")]),
        ]
        results = run_cases(cases, RunOptions())
        assert len(results) == 2
        assert all(not r.failed for r in results)
        for r in results:
            assert r.f1 == 1.0 or r.tp >= 1

    def test_carries_case_metadata(self):
        cases = [
            _case("x/y/z", ["a"], ["a"], [("a", "a")],
                  tags=["alias_dominant", "small"], category="synthetic", difficulty="hard"),
        ]
        results = run_cases(cases, RunOptions())
        r = results[0]
        assert r.case_id == "x/y/z"
        assert r.category == "synthetic"
        assert r.difficulty == "hard"
        assert r.tags == ["alias_dominant", "small"]

    def test_empty_case_list_returns_empty(self):
        results = run_cases([], RunOptions())
        assert results == []

    def test_populates_predictions_for_ece(self):
        cases = [_case("a", ["first_name", "email"], ["first_name", "email"],
                        [("first_name", "first_name"), ("email", "email")])]
        results = run_cases(cases, RunOptions())
        assert len(results[0].predictions) > 0
        for p in results[0].predictions:
            assert hasattr(p, "confidence")
            assert hasattr(p, "correct")


class TestFailureHandling:
    def test_exception_from_scorer_recorded_as_failed(self):
        import infermap_bench.runner as runner_mod

        original = runner_mod._make_engine

        def broken_engine(opts):
            engine = original(opts)
            def bad_map(src, tgt):
                raise RuntimeError("intentional test failure")
            engine.map = bad_map  # type: ignore[method-assign]
            return engine

        runner_mod._make_engine = broken_engine  # type: ignore[assignment]
        try:
            cases = [_case("a", ["x"], ["x"], [("x", "x")])]
            results = run_cases(cases, RunOptions(failure_budget=1.0))
            assert len(results) == 1
            assert results[0].failed is True
            assert "RuntimeError" in (results[0].failure_reason or "")
            assert results[0].f1 == 0.0
            assert results[0].tp == 0
            assert results[0].fn == 1
        finally:
            runner_mod._make_engine = original

    def test_failure_budget_exceeded_aborts(self):
        import infermap_bench.runner as runner_mod
        original = runner_mod._make_engine

        def broken_engine(opts):
            engine = original(opts)
            def bad_map(src, tgt):
                raise RuntimeError("boom")
            engine.map = bad_map  # type: ignore[method-assign]
            return engine

        runner_mod._make_engine = broken_engine  # type: ignore[assignment]
        try:
            cases = [_case(f"c{i}", ["a"], ["a"], [("a", "a")]) for i in range(11)]
            with pytest.raises(FailureBudgetExceededError):
                run_cases(cases, RunOptions(failure_budget=0.10))
        finally:
            runner_mod._make_engine = original

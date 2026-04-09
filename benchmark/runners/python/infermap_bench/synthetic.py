"""Synthetic case generator — canonical (Python-only) per spec §7.4.

This module contains the pure transform primitives used by the generator
(Task 4.3). Every transform preserves `Field.canonical_name` so the generator
can recover ground truth regardless of name/sample mutations.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Field:
    name: str
    dtype: str
    samples: list[str]
    canonical_name: str | None = None


@dataclass(frozen=True)
class Schema:
    name: str
    fields: list[Field]


# ---------- helpers ----------

_SNAKE_RE = re.compile(r"_([a-z])")
_CAMEL_RE = re.compile(r"([a-z])([A-Z])")


def _to_camel(snake: str) -> str:
    return _SNAKE_RE.sub(lambda m: m.group(1).upper(), snake)


def _to_snake(camel: str) -> str:
    return _CAMEL_RE.sub(lambda m: f"{m.group(1)}_{m.group(2)}", camel).lower()


# ---------- transforms ----------


def case_change(schema: Schema, rng: random.Random) -> Schema:
    """Randomly convert each field name to camelCase, PascalCase, or UPPERCASE."""
    choices = ("camel", "pascal", "upper")
    new_fields = []
    for f in schema.fields:
        mode = rng.choice(choices)
        camel = _to_camel(f.name)
        if mode == "camel":
            new_name = camel
        elif mode == "pascal":
            new_name = camel[:1].upper() + camel[1:] if camel else ""
        else:
            new_name = f.name.upper()
        new_fields.append(replace(f, name=new_name))
    return replace(schema, fields=new_fields)


def snake_camel_swap(schema: Schema, rng: random.Random) -> Schema:
    """Flip snake_case fields to camelCase and vice versa."""
    new_fields = []
    for f in schema.fields:
        if "_" in f.name:
            new_fields.append(replace(f, name=_to_camel(f.name)))
        else:
            new_fields.append(replace(f, name=_to_snake(f.name)))
    return replace(schema, fields=new_fields)


def abbreviate(
    schema: Schema,
    rng: random.Random,
    table: dict[str, str],
) -> Schema:
    """Replace field names that appear in the abbreviation table."""
    new_fields = [
        replace(f, name=table.get(f.name, f.name)) for f in schema.fields
    ]
    return replace(schema, fields=new_fields)


def prefix_add(
    schema: Schema,
    rng: random.Random,
    choices: list[str],
) -> Schema:
    """Prepend a randomly chosen prefix to every field."""
    prefix = rng.choice(choices)
    new_fields = [replace(f, name=f"{prefix}_{f.name}") for f in schema.fields]
    return replace(schema, fields=new_fields)


def suffix_add(
    schema: Schema,
    rng: random.Random,
    choices: list[str],
) -> Schema:
    """Append a randomly chosen suffix to every field."""
    suffix = rng.choice(choices)
    new_fields = [replace(f, name=f"{f.name}_{suffix}") for f in schema.fields]
    return replace(schema, fields=new_fields)


def column_reorder(schema: Schema, rng: random.Random) -> Schema:
    """Shuffle the field order."""
    new_fields = list(schema.fields)
    rng.shuffle(new_fields)
    return replace(schema, fields=new_fields)


def drop_columns(
    schema: Schema,
    rng: random.Random,
    count: int = 1,
) -> tuple[Schema, list[str]]:
    """Remove up to `count` random fields.

    Returns (new_schema, dropped_canonical_names).
    """
    if len(schema.fields) <= count:
        return schema, []
    indices = sorted(rng.sample(range(len(schema.fields)), count))
    dropped_canonical = [
        schema.fields[i].canonical_name
        for i in indices
        if schema.fields[i].canonical_name is not None
    ]
    kept = [f for i, f in enumerate(schema.fields) if i not in indices]
    return replace(schema, fields=kept), dropped_canonical


def add_distractors(
    schema: Schema,
    rng: random.Random,
    pool: list[Field],
    count: int = 1,
) -> tuple[Schema, list[str]]:
    """Insert up to `count` distractor fields from the pool."""
    if not pool or count <= 0:
        return schema, []
    take = min(count, len(pool))
    added = rng.sample(pool, take)
    new_fields = list(schema.fields) + list(added)
    rng.shuffle(new_fields)
    return replace(schema, fields=new_fields), [f.name for f in added]


def dtype_string_cast(schema: Schema, rng: random.Random) -> Schema:
    """Convert non-string dtypes to string."""
    new_fields = [
        replace(f, dtype="string") if f.dtype != "string" else f
        for f in schema.fields
    ]
    return replace(schema, fields=new_fields)


def sample_typos(
    schema: Schema,
    rng: random.Random,
    rate: float = 0.15,
) -> Schema:
    """Introduce character-level typos in sample values.

    Only affects samples with length > 1. Never mutates name/canonical_name.
    """
    new_fields = []
    for f in schema.fields:
        mutated = []
        for sample in f.samples:
            if rng.random() < rate and len(sample) > 1:
                idx = rng.randrange(len(sample))
                replacement = chr(ord("a") + rng.randrange(26))
                mutated.append(sample[:idx] + replacement + sample[idx + 1:])
            else:
                mutated.append(sample)
        new_fields.append(replace(f, samples=mutated))
    return replace(schema, fields=new_fields)

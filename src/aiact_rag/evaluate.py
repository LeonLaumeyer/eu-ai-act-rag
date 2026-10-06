"""Question set loading, retrieval metrics and bootstrap statistics."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from aiact_rag.models import SearchResult

LEVELS = ("unit", "article")


@dataclass
class Question:
    id: str
    lang: str  # "en" | "de"
    question: str
    gold: list[str]  # unit ids; empty for unanswerable questions
    answerable: bool = True
    type: str = "factual"  # factual | definition | cross_reference | procedural | trick
    reference_answer: str = ""
    pair: str = ""  # same id for a question and its translation


def load_questions(path: Path, known_units: set[str] | None = None) -> list[Question]:
    """Read the question file and validate ids, languages and gold units.

    Raises ``ValueError`` naming the question for a duplicate id, an unknown language, an
    answerable question without gold units, an unanswerable one with gold units, or a gold
    unit that is not in ``known_units``.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or []
    questions = []
    seen = set()
    for item in raw:
        question = Question(**{"gold": [], **item})
        if question.id in seen:
            raise ValueError(f"duplicate question id: {question.id}")
        seen.add(question.id)
        if question.lang not in ("en", "de"):
            raise ValueError(f"{question.id}: unknown lang {question.lang!r}")
        if question.answerable and not question.gold:
            raise ValueError(f"{question.id}: answerable but no gold units")
        if not question.answerable and question.gold:
            raise ValueError(f"{question.id}: unanswerable but has gold units")
        if known_units is not None:
            unknown = [unit for unit in question.gold if unit not in known_units]
            if unknown:
                raise ValueError(f"{question.id}: unknown gold units {unknown}")
        questions.append(question)
    return questions


def to_article(unit_id: str) -> str:
    """The article / annex / recital a unit belongs to."""
    return unit_id.split("-")[0]


def units_of(result: SearchResult, level: str = "unit") -> set[str]:
    """The units one result covers, as a set."""

    if level == "unit":
        return {id for id in result.chunk.unit_ids}
    if level == "article":
        return {to_article(a) for a in result.chunk.unit_ids}
    raise ValueError(f"Level {level} is not supported")


def hit_at_k(results: list[SearchResult], gold: list[str], k: int, level: str = "unit") -> bool:
    """True if at least one of the first ``k`` results covers at least one gold unit."""
    return reciprocal_rank(results, gold, k, level) > 0


def reciprocal_rank(
    results: list[SearchResult], gold: list[str], k: int = 10, level: str = "unit"
) -> float:
    """``1 / rank`` of the first of the top ``k`` results that covers a gold unit, else ``0.0``."""
    wanted = _gold_set(gold, level)
    for rank, result in enumerate(results[:k], start=1):
        if units_of(result, level) & wanted:
            return 1 / rank
    return 0.0


def recall_at_k(results: list[SearchResult], gold: list[str], k: int, level: str = "unit") -> float:
    """The fraction of gold units covered by the first ``k`` results together."""
    wanted = _gold_set(gold, level)
    found = set()
    for result in results[:k]:
        found |= units_of(result, level) & wanted
    return len(found) / len(wanted)


def _gold_set(gold: list[str], level: str) -> set[str]:
    if not gold:
        raise ValueError("gold is empty")
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    return {to_article(unit) for unit in gold} if level == "article" else set(gold)


def bootstrap_ci(
    values: list[float], n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the MEAN of ``values``."""
    if len(values) == 0:
        raise ValueError("values is empty")
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(data), size=(n_resamples, len(data)))
    means = data[samples].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def paired_bootstrap_p(
    baseline: list[float],
    improved: list[float],
    n_resamples: int = 10000,
    seed: int = 0,
) -> float:
    """One-sided paired bootstrap p-value: the share of resampled mean differences <= 0."""
    if len(baseline) == 0 or len(baseline) != len(improved):
        raise ValueError("need two non-empty lists of the same length")
    diffs = np.asarray(improved, dtype=float) - np.asarray(baseline, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(diffs), size=(n_resamples, len(diffs)))
    means = diffs[samples].mean(axis=1)
    return float((means <= 0).mean())

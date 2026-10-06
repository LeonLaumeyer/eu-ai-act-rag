"""Retrieval evaluation across chunking and retrieval settings.

    python -m aiact_rag.retrieval_eval [--no-rerank]

Writes retrieval.md, retrieval_rows.jsonl and failures.md to results/.
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from aiact_rag.config import RetrievalConfig
from aiact_rag.evaluate import (
    Question,
    bootstrap_ci,
    hit_at_k,
    load_questions,
    paired_bootstrap_p,
    recall_at_k,
    reciprocal_rank,
)
from aiact_rag.models import PROCESSED, ROOT
from aiact_rag.pipeline import Resources, build_retriever

EVAL_TOP_K = 10


@dataclass(frozen=True)
class Setting:
    chunking: str
    mode: str
    rerank: bool

    @property
    def name(self) -> str:
        return f"{self.chunking} + {self.mode}" + (" + rerank" if self.rerank else "")


BASELINE = Setting("fixed", "dense", False)
SETTINGS = [
    BASELINE,
    Setting("article", "dense", False),
    Setting("article", "bm25", False),
    Setting("article", "hybrid", False),
    Setting("article", "dense", True),
    Setting("article", "bm25", True),
    Setting("article", "hybrid", True),
]


@dataclass
class Row:
    """The result of one question under one setting."""

    setting: str
    question_id: str
    lang: str
    type: str
    hit1: float
    hit5: float
    hit5_article: float
    recall5: float
    rr10: float
    seconds: float
    gold: list[str]
    top5: list[str]  # citations of the first five results
    top5_units: list[list[str]]


def load_known_units(processed_dir: Path = PROCESSED) -> set[str]:
    """All unit ids of the parsed Act (the same in both languages)."""
    with open(processed_dir / "units_en.jsonl", encoding="utf-8") as f:
        return {json.loads(line)["unit_id"] for line in f}


def evaluate_setting(name: str, retriever, questions: list[Question]) -> list[Row]:
    """Run all ANSWERABLE questions through ``retriever``; one ``Row`` each."""
    rows = []
    for q in questions:
        if not q.answerable:
            continue
        start = time.perf_counter()
        results = retriever.retrieve(q.question, q.lang)
        seconds = time.perf_counter() - start
        rows.append(
            Row(
                setting=name,
                question_id=q.id,
                lang=q.lang,
                type=q.type,
                hit1=float(hit_at_k(results, q.gold, 1)),
                hit5=float(hit_at_k(results, q.gold, 5)),
                hit5_article=float(hit_at_k(results, q.gold, 5, level="article")),
                recall5=recall_at_k(results, q.gold, 5),
                rr10=reciprocal_rank(results, q.gold, 10),
                seconds=seconds,
                gold=q.gold,
                top5=[r.chunk.citation for r in results[:5]],
                top5_units=[r.chunk.unit_ids for r in results[:5]],
            )
        )
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def summarize(rows: list[Row]) -> dict:
    """The numbers of one table line. ``rows`` must not be empty."""
    hits = [r.hit5 for r in rows]
    low, high = bootstrap_ci(hits)
    return {
        "n": len(rows),
        "hit1": mean([r.hit1 for r in rows]),
        "hit5": mean(hits),
        "hit5_low": low,
        "hit5_high": high,
        "hit5_article": mean([r.hit5_article for r in rows]),
        "recall5": mean([r.recall5 for r in rows]),
        "mrr10": mean([r.rr10 for r in rows]),
        "ms": 1000 * mean([r.seconds for r in rows]),
    }


def pct(value: float) -> str:
    return f"{100 * value:.0f}%"


def results_table(rows_by_setting: dict[str, list[Row]]) -> str:
    """The main table: one line per setting."""
    lines = [
        "| Setting | Hit@1 | Hit@5 (95% CI) | Hit@5 article level | Recall@5 | MRR@10 | ms / question |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, rows in rows_by_setting.items():
        s = summarize(rows)
        lines.append(
            f"| {name} | {pct(s['hit1'])} | {pct(s['hit5'])} ({pct(s['hit5_low'])}–{pct(s['hit5_high'])}) "
            f"| {pct(s['hit5_article'])} | {pct(s['recall5'])} | {s['mrr10']:.2f} | {s['ms']:.0f} |"
        )
    return "\n".join(lines)


def breakdown_table(rows_by_setting: dict[str, list[Row]], field: str) -> str:
    """Hit@5 per setting, split by ``field`` (``"lang"`` or ``"type"``)."""
    groups = sorted({getattr(r, field) for rows in rows_by_setting.values() for r in rows})
    sizes = {g: 0 for g in groups}
    for row in next(iter(rows_by_setting.values()), []):
        sizes[getattr(row, field)] += 1
    lines = [
        "| Setting | " + " | ".join(f"{g} (n={sizes[g]})" for g in groups) + " |",
        "|---|" + "---|" * len(groups),
    ]
    for name, rows in rows_by_setting.items():
        cells = []
        for g in groups:
            group = [r.hit5 for r in rows if getattr(r, field) == g]
            cells.append(pct(mean(group)) if group else "–")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def best_setting(rows_by_setting: dict[str, list[Row]]) -> str:
    """The setting with the highest Hit@5; MRR@10 decides a tie."""

    def score(name: str) -> tuple[float, float]:
        s = summarize(rows_by_setting[name])
        return s["hit5"], s["mrr10"]

    return max(rows_by_setting, key=score)


def failures_markdown(name: str, rows: list[Row], questions: list[Question]) -> str:
    """Every question that ``name`` missed at Hit@5, with gold and what came back."""
    text = {q.id: q.question for q in questions}
    misses = [r for r in rows if not r.hit5]
    lines = [
        f"# Misses of `{name}` (Hit@5, unit level)",
        "",
        f"{len(misses)} of {len(rows)} questions.",
        "",
    ]
    for r in misses:
        lines += [
            f"## {r.question_id} ({r.lang}, {r.type})",
            f"**Question:** {text[r.question_id]}  ",
            f"**Gold:** {', '.join(r.gold)}  ",
            f"**Right article in top 5:** {'yes' if r.hit5_article else 'no'}  ",
            "**Top 5:** " + ("; ".join(r.top5) or "nothing found"),
            "",
        ]
    return "\n".join(lines)


def report(rows_by_setting: dict[str, list[Row]]) -> str:
    """The content of ``results/retrieval.md``."""
    best = best_setting(rows_by_setting)
    parts = [
        "# Retrieval results",
        "",
        results_table(rows_by_setting),
        "",
        f"Best setting: **{best}**.",
    ]
    if BASELINE.name in rows_by_setting and best != BASELINE.name:
        p = paired_bootstrap_p(
            [r.hit5 for r in rows_by_setting[BASELINE.name]],
            [r.hit5 for r in rows_by_setting[best]],
        )
        parts.append(f"Paired bootstrap, Hit@5, `{BASELINE.name}` vs `{best}`: p = {p:.3f}.")
    parts += [
        "",
        "## Hit@5 by language",
        "",
        breakdown_table(rows_by_setting, "lang"),
        "",
        "## Hit@5 by question type",
        "",
        breakdown_table(rows_by_setting, "type"),
        "",
    ]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Retrieval evaluation over several settings.")
    parser.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--no-rerank", action="store_true", help="skip the settings with the reranker"
    )
    args = parser.parse_args(argv)

    questions = load_questions(args.questions, known_units=load_known_units())
    resources = Resources()
    rows_by_setting: dict[str, list[Row]] = {}
    for setting in SETTINGS:
        if setting.rerank and args.no_rerank:
            continue
        config = RetrievalConfig(
            chunking=setting.chunking, mode=setting.mode, rerank=setting.rerank
        )
        retriever = build_retriever(config, resources, top_k=EVAL_TOP_K)
        rows_by_setting[setting.name] = evaluate_setting(setting.name, retriever, questions)
        print(
            f"{setting.name}: Hit@5 {pct(summarize(rows_by_setting[setting.name])['hit5'])}",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "retrieval_rows.jsonl", "w", encoding="utf-8") as f:
        for rows in rows_by_setting.values():
            f.writelines(json.dumps(asdict(row), ensure_ascii=False) + "\n" for row in rows)
    text = report(rows_by_setting)
    (args.out / "retrieval.md").write_text(text, encoding="utf-8")
    best = best_setting(rows_by_setting)
    (args.out / "failures.md").write_text(
        failures_markdown(best, rows_by_setting[best], questions), encoding="utf-8"
    )
    print()
    print(text)


if __name__ == "__main__":
    main()

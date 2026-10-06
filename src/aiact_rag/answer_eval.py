"""End-to-end answer evaluation: citations, refusals and faithfulness.

    python -m aiact_rag.answer_eval [--provider P] [--judge P] [--mode M] [--no-rerank] [--limit N]
                                    [--resume]

Writes answers_<provider>.jsonl and answers_<provider>.md to results/. Every answer is written
to the .jsonl as soon as it is there, so a run that stops halfway can be continued with --resume.
"""

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from aiact_rag.config import PROVIDERS, load_config, load_llm_config
from aiact_rag.evaluate import Question, load_questions
from aiact_rag.generate import Answer
from aiact_rag.judge import faithfulness
from aiact_rag.models import ROOT, GenerateFn
from aiact_rag.pipeline import Resources, build_answer_fn, build_llm, build_retriever
from aiact_rag.retrieval_eval import load_known_units, mean, pct
from aiact_rag.retrieve.hybrid import MODES


@dataclass
class AnswerRow:
    """One question with its answer and everything measured on it."""

    question_id: str
    lang: str
    type: str
    answerable: bool
    refused: bool
    valid_citations: list[str]
    invalid_citations: list[str]
    cites_gold: bool  # a valid citation points at a gold unit
    faithfulness: float | None  # None: not judged (see judge.faithfulness)
    seconds: float
    question: str
    answer: str
    sources: list[str]


def cites_gold(answer: Answer, gold: list[str]) -> bool:
    """True if a valid citation of the answer belongs to a source that covers a gold unit."""
    cited_units = {
        unit
        for source in answer.sources
        if source.chunk.citation in answer.valid_citations
        for unit in source.chunk.unit_ids
    }
    return bool(cited_units & set(gold))


def evaluate_answers(
    questions: list[Question],
    answer_fn: Callable[[str, str], Answer],
    judge: GenerateFn | None = None,
    done: dict[str, AnswerRow] | None = None,
    on_row: Callable[[AnswerRow], None] | None = None,
) -> list[AnswerRow]:
    """Answer every question (also the unanswerable ones) and measure the answer.

    A question whose id is in ``done`` is not asked again; its row is taken from there.
    ``on_row`` is called with every newly made row, right after it was made.
    """
    done = done or {}
    rows = []
    for q in questions:
        if q.id in done:
            rows.append(done[q.id])
            continue
        start = time.perf_counter()
        answer = answer_fn(q.question, q.lang)
        seconds = time.perf_counter() - start
        row = AnswerRow(
            question_id=q.id,
            lang=q.lang,
            type=q.type,
            answerable=q.answerable,
            refused=answer.refused,
            valid_citations=answer.valid_citations,
            invalid_citations=answer.invalid_citations,
            cites_gold=cites_gold(answer, q.gold),
            faithfulness=faithfulness(answer, judge) if judge is not None else None,
            seconds=seconds,
            question=q.question,
            answer=answer.text,
            sources=[s.chunk.citation for s in answer.sources],
        )
        if on_row is not None:
            on_row(row)
        rows.append(row)
    return rows


def row_line(row: AnswerRow) -> str:
    return json.dumps(asdict(row), ensure_ascii=False) + "\n"


def load_rows(path: Path) -> dict[str, AnswerRow]:
    """The rows of an earlier run by question id. A line cut off by a crash is skipped."""
    rows: dict[str, AnswerRow] = {}
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                row = AnswerRow(**json.loads(line))
            except (ValueError, TypeError):
                continue
            rows[row.question_id] = row
    return rows


def rows_to_resume(
    rows_path: Path, settings_path: Path, settings: dict, questions: list[Question]
) -> dict[str, AnswerRow]:
    """The rows of an unfinished run that ``--resume`` may keep.

    Raises ValueError if the earlier run used other settings (or did not record them): mixing
    answers of two models or retrieval modes in one table would make the numbers meaningless.
    The rows file is rewritten with only the kept rows, so new rows can be appended to it.
    """
    rows = load_rows(rows_path)
    if not rows:
        return {}
    earlier = (
        json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else None
    )
    if earlier != settings:
        raise ValueError(
            f"{rows_path.name} was made with other settings ({earlier}) than this run ({settings}). "
            "Run without --resume to start again."
        )
    ids = {q.id for q in questions}
    kept = {question_id: row for question_id, row in rows.items() if question_id in ids}
    temporary = rows_path.with_suffix(".jsonl.tmp")
    with open(temporary, "w", encoding="utf-8") as f:
        f.writelines(row_line(row) for row in kept.values())
    temporary.replace(rows_path)
    return kept


def share(flags: list[bool]) -> float | None:
    """The fraction of True values; None for an empty list (nothing to measure)."""
    return sum(flags) / len(flags) if flags else None


def summarize_answers(rows: list[AnswerRow]) -> dict:
    """The summary numbers. A value is None when there was nothing to measure it on."""
    unanswerable = [r for r in rows if not r.answerable]
    answerable = [r for r in rows if r.answerable]
    answered = [r for r in answerable if not r.refused]
    judged = [r.faithfulness for r in rows if r.faithfulness is not None]
    return {
        "n": len(rows),
        "refused_unanswerable": share([r.refused for r in unanswerable]),
        "refused_answerable": share([r.refused for r in answerable]),
        "n_answered": len(answered),
        "has_citation": share([bool(r.valid_citations or r.invalid_citations) for r in answered]),
        "all_citations_valid": share(
            [bool(r.valid_citations) and not r.invalid_citations for r in answered]
        ),
        "cites_gold": share([r.cites_gold for r in answered]),
        "faithfulness": mean(judged) if judged else None,
        "n_judged": len(judged),
        "seconds": mean([r.seconds for r in rows]) if rows else None,
    }


def answers_table(summary: dict, label: str) -> str:
    def show(value: float | None) -> str:
        return "–" if value is None else pct(value)

    seconds = "–" if summary["seconds"] is None else f"{summary['seconds']:.1f}"
    return "\n".join(
        [
            f"| Measure | {label} |",
            "|---|---|",
            f"| Questions | {summary['n']} |",
            f"| Unanswerable questions refused | {show(summary['refused_unanswerable'])} |",
            f"| Answerable questions refused by mistake | {show(summary['refused_answerable'])} |",
            f"| Answers with at least one citation | {show(summary['has_citation'])} |",
            f"| Answers whose citations are all valid | {show(summary['all_citations_valid'])} |",
            f"| Answers that cite a gold unit | {show(summary['cites_gold'])} |",
            f"| Faithfulness, judged on {summary['n_judged']} answers | {show(summary['faithfulness'])} |",
            f"| Seconds per question | {seconds} |",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Answer evaluation with the full pipeline.")
    parser.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        help="LLM that answers (default: config.yaml)",
    )
    parser.add_argument(
        "--judge",
        choices=PROVIDERS,
        help="LLM that judges faithfulness (default: no judge)",
    )
    parser.add_argument("--limit", type=int, help="only the first N questions (for a quick try)")
    parser.add_argument("--mode", choices=MODES, help="retrieval mode (default: config.yaml)")
    parser.add_argument("--no-rerank", action="store_true", help="retrieve without the reranker")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue a run that stopped: keep the answers already in the .jsonl",
    )
    args = parser.parse_args(argv)

    config = load_config()
    generator = load_llm_config(args.provider)
    judge_config = load_llm_config(args.judge) if args.judge else None

    questions = load_questions(args.questions, known_units=load_known_units())[: args.limit]
    retrieval = replace(
        config.retrieval,
        mode=args.mode or config.retrieval.mode,
        rerank=config.retrieval.rerank and not args.no_rerank,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    name = f"answers_{generator.provider}"
    rows_path = args.out / f"{name}.jsonl"
    settings_path = args.out / f"{name}.run.json"
    settings = {
        "model": generator.model,
        "judge": judge_config.model if judge_config else None,
        **asdict(retrieval),
    }
    done: dict[str, AnswerRow] = {}
    if args.resume:
        try:
            done = rows_to_resume(rows_path, settings_path, settings, questions)
        except ValueError as error:
            parser.error(str(error))
        print(f"Resuming: {len(done)} of {len(questions)} questions are already answered.")
    settings_path.write_text(json.dumps(settings) + "\n", encoding="utf-8")

    judge = build_llm(judge_config) if judge_config else None
    retriever = build_retriever(retrieval, Resources())
    with open(rows_path, "a" if done else "w", encoding="utf-8") as f:

        def save(row: AnswerRow) -> None:
            f.write(row_line(row))
            f.flush()

        rows = evaluate_answers(
            questions, build_answer_fn(retriever, build_llm(generator)), judge, done, save
        )
    table = answers_table(summarize_answers(rows), f"{generator.provider}: {generator.model}")
    (args.out / f"{name}.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()

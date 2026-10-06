"""LLM-as-judge faithfulness scoring."""

import json

from aiact_rag.generate import Answer
from aiact_rag.models import GenerateFn, SearchResult

JUDGE_SYSTEM = (
    "You check whether an answer is supported by source excerpts from the EU AI Act.\n"
    "1. Split the answer into its separate factual claims. Ignore the citations in square "
    "brackets and ignore sentences that only say the excerpts do not contain the answer.\n"
    "2. For each claim decide: is it supported by the excerpts? Use ONLY the excerpts, not "
    "your own knowledge. A claim that goes beyond the excerpts is not supported, even if it is true.\n"
    "Reply with JSON only, in exactly this form:\n"
    '{"claims": [{"claim": "<the claim>", "supported": true}, {"claim": "<the claim>", "supported": false}]}'
)


def build_judge_prompt(answer_text: str, sources: list[SearchResult]) -> tuple[str, str]:
    """The ``(system, user)`` messages for the judge."""
    excerpts = "\n\n".join(f"[{s.chunk.citation}]\n{s.chunk.text}" for s in sources)
    return JUDGE_SYSTEM, f"Excerpts:\n\n{excerpts}\n\nAnswer to check:\n{answer_text}"


def parse_verdicts(reply: str) -> list[bool] | None:
    """Read the judge's reply: one bool per claim, or None if it can't be used."""
    start, end = reply.find("{"), reply.rfind("}")
    if start == -1 or end < start:
        return None
    try:
        data = json.loads(reply[start : end + 1])
    except json.JSONDecodeError:
        return None
    claims = data.get("claims") if isinstance(data, dict) else None
    if not isinstance(claims, list) or not claims:
        return None
    verdicts = []
    for claim in claims:
        supported = claim.get("supported") if isinstance(claim, dict) else None
        if not isinstance(supported, bool):
            return None
        verdicts.append(supported)
    return verdicts


def faithfulness(answer: Answer, judge: GenerateFn) -> float | None:
    """Share of supported claims, from 0.0 to 1.0.

    None if there is nothing to judge (a refusal, or no sources) or the judge's reply is unusable.
    """
    if answer.refused or not answer.sources:
        return None
    verdicts = parse_verdicts(judge(*build_judge_prompt(answer.text, answer.sources)))
    if verdicts is None:
        return None
    return sum(verdicts) / len(verdicts)

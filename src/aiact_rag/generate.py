"""Prompt construction, citation checking and answer generation."""

import re
from dataclasses import dataclass

from aiact_rag.models import GenerateFn, SearchResult

REFUSAL = {
    "en": "The provided excerpts of the AI Act do not answer this question.",
    "de": "Die bereitgestellten Auszüge der KI-Verordnung beantworten diese Frage nicht.",
}


@dataclass
class Answer:
    text: str
    sources: list[SearchResult]  # what the LLM was shown
    valid_citations: list[str]  # cited AND among the sources
    invalid_citations: list[str]  # cited but NOT among the sources
    refused: bool  # the answer is the refusal sentence


def normalize(text: str) -> str:
    """Collapse whitespace and ignore case, for comparing text loosely."""
    return " ".join(text.casefold().split())


def build_prompt(question: str, results: list[SearchResult], lang: str) -> tuple[str, str]:
    """Build the ``(system, user)`` messages for the LLM."""
    system_prompt = f"""You are a question-answering assistant. You will be given a question and a set of excerpts, each marked with a label.

        Rules:
        1. Answer only from the excerpts. Do not use outside knowledge, and do not guess or fill in gaps.
        2. After each claim, cite the label of the excerpt that supports it in square brackets, e.g. [label]. If a claim draws on several excerpts, cite each one, e.g. [label1][label2].
        3. If the excerpts do not contain the answer, reply with exactly this sentence and nothing else:
        {REFUSAL[lang]}
        4. Answer in the same language as the question."""
    excerpts = "".join(f"[{r.chunk.citation}]\n{r.chunk.text}\n" for r in results)
    user_prompt = f"Excerpts:\n{excerpts}Question: {question}"
    return system_prompt, user_prompt


def extract_citations(answer: str) -> list[str]:
    """Everything the answer puts in square brackets, as written."""
    citations = [c.strip() for c in re.findall(r"\[([^\[\]]*)\]", answer)]
    return list(dict.fromkeys(c for c in citations if c))


def check_citations(answer: str, results: list[SearchResult]) -> tuple[list[str], list[str]]:
    """Split the citations of ``answer`` into ``(valid, invalid)``.

    A citation is valid if it names a retrieved source or a point inside one
    (``Art. 5(1)(d)`` for the source ``Art. 5(1)``).
    """
    known = {normalize(r.chunk.citation): r.chunk.citation for r in results}
    valid, invalid = [], []
    for citation in extract_citations(answer):
        cited = normalize(citation)
        source = known.get(cited)
        if source is None:
            for key in sorted(known, key=len, reverse=True):
                if cited.startswith(key) and cited[len(key)] in "(, ":
                    source = known[key]
                    break
        if source is None:
            invalid.append(citation)
        elif source not in valid:
            valid.append(source)
    return valid, invalid


def is_refusal(answer: str, lang: str) -> bool:
    """True if ``answer`` contains ``REFUSAL[lang]``, compared with ``normalize``."""
    return normalize(REFUSAL[lang]) in normalize(answer)


def answer_question(question: str, lang: str, retriever, generate: GenerateFn) -> Answer:
    """Retrieve, generate and check the citations; refuse directly if nothing is retrieved."""
    results = retriever.retrieve(question, lang)
    if not results:
        return Answer(REFUSAL[lang], [], [], [], refused=True)
    system_prompt, user_prompt = build_prompt(question, results, lang)
    text = generate(system_prompt, user_prompt)
    valid, invalid = check_citations(text, results)
    return Answer(text, results, valid, invalid, refused=is_refusal(text, lang))

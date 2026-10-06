"""FastAPI application."""

import re
from collections.abc import Callable
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from aiact_rag.generate import Answer
from aiact_rag.llm.base import LLMError

GERMAN_WORDS = frozenset(
    [
        "der",
        "die",
        "das",
        "und",
        "ist",
        "nicht",
        "ein",
        "eine",
        "für",
        "mit",
        "von",
        "den",
        "dem",
        "des",
        "im",
        "auf",
        "sich",
        "auch",
        "wie",
        "was",
        "wer",
        "wann",
        "welche",
        "welcher",
        "welches",
        "wird",
        "werden",
        "sind",
        "darf",
        "dürfen",
        "muss",
        "müssen",
        "gilt",
        "gelten",
        "gibt",
        "laut",
        "nach",
        "bei",
        "oder",
        "zu",
        "zur",
        "zum",
        "einer",
        "eines",
        "als",
    ]
)
ENGLISH_WORDS = frozenset(
    [
        "the",
        "and",
        "is",
        "not",
        "a",
        "an",
        "for",
        "with",
        "of",
        "to",
        "in",
        "on",
        "it",
        "its",
        "also",
        "how",
        "what",
        "who",
        "when",
        "which",
        "are",
        "does",
        "do",
        "must",
        "may",
        "can",
        "shall",
        "under",
        "by",
        "or",
        "at",
        "as",
        "be",
        "that",
        "this",
    ]
)

GERMAN_LETTERS = "äöüß"


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    lang: Literal["en", "de"] | None = None


class Source(BaseModel):
    citation: str
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    lang: str
    refused: bool
    citations: list[str]
    invalid_citations: list[str]
    sources: list[Source]


def detect_lang(question: str) -> str:
    """Guess ``"en"`` or ``"de"`` from umlauts and common function words."""
    text = question.lower()
    if any(letter in text for letter in GERMAN_LETTERS):
        return "de"
    words = re.findall(r"\w+", text)
    german = sum(word in GERMAN_WORDS for word in words)
    english = sum(word in ENGLISH_WORDS for word in words)
    return "de" if german > english else "en"


def create_app(answer_fn: Callable[[str, str], Answer]) -> FastAPI:
    """Build the FastAPI app around ``answer_fn(question, lang) -> Answer``."""
    app = FastAPI(title="EU AI Act RAG")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        lang = request.lang or detect_lang(request.question)
        try:
            answer = answer_fn(request.question, lang)
        except LLMError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return AskResponse(
            answer=answer.text,
            lang=lang,
            refused=answer.refused,
            citations=answer.valid_citations,
            invalid_citations=answer.invalid_citations,
            sources=[
                Source(citation=s.chunk.citation, text=s.chunk.text, score=s.score)
                for s in answer.sources
            ],
        )

    return app

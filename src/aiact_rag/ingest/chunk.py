"""Turns parsed units into chunks: one per unit (article) or fixed-size token windows (fixed)."""

import json
import re

from transformers import AutoTokenizer

from aiact_rag.models import PROCESSED

MAX_TOKENS = 400
WINDOW = 512
OVERLAP = 64

tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")


def n_tokens(text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def header(unit: dict, lang: str) -> str:
    """The search header of a chunk: ``[EU AI Act · Art. 6(2) · <article title>]``."""
    parts = ["EU AI Act" if lang == "en" else "KI-Verordnung", unit["citation"]]
    if unit["title"]:
        parts.append(unit["title"])
    return "[" + " · ".join(parts) + "]"


def split_long(text: str) -> list[str]:
    """Split a long unit at sentence and point boundaries; every piece repeats the lead-in."""
    colon = text.find(":")
    lead_in = text[: colon + 1] if 0 < colon < 400 else ""
    body = text[len(lead_in) :]
    sentences = re.split(r"(?<=[.;])\s+(?=\(?[A-Za-zÄÖÜ0-9])", body)

    pieces, current = [], []
    for sentence in sentences:
        if current and n_tokens(" ".join([lead_in, *current, sentence])) > MAX_TOKENS:
            pieces.append(" ".join([lead_in, *current]).strip())
            current = []
        current.append(sentence.strip())
    pieces.append(" ".join([lead_in, *current]).strip())
    return pieces


def article_chunks(units: list[dict], lang: str) -> list[dict]:
    """One chunk per unit; units longer than ``MAX_TOKENS`` are split into numbered pieces."""
    chunks = []
    for unit in units:
        text = unit["text"]
        pieces = split_long(text) if n_tokens(text) > MAX_TOKENS else [text]
        for i, piece in enumerate(pieces):
            suffix = f"#{i}" if len(pieces) > 1 else ""
            chunks.append(
                {
                    "chunk_id": f"{lang}-article-{unit['unit_id']}{suffix}",
                    "lang": lang,
                    "strategy": "article",
                    "kind": unit["kind"],
                    "unit_ids": [unit["unit_id"]],
                    "citation": unit["citation"],
                    "text": piece,
                    "embed_text": f"{header(unit, lang)}\n{piece}",
                }
            )
    return chunks


def fixed_chunks(units: list[dict], lang: str) -> list[dict]:
    """Windows of ``WINDOW`` tokens with ``OVERLAP`` over the concatenated text.

    Each chunk records every unit it overlaps, so it can be scored against unit-level gold labels.
    """
    full, spans = "", []
    for unit in units:
        start = len(full)
        full += unit["text"] + "\n\n"
        spans.append((start, len(full), unit))

    offsets = tok(full, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]

    chunks = []
    for i, t in enumerate(range(0, len(offsets), WINDOW - OVERLAP)):
        window = offsets[t : t + WINDOW]
        c_start, c_end = window[0][0], window[-1][-1]
        covered = [unit for start, end, unit in spans if start < c_end and end > c_start]
        text = full[c_start:c_end]
        chunks.append(
            {
                "chunk_id": f"{lang}-fixed-{i}",
                "lang": lang,
                "strategy": "fixed",
                "kind": covered[0]["kind"],
                "unit_ids": [unit["unit_id"] for unit in covered],
                "citation": ", ".join(unit["citation"] for unit in covered),
                "text": text,
                "embed_text": text,
            }
        )
        if t + WINDOW >= len(offsets):
            break
    return chunks


if __name__ == "__main__":
    for lang in ["en", "de"]:
        with open(PROCESSED / f"units_{lang}.jsonl", encoding="utf-8") as f:
            units = [json.loads(line) for line in f]
        for name, make_chunks in [("article", article_chunks), ("fixed", fixed_chunks)]:
            chunks = make_chunks(units, lang)
            with open(PROCESSED / f"chunks_{name}_{lang}.jsonl", "w", encoding="utf-8") as f:
                f.writelines(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks)
            print(lang, name, len(chunks), "chunks")

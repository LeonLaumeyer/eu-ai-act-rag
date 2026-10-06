"""Tests for the chunker.

Importing chunk.py loads the bge-m3 tokenizer, so this file takes a few seconds.
"""

import pytest

from aiact_rag.ingest.chunk import (
    MAX_TOKENS,
    WINDOW,
    article_chunks,
    fixed_chunks,
    header,
    n_tokens,
    split_long,
)

LEAD_IN = "The following AI practices shall be prohibited:"
POINTS = [
    f"({letter}) the placing on the market of systems for purpose number {i} that exploit "
    f"vulnerabilities of natural persons due to their age or disability in scenario {i};"
    for i, letter in enumerate("abcdefghijklmnopqrstuvwxyz")
]
LONG_TEXT = f"{LEAD_IN} {' '.join(POINTS)}"


def unit(unit_id, text, kind="article", citation=None, title="Some title"):
    return {
        "unit_id": unit_id,
        "kind": kind,
        "citation": citation or unit_id.upper(),
        "title": title,
        "text": text,
    }


class TestSplitLong:
    def test_the_test_text_is_really_long(self):
        assert n_tokens(LONG_TEXT) > 2 * MAX_TOKENS

    def test_several_pieces(self):
        assert len(split_long(LONG_TEXT)) >= 3

    def test_every_piece_starts_with_the_lead_in(self):
        for piece in split_long(LONG_TEXT):
            assert piece.startswith(LEAD_IN + " (")

    def test_no_piece_is_too_long(self):
        for piece in split_long(LONG_TEXT):
            assert n_tokens(piece) <= MAX_TOKENS

    def test_every_point_is_in_exactly_one_piece_and_whole(self):
        pieces = split_long(LONG_TEXT)
        for point in POINTS:
            assert sum(point in piece for piece in pieces) == 1

    def test_points_stay_in_order(self):
        joined = " ".join(split_long(LONG_TEXT))
        positions = [joined.index(point) for point in POINTS]
        assert positions == sorted(positions)

    def test_text_without_colon_loses_nothing(self):
        text = " ".join(
            f"Sentence number {i} talks about providers and deployers." for i in range(120)
        )
        pieces = split_long(text)
        assert len(pieces) > 1
        assert " ".join(pieces).split() == text.split()

    def test_no_empty_pieces(self):
        assert all(piece.strip() for piece in split_long(LONG_TEXT))


class TestHeader:
    def test_contains_citation_and_title(self):
        u = unit("art6-2", "text", citation="Art. 6(2)", title="Classification rules")
        assert header(u, "en") == "[EU AI Act · Art. 6(2) · Classification rules]"

    def test_german(self):
        u = unit("art6-2", "text", citation="Art. 6 Abs. 2", title="Einstufungsvorschriften")
        assert header(u, "de") == "[KI-Verordnung · Art. 6 Abs. 2 · Einstufungsvorschriften]"

    def test_no_title(self):
        u = unit("rct27", "text", kind="recital", citation="Recital 27", title="")
        assert header(u, "en") == "[EU AI Act · Recital 27]"


class TestArticleChunks:
    UNITS = [
        unit("art6-2", "Systems in Annex III are high-risk.", citation="Art. 6(2)"),
        unit("art5-1", LONG_TEXT, citation="Art. 5(1)"),
        unit("rct27", "A recital.", kind="recital", citation="Recital 27", title=""),
    ]

    @pytest.fixture
    def chunks(self):
        return article_chunks(self.UNITS, "en")

    def test_short_unit_is_one_chunk(self, chunks):
        assert chunks[0] == {
            "chunk_id": "en-article-art6-2",
            "lang": "en",
            "strategy": "article",
            "kind": "article",
            "unit_ids": ["art6-2"],
            "citation": "Art. 6(2)",
            "text": "Systems in Annex III are high-risk.",
            "embed_text": "[EU AI Act · Art. 6(2) · Some title]\nSystems in Annex III are high-risk.",
        }

    def test_long_unit_gets_numbered_pieces(self, chunks):
        pieces = [c for c in chunks if c["unit_ids"] == ["art5-1"]]
        assert len(pieces) >= 3
        assert [c["chunk_id"] for c in pieces] == [
            f"en-article-art5-1#{i}" for i in range(len(pieces))
        ]
        assert {c["citation"] for c in pieces} == {"Art. 5(1)"}

    def test_every_piece_has_the_header(self, chunks):
        for c in chunks:
            if c["unit_ids"] == ["art5-1"]:
                assert c["embed_text"] == "[EU AI Act · Art. 5(1) · Some title]\n" + c["text"]

    def test_chunk_ids_are_unique(self, chunks):
        assert len({c["chunk_id"] for c in chunks}) == len(chunks)

    def test_kind_is_copied(self, chunks):
        assert chunks[-1]["kind"] == "recital"

    def test_lang(self):
        chunks = article_chunks(self.UNITS[:1], "de")
        assert chunks[0]["chunk_id"] == "de-article-art6-2"
        assert chunks[0]["lang"] == "de"

    def test_does_not_modify_units(self):
        units = [dict(u) for u in self.UNITS]
        article_chunks(units, "en")
        assert units == self.UNITS


class TestFixedChunks:
    # 40 units of about 90 tokens each: several windows, each covering several units.
    UNITS = [
        unit(
            f"art{i}-1",
            " ".join(f"Rule {i} sentence {j} concerns providers of systems." for j in range(9)),
            kind="article" if i < 30 else "annex",
        )
        for i in range(40)
    ]

    @pytest.fixture
    def chunks(self):
        return fixed_chunks(self.UNITS, "en")

    def test_several_chunks(self, chunks):
        total = sum(n_tokens(u["text"]) for u in self.UNITS)
        assert total / WINDOW <= len(chunks) <= 2 * total / WINDOW + 1

    def test_ids_and_fields(self, chunks):
        assert [c["chunk_id"] for c in chunks] == [f"en-fixed-{i}" for i in range(len(chunks))]
        assert {c["strategy"] for c in chunks} == {"fixed"}
        assert {c["lang"] for c in chunks} == {"en"}
        assert all(c["embed_text"] == c["text"] for c in chunks)

    def test_no_chunk_is_longer_than_the_window(self, chunks):
        for c in chunks:
            assert n_tokens(c["text"]) <= WINDOW + 2  # +2: cutting can change a token at each edge

    def test_all_but_the_last_chunk_are_full(self, chunks):
        for c in chunks[:-1]:
            assert n_tokens(c["text"]) >= WINDOW - 2

    def test_every_unit_is_covered(self, chunks):
        covered = {u for c in chunks for u in c["unit_ids"]}
        assert covered == {u["unit_id"] for u in self.UNITS}

    def test_unit_ids_are_neighbours_in_document_order(self, chunks):
        order = [u["unit_id"] for u in self.UNITS]
        for c in chunks:
            first = order.index(c["unit_ids"][0])
            assert c["unit_ids"] == order[first : first + len(c["unit_ids"])]

    def test_a_unit_that_is_fully_inside_is_listed(self, chunks):
        for c in chunks:
            for u in self.UNITS:
                if u["text"] in c["text"]:
                    assert u["unit_id"] in c["unit_ids"]

    def test_a_listed_unit_really_overlaps_the_chunk(self, chunks):
        # Every unit has its own number in every sentence ("Rule 7 sentence ...").
        for c in chunks:
            for unit_id in c["unit_ids"][1:-1]:  # inner units are completely inside
                number = unit_id[3:].split("-")[0]
                assert f"Rule {number} sentence" in c["text"]

    def test_first_and_last(self, chunks):
        assert chunks[0]["unit_ids"][0] == "art0-1"
        assert chunks[-1]["unit_ids"][-1] == "art39-1"
        assert chunks[0]["text"].startswith("Rule 0 sentence 0")
        assert (
            chunks[-1]["text"]
            .rstrip()
            .endswith("Rule 39 sentence 8 concerns providers of systems.")
        )

    def test_neighbouring_chunks_overlap(self, chunks):
        for previous, current in zip(chunks, chunks[1:]):
            assert current["text"][:40] in previous["text"]

    def test_citation_lists_all_covered_units(self, chunks):
        for c in chunks:
            assert c["citation"] == ", ".join(u.upper() for u in c["unit_ids"])

    def test_kind_of_the_first_covered_unit(self, chunks):
        assert chunks[0]["kind"] == "article"
        assert chunks[-1]["kind"] == "annex"

    def test_one_short_unit_is_one_chunk(self):
        chunks = fixed_chunks([unit("art1-1", "Just one short sentence.")], "en")
        assert len(chunks) == 1
        assert chunks[0]["unit_ids"] == ["art1-1"]
        assert chunks[0]["text"].strip() == "Just one short sentence."

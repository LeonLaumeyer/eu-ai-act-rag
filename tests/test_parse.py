from pathlib import Path

import pytest

from aiact_rag.ingest.parse import parse

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def units():
    return {
        lang: {
            u.unit_id: u
            for u in parse(
                (ROOT / "data" / "raw" / f"ai_act_{lang}.html").read_text(encoding="utf-8"),
                lang,
            )
        }
        for lang in ("en", "de")
    }


def test_same_unit_ids_in_both_languages(units):
    assert list(units["en"]) == list(units["de"])


def test_counts(units):
    en = units["en"].values()
    assert len({u.article for u in en if u.kind == "article"}) == 113
    assert sum(u.kind == "recital" for u in en) == 180
    assert len({u.article for u in en if u.kind == "annex"}) == 13


def test_art3_has_68_definitions(units):
    defs = [i for i in units["en"] if i.startswith("art3-p")]
    assert len(defs) == 68
    assert units["en"]["art3-p1"].text.startswith("‘AI system’ means")


def test_art6_2(units):
    en, de = units["en"]["art6-2"], units["de"]["art6-2"]
    assert en.citation == "Art. 6(2)" and de.citation == "Art. 6 Abs. 2"
    assert en.text.startswith("In addition to the high-risk AI systems referred to in paragraph 1")
    assert en.chapter == "III"


def test_annex_citations(units):
    assert units["en"]["anx3-4"].citation == "Annex III, point 4"
    assert units["de"]["anx3-4"].citation == "Anhang III Nummer 4"
    assert units["en"]["anx8-s1"].citation == "Annex VIII, Section A"
    assert "Employment" in units["en"]["anx3-4"].text


def test_no_empty_or_tiny_units(units):
    for lang in units:
        for u in units[lang].values():
            assert len(u.text.split()) >= 5, (lang, u.unit_id, u.text)

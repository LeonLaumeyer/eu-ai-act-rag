"""Parse the EUR-Lex HTML of the AI Act (Regulation (EU) 2024/1689) into citable units.

One unit per:
  - article paragraph        art6-2    -> Art. 6(2)
  - article without paragraphs  art4   -> Art. 4
  - Art. 3 definition        art3-p12  -> Art. 3(12)
  - recital                  rct27     -> Recital 27
  - annex: whole annex if short, else one unit per section (Annex I, VII, VIII, X, XI)
    or per top-level point (anx3-4 -> Annex III, point 4)

unit_ids are language-independent, so EN and DE produce the same set of ids.
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from bs4 import BeautifulSoup, Tag

ROOT = Path(__file__).resolve().parents[3]

SMALL_ANNEX_WORDS = 300

ARTICLE_ID = re.compile(r"^art_(\d+)$")
PARAGRAPH_ID = re.compile(r"^(\d{3})\.(\d{3})$")
RECITAL_ID = re.compile(r"^rct_(\d+)$")
ANNEX_ID = re.compile(r"^anx_([IVXL]+)$")
CHAPTER_ID = re.compile(r"^cpt_([IVXL]+)$")


@dataclass
class Unit:
    unit_id: str  # language-independent, e.g. "art6-2", "art3-p12", "anx3-4", "rct27"
    lang: str  # "en" | "de"
    kind: str  # "article" | "annex" | "recital"
    article: str | None  # "6", or the roman numeral for annexes ("III")
    paragraph: str | None  # "2"
    point: str | None  # "4" (annex point) or "12" (Art. 3 definition)
    title: str  # "Classification rules for high-risk AI systems"
    chapter: str  # "III"
    citation: str  # "Art. 6(2)" / "Art. 6 Abs. 2"
    text: str


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def text_of(el: Tag) -> str:
    return clean(el.get_text(" ", strip=True))


def roman_to_int(roman: str) -> int:
    values = {"I": 1, "V": 5, "X": 10, "L": 50}
    total = 0
    for a, b in zip(roman, roman[1:] + " "):
        total += -values[a] if values.get(b, 0) > values[a] else values[a]
    return total


def citation(
    lang: str,
    kind: str,
    article: str | None = None,
    paragraph: str | None = None,
    point: str | None = None,
    section: str | None = None,
) -> str:
    de = lang == "de"
    if kind == "recital":
        return f"{'Erwägungsgrund' if de else 'Recital'} {article}"
    if kind == "annex":
        base = f"{'Anhang' if de else 'Annex'} {article}"
        if section:
            return f"{base} {section}" if de else f"{base}, {section}"
        if point:
            return f"{base} Nummer {point}" if de else f"{base}, point {point}"
        return base
    if paragraph:
        return f"Art. {article} Abs. {paragraph}" if de else f"Art. {article}({paragraph})"
    if point:
        return f"Art. {article} Nr. {point}" if de else f"Art. {article}({point})"
    return f"Art. {article}"


def label_of(text: str) -> str:
    """Normalise an enumeration label: '(12)', '12.' and '12' all become '12'."""
    return re.sub(r"[().\s]", "", text)


def strip_label(text: str) -> str:
    """Drop a leading enumeration label such as '(12)', '3.' or '(a)'."""
    return re.sub(r"^(\(\w+\)|\d+(\.\d+)*\.)\s*", "", text)


def parse_articles(soup: BeautifulSoup, lang: str) -> list[Unit]:
    units = []
    for art in soup.find_all("div", id=ARTICLE_ID):
        num = ARTICLE_ID.match(art["id"]).group(1)
        title_el = art.select_one(".oj-sti-art")
        title = text_of(title_el) if title_el else ""
        chapter_el = art.find_parent("div", id=CHAPTER_ID)
        chapter = CHAPTER_ID.match(chapter_el["id"]).group(1) if chapter_el else ""

        def unit(unit_id, text, paragraph=None, point=None):
            return Unit(
                unit_id,
                lang,
                "article",
                num,
                paragraph,
                point,
                title,
                chapter,
                citation(lang, "article", num, paragraph, point),
                text,
            )

        paragraphs = art.find_all("div", id=PARAGRAPH_ID, recursive=False)
        if paragraphs:
            for p in paragraphs:
                pnum = str(int(PARAGRAPH_ID.match(p["id"]).group(2)))
                units.append(unit(f"art{num}-{pnum}", strip_label(text_of(p)), paragraph=pnum))
        elif num == "3":
            for table in art.find_all("table", recursive=False):
                label_cell, body_cell = table.find_all("td", recursive=True)[:2]
                point = label_of(text_of(label_cell))
                units.append(unit(f"art3-p{point}", text_of(body_cell), point=point))
        else:
            body = [
                c
                for c in art.find_all(recursive=False)
                if not ({"oj-ti-art", "eli-title"} & set(c.get("class") or []))
            ]
            units.append(unit(f"art{num}", clean(" ".join(text_of(c) for c in body))))
    return units


def parse_recitals(soup: BeautifulSoup, lang: str) -> list[Unit]:
    units = []
    for rct in soup.find_all("div", id=RECITAL_ID):
        num = RECITAL_ID.match(rct["id"]).group(1)
        units.append(
            Unit(
                f"rct{num}",
                lang,
                "recital",
                num,
                None,
                None,
                "",
                "",
                citation(lang, "recital", num),
                strip_label(text_of(rct)),
            )
        )
    return units


def parse_annexes(soup: BeautifulSoup, lang: str) -> list[Unit]:
    units = []
    for anx in soup.find_all("div", id=ANNEX_ID):
        roman = ANNEX_ID.match(anx["id"]).group(1)
        n = roman_to_int(roman)
        headings = anx.find_all("p", class_="oj-doc-ti", recursive=False)
        title = text_of(headings[1]) if len(headings) > 1 else ""
        body = [c for c in anx.find_all(recursive=False) if c not in headings]

        def unit(unit_id, text, point=None, section=None, section_title=None):
            full_title = f"{title} – {section_title}" if section_title else title
            return Unit(
                unit_id,
                lang,
                "annex",
                roman,
                None,
                point,
                full_title,
                "",
                citation(lang, "annex", roman, point=point, section=section),
                text,
            )

        if len(text_of(anx).split()) <= SMALL_ANNEX_WORDS:
            units.append(unit(f"anx{n}", clean(" ".join(text_of(c) for c in body))))

        elif anx.find("p", class_="oj-ti-grseq-1", recursive=False):
            sections: list[tuple[str, list[str]]] = [("", [])]
            for c in body:
                if "oj-ti-grseq-1" in (c.get("class") or []):
                    if len(sections) > 1 and not sections[-1][1]:
                        sections[-1] = (f"{sections[-1][0]} — {text_of(c)}", [])
                    else:
                        sections.append((text_of(c), []))
                else:
                    sections[-1][1].append(text_of(c))
            if len(sections) > 1 and sections[0][1]:
                sections[1] = (sections[1][0], sections[0][1] + sections[1][1])
            for i, (heading, parts) in enumerate(sections[1:], start=1):
                named = re.match(r"^((?:Section|Abschnitt) \w+)", heading)
                numbered = re.match(r"^(\d+)\.", heading)
                units.append(
                    unit(
                        f"anx{n}-s{i}",
                        clean(" ".join([heading] + parts)),
                        point=numbered.group(1) if numbered else None,
                        section=named.group(1) if named else None,
                        section_title=heading,
                    )
                )

        else:
            lead_in = " ".join(text_of(c) for c in body if c.name == "p")
            for c in body:
                if c.name == "p":
                    continue
                label = label_of(text_of(c.find("td") or c.find("p")))
                text = strip_label(text_of(c))
                units.append(unit(f"anx{n}-{label}", clean(f"{lead_in} {text}"), point=label))
    return units


def parse(html: str, lang: str) -> list[Unit]:
    soup = BeautifulSoup(html, "lxml")
    for note in soup.select("span.oj-note-tag"):
        note.decompose()
    units = parse_articles(soup, lang) + parse_annexes(soup, lang) + parse_recitals(soup, lang)
    ids = [u.unit_id for u in units]
    duplicates = {i for i in ids if ids.count(i) > 1}
    assert not duplicates, f"duplicate unit_ids: {sorted(duplicates)}"
    return units


if __name__ == "__main__":
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    for lang in ["en", "de"]:
        html = (ROOT / "data" / "raw" / f"ai_act_{lang}.html").read_text(encoding="utf-8")
        units = parse(html, lang)
        with open(out_dir / f"units_{lang}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(asdict(u), ensure_ascii=False) + "\n" for u in units)
        counts = {k: sum(u.kind == k for u in units) for k in ("article", "annex", "recital")}
        print(f"{lang}: {len(units)} units {counts}")

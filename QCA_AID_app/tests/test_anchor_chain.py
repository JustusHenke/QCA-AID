"""
Tests fuer die Textanker-Anker-Kette (Spec text-anker-export, Phase 2).

Deckt: Parsen der Textanker-Zelle (R8/R11), Modell-Erweiterung anchors/
text_anchors (E4) und die Anker-Kette in anchor_codings (R9/R12): Anker
zuerst, Voll-Fallback auf Chunk-Stufen nur wenn KEIN Anker aufgeloest wird,
Teil-Erfolg markiert nur die Treffer.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

fitz = pytest.importorskip("fitz")

from QCA_AID_assets.annotation.model import Anchor, Coding  # noqa: E402
from QCA_AID_assets.preprocessing.document_index import DocumentIndexer  # noqa: E402
from webapp_logic.review_service import (  # noqa: E402
    TEXTANKER_MIN_CHARS,
    TEXTANKER_MIN_WORDS,
    anchor_codings,
    coding_from_record,
    parse_text_anchors,
)

QUOTE_P1 = "Die Befragten berichten von Zeitdruck."
QUOTE_P2 = "Zweitens fehlen Ressourcen fuer die Lehre."
CHUNK_TEXT = f"{QUOTE_P1} {QUOTE_P2}"


def make_pdf(path: Path, pages=(QUOTE_P1,), size=(595, 842)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for text in pages:
        page = doc.new_page(width=size[0], height=size[1])
        page.insert_textbox(
            fitz.Rect(72, 100, size[0] - 72, size[1] - 100),
            text,
            fontsize=11,
        )
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def two_page_pdf(tmp_path) -> Path:
    return make_pdf(tmp_path / "doc.pdf", pages=(QUOTE_P1, QUOTE_P2))


@pytest.fixture
def single_page_pdf(tmp_path) -> Path:
    return make_pdf(tmp_path / "doc1.pdf", pages=(QUOTE_P1,))


@pytest.fixture
def long_page_pdf(tmp_path) -> Path:
    lang = " ".join(["wort"] * 200)
    return make_pdf(tmp_path / "doc3.pdf", pages=(lang,))


@pytest.fixture
def fremd_pdf(tmp_path) -> Path:
    return make_pdf(tmp_path / "docf.pdf", pages=("Völlig anderer Inhalt steht hier.",))


def index_for(pdf_path: Path, doc_id: str = "doc"):
    return DocumentIndexer().build(pdf_path, doc_id)


def record(**overrides):
    base = {
        "Dokument": "doc.pdf",
        "Text": CHUNK_TEXT,
        "Hauptkategorie": "Zeitdruck",
        "Subkategorien": "Deadlines; Ueberstunden",
        "Begründung": "explizite Nennung",
        "Konfidenz": "0.82",
        "Kodiert": True,
        "Chunk_Nr": "chunk_7",
        "Textanker": f"{QUOTE_P1} | {QUOTE_P2}",
    }
    base.update(overrides)
    return base


def coding_from(**overrides):
    coding = coding_from_record(record(**overrides), doc_id="doc", fallback_segment_id="fb")
    assert coding is not None
    return coding


# ---------------------------------------------------------------------------
# parse_text_anchors (R8/R11)
# ---------------------------------------------------------------------------


class TestParseTextAnchors:
    def test_pipe_split_and_strip(self):
        out = parse_text_anchors(f"  {QUOTE_P1}  |{QUOTE_P2} ")
        assert out == [QUOTE_P1, QUOTE_P2]

    def test_empty_variants(self):
        assert parse_text_anchors("") == []
        assert parse_text_anchors(None) == []
        assert parse_text_anchors("nan") == []
        assert parse_text_anchors("  |   | ") == []

    def test_too_short_skipped(self):
        assert parse_text_anchors("kurz | zweiter") == []
        assert parse_text_anchors(f"k | {QUOTE_P1}") == [QUOTE_P1]

    def test_min_lengths(self):
        assert len(QUOTE_P1) >= TEXTANKER_MIN_CHARS
        assert len(QUOTE_P1.split()) >= TEXTANKER_MIN_WORDS
        two_words = "eins zwei drei vier"[:TEXTANKER_MIN_CHARS]
        # >= 20 Zeichen aber nur 4 Woerter -> ok; waere < 3 Woerter, faelle er
        assert len(two_words.split()) >= TEXTANKER_MIN_WORDS

    def test_deduplicates(self):
        out = parse_text_anchors(f"{QUOTE_P1} | {QUOTE_P1} | {QUOTE_P2}")
        assert out == [QUOTE_P1, QUOTE_P2]

    def test_whitespace_normalised(self):
        out = parse_text_anchors("Die  Befragten\nberichten | von Zeitdruck.")
        assert out[0] == "Die Befragten berichten"


# ---------------------------------------------------------------------------
# coding_from_record (R8)
# ---------------------------------------------------------------------------


class TestCodingFromRecordTextAnker:
    def test_textanker_geparst(self):
        coding = coding_from()
        assert coding.text_anchors == [QUOTE_P1, QUOTE_P2]

    def test_alte_dateien_ohne_spalte(self):
        rec = record()
        rec.pop("Textanker")
        coding = coding_from_record(rec, doc_id="doc", fallback_segment_id="fb")
        assert coding.text_anchors == []

    def test_garbage_anker_gefiltert(self):
        coding = coding_from(Textanker="kurz | ein einziger Langziegelstein")
        # "kurz" zu kurz, "ein einziger Langziegelstein" hat 3 Woerter und
        # 28 Zeichen -> bleibt drin (Verifikation passiert erst am PDF)
        assert coding.text_anchors == ["ein einziger Langziegelstein"]


# ---------------------------------------------------------------------------
# Modell: anchors-Liste + Kompatibilitaet (E4)
# ---------------------------------------------------------------------------


def _anchor(page=1, quote="x", anchored=True):
    return Anchor(
        doc_id="doc",
        page=page,
        quote=quote,
        quads=[[0, 0, 10, 10]] if anchored else [],
        anchored=anchored,
    )


class TestCodingModelAnchors:
    def test_nur_anchor_gesetzt_anchors_wird_abgeleitet(self):
        a = _anchor()
        coding = Coding(
            segment_id="s", doc_id="doc", category="K", anchor=a
        )
        assert coding.anchors == [a]
        assert coding.anchor is a

    def test_nur_anchors_gesetzt_anchor_zeigt_auf_erste(self):
        a1, a2 = _anchor(page=1), _anchor(page=2)
        coding = Coding(
            segment_id="s", doc_id="doc", category="K", anchors=[a1, a2]
        )
        assert coding.anchor is a1
        assert len(coding.anchors) == 2

    def test_leere_kodierung_ohne_anker_ok(self):
        coding = Coding(segment_id="s", doc_id="doc", category="K")
        assert coding.anchor is None
        assert coding.anchors == []

    def test_validate_prueft_zusatzanker(self):
        bad = Anchor(doc_id="doc", page=2, quote="y", quads=[], anchored=True)
        good = _anchor()
        coding = Coding(
            segment_id="s", doc_id="doc", category="K",
            anchor=good, anchors=[good, bad],
        )
        ok, errors = coding.validate()
        assert not ok
        assert any("Zusatzanker 2" in e for e in errors)

    def test_roundtrip_to_dict_from_dict(self):
        a1, a2 = _anchor(page=1), _anchor(page=2)
        coding = Coding(
            segment_id="s", doc_id="doc", category="K",
            anchors=[a1, a2],
            text_anchors=["erstens", "zweitens"],
        )
        data = coding.to_dict()
        assert data["text_anchors"] == ["erstens", "zweitens"]
        assert len(data["anchors"]) == 2
        restored = Coding.from_dict(data)
        assert restored.text_anchors == ["erstens", "zweitens"]
        assert len(restored.anchors) == 2
        assert restored.anchor.page == 1

    def test_from_dict_ohne_neue_felder(self):
        data = {
            "segment_id": "s",
            "doc_id": "doc",
            "category": "K",
            "anchor": _anchor().to_dict(),
        }
        coding = Coding.from_dict(data)
        assert coding.text_anchors == []
        assert len(coding.anchors) == 1


# ---------------------------------------------------------------------------
# Anker-Kette (R9/R12)
# ---------------------------------------------------------------------------


class TestAnchorChain:
    def test_beide_anker_loesen_auf_zwei_seiten(self, two_page_pdf):
        coding = coding_from()
        resolved = anchor_codings(two_page_pdf, [coding], index=index_for(two_page_pdf))

        assert len(resolved) == 1
        out = resolved[0]
        assert out.anchor.anchored
        assert len(out.anchors) == 2
        assert {a.page for a in out.anchors} == {1, 2}
        # Primaerseite = erste aufgeloeste Stelle (R10)
        assert out.page == out.anchors[0].page
        assert out.text_anchors == [QUOTE_P1, QUOTE_P2]

    def test_teil_erfolg_nur_treffer_kein_chunk_fallback(self, two_page_pdf):
        coding = coding_from(Textanker=f"{QUOTE_P2} | voellig unbekannter Absatz hier")
        resolved = anchor_codings(two_page_pdf, [coding], index=index_for(two_page_pdf))

        out = resolved[0]
        assert out.anchor.anchored
        assert len(out.anchors) == 1
        # Der Treffer ist der Textanker, nicht der Chunkkopf
        assert "Zweitens fehlen" in out.anchor.quote
        assert out.page == 2

    def test_voll_fallback_auf_chunk_stufen(self, single_page_pdf):
        coding = coding_from(Textanker="voellig unbekannter Absatz | noch ein fremder Satz", Text=QUOTE_P1)
        resolved = anchor_codings(single_page_pdf, [coding], index=index_for(single_page_pdf))

        out = resolved[0]
        assert out.anchor.anchored
        # Chunk-Fallback: der Zitatkopf ist der Chunkanfang
        assert out.anchor.quote.startswith("Die Befragten")
        assert len(out.anchors) == 1
        assert out.text_anchors != []

    def test_alle_faelle_scheitern_ohne_anker(self, fremd_pdf):
        coding = coding_from(Textanker="voellig unbekannter Absatz hier")
        resolved = anchor_codings(fremd_pdf, [coding], index=index_for(fremd_pdf))

        out = resolved[0]
        assert out.anchor is None or not out.anchor.anchored

    def test_ohne_textanker_alles_wie_frueher(self, single_page_pdf):
        rec = record()
        rec.pop("Textanker")
        rec["Text"] = QUOTE_P1
        coding = coding_from_record(rec, doc_id="doc", fallback_segment_id="fb")
        resolved = anchor_codings(single_page_pdf, [coding], index=index_for(single_page_pdf))

        out = resolved[0]
        assert out.anchor.anchored
        assert out.anchor.quote.startswith("Die Befragten")

    def test_langer_anker_faellt_auf_praefix(self, long_page_pdf):
        langer_text = " ".join(["wort"] * 200)
        langer_anker = langer_text + " Zusatz der nicht im PDF steht"
        coding = coding_from(Textanker=langer_anker, Text=langer_text)
        resolved = anchor_codings(long_page_pdf, [coding], index=index_for(long_page_pdf))

        out = resolved[0]
        assert out.anchor.anchored
        # Praefix-Stufe: gekuerzt, aber weiterhin Wortlaut des Chunks
        assert len(out.anchor.quote) < len(langer_anker)
        assert not out.anchor.quote.endswith("steht")

    def test_bereits_verankerte_bleiben_unangetastet(self, two_page_pdf):
        fixed = Coding(
            segment_id="s", doc_id="doc", category="K",
            text_anchors=["voellig unbekannter Absatz hier"],
            anchor=_anchor(page=2, quote="fix"),
        )
        resolved = anchor_codings(two_page_pdf, [fixed], index=index_for(two_page_pdf))

        assert resolved[0] is fixed
        assert resolved[0].anchor.quote == "fix"

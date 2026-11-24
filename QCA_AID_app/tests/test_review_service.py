"""
Tests fuer die Review-Logik (Analyse-XLSX -> ankerbasierte Kodierungen -> SVG).

Deckt: Zitatauswahl, Seitenfindung, XLSX-Parsing, Anker-Aufloesung und
Overlay-Erzeugung. Diese Schicht ist Streamlit-frei und damit schnell testbar.
"""

import sys
from pathlib import Path
from xml.etree import ElementTree as ET

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
    QUOTE_MAX_CHARS,
    QUOTE_STAGES,
    anchor_codings,
    available_coding_sources,
    build_overlay_map,
    coding_from_record,
    codings_from_records,
    find_page_for_quote,
    load_coding_source,
    load_codings_from_xlsx,
    normalize_doc_name,
    normalize_ws,
    quote_stages,
    select_quote,
)

QUOTE_P1 = "Die Befragten berichten von Zeitdruck."
QUOTE_P2 = "Zweitens fehlen Ressourcen fuer die Lehre."


def make_pdf(path: Path, pages=(QUOTE_P1,), size=(595, 842)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for text in pages:
        page = doc.new_page(width=size[0], height=size[1])
        # insert_textbox bricht um: laengere Testtexte bleiben vollstaendig im
        # Seitenraum (insert_text zeichnet eine Zeile ueber die Seitenbreite
        # hinaus, und Worte jenseits der Seite verschwinden aus der Extraktion).
        page.insert_textbox(
            fitz.Rect(72, 100, size[0] - 72, size[1] - 100),
            text,
            fontsize=11,
        )
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def single_page_pdf(tmp_path) -> Path:
    return make_pdf(tmp_path / "doc.pdf", pages=(QUOTE_P1,))


@pytest.fixture
def two_page_pdf(tmp_path) -> Path:
    return make_pdf(tmp_path / "doc2.pdf", pages=(QUOTE_P1, QUOTE_P2))


def index_for(pdf_path: Path, doc_id: str = "doc"):
    return DocumentIndexer().build(pdf_path, doc_id)


# ---------------------------------------------------------------------------
# Zitatauswahl
# ---------------------------------------------------------------------------


class TestSelectQuote:
    def test_short_text_unchanged(self):
        assert select_quote("Kurzer Satz.") == "Kurzer Satz."

    def test_whitespace_normalised(self):
        assert select_quote("Zeile eins.\n\n  Zeile zwei.") == "Zeile eins. Zeile zwei."

    def test_truncates_at_word_boundary(self):
        text = " ".join(["wort"] * 200)

        quote = select_quote(text, max_chars=50)

        assert len(quote) <= 50
        assert not quote.endswith("wor")
        assert text.startswith(quote)

    def test_strips_path_artifacts(self):
        quote = select_quote("C:\\Users\\justu\\Dokumente\\interview.pdf Inhalt bleibt.")

        assert "Users" not in quote
        assert "interview.pdf" not in quote
        assert "Inhalt bleibt." in quote

    def test_only_artifacts_returns_empty(self):
        assert select_quote("file:///C:/tmp/x.pdf") == ""

    def test_empty_input(self):
        assert select_quote("") == ""
        assert select_quote(None) == ""

    def test_default_limit_is_respected(self):
        assert len(select_quote("a " * 400)) <= QUOTE_MAX_CHARS


class TestQuoteStages:
    def test_long_text_produces_descending_stages(self):
        text = " ".join(f"Wort{i}" for i in range(400))  # ~3.000 Zeichen

        stages = quote_stages(text)

        assert len(stages) == len(QUOTE_STAGES)
        lengths = [len(stage) for stage in stages]
        assert lengths == sorted(lengths, reverse=True)
        cleaned = normalize_ws(text)
        for stage in stages:
            assert cleaned.startswith(stage)

    def test_short_text_is_single_stage(self):
        assert quote_stages("Kurzer Satz.") == ["Kurzer Satz."]

    def test_duplicate_stages_are_removed(self):
        stages = quote_stages("a " * 130)  # ~260 Zeichen: 800 und 500 schneiden nicht

        assert len(stages) == 2
        assert stages[0].startswith(stages[1])

    def test_empty_text_has_no_stages(self):
        assert quote_stages("   ") == []

    def test_default_limit_is_longest_stage(self):
        text = " ".join(["wort"] * 200)

        assert quote_stages(text)[0] == select_quote(text, max_chars=QUOTE_STAGES[0])


class TestNormalizeWs:
    def test_collapses_all_whitespace(self):
        assert normalize_ws(" a\t b\r\n c ") == "a b c"

    def test_soft_hyphen_removed(self):
        assert normalize_ws("Sil\u00adben") == "Silben"


class TestFindPage:
    def test_finds_page_with_quote(self, two_page_pdf):
        index = index_for(two_page_pdf)

        assert find_page_for_quote(index, "Zweitens fehlen Ressourcen") == 2

    def test_normalises_whitespace_for_match(self, single_page_pdf):
        index = index_for(single_page_pdf)

        assert find_page_for_quote(index, "Die   Befragten\nberichten") == 1

    def test_returns_none_when_absent(self, single_page_pdf):
        index = index_for(single_page_pdf)

        assert find_page_for_quote(index, "voellig anderer text") is None

    def test_none_index(self):
        assert find_page_for_quote(None, QUOTE_P1) is None

    def test_empty_quote(self, single_page_pdf):
        assert find_page_for_quote(index_for(single_page_pdf), "") is None

    def test_long_quote_falls_back_to_head(self, single_page_pdf):
        # Zitat laeuft ueber das Seitenende hinaus: der Kopf muss reichen
        index = index_for(single_page_pdf)

        quote = QUOTE_P1 + " " + "Nachtrag " * 40

        assert find_page_for_quote(index, quote) == 1


# ---------------------------------------------------------------------------
# XLSX-Zeilen -> Coding
# ---------------------------------------------------------------------------


def record(**overrides):
    base = {
        "Dokument": "interview.pdf",
        "Text": QUOTE_P1,
        "Hauptkategorie": "Zeitdruck",
        "Subkategorien": "Deadlines; Ueberstunden",
        "Begründung": "explizite Nennung",
        "Konfidenz": "0.82",
        "Kodiert": True,
        "Chunk_Nr": "chunk_7",
    }
    base.update(overrides)
    return base


class TestCodingFromRecord:
    def test_maps_fields(self):
        coding = coding_from_record(record(), doc_id="doc", fallback_segment_id="fb")

        assert coding is not None
        assert coding.segment_id == "chunk_7"
        assert coding.category == "Zeitdruck"
        assert coding.subcategories == ["Deadlines", "Ueberstunden"]
        assert coding.confidence == pytest.approx(0.82)
        assert coding.status == "proposed"
        assert coding.created_by == "ai"

    def test_skips_uncoded_rows(self):
        assert coding_from_record(record(Kodiert="Nein"), "doc", "fb") is None
        assert coding_from_record(record(Kodiert=0), "doc", "fb") is None

    def test_skips_empty_category(self):
        assert coding_from_record(record(Hauptkategorie=""), "doc", "fb") is None
        assert coding_from_record(record(Hauptkategorie="nan"), "doc", "fb") is None

    def test_skips_rows_without_usable_text(self):
        assert coding_from_record(record(Text="file:///C:/tmp/a.pdf"), "doc", "fb") is None

    def test_fallback_segment_id(self):
        coding = coding_from_record(record(Chunk_Nr=""), "doc", "doc_row_4")

        assert coding is not None
        assert coding.segment_id == "doc_row_4"

    def test_comma_decimal_confidence(self):
        coding = coding_from_record(record(Konfidenz="0,75"), "doc", "fb")

        assert coding is not None
        assert coding.confidence == pytest.approx(0.75)

    def test_nested_confidence_dict(self):
        coding = coding_from_record(record(Konfidenz={"total": 0.6}), "doc", "fb")

        assert coding is not None
        assert coding.confidence == pytest.approx(0.6)

    def test_percent_confidence(self):
        coding = coding_from_record(record(Konfidenz=85), "doc", "fb")

        assert coding is not None
        assert coding.confidence == pytest.approx(0.85)

    def test_unparsable_confidence_is_none(self):
        coding = coding_from_record(record(Konfidenz="n/a"), "doc", "fb")

        assert coding is not None
        assert coding.confidence is None

    def test_inference_without_justification_downgraded(self):
        coding = coding_from_record(record(Inferenz="Ja", Begründung=""), "doc", "fb")

        assert coding is not None
        assert coding.inference is False

    def test_inference_with_justification_kept(self):
        coding = coding_from_record(record(Inferenz="Ja"), "doc", "fb")

        assert coding is not None
        assert coding.inference is True

    def test_long_source_text_kept_for_staged_anchoring(self):
        text = " ".join(f"Wort{i}" for i in range(300))  # ~2.100 Zeichen

        coding = coding_from_record(record(Text=text), "doc", "fb")

        assert coding is not None
        assert coding.anchor.quote == normalize_ws(text)

    def test_comma_separated_subcategories(self):
        coding = coding_from_record(record(Subkategorien="A, B , C"), "doc", "fb")

        assert coding is not None
        assert coding.subcategories == ["A", "B", "C"]


class TestCodingsFromRecords:
    def test_filters_by_document_name(self):
        records = [record(), record(Dokument="anderes.pdf")]

        codings = codings_from_records(records, doc_id="doc", doc_name="interview.pdf")

        assert len(codings) == 1

    def test_without_filter_keeps_all(self):
        records = [record(), record(Dokument="anderes.pdf")]

        assert len(codings_from_records(records, doc_id="doc")) == 2

    def test_name_match_ignores_path_extension_case(self):
        records = [record(Dokument="C:/input/Interview_Final.DOCX")]

        codings = codings_from_records(records, doc_id="doc", doc_name="interview_final.pdf")

        assert len(codings) == 1

    def test_unlinked_document_yields_nothing(self):
        """Ein nicht kodiertes Dokument darf keine fremden Kodierungen kriegen."""
        records = [record(), record(Dokument="anderes.pdf")]

        assert codings_from_records(records, doc_id="doc", doc_name="ohne_kodierungen") == []

    def test_blank_document_column_is_kept(self):
        """Zeilen ohne Dokumentangabe werden nicht weggefiltert (alte Exporte)."""
        records = [record(Dokument="")]

        assert len(codings_from_records(records, doc_id="doc", doc_name="interview.pdf")) == 1


class TestLoadFromXlsx:
    def test_roundtrip(self, tmp_path):
        pd = pytest.importorskip("pandas")
        pytest.importorskip("openpyxl")
        path = tmp_path / "QCA-AID_Analysis_deductive_2026.xlsx"
        pd.DataFrame([record(), record(Dokument="anderes.pdf")]).to_excel(path, index=False)

        codings = load_codings_from_xlsx(path, doc_id="doc", doc_name="interview.pdf")

        assert len(codings) == 1
        assert codings[0].category == "Zeitdruck"

    def test_source_meta_lists_documents(self, tmp_path):
        pd = pytest.importorskip("pandas")
        pytest.importorskip("openpyxl")
        path = tmp_path / "QCA-AID_Analysis_deductive_2026.xlsx"
        pd.DataFrame([record(), record(Dokument="anderes.pdf")]).to_excel(path, index=False)

        records, documents = load_coding_source(path)

        assert sorted(documents) == ["anderes", "interview"]
        assert len(records) == 2

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_codings_from_xlsx(tmp_path / "fehlt.xlsx", doc_id="doc")


class TestAvailableSources:
    def test_sorted_newest_first(self, tmp_path):
        import os
        import time

        old = tmp_path / "QCA-AID_Analysis_a.xlsx"
        new = tmp_path / "QCA-AID_Analysis_b.xlsx"
        old.write_bytes(b"x")
        new.write_bytes(b"x")
        now = time.time()
        os.utime(old, (now - 100, now - 100))
        os.utime(new, (now, now))

        assert [p.name for p in available_coding_sources(tmp_path)] == [
            "QCA-AID_Analysis_b.xlsx",
            "QCA-AID_Analysis_a.xlsx",
        ]

    def test_missing_dir(self, tmp_path):
        assert available_coding_sources(tmp_path / "nix") == []


# ---------------------------------------------------------------------------
# Anker
# ---------------------------------------------------------------------------


def unresolved_coding(quote: str, segment_id: str = "seg_1") -> Coding:
    """Kodierung wie sie aus der XLSX kommt: Zitat bekannt, Anker unaufgeloest."""
    return Coding(
        segment_id=segment_id,
        doc_id="doc",
        category="Zeitdruck",
        anchor=Anchor(doc_id="doc", page=1, quote=quote, anchored=False),
    )


def anchored_coding(
    page: int = 1,
    segment_id: str = "seg_1",
    category: str = "Zeitdruck",
    quads=None,
) -> Coding:
    anchor = Anchor(
        doc_id="doc",
        page=page,
        quote=QUOTE_P1,
        quads=quads if quads is not None else [[72.0, 108.0, 300.0, 122.0]],
        anchored=True,
    )
    return Coding(segment_id=segment_id, doc_id="doc", category=category, anchor=anchor)


class TestAnchorCodings:
    def test_anchors_resolvable_coding(self, single_page_pdf):
        index = index_for(single_page_pdf)

        result = anchor_codings(single_page_pdf, [unresolved_coding(QUOTE_P1)], index=index)

        assert len(result) == 1
        anchor = result[0].anchor
        assert anchor is not None
        assert anchor.anchored is True
        assert anchor.page == 1
        assert anchor.quads
        assert result[0].page == 1

    def test_unresolvable_coding_keeps_quote_but_no_anchor(self, single_page_pdf):
        index = index_for(single_page_pdf)

        result = anchor_codings(
            single_page_pdf,
            [unresolved_coding("Dieser Satz steht nirgends im Dokument.")],
            index=index,
        )

        anchor = result[0].anchor
        assert anchor is not None
        assert anchor.anchored is False
        assert anchor.quote == "Dieser Satz steht nirgends im Dokument."

    def test_already_anchored_coding_is_untouched(self, single_page_pdf):
        coding = anchored_coding()

        result = anchor_codings(single_page_pdf, [coding], index=index_for(single_page_pdf))

        assert result[0] is coding

    def test_long_chunk_on_one_page_is_anchored_in_full(self, tmp_path):
        """Variante 1: bis zu 800 Zeichen werden markiert, nicht nur der Kopf."""
        continuation = " ".join(f"Wort{i}" for i in range(60))  # ~420 Zeichen
        body = f"{QUOTE_P1} {continuation}"

        pdf = make_pdf(tmp_path / "lang.pdf", pages=(body,))

        result = anchor_codings(pdf, [unresolved_coding(body)], index=index_for(pdf))[0]

        anchor = result.anchor
        assert anchor.anchored is True
        assert len(anchor.quote) > QUOTE_MAX_CHARS  # laengere Stufe als der Kopf
        assert anchor.quote.startswith(QUOTE_P1)

    def test_falls_back_to_shorter_stage_when_text_deviates(self, tmp_path):
        """Stimmt nur der Kopf, greift die letzte Stufe - nie 'ohne Anker'."""
        page_body = f"{QUOTE_P1} {'A' * 200} {'B' * 200}"
        source_text = f"{QUOTE_P1} {'A' * 200} {'Störfall ' * 20}{'B' * 200}"

        pdf = make_pdf(tmp_path / "abweichung.pdf", pages=(page_body,))

        result = anchor_codings(pdf, [unresolved_coding(source_text)], index=index_for(pdf))[0]

        anchor = result.anchor
        assert anchor.anchored is True
        # Die letzte passende Stufe ist der Kopf bis zur Abweichung - laenger
        # als der 240er-Default, aber ohne den abweichenden Teil.
        assert anchor.quote == quote_stages(source_text)[-1]
        assert len(anchor.quote) < len(source_text)
        assert anchor.quote.startswith(QUOTE_P1)

    def test_short_quote_behaviour_is_unchanged(self, single_page_pdf):
        index = index_for(single_page_pdf)

        result = anchor_codings(single_page_pdf, [unresolved_coding(QUOTE_P1)], index=index)

        assert result[0].anchor.quote == QUOTE_P1

    def test_keeps_category_and_status(self, single_page_pdf):
        coding = Coding(
            segment_id="s",
            doc_id="doc",
            category="Zeitdruck",
            subcategories=["Sub"],
            confidence=0.7,
            status="accepted",
            anchor=None,
        )

        result = anchor_codings(single_page_pdf, [coding], index=index_for(single_page_pdf))[0]

        assert result.category == "Zeitdruck"
        assert result.subcategories == ["Sub"]
        assert result.confidence == pytest.approx(0.7)
        assert result.status == "accepted"

    def test_second_page_detected(self, two_page_pdf):
        index = index_for(two_page_pdf)

        result = anchor_codings(two_page_pdf, [unresolved_coding(QUOTE_P2)], index=index)[0]

        assert result.anchor is not None
        assert result.anchor.page == 2

    def test_resolver_error_does_not_break_batch(self, single_page_pdf):
        index = index_for(single_page_pdf)

        class ExplodingResolver:
            def resolve(self, **kwargs):
                raise RuntimeError("kaputt")

        result = anchor_codings(
            single_page_pdf,
            [unresolved_coding(QUOTE_P1, "a"), unresolved_coding(QUOTE_P1, "b")],
            index=index,
            resolver=ExplodingResolver(),
        )

        assert len(result) == 2
        assert all(coding.anchor is not None and coding.anchor.anchored is False for coding in result)


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------


def _layer_children(svg: str, css_class: str) -> list:
    root = ET.fromstring(svg)
    for element in root.iter():
        if element.get("class") == css_class:
            return list(element)
    return []


class TestBuildOverlayMap:
    def test_empty_without_anchors(self):
        assert build_overlay_map([(595.0, 842.0)], [unresolved_coding(QUOTE_P1)]) == {}

    def test_svg_per_page(self):
        overlays = build_overlay_map(
            [(595.0, 842.0), (595.0, 842.0)],
            [anchored_coding(1, "a"), anchored_coding(2, "b")],
            active_coding_id="a",
        )

        assert set(overlays) == {1, 2}
        assert "<svg" in overlays[1]
        assert 'viewBox="0 0 595.0 842.0"' in overlays[1]

    def test_multiply_blend_and_opacity(self):
        overlays = build_overlay_map([(595.0, 842.0)], [anchored_coding()])

        assert "mix-blend-mode:multiply" in overlays[1]
        assert "opacity:0.4" in overlays[1]

    def test_highlight_mode_draws_in_text(self):
        overlays = build_overlay_map([(595.0, 842.0)], [anchored_coding()], mode="highlight")

        assert _layer_children(overlays[1], "qca-highlights")
        assert _layer_children(overlays[1], "qca-margin-overlay") == []

    def test_margin_mode_draws_no_in_text_marks(self):
        overlays = build_overlay_map([(595.0, 842.0)], [anchored_coding()], mode="margin")

        assert _layer_children(overlays[1], "qca-highlights") == []
        assert _layer_children(overlays[1], "qca-margin-overlay")

    def test_page_out_of_range_is_skipped(self):
        assert build_overlay_map([(595.0, 842.0)], [anchored_coding(page=5)]) == {}

    def test_two_codings_of_one_segment_active_in_text_other_margin(self):
        overlays = build_overlay_map(
            [(595.0, 842.0)],
            [anchored_coding(1, "a", "Zeitdruck"), anchored_coding(1, "b", "Andere")],
            active_coding_id="a",
        )

        highlight = _layer_children(overlays[1], "qca-highlights")
        solid = [el for el in highlight if not el.get("fill", "").startswith("url(")]
        hatched = [el for el in highlight if el.get("fill", "").startswith("url(")]
        assert [el.get("data-segment-id") for el in solid] == ["a"]
        assert [el.get("data-segment-id") for el in hatched] == ["b"]
        assert len(_layer_children(overlays[1], "qca-margin-overlay")) == 1

    def test_distinct_segments_are_both_in_text(self):
        overlays = build_overlay_map(
            [(595.0, 842.0)],
            [
                anchored_coding(1, "a"),
                anchored_coding(1, "b", "Andere", quads=[[72.0, 140.0, 300.0, 154.0]]),
            ],
            active_coding_id="a",
        )

        assert len(_layer_children(overlays[1], "qca-highlights")) == 2

    def test_same_text_area_is_highlighted_once_with_a_stripe_for_the_twin(self):
        """Mehrfachkodierung desselben Chunks: gleiche Quads, zwei Kodierungen."""
        overlays = build_overlay_map(
            [(595.0, 842.0)],
            [anchored_coding(1, "a"), anchored_coding(1, "b", "Andere")],
            active_coding_id="a",
        )

        highlight = _layer_children(overlays[1], "qca-highlights")
        solid = [el for el in highlight if not el.get("fill", "").startswith("url(")]
        hatched = [el for el in highlight if el.get("fill", "").startswith("url(")]
        margin = _layer_children(overlays[1], "qca-margin-overlay")
        assert [el.get("data-segment-id") for el in solid] == ["a"]
        assert [el.get("data-segment-id") for el in hatched] == ["b"]
        assert [el.get("data-segment-id") for el in margin] == ["b"]

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            build_overlay_map([(595.0, 842.0)], [anchored_coding()], mode="unbekannt")

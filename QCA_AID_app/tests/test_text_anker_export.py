"""
Tests fuer die Textanker-Spalte in der Analyse-XLSX (Spec text-anker-export).

Deckt Phase 1 ab: format_text_anchors (R2-R5), Spaltenintegration im
Export-Zeilen-Dict (R1) und die Prompt-/Datenfluss-Anbindung (T1-Fixes).
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from QCA_AID_assets.export.results_exporter import (  # noqa: E402
    TEXTANKER_MAX_CHARS,
    TEXTANKER_MAX_COUNT,
    format_text_anchors,
    normalize_ws,
)

CHUNK = (
    "Die Hochschule steht unter finanziellem Druck. Die Grundfinanzierung "
    "sinkt seit Jahren, waehrend die Studierendenzahlen steigen. "
    "Deshalb werden Third-Mittel immer wichtiger."
)

REF1 = "Die Grundfinanzierung sinkt seit Jahren"
REF2 = "Third-Mittel immer wichtiger"


def _coding(category="Legitimationsstrategien", refs=None, **extra):
    return {"category": category, "text_references": refs or [], **extra}


# --------------------------------------------------------------------- R2/R5

def test_kodierte_zeile_bekommt_anker_mit_pipe_trenner():
    out = format_text_anchors(_coding(refs=[REF1, REF2]), CHUNK)
    assert out == f"{REF1} | {REF2}"
    assert "\n" not in out


def test_leere_referenzen_leere_zelle():
    assert format_text_anchors(_coding(refs=[]), CHUNK) == ""
    assert format_text_anchors(_coding(refs=None), CHUNK) == ""


# ----------------------------------------------------------------------- R3

def test_paraphrase_wird_verworfen():
    out = format_text_anchors(
        _coding(refs=[REF1, "Die Hochschule hat Geldprobleme"]), CHUNK
    )
    assert out == REF1


def test_anker_ausserhalb_des_chunks_wird_verworfen():
    assert format_text_anchors(_coding(refs=["Vollig anderer Text"]), CHUNK) == ""


def test_whitespace_wird_normalisiert_und_trotzdem_gewonnen():
    ref_mit_umbruechen = "Die Grundfinanzierung\n  sinkt   seit Jahren"
    out = format_text_anchors(_coding(refs=[ref_mit_umbruechen]), CHUNK)
    assert out == REF1


def test_chunktext_normalisiert_verglichen():
    chunk_mit_umbruechen = CHUNK.replace("sinkt seit Jahren", "sinkt\nseit   Jahren")
    assert format_text_anchors(_coding(refs=[REF1]), chunk_mit_umbruechen) == REF1


def test_control_zeichen_blockieren_nicht():
    ref = "Die Grundfinanzierung \x00 sinkt seit Jahren"
    out = format_text_anchors(_coding(refs=[ref]), CHUNK)
    assert out == REF1


# ----------------------------------------------------------------------- R2

def test_pipe_im_auszug_wird_verworfen():
    ref_mit_pipe = "Die Grundfinanzierung | sinkt seit Jahren"
    out = format_text_anchors(_coding(refs=[ref_mit_pipe, REF2]), CHUNK)
    assert out == REF2


# ----------------------------------------------------------------------- R4

def test_max_count_wird_eingehalten():
    refs = [f"Ausschnitt {i} steht hier" for i in range(10)]
    # Nur echte Teile des Chunks zaehlen - baue 8 gueltige Varianten:
    basis = [
        "Die Hochschule steht unter finanziellem Druck",
        REF1,
        REF2,
        "waehrend die Studierendenzahlen steigen",
        "Deshalb werden Third-Mittel",
        "unter finanziellem Druck",
        "immer wichtiger",
        "sinkt seit Jahren",
    ]
    refs = basis + ["nicht im chunk", "auch nicht"]
    out = format_text_anchors(_coding(refs=refs), CHUNK)
    assert len(out.split(" | ")) == TEXTANKER_MAX_COUNT


def test_ueberlanger_anker_wird_verworfen():
    ref = "A" * (TEXTANKER_MAX_CHARS + 1)
    out = format_text_anchors(_coding(refs=[ref, REF1]), CHUNK)
    assert out == REF1


# ----------------------------------------------------------------- R3a / R5

@pytest.mark.parametrize("category", ["Nicht kodiert", "nicht kodiert"])
def test_nicht_kodierte_kategorie_bekommt_keine_anker(category):
    assert format_text_anchors(_coding(category=category, refs=[REF1]), CHUNK) == ""


def test_leere_kategorie_gilt_als_unkodiert():
    assert format_text_anchors(_coding(category="", refs=[REF1]), CHUNK) == ""


def test_andere_kategorie_bekommt_anker():
    out = format_text_anchors(
        _coding(category="Keine passende Kategorie", refs=[REF1]), CHUNK
    )
    assert out == REF1


def test_leerer_chunk_leere_zelle():
    assert format_text_anchors(_coding(refs=[REF1]), "") == ""


# -------------------------------------------------------------- Duplikate

def test_duplikate_werden_entfernt():
    out = format_text_anchors(_coding(refs=[REF1, REF1]), CHUNK)
    assert out == REF1


def test_nicht_string_referenzen_werden_ignoriert():
    out = format_text_anchors(_coding(refs=[42, None, REF1]), CHUNK)
    assert out == REF1


# ----------------------------------------------------------------- Utilities

def test_normalize_ws():
    assert normalize_ws("  a \n\t b  ") == "a b"
    assert normalize_ws(None) == ""


# ------------------------------------------------- Datenfluss (T1-Regression)

def test_unified_analysis_result_traegt_text_references():
    from QCA_AID_assets.optimization.unified_relevance_analyzer import (
        UnifiedAnalysisResult,
    )

    result = UnifiedAnalysisResult(
        segment_id="doc_chunk_0",
        primary_category="Kategorie A",
        all_categories=[],
        relevance_scores={"Kategorie A": 0.9},
        confidence=0.9,
        requires_multiple_coding=False,
        text_references=[REF1],
    )
    assert result.text_references == [REF1]


def test_parse_comprehensive_result_liest_text_references():
    from QCA_AID_assets.optimization.unified_relevance_analyzer import (
        UnifiedRelevanceAnalyzer,
    )

    self_fake = SimpleNamespace(multiple_coding_threshold=0.7)
    parsed = UnifiedRelevanceAnalyzer._parse_comprehensive_result(
        self_fake,
        {
            "segment_id": "doc_chunk_0",
            "primary_category": "Kategorie A",
            "confidence": 0.9,
            "justification": "weil",
            "text_references": [REF1, REF2],
        },
        "doc_chunk_0",
    )
    assert parsed.text_references == [REF1, REF2]


def test_parse_comprehensive_result_ohne_text_references_leer():
    from QCA_AID_assets.optimization.unified_relevance_analyzer import (
        UnifiedRelevanceAnalyzer,
    )

    self_fake = SimpleNamespace(multiple_coding_threshold=0.7)
    parsed = UnifiedRelevanceAnalyzer._parse_comprehensive_result(
        self_fake,
        {"segment_id": "doc_chunk_0", "primary_category": "Kategorie A"},
        "doc_chunk_0",
    )
    assert parsed.text_references == []


# ------------------------------------------------------- Prompts (E1/T3)

def test_prompts_fordern_text_references():
    src = (REPO_ROOT / "QCA_AID_assets" / "QCA_Prompts.py").read_text(
        encoding="utf-8"
    )
    # Einzeln + Fokus + Fokus-Kontext + Batch = 4 Schema-Stellen
    assert src.count('"text_references":') >= 4
    # Wortlaut-Regel im Prompt verankert
    assert "wortwoertlicher" in src
    # Batch-Anweisung 5 (TEXTANKER) vorhanden
    assert "TEXTANKER (text_references)" in src


def test_focus_fallback_nimmt_llm_referenzen():
    """code_chunk_with_focus nutzt LLM-Referenzen statt nur chunk[:100]."""
    src = (
        REPO_ROOT / "QCA_AID_assets" / "analysis" / "deductive_coding.py"
    ).read_text(encoding="utf-8")
    assert 'result.get("text_references", []) or [chunk[:100]]' in src


# ------------------------------------------------- Export-Zeile (R1, T5)

def test_export_zeile_fuehrt_textanker_nach_begruendung():
    src = (
        REPO_ROOT / "QCA_AID_assets" / "export" / "results_exporter.py"
    ).read_text(encoding="utf-8")
    begr = src.index("'Begründung': sanitize_text_for_excel(justification),")
    anker = src.index("'Textanker': format_text_anchors(coding, text),")
    conf = src.index("'Konfidenz':", anker)
    assert begr < anker < conf

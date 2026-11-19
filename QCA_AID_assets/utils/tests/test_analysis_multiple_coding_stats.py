"""
Regressionstests fuer `calculate_multiple_coding_stats`.

Hintergrund: Im induktiven Modus konnten Kodierungen mit `category = None`
entstehen (ReviewManager: `target_category = None` fuer Segmente ohne gueltige
Kategorie, sowie LLM-Antworten mit `null`). Die Statistik-Funktion stuerzte
dann mit
`TypeError: '<' not supported between instances of 'NoneType' and 'str'`
ab - nachdem der Export bereits geschrieben war.

Die Tests stellen sicher, dass `None`/leere Werte in Kategorie und Coder-ID
nicht mehr zu Abstuerzen fuehren und dass die Kombinationen lesbar bleiben.
"""

import pytest

from QCA_AID_assets.utils.analysis import (
    EMPTY_CATEGORY_LABEL,
    calculate_multiple_coding_stats,
)


def _coding(segment_id="doc_chunk_0", coder_id="auto_1", category="A", **extra):
    coding = {
        "segment_id": segment_id,
        "coder_id": coder_id,
        "category": category,
        "subcategories": [],
        "total_coding_instances": 1,
    }
    coding.update(extra)
    return coding


def test_none_category_does_not_raise():
    codings = [
        _coding(coder_id="auto_1", category=None),
        _coding(coder_id="auto_2", category="Kategorie B"),
    ]

    stats = calculate_multiple_coding_stats(codings)

    assert stats["segments_with_multiple_categories"] == 1
    assert stats["top_combinations"] == [f"{EMPTY_CATEGORY_LABEL} + Kategorie B"]
    assert stats["total_segments"] == 1
    assert stats["segments_with_multiple_coders"] == 1


def test_all_categories_none_does_not_raise():
    codings = [
        _coding(coder_id="auto_1", category=None),
        _coding(coder_id="auto_2", category=None),
    ]

    stats = calculate_multiple_coding_stats(codings)

    # Zwei Kodierer, aber (nach Normalisierung) nur eine Kategorie
    assert stats["segments_with_multiple_coders"] == 1
    assert stats["segments_with_multiple_categories"] == 0
    assert stats["top_combinations"] == []


def test_none_coder_id_and_instances_do_not_raise():
    codings = [
        _coding(coder_id=None, category="A", total_coding_instances=None),
        _coding(coder_id="auto_2", category="B", total_coding_instances="2"),
    ]

    stats = calculate_multiple_coding_stats(codings)

    assert stats["segments_with_multiple_coders"] == 1
    assert stats["segments_with_multiple_categories"] == 1
    # "2" (String) muss als 2 erkannt werden
    assert stats["segments_with_true_multiple_coding"] == 1


def test_none_segment_id_groups_instead_of_crashing():
    codings = [
        _coding(segment_id=None, coder_id="auto_1", category="A"),
        _coding(segment_id=None, coder_id="auto_2", category="B"),
    ]

    stats = calculate_multiple_coding_stats(codings)

    assert stats["total_segments"] == 1
    assert stats["segments_with_multiple_categories"] == 1


def test_normal_codings_stay_unchanged():
    codings = [
        _coding(segment_id="d_chunk_1", coder_id="auto_1", category="Alpha"),
        _coding(segment_id="d_chunk_1", coder_id="auto_2", category="Beta"),
        _coding(segment_id="d_chunk_2", coder_id="auto_1", category="Alpha"),
        _coding(segment_id="d_chunk_2", coder_id="auto_2", category="Alpha"),
    ]

    stats = calculate_multiple_coding_stats(codings)

    assert stats["total_segments"] == 2
    assert stats["segments_with_multiple"] == 2
    assert stats["segments_with_multiple_categories"] == 1
    assert stats["top_combinations"] == ["Alpha + Beta"]
    assert stats["avg_codings_per_segment"] == 2.0


def test_focus_adherence_with_none_values_does_not_raise():
    codings = [
        _coding(category=None, target_category=None, category_focus_used=True),
        _coding(
            coder_id="auto_2",
            category="Alpha",
            target_category="Alpha",
            category_focus_used=True,
        ),
    ]

    stats = calculate_multiple_coding_stats(codings)

    # Beide Kodierungen erfuellen den Fokus (None == None, Alpha == Alpha)
    assert stats["focus_adherence_rate"] == 1.0


@pytest.mark.parametrize("codings", [[], [{}]])
def test_edge_cases_do_not_raise(codings):
    stats = calculate_multiple_coding_stats(codings)

    assert stats["total_segments"] == len(codings)
    assert stats["avg_codings_per_segment"] in (0, 1.0)

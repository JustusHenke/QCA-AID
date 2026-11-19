"""
Regressionstest fuer den ReviewManager: Kategorien duerfen nicht `None` werden.

Im induktiven Modus erzeugte `_create_category_specific_segments` fuer Segmente
ohne gueltige Kategorie ein `target_category = None`. Der Consensus-Pfad
schrieb dieses `None` anschliessend als `category` in die finalen Kodierungen.
Folge: `calculate_multiple_coding_stats` stuerzte mit
`TypeError: '<' not supported between instances of 'NoneType' and 'str'` ab,
und der Export enthielt Kodierungen ohne Kategorienamen.
"""

from QCA_AID_assets.quality.review_manager import ReviewManager


def _coding(coder_id="auto_1", category="A", confidence=0.9):
    return {
        "segment_id": "doc_chunk_0",
        "coder_id": coder_id,
        "category": category,
        "subcategories": [],
        "confidence": {"total": confidence, "category": confidence},
        "justification": "",
    }


def test_segment_without_valid_category_gets_fallback_category():
    manager = ReviewManager(output_dir=".")
    codings = [
        _coding(coder_id="auto_1", category="Nicht kodiert"),
        _coding(coder_id="auto_2", category="Nicht kodiert", confidence=0.7),
    ]

    segments = manager._create_category_specific_segments(codings)

    assert len(segments) == 1
    assert segments[0]["target_category"] is None

    consensus = manager._get_consensus_for_category_segment(segments[0])

    assert consensus is not None
    assert consensus["category"] == "Nicht kodiert"


def test_none_categories_get_fallback_instead_of_none():
    manager = ReviewManager(output_dir=".")
    codings = [
        _coding(coder_id="auto_1", category=None),
        _coding(coder_id="auto_2", category=None, confidence=0.6),
    ]

    segments = manager._create_category_specific_segments(codings)
    assert segments[0]["target_category"] is None

    consensus = manager._get_consensus_for_category_segment(segments[0])

    assert consensus is not None
    assert consensus["category"] == "Nicht kodiert"
    assert not isinstance(consensus["category"], type(None))


def test_valid_target_category_is_preserved():
    manager = ReviewManager(output_dir=".")
    codings = [
        _coding(coder_id="auto_1", category="Alpha"),
        _coding(coder_id="auto_2", category="Alpha", confidence=0.5),
    ]

    segments = manager._create_category_specific_segments(codings)
    consensus = manager._get_consensus_for_category_segment(segments[0])

    assert consensus is not None
    assert consensus["category"] == "Alpha"

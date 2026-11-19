"""
Tests für Krippendorff's Alpha (QCA_AID_assets/quality/reliability.py)
======================================================================

Referenzwerte wurden gegen die Referenzimplementierung des Pakets
`krippendorff` (0.9.0, Umsetzung von Krippendorff 2011) geprüft.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from QCA_AID_assets.quality.reliability import ReliabilityCalculator


@pytest.fixture()
def calc():
    return ReliabilityCalculator(verbose=False)


def make_coding(segment_id, coder_id, category, subcategories=None):
    return {
        "segment_id": segment_id,
        "coder_id": coder_id,
        "category": category,
        "subcategories": list(subcategories or []),
    }


# ----------------------------------------------------------------------
# Numerische Validierung gegen die Referenzimplementierung
# ----------------------------------------------------------------------
def test_nominal_matches_reference_implementation(calc):
    """Krippendorff (2011), Beispieldatensatz: 4 Kodierer, 12 Einheiten."""
    table = [
        [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None],
        [1, 2, 3, 3, 2, 1, 4, 1, 2, 5, None, 3],
        [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None],
        [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None],
    ]

    units = []
    for unit_index in range(12):
        values = [row[unit_index] for row in table if row[unit_index] is not None]
        if len(values) >= 2:
            units.append(values)

    result = calc._alpha(units, calc._nominal_distance)

    # Referenzwert: krippendorff.alpha(..., level_of_measurement="nominal")
    assert result.alpha == pytest.approx(0.7659574468085106, abs=1e-9)
    assert result.n_values == pytest.approx(40)
    assert result.degenerate is False


def test_two_coder_small_sample_hand_computed(calc):
    """(A,A) (A,B) (B,B): Do = 1/3, De = 0.6 -> alpha = 4/9."""
    units = [["A", "A"], ["A", "B"], ["B", "B"]]
    result = calc._alpha(units, calc._nominal_distance)

    assert result.observed_disagreement == pytest.approx(1 / 3)
    assert result.expected_disagreement == pytest.approx(0.6)
    assert result.alpha == pytest.approx(4 / 9)


def test_perfect_agreement_is_one(calc):
    units = [["A", "A"], ["B", "B"], ["A", "A"]]
    assert calc._alpha(units, calc._nominal_distance).alpha == pytest.approx(1.0)


def test_systematic_disagreement_is_negative(calc):
    """Spiegelverkehrte Kodierung: alpha muss negativ und darf nicht gekappt werden."""
    units = [["A", "B"], ["B", "A"], ["A", "B"], ["B", "A"]]
    result = calc._alpha(units, calc._nominal_distance)

    assert result.alpha == pytest.approx(-0.75)
    assert result.alpha < 0


def test_negative_alpha_survives_report(calc):
    codings = []
    pattern = ["A", "B", "B", "A"]
    for index, category in enumerate(pattern):
        opposite = "B" if category == "A" else "A"
        codings.append(make_coding(f"seg_{index}", "coder1", category))
        codings.append(make_coding(f"seg_{index}", "coder2", opposite))

    report = calc.calculate_comprehensive_reliability(codings)
    assert report["overall_alpha"] < 0


# ----------------------------------------------------------------------
# Alpha-Eigenschaften, die die alte Berechnung verletzt hat
# ----------------------------------------------------------------------
def test_expected_agreement_follows_marginals(calc):
    """
    Die erwartete Übereinstimmung muss aus den Randverteilungen folgen.
    Zwei Datensätze mit gleicher beobachteter Übereinstimmung, aber
    unterschiedlicher Kategorienverteilung müssen unterschiedliches Alpha
    ergeben (vorher: fix 0.25 für alle Ebenen).
    """
    # 4 Einheiten, 2 Kodierer, je 2 übereinstimmend / 2 nicht
    units_balanced = [["A", "A"], ["B", "B"], ["A", "B"], ["B", "A"]]
    # gleiche Anzahl Übereinstimmungen, aber schiefe Verteilung
    units_skewed = [["A", "A"], ["A", "A"], ["A", "B"], ["B", "A"]]

    alpha_balanced = calc._alpha(units_balanced, calc._nominal_distance)
    alpha_skewed = calc._alpha(units_skewed, calc._nominal_distance)

    assert alpha_balanced.observed_disagreement == pytest.approx(
        alpha_skewed.observed_disagreement
    )
    assert alpha_balanced.expected_disagreement != pytest.approx(
        alpha_skewed.expected_disagreement
    )
    assert alpha_balanced.alpha != pytest.approx(alpha_skewed.alpha)


def test_missing_data_units_are_excluded(calc):
    """Einheiten mit nur einem Kodierer tragen nichts zu Alpha bei."""
    base = [["A", "A"], ["B", "B"], ["A", "A"]]
    alpha_base = calc._alpha(base, calc._nominal_distance).alpha

    codings = []
    for index, value in enumerate(["A", "B", "A"]):
        codings.append(make_coding(f"seg_{index}", "coder1", value))
        codings.append(make_coding(f"seg_{index}", "coder2", value))
    # zusätzliche Einheit, die nur ein Kodierer gesehen hat
    codings.append(make_coding("seg_99", "coder1", "C"))

    calc.verbose = False
    report = calc.calculate_comprehensive_reliability(codings)

    assert report["main_categories_alpha"] == pytest.approx(alpha_base)
    assert report["statistics"]["vergleichbare_segmente"] == 3


def test_multiple_codings_same_coder_are_pooled(calc):
    """Zwei Kodierungen von coder1 und eine von coder2 bilden eine Einheit mit m=3."""
    codings = [
        make_coding("seg_1", "coder1", "A"),
        make_coding("seg_1-1", "coder1", "A"),
        make_coding("seg_1", "coder2", "A"),
        make_coding("seg_2", "coder1", "B"),
        make_coding("seg_2-1", "coder1", "B"),
        make_coding("seg_2", "coder2", "A"),
    ]
    report = calc.calculate_comprehensive_reliability(codings)

    # Hauptkategorien: 2 Einheiten, je m=3, eine übereinstimmend
    assert report["main_categories_alpha"] < 1.0
    assert report["alpha_details"]["main_categories"]["n_units"] == 2


# ----------------------------------------------------------------------
# Set-basierte Ebenen
# ----------------------------------------------------------------------
def test_subcategory_jaccard_partial_agreement(calc):
    """Überlappende Subkategorienmengen ergeben partielle, keine volle Übereinstimmung."""
    # jede Einheit enthält einen Wert je Kodierer
    units_identical = [
        [frozenset({"x", "y"}), frozenset({"x", "y"})],
        [frozenset({"x", "z"}), frozenset({"x", "z"})],
    ]
    units_partial = [
        [frozenset({"x", "y"}), frozenset({"x", "z"})],
        [frozenset({"y"}), frozenset({"y", "z"})],
    ]

    assert calc._jaccard_distance(frozenset({"x", "y"}), frozenset({"x", "y"})) == 0.0
    assert calc._jaccard_distance(frozenset({"x", "y"}), frozenset({"x", "z"})) == pytest.approx(
        (1 - 1 / 3) ** 2
    )
    assert calc._jaccard_distance(frozenset(), frozenset()) == 0.0

    alpha_identical = calc._alpha(units_identical, calc._jaccard_distance)
    assert alpha_identical.alpha == pytest.approx(1.0)

    alpha_partial = calc._alpha(units_partial, calc._jaccard_distance)
    assert 0.0 < alpha_partial.alpha < 1.0


def test_composite_distance_averages_levels(calc):
    d = calc._composite_distance(("A", frozenset({"x"})), ("B", frozenset({"x"})))
    assert d == pytest.approx(0.5)

    d = calc._composite_distance(("A", frozenset({"x"})), ("A", frozenset({"y"})))
    assert d == pytest.approx(0.5)

    d = calc._composite_distance(("A", frozenset({"x"})), ("A", frozenset({"x"})))
    assert d == 0.0


def test_missing_category_is_missing_value_not_empty_set(calc):
    """'Nicht kodiert' darf nicht als leere Kategorie in die Rechnung eingehen."""
    codings = [
        make_coding("seg_1", "coder1", "A", ["x"]),
        make_coding("seg_1", "coder2", "Nicht kodiert"),
        make_coding("seg_2", "coder1", "A", ["x"]),
        make_coding("seg_2", "coder2", "A", ["x"]),
    ]
    report = calc.calculate_comprehensive_reliability(codings)

    assert report["alpha_details"]["main_categories"]["n_units"] == 1
    assert report["main_categories_alpha"] == pytest.approx(1.0)


# ----------------------------------------------------------------------
# Berichtsstruktur / Kategorie-Diagnostik
# ----------------------------------------------------------------------
def test_report_keeps_legacy_keys(calc):
    codings = [
        make_coding("seg_1", "coder1", "A", ["x"]),
        make_coding("seg_1", "coder2", "A", ["y"]),
        make_coding("seg_2", "coder1", "B", ["x"]),
        make_coding("seg_2", "coder2", "B", ["x"]),
    ]
    report = calc.calculate_comprehensive_reliability(codings)

    for key in (
        "overall_alpha",
        "main_categories_alpha",
        "subcategories_alpha",
        "agreement_analysis",
        "statistics",
    ):
        assert key in report

    assert isinstance(report["overall_alpha"], float)
    assert isinstance(report["main_categories_alpha"], float)
    assert isinstance(report["subcategories_alpha"], float)
    assert report["statistics"]["anzahl_kodierer"] == 2
    assert set(report["alpha_details"]) == {
        "overall",
        "main_categories",
        "subcategories",
        "overall_binary",
        "subcategories_binary",
    }


def test_category_level_alphas_identify_weak_category(calc):
    codings = []
    # Kategorie A: perfekt übereinstimmend (4 Einheiten)
    for index in range(4):
        codings.append(make_coding(f"a_{index}", "coder1", "A"))
        codings.append(make_coding(f"a_{index}", "coder2", "A"))
    # Kategorie B: systematisch uneinig (4 Einheiten)
    for index in range(4):
        codings.append(make_coding(f"b_{index}", "coder1", "B"))
        codings.append(make_coding(f"b_{index}", "coder2", "C"))

    report = calc.calculate_comprehensive_reliability(codings)
    per_category = report["category_alphas"]

    assert per_category["A"] == pytest.approx(1.0)
    assert per_category["B"] < per_category["A"]


def test_empty_input_returns_zero_report(calc):
    report = calc.calculate_comprehensive_reliability([])

    assert report["overall_alpha"] == 0.0
    assert report["statistics"]["vergleichbare_segmente"] == 0
    assert report["category_alphas"] == {}


def test_single_coder_yields_no_alpha(calc):
    codings = [
        make_coding("seg_1", "coder1", "A"),
        make_coding("seg_2", "coder1", "B"),
    ]
    report = calc.calculate_comprehensive_reliability(codings)
    assert report["overall_alpha"] == 0.0


# ----------------------------------------------------------------------
# Robustheitsvariante: binäre Hamming-Distanz über das Kategorienuniversum
# ----------------------------------------------------------------------
def test_hamming_variant_hand_computed(calc):
    """
    {'a'} vs {'a'}, {'a','b'} vs {'a'} bei K = 2:
    Do = 0.25, De = 0.25 -> alpha = 0.
    """
    units = [
        [frozenset({"a"}), frozenset({"a"})],
        [frozenset({"a", "b"}), frozenset({"a"})],
    ]
    result = calc._alpha(units, calc._hamming_distance(2))

    assert result.observed_disagreement == pytest.approx(0.25)
    assert result.expected_disagreement == pytest.approx(0.25)
    assert result.alpha == pytest.approx(0.0)


def test_hamming_is_invariant_to_universe_size(calc):
    """Normierung auf K ist kosmetisch: Do und De skalieren gemeinsam."""
    units = [
        [frozenset({"a"}), frozenset({"a"})],
        [frozenset({"a", "b"}), frozenset({"b"})],
        [frozenset({"c"}), frozenset({"c", "d"})],
    ]
    alpha_k4 = calc._alpha(units, calc._hamming_distance(4)).alpha
    alpha_k20 = calc._alpha(units, calc._hamming_distance(20)).alpha

    assert alpha_k4 == pytest.approx(alpha_k20, abs=1e-12)


def test_hamming_perfect_agreement_is_one(calc):
    units = [
        [frozenset({"a", "b"}), frozenset({"a", "b"})],
        [frozenset({"c"}), frozenset({"c"})],
    ]
    assert calc._alpha(units, calc._hamming_distance(3)).alpha == pytest.approx(1.0)


def test_binary_variant_differs_from_jaccard(calc):
    """
    Jaccard bestraft asymmetrische Setgrößen anders als Hamming; die beiden
    Varianten dürfen daher auseinanderfallen - genau das ist die Diagnose.
    """
    codings = [
        make_coding("seg_1", "coder1", "A", ["Strategie", "Infrastruktur"]),
        make_coding("seg_1", "coder2", "A", ["Strategie"]),
        make_coding("seg_2", "coder1", "A", ["Strategie"]),
        make_coding("seg_2", "coder2", "A", ["Strategie"]),
        make_coding("seg_3", "coder1", "A", ["Governance"]),
        make_coding("seg_3", "coder2", "A", ["Strategie"]),
    ]
    report = calc.calculate_comprehensive_reliability(codings)

    assert -1.0 <= report["subcategories_alpha_binary"] <= 1.0
    assert -1.0 <= report["overall_alpha_binary"] <= 1.0
    assert report["subcategories_alpha"] != pytest.approx(
        report["subcategories_alpha_binary"]
    )


def test_binary_keys_in_empty_report(calc):
    report = calc.calculate_comprehensive_reliability([])

    assert report["overall_alpha_binary"] == 0.0
    assert report["subcategories_alpha_binary"] == 0.0

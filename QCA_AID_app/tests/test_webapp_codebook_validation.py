"""
Unit-Tests fuer die mode-aware Codebook-Validierung (CodebookData.validate).

Hintergrund: In den emergenten Modi (``grounded``, ``inductive``) entsteht das
Kategoriensystem waehrend der Analyse aus dem Material (siehe
``QCA_AID_assets/analysis/analysis_manager.py`` -> ``_analyze_normal_modes``:
im induktiven Modus werden ``current_categories``/``deductive_categories``
geleert). Vorab definierte Kategorien sind dort methodisch nicht erforderlich -
verpflichtend ist nur die Forschungsfrage.
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from webapp_models.codebook_data import CategoryData, CodebookData  # noqa: E402


def make_codebook(with_categories: bool = False, with_forschungsfrage: bool = True):
    categories = {}
    if with_categories:
        categories["Kategorie 1"] = CategoryData(
            name="Kategorie 1",
            definition="Definition",
            rules=[],
            examples=["Beleg"],
            subcategories={"Sub 1": "Subdefinition"},
        )
    return CodebookData(
        forschungsfrage="Wie wirkt X?" if with_forschungsfrage else "",
        kodierregeln={"general": [], "format": [], "exclusion": []},
        deduktive_kategorien=categories,
    )


class TestEmergentModesNeedNoCategories:
    def test_inductive_valid_with_forschungsfrage_only(self):
        is_valid, errors = make_codebook().validate(analysis_mode="inductive")
        assert is_valid, errors

    def test_grounded_valid_with_forschungsfrage_only(self):
        is_valid, errors = make_codebook().validate(analysis_mode="grounded")
        assert is_valid, errors

    def test_inductive_still_requires_forschungsfrage(self):
        is_valid, errors = make_codebook(with_forschungsfrage=False).validate(
            analysis_mode="inductive"
        )
        assert not is_valid
        assert any("Research question" in e for e in errors)

    def test_inductive_with_categories_is_valid_too(self):
        # Kategorien duerfen im Codebook stehen - sie werden im induktiven
        # Modus nur nicht als Startpunkt verwendet.
        is_valid, errors = make_codebook(with_categories=True).validate(
            analysis_mode="inductive"
        )
        assert is_valid, errors


class TestCategoryBasedModesStillRequireCategories:
    def test_deductive_requires_categories(self):
        is_valid, errors = make_codebook().validate(analysis_mode="deductive")
        assert not is_valid
        assert any("At least one category" in e for e in errors)

    def test_abductive_requires_categories(self):
        is_valid, errors = make_codebook().validate(analysis_mode="abductive")
        assert not is_valid
        assert any("At least one category" in e for e in errors)

    def test_no_mode_kwarg_keeps_strict_default(self):
        # Rueckwaertskompatibel: ohne Modus bleiben Kategorien Pflicht.
        is_valid, _errors = make_codebook().validate()
        assert not is_valid

    def test_deductive_with_categories_is_valid(self):
        is_valid, errors = make_codebook(with_categories=True).validate(
            analysis_mode="deductive"
        )
        assert is_valid, errors

"""
Reliabilitäts-Berechnungen für QCA-AID
=======================================
Berechnet Inter-Coder-Reliabilität als **Krippendorff's Alpha** über das
Koinzidenzmatrix-Verfahren (Krippendorff 2004, 2011; Hayes & Krippendorff 2007).

Warum Koinzidenzmatrix und nicht paarweise Mittelwerte?
------------------------------------------------------
Alpha ist definiert als

    alpha = 1 - Do / De

mit der beobachteten Unähnlichkeit Do und der aus den Randverteilungen
erwarteten Unähnlichkeit De. Beide Größen werden aus *einer* Koinzidenzmatrix
über alle Einheiten und alle Kodierungen gewonnen. Nur so gilt:

* fehlende Werte sind zulässig (nicht jedes Segment von jedem Kodierer kodiert),
* beliebig viele Kodierer, auch ungleich viele Kodierungen pro Einheit,
* mehrere Kodierungen desselben Kodierers pro Einheit (MULTIPLE_CODINGS),
* eine einzige, konsistente Fehlerrechnung - keine hartkodierte
  Zufallserwartung,
* Alpha darf negativ werden (systematische Nicht-Übereinstimmung) und wird
  nicht auf 0 abgeschnitten.

Distanzmetriken
---------------
* Hauptkategorien: nominal (0 = gleich, 1 = ungleich)
* Subkategorien: Jaccard-Distanz auf den Kategorienmengen,
  d^2 = (1 - |S1 & S2| / |S1 | S2|)^2
* Overall: Mittel der Distanzen beider Ebenen,
  d^2 = (d^2_haupt + d^2_sub) / 2 - also Alpha über die vollständige
  Kodierentscheidung, ohne dass eine Ebene die andere dominiert.

Robustheitsvariante
-------------------
Für mengenwertige Kodierungen gibt es keine kanonische Alpha-Variante; die
Wahl der Mengenmetrik ist eine Modellentscheidung (vgl. MASI, Passonneau 2006,
das ebenfalls Jaccard-basiert ist). Um sie prüfbar zu machen, wird zusätzlich
eine Variante mit **binären Indikatorvektoren** berechnet:

    d = |S1 symmetrische Differenz S2| / K

Das ist die Distanz einer multivariaten binären Variable (Intervall-Fall) über
das Kategorienuniversum K. Die Normierung auf K ist kosmetisch - Do und De
skalieren gemeinsam, Alpha bleibt unverändert; sie hält Do/De nur im Bereich
[0, 1]. Liegen Jaccard- und Hamming-Variante nah beieinander, ist die
Metrikwahl empirisch unkritisch und muss nicht verteidigt werden.

Referenzen
----------
Krippendorff, K. (2004). Content Analysis: An Introduction to Its Methodology.
Krippendorff, K. (2011). Computing Krippendorff's Alpha-Reliability.
Hayes, A. F., & Krippendorff, K. (2007). Answering the call for a standard
    reliability measure for coding data. Communication Methods and Measures, 1(1), 77-89.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import Counter, defaultdict

# Werte, die "keine Kodierung" bedeuten und als fehlender Wert behandelt werden
MISSING_CODES = {"", "nicht kodiert", "kein kodierkonsens", "none", "nan"}

# Schwellen nach Krippendorff (2004, S. 241f.)
THRESHOLD_RELIABLE = 0.800
THRESHOLD_EXPLORATORY = 0.667

# Ab dieser Differenz zwischen Jaccard- und Hamming-Variante ist die Wahl der
# Mengenmetrik inhaltlich relevant und sollte im Methodenteil diskutiert werden.
THRESHOLD_METRIC_ROBUSTNESS = 0.05


class AlphaResult:
    """Ergebnis einer Alpha-Berechnung inkl. Diagnosegrößen."""

    __slots__ = (
        "alpha",
        "observed_disagreement",
        "expected_disagreement",
        "n_units",
        "n_values",
        "n_categories",
        "degenerate",
    )

    def __init__(
        self,
        alpha: Optional[float],
        observed_disagreement: float = 0.0,
        expected_disagreement: float = 0.0,
        n_units: int = 0,
        n_values: int = 0,
        n_categories: int = 0,
        degenerate: bool = False,
    ):
        self.alpha = alpha
        self.observed_disagreement = observed_disagreement
        self.expected_disagreement = expected_disagreement
        self.n_units = n_units
        self.n_values = n_values
        self.n_categories = n_categories
        self.degenerate = degenerate

    @property
    def alpha_or_zero(self) -> float:
        """Alpha als Float; nicht berechenbare Fälle werden als 0.0 gemeldet."""
        return 0.0 if self.alpha is None else self.alpha

    def as_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "observed_disagreement": self.observed_disagreement,
            "expected_disagreement": self.expected_disagreement,
            "n_units": self.n_units,
            "n_values": self.n_values,
            "n_categories": self.n_categories,
            "degenerate": self.degenerate,
        }


class ReliabilityCalculator:
    """
    Berechnet Inter-Coder-Reliabilität für QCA-AID.

    Öffentliche API (stabil gehalten):
        calculate_comprehensive_reliability(codings) -> dict
        calculate_reliability(codings) -> float
    """

    def __init__(self, verbose: bool = True, decimals: int = 3):
        self.verbose = verbose
        self.decimals = decimals
        self._last_results: Dict[str, AlphaResult] = {}

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------
    def calculate_comprehensive_reliability(self, codings: List[Dict]) -> dict:
        """Berechnet alle Reliabilitätskennwerte für einen Kodierlauf."""
        original_codings = self._filter_original_codings(codings)

        if len(original_codings) < 2:
            self._log("Zu wenige ursprüngliche Kodierungen für Reliabilitätsanalyse")
            return self._create_empty_reliability_report()

        statistics = self._calculate_basic_statistics(original_codings)

        overall = self._alpha_overall(original_codings)
        main = self._alpha_main_categories(original_codings)
        subs = self._alpha_subcategories(original_codings)
        # Robustheitsvariante: Indikatorvektoren über das Kategorienuniversum
        # (Hamming-Distanz) statt Jaccard-Distanz auf den Mengen
        overall_binary = self._alpha_overall_binary(original_codings)
        subs_binary = self._alpha_subcategories_binary(original_codings)

        self._last_results = {
            "overall": overall,
            "main": main,
            "sub": subs,
            "overall_binary": overall_binary,
            "sub_binary": subs_binary,
        }

        report = {
            "overall_alpha": overall.alpha_or_zero,
            "main_categories_alpha": main.alpha_or_zero,
            "subcategories_alpha": subs.alpha_or_zero,
            "overall_alpha_binary": overall_binary.alpha_or_zero,
            "subcategories_alpha_binary": subs_binary.alpha_or_zero,
            "agreement_analysis": self._calculate_detailed_agreement_analysis(
                original_codings
            ),
            "statistics": statistics,
            # Neue, diagnostische Felder (abwärtskompatibel ergänzt)
            "alpha_details": {
                "overall": overall.as_dict(),
                "main_categories": main.as_dict(),
                "subcategories": subs.as_dict(),
                "overall_binary": overall_binary.as_dict(),
                "subcategories_binary": subs_binary.as_dict(),
            },
            "category_alphas": self._category_level_alphas(original_codings),
        }

        self._print_reliability_summary(report)
        return report

    def calculate_reliability(self, all_codings: List[Dict]) -> float:
        """Gibt Overall-Alpha als Float zurück (Rückwärtskompatibilität)."""
        report = self.calculate_comprehensive_reliability(all_codings)
        overall_alpha = report["overall_alpha"]
        if isinstance(overall_alpha, dict):  # Altformat-Schutz
            return float(overall_alpha.get("alpha", 0.0) or 0.0)
        return float(overall_alpha)

    # ------------------------------------------------------------------
    # Alpha-Kern: Koinzidenzmatrix
    # ------------------------------------------------------------------
    def _alpha(
        self, units: List[List[Any]], distance: Callable[[Any, Any], float]
    ) -> AlphaResult:
        """
        Krippendorff's Alpha aus einer Liste von Einheiten (je Einheit eine
        Liste von Werten, ein Wert pro Kodierung).

        units: [[v1, v2, ...], ...] - fehlende Werte sind bereits entfernt.
        distance: liefert die *quadrierte* Distanz zweier Werte.
        """
        coincidence: Dict[Tuple[Any, Any], float] = defaultdict(float)
        usable_units = 0

        for unit_values in units:
            m = len(unit_values)
            if m < 2:
                continue
            usable_units += 1
            weight = 1.0 / (m - 1)
            for i, value_i in enumerate(unit_values):
                for j, value_j in enumerate(unit_values):
                    if i == j:
                        continue
                    coincidence[(value_i, value_j)] += weight

        if not coincidence:
            return AlphaResult(alpha=None, n_units=usable_units)

        marginals: Counter = Counter()
        for (value_i, _), count in coincidence.items():
            marginals[value_i] += count
        n = sum(marginals.values())

        if n < 2:
            return AlphaResult(alpha=None, n_units=usable_units, n_values=n)

        do_sum = 0.0
        for (value_i, value_j), count in coincidence.items():
            do_sum += count * distance(value_i, value_j)

        de_sum = 0.0
        values = list(marginals.keys())
        for value_i in values:
            for value_j in values:
                de_sum += (
                    marginals[value_i]
                    * marginals[value_j]
                    * distance(value_i, value_j)
                )

        observed = do_sum / n
        expected = de_sum / (n * (n - 1))

        # De = 0 bedeutet: alle Werte identisch -> perfekte Übereinstimmung
        if expected <= 0.0:
            alpha = 1.0 if observed <= 0.0 else 0.0
            return AlphaResult(
                alpha=alpha,
                observed_disagreement=observed,
                expected_disagreement=0.0,
                n_units=usable_units,
                n_values=n,
                n_categories=len(marginals),
                degenerate=True,
            )

        alpha = 1.0 - (observed / expected)
        # numerische Sicherung, keine inhaltliche Kappung (negative Alpha bleibt negativ)
        alpha = max(-1.0, min(1.0, alpha))

        return AlphaResult(
            alpha=alpha,
            observed_disagreement=observed,
            expected_disagreement=expected,
            n_units=usable_units,
            n_values=n,
            n_categories=len(marginals),
        )

    @staticmethod
    def _nominal_distance(value_i: Any, value_j: Any) -> float:
        return 0.0 if value_i == value_j else 1.0

    @staticmethod
    def _jaccard_distance(set_i: frozenset, set_j: frozenset) -> float:
        if not set_i and not set_j:
            return 0.0  # zwei leere Mengen sind identisch
        union = len(set_i | set_j)
        if union == 0:
            return 0.0
        similarity = len(set_i & set_j) / union
        return (1.0 - similarity) ** 2

    @staticmethod
    def _composite_distance(
        value_i: Tuple[str, frozenset], value_j: Tuple[str, frozenset]
    ) -> float:
        main_i, subs_i = value_i
        main_j, subs_j = value_j
        d_main = ReliabilityCalculator._nominal_distance(main_i, main_j)
        d_subs = ReliabilityCalculator._jaccard_distance(subs_i, subs_j)
        return 0.5 * (d_main + d_subs)

    # ------------------------------------------------------------------
    # Werteextraktion aus Kodierungen
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_codes(raw: Any) -> frozenset:
        """Bringt subcategories in eine hashbare, vergleichbare Menge."""
        if raw is None:
            return frozenset()
        if isinstance(raw, dict):
            candidates = list(raw.keys())
        elif isinstance(raw, str):
            candidates = raw.split(",")
        elif isinstance(raw, (list, tuple, set, frozenset)):
            candidates = list(raw)
        else:
            candidates = [raw]

        cleaned = {
            str(item).strip()
            for item in candidates
            if item is not None and str(item).strip().lower() not in MISSING_CODES
        }
        return frozenset(cleaned)

    @classmethod
    def _main_value(cls, coding: Dict) -> Optional[str]:
        category = coding.get("category")
        if category is None:
            return None
        category = str(category).strip()
        if category.lower() in MISSING_CODES:
            return None
        return category

    @classmethod
    def _sub_value(cls, coding: Dict) -> frozenset:
        return cls._normalize_codes(coding.get("subcategories"))

    @classmethod
    def _combined_value(cls, coding: Dict) -> Optional[Tuple[str, frozenset]]:
        main = cls._main_value(coding)
        subs = cls._sub_value(coding)
        if main is None and not subs:
            return None  # gar keine Kodierung -> fehlender Wert
        return (main or "", subs)

    def _collect_units(
        self, codings: List[Dict], value_fn: Callable[[Dict], Any]
    ) -> List[List[Any]]:
        """
        Baut die Einheiten-Tabelle: je Segment eine Liste von Werten.

        Nur Segmente mit Kodierungen von mindestens zwei *verschiedenen*
        Kodierern gehen ein - Intra-Kodierer-Stabilität ist keine
        Inter-Coder-Reliabilität.
        """
        segment_data = self._group_by_original_segments(codings)
        units: List[List[Any]] = []

        for _, coders_data in segment_data.items():
            if len(coders_data) < 2:
                continue
            values: List[Any] = []
            for _, coder_codings in coders_data.items():
                for coding in coder_codings:
                    value = value_fn(coding)
                    if value is not None:
                        values.append(value)
            if len(values) >= 2:
                units.append(values)

        return units

    # ------------------------------------------------------------------
    # Alpha je Ebene
    # ------------------------------------------------------------------
    def _alpha_overall(self, codings: List[Dict]) -> AlphaResult:
        units = self._collect_units(codings, self._combined_value)
        return self._alpha(units, self._composite_distance)

    def _alpha_main_categories(self, codings: List[Dict]) -> AlphaResult:
        units = self._collect_units(codings, self._main_value)
        return self._alpha(units, self._nominal_distance)

    def _alpha_subcategories(self, codings: List[Dict]) -> AlphaResult:
        # Kodierungen ohne jede Kodierung ausschließen, leere Subkategorie-
        # Mengen aber als gültige Beobachtung behandeln.
        units = self._subcategory_units(codings)
        return self._alpha(units, self._jaccard_distance)

    # ------------------------------------------------------------------
    # Robustheitsvariante: binäre Indikatorvektoren (Hamming)
    # ------------------------------------------------------------------
    def _subcategory_units(self, codings: List[Dict]) -> List[List[frozenset]]:
        """Einheiten-Tabelle für die Subkategorie-Ebene."""

        def value_fn(coding: Dict):
            if self._main_value(coding) is None and not self._sub_value(coding):
                return None
            return self._sub_value(coding)

        return self._collect_units(codings, value_fn)

    @staticmethod
    def _hamming_distance(universe_size: int) -> Callable[[frozenset, frozenset], float]:
        """
        Quadrierte Hamming-Distanz auf binären Indikatorvektoren, normiert auf
        das Kategorienuniversum: d = |S1 Δ S2| / K.

        Das ist die Distanz einer multivariaten binären Variable (Krippendorff
        2004, Intervall-Fall). Die Normierung auf K ist kosmetisch: Da Do und De
        mit demselben Faktor skaliert werden, ist Alpha invariant gegen die Wahl
        von K - sie macht Do/De nur im Bereich [0, 1] interpretierbar.
        """
        size = max(1, universe_size)

        def distance(set_i: frozenset, set_j: frozenset) -> float:
            return len(set_i ^ set_j) / size

        return distance

    def _alpha_subcategories_binary(self, codings: List[Dict]) -> AlphaResult:
        """Alpha der Subkategorien mit Hamming-Distanz (binäre Indikatorvektoren)."""
        units = self._subcategory_units(codings)
        universe = set()
        for unit in units:
            for value in unit:
                universe.update(value)
        return self._alpha(units, self._hamming_distance(len(universe)))

    def _alpha_overall_binary(self, codings: List[Dict]) -> AlphaResult:
        """Alpha der vollständigen Kodierentscheidung mit Hamming-Distanz."""
        units = self._collect_units(codings, self._combined_value)
        universe = set()
        for unit in units:
            for _, subs in unit:
                universe.update(subs)
        sub_distance = self._hamming_distance(len(universe))

        def distance(value_i, value_j) -> float:
            return 0.5 * (
                self._nominal_distance(value_i[0], value_j[0])
                + sub_distance(value_i[1], value_j[1])
            )

        return self._alpha(units, distance)

    def _category_level_alphas(self, codings: List[Dict]) -> Dict[str, float]:
        """
        Alpha je Hauptkategorie als binäre Variable (vorhanden / nicht vorhanden).

        Nützlich, um einzelne Problemkategorien zu identifizieren, die den
        Gesamt-Alpha drücken.
        """
        segment_data = self._group_by_original_segments(codings)
        all_categories = set()
        for coders_data in segment_data.values():
            if len(coders_data) < 2:
                continue
            for coder_codings in coders_data.values():
                for coding in coder_codings:
                    main = self._main_value(coding)
                    if main is not None:
                        all_categories.add(main)

        results: Dict[str, float] = {}
        for category in sorted(all_categories):
            units: List[List[Any]] = []
            for _, coders_data in segment_data.items():
                if len(coders_data) < 2:
                    continue
                values = []
                for _, coder_codings in coders_data.items():
                    coder_values = []
                    for coding in coder_codings:
                        main = self._main_value(coding)
                        if main is None:
                            continue
                        coder_values.append(1 if main == category else 0)
                    # ein Kodierer trägt höchstens einen Wert bei: Mehrfach-
                    # kodierungen werden gemittelt (Mehrheitsentscheid)
                    if coder_values:
                        values.append(round(sum(coder_values) / len(coder_values)))
                if len(values) >= 2:
                    units.append(values)

            result = self._alpha(units, self._nominal_distance)
            if result.alpha is not None and not result.degenerate:
                results[category] = result.alpha

        return results

    # ------------------------------------------------------------------
    # Abwärtskompatible Einzelmethoden
    # ------------------------------------------------------------------
    def _calculate_multiple_coding_alpha(self, codings: List[Dict]) -> float:
        """Deprecated: Overall-Alpha (Alias auf _alpha_overall)."""
        return self._alpha_overall(codings).alpha_or_zero

    def _calculate_main_categories_alpha(self, codings: List[Dict]) -> float:
        """Deprecated: Alpha der Hauptkategorien."""
        return self._alpha_main_categories(codings).alpha_or_zero

    def _calculate_subcategories_alpha(self, codings: List[Dict]) -> float:
        """Deprecated: Alpha der Subkategorien."""
        return self._alpha_subcategories(codings).alpha_or_zero

    def _calculate_combined_sets_alpha(self, codings: List[Dict]) -> float:
        """Deprecated: Overall-Alpha (Alias)."""
        return self._alpha_overall(codings).alpha_or_zero

    # ------------------------------------------------------------------
    # Deskriptive Übereinstimmungsanalyse
    # ------------------------------------------------------------------
    def _calculate_detailed_agreement_analysis(self, codings: List[Dict]) -> dict:
        """Zählt paarweise Übereinstimmungsmuster (deskriptiv, nicht Alpha)."""
        segment_data = self._group_by_original_segments(codings)

        agreement_stats = {
            "Vollständige Übereinstimmung": 0,
            "Hauptkategorie gleich, Subkat. unterschiedlich": 0,
            "Hauptkategorie unterschiedlich": 0,
        }

        for _, coder_data in segment_data.items():
            all_segment_codings = []
            for _, coder_codings in coder_data.items():
                all_segment_codings.extend(coder_codings)

            if len(all_segment_codings) < 2:
                continue

            for i in range(len(all_segment_codings)):
                for j in range(i + 1, len(all_segment_codings)):
                    coding1 = all_segment_codings[i]
                    coding2 = all_segment_codings[j]

                    main_cat1 = coding1.get("category", "")
                    main_cat2 = coding2.get("category", "")
                    subcats1 = self._sub_value(coding1)
                    subcats2 = self._sub_value(coding2)

                    if main_cat1 == main_cat2 and subcats1 == subcats2:
                        agreement_stats["Vollständige Übereinstimmung"] += 1
                    elif main_cat1 == main_cat2:
                        agreement_stats["Hauptkategorie gleich, Subkat. unterschiedlich"] += 1
                    else:
                        agreement_stats["Hauptkategorie unterschiedlich"] += 1

        agreement_stats["Gesamt"] = sum(
            agreement_stats[key]
            for key in (
                "Vollständige Übereinstimmung",
                "Hauptkategorie gleich, Subkat. unterschiedlich",
                "Hauptkategorie unterschiedlich",
            )
        )
        return agreement_stats

    # ------------------------------------------------------------------
    # Gruppierung / Filter / Statistik
    # ------------------------------------------------------------------
    def _filter_original_codings(self, codings: List[Dict]) -> List[Dict]:
        """Entfernt Konsens-/Review-Artefakte, behält die ursprünglichen Kodierungen."""
        original_codings = []
        for coding in codings:
            coder_id = coding.get("coder_id", "")
            consensus_info = coding.get("consensus_info", {}) or {}
            manual_review = coding.get("manual_review", False)
            selection_type = consensus_info.get("selection_type", "")

            is_excluded = (
                coder_id in ["consensus", "majority", "review"]
                or manual_review is True
                or selection_type in ["consensus", "majority", "manual_consensus"]
            )
            if not is_excluded:
                original_codings.append(coding)

        if len(original_codings) < 2:
            self._log("Zu wenige Kodierungen nach strengem Filter - verwende gelockerten Filter")
            original_codings = [
                coding
                for coding in codings
                if coding.get("coder_id", "") not in ["consensus", "majority", "review"]
            ]

        self._log(f"Gefilterte ursprüngliche Kodierungen: {len(original_codings)}")
        return original_codings

    def _extract_base_segment_id(self, coding: Dict) -> str:
        """Entfernt Mehrfachkodierungs-Suffixe (z.B. 'seg_5-1' -> 'seg_5')."""
        segment_id = coding.get("segment_id", "")

        if "-" in segment_id:
            parts = segment_id.rsplit("-", 1)
            if len(parts) == 2 and parts[1].isdigit():
                return parts[0]

        return segment_id

    def _extract_original_segment_id(self, coding: Dict) -> str:
        """Bevorzugt die im consensus_info hinterlegte Original-ID."""
        consensus_info = coding.get("consensus_info", {}) or {}
        if consensus_info.get("original_segment_id"):
            return consensus_info["original_segment_id"]
        return self._extract_base_segment_id(coding)

    def _group_by_original_segments(self, codings: List[Dict]) -> dict:
        """
        Gruppiert Kodierungen nach Basis-Segment und Kodierer.

        Rückgabe: {base_segment_id: {coder_id: [codings]}}
        """
        segment_data: Dict[str, Dict[str, List[Dict]]] = {}

        for coding in codings:
            original_id = self._extract_original_segment_id(coding)
            coder_id = coding.get("coder_id", "unknown")

            segment_data.setdefault(original_id, {}).setdefault(coder_id, []).append(
                coding
            )

        return segment_data

    def _calculate_basic_statistics(self, codings: List[Dict]) -> dict:
        """Basis-Statistiken für den Bericht."""
        if not codings:
            return {
                "total_codings": 0,
                "vergleichbare_segmente": 0,
                "total_segmente": 0,
                "anzahl_kodierer": 0,
                "mittelwert_kodierungen": 0.0,
            }

        segment_data = self._group_by_original_segments(codings)
        vergleichbare_segmente = sum(
            1 for data in segment_data.values() if len(data) >= 2
        )

        coders = {
            coding.get("coder_id", "unknown")
            for coding in codings
            if coding.get("coder_id")
        }
        anzahl_kodierer = len(coders) if coders else 1

        return {
            "total_codings": len(codings),
            "vergleichbare_segmente": vergleichbare_segmente,
            "total_segmente": len(segment_data),
            "anzahl_kodierer": anzahl_kodierer,
            "mittelwert_kodierungen": len(codings) / anzahl_kodierer,
        }

    def _create_empty_reliability_report(self) -> dict:
        """Vollständiger Leerbericht (gleiche Struktur wie der normale Bericht)."""
        empty_stats = {
            "total_codings": 0,
            "vergleichbare_segmente": 0,
            "total_segmente": 0,
            "anzahl_kodierer": 0,
            "mittelwert_kodierungen": 0.0,
        }
        return {
            "overall_alpha": 0.0,
            "main_categories_alpha": 0.0,
            "subcategories_alpha": 0.0,
            "overall_alpha_binary": 0.0,
            "subcategories_alpha_binary": 0.0,
            "agreement_analysis": {
                "Vollständige Übereinstimmung": 0,
                "Hauptkategorie gleich, Subkat. unterschiedlich": 0,
                "Hauptkategorie unterschiedlich": 0,
                "Gesamt": 0,
            },
            "statistics": empty_stats,
            "alpha_details": {
                "overall": AlphaResult(None).as_dict(),
                "main_categories": AlphaResult(None).as_dict(),
                "subcategories": AlphaResult(None).as_dict(),
                "overall_binary": AlphaResult(None).as_dict(),
                "subcategories_binary": AlphaResult(None).as_dict(),
            },
            "category_alphas": {},
        }

    # ------------------------------------------------------------------
    # Ausgabe
    # ------------------------------------------------------------------
    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _fmt(self, value: Optional[float]) -> str:
        if value is None:
            return "n. b."
        return f"{value:.{self.decimals}f}"

    def _print_reliability_summary(self, report: dict):
        """Kompakte Zusammenfassung inkl. der Größen, die Alpha ausmachen."""
        overall = report["alpha_details"]["overall"]
        main = report["alpha_details"]["main_categories"]
        sub = report["alpha_details"]["subcategories"]
        overall_binary = report["alpha_details"].get("overall_binary") or {}
        sub_binary = report["alpha_details"].get("subcategories_binary") or {}
        stats = report["statistics"]

        self._log("\nKrippendorff's Alpha - Reliabilitäts-Analyse")
        self._log("=" * 60)
        self._log(f"Overall (Haupt + Sub):     {self._fmt(overall['alpha'])}   [Jaccard]")
        self._log(f"Overall (binär):           {self._fmt(overall_binary.get('alpha'))}   [Hamming]")
        self._log(f"Hauptkategorien:           {self._fmt(main['alpha'])}")
        self._log(f"Subkategorien:             {self._fmt(sub['alpha'])}   [Jaccard]")
        self._log(f"Subkategorien (binär):     {self._fmt(sub_binary.get('alpha'))}   [Hamming]")
        self._log(f"Vergleichbare Segmente:    {stats['vergleichbare_segmente']}")
        self._log(f"Kodierer:                  {stats['anzahl_kodierer']}")
        self._log(f"Parierbare Werte:          {int(round(overall['n_values']))}")
        self._log(
            f"Do / De (Overall):         "
            f"{self._fmt(overall['observed_disagreement'])} / "
            f"{self._fmt(overall['expected_disagreement'])}"
        )

        self._log("\nMethodik:")
        self._log("   - Koinzidenzmatrix nach Krippendorff (2011), fehlende Werte zulässig")
        self._log("   - Hauptkategorien: nominale Distanz")
        self._log("   - Subkategorien: Jaccard-Distanz auf Kategorienmengen")
        self._log("   - Subkategorien (binär): Hamming-Distanz auf Indikatorvektoren")
        self._log("   - Overall: Mittel der Distanzen beider Ebenen")

        # Metrik-Robustheit: Jaccard vs. Hamming auf derselben Ebene
        self._log_metric_robustness(
            ("Subkategorien", sub, sub_binary),
            ("Overall", overall, overall_binary),
        )

        # Diagnose: degenerierte oder nicht berechenbare Fälle
        for label, detail in (
            ("Overall", overall),
            ("Hauptkategorien", main),
            ("Subkategorien", sub),
        ):
            if detail["alpha"] is None:
                self._log(
                    f"   ! {label}: Alpha nicht berechenbar "
                    f"(zu wenige Einheiten oder Werte)"
                )
            elif detail["degenerate"]:
                self._log(
                    f"   ! {label}: keine Varianz in den Randverteilungen "
                    f"- Alpha ist hier nicht interpretierbar"
                )

        # Hinweis: Overall muss nicht zwischen den Ebenen liegen
        if all(
            detail["alpha"] is not None and not detail["degenerate"]
            for detail in (overall, main, sub)
        ):
            low, high = sorted([main["alpha"], sub["alpha"]])
            if not (low <= overall["alpha"] <= high):
                self._log(
                    "   i Overall liegt außerhalb des Bereichs der Einzelebenen - "
                    "das ist bei unterschiedlichen De-Werten normal, kein Rechenfehler."
                )

        if stats["vergleichbare_segmente"] < 10:
            self._log(
                f"   ! Nur {stats['vergleichbare_segmente']} vergleichbare Segmente: "
                f"Alpha ist auf so wenigen Einheiten statistisch nicht belastbar."
            )

        self._log("\nBewertung Overall Alpha:   " + self._rating(overall["alpha"]))

        category_alphas = report.get("category_alphas") or {}
        if category_alphas:
            weak = sorted(
                (
                    (name, value)
                    for name, value in category_alphas.items()
                    if value < THRESHOLD_EXPLORATORY
                ),
                key=lambda item: item[1],
            )
            if weak:
                self._log(
                    f"\nKategorien unter der Explorationsschwelle "
                    f"({len(weak)} von {len(category_alphas)}, binäres Alpha):"
                )
                for name, value in weak[:5]:
                    self._log(f"   - {name}: {self._fmt(value)}")
                if len(weak) > 5:
                    self._log(f"   ... und {len(weak) - 5} weitere")
            else:
                self._log(
                    f"\nAlle {len(category_alphas)} Hauptkategorien liegen "
                    f"im binären Alpha über {THRESHOLD_EXPLORATORY:.3f}."
                )

    def _log_metric_robustness(self, *pairs) -> None:
        """
        Vergleicht Jaccard- und Hamming-Variante je Ebene.

        Kleine Abweichung = die Wahl der Mengenmetrik ändert das Ergebnis nicht;
        große Abweichung = die Übereinstimmung hängt an der Metrik und muss
        begründet werden.
        """
        deltas = []
        for label, jaccard_detail, hamming_detail in pairs:
            if not jaccard_detail or not hamming_detail:
                continue
            alpha_jaccard = jaccard_detail.get("alpha")
            alpha_hamming = hamming_detail.get("alpha")
            if alpha_jaccard is None or alpha_hamming is None:
                continue
            if jaccard_detail.get("degenerate") or hamming_detail.get("degenerate"):
                continue
            deltas.append((label, abs(alpha_jaccard - alpha_hamming)))

        if not deltas:
            return

        worst_label, worst_delta = max(deltas, key=lambda item: item[1])
        if worst_delta <= THRESHOLD_METRIC_ROBUSTNESS:
            self._log(
                f"\nMetrik-Robustheit: max. Abweichung {worst_delta:.3f} "
                f"({worst_label}) - Jaccard und Hamming stimmen praktisch überein."
            )
        else:
            self._log(
                f"\n! Metrik-Robustheit: Abweichung {worst_delta:.3f} ({worst_label}) - "
                f"Jaccard und Hamming fallen auseinander; die Übereinstimmung hängt "
                f"an der Metrik und sollte im Methodenteil begründet werden."
            )

    @staticmethod
    def _rating(alpha: Optional[float]) -> str:
        if alpha is None:
            return "nicht berechenbar"
        if alpha >= THRESHOLD_RELIABLE:
            return "reliabel (>= .800)"
        if alpha >= THRESHOLD_EXPLORATORY:
            return "nur explorativ belastbar (.667 - .800)"
        if alpha > 0.0:
            return "unzureichend (< .667)"
        return "systematische Nicht-Übereinstimmung (<= 0)"

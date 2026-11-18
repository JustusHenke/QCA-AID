"""Regressionstests fuer die Batch-/Phasen-Callbacks des OptimizationControllers.

Hintergrund (Bug in 0.13.5): ``_analyze_inductive()`` benutzte den Namen
``batch_callback``, ohne ihn als Parameter zu deklarieren. Der Name kam per
``**kwargs`` aus ``analyze_segments()`` herein, wurde aber nie gebunden ->
``NameError: name 'batch_callback' is not defined`` bei jedem induktiven Lauf,
gefolgt von einem stillen Fallback auf die Standard-Analyse.
"""

import asyncio
import inspect
import types

from QCA_AID_assets.optimization.controller import AnalysisMode, OptimizationController


class _StubAnalyzer:
    """Minimaler UnifiedAnalyzer-Ersatz: findet keine relevanten Segmente."""

    def __init__(self):
        self.calls = []

    async def analyze_relevance_simple(self, **kwargs):
        self.calls.append(kwargs)
        return []


def _fake_controller(analyzer=None):
    return types.SimpleNamespace(
        config={AnalysisMode.INDUCTIVE: {"batch_size": 10}},
        unified_analyzer=analyzer or _StubAnalyzer(),
    )


def test_analyze_inductive_declares_batch_callback():
    """Der Parameter muss in der Signatur stehen (Kern des 0.13.5-Bugs)."""
    sig = inspect.signature(OptimizationController._analyze_inductive)
    assert "batch_callback" in sig.parameters
    assert sig.parameters["batch_callback"].default is None

    # _analyze_grounded hatte den Parameter schon immer - bleibt Referenz.
    grounded_sig = inspect.signature(OptimizationController._analyze_grounded)
    assert "batch_callback" in grounded_sig.parameters


def test_analyze_inductive_reports_phase_without_name_error():
    """Relevanzpruefung muss den Callback aufrufen und ohne API-Calls aussteigen."""
    phases = []

    def _callback(batch_num, total_batches, coder_id="", phase=""):
        phases.append(phase)

    analyzer = _StubAnalyzer()

    async def _run():
        return await OptimizationController._analyze_inductive(
            _fake_controller(analyzer),
            segments=[{"segment_id": "seg_1", "text": "Beispieltext"}],
            research_question="Wie werden Ziele formuliert?",
            batch_callback=_callback,
        )

    results = asyncio.run(_run())

    assert results == []
    assert phases == ["Phase 1: Relevanzprüfung"]
    # Abbruch vor dem ersten Kodier-Batch: genau ein Relevanz-Call
    assert len(analyzer.calls) == 1


def test_analyze_inductive_works_without_callback():
    """Ohne Callback darf die Analyse weiterhin laufen (Default None)."""

    async def _run():
        return await OptimizationController._analyze_inductive(
            _fake_controller(),
            segments=[{"segment_id": "seg_1", "text": "Beispieltext"}],
            research_question="Wie werden Ziele formuliert?",
        )

    assert asyncio.run(_run()) == []

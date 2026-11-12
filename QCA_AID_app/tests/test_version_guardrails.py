"""Guardrail-Tests: Version und Changelog müssen synchron sein.

Stellt sicher, dass ein Versionsprung alle Stellen erreicht:
  1. pyproject.toml ([project] version)
  2. QCA_AID_assets/__version__.py (__version__ + __version_date__)
  3. CHANGELOG.md (neuester '## Neu in <version> (<datum>)'-Eintrag)

Außerdem: Changelog-Einträge streng fallend, keine Duplikate.
Die Prüflogik lebt in scripts/version_guardrail.py und wird hier gegen
das echte Repo UND gegen synthetische Fixtures getestet.
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARDRAIL_SCRIPT = REPO_ROOT / "scripts" / "version_guardrail.py"

_spec = importlib.util.spec_from_file_location("version_guardrail", GUARDRAIL_SCRIPT)
guardrail = importlib.util.module_from_spec(_spec)
sys.modules["version_guardrail"] = guardrail
_spec.loader.exec_module(guardrail)


def _fixture(pyproject=None, version_module=None, changelog=None):
    """Standard-Fixture: konsistenter Zustand mit Version 1.2.3."""
    if pyproject is None:
        pyproject = '[project]\nname = "qca-aid"\nversion = "1.2.3"\n'
    if version_module is None:
        version_module = (
            '__version__ = "1.2.3"\n'
            '__version_date__ = "2026-07-24"\n'
        )
    if changelog is None:
        changelog = (
            "# Changelog\n"
            "\n"
            "## Versionen und Updates\n"
            "\n"
            "---\n"
            "\n"
            "## Neu in 1.2.3 (2026-07-24)\n"
            "\n"
            "### Feature X\n"
            "\n"
            "## Neu in 1.2.2 (2026-07-01)\n"
            "\n"
            "### Feature Y\n"
        )
    return pyproject, version_module, changelog


class TestRealRepo:
    """Das echte Repo muss konsistent sein."""

    def test_check_returns_no_errors(self):
        errors = guardrail.check(
            guardrail._read(guardrail.PYPROJECT_PATH),
            guardrail._read(guardrail.VERSION_MODULE_PATH),
            guardrail._read(guardrail.CHANGELOG_PATH),
        )
        assert errors == []

    def test_all_sources_match(self):
        py = guardrail.get_pyproject_version(
            guardrail._read(guardrail.PYPROJECT_PATH))
        mod, _ = guardrail.get_version_module(
            guardrail._read(guardrail.VERSION_MODULE_PATH))
        entries = guardrail.get_changelog_entries(
            guardrail._read(guardrail.CHANGELOG_PATH))
        assert py == mod
        assert entries[0][0] == py

    def test_changelog_entries_strictly_descending(self):
        entries = guardrail.get_changelog_entries(
            guardrail._read(guardrail.CHANGELOG_PATH))
        keys = [guardrail.parse_version(v) for v, _ in entries]
        assert all(k is not None for k in keys)
        assert keys == sorted(keys, reverse=True)
        assert len(set(keys)) == len(keys)  # keine Duplikate


class TestCheckLogic:
    """Die Prüflogik erkennt jede Art von Inkonsistenz."""

    def _check(self, py=None, mod=None, chlog=None):
        p, m, c = _fixture(pyproject=py, version_module=mod, changelog=chlog)
        return guardrail.check(p, m, c)

    def test_consistent_state_passes(self):
        assert self._check() == []

    def test_missing_changelog_entry_detected(self):
        # pyproject ist 1.2.3, neuester Changelog-Eintrag aber 1.2.2
        errors = self._check(
            chlog="## Neu in 1.2.2 (2026-07-01)\n"
        )
        assert any("CHANGELOG.md" in e and "1.2.2" in e for e in errors)

    def test_pyproject_mismatch_detected(self):
        errors = self._check(py='version = "1.3.0"\n')
        assert any("Versions-Mismatch" in e for e in errors)

    def test_version_date_mismatch_detected(self):
        errors = self._check(mod='__version__ = "1.2.3"\n__version_date__ = "2026-01-01"\n')
        assert any("Datums-Mismatch" in e for e in errors)

    def test_duplicate_version_detected(self):
        chlog = ("## Neu in 1.2.3 (2026-07-24)\n\n"
                 "## Neu in 1.2.3 (2026-07-01)\n")
        errors = self._check(chlog=chlog)
        assert any("doppelt" in e for e in errors)

    def test_ascending_order_detected(self):
        chlog = ("## Neu in 1.2.2 (2026-07-01)\n\n"
                 "## Neu in 1.2.3 (2026-07-24)\n")
        errors = self._check(chlog=chlog)
        assert any("fallend" in e for e in errors)

    def test_invalid_version_format_detected(self):
        errors = self._check(py='version = "abc"\n')
        assert any("gültiges Format" in e for e in errors)

    def test_empty_changelog_detected(self):
        errors = self._check(chlog="# Changelog\n")
        assert any("kein Eintrag" in e for e in errors)


class TestBumpLogic:
    """Der Versionsprung aktualisiert alle Stellen synchron oder gar nicht."""

    def _bump(self, new_version, py=None, mod=None, chlog=None, date_="2026-11-12"):
        p, m, c = _fixture(pyproject=py, version_module=mod, changelog=chlog)
        return guardrail.bump(new_version, "Test-Feature", p, m, c, entry_date=date_)

    def test_bump_updates_all_three_sources(self):
        p, m, c, errors = self._bump("1.3.0")
        assert errors == []
        assert 'version = "1.3.0"' in p
        assert '__version__ = "1.3.0"' in m
        assert '__version_date__ = "2026-11-12"' in m
        assert "## Neu in 1.3.0 (2026-11-12)" in c
        assert "### Test-Feature" in c

    def test_bump_refuses_lower_version(self):
        _, _, _, errors = self._bump("1.2.2")
        assert any("größer" in e for e in errors)

    def test_bump_refuses_equal_version(self):
        _, _, _, errors = self._bump("1.2.3")
        assert errors != []

    def test_bump_refuses_invalid_format(self):
        _, _, _, errors = self._bump("nonsense")
        assert any("gültiges Format" in e for e in errors)

    def test_bump_refuses_inconsistent_state(self):
        # Neuester Changelog-Eintrag gehört nicht zur aktuellen Version
        _, _, _, errors = self._bump(
            "1.3.0", chlog="## Neu in 1.2.0 (2026-06-01)\n")
        assert errors != []

    def test_bump_inserts_entry_before_existing(self):
        _, _, c, errors = self._bump("1.3.0")
        assert errors == []
        pos_new = c.index("## Neu in 1.3.0")
        pos_old = c.index("## Neu in 1.2.3")
        assert pos_new < pos_old

    def test_bump_matches_file_line_endings(self):
        # LF-Fixture → neuer Eintrag mit LF
        _, _, c, errors = self._bump("1.3.0")
        assert errors == []
        assert "\n\n## Neu in 1.3.0" in c
        # CRLF-Fixture → neuer Eintrag mit CRLF (reales Repo ist CRLF)
        p, m, c = _fixture()
        c_crlf = c.replace("\n", "\r\n")
        p_new, m_new, c_new, errors = guardrail.bump(
            "1.3.0", "Test-Feature", p, m, c_crlf, entry_date="2026-11-12")
        assert errors == []
        assert "\r\n\r\n## Neu in 1.3.0" in c_new
        assert guardrail.check(p_new, m_new, c_new) == []

    def test_bumped_state_passes_check(self):
        p, m, c, errors = self._bump("1.3.0")
        assert errors == []
        assert guardrail.check(p, m, c) == []

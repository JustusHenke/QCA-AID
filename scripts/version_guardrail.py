#!/usr/bin/env python3
"""Version- und Changelog-Guardrail für QCA-AID.

Stellt sicher, dass Versionsprünge konsistent über alle Stellen propagiert
werden und der Changelog nicht vergessen wird:

1. ``pyproject.toml``  → ``[project] version``
2. ``QCA_AID_assets/__version__.py`` → ``__version__`` + ``__version_date__``
3. ``CHANGELOG.md``    → neuester Eintrag ``## Neu in <version> (<datum>)``

Zusätzlich wird geprüft, dass die Changelog-Einträge streng fallend sortiert
sind und keine Version doppelt vorkommt.

Verwendung::

    python scripts/version_guardrail.py check
    python scripts/version_guardrail.py bump 0.13.2 --summary "Kurzbeschreibung"

``check`` beendet sich mit Exit-Code 1, wenn Inkonsistenzen gefunden wurden
(geeignet für CI oder als Vor-Commit-Prüfung).
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
VERSION_MODULE_PATH = REPO_ROOT / "QCA_AID_assets" / "__version__.py"
CHANGELOG_PATH = REPO_ROOT / "CHANGELOG.md"

CHANGELOG_HEADER = "## Versionen und Updates"

# "## Neu in 0.13.1 (2026-07-24)"
# "\r" tolerieren: CRLF-Dateien haben vor dem \n ein \r, das $ nicht schluckt
CHANGELOG_ENTRY_RE = re.compile(
    r"(?m)^##\s+Neu in\s+(\S+)\s+\((\d{4}-\d{2}-\d{2})\)[ \t]*\r?$"
)
PYPROJECT_VERSION_RE = re.compile(r'(?m)^version\s*=\s*"([^"]+)"')
VERSION_RE = re.compile(r'(?m)^__version__\s*=\s*"([^"]+)"')
VERSION_DATE_RE = re.compile(r'(?m)^__version_date__\s*=\s*"([^"]+)"')


def _read(path: Path) -> str:
    """Datei als UTF-8 lesen, Zeilenenden unverändert lassen (CRLF-sicher)."""
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return fh.read()


def _write(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def parse_version(v: str) -> Optional[Tuple[int, ...]]:
    """Version in vergleichbare Tuple zerlegen ('0.13.1.2' → (0,13,1,2))."""
    try:
        return tuple(int(part) for part in v.strip().split("."))
    except ValueError:
        return None


def get_pyproject_version(text: str) -> Optional[str]:
    match = PYPROJECT_VERSION_RE.search(text)
    return match.group(1) if match else None


def get_version_module(text: str) -> Tuple[Optional[str], Optional[str]]:
    v_match = VERSION_RE.search(text)
    d_match = VERSION_DATE_RE.search(text)
    return (
        v_match.group(1) if v_match else None,
        d_match.group(1) if d_match else None,
    )


def get_changelog_entries(text: str) -> List[Tuple[str, str]]:
    """Alle (version, datum)-Paare in Dateireihenfolge (neueste zuerst)."""
    return CHANGELOG_ENTRY_RE.findall(text)


def check(
    pyproject_text: str,
    version_module_text: str,
    changelog_text: str,
) -> List[str]:
    """Führt alle Konsistenzprüfungen aus. Rückgabe: Liste von Fehlern."""
    errors: List[str] = []

    py_version = get_pyproject_version(pyproject_text)
    mod_version, mod_date = get_version_module(version_module_text)
    entries = get_changelog_entries(changelog_text)

    if py_version is None:
        errors.append("pyproject.toml: keine `version = \"...\"` gefunden")
    if mod_version is None:
        errors.append("__version__.py: kein `__version__ = \"...\"` gefunden")
    if mod_date is None:
        errors.append("__version__.py: kein `__version_date__ = \"...\"` gefunden")

    # Versionen müssen formal parsebar sein
    for label, v in (("pyproject.toml", py_version), ("__version__.py", mod_version)):
        if v is not None and parse_version(v) is None:
            errors.append(f"{label}: Version '{v}' ist kein gültiges Format (x.y.z[.w])")

    if errors:
        return errors

    # 1) pyproject.toml == __version__.py
    if py_version != mod_version:
        errors.append(
            f"Versions-Mismatch: pyproject.toml='{py_version}' "
            f"vs. __version__.py='{mod_version}'"
        )

    # 2) Changelog überhaupt vorhanden?
    if not entries:
        errors.append("CHANGELOG.md: kein Eintrag '## Neu in <version> (<datum>)' gefunden")
        return errors

    latest_version, latest_date = entries[0]

    # 3) Neuester Changelog-Eintrag == aktuelle Version
    if latest_version != py_version:
        errors.append(
            f"CHANGELOG.md: neuester Eintrag ist {latest_version}, "
            f"aber aktuelle Version ist {py_version} — Changelog-Eintrag fehlt?"
        )

    # 4) Changelog-Datum == __version_date__
    if latest_version == py_version and latest_date != mod_date:
        errors.append(
            f"Datums-Mismatch: __version__.py='{mod_date}' "
            f"vs. CHANGELOG.md-Eintrag='{latest_date}'"
        )

    # 5) Changelog-Einträge streng fallend, keine Duplikate
    parsed: List[Tuple[Tuple[int, ...], str]] = []
    for entry_version, entry_date in entries:
        key = parse_version(entry_version)
        if key is None:
            errors.append(
                f"CHANGELOG.md: Eintrag '{entry_version}' ist kein gültiges Versionsformat"
            )
            continue
        parsed.append((key, entry_version))

    seen = set()
    for key, entry_version in parsed:
        if entry_version in seen:
            errors.append(f"CHANGELOG.md: Version {entry_version} ist doppelt eingetragen")
        seen.add(entry_version)

    for (key_a, ver_a), (key_b, ver_b) in zip(parsed, parsed[1:]):
        if key_a <= key_b:
            errors.append(
                f"CHANGELOG.md: Einträge nicht streng fallend sortiert "
                f"('{ver_a}' erscheint vor '{ver_b}')"
            )

    return errors


def _build_changelog_entry(new_version: str, entry_date: str, summary: str,
                           eol: str = "\r\n") -> str:
    lines = [
        f"## Neu in {new_version} ({entry_date})",
        "",
        f"### {summary}",
        "",
        "#### Details",
        "- (Hier Änderungen ergänzen)",
        "",
        "### 🧹 Sonstiges",
        "",
        "",
    ]
    return eol.join(lines)


def bump(
    new_version: str,
    summary: str,
    pyproject_text: str,
    version_module_text: str,
    changelog_text: str,
    entry_date: Optional[str] = None,
) -> Tuple[str, str, str, List[str]]:
    """Versionsprung über alle drei Stellen synchron ausführen.

    Rückgabe: (neues_pyproject, neues_version_module, neues_changelog, fehler).
    Bei Fehlern in ``fehler`` werden die Rückgabetexte unverändert übernommen.
    """
    errors = check(pyproject_text, version_module_text, changelog_text)
    if errors:
        return pyproject_text, version_module_text, changelog_text, errors

    current = get_pyproject_version(pyproject_text)
    _, mod_date = get_version_module(version_module_text)

    new_key = parse_version(new_version)
    cur_key = parse_version(current or "")
    if new_key is None:
        return (pyproject_text, version_module_text, changelog_text,
                [f"Neue Version '{new_version}' ist kein gültiges Format (x.y.z[.w])"])
    if new_key <= cur_key:  # type: ignore[operator]
        return (pyproject_text, version_module_text, changelog_text,
                [f"Neue Version {new_version} muss größer als {current} sein"])

    entry_date = entry_date or date.today().isoformat()

    # pyproject.toml
    new_pyproject = PYPROJECT_VERSION_RE.sub(
        f'version = "{new_version}"', pyproject_text, count=1
    )

    # __version__.py
    new_module = VERSION_RE.sub(f'__version__ = "{new_version}"', version_module_text, count=1)
    new_module = VERSION_DATE_RE.sub(
        f'__version_date__ = "{entry_date}"', new_module, count=1
    )

    # CHANGELOG.md: neuen Eintrag vor dem ersten bestehenden Eintrag einfügen
    # Zeilenende der Zieldatei übernehmen (CRLF/LFR-mischung vermeiden)
    eol = "\r\n" if "\r\n" in changelog_text else "\n"
    entry = _build_changelog_entry(new_version, entry_date, summary, eol=eol)
    match = CHANGELOG_ENTRY_RE.search(changelog_text)
    if match:
        pos = match.start()
        new_changelog = changelog_text[:pos] + entry + changelog_text[pos:]
    else:
        # Kein bestehender Eintrag: nach dem Header anhängen
        header_pos = changelog_text.find(CHANGELOG_HEADER)
        if header_pos >= 0:
            header_end = header_pos + len(CHANGELOG_HEADER)
            new_changelog = changelog_text[:header_end] + eol + eol + entry + changelog_text[header_end:]
        else:
            new_changelog = changelog_text.rstrip("\r\n") + eol + eol + entry

    return new_pyproject, new_module, new_changelog, []


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Konsistenzprüfung Version/Changelog")

    bump_parser = sub.add_parser("bump", help="Versionsprung über alle Stellen ausführen")
    bump_parser.add_argument("new_version", help="Neue Version, z.B. 0.13.2")
    bump_parser.add_argument("--summary", required=True,
                             help="Kurzbeschreibung für den neuen Changelog-Eintrag")
    bump_parser.add_argument("--date", default=None,
                             help="Eintragsdatum (ISO), default: heute")

    args = parser.parse_args(argv)

    pyproject_text = _read(PYPROJECT_PATH)
    version_module_text = _read(VERSION_MODULE_PATH)
    changelog_text = _read(CHANGELOG_PATH)

    if args.command == "check":
        errors = check(pyproject_text, version_module_text, changelog_text)
        if errors:
            print("[FEHLER] Version/Changelog inkonsistent:")
            for err in errors:
                print(f"  - {err}")
            return 1
        v = get_pyproject_version(pyproject_text)
        print(f"[OK] Version/Changelog konsistent (Version {v})")
        return 0

    # bump
    new_py, new_mod, new_chlog, errors = bump(
        args.new_version, args.summary,
        pyproject_text, version_module_text, changelog_text,
        entry_date=args.date,
    )
    if errors:
        print("[FEHLER] Versionsprung abgelehnt:")
        for err in errors:
            print(f"  - {err}")
        return 1

    _write(PYPROJECT_PATH, new_py)
    _write(VERSION_MODULE_PATH, new_mod)
    _write(CHANGELOG_PATH, new_chlog)
    print(f"[OK] Versionsprung auf {args.new_version} über alle Stellen durchgeführt:")
    print(f"  - {PYPROJECT_PATH.name}")
    print(f"  - {VERSION_MODULE_PATH.relative_to(REPO_ROOT)}")
    print(f"  - {CHANGELOG_PATH.name} (neuer Eintrag: '{args.summary}')")
    return 0


if __name__ == "__main__":
    sys.exit(main())

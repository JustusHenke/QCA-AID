"""Lint: In der Webapp-UI duerfen keine Emoji-Strings mehr vorkommen.

Migration auf Material-Symbols-Shorthand (``:material/<name>:``) ist abgeschlossen
(siehe docs/ICON_MIGRATION_PLAN.md). Erlaubt bleiben Emojis in:

- Kommentaren
- Docstrings
- Log-/CLI-Strings (print, add_log, logger.* etc. -- bewusste Nutzer-Entscheidung)
- webapp_components/icons.py (Emoji->Icon-Mapping-Legende)

Falls ein neues UI-Emoji auftaucht: stattdessen ein Icon aus icons.py verwenden.
"""
from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent

EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"
    "\u2600-\u27BF"
    "\u2B00-\u2BFF"
    "\u2049\u203C\u2122\u2139\u231A\u231B\u2328\u23CF\u23E9-\u23FA"
    "\u24C2\u25AA\u25AB\u25B6\u25C0\u2934\u2935\u3030\u303D\u3297\u3299\uFE0F"
    "]"
)

SKIP_PLAIN_RE = re.compile(r"^(print|add_log|log_message|append_log|log|debug)$", re.I)
SKIP_RECEIVER_RE = re.compile(r"^(logger|logging|log|self)$", re.I)
SKIP_ATTR_RE = re.compile(
    r"^(add_log|log_message|append_log|log|debug|info|warning|error|critical)$", re.I
)


def _ui_emoji_strings(path: Path) -> list[tuple[int, str]]:
    """Liefert (Zeile, String-Praefix) fuer UI-Strings mit Emojis."""
    src = path.read_text(encoding="utf-8", errors="ignore")
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    hits: list[tuple[int, str]] = []
    for i, tok in enumerate(toks):
        if tok.type != tokenize.STRING or not EMOJI_RE.search(tok.string):
            continue
        ctx_name, ctx_receiver = None, None
        for j in range(i - 1, max(i - 10, -1), -1):
            t = toks[j]
            if t.type == tokenize.OP and t.string == "(":
                if j >= 1 and toks[j - 1].type == tokenize.NAME:
                    ctx_name = toks[j - 1].string
                elif (
                    j >= 3
                    and toks[j - 1].string == "."
                    and toks[j - 2].type == tokenize.NAME
                ):
                    ctx_name = toks[j - 2].string
                    ctx_receiver = (
                        toks[j - 3].string
                        if toks[j - 3].type == tokenize.NAME
                        else None
                    )
                break
            if t.type in (
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.STRING,
            ):
                break
        skip = (
            bool(ctx_name and SKIP_PLAIN_RE.match(ctx_name))
            if ctx_receiver is None
            else bool(
                ctx_receiver
                and SKIP_RECEIVER_RE.match(ctx_receiver)
                and ctx_name
                and SKIP_ATTR_RE.match(ctx_name)
            )
        )
        if not skip:
            hits.append((tok.start[0], tok.string[:60]))
    return hits


def test_no_emoji_in_ui_strings():
    offenders = []
    targets = [APP_DIR / "webapp.py", APP_DIR / "webapp_performance_optimizations.py"]
    targets += sorted((APP_DIR / "webapp_components").glob("*.py"))
    targets += sorted((APP_DIR / "webapp_logic").glob("*.py"))
    for path in targets:
        if path.name == "icons.py":
            continue
        for line, preview in _ui_emoji_strings(path):
            offenders.append(f"{path.name}:{line}: {preview!r}")
    assert not offenders, (
        "UI-Strings enthalten Emojis -- bitte :material/-Icons aus "
        "webapp_components/icons.py verwenden:\n" + "\n".join(offenders)
    )


# --- \\u-Escape-Variante: Emojis koennen auch als Quelltext-Escapes versteckt sein
ESCAPE_RE = re.compile(r"\\U([0-9A-Fa-f]{8})|\\u([0-9A-Fa-f]{4})")


def _is_emoji_cp(cp: int) -> bool:
    return (
        0x1F000 <= cp <= 0x1FAFF
        or 0x2600 <= cp <= 0x27BF
        or 0x2B00 <= cp <= 0x2BFF
        or cp in (0x2049, 0x203C, 0x2122, 0x2139, 0x231A, 0x231B, 0xFE0F)
    )


def test_no_emoji_unicode_escapes_in_source():
    offenders = []
    targets = [APP_DIR / "webapp.py", APP_DIR / "webapp_performance_optimizations.py"]
    targets += sorted((APP_DIR / "webapp_components").glob("*.py"))
    targets += sorted((APP_DIR / "webapp_logic").glob("*.py"))
    for path in targets:
        if path.name == "icons.py":
            continue
        for i, line in enumerate(
            path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
        ):
            for m in ESCAPE_RE.finditer(line):
                cp = int(m.group(1) or m.group(2), 16)
                if _is_emoji_cp(cp):
                    offenders.append(f"{path.name}:{i}: U+{cp:05X} {chr(cp)}")
                    break
    assert not offenders, (
        "UI-Quelltext enthaelt \\u-Escapes von Emojis "
        "(zur Laufzeit Emoji, fuer den Lint unsichtbar):\n" + "\n".join(offenders)
    )

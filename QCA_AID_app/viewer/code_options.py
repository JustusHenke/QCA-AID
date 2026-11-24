"""
Code Options
============
Auswahlbare Codes fuer die Pruf-Ansicht: Hauptkategorien und Subcodes aus dem
Codebook plus die im Material bereits vergebenen Codes
(siehe .kiro/specs/pdf-preview-annotation, Anforderung 6.5 "Change").

Reine Logik (headless testbar):
  - CodeOption:         ein auswaehlbarer Code (Kategorie + Subcodes)
  - build_code_options: Optionen aus Codebook und geladenen Kodierungen
  - parse_code_text:    Freitext "Kategorie > Subcode, Subcode2" -> CodeOption
  - recode:             Kodierung auf einen anderen Code umschreiben

Warum die geladenen Kodierungen mit einfliessen: LLM-Kodierungen koennen
Kategorien enthalten, die (noch) nicht im Codebook stehen - etwa im Modus
"induktiv". Diese Codes muessen auswaehlbar bleiben, sonst wuerde ein
Korrekturversuch die vorhandene Information vernichten.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from QCA_AID_assets.annotation.model import Coding

#: Trenner zwischen Kategorie und Subcode im Anzeigelabel und im Freitext.
PATH_SEPARATOR = "›"
#: Im Freitext werden auch ASCII-Varianten akzeptiert.
_ACCEPTED_SEPARATORS = (PATH_SEPARATOR, ">")
SUB_SEPARATOR = ", "


@dataclass(frozen=True)
class CodeOption:
    """Ein auswaehlbarer Code: Kategorie und optionale Subcodes."""

    category: str
    subcategories: Tuple[str, ...] = ()

    @property
    def label(self) -> str:
        """Anzeigetext, z.B. ``Steuerung › Legitimitaet``."""
        return code_label(self.category, self.subcategories)

    def matches(self, category: str, subcategories: Sequence[str]) -> bool:
        """True, wenn der Code exakt dieser Kombination entspricht."""
        return self.category == category and tuple(self.subcategories) == tuple(
            subcategories
        )


def code_label(category: str, subcategories: Sequence[str] = ()) -> str:
    """Baut den Anzeigetext eines Codes."""
    category = (category or "").strip()
    subs = [str(sub).strip() for sub in subcategories if str(sub).strip()]
    if not subs:
        return category
    return f"{category} {PATH_SEPARATOR} {SUB_SEPARATOR.join(subs)}"


# --------------------------------------------------------------------------
# Codebook lesen (tolerant gegen Dict- und Dataclass-Formen)
# --------------------------------------------------------------------------


def _category_name(name: Any, entry: Any) -> str:
    if isinstance(entry, dict):
        return str(entry.get("name") or name or "").strip()
    return str(getattr(entry, "name", None) or name or "").strip()


def _subcategory_names(entry: Any) -> List[str]:
    subs = entry.get("subcategories") if isinstance(entry, dict) else getattr(
        entry, "subcategories", None
    )
    if isinstance(subs, dict):
        names: Iterable[Any] = subs.keys()
    elif isinstance(subs, (list, tuple, set)):
        names = subs
    else:
        return []
    return [str(name).strip() for name in names if str(name).strip()]


def iter_codebook_categories(
    codebook_data: Any,
) -> Iterator[Tuple[str, List[str]]]:
    """Liefert (Kategoriename, Subcodes) aus CodebookData, dict oder None."""
    if codebook_data is None:
        return

    if isinstance(codebook_data, dict):
        categories = codebook_data.get("deduktive_kategorien", codebook_data)
    else:
        categories = getattr(codebook_data, "deduktive_kategorien", None)

    if not isinstance(categories, dict):
        return

    for name, entry in categories.items():
        category = _category_name(name, entry)
        if category:
            yield category, _subcategory_names(entry)


# --------------------------------------------------------------------------
# Optionen
# --------------------------------------------------------------------------


def build_code_options(
    codebook_data: Any = None,
    codings: Optional[Sequence[Coding]] = None,
) -> List[CodeOption]:
    """Sammelt alle auswaehlbaren Codes.

    Reihenfolge: alphabetisch nach Kategorie/Subcode, Duplikate entfernt.
    Jede Kategorie ist auch ohne Subcode waehlbar (Subcode entfaellt).
    """
    seen: Dict[Tuple[str, Tuple[str, ...]], None] = {}
    options: List[CodeOption] = []

    def add(category: str, subcategories: Sequence[str]) -> None:
        category = (category or "").strip()
        if not category:
            return
        subs = tuple(str(sub).strip() for sub in subcategories if str(sub).strip())
        key = (category, subs)
        if key in seen:
            return
        seen[key] = None
        options.append(CodeOption(category=category, subcategories=subs))

    for category, subs in iter_codebook_categories(codebook_data):
        add(category, ())
        for sub in subs:
            add(category, (sub,))

    for coding in codings or ():
        add(getattr(coding, "category", ""), tuple(getattr(coding, "subcategories", ()) or ()))

    options.sort(
        key=lambda option: (
            option.category.casefold(),
            tuple(sub.casefold() for sub in option.subcategories),
        )
    )
    return options


def find_option(
    options: Sequence[CodeOption],
    category: str,
    subcategories: Sequence[str] = (),
) -> Optional[CodeOption]:
    """Sucht den Code, der einer Kodierung entspricht."""
    for option in options:
        if option.matches(category, subcategories):
            return option
    return None


def option_index(
    options: Sequence[CodeOption],
    category: str,
    subcategories: Sequence[str] = (),
) -> Optional[int]:
    """Position des passenden Codes (fuer die Vorauswahl im Selectbox)."""
    for index, option in enumerate(options):
        if option.matches(category, subcategories):
            return index
    return None


# --------------------------------------------------------------------------
# Freitext
# --------------------------------------------------------------------------


def parse_code_text(text: str) -> Optional[CodeOption]:
    """Parst ``Kategorie > Subcode, Subcode2`` in eine CodeOption.

    Returns:
        CodeOption oder None, wenn keine Kategorie angegeben wurde.
    """
    text = (text or "").strip()
    if not text:
        return None

    category, _, rest = text, "", ""
    for separator in _ACCEPTED_SEPARATORS:
        if separator in text:
            category, _, rest = text.partition(separator)
            break

    category = category.strip()
    if not category:
        return None

    subs = tuple(
        part.strip()
        for part in rest.replace(";", ",").split(",")
        if part.strip()
    )
    return CodeOption(category=category, subcategories=subs)


# --------------------------------------------------------------------------
# Umschreiben
# --------------------------------------------------------------------------


def recode(coding: Coding, option: CodeOption, accept: bool = True) -> Coding:
    """Schreibt eine Kodierung auf einen anderen Code um.

    Die Aenderung ist eine menschliche Entscheidung: ``created_by`` wird
    ``human`` und der Status (per Default) ``accepted`` - eine Korrektur ist
    zugleich die Pruefung. Begruendung, Konfidenz und Anker bleiben unberuehrt.
    """
    return replace(
        coding,
        category=option.category,
        subcategories=list(option.subcategories),
        created_by="human",
        status="accepted" if accept else coding.status,
    )


__all__ = [
    "PATH_SEPARATOR",
    "CodeOption",
    "code_label",
    "iter_codebook_categories",
    "build_code_options",
    "find_option",
    "option_index",
    "parse_code_text",
    "recode",
]

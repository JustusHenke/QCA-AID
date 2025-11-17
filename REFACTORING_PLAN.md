# QCA_Utils.py Refactoring Plan

## Übersicht: Monolith → Modular Architecture

**Aktuelle Situation:**
- QCA_Utils.py: ~3954+ Zeilen
- 15 Klassen mit vollkommen unterschiedlichen Verantwortlichkeiten
- Schwer zu testen, warten und erweitern
- Imports sind chaotisch

**Ziel:** Zerlegung in 6 spezialisierte Module

---

## 🎯 Zielarchitektur

```
QCA_AID_assets/
├── utils/                          # Neues Modul
│   ├── __init__.py
│   ├── llm/                        # LLM-Provider und Responses
│   │   ├── __init__.py
│   │   ├── base.py                 # LLMProvider (abstract)
│   │   ├── openai_provider.py      # OpenAIProvider
│   │   ├── mistral_provider.py     # MistralProvider
│   │   ├── factory.py              # LLMProviderFactory
│   │   └── response.py             # LLMResponse
│   │
│   ├── config/                     # Konfiguration laden & validieren
│   │   ├── __init__.py
│   │   └── loader.py               # ConfigLoader
│   │
│   ├── tracking/                   # Token-Tracking & Kosten
│   │   ├── __init__.py
│   │   ├── token_tracker.py        # TokenTracker
│   │   └── token_counter.py        # TokenCounter (legacy)
│   │
│   ├── dialog/                     # Tkinter GUI Dialoge
│   │   ├── __init__.py
│   │   ├── widgets.py              # MultiSelectListbox
│   │   └── multiple_coding.py      # ManualMultipleCodingDialog
│   │
│   ├── export/                     # Export & Annotation
│   │   ├── __init__.py
│   │   ├── pdf_annotator.py        # PDFAnnotator
│   │   └── review.py               # ManualReviewGUI, ManualReviewComponent
│   │
│   ├── io/                         # Input/Output
│   │   ├── __init__.py
│   │   ├── document_reader.py      # DocumentReader
│   │   └── escape_handler.py       # EscapeHandler
│   │
│   └── common.py                   # Shared utilities (Konstanten, Helper)
│
├── QCA_Utils.py                    # DEPRECATED - nur noch imports für Rückwärtskompatibilität
```

---

## 📋 Detaillierte Zerlegung

### 1️⃣ **utils/llm/** - LLM Provider System

**Files:**
- `base.py` (100 Zeilen)
- `openai_provider.py` (130 Zeilen)
- `mistral_provider.py` (80 Zeilen)
- `factory.py` (45 Zeilen)
- `response.py` (30 Zeilen)

**Includes:**
- ✅ `LLMProvider` (abstract base class)
- ✅ `OpenAIProvider` mit Capability-Testing
- ✅ `MistralProvider`
- ✅ `LLMProviderFactory`
- ✅ `LLMResponse`

**Dependencies:** openai, mistralai

**Exports:** LLMProvider, OpenAIProvider, MistralProvider, LLMProviderFactory, LLMResponse

---

### 2️⃣ **utils/config/** - Configuration Loading

**Files:**
- `loader.py` (500 Zeilen)

**Includes:**
- ✅ `ConfigLoader` - vollständig
- Dependencies: openpyxl, pandas

**Key Features:**
- Excel Workbook Loading
- Category Definition Parsing
- Validation & Sanitization
- Multi-coder Settings

**Exports:** ConfigLoader

---

### 3️⃣ **utils/tracking/** - Token & Cost Tracking

**Files:**
- `token_tracker.py` (360 Zeilen)
- `token_counter.py` (55 Zeilen)

**Includes:**
- ✅ `TokenTracker` - vollständig
- ✅ `TokenCounter` - legacy support

**Dependencies:** datetime

**Exports:** TokenTracker, TokenCounter

---

### 4️⃣ **utils/dialog/** - Tkinter GUI Components

**Files:**
- `widgets.py` (60 Zeilen)
- `multiple_coding.py` (130 Zeilen)

**Includes:**
- ✅ `MultiSelectListbox`
- ✅ `ManualMultipleCodingDialog`

**Dependencies:** tkinter

**Exports:** MultiSelectListbox, ManualMultipleCodingDialog

---

### 5️⃣ **utils/export/** - PDF Export & Manual Review

**Files:**
- `pdf_annotator.py` (85 Zeilen)
- `review.py` (1000+ Zeilen)

**Includes (review.py):**
- ✅ `ManualReviewGUI`
- ✅ `ManualReviewComponent`

**Includes (pdf_annotator.py):**
- ✅ `PDFAnnotator`

**Dependencies:** tkinter, pypdf, reportlab

**Exports:** PDFAnnotator, ManualReviewGUI, ManualReviewComponent

---

### 6️⃣ **utils/io/** - Input/Output & Handlers

**Files:**
- `document_reader.py` (310 Zeilen)
- `escape_handler.py` (400 Zeilen)

**Includes:**
- ✅ `DocumentReader` - TXT/DOCX/PDF parsing
- ✅ `EscapeHandler` - ESC-key management

**Dependencies:** python-docx, PyPDF2, os, signal

**Exports:** DocumentReader, EscapeHandler

---

## 🔄 Migration Path

### Phase 1: Struktur erstellen (1h)
1. Neue Directory-Struktur erstellen
2. Leere `__init__.py` files
3. Basis-imports definieren

### Phase 2: LLM-System migrieren (1.5h)
1. `utils/llm/base.py` - LLMProvider abstract
2. `utils/llm/response.py` - LLMResponse
3. `utils/llm/openai_provider.py` - OpenAI implementation
4. `utils/llm/mistral_provider.py` - Mistral implementation
5. `utils/llm/factory.py` - Factory pattern
6. Tests: Token-Tracking, Capability detection

### Phase 3: Konfiguration migrieren (0.5h)
1. `utils/config/loader.py` - ConfigLoader
2. Tests: Excel-Laden, Validierung

### Phase 4: Tracking migrieren (0.5h)
1. `utils/tracking/token_tracker.py`
2. `utils/tracking/token_counter.py`
3. Tests: Cost calculation, session persistence

### Phase 5: GUI migrieren (1h)
1. `utils/dialog/widgets.py`
2. `utils/dialog/multiple_coding.py`
3. Tests: Manual coding workflows

### Phase 6: Export migrieren (2h)
1. `utils/export/review.py` - Review GUI (komplexeste!)
2. `utils/export/pdf_annotator.py`
3. Tests: PDF generation, review workflows

### Phase 7: IO migrieren (1h)
1. `utils/io/document_reader.py`
2. `utils/io/escape_handler.py`
3. Tests: Document parsing

### Phase 8: Compatibility Layer (0.5h)
1. Alte `QCA_Utils.py` → nur imports
2. Update alle `from QCA_Utils import` → `from utils import`
3. Backward-compatibility sichern

---

## 📊 Impact Analysis

### Dateien die angepasst werden müssen:
```
grep -r "from.*QCA_Utils import" QCA_AID_assets/
grep -r "import QCA_Utils" QCA_AID_assets/
```

Voraussichtlich:
- `main.py`
- `analysis_manager.py`
- `deductive_coding.py`
- `inductive_coding.py`
- `relevance_checker.py`
- `QCA_Prompts.py` (wenn QCA_Utils importiert)
- `results_exporter.py`
- Alle Test-Dateien

### Dependencies bleiben gleich:
```
openai, mistralai, python-docx, PyPDF2, pandas, openpyxl, tkinter
```

---

## ✅ Testing Strategy

Für jeden Schritt:
1. Unit-Tests der neuen Module
2. Import-Tests (alte API noch funktioniert?)
3. Integration-Tests mit realen Workflows

Critical paths to test:
- ✅ LLM API calls (mit capacity detection)
- ✅ Config loading (Excel parsing)
- ✅ Token tracking (cost calculation)
- ✅ GUI dialogs (tkinter)
- ✅ Document parsing (TXT/DOCX/PDF)
- ✅ Manual review (complex state management)

---

## 🎁 Benefits nach Refactoring

| Aspekt | Vorher | Nachher |
|--------|--------|---------|
| **Dateigröße** | 3954+ Zeilen | 6 Module mit ~200-500 Zeilen max |
| **Testbarkeit** | Schwer (Monolith) | Einfach (isolierte Module) |
| **Imports** | Chaotisch | Klar: `from utils.llm import ...` |
| **Wartbarkeit** | Hoch (alles durchsuchen) | Niedrig (klare Struktur) |
| **Reusability** | Schwer (alles gekoppelt) | Einfach (standalone modules) |
| **Entwicklung** | Merge-conflicts | Parallel möglich |
| **Onboarding** | Schwierig | Einfach (Modul = Konzept) |

---

## 🚀 Nächste Schritte

1. **Bestätigung dieses Plans** - Alle 6 Module okay?
2. **Phase-by-Phase Execution** - Mit Tests nach jeder Phase
3. **Parallel Testing** - Old vs New API compatibility
4. **Documentation** - Module docstrings, usage examples
5. **Grad ual Migration** - Code-by-code nicht alles auf einmal

---

## 📝 Notizen

- **Rückwärtskompatibilität:** Alte `from QCA_Utils import X` funktioniert noch (via Proxy-Imports)
- **Keine Breaking Changes** - Externe API bleibt identisch
- **Git-freundlich** - Kleine, fokussierte Commits pro Modul
- **Type Hints** - Alle neuen Module mit vollständigen type hints

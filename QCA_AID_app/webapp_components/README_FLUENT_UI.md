# Fluent UI Design System für QCA-AID Webapp

## 🎨 Übersicht

Dieses Modul implementiert Microsoft Fluent UI Design-Prinzipien für die QCA-AID Streamlit Webapp. Es bietet ein konsistentes, modernes und zugängliches Design-System.

## 📦 Module

### `fluent_styles.py`
Zentrale Design-Token und CSS-Generierung:
- **FluentColors**: Farbpalette (Brand, Neutral, Semantic)
- **FluentTypography**: Schriftarten, Größen, Gewichte
- **FluentSpacing**: 4px Grid-System für Abstände
- **FluentShadows**: Elevation/Schatten-System
- **FluentBorders**: Border Radius und Widths
- **get_fluent_css()**: Generiert vollständiges CSS

### `fluent_components.py`
Wiederverwendbare UI-Komponenten:
- `fluent_card()`: Fluent UI Card mit Titel und Inhalt
- `fluent_section_header()`: Section Header mit Icon
- `fluent_status_badge()`: Status-Badge (Success, Warning, Error, etc.)
- `fluent_divider()`: Horizontaler Trenner
- `fluent_info_box()`: Info/Success/Warning/Error Box
- `fluent_metric_card()`: Metrik-Anzeige mit Delta
- `fluent_button_group()`: Gruppe von Buttons

## 🚀 Quick Start

### 1. Automatische Anwendung (bereits integriert)

Das Fluent UI Design wird automatisch in `webapp.py` geladen:

```python
from webapp_components import get_fluent_css

st.markdown(get_fluent_css(), unsafe_allow_html=True)
```

### 2. Komponenten verwenden

```python
from webapp_components import (
    fluent_section_header,
    fluent_card,
    fluent_info_box
)

# Section Header
fluent_section_header(
    title="Konfiguration",
    subtitle="Verwalten Sie Ihre Einstellungen",
    icon="⚙️"
)

# Card
fluent_card(
    title="Willkommen",
    content="Dies ist eine Fluent UI Card",
    icon="👋",
    elevated=True
)

# Info Box
fluent_info_box(
    message="Erfolgreich gespeichert",
    box_type="success"
)
```

## 🎯 Design-Prinzipien

### Farben
- **Primary**: #0078D4 (Fluent Blue)
- **Neutral Background**: #FFFFFF
- **Neutral Foreground**: #242424
- **Success**: #107C10
- **Warning**: #F7630C
- **Error**: #D13438

### Typografie
- **Font Family**: Segoe UI (mit Fallbacks)
- **Sizes**: 10px - 40px (Scale 100-900)
- **Weights**: Regular (400), Semibold (600), Bold (700)

### Spacing (4px Grid)
- XXS: 4px
- XS: 8px
- S: 12px
- M: 16px ← Standard
- L: 20px
- XL: 24px
- XXL: 32px

### Shadows
- Shadow 2: Subtil (Cards, Buttons)
- Shadow 4: Leicht (Hover)
- Shadow 8: Mittel (Dialoge)
- Shadow 16: Hoch (Modals)

### Borders
- Radius: 4px (Standard)
- Width: 1px (Standard)

## 📚 Dokumentation

- **FLUENT_UI_GUIDE.md**: Vollständige Dokumentation
- **FLUENT_UI_MIGRATION_EXAMPLE.md**: Vorher/Nachher Beispiele

## 🔗 Referenzen

- [Fluent UI Layout](https://fluent2.microsoft.design/layout)
- [Fluent UI Colors](https://storybooks.fluentui.dev/react/?path=/docs/theme-colors--docs)
- [Fluent UI Typography](https://storybooks.fluentui.dev/react/?path=/docs/theme-typography--docs)
- [Fluent UI Shadows](https://storybooks.fluentui.dev/react/?path=/docs/theme-shadows--docs)
- [Fluent UI Spacing](https://storybooks.fluentui.dev/react/?path=/docs/theme-spacing--docs)

## ✅ Features

- ✅ Vollständig rückwärtskompatibel
- ✅ Automatische Anwendung auf alle Streamlit-Komponenten
- ✅ Wiederverwendbare Komponenten
- ✅ Konsistentes Design-System
- ✅ Keine externen Abhängigkeiten
- ✅ Performance-optimiert
- ✅ Browser-kompatibel (Chrome, Firefox, Safari, Edge)

## 🛠️ Anpassung

Um das Design anzupassen, bearbeiten Sie die Klassen in `fluent_styles.py`:

```python
class FluentColors:
    BRAND_PRIMARY = "#0078D4"  # Ihre Farbe hier

class FluentSpacing:
    M = "16px"  # Ihr Standard-Abstand
```

Änderungen werden automatisch auf die gesamte App angewendet.

## 📝 Beispiele

### Section Header
```python
fluent_section_header(
    title="Analyse",
    subtitle="Starten Sie Ihre Analyse",
    icon="🔬"
)
```

### Metric Card
```python
fluent_metric_card(
    label="Dokumente",
    value="42",
    delta="+5",
    delta_positive=True,
    icon="📄"
)
```

### Status Badge
```python
badge = fluent_status_badge("Aktiv", status="success")
st.markdown(badge, unsafe_allow_html=True)
```

## 🤝 Beitragen

Bei Fragen oder Verbesserungsvorschlägen:
1. Dokumentation in `FLUENT_UI_GUIDE.md` prüfen
2. Beispiele in `FLUENT_UI_MIGRATION_EXAMPLE.md` ansehen
3. Code in `fluent_styles.py` und `fluent_components.py` anpassen

## 📄 Lizenz

Teil des QCA-AID Projekts. Siehe Haupt-LICENSE Datei.

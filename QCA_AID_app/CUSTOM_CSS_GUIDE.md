# Custom CSS Styles - Quick Guide

## 📝 Wo CSS-Styles ändern?

**Datei**: `QCA_AID_app/webapp_components/fluent_styles.py`

**Funktion**: `get_fluent_css()`

**Bereich**: Am Ende vor `</style>` (Abschnitt "CUSTOM STYLES")

---

## 🎯 Schritt-für-Schritt Anleitung

### 1. CSS-Klasse finden

**Browser DevTools öffnen**:
- Windows/Linux: `F12` oder `Ctrl+Shift+I`
- macOS: `Cmd+Option+I`

**Element inspizieren**:
1. Rechtsklick auf Element → "Untersuchen" / "Inspect"
2. CSS-Klassen im Inspector ansehen
3. Klasse kopieren (z.B. `.st-emotion-cache-1234`)

### 2. Style hinzufügen

**Öffne**: `QCA_AID_app/webapp_components/fluent_styles.py`

**Suche nach**: `/* ===== CUSTOM STYLES ===== */`

**Füge hinzu**:
```python
    /* ===== CUSTOM STYLES ===== */
    
    /* Dein Custom Style */
    .st-emotion-cache-1234 {
        background-color: #F5F5F5;
        padding: 16px;
        border-radius: 4px;
    }
```

### 3. Webapp neu starten

```bash
streamlit run QCA_AID_app/webapp.py
```

---

## 💡 Beispiele

### Beispiel 1: Alle st-emotion Klassen stylen

```python
    /* Alle st-emotion Klassen */
    [class*="st-emotion"] {
        font-family: 'Segoe UI', sans-serif;
    }
```

### Beispiel 2: Spezifische Komponente

```python
    /* Spezifische Komponente */
    .st-emotion-cache-xyz {
        background-color: #F5F5F5;
        border: 1px solid #E1DFDD;
        border-radius: 4px;
        padding: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
```

### Beispiel 3: Container stylen

```python
    /* Container */
    .st-emotion-cache-container {
        max-width: 1200px;
        margin: 0 auto;
    }
```

### Beispiel 4: Buttons anpassen

```python
    /* Custom Button Style */
    .st-emotion-cache-button {
        background-color: #0078D4;
        color: white;
        border-radius: 4px;
        padding: 8px 16px;
    }
    
    .st-emotion-cache-button:hover {
        background-color: #106EBE;
    }
```

### Beispiel 5: Cards stylen

```python
    /* Card Style */
    .st-emotion-cache-card {
        background-color: white;
        border: 1px solid #E1DFDD;
        border-radius: 4px;
        padding: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
```

---

## 🎨 Fluent UI Farben verwenden

Du kannst die Fluent UI Farben aus den Klassen verwenden:

```python
    /* Mit Fluent UI Farben */
    .st-emotion-cache-xyz {{
        background-color: {FluentColors.NEUTRAL_BACKGROUND_SECONDARY};
        border-color: {FluentColors.NEUTRAL_STROKE};
        color: {FluentColors.NEUTRAL_FOREGROUND};
    }}
```

**Verfügbare Farben**:
- `{FluentColors.BRAND_PRIMARY}` - #0078D4 (Fluent Blue)
- `{FluentColors.NEUTRAL_BACKGROUND}` - #FFFFFF
- `{FluentColors.NEUTRAL_BACKGROUND_SECONDARY}` - #F5F5F5
- `{FluentColors.NEUTRAL_STROKE}` - #E1DFDD
- `{FluentColors.NEUTRAL_FOREGROUND}` - #242424
- `{FluentColors.SUCCESS}` - #107C10
- `{FluentColors.WARNING}` - #F7630C
- `{FluentColors.ERROR}` - #D13438

---

## ⚠️ Wichtige Hinweise

### ✅ Sicher (nur Farben/Font)
```css
/* Sicher */
.st-emotion-cache-xyz {
    background-color: #F5F5F5;
    color: #242424;
    border-color: #E1DFDD;
}
```

### ⚠️ Vorsichtig (Layout-Eigenschaften)
```css
/* Kann Layout beeinflussen */
.st-emotion-cache-xyz {
    padding: 16px;        /* Ändert Größe */
    margin: 20px;         /* Verschiebt Position */
    border-width: 2px;    /* Ändert Größe */
    height: 100px;        /* Ändert Größe */
}
```

### ❌ Vermeiden (Breaking Changes)
```css
/* Kann Layout zerstören */
.st-emotion-cache-xyz {
    display: none;        /* Versteckt Element */
    position: absolute;   /* Verschiebt aus Flow */
    width: 100%;          /* Überschreibt Breite */
}
```

---

## 🔍 Debugging

### CSS funktioniert nicht?

1. **Cache leeren**: Browser-Cache leeren (Ctrl+Shift+Delete)
2. **Hard Reload**: Ctrl+F5 (Windows) / Cmd+Shift+R (macOS)
3. **DevTools prüfen**: Ist der Style angewendet? Wird er überschrieben?
4. **Spezifität erhöhen**: Mehr spezifische Selektoren verwenden

### Beispiel: Spezifität erhöhen
```css
/* Niedrige Spezifität */
.st-emotion-cache-xyz {
    color: red;
}

/* Höhere Spezifität */
.stApp .st-emotion-cache-xyz {
    color: red;
}

/* Noch höhere Spezifität */
div.stApp > div > .st-emotion-cache-xyz {
    color: red;
}

/* Maximum (nur wenn nötig) */
.st-emotion-cache-xyz {
    color: red !important;
}
```

---

## 📚 Weitere Ressourcen

### Streamlit CSS-Klassen
- `.stApp` - Haupt-App-Container
- `.stButton` - Button-Container
- `.stTextInput` - Text Input-Container
- `.stSelectbox` - Select Box-Container
- `.stTabs` - Tabs-Container
- `[data-testid="stSidebar"]` - Sidebar
- `[data-testid="stHeader"]` - Header

### Fluent UI Referenzen
- [Fluent UI Colors](https://storybooks.fluentui.dev/react/?path=/docs/theme-colors--docs)
- [Fluent UI Typography](https://storybooks.fluentui.dev/react/?path=/docs/theme-typography--docs)
- [Fluent UI Spacing](https://storybooks.fluentui.dev/react/?path=/docs/theme-spacing--docs)

---

## 💡 Tipps

1. **Klein anfangen**: Teste mit einem Element
2. **DevTools nutzen**: Live-Editing im Browser
3. **Backup machen**: Kopiere `fluent_styles.py` vor Änderungen
4. **Dokumentieren**: Kommentiere deine Custom Styles
5. **Testen**: Prüfe in verschiedenen Browsern

---

## 🚀 Quick Start

```python
# 1. Öffne fluent_styles.py
# 2. Suche nach "CUSTOM STYLES"
# 3. Füge hinzu:

    /* Mein Custom Style */
    .st-emotion-cache-xyz {
        background-color: #F5F5F5;
        border-radius: 4px;
    }

# 4. Speichern
# 5. Webapp neu starten
# 6. Testen!
```

---

**Datei**: `QCA_AID_app/webapp_components/fluent_styles.py`  
**Bereich**: `/* ===== CUSTOM STYLES ===== */`  
**Tipp**: Browser DevTools (F12) zum Inspizieren nutzen!

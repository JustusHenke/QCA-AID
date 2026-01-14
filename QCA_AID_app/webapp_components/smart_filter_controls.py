"""
Smart Filter Controls for QCA-AID Explorer Webapp
=================================================

This module provides intelligent dropdown filter controls that automatically
load available categories from the Excel analysis results.
"""

import streamlit as st
from typing import Dict, List, Optional, Any, Tuple
from webapp_models.explorer_config_data import AnalysisConfig
from webapp_logic.category_loader import CategoryLoader


def render_smart_filter_controls(analysis: AnalysisConfig, index: int, category_loader: Optional[CategoryLoader] = None) -> bool:
    """
    Rendert intelligente Filter-Controls mit Dropdown-Unterstützung.
    
    Args:
        analysis: AnalysisConfig to render filters for
        index: Index of the analysis
        category_loader: CategoryLoader instance for intelligent dropdowns
        
    Returns:
        bool: True if filters were updated
    """
    st.markdown("Filtern Sie die Daten vor der Analyse:")
    
    filters = analysis.filters.copy()
    updated = False
    
    # Check if we have a category loader with filter values
    has_filter_values = category_loader is not None
    
    # Document filter - with dropdown if values available
    if has_filter_values:
        documents = category_loader.get_documents()
        if documents:
            doc_options = ["(Alle Dokumente)"] + documents
            current_doc = filters.get('Dokument')
            doc_index = documents.index(current_doc) + 1 if current_doc and current_doc in documents else 0
            
            selected_doc = st.selectbox(
                "Dokument:",
                options=doc_options,
                index=doc_index,
                key=f"filter_dokument_dropdown_{index}",
                help="Wählen Sie ein Dokument oder lassen Sie 'Alle' für keine Filterung"
            )
            
            new_dokument = selected_doc if selected_doc != "(Alle Dokumente)" else None
            if new_dokument != filters.get('Dokument'):
                filters['Dokument'] = new_dokument
                updated = True
        else:
            # Fallback to text input
            new_dokument = st.text_input(
                "Dokument:",
                value=filters.get('Dokument') or '',
                key=f"filter_dokument_{index}",
                help="Leer lassen für alle Dokumente"
            )
            if (new_dokument or None) != filters.get('Dokument'):
                filters['Dokument'] = new_dokument if new_dokument else None
                updated = True
    else:
        # Fallback to text input
        new_dokument = st.text_input(
            "Dokument:",
            value=filters.get('Dokument') or '',
            key=f"filter_dokument_{index}",
            help="Leer lassen für alle Dokumente"
        )
        if (new_dokument or None) != filters.get('Dokument'):
            filters['Dokument'] = new_dokument if new_dokument else None
            updated = True
    
    # Intelligente Kategorie-Filter
    if category_loader and category_loader.is_loaded:
        # Zeige Statistiken
        stats = category_loader.get_statistics()
        st.info(f"📊 Verfügbare Kategorien: {stats['total_main_categories']} Haupt-, {stats['total_subcategories']} Subkategorien")
        
        # Hauptkategorie mit Dropdown
        main_categories = category_loader.get_main_categories()
        current_main = filters.get('Hauptkategorie')
        
        # Erstelle Optionen für Selectbox (mit "Alle" Option)
        main_options = ["(Alle Hauptkategorien)"] + main_categories
        
        # Bestimme aktuellen Index
        if current_main and current_main in main_categories:
            main_index = main_categories.index(current_main) + 1  # +1 wegen "Alle" Option
        else:
            main_index = 0
        
        selected_main = st.selectbox(
            "Hauptkategorie:",
            options=main_options,
            index=main_index,
            key=f"filter_hauptkat_dropdown_{index}",
            help="Wählen Sie eine Hauptkategorie oder lassen Sie 'Alle' für keine Filterung"
        )
        
        # Aktualisiere Filter
        new_hauptkat = selected_main if selected_main != "(Alle Hauptkategorien)" else None
        if new_hauptkat != current_main:
            filters['Hauptkategorie'] = new_hauptkat
            updated = True
            
            # Wenn sich die Hauptkategorie ändert, setze Subkategorien zurück
            if filters.get('Subkategorien'):
                filters['Subkategorien'] = None
                updated = True
        
        # Subkategorien mit Multiselect (abhängig von Hauptkategorie)
        if new_hauptkat:
            # Zeige nur Subkategorien der gewählten Hauptkategorie
            available_subcategories = category_loader.get_subcategories(new_hauptkat)
            
            # Zeige Kategorie-Info
            category_info = category_loader.get_category_info(new_hauptkat)
            if category_info and category_info.get('definition'):
                with st.expander(f"ℹ️ Info zu '{new_hauptkat}'", expanded=False):
                    st.markdown(f"**Typ:** {category_info.get('typ', 'N/A')}")
                    st.markdown(f"**Definition:** {category_info.get('definition', 'N/A')}")
        else:
            # Zeige alle Subkategorien
            available_subcategories = category_loader.get_all_subcategories()
        
        if available_subcategories:
            # Parse aktuelle Subkategorien
            current_subcategories = []
            current_subs_str = filters.get('Subkategorien')
            if current_subs_str:
                current_subcategories = [sub.strip() for sub in current_subs_str.split(',') if sub.strip()]
            
            # Filtere nur gültige Subkategorien (falls sich Hauptkategorie geändert hat)
            valid_current_subcategories = [
                sub for sub in current_subcategories 
                if sub in available_subcategories
            ]
            
            selected_subcategories = st.multiselect(
                "Subkategorien:",
                options=available_subcategories,
                default=valid_current_subcategories,
                key=f"filter_subkat_multiselect_{index}",
                help="Wählen Sie eine oder mehrere Subkategorien (Strg+Klick für Mehrfachauswahl)"
            )
            
            # Aktualisiere Filter
            new_subkat = ', '.join(selected_subcategories) if selected_subcategories else None
            if new_subkat != filters.get('Subkategorien'):
                filters['Subkategorien'] = new_subkat
                updated = True
        else:
            st.info("Keine Subkategorien verfügbar für die gewählte Hauptkategorie")
            if filters.get('Subkategorien'):
                filters['Subkategorien'] = None
                updated = True
    
    else:
        # Fallback: Normale Textfelder wenn keine Kategorien verfügbar
        st.warning("⚠️ Keine Kategorien verfügbar - verwende Freitext-Eingabe")
        
        # Hauptkategorie als Textfeld
        new_hauptkat = st.text_input(
            "Hauptkategorie:",
            value=filters.get('Hauptkategorie') or '',
            key=f"filter_hauptkat_text_{index}",
            help="Leer lassen für alle Hauptkategorien"
        )
        if (new_hauptkat or None) != filters.get('Hauptkategorie'):
            filters['Hauptkategorie'] = new_hauptkat if new_hauptkat else None
            updated = True
        
        # Subkategorien als Textfeld
        new_subkat = st.text_input(
            "Subkategorien:",
            value=filters.get('Subkategorien') or '',
            key=f"filter_subkat_text_{index}",
            help="Komma-getrennte Liste oder leer für alle Subkategorien"
        )
        if (new_subkat or None) != filters.get('Subkategorien'):
            filters['Subkategorien'] = new_subkat if new_subkat else None
            updated = True
    
    # Attribute Filter - with dropdowns if values available
    col1, col2 = st.columns(2)
    
    with col1:
        # Attribut1 filter - with dropdown if values available
        if has_filter_values:
            attribut1_values = category_loader.get_attribut1_values()
            if attribut1_values:
                attr1_options = ["(Alle Werte)"] + attribut1_values
                current_attr1 = filters.get('Attribut_1')
                attr1_index = attribut1_values.index(current_attr1) + 1 if current_attr1 and current_attr1 in attribut1_values else 0
                
                selected_attr1 = st.selectbox(
                    "Attribut 1:",
                    options=attr1_options,
                    index=attr1_index,
                    key=f"filter_attr1_dropdown_{index}",
                    help="Wählen Sie einen Wert für Attribut 1 oder lassen Sie 'Alle' für keine Filterung"
                )
                
                new_attr1 = selected_attr1 if selected_attr1 != "(Alle Werte)" else None
                if new_attr1 != filters.get('Attribut_1'):
                    filters['Attribut_1'] = new_attr1
                    updated = True
            else:
                # Fallback to text input
                new_attr1 = st.text_input(
                    "Attribut 1:",
                    value=filters.get('Attribut_1') or '',
                    key=f"filter_attr1_{index}",
                    help="Leer lassen für alle Werte"
                )
                if (new_attr1 or None) != filters.get('Attribut_1'):
                    filters['Attribut_1'] = new_attr1 if new_attr1 else None
                    updated = True
        else:
            # Fallback to text input
            new_attr1 = st.text_input(
                "Attribut 1:",
                value=filters.get('Attribut_1') or '',
                key=f"filter_attr1_{index}",
                help="Leer lassen für alle Werte"
            )
            if (new_attr1 or None) != filters.get('Attribut_1'):
                filters['Attribut_1'] = new_attr1 if new_attr1 else None
                updated = True
    
    with col2:
        # Attribut2 filter - with dropdown if values available
        if has_filter_values:
            attribut2_values = category_loader.get_attribut2_values()
            if attribut2_values:
                attr2_options = ["(Alle Werte)"] + attribut2_values
                current_attr2 = filters.get('Attribut_2')
                attr2_index = attribut2_values.index(current_attr2) + 1 if current_attr2 and current_attr2 in attribut2_values else 0
                
                selected_attr2 = st.selectbox(
                    "Attribut 2:",
                    options=attr2_options,
                    index=attr2_index,
                    key=f"filter_attr2_dropdown_{index}",
                    help="Wählen Sie einen Wert für Attribut 2 oder lassen Sie 'Alle' für keine Filterung"
                )
                
                new_attr2 = selected_attr2 if selected_attr2 != "(Alle Werte)" else None
                if new_attr2 != filters.get('Attribut_2'):
                    filters['Attribut_2'] = new_attr2
                    updated = True
            else:
                # Fallback to text input
                new_attr2 = st.text_input(
                    "Attribut 2:",
                    value=filters.get('Attribut_2') or '',
                    key=f"filter_attr2_{index}",
                    help="Leer lassen für alle Werte"
                )
                if (new_attr2 or None) != filters.get('Attribut_2'):
                    filters['Attribut_2'] = new_attr2 if new_attr2 else None
                    updated = True
        else:
            # Fallback to text input
            new_attr2 = st.text_input(
                "Attribut 2:",
                value=filters.get('Attribut_2') or '',
                key=f"filter_attr2_{index}",
                help="Leer lassen für alle Werte"
            )
            if (new_attr2 or None) != filters.get('Attribut_2'):
                filters['Attribut_2'] = new_attr2 if new_attr2 else None
                updated = True
    
    # Validierung anzeigen (falls Kategorien verfügbar)
    if category_loader and category_loader.is_loaded and updated:
        main_category = filters.get('Hauptkategorie')
        subcategories_str = filters.get('Subkategorien')
        
        # Parse Subkategorien
        subcategories = None
        if subcategories_str:
            subcategories = [sub.strip() for sub in subcategories_str.split(',') if sub.strip()]
        
        # Validiere
        is_valid, validation_errors = category_loader.validate_filter_values(main_category, subcategories)
        
        if not is_valid:
            st.error("❌ **Filter-Validierungsfehler:**")
            for error in validation_errors:
                st.error(f"• {error}")
        else:
            st.success("✅ Filter sind gültig")
    
    # Aktualisiere Analysis-Objekt falls nötig
    if updated:
        # Verwende die handle_update_analysis Funktion aus der Explorer UI
        # (wird über Session State kommuniziert)
        st.session_state[f'filter_update_{index}'] = filters
    
    return updated


def render_category_statistics(category_loader: CategoryLoader) -> None:
    """
    Zeigt Statistiken über verfügbare Kategorien an.
    
    Args:
        category_loader: CategoryLoader instance
    """
    if not category_loader or not category_loader.is_loaded:
        st.warning("Keine Kategorien verfügbar")
        return
    
    stats = category_loader.get_statistics()
    
    st.subheader("📊 Kategorie-Statistiken")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Hauptkategorien", stats['total_main_categories'])
    
    with col2:
        st.metric("Subkategorien", stats['total_subcategories'])
    
    with col3:
        st.metric("Ø Subkategorien", stats['average_subcategories_per_main'])
    
    # Detaillierte Aufschlüsselung
    with st.expander("📋 Detaillierte Aufschlüsselung", expanded=False):
        main_categories = category_loader.get_main_categories()
        category_mapping = category_loader.get_category_mapping()
        
        for main_cat in main_categories:
            subcats = category_mapping.get(main_cat, [])
            st.markdown(f"**{main_cat}** ({len(subcats)} Subkategorien)")
            
            if subcats:
                # Zeige Subkategorien in Spalten für bessere Übersicht
                if len(subcats) <= 3:
                    cols = st.columns(len(subcats))
                    for i, subcat in enumerate(subcats):
                        cols[i].caption(f"• {subcat}")
                else:
                    # Bei vielen Subkategorien, zeige als Liste
                    subcats_text = ", ".join(subcats)
                    st.caption(subcats_text)
            else:
                st.caption("(Keine Subkategorien)")
            
            st.markdown("---")


def render_filter_validation_summary(analysis_configs: List[AnalysisConfig], category_loader: Optional[CategoryLoader]) -> None:
    """
    Zeigt eine Zusammenfassung der Filter-Validierung für alle Analysen.
    
    Args:
        analysis_configs: Liste aller Analysekonfigurationen
        category_loader: CategoryLoader instance für Validierung
    """
    if not category_loader or not category_loader.is_loaded:
        return
    
    st.subheader("🔍 Filter-Validierung")
    
    all_valid = True
    validation_results = []
    
    for i, analysis in enumerate(analysis_configs):
        errors = []
        
        # Validiere Filter
        main_category = analysis.filters.get('Hauptkategorie')
        subcategories_str = analysis.filters.get('Subkategorien')
        
        # Parse Subkategorien
        subcategories = None
        if subcategories_str:
            subcategories = [sub.strip() for sub in subcategories_str.split(',') if sub.strip()]
        
        # Validiere gegen Kategorien
        is_valid, validation_errors = category_loader.validate_filter_values(main_category, subcategories)
        
        validation_results.append({
            'name': analysis.name,
            'index': i,
            'is_valid': is_valid,
            'errors': validation_errors,
            'active': analysis.active
        })
        
        if not is_valid and analysis.active:
            all_valid = False
    
    # Zeige Gesamtstatus
    if all_valid:
        st.success("✅ Alle aktiven Analysen haben gültige Filter")
    else:
        st.error("❌ Einige aktive Analysen haben ungültige Filter")
    
    # Zeige Details für jede Analyse
    for result in validation_results:
        if result['is_valid']:
            if result['active']:
                st.success(f"✅ **{result['name']}** - Filter gültig")
            else:
                st.info(f"ℹ️ **{result['name']}** - Filter gültig (Analyse deaktiviert)")
        else:
            if result['active']:
                st.error(f"❌ **{result['name']}** - Filter ungültig:")
                for error in result['errors']:
                    st.error(f"   • {error}")
            else:
                st.warning(f"⚠️ **{result['name']}** - Filter ungültig (Analyse deaktiviert):")
                for error in result['errors']:
                    st.warning(f"   • {error}")


def get_category_suggestions(category_loader: CategoryLoader, query: str, filter_type: str = 'all') -> List[str]:
    """
    Gibt Kategorie-Vorschläge basierend auf einer Suchanfrage zurück.
    
    Args:
        category_loader: CategoryLoader instance
        query: Suchbegriff
        filter_type: Typ der Suche ('main', 'sub', 'all')
        
    Returns:
        Liste von passenden Kategorien
    """
    if not category_loader or not category_loader.is_loaded:
        return []
    
    query_lower = query.lower()
    suggestions = []
    
    if filter_type in ['main', 'all']:
        # Suche in Hauptkategorien
        main_categories = category_loader.get_main_categories()
        for cat in main_categories:
            if query_lower in cat.lower():
                suggestions.append(cat)
    
    if filter_type in ['sub', 'all']:
        # Suche in Subkategorien
        all_subcategories = category_loader.get_all_subcategories()
        for cat in all_subcategories:
            if query_lower in cat.lower():
                suggestions.append(cat)
    
    return suggestions[:10]  # Limitiere auf 10 Vorschläge
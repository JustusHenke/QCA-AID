"""
Codebook UI Component
=====================
Streamlit UI component for managing QCA-AID codebook.

Requirements: 1.2, 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.2, 4.3, 4.4, 4.5, 5.1, 5.2, 5.3, 5.4, 5.5
"""

import streamlit as st
from typing import Dict, List
from datetime import datetime
from pathlib import Path

from webapp_models.codebook_data import CodebookData, CategoryData
from webapp_logic.file_browser_service import FileBrowserService
from webapp_logic.inductive_code_extractor import InductiveCodeExtractor
from webapp_logic.ui_helpers import (
    truncate_path_for_display,
    show_file_operation_success,
    show_file_operation_error,
    show_real_time_path_validation
)


def render_codebook_tab():
    """
    Rendert Codebook-Reiter als Hauptlayout.
    
    Requirements:
    - 1.2: Display project root directory
    - 3.1-3.5: File browser for codebook files
    - 4.1-4.5: Kategorieverwaltung
    - 4.1-4.3: Inductive code detection and import
    - 5.1-5.5: Forschungsfrage und Kodierregeln
    """
    st.header("📚 Codebook")
    st.markdown("Verwalten Sie Forschungsfrage, Kodierregeln und Kategorien")
    
    # Display project root prominently (Requirement 1.2)
    project_manager = st.session_state.project_manager
    project_root = project_manager.get_root_directory()
    st.info(f"📁 **Projekt-Verzeichnis:** `{project_root}`")
    
    # Workflow hint - check if codebook needs attention
    if 'codebook_data' in st.session_state and st.session_state.codebook_data:
        codebook = st.session_state.codebook_data
        if len(codebook.deduktive_kategorien) == 0:
            st.warning("⚠️ **Aktion erforderlich:** Ihr Codebook enthält keine Kategorien. Laden Sie ein bestehendes Codebook oder erstellen Sie neue Kategorien.")
        elif not codebook.forschungsfrage or not codebook.forschungsfrage.strip():
            st.info("💡 **Empfehlung:** Definieren Sie eine Forschungsfrage für bessere Analyseergebnisse.")
        else:
            st.success(f"✅ Codebook bereit: {len(codebook.deduktive_kategorien)} Kategorien definiert")
    
    # Get or initialize codebook from session state
    if 'codebook_data' not in st.session_state or st.session_state.codebook_data is None:
        # Initialize with empty codebook
        st.session_state.codebook_data = CodebookData(
            forschungsfrage='',
            kodierregeln={'general': [], 'format': [], 'exclusion': []},
            deduktive_kategorien={}
        )
        st.session_state.codebook_modified = False
    
    codebook = st.session_state.codebook_data
    
    # Check for inductive codes on tab open (Requirements 4.1, 4.2)
    check_inductive_codes_available()
    
    # File operations section
    render_file_operations()
    
    # Render merge dialog if active
    render_inductive_merge_dialog()
    
    st.markdown("---")
    
    # Create two columns for layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Research question
        st.subheader("🔍 Forschungsfrage")
        render_research_question()
        
        st.markdown("---")
        
        # Coding rules
        st.subheader("📋 Kodierregeln")
        render_coding_rules()
        
        st.markdown("---")
        
        # Categories
        st.subheader("🏷️ Kategorien")
        render_categories()
    
    with col2:
        # JSON Preview
        st.subheader("👁️ JSON-Vorschau")
        render_json_preview()
        
        st.markdown("---")
        
        # Validation status
        render_validation_status()


def render_file_operations():
    """
    Rendert Laden/Speichern Buttons für Codebook mit File Browser Integration.
    
    Requirements:
    - 3.1: Display current file path in input field
    - 3.2: Show default path when empty
    - 3.3: File browser button next to path input
    - 3.4: Display full path when file selected
    - 3.5: Save dialog with suggested filename
    """
    # Show current codebook status
    codebook = st.session_state.codebook_data
    if codebook and st.session_state.get('codebook_loaded_from') == 'file':
        st.success(f"✅ Codebook geladen ({len(codebook.deduktive_kategorien)} Kategorien)")
    elif not codebook or len(codebook.deduktive_kategorien) == 0:
        st.info("ℹ️ Kein Codebook geladen - Laden Sie ein bestehendes oder erstellen Sie ein neues")
    
    col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 1, 1])
    
    with col1:
        st.markdown("**Dateioperationen**")
    
    with col2:
        # Load button - highlight if no categories exist
        button_type = "primary" if not codebook or len(codebook.deduktive_kategorien) == 0 else "secondary"
        if st.button("📂 Codebook laden", use_container_width=True, key="codebook_load_main_btn", type=button_type):
            st.session_state.show_codebook_load_dialog = True
    
    with col3:
        # Save button - highlight if modified
        button_type = "primary" if st.session_state.get('codebook_modified', False) else "secondary"
        if st.button("💾 Speichern", use_container_width=True, key="codebook_save_main_btn", type=button_type):
            st.session_state.show_codebook_save_dialog = True
    
    with col4:
        # Remove codebook button
        if codebook and len(codebook.deduktive_kategorien) > 0:
            if st.button("🗑️ Entfernen", use_container_width=True, key="codebook_remove_btn", 
                        help="Aktuelles Codebook entfernen", type="secondary"):
                st.session_state.show_codebook_remove_dialog = True
    
    with col5:
        # Import inductive codes button (Requirement 4.3)
        if st.session_state.get('inductive_codes_available', False):
            if st.button("📥 Import", use_container_width=True, key="codebook_import_inductive_btn", 
                        help="Induktive Codes importieren"):
                st.session_state.show_inductive_import_dialog = True
    
    # Remove codebook confirmation dialog
    if st.session_state.get('show_codebook_remove_dialog', False):
        with st.expander("🗑️ Codebook entfernen", expanded=True):
            st.warning("⚠️ Möchten Sie das aktuelle Codebook wirklich entfernen?")
            st.info("Das Codebook wird nur aus der Anwendung entfernt, nicht von der Festplatte gelöscht.")
            
            col_remove1, col_remove2 = st.columns(2)
            
            with col_remove1:
                if st.button("Ja, entfernen", use_container_width=True, key='codebook_remove_confirm', type="primary"):
                    # Reset codebook to empty state
                    from webapp_models.codebook_data import CodebookData
                    st.session_state.codebook_data = CodebookData(
                        forschungsfrage='',
                        kodierregeln={'general': [], 'format': [], 'exclusion': []},
                        deduktive_kategorien={}
                    )
                    st.session_state.codebook_modified = False
                    st.session_state.codebook_loaded_from = "none"
                    st.session_state.show_codebook_remove_dialog = False
                    st.success("✅ Codebook entfernt")
                    st.rerun()
            
            with col_remove2:
                if st.button("Abbrechen", use_container_width=True, key='codebook_remove_cancel'):
                    st.session_state.show_codebook_remove_dialog = False
                    st.rerun()
    
    # Load dialog
    if st.session_state.get('show_codebook_load_dialog', False):
        with st.expander("📂 Codebook laden", expanded=True):
            # Use detected format if available, otherwise default to json
            default_format = st.session_state.get('codebook_load_format_detected', 'json')
            format_index = 0 if default_format == 'json' else 1
            
            load_format = st.radio(
                "Format wählen:",
                options=['json', 'xlsx'],
                format_func=lambda x: 'JSON' if x == 'json' else 'Excel (XLSX)',
                horizontal=True,
                index=format_index,
                key='codebook_load_format'
            )
            
            # Clear detected format after using it
            if 'codebook_load_format_detected' in st.session_state:
                del st.session_state.codebook_load_format_detected
            
            # Get project manager for default path
            project_manager = st.session_state.project_manager
            default_filename = f"QCA-AID-Codebook.{load_format}"
            default_path = project_manager.get_codebook_path(default_filename)
            
            # Display current file path with default (Requirements 3.1, 3.2)
            col_path, col_browse = st.columns([4, 1])
            
            with col_path:
                # Initialize session state for file path if not exists
                if 'codebook_load_path_input' not in st.session_state:
                    st.session_state.codebook_load_path_input = str(default_path)
                
                # Requirement 8.3: Truncate long paths with ellipsis
                help_text = "Dateipfad zum Laden (Standard wird angezeigt)"
                current_path = st.session_state.codebook_load_path_input
                if current_path and len(current_path) > 50:
                    # Show full path in tooltip
                    help_text = f"Vollständiger Pfad: {current_path}\n\n{help_text}"
                
                # Use session state key directly for two-way binding
                file_path = st.text_input(
                    "Dateipfad:",
                    value=st.session_state.codebook_load_path_input,
                    help=help_text,
                    key='codebook_load_path_widget',
                    placeholder=str(default_path)
                )
                
                # Update session state when user types
                if file_path != st.session_state.codebook_load_path_input:
                    st.session_state.codebook_load_path_input = file_path
                
                # Requirement 8.4: Real-time path validation
                if file_path and file_path.strip():
                    path_resolver = project_manager.path_resolver
                    show_real_time_path_validation(file_path, path_resolver, check_writable=False)
            
            with col_browse:
                # File browser button (Requirement 3.3, 8.1, 8.2)
                st.markdown("<br>", unsafe_allow_html=True)  # Align with input
                if st.button("📁", key='codebook_load_browse_btn', help="Datei durchsuchen"):
                    # Open file browser dialog
                    selected_path = FileBrowserService.open_codebook_file_dialog(
                        initial_dir=project_manager.get_root_directory()
                    )
                    
                    if selected_path:
                        # Update path input (Requirement 3.4)
                        st.session_state.codebook_load_path_input = str(selected_path)
                        # Store detected format in separate key to avoid widget conflict
                        if selected_path.suffix.lower() == '.json':
                            st.session_state.codebook_load_format_detected = 'json'
                        elif selected_path.suffix.lower() == '.xlsx':
                            st.session_state.codebook_load_format_detected = 'xlsx'
                        st.rerun()
            
            col_load1, col_load2 = st.columns(2)
            
            with col_load1:
                if st.button("Laden", use_container_width=True, key='codebook_load_btn'):
                    from webapp_logic.codebook_manager import CodebookManager
                    manager = CodebookManager(project_manager.get_root_directory())
                    
                    # Use file path from input or default
                    load_path = file_path.strip() if file_path.strip() else str(default_path)
                    
                    # Load codebook
                    success, codebook_data, errors = manager.load_codebook(
                        file_path=load_path,
                        format=load_format
                    )
                    
                    if success:
                        st.session_state.codebook_data = codebook_data
                        st.session_state.codebook_modified = False
                        project_manager.update_last_codebook_file(Path(load_path))
                        
                        # Requirement 8.5: Success confirmation with file info
                        additional_info = {
                            "Kategorien": len(codebook_data.deduktive_kategorien),
                            "Kodierregeln": sum(len(rules) for rules in codebook_data.kodierregeln.values())
                        }
                        show_file_operation_success("geladen", Path(load_path), additional_info)
                        
                        st.session_state.show_codebook_load_dialog = False
                        st.rerun()
                    else:
                        # Enhanced error display
                        show_file_operation_error(
                            "Laden",
                            Path(load_path),
                            "\n".join(errors),
                            ["Überprüfen Sie das Dateiformat", "Stellen Sie sicher, dass die Datei existiert"]
                        )
            
            with col_load2:
                if st.button("Abbrechen", use_container_width=True, key='codebook_load_cancel'):
                    st.session_state.show_codebook_load_dialog = False
                    st.rerun()
    
    # Save dialog
    if st.session_state.get('show_codebook_save_dialog', False):
        with st.expander("💾 Codebook speichern", expanded=True):
            save_format = st.radio(
                "Format wählen:",
                options=['json', 'xlsx'],
                format_func=lambda x: 'JSON (empfohlen)' if x == 'json' else 'Excel (XLSX)',
                horizontal=True,
                help="JSON ist das empfohlene Format",
                key='codebook_save_format'
            )
            
            # Get project manager for default path
            project_manager = st.session_state.project_manager
            default_filename = f"QCA-AID-Codebook.{save_format}"
            default_path = project_manager.get_codebook_path(default_filename)
            
            # Display current file path with default (Requirements 3.1, 3.2)
            col_path, col_browse = st.columns([4, 1])
            
            with col_path:
                # Initialize session state for save path if not exists
                if 'codebook_save_path_input' not in st.session_state:
                    st.session_state.codebook_save_path_input = str(default_path)
                
                # Requirement 8.3: Truncate long paths with ellipsis
                help_text = "Dateipfad zum Speichern (Standard wird angezeigt)"
                current_path = st.session_state.codebook_save_path_input
                if current_path and len(current_path) > 50:
                    # Show full path in tooltip
                    help_text = f"Vollständiger Pfad: {current_path}\n\n{help_text}"
                
                # Use session state key directly for two-way binding
                file_path = st.text_input(
                    "Dateipfad:",
                    value=st.session_state.codebook_save_path_input,
                    help=help_text,
                    key='codebook_save_path_widget',
                    placeholder=str(default_path)
                )
                
                # Update session state when user types
                if file_path != st.session_state.codebook_save_path_input:
                    st.session_state.codebook_save_path_input = file_path
                
                # Requirement 8.4: Real-time path validation
                if file_path and file_path.strip():
                    path_resolver = project_manager.path_resolver
                    show_real_time_path_validation(file_path, path_resolver, check_writable=True)
            
            with col_browse:
                # File browser button for save (Requirement 3.5, 8.1, 8.2)
                st.markdown("<br>", unsafe_allow_html=True)  # Align with input
                if st.button("📁", key='codebook_save_browse_btn', help="Speicherort wählen"):
                    # Open save dialog with suggested filename
                    selected_path = FileBrowserService.save_codebook_file_dialog(
                        initial_dir=project_manager.get_root_directory(),
                        default_filename=default_filename
                    )
                    
                    if selected_path:
                        # Update path input (Requirement 3.4)
                        st.session_state.codebook_save_path_input = str(selected_path)
                        st.rerun()
            
            col_save1, col_save2 = st.columns(2)
            
            with col_save1:
                if st.button("Speichern", use_container_width=True, key='codebook_save_btn'):
                    from webapp_logic.codebook_manager import CodebookManager
                    manager = CodebookManager(project_manager.get_root_directory())
                    codebook = st.session_state.codebook_data
                    
                    # Use file path from input or default
                    save_path = file_path.strip() if file_path.strip() else str(default_path)
                    
                    # Save codebook
                    success, errors = manager.save_codebook(
                        codebook=codebook,
                        file_path=save_path,
                        format=save_format
                    )
                    
                    if success:
                        st.session_state.codebook_modified = False
                        project_manager.update_last_codebook_file(Path(save_path))
                        
                        # Requirement 8.5: Success confirmation with file info
                        additional_info = {
                            "Kategorien": len(codebook.deduktive_kategorien),
                            "Kodierregeln": sum(len(rules) for rules in codebook.kodierregeln.values())
                        }
                        show_file_operation_success("gespeichert", Path(save_path), additional_info)
                        
                        st.session_state.show_codebook_save_dialog = False
                        st.rerun()
                    else:
                        # Enhanced error display
                        show_file_operation_error(
                            "Speichern",
                            Path(save_path),
                            "\n".join(errors),
                            ["Überprüfen Sie die Schreibrechte", "Stellen Sie sicher, dass das Verzeichnis existiert"]
                        )
            
            with col_save2:
                if st.button("Abbrechen", use_container_width=True, key='codebook_save_cancel'):
                    st.session_state.show_codebook_save_dialog = False
                    st.rerun()
    
    # Inductive code import dialog (Requirement 4.4)
    if st.session_state.get('show_inductive_import_dialog', False):
        render_inductive_import_dialog()


def render_research_question():
    """
    Rendert Forschungsfrage-Editor.
    
    Requirement 5.1: WHEN der Codebook-Reiter angezeigt wird 
                    THEN das System SHALL ein Textfeld für die Forschungsfrage anzeigen
    Requirement 5.5: WHEN die Forschungsfrage leer ist 
                    THEN das System SHALL eine Warnung anzeigen
    """
    codebook = st.session_state.codebook_data
    
    new_question = st.text_area(
        "Forschungsfrage",
        value=codebook.forschungsfrage,
        height=100,
        help="Formulieren Sie Ihre zentrale Forschungsfrage",
        placeholder="z.B. Wie beeinflussen soziale Medien das politische Engagement junger Erwachsener?"
    )
    
    if new_question != codebook.forschungsfrage:
        codebook.forschungsfrage = new_question
        st.session_state.codebook_modified = True
    
    # Show warning if empty
    if not new_question.strip():
        st.warning("⚠️ Bitte geben Sie eine Forschungsfrage ein")


def render_coding_rules():
    """
    Rendert Kodierregeln-Editor mit drei Textbereichen.
    
    Requirement 5.2: WHEN der Codebook-Reiter angezeigt wird 
                    THEN das System SHALL separate Textbereiche für allgemeine Kodierregeln, 
                    Formatregeln und Ausschlussregeln anzeigen
    Requirement 5.3: WHEN ein Benutzer Regeln eingibt 
                    THEN das System SHALL jede Zeile als separate Regel behandeln
    Requirement 5.4: WHEN Regeln gespeichert werden 
                    THEN das System SHALL leere Zeilen automatisch entfernen
    """
    codebook = st.session_state.codebook_data
    
    # Ensure kodierregeln structure exists
    if not isinstance(codebook.kodierregeln, dict):
        codebook.kodierregeln = {'general': [], 'format': [], 'exclusion': []}
    
    # General rules
    st.markdown("**Allgemeine Kodierregeln**")
    general_text = '\n'.join(codebook.kodierregeln.get('general', []))
    new_general = st.text_area(
        "Allgemeine Regeln",
        value=general_text,
        height=100,
        help="Eine Regel pro Zeile. Leere Zeilen werden automatisch entfernt.",
        placeholder="z.B.\n- Kodiere nur explizite Aussagen\n- Berücksichtige den Kontext",
        label_visibility="collapsed"
    )
    
    # Parse rules (one per line, remove empty lines)
    new_general_rules = [line.strip() for line in new_general.split('\n') if line.strip()]
    if new_general_rules != codebook.kodierregeln.get('general', []):
        codebook.kodierregeln['general'] = new_general_rules
        st.session_state.codebook_modified = True
    
    # Format rules
    st.markdown("**Formatregeln**")
    format_text = '\n'.join(codebook.kodierregeln.get('format', []))
    new_format = st.text_area(
        "Formatregeln",
        value=format_text,
        height=100,
        help="Eine Regel pro Zeile. Leere Zeilen werden automatisch entfernt.",
        placeholder="z.B.\n- Verwende vollständige Sätze\n- Zitiere wörtlich",
        label_visibility="collapsed"
    )
    
    new_format_rules = [line.strip() for line in new_format.split('\n') if line.strip()]
    if new_format_rules != codebook.kodierregeln.get('format', []):
        codebook.kodierregeln['format'] = new_format_rules
        st.session_state.codebook_modified = True
    
    # Exclusion rules
    st.markdown("**Ausschlussregeln**")
    exclusion_text = '\n'.join(codebook.kodierregeln.get('exclusion', []))
    new_exclusion = st.text_area(
        "Ausschlussregeln",
        value=exclusion_text,
        height=100,
        help="Eine Regel pro Zeile. Leere Zeilen werden automatisch entfernt.",
        placeholder="z.B.\n- Ignoriere Werbung\n- Schließe Metadaten aus",
        label_visibility="collapsed"
    )
    
    new_exclusion_rules = [line.strip() for line in new_exclusion.split('\n') if line.strip()]
    if new_exclusion_rules != codebook.kodierregeln.get('exclusion', []):
        codebook.kodierregeln['exclusion'] = new_exclusion_rules
        st.session_state.codebook_modified = True


def render_categories():
    """
    Rendert Kategorien-Editor mit Expand/Collapse.
    
    Requirement 4.1: WHEN der Codebook-Reiter angezeigt wird 
                    THEN das System SHALL eine Liste aller Hauptkategorien mit Expand/Collapse-Funktionalität anzeigen
    Requirement 4.3: WHEN der Codebook-Reiter angezeigt wird 
                    THEN das System SHALL eine Schaltfläche zum Hinzufügen neuer Hauptkategorien anzeigen
    Requirement 5.1: Display imported codes in separate section
    """
    codebook = st.session_state.codebook_data
    
    # Separate deductive and inductive categories (Requirement 5.1)
    from webapp_models.inductive_code_data import InductiveCodeData
    
    deductive_categories = {}
    inductive_categories = {}
    
    for cat_name, category in codebook.deduktive_kategorien.items():
        if isinstance(category, InductiveCodeData) and category.is_inductive:
            inductive_categories[cat_name] = category
        else:
            deductive_categories[cat_name] = category
    
    # Display deductive categories section
    if deductive_categories or not inductive_categories:
        st.markdown("### 📚 Deduktive Kategorien")
        
        if deductive_categories:
            for cat_name in list(deductive_categories.keys()):
                category = deductive_categories[cat_name]
                
                with st.expander(f"📁 {cat_name}", expanded=False):
                    render_category_editor(cat_name, category)
        else:
            st.info("ℹ️ Noch keine deduktiven Kategorien vorhanden. Fügen Sie eine neue Kategorie hinzu.")
    
    # Display inductive categories section (Requirement 5.1)
    if inductive_categories:
        st.markdown("---")
        st.markdown("### 🔬 Induktive Kategorien")
        st.caption("Diese Kategorien wurden aus vorherigen Analysen importiert")
        
        for cat_name in list(inductive_categories.keys()):
            category = inductive_categories[cat_name]
            
            # Show source attribution in the expander title (Requirement 5.3)
            source_info = ""
            if hasattr(category, 'source_file') and category.source_file:
                source_info = f" (aus: {category.source_file})"
            
            with st.expander(f"🔬 {cat_name}{source_info}", expanded=False):
                # Show import metadata at the top
                if hasattr(category, 'import_date') and category.import_date:
                    st.caption(f"📅 Importiert: {category.import_date}")
                if hasattr(category, 'original_frequency') and category.original_frequency:
                    st.caption(f"📊 Häufigkeit in Analyse: {category.original_frequency}")
                
                st.markdown("---")
                
                # Render the category editor (Requirement 5.4: Edit operation parity)
                render_category_editor(cat_name, category)
    
    # Add new category button
    st.markdown("---")
    codebook = st.session_state.codebook_data
    button_type = "primary" if not codebook or len(codebook.deduktive_kategorien) == 0 else "secondary"
    button_help = "Erstellen Sie Ihre erste Kategorie" if not codebook or len(codebook.deduktive_kategorien) == 0 else "Weitere Kategorie hinzufügen"
    if st.button("➕ Neue Kategorie hinzufügen", use_container_width=True, key="add_category_main_btn", type=button_type, help=button_help):
        st.session_state.show_add_category_dialog = True
    
    # Add category dialog
    if st.session_state.get('show_add_category_dialog', False):
        with st.expander("➕ Neue Kategorie hinzufügen", expanded=True):
            render_add_category_form()


def render_category_editor(cat_name: str, category: CategoryData):
    """
    Rendert Editor für einzelne Kategorie.
    
    Requirement 4.2: WHEN eine Kategorie erweitert wird 
                    THEN das System SHALL Eingabefelder für Definition, Regeln und Beispiele anzeigen
    Requirement 4.4: WHEN eine Kategorie ausgewählt ist 
                    THEN das System SHALL die Verwaltung von Subkategorien mit Hinzufügen/Entfernen-Funktionen ermöglichen
    """
    codebook = st.session_state.codebook_data
    
    # Category name (read-only, shown as info)
    st.markdown(f"**Kategoriename:** {cat_name}")
    st.caption(f"Erstellt: {category.added_date} | Geändert: {category.modified_date}")
    
    # Definition
    new_definition = st.text_area(
        "Definition",
        value=category.definition,
        height=100,
        help="Beschreibung der Kategorie",
        key=f"cat_def_{cat_name}"
    )
    
    if new_definition != category.definition:
        category.definition = new_definition
        category.modified_date = datetime.now().strftime("%Y-%m-%d")
        st.session_state.codebook_modified = True
    
    # Show info about definition length (not a warning)
    word_count = len(new_definition.split())
    if word_count > 0 and word_count < 10:
        st.info(f"ℹ️ Definition: {word_count} Wörter (empfohlen: 10+)")
    
    # Rules
    st.markdown("**Regeln**")
    rules_text = '\n'.join(category.rules)
    new_rules_text = st.text_area(
        "Regeln (eine pro Zeile)",
        value=rules_text,
        height=80,
        key=f"cat_rules_{cat_name}",
        label_visibility="collapsed"
    )
    
    new_rules = [line.strip() for line in new_rules_text.split('\n') if line.strip()]
    if new_rules != category.rules:
        category.rules = new_rules
        category.modified_date = datetime.now().strftime("%Y-%m-%d")
        st.session_state.codebook_modified = True
    
    # Examples
    st.markdown("**Beispiele** (optional, empfohlen: 2+)")
    examples_text = '\n'.join(category.examples)
    new_examples_text = st.text_area(
        "Beispiele (eines pro Zeile)",
        value=examples_text,
        height=80,
        key=f"cat_examples_{cat_name}",
        label_visibility="collapsed"
    )
    
    new_examples = [line.strip() for line in new_examples_text.split('\n') if line.strip()]
    if new_examples != category.examples:
        category.examples = new_examples
        category.modified_date = datetime.now().strftime("%Y-%m-%d")
        st.session_state.codebook_modified = True
    
    # Show info about examples (not a warning)
    if len(new_examples) == 1:
        st.info(f"ℹ️ {len(new_examples)} Beispiel vorhanden (empfohlen: 2+)")
    
    # Subcategories
    st.markdown("**Subkategorien** (optional, empfohlen: 2+)")
    
    # Show info about subcategories (not a warning)
    if len(category.subcategories) == 1:
        st.info(f"ℹ️ {len(category.subcategories)} Subkategorie vorhanden (empfohlen: 2+)")
    
    # Display existing subcategories
    subcats_to_remove = []
    for subcat_key, subcat_label in list(category.subcategories.items()):
        col1, col2, col3 = st.columns([2, 3, 1])
        
        with col1:
            new_key = st.text_input(
                "Schlüssel",
                value=subcat_key,
                key=f"subcat_key_{cat_name}_{subcat_key}",
                label_visibility="collapsed"
            )
        
        with col2:
            new_label = st.text_input(
                "Bezeichnung",
                value=subcat_label,
                key=f"subcat_label_{cat_name}_{subcat_key}",
                label_visibility="collapsed"
            )
            
            # Update if changed
            if new_label != subcat_label:
                category.subcategories[subcat_key] = new_label
                category.modified_date = datetime.now().strftime("%Y-%m-%d")
                st.session_state.codebook_modified = True
        
        with col3:
            if st.button("🗑️", key=f"remove_subcat_{cat_name}_{subcat_key}", help="Subkategorie entfernen"):
                subcats_to_remove.append(subcat_key)
        
        # Handle key change
        if new_key != subcat_key and new_key not in category.subcategories:
            category.subcategories[new_key] = category.subcategories.pop(subcat_key)
            category.modified_date = datetime.now().strftime("%Y-%m-%d")
            st.session_state.codebook_modified = True
            st.rerun()
    
    # Process removals
    for key in subcats_to_remove:
        if key in category.subcategories:
            del category.subcategories[key]
            category.modified_date = datetime.now().strftime("%Y-%m-%d")
            st.session_state.codebook_modified = True
            st.rerun()
    
    # Add new subcategory
    with st.expander("➕ Subkategorie hinzufügen"):
        col1, col2 = st.columns(2)
        
        with col1:
            new_subcat_key = st.text_input(
                "Neuer Schlüssel",
                value="",
                key=f"new_subcat_key_{cat_name}",
                help="z.B. 'positiv', 'negativ'"
            )
        
        with col2:
            new_subcat_label = st.text_input(
                "Neue Bezeichnung",
                value="",
                key=f"new_subcat_label_{cat_name}",
                help="z.B. 'Positiv', 'Negativ'"
            )
        
        if st.button("Subkategorie hinzufügen", key=f"add_subcat_btn_{cat_name}"):
            if new_subcat_key and new_subcat_label:
                if new_subcat_key not in category.subcategories:
                    category.subcategories[new_subcat_key] = new_subcat_label
                    category.modified_date = datetime.now().strftime("%Y-%m-%d")
                    st.session_state.codebook_modified = True
                    st.success(f"✅ Subkategorie '{new_subcat_key}' hinzugefügt")
                    st.rerun()
                else:
                    st.error(f"❌ Subkategorie '{new_subcat_key}' existiert bereits")
            else:
                st.error("❌ Bitte beide Felder ausfüllen")
    
    # Remove category button
    st.markdown("---")
    if st.button(f"🗑️ Kategorie '{cat_name}' entfernen", key=f"remove_cat_{cat_name}", type="secondary"):
        if len(codebook.deduktive_kategorien) > 1:
            del codebook.deduktive_kategorien[cat_name]
            st.session_state.codebook_modified = True
            st.success(f"✅ Kategorie '{cat_name}' entfernt")
            st.rerun()
        else:
            st.error("❌ Mindestens eine Kategorie muss vorhanden sein")


def render_add_category_form():
    """
    Rendert Formular zum Hinzufügen einer neuen Kategorie.
    """
    codebook = st.session_state.codebook_data
    
    # Category name
    new_cat_name = st.text_input(
        "Kategoriename",
        value="",
        help="3-50 Zeichen",
        key="new_cat_name"
    )
    
    # Definition
    new_cat_def = st.text_area(
        "Definition",
        value="",
        height=100,
        help="Beschreibung der Kategorie",
        key="new_cat_def"
    )
    
    # Rules
    new_cat_rules = st.text_area(
        "Regeln (eine pro Zeile, optional)",
        value="",
        height=80,
        key="new_cat_rules"
    )
    
    # Examples
    new_cat_examples = st.text_area(
        "Beispiele (eines pro Zeile, optional)",
        value="",
        height=80,
        key="new_cat_examples"
    )
    
    # Subcategories
    st.markdown("**Subkategorien** (optional)")
    st.caption("Format: schlüssel:Bezeichnung (eine pro Zeile)")
    new_cat_subcats = st.text_area(
        "Subkategorien",
        value="",
        height=80,
        placeholder="z.B.\npositiv:Positiv\nnegativ:Negativ\nneutral:Neutral",
        key="new_cat_subcats",
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Kategorie hinzufügen", use_container_width=True, key="add_cat_btn"):
            # Validate inputs
            errors = []
            
            if not new_cat_name.strip():
                errors.append("Kategoriename darf nicht leer sein")
            elif len(new_cat_name.strip()) < 3:
                errors.append("Kategoriename muss mindestens 3 Zeichen haben")
            elif new_cat_name in codebook.deduktive_kategorien:
                errors.append(f"Kategorie '{new_cat_name}' existiert bereits")
            
            if not new_cat_def.strip():
                errors.append("Definition darf nicht leer sein")
            
            # Parse rules (optional)
            rules = [line.strip() for line in new_cat_rules.split('\n') if line.strip()]
            
            # Parse examples (optional)
            examples = [line.strip() for line in new_cat_examples.split('\n') if line.strip()]
            
            # Parse subcategories (optional)
            subcats = {}
            for line in new_cat_subcats.split('\n'):
                line = line.strip()
                if line and ':' in line:
                    key, label = line.split(':', 1)
                    subcats[key.strip()] = label.strip()
            
            if errors:
                for error in errors:
                    st.error(f"❌ {error}")
            else:
                # Create new category
                new_category = CategoryData(
                    name=new_cat_name.strip(),
                    definition=new_cat_def.strip(),
                    rules=rules,
                    examples=examples,
                    subcategories=subcats,
                    added_date=datetime.now().strftime("%Y-%m-%d"),
                    modified_date=datetime.now().strftime("%Y-%m-%d")
                )
                
                # Add to codebook
                codebook.deduktive_kategorien[new_cat_name.strip()] = new_category
                st.session_state.codebook_modified = True
                st.session_state.show_add_category_dialog = False
                st.success(f"✅ Kategorie '{new_cat_name}' hinzugefügt")
                st.rerun()
    
    with col2:
        if st.button("Abbrechen", use_container_width=True, key="add_cat_cancel"):
            st.session_state.show_add_category_dialog = False
            st.rerun()


def render_json_preview():
    """
    Rendert JSON-Vorschau mit Syntax-Highlighting.
    
    Requirement 4.5: WHEN Änderungen am Codebook vorgenommen werden 
                    THEN das System SHALL eine Vorschau der JSON-Struktur anzeigen
    """
    codebook = st.session_state.codebook_data
    
    # Generate JSON preview
    from webapp_logic.codebook_manager import CodebookManager
    manager = CodebookManager()
    manager.codebook = codebook
    
    json_preview = manager.to_json_preview()
    
    # Display with syntax highlighting
    st.code(json_preview, language='json', line_numbers=False)
    
    # Show size info
    json_size = len(json_preview.encode('utf-8'))
    st.caption(f"Größe: {json_size:,} Bytes")


def render_validation_status():
    """
    Zeigt Validierungsstatus des Codebooks an.
    """
    codebook = st.session_state.codebook_data
    
    # Validate codebook
    is_valid, errors = codebook.validate()
    
    if is_valid:
        st.success("✅ Codebook ist gültig")
    else:
        st.error("❌ Codebook enthält Fehler:")
        for error in errors:
            st.error(f"  • {error}")
    
    # Show modification status
    if st.session_state.codebook_modified:
        st.warning("⚠️ Ungespeicherte Änderungen vorhanden")
    
    # Show statistics
    st.markdown("**Statistik:**")
    st.text(f"Kategorien: {len(codebook.deduktive_kategorien)}")
    
    total_subcats = sum(len(cat.subcategories) for cat in codebook.deduktive_kategorien.values())
    st.text(f"Subkategorien: {total_subcats}")
    
    total_rules = (
        len(codebook.kodierregeln.get('general', [])) +
        len(codebook.kodierregeln.get('format', [])) +
        len(codebook.kodierregeln.get('exclusion', []))
    )
    st.text(f"Kodierregeln: {total_rules}")



def check_inductive_codes_available():
    """
    Check for available inductive codes in output directory.
    
    Requirements:
    - 4.1: Scan output directory for analysis files
    - 4.2: Display notification when inductive codes found
    """
    # Only check once per session or when explicitly requested
    if 'inductive_codes_checked' not in st.session_state:
        st.session_state.inductive_codes_checked = True
        
        # Get output directory from project manager
        project_manager = st.session_state.project_manager
        output_dir = project_manager.get_output_dir()
        
        # Check if output directory exists
        if not output_dir.exists():
            st.session_state.inductive_codes_available = False
            return
        
        # Create extractor and check for inductive codes
        extractor = InductiveCodeExtractor(output_dir)
        has_codes = extractor.has_inductive_codes_available()
        
        st.session_state.inductive_codes_available = has_codes
        
        # Display notification if codes found (Requirement 4.2)
        if has_codes:
            st.info("ℹ️ **Induktive Codes verfügbar!** Klicken Sie auf 'Import' um Codes aus vorherigen Analysen zu importieren.")


def render_inductive_import_dialog():
    """
    Render dialog for importing inductive codes.
    
    Requirements:
    - 4.4: Display available analysis files
    - 4.5: Extract codes from selected file
    """
    with st.expander("📥 Induktive Codes importieren", expanded=True):
        st.markdown("**Verfügbare Analyse-Dateien**")
        
        # Get output directory
        project_manager = st.session_state.project_manager
        output_dir = project_manager.get_output_dir()
        
        # Create extractor
        extractor = InductiveCodeExtractor(output_dir)
        
        # Check for JSON codebook first (primary source)
        if extractor.has_inductive_codebook():
            st.success("✅ Induktives Codebook gefunden: `codebook_inductive.json`")
            
            # Load and show preview of codes (Requirement 4.4)
            inductive_codes_preview = extractor.load_inductive_codebook()
            
            if inductive_codes_preview:
                # Display metadata (Requirement 4.4)
                st.markdown("**Codebook-Informationen:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Induktive Kategorien", len(inductive_codes_preview))
                
                with col2:
                    total_subcats = sum(len(code.subcategories) for code in inductive_codes_preview.values())
                    st.metric("Subkategorien", total_subcats)
                
                with col3:
                    total_examples = sum(len(code.examples) for code in inductive_codes_preview.values())
                    st.metric("Beispiele", total_examples)
                
                # Show preview of first few codes
                st.markdown("**Vorschau (erste 3 Codes):**")
                for idx, (code_name, code_data) in enumerate(list(inductive_codes_preview.items())[:3]):
                    st.caption(f"• **{code_name}**: {code_data.definition[:80]}{'...' if len(code_data.definition) > 80 else ''}")
                
                if len(inductive_codes_preview) > 3:
                    st.caption(f"... und {len(inductive_codes_preview) - 3} weitere")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Codes importieren", use_container_width=True, key='import_from_json_btn'):
                    # Extract codes from JSON
                    inductive_codes = extractor.load_inductive_codebook()
                    
                    if inductive_codes:
                        # Store in session state for merge
                        st.session_state.inductive_codes_to_import = inductive_codes
                        st.session_state.inductive_import_source = "codebook_inductive.json"
                        st.session_state.show_inductive_import_dialog = False
                        st.session_state.show_inductive_merge_dialog = True
                        st.rerun()
                    else:
                        st.error("❌ Keine induktiven Codes im Codebook gefunden")
            
            with col2:
                if st.button("Abbrechen", use_container_width=True, key='import_cancel_json_btn'):
                    st.session_state.show_inductive_import_dialog = False
                    st.rerun()
        
        else:
            # Fallback to XLSX files
            analysis_files = extractor.find_analysis_files()
            
            if not analysis_files:
                st.warning("⚠️ Keine Analyse-Dateien im Output-Verzeichnis gefunden")
                
                if st.button("Schließen", use_container_width=True, key='import_close_btn'):
                    st.session_state.show_inductive_import_dialog = False
                    st.rerun()
                return
            
            # Filter files that have inductive codes
            files_with_codes = []
            for file_path in analysis_files:
                if extractor.has_inductive_codes(file_path):
                    metadata = extractor.get_analysis_metadata(file_path)
                    files_with_codes.append((file_path, metadata))
            
            if not files_with_codes:
                st.warning("⚠️ Keine induktiven Codes in den Analyse-Dateien gefunden")
                
                if st.button("Schließen", use_container_width=True, key='import_close_no_codes_btn'):
                    st.session_state.show_inductive_import_dialog = False
                    st.rerun()
                return
            
            # Display files with metadata
            st.markdown(f"Gefunden: **{len(files_with_codes)}** Datei(en) mit induktiven Codes")
            
            # Let user select a file
            selected_file_index = st.selectbox(
                "Datei auswählen:",
                range(len(files_with_codes)),
                format_func=lambda i: files_with_codes[i][0].name,
                key='inductive_file_select'
            )
            
            if selected_file_index is not None:
                selected_file, metadata = files_with_codes[selected_file_index]
                
                # Display metadata (Requirement 4.4)
                st.markdown("**Datei-Informationen:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Dokumente", metadata.get('document_count', 0))
                
                with col2:
                    st.metric("Kategorien", metadata.get('inductive_category_count', 0))
                
                with col3:
                    st.metric("Kodierungen", metadata.get('total_codings', 0))
                
                st.caption(f"Geändert: {metadata.get('modified_date', 'Unbekannt')}")
                
                # Show preview of codes from this file (Requirement 4.5)
                st.markdown("**Vorschau der Codes:**")
                preview_codes = extractor.extract_inductive_codes_from_xlsx(selected_file)
                
                if preview_codes:
                    for idx, (code_name, code_data) in enumerate(list(preview_codes.items())[:3]):
                        freq_info = f" ({code_data.original_frequency}x)" if code_data.original_frequency else ""
                        st.caption(f"• **{code_name}**{freq_info}: {code_data.definition[:80]}{'...' if len(code_data.definition) > 80 else ''}")
                    
                    if len(preview_codes) > 3:
                        st.caption(f"... und {len(preview_codes) - 3} weitere")
                else:
                    st.warning("⚠️ Keine Codes in Vorschau verfügbar")
                
                # Import button
                col_import1, col_import2 = st.columns(2)
                
                with col_import1:
                    if st.button("Codes importieren", use_container_width=True, key='import_from_xlsx_btn'):
                        # Extract codes from XLSX
                        inductive_codes = extractor.extract_inductive_codes_from_xlsx(selected_file)
                        
                        if inductive_codes:
                            # Store in session state for merge
                            st.session_state.inductive_codes_to_import = inductive_codes
                            st.session_state.inductive_import_source = selected_file.name
                            st.session_state.show_inductive_import_dialog = False
                            st.session_state.show_inductive_merge_dialog = True
                            st.rerun()
                        else:
                            st.error("❌ Keine induktiven Codes in der Datei gefunden")
                
                with col_import2:
                    if st.button("Abbrechen", use_container_width=True, key='import_cancel_xlsx_btn'):
                        st.session_state.show_inductive_import_dialog = False
                        st.rerun()


def render_inductive_merge_dialog():
    """
    Render dialog for merging imported inductive codes with existing codebook.
    
    This is called from the main render function after codes are selected for import.
    
    Requirements:
    - 4.4: Display file metadata
    - 4.5: Show preview of codes to be imported
    - 5.1: Display imported codes in separate section
    - 5.2: Implement conflict detection and warning display
    - 5.3: Mark codes with source file attribution
    """
    if not st.session_state.get('show_inductive_merge_dialog', False):
        return
    
    with st.expander("🔀 Induktive Codes zusammenführen", expanded=True):
        inductive_codes = st.session_state.get('inductive_codes_to_import', {})
        source_file = st.session_state.get('inductive_import_source', 'Unbekannt')
        
        st.markdown(f"**Quelle:** `{source_file}`")
        st.markdown(f"**Anzahl Codes:** {len(inductive_codes)}")
        
        # Check for conflicts
        from webapp_logic.code_merger import CodeMerger
        codebook = st.session_state.codebook_data
        merger = CodeMerger()
        
        conflicts = merger.detect_conflicts(
            codebook.deduktive_kategorien,
            {name: code for name, code in inductive_codes.items()}
        )
        
        # Initialize rename map in session state if not exists
        if 'inductive_rename_map' not in st.session_state:
            st.session_state.inductive_rename_map = {}
        
        # Display conflicts with rename options (Requirement 5.2)
        if conflicts:
            st.warning(f"⚠️ **{len(conflicts)} Namenskonflikte gefunden:**")
            st.markdown("**Konfliktlösung:** Benennen Sie die konfliktierenden Kategorien um oder überspringen Sie sie.")
            
            for conflict_name, conflict_type in conflicts:
                with st.container():
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.error(f"**Konflikt:** '{conflict_name}'")
                        st.caption(f"Typ: {conflict_type}")
                    
                    with col2:
                        # Suggest alternative name
                        existing_names = set(codebook.deduktive_kategorien.keys())
                        suggested_name = merger.suggest_rename(conflict_name, existing_names)
                        
                        # Allow user to edit the suggested name
                        new_name = st.text_input(
                            "Neuer Name:",
                            value=st.session_state.inductive_rename_map.get(conflict_name, suggested_name),
                            key=f"rename_{conflict_name}",
                            help="Geben Sie einen neuen Namen ein oder übernehmen Sie den Vorschlag"
                        )
                        
                        # Store in rename map
                        if new_name and new_name != conflict_name:
                            st.session_state.inductive_rename_map[conflict_name] = new_name
                    
                    with col3:
                        # Option to skip this code
                        skip = st.checkbox(
                            "Überspringen",
                            key=f"skip_{conflict_name}",
                            help="Diese Kategorie nicht importieren"
                        )
                        
                        if skip and conflict_name in st.session_state.inductive_rename_map:
                            del st.session_state.inductive_rename_map[conflict_name]
            
            st.markdown("---")
        
        # Preview codes to be imported (Requirement 4.5)
        st.markdown("**Vorschau der zu importierenden Codes:**")
        
        # Show expandable preview for each code
        preview_count = min(len(inductive_codes), 10)  # Show up to 10
        
        for idx, (code_name, code_data) in enumerate(list(inductive_codes.items())[:preview_count]):
            # Check if this code will be renamed or skipped
            display_name = code_name
            status_icon = "📥"
            status_text = ""
            
            if code_name in [c[0] for c in conflicts]:
                if st.session_state.get(f"skip_{code_name}", False):
                    status_icon = "⏭️"
                    status_text = " (wird übersprungen)"
                elif code_name in st.session_state.inductive_rename_map:
                    new_name = st.session_state.inductive_rename_map[code_name]
                    status_icon = "✏️"
                    status_text = f" → '{new_name}'"
                    display_name = new_name
                else:
                    status_icon = "⚠️"
                    status_text = " (Konflikt)"
            
            with st.expander(f"{status_icon} **{code_name}**{status_text}", expanded=False):
                st.markdown(f"**Definition:** {code_data.definition}")
                
                if code_data.rules:
                    st.markdown(f"**Regeln:** {len(code_data.rules)}")
                    for rule in code_data.rules[:3]:
                        st.caption(f"  • {rule}")
                    if len(code_data.rules) > 3:
                        st.caption(f"  ... und {len(code_data.rules) - 3} weitere")
                
                if code_data.examples:
                    st.markdown(f"**Beispiele:** {len(code_data.examples)}")
                    for example in code_data.examples[:2]:
                        st.caption(f"  • {example[:100]}{'...' if len(example) > 100 else ''}")
                    if len(code_data.examples) > 2:
                        st.caption(f"  ... und {len(code_data.examples) - 2} weitere")
                
                if code_data.subcategories:
                    st.markdown(f"**Subkategorien:** {len(code_data.subcategories)}")
                    st.caption(", ".join(code_data.subcategories.keys()))
                
                # Show source attribution (Requirement 5.3)
                if hasattr(code_data, 'source_file') and code_data.source_file:
                    st.caption(f"📄 Quelle: {code_data.source_file}")
                if hasattr(code_data, 'original_frequency') and code_data.original_frequency:
                    st.caption(f"📊 Häufigkeit in Analyse: {code_data.original_frequency}")
        
        if len(inductive_codes) > preview_count:
            st.caption(f"... und {len(inductive_codes) - preview_count} weitere Codes")
        
        st.markdown("---")
        
        # Summary of import action
        codes_to_import = len(inductive_codes)
        codes_to_skip = sum(1 for c in conflicts if st.session_state.get(f"skip_{c[0]}", False))
        codes_to_rename = len(st.session_state.inductive_rename_map)
        codes_final = codes_to_import - codes_to_skip
        
        st.info(f"📊 **Import-Zusammenfassung:** {codes_final} von {codes_to_import} Codes werden importiert "
                f"({codes_to_rename} umbenannt, {codes_to_skip} übersprungen)")
        
        # Merge options
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Importieren", use_container_width=True, key='merge_confirm_btn', type="primary"):
                # Apply renames and filter skipped codes
                codes_to_merge = {}
                
                for code_name, code_data in inductive_codes.items():
                    # Check if skipped
                    if st.session_state.get(f"skip_{code_name}", False):
                        continue
                    
                    # Apply rename if exists
                    final_name = st.session_state.inductive_rename_map.get(code_name, code_name)
                    
                    # Update the code data with final name
                    if final_name != code_name:
                        code_data.name = final_name
                    
                    codes_to_merge[final_name] = code_data
                
                # Perform merge
                merged_codes, warnings = merger.merge_codes(
                    codebook.deduktive_kategorien,
                    codes_to_merge,
                    source_file
                )
                
                # Update codebook
                codebook.deduktive_kategorien = merged_codes
                st.session_state.codebook_modified = True
                
                # Clear import state
                st.session_state.show_inductive_merge_dialog = False
                st.session_state.inductive_codes_to_import = None
                st.session_state.inductive_import_source = None
                st.session_state.inductive_rename_map = {}
                
                # Show success message
                st.success(f"✅ {len(codes_to_merge)} induktive Codes erfolgreich importiert!")
                
                if warnings:
                    with st.expander("⚠️ Hinweise anzeigen", expanded=False):
                        for warning in warnings:
                            st.warning(f"  • {warning}")
                
                st.rerun()
        
        with col2:
            if st.button("❌ Abbrechen", use_container_width=True, key='merge_cancel_btn'):
                st.session_state.show_inductive_merge_dialog = False
                st.session_state.inductive_codes_to_import = None
                st.session_state.inductive_import_source = None
                st.session_state.inductive_rename_map = {}
                st.rerun()

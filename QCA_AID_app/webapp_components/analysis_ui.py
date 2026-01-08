"""
Analysis UI Component
=====================
Streamlit UI component for managing QCA-AID analysis execution.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 12.1, 12.2, 12.3, 12.4, 12.5
"""

import streamlit as st
from typing import List
import os
from pathlib import Path

from webapp_logic.file_manager import FileManager
from webapp_logic.analysis_runner import AnalysisRunner


def render_analysis_tab():
    """
    Rendert Analyse-Reiter als Hauptlayout.
    
    Requirements:
    - 8.1-8.5: Analyse-Steuerung und Fortschrittsanzeige
    - 12.1-12.5: Eingabedateiverwaltung
    """
    st.header("🔬 Analyse")
    st.markdown("Starten und überwachen Sie QCA-AID Analysen")
    
    # Comprehensive workflow readiness check
    config = st.session_state.config_data
    project_root = st.session_state.project_manager.get_root_directory()
    
    readiness_issues = []
    readiness_warnings = []
    
    # 1. Check config validity
    is_config_valid, config_errors = config.validate()
    if not is_config_valid:
        readiness_issues.append(f"Konfiguration ungültig: {', '.join(config_errors[:2])}")
    
    # 2. Check if config is saved
    config_manager = st.session_state.config_manager
    config_saved = config_manager.json_path.exists() or config_manager.xlsx_path.exists()
    if not config_saved:
        readiness_warnings.append("Konfiguration noch nicht gespeichert")
    
    # 3. Check codebook
    from webapp_logic.codebook_manager import CodebookManager
    codebook_manager = CodebookManager(project_root)
    success, codebook_data, errors = codebook_manager.load_codebook()
    
    if not success or not codebook_data:
        readiness_issues.append("Codebook konnte nicht geladen werden")
    elif len(codebook_data.deduktive_kategorien) == 0:
        readiness_issues.append("Codebook enthält keine Kategorien")
    else:
        # Check if categories have subcategories
        categories_without_subcats = [
            name for name, cat in codebook_data.deduktive_kategorien.items()
            if not cat.subcategories or len(cat.subcategories) == 0
        ]
        if categories_without_subcats:
            readiness_warnings.append(f"{len(categories_without_subcats)} Kategorie(n) ohne Subkategorien")
    
    # 4. Check input files
    input_dir = Path(project_root) / config.data_dir
    if not input_dir.exists():
        readiness_issues.append(f"Eingabeverzeichnis existiert nicht: {config.data_dir}")
    else:
        input_files = list(input_dir.glob('*.txt')) + list(input_dir.glob('*.pdf')) + list(input_dir.glob('*.docx'))
        if len(input_files) == 0:
            readiness_issues.append("Keine Eingabedateien (.txt, .pdf, .docx) gefunden")
        else:
            # Check file sizes
            empty_files = [f.name for f in input_files if f.stat().st_size == 0]
            if empty_files:
                readiness_warnings.append(f"{len(empty_files)} leere Datei(en) gefunden")
    
    # 5. Check output directory
    output_dir = Path(project_root) / config.output_dir
    if not output_dir.exists():
        readiness_warnings.append(f"Ausgabeverzeichnis wird erstellt: {config.output_dir}")
    
    # Display readiness status
    if readiness_issues:
        st.error(f"❌ **Nicht bereit zur Analyse:** {len(readiness_issues)} Problem(e) gefunden")
        for issue in readiness_issues:
            st.error(f"   • {issue}")
    elif readiness_warnings:
        st.warning(f"⚠️ **Bereit mit Einschränkungen:** {len(readiness_warnings)} Warnung(en)")
        for warning in readiness_warnings:
            st.warning(f"   • {warning}")
        st.info("💡 Sie können die Analyse starten, aber die Warnungen sollten überprüft werden.")
    else:
        st.success("✅ **Bereit zur Analyse:** Alle Voraussetzungen erfüllt")
        if codebook_data:
            st.info(f"📊 {len(codebook_data.deduktive_kategorien)} Kategorien • {len(input_files)} Eingabedatei(en)")
    
    # Initialize analysis runner in session state if not exists
    if 'analysis_runner' not in st.session_state:
        config = st.session_state.config_data
        project_manager = st.session_state.project_manager
        project_root = project_manager.get_root_directory()
        
        # Resolve output_dir relative to project root
        output_dir = config.output_dir
        if not Path(output_dir).is_absolute():
            output_dir = str(project_root / output_dir)
        
        st.session_state.analysis_runner = AnalysisRunner(output_dir=output_dir)
    
    # Get analysis runner
    runner = st.session_state.analysis_runner
    
    # Create two columns for layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Input files section
        st.subheader("📁 Eingabedateien")
        render_input_files()
        
        st.markdown("---")
        
        # Analysis controls
        st.subheader("▶️ Analyse-Steuerung")
        render_analysis_controls()
        
        st.markdown("---")
        
        # Analysis progress
        if runner.is_running() or runner.get_status().output_file:
            st.subheader("📊 Fortschritt")
            render_analysis_progress()
    
    with col2:
        # Results section
        st.subheader("📄 Ergebnisse")
        render_results()
    
    # Log streaming - OUTSIDE columns for full width
    if runner.is_running() or runner.get_logs():
        st.markdown("---")
        st.subheader("📋 Logs")
        render_log_stream()


def render_input_files():
    """
    Zeigt Eingabedateien mit Vorschau.
    
    Requirement 12.1: WHEN der Analyse-Reiter angezeigt wird 
                     THEN das System SHALL alle Dateien im INPUT_DIR auflisten
    Requirement 12.2: WHEN Eingabedateien angezeigt werden 
                     THEN das System SHALL Dateiname, Typ und Größe anzeigen
    Requirement 12.3: WHEN ein Benutzer eine Datei auswählt 
                     THEN das System SHALL eine Textvorschau der ersten 500 Zeichen anzeigen
    Requirement 12.4: WHEN keine Eingabedateien vorhanden sind 
                     THEN das System SHALL eine Meldung mit Upload-Hinweis anzeigen
    Requirement 12.5: WHEN der INPUT_DIR nicht existiert 
                     THEN das System SHALL den Ordner automatisch erstellen
    """
    config = st.session_state.config_data
    input_dir = config.data_dir
    
    # Get project root from project manager
    project_manager = st.session_state.project_manager
    project_root = project_manager.get_root_directory()
    
    # Initialize file manager with project root
    file_manager = FileManager(base_dir=str(project_root))
    
    # Resolve input_dir relative to project root if it's a relative path
    if not Path(input_dir).is_absolute():
        input_dir = str(project_root / input_dir)
    
    # Ensure input directory exists
    success, error_msg = file_manager.ensure_directory(input_dir)
    if not success:
        st.error(f"❌ Fehler beim Erstellen des Eingabeverzeichnisses: {error_msg}")
        return
    elif error_msg is None and not Path(input_dir).exists():
        # Directory was just created
        st.info(f"ℹ️ Eingabeverzeichnis wurde erstellt: {input_dir}")
    
    # List files in input directory
    try:
        files = file_manager.list_files(
            directory=input_dir,
            extensions=['.txt', '.pdf', '.docx', '.doc']
        )
    except Exception as e:
        st.error(f"❌ Fehler beim Auflisten der Dateien: {str(e)}")
        return
    
    # Display files
    if not files:
        st.info("ℹ️ Keine Eingabedateien gefunden. Legen Sie Dateien im Eingabeverzeichnis ab.")
        st.code(f"Eingabeverzeichnis: {input_dir}")
        return
    
    # Performance Optimization: Pagination for long file lists
    total_files = len(files)
    st.markdown(f"**{total_files} Datei(en) gefunden**")
    
    # Pagination controls
    if total_files > st.session_state.files_per_page:
        # Get current page
        current_page = st.session_state.get('input_files_page', 1)
        
        # Paginate files
        paginated_files, total_pages, _ = file_manager.paginate_files(
            files,
            page=current_page,
            page_size=st.session_state.files_per_page
        )
        
        # Pagination UI
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("◀ Zurück", disabled=(current_page <= 1), key="input_prev"):
                st.session_state.input_files_page = max(1, current_page - 1)
                st.rerun()
        
        with col2:
            st.markdown(f"<div style='text-align: center'>Seite {current_page} von {total_pages}</div>", 
                       unsafe_allow_html=True)
        
        with col3:
            if st.button("Weiter ▶", disabled=(current_page >= total_pages), key="input_next"):
                st.session_state.input_files_page = min(total_pages, current_page + 1)
                st.rerun()
        
        st.markdown("---")
        
        # Use paginated files
        files = paginated_files
    
    # Create file selection
    file_options = {f"{f.name} ({f.format_size()})": f for f in files}
    
    if file_options:
        selected_file_label = st.selectbox(
            "Datei auswählen für Vorschau:",
            options=list(file_options.keys()),
            key="selected_input_file"
        )
        
        if selected_file_label:
            selected_file = file_options[selected_file_label]
            
            # Display file info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Typ", selected_file.extension)
            
            with col2:
                st.metric("Größe", selected_file.format_size())
            
            with col3:
                st.metric("Geändert", selected_file.format_date())
            
            # Display file preview
            st.markdown("**Vorschau:**")
            try:
                preview = file_manager.get_file_preview(
                    file_path=selected_file.path,
                    max_chars=500
                )
                st.text_area(
                    "Dateiinhalt",
                    value=preview,
                    height=150,
                    disabled=True,
                    label_visibility="collapsed"
                )
            except Exception as e:
                st.warning(f"⚠️ Vorschau nicht verfügbar: {str(e)}")
    
    # Show directory stats
    with st.expander("📊 Verzeichnis-Statistik"):
        try:
            stats = file_manager.get_directory_stats(input_dir)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Dateien gesamt", stats['total_files'])
            
            with col2:
                # Format total size
                total_size = stats['total_size']
                for unit in ['B', 'KB', 'MB', 'GB']:
                    if total_size < 1024.0:
                        size_str = f"{total_size:.1f} {unit}"
                        break
                    total_size /= 1024.0
                else:
                    size_str = f"{total_size:.1f} TB"
                
                st.metric("Gesamtgröße", size_str)
            
            # File types breakdown
            if stats['file_types']:
                st.markdown("**Dateitypen:**")
                for ext, count in sorted(stats['file_types'].items()):
                    st.text(f"  {ext}: {count}")
        
        except Exception as e:
            st.error(f"❌ Fehler beim Laden der Statistik: {str(e)}")


def render_analysis_controls():
    """
    Rendert Analyse-Start-Button und Einstellungen.
    
    Requirement 8.1: WHEN der Analyse-Reiter angezeigt wird 
                    THEN das System SHALL eine Schaltfläche "Analyse starten" anzeigen
    Requirement 8.2: WHEN ein Benutzer auf "Analyse starten" klickt 
                    THEN das System SHALL die aktuelle Konfiguration validieren
    """
    runner = st.session_state.analysis_runner
    config = st.session_state.config_data
    
    # Check if analysis is running
    is_running = runner.is_running()
    
    # Display current configuration summary
    with st.expander("⚙️ Aktuelle Konfiguration", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.text(f"Modell: {config.model_provider} / {config.model_name}")
            st.text(f"Analyse-Modus: {config.analysis_mode}")
            st.text(f"Chunk-Größe: {config.chunk_size}")
        
        with col2:
            st.text(f"Eingabe: {config.data_dir}")
            st.text(f"Ausgabe: {config.output_dir}")
            st.text(f"Coder: {len(config.coder_settings)}")
    
    # Validate configuration before allowing start
    is_valid, errors = config.validate()
    
    if not is_valid:
        st.error("❌ Konfiguration ist ungültig:")
        for error in errors:
            st.error(f"  • {error}")
        st.warning("⚠️ Bitte korrigieren Sie die Konfiguration im Konfigurationsreiter")
        return
    
    # Control buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if not is_running:
            if st.button("▶️ Analyse starten", use_container_width=True, type="primary"):
                # Get project root
                project_manager = st.session_state.project_manager
                project_root = project_manager.get_root_directory()
                
                # Save config before starting analysis so QCA-AID.py can read it
                config_manager = st.session_state.config_manager
                save_success, save_errors = config_manager.save_config(config, format='json')
                if not save_success:
                    st.error("❌ Fehler beim Speichern der Konfiguration:")
                    for error in save_errors:
                        st.error(f"  • {error}")
                    st.stop()
                
                # Resolve paths relative to project root
                data_dir = config.data_dir
                if not Path(data_dir).is_absolute():
                    data_dir = str(project_root / data_dir)
                
                output_dir = config.output_dir
                if not Path(output_dir).is_absolute():
                    output_dir = str(project_root / output_dir)
                
                # Convert config to dict for analysis runner
                config_dict = {
                    'MODEL_PROVIDER': config.model_provider,
                    'MODEL_NAME': config.model_name,
                    'DATA_DIR': data_dir,
                    'OUTPUT_DIR': output_dir,
            'PROJECT_ROOT': str(project_root),
                    'CHUNK_SIZE': config.chunk_size,
                    'CHUNK_OVERLAP': config.chunk_overlap,
                    'BATCH_SIZE': config.batch_size,
                    'CODE_WITH_CONTEXT': config.code_with_context,
                    'MULTIPLE_CODINGS': config.multiple_codings,
                    'MULTIPLE_CODING_THRESHOLD': config.multiple_coding_threshold,
                    'ANALYSIS_MODE': config.analysis_mode,
                    'REVIEW_MODE': config.review_mode,
                    'MANUAL_CODING_ENABLED': config.manual_coding_enabled,
                    'CODER_SETTINGS': [
                        {'temperature': c.temperature, 'coder_id': c.coder_id}
                        for c in config.coder_settings
                    ],
                    'EXPORT_ANNOTATED_PDFS': config.export_annotated_pdfs,
                    'PDF_ANNOTATION_FUZZY_THRESHOLD': config.pdf_annotation_fuzzy_threshold
                }
                
                # Start analysis with improved error handling
                success, messages = runner.start_analysis(config_dict)
                
                if success:
                    st.success("✅ Analyse gestartet!")
                    # Show any warnings
                    for msg in messages:
                        if 'erstellt' in msg.lower() or 'warnung' in msg.lower():
                            st.info(f"ℹ️ {msg}")
                    st.rerun()
                else:
                    st.error("❌ Fehler beim Starten der Analyse:")
                    for msg in messages:
                        st.error(f"  • {msg}")
        else:
            # FIX: Zeige Status-Info statt "Analyse läuft" wenn abgeschlossen
            status = runner.get_status()
            if status.current_step == "Abgeschlossen":
                st.success("✅ Fertig")
            elif status.current_step == "Abgebrochen":
                st.warning("⚠️ Abgebrochen")
            elif status.error:
                st.error("❌ Fehler")
            else:
                st.info("🔄 Analyse läuft...")
    
    with col2:
        # FIX: Zeige "Analyse stoppen" Button nur wenn Analyse wirklich läuft
        if is_running:
            if st.button("⏹️ Analyse stoppen", use_container_width=True, type="secondary"):
                success, error_msg = runner.stop_analysis()
                
                if success:
                    st.warning("⚠️ Analyse gestoppt")
                    st.rerun()
                else:
                    st.error(f"❌ Fehler beim Stoppen der Analyse: {error_msg}")
        else:
            # FIX: Zeige alternativen Button oder leeren Platz wenn Analyse nicht läuft
            status = runner.get_status()
            if status.current_step == "Abgeschlossen" or status.output_file:
                st.success("🎉 Abgeschlossen")
            elif status.current_step == "Abgebrochen":
                if st.button("🔄 Erneut versuchen", use_container_width=True, type="primary"):
                    # Reset status für erneuten Versuch
                    runner.status = runner.status.create_initial()
                    st.rerun()
            elif status.error:
                if st.button("🔄 Erneut versuchen", use_container_width=True, type="primary"):
                    # Reset status für erneuten Versuch
                    runner.status = runner.status.create_initial()
                    st.rerun()
            else:
                # Leerer Platz für bessere Ausrichtung
                st.empty()


def render_analysis_progress():
    """
    Zeigt Analyse-Status.
    
    Requirement 8.3: WHEN die Analyse gestartet wird 
                    THEN das System SHALL Statusmeldungen anzeigen
    Requirement 8.4: WHEN die Analyse läuft 
                    THEN das System SHALL Log-Ausgaben in Echtzeit anzeigen
    """
    runner = st.session_state.analysis_runner
    status = runner.get_status()
    
    # Show simple status indicator
    if status.is_running:
        st.info("🔄 Analyse läuft... Siehe Logs unten für Details")
    elif status.output_file and not status.is_running:
        st.success(f"✅ Analyse abgeschlossen!")
    elif status.error:
        st.error(f"❌ Analyse fehlgeschlagen: {status.error}")
    elif status.current_step == "Abgeschlossen":
        st.success(f"✅ Analyse erfolgreich abgeschlossen!")
    elif status.current_step == "Abgebrochen":
        st.warning(f"⚠️ Analyse wurde abgebrochen")
    elif not status.is_running and status.current_step != "Not started":
        st.info(f"ℹ️ Status: {status.current_step}")
        
        # Find the newest Excel file in the output directory (in case the old one is locked)
        output_dir = Path(status.output_file).parent
        analysis_mode = st.session_state.get('analysis_mode', 'deductive')
        
        # Look for Excel files matching the pattern
        excel_files = list(output_dir.glob(f"QCA-AID_Analysis_{analysis_mode}_*.xlsx"))
        
        if excel_files:
            # Sort by modification time and get the newest
            newest_file = max(excel_files, key=lambda p: p.stat().st_mtime)
            output_path = newest_file
        else:
            # Fallback to the status output file
            output_path = Path(status.output_file)
        
        if output_path.exists():
            filename = output_path.name
            
            # Show filename and provide download button
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"📄 Ergebnisdatei: {filename}")
            with col2:
                # Read file for download with error handling
                try:
                    with open(output_path, 'rb') as f:
                        file_data = f.read()
                    
                    st.download_button(
                        label="⬇️ Download",
                        data=file_data,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                except PermissionError:
                    st.warning(f"⚠️ Datei ist noch geöffnet. Bitte schließ Sie {filename} in Excel und laden Sie die Seite neu.")
                except Exception as e:
                    st.error(f"❌ Fehler beim Lesen der Datei: {str(e)}")
            
            # Show file path for reference
            st.caption(f"Speicherort: {output_path.absolute()}")
        else:
            st.info(f"📄 Ergebnisdatei: {os.path.basename(status.output_file)}")
    
    if status.error:
        st.error(f"❌ Fehler: {status.error}")


def render_log_stream():
    """
    Zeigt Echtzeit-Log-Streaming.
    
    Requirement 8.4: WHEN die Analyse läuft 
                    THEN das System SHALL Log-Ausgaben in Echtzeit anzeigen
    """
    runner = st.session_state.analysis_runner
    
    # Get logs
    logs = runner.get_logs()
    
    if not logs:
        st.info("ℹ️ Keine Logs verfügbar")
        return
    
    # Display logs in a text area
    log_text = '\n'.join(logs[-50:])  # Show last 50 log entries
    
    st.text_area(
        "Log-Ausgabe",
        value=log_text,
        height=300,
        disabled=True,
        label_visibility="collapsed"
    )
    
    # Show log count
    st.caption(f"{len(logs)} Log-Einträge (zeige letzte 50)")
    
    # Auto-refresh if analysis is running
    if runner.is_running():
        st.caption("🔄 Logs werden automatisch aktualisiert...")
        # Trigger rerun every 2 seconds
        import time
        time.sleep(2)
        st.rerun()


def render_results():
    """
    Zeigt Analyseergebnisse.
    
    Requirement 8.5: WHEN die Analyse abgeschlossen ist 
                    THEN das System SHALL eine Erfolgsmeldung mit Link zur Output-Datei anzeigen
    Requirement 7.1: WHEN der Explorer-Reiter angezeigt wird 
                    THEN das System SHALL alle XLSX-Dateien im OUTPUT_DIR auflisten
    Requirement 7.2: WHEN Output-Dateien angezeigt werden 
                    THEN das System SHALL Dateiname, Größe und Änderungsdatum anzeigen
    """
    config = st.session_state.config_data
    output_dir = config.output_dir
    
    # Get project root from project manager
    project_manager = st.session_state.project_manager
    project_root = project_manager.get_root_directory()
    
    # Initialize file manager with project root
    file_manager = FileManager(base_dir=str(project_root))
    
    # Resolve output_dir relative to project root if it's a relative path
    if not Path(output_dir).is_absolute():
        output_dir = str(project_root / output_dir)
    
    # Ensure output directory exists
    success, error_msg = file_manager.ensure_directory(output_dir)
    if not success:
        st.error(f"❌ Fehler beim Erstellen des Ausgabeverzeichnisses: {error_msg}")
        return
    
    # List XLSX files in output directory
    try:
        files = file_manager.list_files(
            directory=output_dir,
            extensions=['.xlsx']
        )
    except Exception as e:
        st.error(f"❌ Fehler beim Auflisten der Ergebnisse: {str(e)}")
        return
    
    # Display results
    if not files:
        st.info("ℹ️ Noch keine Analyseergebnisse vorhanden")
        return
    
    # Performance Optimization: Pagination for long result lists
    total_files = len(files)
    st.markdown(f"**{total_files} Ergebnisdatei(en) gefunden**")
    
    # Action buttons at the top (appear once for all files)
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        # Open file location button - opens the output directory
        if st.button("📂 Ordner öffnen", key="open_folder_top"):
            import subprocess
            import platform
            
            try:
                if platform.system() == 'Windows':
                    os.startfile(output_dir)
                elif platform.system() == 'Darwin':  # macOS
                    subprocess.run(['open', output_dir])
                else:  # Linux
                    subprocess.run(['xdg-open', output_dir])
                
                st.success("✅ Ordner geöffnet")
            except Exception as e:
                st.error(f"❌ Fehler: {str(e)}")
    
    with col_btn2:
        # Show directory path button
        if st.button("📋 Pfad kopieren", key="copy_path_top"):
            st.code(output_dir, language=None)
    
    st.markdown("---")
    
    # Pagination controls
    if total_files > st.session_state.files_per_page:
        # Get current page
        current_page = st.session_state.get('results_files_page', 1)
        
        # Paginate files
        paginated_files, total_pages, _ = file_manager.paginate_files(
            files,
            page=current_page,
            page_size=st.session_state.files_per_page
        )
        
        # Pagination UI
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("◀ Zurück", disabled=(current_page <= 1), key="results_prev"):
                st.session_state.results_files_page = max(1, current_page - 1)
                st.rerun()
        
        with col2:
            st.markdown(f"<div style='text-align: center'>Seite {current_page} von {total_pages}</div>", 
                       unsafe_allow_html=True)
        
        with col3:
            if st.button("Weiter ▶", disabled=(current_page >= total_pages), key="results_next"):
                st.session_state.results_files_page = min(total_pages, current_page + 1)
                st.rerun()
        
        st.markdown("---")
        
        # Use paginated files
        files = paginated_files
    
    # Display each file with compact layout
    for file_info in files:
        with st.container():
            # Filename in bold
            st.markdown(f"**{file_info.name}**")
            # Size and date as small text below
            st.caption(f"📦 {file_info.format_size()} • 📅 {file_info.format_date()}")
            st.markdown("---")

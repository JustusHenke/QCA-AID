"""
Haupt-Ausführungsskript für QCA-AID
====================================
Koordiniert den gesamten Analyse-Workflow.
"""

import os
import sys
import asyncio
from datetime import datetime
from typing import Dict, List

from .core.config import CONFIG, FORSCHUNGSFRAGE
from .core.data_models import CategoryDefinition, CategoryChange
from .preprocessing.material_loader import MaterialLoader
from .analysis.deductive_coding import DeductiveCategoryBuilder, DeductiveCoder
from .analysis.analysis_manager import IntegratedAnalysisManager
from .analysis.manual_coding import ManualCoder
from .management import DevelopmentHistory, CategoryRevisionManager, CategoryManager
from .quality.reliability import ReliabilityCalculator
from .quality.review_manager import ReviewManager
from .export.results_exporter import ResultsExporter
from .QCA_Utils import (
    ConfigLoader, _patch_tkinter_for_threaded_exit,
    ConsoleLogger, TeeWriter, DocumentReader, get_input_with_timeout, token_counter
)
from .QCA_Utils import _calculate_multiple_coding_stats

# Check if PDF annotation is available
try:
    from .QCA_Utils import PDFAnnotator, DocumentToPDFConverter
    pdf_annotation_available = True
except ImportError:
    pdf_annotation_available = False


async def perform_manual_coding(chunks, categories, manual_coders):
    """
    KORRIGIERT: Behandelt Abbruch und Mehrfachkodierung korrekt
    """
    manual_codings = []
    total_segments = sum(len(chunks[doc]) for doc in chunks)
    processed_segments = 0
    
    # Erstelle eine flache Liste aller zu kodierenden Segmente
    all_segments = []
    for document_name, document_chunks in chunks.items():
        for chunk_id, chunk in enumerate(document_chunks):
            all_segments.append((document_name, chunk_id, chunk))
    
    print(f"\nManuelles Kodieren: Insgesamt {total_segments} Segmente zu kodieren")
    
    try:
        # Verarbeite alle Segmente
        for idx, (document_name, chunk_id, chunk) in enumerate(all_segments):
            processed_segments += 1
            progress_percentage = (processed_segments / total_segments) * 100
            
            print(f"\nManuelles Codieren: Dokument {document_name}, "
                  f"Chunk {chunk_id + 1}/{len(chunks[document_name])} "
                  f"(Gesamt: {processed_segments}/{total_segments}, {progress_percentage:.1f}%)")
            
            # PrÜfe, ob es das letzte Segment ist
            last_segment = (processed_segments == total_segments)
            
            for coder_idx, manual_coder in enumerate(manual_coders):
                try:
                    # Informiere den Benutzer Über den Fortschritt
                    if last_segment:
                        print(f"Dies ist das letzte zu kodierende Segment!")
                    
                    # Übergabe des last_segment Parameters an die code_chunk Methode
                    coding_result = await manual_coder.code_chunk(chunk, categories, is_last_segment=last_segment)
                    
                    # KORRIGIERT: PrÜfe auf ABORT_ALL
                    if coding_result == "ABORT_ALL":
                        print("Manuelles Kodieren wurde vom Benutzer abgebrochen.")
                        
                        # Schlieẞe alle verbliebenen GUI-Fenster
                        for coder in manual_coders:
                            if hasattr(coder, 'root') and coder.root:
                                try:
                                    coder.root.quit()
                                    coder.root.destroy()
                                except:
                                    pass
                        
                        return manual_codings  # Gebe bisher gesammelte Kodierungen zurÜck
                    
                    # KORRIGIERT: Behandle sowohl Liste als auch einzelne Kodierungen
                    if coding_result:
                        if isinstance(coding_result, list):
                            # Mehrfachkodierung: Verarbeite jede Kodierung in der Liste
                            print(f"Mehrfachkodierung erkannt: {len(coding_result)} Kodierungen")
                            
                            for i, single_coding in enumerate(coding_result, 1):
                                # Erstelle Dictionary-Eintrag fuer jede Kodierung
                                coding_entry = {
                                    'segment_id': f"{document_name}_chunk_{chunk_id}",
                                    'coder_id': manual_coder.coder_id,
                                    'category': single_coding.get('category', ''),
                                    'subcategories': single_coding.get('subcategories', []),
                                    'confidence': single_coding.get('confidence', {'total': 1.0}),
                                    'justification': single_coding.get('justification', ''),
                                    'text': chunk,
                                    'document_name': document_name,
                                    'chunk_id': chunk_id,
                                    'manual_coding': True,
                                    'manual_multiple_coding': True,
                                    'multiple_coding_instance': i,
                                    'total_coding_instances': len(coding_result),
                                    'coding_date': datetime.now().isoformat()
                                }
                                
                                # FÜge weitere Attribute hinzu falls vorhanden
                                for attr in ['paraphrase', 'keywords', 'text_references', 'uncertainties']:
                                    if attr in single_coding:
                                        coding_entry[attr] = single_coding[attr]
                                
                                manual_codings.append(coding_entry)
                                print(f"  ✅ Mehrfachkodierung {i}/{len(coding_result)}: {coding_entry['category']}")
                        
                        else:
                            # Einzelkodierung (Dictionary)
                            coding_entry = {
                                'segment_id': f"{document_name}_chunk_{chunk_id}",
                                'coder_id': manual_coder.coder_id,
                                'category': coding_result.get('category', ''),
                                'subcategories': coding_result.get('subcategories', []),
                                'confidence': coding_result.get('confidence', {'total': 1.0}),
                                'justification': coding_result.get('justification', ''),
                                'text': chunk,
                                'document_name': document_name,
                                'chunk_id': chunk_id,
                                'manual_coding': True,
                                'manual_multiple_coding': False,
                                'multiple_coding_instance': 1,
                                'total_coding_instances': 1,
                                'coding_date': datetime.now().isoformat()
                            }
                            
                            # FÜge weitere Attribute hinzu falls vorhanden
                            for attr in ['paraphrase', 'keywords', 'text_references', 'uncertainties']:
                                if attr in coding_result:
                                    coding_entry[attr] = coding_result[attr]
                            
                            manual_codings.append(coding_entry)
                            print(f"✅ Manuelle Kodierung erfolgreich: {coding_entry['category']}")
                    else:
                        print("âš  Manuelle Kodierung Übersprungen")
                        
                except Exception as e:
                    print(f"Fehler bei manuellem Kodierer {manual_coder.coder_id}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue  # Fahre mit dem nÄchsten Chunk fort
                    
                # Kurze Pause zwischen den Chunks
                await asyncio.sleep(0.5)
    
        print("\n[OK] Manueller Kodierungsprozess abgeschlossen")
        print(f"- {len(manual_codings)}/{total_segments} Segmente erfolgreich kodiert")
        
        # Sicherstellen, dass alle Fenster geschlossen sind
        for coder in manual_coders:
            if hasattr(coder, 'root') and coder.root:
                try:
                    coder.root.quit()
                    coder.root.destroy()
                    coder.root = None
                except:
                    pass
    
    except Exception as e:
        print(f"Fehler im manuellen Kodierungsprozess: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Versuche, alle Fenster zu schlieẞen, selbst im Fehlerfall
        for coder in manual_coders:
            if hasattr(coder, 'root') and coder.root:
                try:
                    coder.root.quit()
                    coder.root.destroy()
                    coder.root = None
                except:
                    pass
    
    return manual_codings

# ============================ 
# 5. Hauptprogramm
# ============================ 

# Aufgabe: ZusammenfÜhrung aller Komponenten, Steuerung des gesamten Analyseprozesses
async def main() -> None:
    try:
        # 1. Konfiguration laden - ZUERST damit INPUT/OUTPUT_DIR gesetzt sind
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up one level to root
        
        # Setze Input/Output Verzeichnisse, bevor ConfigLoader verwendet wird
        CONFIG['OUTPUT_DIR'] = os.path.join(script_dir, 'output')
        CONFIG['DATA_DIR'] = os.path.join(script_dir, 'input')
        CONFIG['INPUT_DIR'] = os.path.join(script_dir, 'input')
        
        # Erstelle diese Verzeichnisse, falls sie nicht existieren
        os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
        os.makedirs(CONFIG['DATA_DIR'], exist_ok=True)
        
        # FIX: Console Logging initialisieren
        console_logger = ConsoleLogger(CONFIG['OUTPUT_DIR'])
        console_logger.start_logging()

        print("=== Qualitative Inhaltsanalyse nach Mayring ===")

        config_loader = ConfigLoader(script_dir, CONFIG)
        
        if config_loader.load_codebook():
            print("\nKonfiguration erfolgreich geladen")
        else:
            print("Verwende Standard-Konfiguration")
        
        # Stelle sicher, dass Input/Output Pfade im Root sind, nicht in QCA_AID_assets
        CONFIG['OUTPUT_DIR'] = os.path.join(script_dir, 'output')
        CONFIG['DATA_DIR'] = os.path.join(script_dir, 'input')
        CONFIG['INPUT_DIR'] = os.path.join(script_dir, 'input')

        category_builder = DeductiveCategoryBuilder()
        initial_categories = category_builder.load_theoretical_categories()
        
        # 3. Manager und History initialisieren
        development_history = DevelopmentHistory(CONFIG['OUTPUT_DIR'])
        revision_manager = CategoryRevisionManager(
            output_dir=CONFIG['OUTPUT_DIR'],
            config=CONFIG
        )
        
        # Initiale Kategorien dokumentieren
        for category_name in initial_categories.keys():
            revision_manager.changes.append(CategoryChange(
                category_name=category_name,
                change_type='add',
                description="Initiale deduktive Kategorie",
                timestamp=datetime.now().isoformat(),
                justification="Teil des ursprÜnglichen deduktiven Kategoriensystems"
            ))

        # 4. Dokumente einlesen
        print("\n2. Lese Dokumente ein...")
        reader = DocumentReader(CONFIG['DATA_DIR'])  # Import aus QCA_Utils
        documents = await reader.read_documents()

        if not documents:
            print("\nKeine Dokumente zum Analysieren gefunden.")
            return

        # 4b. Abfrage zur induktiven Kodierung
        print("\n3. Induktive Kodierung konfigurieren...")

        # PrÜfe ob ein induktives Codebook existiert
        codebook_path = os.path.join(CONFIG['OUTPUT_DIR'], "codebook_inductive.json")
        skip_inductive = False

        if os.path.exists(codebook_path):
            print("\nGespeichertes induktives Codebook gefunden.")
            print("Automatische FortfÜhrung in 10 Sekunden...")
            
            use_saved = get_input_with_timeout(
                "\nMÖchten Sie das gespeicherte erweiterte Kodesystem laden? (j/N)",
                timeout=10
            )
            
            if use_saved.lower() == 'j':
                try:
                    with open(codebook_path, 'r', encoding='utf-8') as f:
                        saved_categories = json.load(f)
                        
                    if 'categories' in saved_categories:
                        # Konvertiere JSON zurÜck in CategoryDefinition Objekte
                        for name, cat_data in saved_categories['categories'].items():
                            initial_categories[name] = CategoryDefinition(
                                name=name,
                                definition=cat_data['definition'],
                                examples=cat_data.get('examples', []),
                                rules=cat_data.get('rules', []),
                                subcategories=cat_data.get('subcategories', {}),
                                added_date=cat_data.get('added_date', datetime.now().strftime("%Y-%m-%d")),
                                modified_date=cat_data.get('modified_date', datetime.now().strftime("%Y-%m-%d"))
                            )
                        print(f"\n✅ {len(saved_categories['categories'])} Kategorien aus Codebook geladen")
                        skip_inductive = True
                    else:
                        print("\nWarnung: UngÜltiges Codebook-Format")
                        
                except Exception as e:
                    print(f"\nFehler beim Laden des Codebooks: {str(e)}")
                    print("Fahre mit Standard-Kategorien fort")

        if not skip_inductive:
            default_mode = CONFIG['ANALYSIS_MODE']
            print("\nAktueller Analysemodus aus Codebook: {default_mode}")
            print("Sie haben 10 Sekunden Zeit fuer die Eingabe.")
            print("Optionen:")
            print("1 = inductive (volle induktive Analyse)")
            print("2 = abductive (nur Subkategorien entwickeln)")
            print("3 = deductive (nur deduktiv)")
            print("4 = grounded (Subkategorien sammeln, spaeter Hauptkategorien generieren)")

            analysis_mode = get_input_with_timeout(
                f"\nWelchen Analysemodus moechten Sie verwenden? [1/2/3/4] (Standard: {CONFIG['ANALYSIS_MODE']})", 
                timeout=10
            )

            # Mapping von Zahlen zu Modi
            mode_mapping = {
                '1': 'inductive',
                '2': 'abductive',
                '3': 'deductive',
                '4': 'grounded'
            }

            # Verarbeite Zahlen oder direkte Modusangaben, behalte Default wenn leere oder ungÜltige Eingabe
            if analysis_mode:  # Nur wenn etwas eingegeben wurde
                if analysis_mode in mode_mapping:
                    CONFIG['ANALYSIS_MODE'] = mode_mapping[analysis_mode]
                elif analysis_mode.lower() in mode_mapping.values():
                    CONFIG['ANALYSIS_MODE'] = analysis_mode.lower()
                else:
                    print(f"\nUngueltiger Modus '{analysis_mode}'. Verwende Default-Modus '{default_mode}'.")
                    # Keine Änderung an CONFIG['ANALYSIS_MODE'], Default bleibt bestehen
            else:
                print(f"Keine Eingabe. Verwende Default-Modus '{default_mode}'.")

            # Bestimme, ob induktive Analyse Übersprungen wird
            skip_inductive = CONFIG['ANALYSIS_MODE'] == 'deductive'

            print(f"\nAnalysemodus: {CONFIG['ANALYSIS_MODE']} {'(Skip induktiv)' if skip_inductive else ''}")

        # Grounded Theory Modus Info entfernt (Unicode-Probleme)

        # 5. Kodierer konfigurieren
        print("\n4. Konfiguriere Kodierer...")
        # Automatische Kodierer
        auto_coders = [
            DeductiveCoder(
                model_name=CONFIG['MODEL_NAME'],
                temperature=coder_config['temperature'],
                coder_id=coder_config['coder_id']
            )
            for coder_config in CONFIG['CODER_SETTINGS']
        ]

        # Manuelle Kodierung konfigurieren
        print("\nKonfiguriere manuelle Kodierung...")
        print("Sie haben 10 Sekunden Zeit fuer die Eingabe.")
        print("Druecken Sie 'j' fuer manuelle Kodierung oder 'n' zum Überspringen.")

        manual_coders = []
        user_input = get_input_with_timeout(
            "\nMÖchten Sie manuell kodieren? (j/N)",
            timeout=10
        )
        
        if user_input.lower() == 'j':
            manual_coders.append(ManualCoder(coder_id="human_1"))
            print("\n✅ Manueller Kodierer wurde hinzugefÜgt")
        else:
            print("\n[INFO] Keine manuelle Kodierung - nur automatische Kodierung wird durchgefÜhrt")

        # 6. Material vorbereiten
        print("\n5. Bereite Material vor...")
        loader = MaterialLoader(
            data_dir=CONFIG['DATA_DIR'],
            chunk_size=CONFIG['CHUNK_SIZE'],
            chunk_overlap=CONFIG['CHUNK_OVERLAP']
        )
        chunks = {}
        for doc_name, doc_text in documents.items():
            chunks[doc_name] = loader.chunk_text(doc_text)
            print(f"- {doc_name}: {len(chunks[doc_name])} Chunks erstellt")

        # 7. Manuelle Kodierung durchfÜhren
        manual_codings = []
        if manual_coders:
            print("\n6. Starte manuelle Kodierung...")
            
            # Verwende die verbesserte perform_manual_coding Funktion
            manual_coding_result = await perform_manual_coding(
                chunks=chunks, 
                categories=initial_categories,
                manual_coders=manual_coders
            )
            
            if manual_coding_result == "ABORT_ALL":
                print("Manuelle Kodierung abgebrochen. Beende Programm.")
                return
                
            manual_codings = manual_coding_result
            print(f"Manuelle Kodierung abgeschlossen: {len(manual_codings)} Kodierungen")
            
            # Stelle sicher, dass alle Kodierer-Fenster geschlossen sind
            for coder in manual_coders:
                if hasattr(coder, 'root') and coder.root:
                    try:
                        coder.root.quit()
                        coder.root.destroy()
                        coder.root = None
                    except:
                        pass


        # 8. Integrierte Analyse starten
        print("\n7. Starte integrierte Analyse...")

        # Zeige Kontext-Modus an
        print(f"\nKodierungsmodus: {'Mit progressivem Kontext' if CONFIG.get('CODE_WITH_CONTEXT', False) else 'Ohne Kontext'}")
        
        analysis_manager = IntegratedAnalysisManager(CONFIG)

        # Initialisiere FortschrittsÜberwachung
        progress_task = asyncio.create_task(
            monitor_progress(analysis_manager)
        )

        try:
            # Starte die Hauptanalyse
            final_categories, coding_results = await analysis_manager.analyze_material(
                chunks=chunks,
                initial_categories=initial_categories,
                skip_inductive=skip_inductive
            )

            # Beende FortschrittsÜberwachung
            progress_task.cancel()
            await progress_task

            # Kombiniere alle Kodierungen
            all_codings = []
            if coding_results and len(coding_results) > 0:
                print(f"\nFÜge {len(coding_results)} automatische Kodierungen hinzu")
                for coding in coding_results:
                    if isinstance(coding, dict) and 'segment_id' in coding:
                        all_codings.append(coding)
                    else:
                        print(f"Überspringe ungÜltige Kodierung: {coding}")

            # FÜge manuelle Kodierungen hinzu
            if manual_codings and len(manual_codings) > 0:
                print(f"FÜge {len(manual_codings)} manuelle Kodierungen hinzu")
                all_codings.extend(manual_codings)

            print(f"\nGesamtzahl Kodierungen: {len(all_codings)}")


            # 8.  Intercoder-ReliabilitÄt mit kategorie-spezifischer Berechnung
            if all_codings:
                print("\n8. Berechne korrekte Intercoder-ReliabilitÄt...")
                
                # FIX: SICHER ursprÜngliche Kodierungen BEVOR Review-Prozess
                original_codings_for_reliability = all_codings.copy()  # Kopie der ursprÜnglichen Kodierungen
                
                # NEUE LOGIK: Verwende korrigierte ReliabilityCalculator
                reliability_calculator = ReliabilityCalculator()
                reliability = reliability_calculator.calculate_reliability(original_codings_for_reliability)
                
                print(f"📈 Krippendorff's Alpha (korrigiert fuer Mehrfachkodierungen): {reliability:.3f}")
            else:
                print("\nKeine Kodierungen fuer ReliabilitÄtsberechnung")
                reliability = 0.0
                original_codings_for_reliability = []

            # 9. Review-Behandlung mit kategorie-zentrierter Mehrfachkodierungs-Logik
            print(f"\n9. FÜhre kategorie-zentrierten Review-Prozess durch...")

            # Gruppiere Kodierungen nach Segmenten fuer Review
            segment_codings = {}
            for coding in all_codings:
                segment_id = coding.get('segment_id')
                if segment_id:
                    if segment_id not in segment_codings:
                        segment_codings[segment_id] = []
                    segment_codings[segment_id].append(coding)
            
            # Erkenne manuelle Kodierer
            manual_coders = set()
            for coding in all_codings:
                coder_id = coding.get('coder_id', '')
                if 'manual' in coder_id.lower() or 'human' in coder_id.lower():
                    manual_coders.add(coder_id)
            
            # Bestimme Review-Modus
            review_mode = CONFIG.get('REVIEW_MODE', 'consensus')

            if manual_coders:
                print(f"🎯 Manuelle Kodierung erkannt von {len(manual_coders)} Kodierern")
                if review_mode == 'manual':
                    print("   Manueller Review-Modus aus CONFIG aktiviert")
                else:
                    print(f"   CONFIG-Einstellung '{review_mode}' wird verwendet (nicht automatisch auf 'manual' geÄndert)")
            else:
                if review_mode == 'manual':
                    print("   Manueller Review-Modus aus CONFIG aktiviert (auch ohne manuelle Kodierer)")
            
            print(f"🔀‹ Review-Modus: {review_mode}")
            print(f"📈 Eingabe: {len(all_codings)} ursprÜngliche Kodierungen")
            
            review_manager = ReviewManager(CONFIG['OUTPUT_DIR'])
            
            try:
                # FÜhre kategorie-zentrierten Review durch
                reviewed_codings = review_manager.process_coding_review(all_codings, review_mode)
                
                print(f"[OK] Review abgeschlossen: {len(reviewed_codings)} finale Kodierungen")
                
                # Überschreibe all_codings mit den reviewten Ergebnissen
                all_codings = reviewed_codings
                
                # Setze Export-Modus
                export_mode = review_mode

                if 'console_logger' in locals():
                    console_logger.stop_logging()    

            except Exception as e:
                print(f"❌ Fehler beim Review-Prozess: {str(e)}")
                print("📝 Verwende ursprÜngliche Kodierungen ohne Review")
                # all_codings bleibt unverÄndert
                export_mode = review_mode
                import traceback
                traceback.print_exc()
                if 'console_logger' in locals():
                    console_logger.stop_logging() 
            

            # 10. Speichere induktiv erweitertes Codebook
            # Hier die Zusammenfassung der finalen Kategorien vor dem Speichern:
            print("\nFinales Kategoriensystem komplett:")
            print(f"- Insgesamt {len(final_categories)} Hauptkategorien")
            print(f"- Davon {len(final_categories) - len(initial_categories)} neu entwickelt")
            
            # ZÄhle Subkategorien fuer zusammenfassende Statistik
            total_subcats = sum(len(cat.subcategories) for cat in final_categories.values())
            print(f"- Insgesamt {total_subcats} Subkategorien")
            
            # 10. Speichere induktiv erweitertes Codebook
            if final_categories:
                category_manager = CategoryManager(CONFIG['OUTPUT_DIR'])
                category_manager.save_codebook(
                    categories=final_categories,
                    filename="codebook_inductive.json"
                )
                print(f"\nCodebook erfolgreich gespeichert mit {len(final_categories)} Hauptkategorien und {total_subcats} Subkategorien")

            # 11. Export der Ergebnisse
            print("\n10. Exportiere Ergebnisse...")
            if all_codings:
                exporter = ResultsExporter(
                    output_dir=CONFIG['OUTPUT_DIR'],
                    attribute_labels=CONFIG['ATTRIBUTE_LABELS'],
                    analysis_manager=analysis_manager,
                    inductive_coder=reliability_calculator
                )

                exporter.current_categories = final_categories 
                
                # FIX: Store original codings in exporter for reliability calculation
                exporter.original_codings_for_reliability = original_codings_for_reliability
                
                # Exportiere Ergebnisse mit Document-Summaries, wenn vorhanden
                summary_arg = analysis_manager.document_summaries if CONFIG.get('CODE_WITH_CONTEXT', True) else None

                # FIX: VERWENDE den bereits bestimmten export_mode, lade NICHT nochmal aus CONFIG
                # ENTFERNT: export_mode = CONFIG.get('REVIEW_MODE', 'consensus') 


                # Validiere und mappe den Export-Modus
                if export_mode == 'auto':
                    export_mode = 'consensus'  # 'auto' ist ein Alias fuer 'consensus'
                elif export_mode not in ['consensus', 'majority', 'manual_priority', 'manual']:
                    print(f"Warnung: Unbekannter export_mode '{export_mode}', verwende 'consensus'")
                    export_mode = 'consensus'
                
                # FIX: Mappe 'manual' auf 'manual_priority' fuer Export
                if export_mode == 'manual':
                    export_mode = 'manual_priority'

                print(f"Export wird mit Modus '{export_mode}' durchgefÜhrt")

                await exporter.export_results(
                    codings=all_codings,  # Review-Ergebnisse fuer Export  
                    reliability=reliability,  # Bereits berechnete ReliabilitÄt
                    categories=final_categories,
                    chunks=chunks,  
                    revision_manager=revision_manager,
                    export_mode=export_mode,
                    original_categories=initial_categories,
                    inductive_coder=reliability_calculator,  
                    document_summaries=summary_arg,
                    original_codings=original_codings_for_reliability,
                    is_intermediate_export=False
                )

                # Ausgabe der finalen Summaries, wenn vorhanden
                if CONFIG.get('CODE_WITH_CONTEXT', True) and analysis_manager.document_summaries:
                    print("\nFinale Document-Summaries:")
                    for doc_name, summary in analysis_manager.document_summaries.items():
                        print(f"\n📋 {doc_name}:")
                        print(f"  {summary}")

                # FIX: Korrekte PrÜfung von EXPORT_ANNOTATED_PDFS
                export_pdfs_enabled = CONFIG.get('EXPORT_ANNOTATED_PDFS', True)
                print(f"DEBUG: EXPORT_ANNOTATED_PDFS Wert: {export_pdfs_enabled} (Typ: {type(export_pdfs_enabled)})")
                
                if export_pdfs_enabled is False or str(export_pdfs_enabled).lower() in ['false', '0', 'no', 'nein', 'off']:
                    print("\n   [INFO]⚠️ PDF-Annotation deaktiviert (EXPORT_ANNOTATED_PDFS=False)")
                elif not pdf_annotation_available:
                    print("\n   [INFO]⚠️ PDF-Annotation nicht verfÜgbar (PyMuPDF/ReportLab fehlt)")
                    print("   ℹ️ Installieren Sie mit: pip install PyMuPDF reportlab")
                else:
                    # PDF-Annotation ist aktiviert und verfÜgbar
                    try:
                        print("\nðŸŽ¨ Exportiere annotierte PDFs fuer alle Dateiformate...")
                        
                        # FIX: Verwende erweiterte Methode fuer alle Formate
                        annotated_pdfs = exporter.export_annotated_pdfs_all_formats(
                            codings=all_codings,
                            chunks=chunks,
                            data_dir=CONFIG['DATA_DIR']
                        )
                        
                        if annotated_pdfs:
                            print(f"📋 {len(annotated_pdfs)} annotierte PDFs erstellt:")
                            for pdf_path in annotated_pdfs:
                                print(f"   - {os.path.basename(pdf_path)}")
                        else:
                            print("   [INFO]⚠️ Keine Dateien fuer Annotation gefunden")
                            
                    except Exception as e:
                        print(f"   ❌ Fehler bei erweiterter PDF-Annotation: {e}")
                        print("   ℹ️ PDF-Annotation Übersprungen, normaler Export fortgesetzt")

                print("Export erfolgreich abgeschlossen")

            else:
                print("Keine Kodierungen zum Exportieren vorhanden")


            # 12. Zeige finale Statistiken
            print("\nAnalyse abgeschlossen:")
            print(analysis_manager.get_analysis_report())

            if CONFIG.get('MULTIPLE_CODINGS', True):
                # Verwende die ursprÜnglichen Kodierungen fuer Mehrfachkodierungs-Statistiken
                codings_for_stats = original_codings_for_reliability if original_codings_for_reliability else all_codings
                
                if codings_for_stats:
                    multiple_coding_stats = _calculate_multiple_coding_stats(codings_for_stats)
                    
                    # FIX: ZeroDivisionError bei Division durch Null verhindern
                    auto_coder_ids = set(c.get('coder_id', '') for c in codings_for_stats if c.get('coder_id', '').startswith('auto'))
        # Mehrfachkodierungs-Statistiken entfernt (Unicode-Probleme)
                else:
                    print("\n                    Mehrfachkodierungs-Statistiken: Keine Kodierungen fuer Analyse verfÜgbar")
            else:
                print("\n                    Mehrfachkodierungs-Statistiken: DEAKTIVIERT")
            
            # Token-Statistiken
            print("\nToken-Nutzung:")
            print(token_counter.get_report())
            
            # Relevanz-Statistiken
            relevance_stats = analysis_manager.relevance_checker.get_statistics()
            print("\nRelevanz-Statistiken:")
            print(f"- Segmente analysiert: {relevance_stats['total_segments']}")
            print(f"- Relevante Segmente: {relevance_stats['relevant_segments']}")
            print(f"- Relevanzrate: {relevance_stats['relevance_rate']*100:.1f}%")
            print(f"- API-Calls gespart: {relevance_stats['total_segments'] - relevance_stats['api_calls']}")
            print(f"- Cache-Nutzung: {relevance_stats['cache_size']} EintrÄge")

            if 'console_logger' in locals():
                console_logger.stop_logging() 

        except asyncio.CancelledError:
            print("\nAnalyse wurde abgebrochen.")
            if 'console_logger' in locals():
                console_logger.stop_logging() 
        finally:
            # Stelle sicher, dass die FortschrittsÜberwachung beendet wird
            if not progress_task.done():
                progress_task.cancel()
                try:
                    await progress_task
                    if 'console_logger' in locals():
                        console_logger.stop_logging() 
                except asyncio.CancelledError:
                    pass
                    if 'console_logger' in locals():
                        console_logger.stop_logging() 

        if 'console_logger' in locals():
            console_logger.stop_logging() 

    except Exception as e:
        import traceback
        print(f"Fehler in der HauptausfÜhrung: {str(e)}")
        traceback.print_exc()
        if 'console_logger' in locals():
            console_logger.stop_logging() 

        try:
            if 'analysis_manager' in locals() and hasattr(analysis_manager, 'coding_results'):
                print("\nVersuche Zwischenergebnisse zu exportieren...")
                await analysis_manager._export_intermediate_results(
                    chunks=chunks if 'chunks' in locals() else {},
                    current_categories=final_categories if 'final_categories' in locals() else {},
                    deductive_categories=initial_categories if 'initial_categories' in locals() else {},
                    initial_categories=initial_categories if 'initial_categories' in locals() else {}
                )
            if 'console_logger' in locals():
                console_logger.stop_logging() 

        except Exception as export_error:
            print(f"Fehler beim Export der Zwischenergebnisse: {str(export_error)}")
            if 'console_logger' in locals():
                console_logger.stop_logging() 


async def monitor_progress(analysis_manager: IntegratedAnalysisManager):
    """
    Überwacht und zeigt den Analysefortschritt an.
    """
    try:
        while True:
            progress = analysis_manager.get_progress_report()
            
            # Formatiere Fortschrittsanzeige
            print("\n--- Analysefortschritt ---")
            print(f"Verarbeitet: {progress['progress']['processed_segments']} Segmente")
            print(f"Geschwindigkeit: {progress['progress']['segments_per_hour']:.1f} Segmente/Stunde")
            print("------------------------")
            
            await asyncio.sleep(30)  # Update alle 30 Sekunden
            
    except asyncio.CancelledError:
        print("\nFortschrittsÜberwachung beendet.")

_patch_tkinter_for_threaded_exit()

if __name__ == "__main__":
    try:
        # Windows-spezifische Event Loop Policy setzen
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # Hauptprogramm ausfÜhren
        asyncio.run(main())
        
    except KeyboardInterrupt:
        print("\nProgramm durch Benutzer beendet")
    except Exception as e:
        print(f"Fehler im Hauptprogramm: {str(e)}")
        raise


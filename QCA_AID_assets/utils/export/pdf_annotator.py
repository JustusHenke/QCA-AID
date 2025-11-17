"""
PDF Annotator

Annotates original PDF files with color-coded highlights based on QCA-AID codings.
Creates visual representation of coded segments with legend and metadata.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from fuzzywuzzy import fuzz
import fitz  # PyMuPDF

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
except ImportError:
    canvas = None
    colors = None


class PDFAnnotator:
    """
    FIX: Annotiert Original-PDFs mit farbkodierten Highlights basierend auf QCA-AID Kodierungen
    
    Diese Klasse erstellt das gleiche PDF wie das Original, nur mit farbigen Highlights
    für kodierte Textabschnitte und vollständigen Annotationen mit Metadaten.
    """
    
    def __init__(self, results_exporter: Any ):
        self.results_exporter = results_exporter
        self.category_colors = {}
        self.fuzzy_match_threshold = 0.85  # Ähnlichkeitsschwelle für Text-Matching
        
        # FIX: Farbpalette für Hauptkategorien (RGB-Werte für PyMuPDF)
        self.default_colors = {
            'Nicht kodiert': (0.8, 0.8, 0.8),  # Grau
            'Kategorie1': (1.0, 0.8, 0.8),     # Hellrot
            'Kategorie2': (0.8, 1.0, 0.8),     # Hellgrün  
            'Kategorie3': (0.8, 0.8, 1.0),     # Hellblau
            'Kategorie4': (1.0, 1.0, 0.8),     # Hellgelb
            'Kategorie5': (1.0, 0.8, 1.0),     # Hellmagenta
            'Kategorie6': (0.8, 1.0, 1.0),     # Hellcyan
        }
    
    def annotate_pdf_with_codings(self, 
                                 pdf_path: str, 
                                 codings: List[Dict], 
                                 chunks: Dict[str, List[str]], 
                                 output_path: str = None) -> str:
        """
        FIX: Hauptmethode zur PDF-Annotation
        
        Args:
            pdf_path: Pfad zum ursprünglichen PDF
            codings: Liste der Kodierungen aus QCA-AID
            chunks: Dictionary mit chunk_id -> text mapping
            output_path: Ausgabepfad (optional)
            
        Returns:
            str: Pfad zur annotierten PDF-Datei
        """
        print(f"\n🎨 Beginne PDF-Annotation: {Path(pdf_path).name}")
        
        # FIX: Öffne Original-PDF
        try:
            doc = fitz.open(pdf_path)
            print(f"   📄 PDF geladen: {len(doc)} Seiten")
        except Exception as e:
            print(f"❌ Fehler beim Öffnen der PDF: {e}")
            return None
        
        # FIX: Bereite Kodierungsdaten vor
        coding_map = self._prepare_coding_map(codings, chunks)
        print(f"   📋 {len(coding_map)} Textabschnitte zu annotieren")
        
        # FIX: Initialisiere Farbschema
        self._initialize_color_scheme(coding_map)
        
        # FIX: Annotiere jede Seite
        total_annotations = 0
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            annotations_on_page = self._annotate_page(page, coding_map, page_num + 1)
            total_annotations += annotations_on_page
            print(f"   📄 Seite {page_num + 1}: {annotations_on_page} Highlights hinzugefügt")
        
        # FIX: Erstelle Legende als erste Seite
        self._add_legend_page(doc)
        
        # FIX: Speichere annotierte PDF
        if not output_path:
            base_name = Path(pdf_path).stem
            output_path = f"{base_name}_annotiert.pdf"
        
        doc.save(output_path)
        doc.close()
        
        print(f"✅ PDF-Annotation abgeschlossen:")
        print(f"   📊 {total_annotations} Highlights erstellt")
        print(f"   💾 Gespeichert als: {output_path}")
        
        return output_path
    
    # FIX: Diese Korrekturen in PDFAnnotator Klasse in QCA_Utils.py vornehmen

    def _prepare_coding_map(self, codings: List[Dict], chunks: Dict[str, List[str]]) -> Dict[str, Dict]:
        """
        FIX: Korrigierte Version mit besserer Dateipfad-Behandlung und None-Kategorie-Handling
        """
        coding_map = {}
        
        print(f"\n   📋 Bereite {len(codings)} Kodierungen vor...")
        print(f"   📁 Verfügbare Dokumente: {len(chunks)}")
        
        for i, coding in enumerate(codings):
            segment_id = coding.get('segment_id', '')
            category = coding.get('category', 'Nicht kodiert')
            
            # FIX: Behandle None-Kategorien
            if category is None:
                category = 'Nicht kodiert'
            
            print(f"\n      [SEARCH] Kodierung {i+1}: {segment_id} → {category}")
            
            try:
                if '_chunk_' not in segment_id:
                    print(f"          ❌ Ungültiges Segment-ID Format (kein '_chunk_')")
                    continue
                
                doc_name = segment_id.split('_chunk_')[0]
                chunk_part = segment_id.split('_chunk_')[1]
                
                if '-' in chunk_part:
                    chunk_id = int(chunk_part.split('-')[0])
                else:
                    chunk_id = int(chunk_part)
                
                print(f"          📂 Parsed: doc='{doc_name}', chunk_id={chunk_id}")
                
                if doc_name not in chunks:
                    print(f"          ❌ Dokument '{doc_name}' nicht in chunks gefunden")
                    continue
                
                doc_chunks = chunks[doc_name]
                
                if chunk_id >= len(doc_chunks):
                    print(f"          ❌ Chunk {chunk_id} nicht vorhanden (nur {len(doc_chunks)} Chunks)")
                    continue
                
                chunk_text = doc_chunks[chunk_id]
                
                if not chunk_text or len(str(chunk_text).strip()) < 10:
                    print(f"          ⚠️ Chunk-Text zu kurz oder leer")
                    continue
                
                text_content = str(chunk_text).strip()
                
                # FIX: Erweiterte Dateipfad-Erkennung und -Bereinigung
                if self._contains_file_path_artifacts(text_content):
                    # Versuche Text zu bereinigen statt komplett zu überspringen
                    cleaned_content = self._remove_file_path_artifacts(text_content)
                    if len(cleaned_content.strip()) >= 50:  # Mindestens 50 Zeichen nach Bereinigung
                        text_content = cleaned_content
                        print(f"          🔧 Dateipfad-Artefakte entfernt, verwende bereinigten Text")
                    else:
                        print(f"          ⚠️ Zu wenig Text nach Dateipfad-Bereinigung, überspringe")
                        continue
                
                clean_text = self._clean_text_for_matching(text_content)
                
                if len(clean_text) < 10:
                    print(f"          ⚠️ Bereinigter Text zu kurz: {len(clean_text)} Zeichen")
                    continue
                
                map_key = f"{segment_id}_{category}"
                
                coding_map[map_key] = {
                    'segment_id': segment_id,
                    'category': category,
                    'subcategories': coding.get('subcategories', []),
                    'justification': coding.get('justification', ''),
                    'confidence': coding.get('confidence', {}),
                    'original_text': text_content,
                    'clean_text': clean_text,
                    'doc_name': doc_name,
                    'chunk_id': chunk_id
                }
                
                print(f"          ✅ Bereit: {len(clean_text)} Zeichen")
                print(f"              Preview: '{clean_text[:80]}...'")
                
            except Exception as e:
                print(f"          ❌ Fehler: {e}")
                continue
        
        print(f"\n   📊 {len(coding_map)} Kodierungen erfolgreich vorbereitet")
        return coding_map

    def _contains_file_path_artifacts(self, text: str) -> bool:
        """
        FIX: Neue Methode zur Erkennung von Dateipfad-Artefakten
        """
        file_path_indicators = [
            'file:///',
            'OneDrive',
            'C:\\Users\\',
            '/Users/',
            '.txt',
            '.pdf',
            '.docx',
            'Projekte/Forschung'
        ]
        
        text_start = text[:200]  # Prüfe nur die ersten 200 Zeichen
        return any(indicator in text_start for indicator in file_path_indicators)

    def _remove_file_path_artifacts(self, text: str) -> str:
        """
        FIX: Neue Methode zur Entfernung von Dateipfad-Artefakten
        """
        import re
        
        # Entferne Zeilen die wie Dateipfade aussehen
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Überspringe Zeilen mit typischen Dateipfad-Mustern
            if (line.startswith('file:///') or 
                'OneDrive' in line or
                re.match(r'^[A-Z]:\\', line) or  # Windows Pfade
                line.startswith('/Users/') or   # Mac Pfade
                line.endswith(('.txt', '.pdf', '.docx')) or
                len(line) < 10):  # Sehr kurze Zeilen
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def _initialize_color_scheme(self, coding_map: Dict[str, Dict]) -> None:
        """
        FIX: Erweiterte Farbschema-Initialisierung mit None-Kategorie-Behandlung
        """
        categories = set()
        for coding_info in coding_map.values():
            category = coding_info['category']
            if category and category != 'None':  # FIX: Ignoriere None und leere Kategorien
                categories.add(category)
        
        # FIX: Nutze bestehende Farben vom ResultsExporter falls verfügbar
        if hasattr(self.results_exporter, 'category_colors') and self.results_exporter.category_colors:
            for category, hex_color in self.results_exporter.category_colors.items():
                if category in categories:  # Nur Kategorien die auch verwendet werden
                    self.category_colors[category] = self._hex_to_rgb(hex_color)
        
        # FIX: Ergänze fehlende Kategorien mit Standard-Farben
        available_colors = list(self.default_colors.values())
        color_index = 0
        
        for category in sorted(categories):
            if category not in self.category_colors:
                if color_index < len(available_colors):
                    self.category_colors[category] = available_colors[color_index]
                    color_index += 1
                else:
                    import random
                    self.category_colors[category] = (
                        0.7 + random.random() * 0.3,
                        0.7 + random.random() * 0.3,
                        0.7 + random.random() * 0.3
                    )
        
        print(f"   🎨 Farbschema initialisiert für {len(self.category_colors)} Kategorien")
        for cat, color in self.category_colors.items():
            print(f"      - {cat}: RGB{color}")

    def _group_codings_by_original_text(self, coding_map: Dict[str, Dict]) -> Dict[str, List[Dict]]:
        """
        FIX: Verbesserte Gruppierung mit None-Kategorie-Behandlung
        """
        text_groups = {}
        
        for map_key, coding_info in coding_map.items():
            category = coding_info['category']
            
            # FIX: Überspringe None-Kategorien bei der Gruppierung
            if not category or category == 'None':
                print(f"      ⚠️ Überspringe Kodierung mit leerer/None Kategorie: {map_key}")
                continue
            
            clean_text = coding_info.get('clean_text', coding_info.get('original_text', ''))
            
            if clean_text not in text_groups:
                text_groups[clean_text] = []
            
            text_groups[clean_text].append(coding_info)
        
        print(f"   📋 Text gruppiert in {len(text_groups)} Gruppen")
        return text_groups
    
    def _clean_text_for_matching(self, text: str) -> str:
        """
        FIX: Weniger aggressive Text-Bereinigung für besseres Matching
        """
        if not text:
            return ""
        
        # FIX: Behalte mehr Zeichen für besseres Matching
        clean = re.sub(r'\s+', ' ', text.strip())
        
        # FIX: Entferne nur wirklich problematische Zeichen
        clean = clean.replace('\uf0b7', '•')
        clean = clean.replace('\u2022', '•')
        
        # FIX: Behalte wichtige Satzzeichen und deutsche Zeichen
        clean = re.sub(r'[^\w\s.,!?;:()\-\"\'äöüßÄÖÜ€%]', '', clean)
        
        return clean
    
    def _initialize_color_scheme(self, coding_map: Dict[str, Dict]) -> None:
        """
        FIX: Initialisiert Farbschema basierend auf gefundenen Kategorien
        """
        # FIX: Sammle alle verwendeten Hauptkategorien
        categories = set()
        for coding_info in coding_map.values():
            categories.add(coding_info['category'])
        
        # FIX: Nutze bestehende Farben vom ResultsExporter falls verfügbar
        if hasattr(self.results_exporter, 'category_colors') and self.results_exporter.category_colors:
            for category, hex_color in self.results_exporter.category_colors.items():
                self.category_colors[category] = self._hex_to_rgb(hex_color)
        
        # FIX: Ergänze fehlende Kategorien mit Standard-Farben
        available_colors = list(self.default_colors.values())
        color_index = 0
        
        for category in sorted(categories):
            if category not in self.category_colors:
                if color_index < len(available_colors):
                    self.category_colors[category] = available_colors[color_index]
                    color_index += 1
                else:
                    # FIX: Generiere zufällige Pastellfarbe falls Standard-Farben aufgebraucht
                    import random
                    self.category_colors[category] = (
                        0.7 + random.random() * 0.3,  # Hell-Bereich 0.7-1.0
                        0.7 + random.random() * 0.3,
                        0.7 + random.random() * 0.3
                    )
        
        print(f"   🎨 Farbschema initialisiert für {len(self.category_colors)} Kategorien")
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[float, float, float]:
        """
        FIX: Konvertiert Hex-Farbe zu RGB (0-1 Bereich für PyMuPDF)
        """
        if hex_color.startswith('#'):
            hex_color = hex_color[1:]
        
        try:
            r = int(hex_color[0:2], 16) / 255.0
            g = int(hex_color[2:4], 16) / 255.0
            b = int(hex_color[4:6], 16) / 255.0
            return (r, g, b)
        except:
            return (0.8, 0.8, 0.8)  # Fallback: Grau
    
    def _annotate_page(self, page, coding_map: Dict[str, Dict], page_num: int) -> int:
        """
        FIX: Überarbeitete Annotation mit präziser Höhen-Kalibrierung
        """
        annotations_added = 0
        page_text = page.get_text()
        
        print(f"\n   📄 Annotiere Seite {page_num} (Text-Länge: {len(page_text)} Zeichen)")
        
        text_groups = self._group_codings_by_original_text(coding_map)
        print(f"      📋 {len(text_groups)} Text-Gruppen zu verarbeiten")
        
        for i, (original_text, codings_group) in enumerate(text_groups.items(), 1):
            print(f"\n      [SEARCH] Gruppe {i}/{len(text_groups)}: {len(codings_group)} Kodierungen")
            print(f"          Text-Länge: {len(original_text)} Zeichen")
            
            # FIX: Strategie 1 - Exakte Text-Grenzen durch Content-Matching
            y_start, y_end, x_left = self._find_exact_text_boundaries_by_content(page, original_text)
            
            if y_start is None:
                # FIX: Strategie 2 - Fallback mit Text-Position-Schätzung
                print(f"          ⚠️ Exakte Grenzen nicht gefunden, verwende Positions-Schätzung")
                matches = self._find_text_matches(page_text, original_text, original_text)
                
                if matches:
                    match_start, match_end = matches[0]
                    y_start, y_end, x_left = self._estimate_chunk_boundaries_from_text_position(
                        page, match_start, match_end, original_text
                    )
                else:
                    print(f"          ❌ Auch Fallback-Strategie fehlgeschlagen")
                    continue
            
            # FIX: Erstelle präzise, nicht-überlappende Sidebar-Balken
            sidebar_rects = self._create_non_overlapping_sidebar_rectangles(y_start, y_end, x_left, codings_group)
            
            # FIX: Erstelle Annotationen
            for rect_idx, (rect, coding_info) in enumerate(sidebar_rects):
                category = coding_info['category']
                color = self.category_colors.get(category, (0.8, 0.8, 0.8))
                
                # FIX: Sidebar-Balken mit hoher Deckkraft
                sidebar_annot = page.add_rect_annot(rect)
                sidebar_annot.set_colors({"stroke": color, "fill": color})
                sidebar_annot.set_opacity(0.9)
                
                # FIX: Annotation mit Details
                annotation_text = self._create_annotation_text(
                    coding_info, 
                    is_multiple=(len(codings_group) > 1), 
                    instance_nr=rect_idx+1
                )
                sidebar_annot.set_info(content=annotation_text)
                sidebar_annot.update()
                
                annotations_added += 1
                # print(f"            ✅ Balken erstellt: {category} (Höhe: {rect.height:.1f})")
        
        print(f"\n   📊 Seite {page_num}: {annotations_added} präzise Balken hinzugefügt")
        return annotations_added
    
    def _group_codings_by_original_text(self, coding_map: Dict[str, Dict]) -> Dict[str, List[Dict]]:
        """
        FIX: Gruppiert Kodierungen nach ursprünglichem Text für Mehrfachkodierungs-Behandlung
        """
        text_groups = {}
        
        for clean_text, coding_info in coding_map.items():
            # FIX: Extrahiere ursprüngliche Segment-ID ohne Mehrfachkodierungs-Suffix
            segment_id = coding_info['segment_id']
            original_segment_id = self._extract_original_segment_id(segment_id)
            
            # FIX: Gruppe nach ursprünglichem Text
            original_text = coding_info['original_text']
            
            if original_text not in text_groups:
                text_groups[original_text] = []
            
            text_groups[original_text].append(coding_info)
        
        return text_groups
    
    def _extract_original_segment_id(self, segment_id: str) -> str:
        """
        FIX: Extrahiert ursprüngliche Segment-ID (entfernt Mehrfachkodierungs-Suffixe)
        
        Beispiele:
        - "TEDFWI-1-01" → "TEDFWI-1"
        - "doc_chunk_5-02" → "doc_chunk_5"
        """
        if '-' in segment_id:
            parts = segment_id.rsplit('-', 1)
            if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) <= 2:
                return parts[0]
        return segment_id
    
    def _create_multiple_coding_rectangles(self, base_rects: List, codings_group: List[Dict], page) -> List[Tuple]:
        """
        FIX: Erstellt Sidebar-Rechtecke für Mehrfachkodierung (vereinfacht)
        
        Bei mehr als einer Kodierung pro Segment: Sofort Sidebar-Markers verwenden
        """
        return self._create_sidebar_rectangles(base_rects, codings_group, page)
    
    def _create_sidebar_rectangles(self, base_rects: List, codings_group: List[Dict], page) -> List[Tuple]:
        """
        FIX: Korrigierte Sidebar-Rechtecke - schmale Streifen AM RAND der Text-Rechtecke
        """
        result_rects = []
        
        for i, coding_info in enumerate(codings_group):
            for base_rect in base_rects:
                # FIX: Schmale Streifen direkt am linken Rand des Text-Bereichs
                bar_width = 5    # Schmal aber sichtbar
                spacing = 1      # Minimaler Abstand
                
                # FIX: Positionierung INNERHALB des Text-Bereichs am linken Rand
                bar_rect = fitz.Rect(
                    base_rect.x0 + (bar_width + spacing) * i,     # Von links nach rechts innerhalb des Textes
                    base_rect.y0,                                  # Gleiche Höhe wie Text
                    base_rect.x0 + (bar_width + spacing) * i + bar_width,  # Breite des Balkens
                    base_rect.y1                                   # Gleiche Höhe wie Text
                )
                
                result_rects.append((bar_rect, coding_info))
                print(f"            📏 Sidebar {i+1}: {bar_rect} für {coding_info['category']}")
        
        return result_rects
    
    def _create_multiple_coding_highlight(self, page, rect, color, instance_nr: int, total_instances: int):
        """
        FIX: Erstellt Rechteck-Annotation für Sidebar-Mehrfachkodierung (vereinfacht)
        """
        # FIX: Verwende immer Rechteck-Annotation für Sidebar-Markers
        highlight = page.add_rect_annot(rect)
        
        # FIX: Setze Farbe mit hoher Deckkraft für bessere Sichtbarkeit
        highlight.set_colors({"stroke": color, "fill": color})
        highlight.set_opacity(0.8)  # 80% Deckkraft
        
        return highlight
    
    def _find_exact_text_boundaries_by_content(self, page, target_text: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        FIX: Findet exakte Text-Grenzen durch direkten Text-Abgleich
        
        Returns:
            Tuple[y_start, y_end, x_left] oder (None, None, None)
        """
        words = page.get_text("words")  # [(x0, y0, x1, y1, "word", block_no, line_no, word_no)]
        page_text = page.get_text()
        
        # FIX: Bereite Target-Text vor
        target_clean = self._clean_text_for_matching(target_text)
        target_words = [w.strip() for w in target_clean.split() if len(w.strip()) > 2]
        
        if len(target_words) < 5:
            print(f"          ❌ Zu wenige charakteristische Wörter: {len(target_words)}")
            return None, None, None
        
        print(f"          [SEARCH] Suche {len(target_words)} Ziel-Wörter im PDF...")
        
        # FIX: Finde zusammenhängenden Textbereich
        # Strategie: Suche nach einer Sequenz von mindestens 5 aufeinanderfolgenden Wörtern
        best_match_start = None
        best_match_end = None
        best_match_score = 0
        
        # FIX: Sliding Window über PDF-Wörter
        for start_idx in range(len(words) - 5):
            # Nimm 10-Wort-Fenster für Vergleich
            window_size = min(10, len(words) - start_idx)
            pdf_window_words = []
            
            for i in range(start_idx, start_idx + window_size):
                pdf_word = words[i][4].lower().strip()
                if len(pdf_word) > 2:
                    pdf_window_words.append(pdf_word)
            
            if len(pdf_window_words) < 5:
                continue
            
            # FIX: Berechne Übereinstimmung mit Target-Wörtern
            matches = 0
            for target_word in target_words[:10]:  # Erste 10 Target-Wörter
                if any(target_word.lower() in pdf_word or pdf_word in target_word.lower() 
                       for pdf_word in pdf_window_words):
                    matches += 1
            
            match_score = matches / min(len(target_words), 10)
            
            # FIX: Wenn gute Übereinstimmung gefunden
            if match_score > 0.6 and match_score > best_match_score:
                best_match_score = match_score
                best_match_start = start_idx
                # Schätze Ende basierend auf Target-Länge
                estimated_length = min(len(target_words), 50)  # Maximal 50 Wörter pro Chunk
                best_match_end = min(start_idx + estimated_length, len(words) - 1)
                
                print(f"          ✅ Match gefunden: Score={match_score:.2f}, Wörter {start_idx}-{best_match_end}")
        
        if best_match_start is None:
            print(f"          ❌ Kein ausreichender Match gefunden (bester Score: {best_match_score:.2f})")
            return None, None, None
        
        # FIX: Bestimme exakte Koordinaten des gefundenen Bereichs
        start_word = words[best_match_start]
        end_word = words[best_match_end]
        
        y_start = start_word[1]  # y0 des ersten Wortes
        y_end = end_word[3]      # y1 des letzten Wortes
        x_left = min(start_word[0], end_word[0])  # Linkeste x-Koordinate
        
        # FIX: Erweitere um ein wenig Puffer für bessere Sichtbarkeit
        line_height = abs(y_end - y_start) / max(1, (best_match_end - best_match_start) // 8)  # Geschätzte Zeilenhöhe
        y_start = y_start - line_height * 0.1  # 10% Puffer oben
        y_end = y_end + line_height * 0.1      # 10% Puffer unten
        
        # print(f"          📐 Exakte Grenzen: Y={y_start:.1f}-{y_end:.1f} (Höhe: {y_end-y_start:.1f}), X={x_left:.1f}")
        
        # FIX: Validiere und korrigiere Koordinaten
        if y_start > y_end:
            print(f"          🔧 Korrigiere vertauschte Y-Koordinaten: {y_start:.1f} ↔ {y_end:.1f}")
            y_start, y_end = y_end, y_start
        
        # FIX: Mindesthöhe sicherstellen
        min_height = 20  # Mindestens 20 Pixel Höhe
        if (y_end - y_start) < min_height:
            print(f"          🔧 Erweitere zu geringe Höhe von {y_end - y_start:.1f} auf {min_height}")
            center_y = (y_start + y_end) / 2
            y_start = center_y - min_height / 2
            y_end = center_y + min_height / 2
        
        # FIX: Prüfe auf gültige Werte
        if any(coord is None or not (-10000 < coord < 10000) for coord in [y_start, y_end, x_left]):
            print(f"          ❌ Ungültige Koordinaten erkannt, verwende Fallback")
            return None, None, None
        
        # print(f"          ✅ Validierte Grenzen: Y={y_start:.1f}-{y_end:.1f} (Höhe: {y_end-y_start:.1f}), X={x_left:.1f}")
        
        return y_start, y_end, x_left
    
    def _find_exact_chunk_boundaries(self, page, target_text: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        FIX: Findet exakte Start- und End-Koordinaten eines Text-Chunks
        
        Returns:
            Tuple[y_start, y_end, x_left, x_right] oder (None, None, None, None) falls nicht gefunden
        """
        words = page.get_text("words")
        page_text = page.get_text()
        
        # FIX: Bereite Suchtext vor - erste und letzte charakteristische Wörter
        target_words = target_text.split()
        if len(target_words) < 3:
            return None, None, None, None
        
        # FIX: Charakteristische Wörter am Anfang und Ende
        start_keywords = []
        end_keywords = []
        
        # Sammle erste 5 längere Wörter (> 4 Zeichen) 
        for word in target_words[:10]:
            if len(word) > 4 and word.isalpha():
                start_keywords.append(word.lower())
                if len(start_keywords) >= 3:
                    break
        
        # Sammle letzte 5 längere Wörter (> 4 Zeichen)
        for word in reversed(target_words[-10:]):
            if len(word) > 4 and word.isalpha():
                end_keywords.append(word.lower())
                if len(end_keywords) >= 3:
                    break
        
        if not start_keywords or not end_keywords:
            return None, None, None, None
        
        print(f"          🎯 Suche Start-Wörter: {start_keywords}")
        print(f"          🎯 Suche End-Wörter: {end_keywords}")
        
        # FIX: Finde Start- und End-Positionen
        start_positions = []
        end_positions = []
        
        for word_info in words:
            word_text = word_info[4].lower()
            word_rect = fitz.Rect(word_info[:4])
            
            # Suche Start-Wörter
            if any(keyword in word_text or word_text in keyword for keyword in start_keywords):
                start_positions.append(word_rect)
                print(f"          ✅ Start-Wort gefunden: '{word_text}' an Y={word_rect.y0:.1f}")
            
            # Suche End-Wörter  
            if any(keyword in word_text or word_text in keyword for keyword in end_keywords):
                end_positions.append(word_rect)
                print(f"          ✅ End-Wort gefunden: '{word_text}' an Y={word_rect.y0:.1f}")
        
        if not start_positions or not end_positions:
            print(f"          ❌ Start oder End nicht gefunden (Start: {len(start_positions)}, End: {len(end_positions)})")
            return None, None, None, None
        
        # FIX: Bestimme Chunk-Grenzen
        # Nimm früheste Start-Position und späteste End-Position
        y_start = min(rect.y0 for rect in start_positions)
        y_end = max(rect.y1 for rect in end_positions)
        x_left = min(rect.x0 for rect in start_positions + end_positions)
        x_right = max(rect.x1 for rect in start_positions + end_positions)
        
        print(f"          📐 Chunk-Grenzen: Y={y_start:.1f}-{y_end:.1f}, X={x_left:.1f}-{x_right:.1f}")
        
        return y_start, y_end, x_left, x_right

    def _create_non_overlapping_sidebar_rectangles(self, y_start: float, y_end: float, x_left: float, codings_group: List[Dict]) -> List[Tuple]:
        """
        FIX: Erstellt nicht-überlappende Sidebar-Rechtecke
        """
        result_rects = []
        
        # FIX: Dynamische Balken-Größe basierend auf verfügbarer Höhe
        available_height = y_end - y_start
        num_codings = len(codings_group)
        
        # FIX: Berechne optimale Balken-Höhe
        if num_codings == 1:
            # Ein Balken: Nutze 80% der verfügbaren Höhe
            bar_height = available_height * 0.8
            bar_spacing = available_height * 0.1
        else:
            # Mehrere Balken: Teile Höhe gleichmäßig auf
            total_spacing = available_height * 0.2  # 20% für Abstände
            available_for_bars = available_height - total_spacing
            bar_height = available_for_bars / num_codings
            bar_spacing = total_spacing / (num_codings + 1)
        
        # FIX: Balken-Breite und Position
        bar_width = 12      # Sichtbare aber nicht störende Breite
        margin_from_text = 15  # Abstand vom Text
        
        # print(f"          📊 Balken-Layout: {num_codings} Balken, Höhe={bar_height:.1f}, Abstand={bar_spacing:.1f}")
        
        for i, coding_info in enumerate(codings_group):
            # FIX: Vertikale Position - von oben nach unten gestapelt
            bar_y_start = y_start + bar_spacing + i * (bar_height + bar_spacing)
            bar_y_end = bar_y_start + bar_height
            
            # FIX: Horizontale Position - links vom Text
            bar_x_start = x_left - margin_from_text - bar_width
            bar_x_end = x_left - margin_from_text
            
            # FIX: Stelle sicher, dass der Balken im sichtbaren Bereich ist
            if bar_x_start < 10:  # Mindestens 10px vom Seitenrand
                bar_x_start = 10
                bar_x_end = bar_x_start + bar_width
            
            bar_rect = fitz.Rect(bar_x_start, bar_y_start, bar_x_end, bar_y_end)
            result_rects.append((bar_rect, coding_info))
            
            # print(f"            📏 Balken {i+1}: {bar_rect} → {coding_info['category']}")
        
        return result_rects
    
    def _create_precise_sidebar_rectangles(self, y_start: float, y_end: float, x_left: float, codings_group: List[Dict]) -> List[Tuple]:
        """
        FIX: Erstellt präzise Sidebar-Rechtecke nur für den Chunk-Bereich
        """
        result_rects = []
        
        # FIX: Balken-Dimensionen
        bar_width = 8      # Sichtbar aber nicht störend
        spacing = 2        # Kleiner Abstand zwischen Balken
        margin = 10        # Abstand vom Text
        
        for i, coding_info in enumerate(codings_group):
            # FIX: Positionierung links vom Text-Bereich
            bar_x_start = x_left - margin - (bar_width + spacing) * (len(codings_group) - i)
            bar_x_end = bar_x_start + bar_width
            
            # FIX: Präzises Rechteck nur für diesen Chunk
            bar_rect = fitz.Rect(
                bar_x_start,    # Links vom Text
                y_start,        # Exakt Start des Chunks
                bar_x_end,      # Balken-Breite
                y_end           # Exakt Ende des Chunks
            )
            
            result_rects.append((bar_rect, coding_info))
            print(f"            📏 Präziser Sidebar {i+1}: {bar_rect} für {coding_info['category']}")
        
        return result_rects

    def _create_fallback_sidebar_rectangles(self, page, match_start: int, match_end: int, codings_group: List[Dict]) -> List[Tuple]:
        """
        FIX: Fallback für Sidebar-Rechtecke wenn präzise Erkennung fehlschlägt
        """
        result_rects = []
        page_rect = page.rect
        page_text = page.get_text()
        
        # FIX: Geschätzte Position basierend auf Text-Position
        text_ratio = match_start / max(1, len(page_text))
        
        # FIX: Realistische Schätzung für Chunk-Größe
        estimated_lines = min(10, max(3, (match_end - match_start) // 80))  # Etwa 80 Zeichen pro Zeile
        line_height = 12  # Typische Zeilenhöhe
        
        y_start = page_rect.y0 + 70 + (text_ratio * (page_rect.height - 140))
        y_end = y_start + (estimated_lines * line_height)
        x_left = page_rect.x0 + 50  # Typischer linker Textrand
        
        print(f"          📍 Fallback Chunk-Schätzung: Y={y_start:.1f}-{y_end:.1f}, Zeilen={estimated_lines}")
        
        # FIX: Erstelle Sidebar-Rechtecke
        bar_width = 8
        spacing = 2
        margin = 10
        
        for i, coding_info in enumerate(codings_group):
            bar_x_start = x_left - margin - (bar_width + spacing) * (len(codings_group) - i)
            bar_x_end = bar_x_start + bar_width
            
            bar_rect = fitz.Rect(bar_x_start, y_start, bar_x_end, y_end)
            result_rects.append((bar_rect, coding_info))
            print(f"            📏 Fallback Sidebar {i+1}: {bar_rect}")
        
        return result_rects
    
    
    def _estimate_chunk_boundaries_from_text_position(self, page, match_start: int, match_end: int, target_text: str) -> Tuple[float, float, float]:
        """
        FIX: Fallback-Methode mit besserer Schätzung
        """
        page_text = page.get_text()
        page_rect = page.rect
        
        # FIX: Genauere Positionsschätzung
        text_length = len(page_text)
        target_length = len(target_text)
        
        # Schätze Zeilen basierend auf Textlänge (etwa 80-100 Zeichen pro Zeile)
        estimated_lines = max(2, min(15, target_length // 85))
        line_height = 14  # Typische Zeilenhöhe in PDFs
        
        # Position im Dokument
        position_ratio = match_start / max(1, text_length)
        
        # FIX: Realistischer Y-Bereich
        text_area_top = page_rect.y0 + 60    # Typischer oberer Rand
        text_area_bottom = page_rect.y1 - 60  # Typischer unterer Rand
        text_area_height = text_area_bottom - text_area_top
        
        y_start = text_area_top + (position_ratio * text_area_height)
        y_end = y_start + (estimated_lines * line_height)
        
        # Stelle sicher, dass wir im Seitenbereich bleiben
        if y_end > text_area_bottom:
            y_end = text_area_bottom
            y_start = y_end - (estimated_lines * line_height)
        
        x_left = page_rect.x0 + 50  # Typischer linker Textrand
        
        print(f"          📍 Fallback-Schätzung: Y={y_start:.1f}-{y_end:.1f} (Zeilen: {estimated_lines}), X={x_left:.1f}")
        
        return y_start, y_end, x_left
    
    def _find_text_matches(self, page_text: str, clean_text: str, original_text: str) -> List[Tuple[int, int]]:
        """
        FIX: Vereinfachte Text-Suche nur für Fallback-Zwecke
        """
        matches = []
        
        # FIX: Nur einfache Strategien für Fallback
        # Strategie 1: Erste 100 Zeichen
        if len(clean_text) > 100:
            short_text = clean_text[:100]
            pos = page_text.find(short_text)
            if pos != -1:
                matches.append((pos, pos + len(clean_text)))
                print(f"      ✅ Fallback-Match gefunden an Position {pos}")
                return matches
        
        # Strategie 2: Erste 5 Wörter
        words = clean_text.split()[:5]
        if len(words) >= 3:
            search_phrase = ' '.join(words)
            pos = page_text.find(search_phrase)
            if pos != -1:
                estimated_end = pos + len(clean_text)
                matches.append((pos, estimated_end))
                print(f"      ✅ Wort-Fallback-Match gefunden an Position {pos}")
        
        return matches
    
    def _fuzzy_text_search(self, page_text: str, search_text: str, threshold: float = 0.7) -> List[Tuple[int, int]]:
        """
        FIX: Verbesserte Fuzzy-Suche mit anpassbarer Schwelle
        """
        from difflib import SequenceMatcher
        
        matches = []
        search_len = len(search_text)
        
        if search_len < 20:
            return matches
        
        # FIX: Suche in überlappenden Fenstern
        window_size = min(search_len, 200)  # Kleinere Fenster für besseres Matching
        step_size = window_size // 4        # Überlappung für bessere Abdeckung
        
        for i in range(0, len(page_text) - window_size + 1, step_size):
            candidate = page_text[i:i + window_size]
            similarity = SequenceMatcher(None, search_text[:window_size], candidate).ratio()
            
            if similarity >= threshold:
                matches.append((i, i + window_size))
                print(f"      ✅ Fuzzy-Match (Ähnlichkeit: {similarity:.2f}) an Position {i}")
                break  # Nehme nur den ersten guten Match
        
        return matches

    def _get_text_rectangles(self, page, match_start: int, match_end: int) -> List:
        """
        FIX: Korrigierte Text-Rechteck-Erkennung - nur der eigentliche Text-Bereich
        """
        rects = []
        
        try:
            # FIX: Nutze get_text("words") für präzise Koordinaten
            words = page.get_text("words")
            page_text = page.get_text()
            
            if match_start < 0 or match_end > len(page_text):
                return self._create_fallback_rect(page, match_start, match_end)
            
            target_text = page_text[match_start:match_end]
            
            # FIX: Erste und letzte Wörter des Targets finden
            target_words = target_text.split()
            if len(target_words) < 2:
                return self._create_fallback_rect(page, match_start, match_end)
            
            first_words = target_words[:3]  # Erste 3 Wörter
            last_words = target_words[-3:]  # Letzte 3 Wörter
            
            print(f"          🎯 Suche Bereich: '{' '.join(first_words)}' bis '{' '.join(last_words)}'")
            
            # FIX: Finde Start- und End-Koordinaten
            start_rects = []
            end_rects = []
            
            for word_info in words:
                word_text = word_info[4].lower()
                word_rect = fitz.Rect(word_info[:4])
                
                # Suche nach ersten Wörtern
                if any(first_word.lower() in word_text or word_text in first_word.lower() 
                       for first_word in first_words):
                    start_rects.append(word_rect)
                
                # Suche nach letzten Wörtern
                if any(last_word.lower() in word_text or word_text in last_word.lower() 
                       for last_word in last_words):
                    end_rects.append(word_rect)
            
            if start_rects and end_rects:
                # FIX: Bestimme Text-Bereich von erstem bis letztem gefundenen Wort
                min_x0 = min(rect.x0 for rect in start_rects + end_rects)
                min_y0 = min(rect.y0 for rect in start_rects + end_rects)
                max_x1 = max(rect.x1 for rect in start_rects + end_rects)
                max_y1 = max(rect.y1 for rect in start_rects + end_rects)
                
                # FIX: Begrenzter Text-Bereich (nicht die ganze Seite)
                text_rect = fitz.Rect(
                    min_x0,      # Linke Kante des Textes
                    min_y0,      # Obere Kante des Textes  
                    max_x1,      # Rechte Kante des Textes
                    max_y1       # Untere Kante des Textes
                )
                
                rects.append(text_rect)
                print(f"          ✅ Präziser Text-Bereich: {text_rect}")
            else:
                return self._create_fallback_rect(page, match_start, match_end)
            
        except Exception as e:
            print(f"          ❌ Fehler bei präziser Koordinaten-Suche: {e}")
            return self._create_fallback_rect(page, match_start, match_end)
        
        return rects
    
    def _create_fallback_rect(self, page, match_start: int, match_end: int) -> List:
        """
        FIX: Realistisches Fallback-Rechteck basierend auf Seitenlayout
        """
        page_rect = page.rect
        
        # FIX: Typisches PDF-Layout - Text nimmt mittleren Bereich ein
        text_margin_left = 50
        text_margin_right = 50  
        text_margin_top = 70
        
        # FIX: Geschätzte Position basierend auf Text-Position
        page_text = page.get_text()
        text_ratio = match_start / max(1, len(page_text))
        
        # FIX: Y-Position basierend auf Position im Text
        available_height = page_rect.height - 2 * text_margin_top
        y_position = page_rect.y0 + text_margin_top + (text_ratio * available_height)
        
        # FIX: Realistisches Text-Rechteck
        fallback_rect = fitz.Rect(
            page_rect.x0 + text_margin_left,           # Realistischer linker Rand
            y_position,                                # Geschätzte Y-Position
            page_rect.x1 - text_margin_right,          # Realistischer rechter Rand
            y_position + 40                            # Moderater Höhe (etwa 2-3 Textzeilen)
        )
        
        print(f"          📍 Fallback Text-Rechteck: {fallback_rect}")
        return [fallback_rect]
    
    def _create_annotation_text(self, coding_info: Dict, is_multiple: bool = False, instance_nr: int = 1) -> str:
        """
        FIX: Erstellt Annotations-Text mit allen Kodierungs-Informationen
        """
        lines = []
        
        # FIX: Mehrfachkodierungs-Header
        if is_multiple:
            lines.append(f"🔄 MEHRFACHKODIERUNG - Teil {instance_nr}")
            lines.append("")
        
        # FIX: Hauptkategorie
        lines.append(f"📋 Kategorie: {coding_info['category']}")
        
        # FIX: Subkategorien
        if coding_info['subcategories']:
            subcats = ', '.join(coding_info['subcategories'])
            lines.append(f"🔖 Subkategorien: {subcats}")
        
        # FIX: Konfidenz
        confidence = coding_info.get('confidence', {})
        if isinstance(confidence, dict) and 'total' in confidence:
            lines.append(f"📊 Konfidenz: {confidence['total']:.2f}")
        
        # FIX: Begründung (gekürzt)
        justification = coding_info.get('justification', '')
        if justification:
            short_justification = justification[:200] + "..." if len(justification) > 200 else justification
            lines.append(f"💭 Begründung: {short_justification}")
        
        # FIX: Segment-ID
        lines.append(f"🔢 Segment: {coding_info['segment_id']}")
        
        return '\n'.join(lines)
    
    def _add_legend_page(self, doc) -> None:
        """
        FIX: Fügt Legende als erste Seite hinzu
        """
        # FIX: Erstelle neue Seite am Anfang
        legend_page = doc.new_page(0, width=595, height=842)  # A4 Format
        
        # FIX: Titel
        title_rect = fitz.Rect(50, 50, 545, 80)
        legend_page.insert_text(title_rect.tl, "QCA-AID Kategorien-Legende", 
                               fontsize=20, color=(0, 0, 0))
        
        # FIX: Legende für jede Kategorie
        y_pos = 120
        for category, color in self.category_colors.items():
            # FIX: Farbfeld
            color_rect = fitz.Rect(50, y_pos, 80, y_pos + 20)
            legend_page.draw_rect(color_rect, color=color, fill=color)
            
            # FIX: Kategorie-Name
            text_rect = fitz.Rect(90, y_pos, 500, y_pos + 20)
            legend_page.insert_text(text_rect.tl, category, 
                                   fontsize=12, color=(0, 0, 0))
            
            y_pos += 30
        
        # FIX: Anweisungen
        instructions = [
            "",
            "Anweisungen:",
            "• Highlights zeigen kodierte Textabschnitte",
            "• Klicken Sie auf Highlights für Details",
            "• Annotationen enthalten Kategorie, Subkategorien und Begründung",
            "• Bei Mehrfachkodierung: Farbbalken links neben dem Text",
            "",
            f"Erstellt mit QCA-AID • {len(self.category_colors)} Kategorien identifiziert"
        ]
        
        y_pos += 30
        for instruction in instructions:
            legend_page.insert_text((50, y_pos), instruction, 
                                   fontsize=10, color=(0.3, 0.3, 0.3))
            y_pos += 15
        
        # print("   📖 Legende als erste Seite hinzugefügt")


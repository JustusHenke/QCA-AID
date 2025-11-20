"""
Material Loader für QCA-AID
============================
Lädt und verarbeitet Interviewdokumente in verschiedenen Formaten.
"""

import os
import re
import spacy
from typing import List, Dict
from ..core.config import CONFIG


class MaterialLoader:
    """Lädt und verarbeitet Interviewdokumente."""
    
    def __init__(self, data_dir: str = CONFIG['DATA_DIR'], 
                 chunk_size: int = CONFIG['CHUNK_SIZE'], 
                 chunk_overlap: int = CONFIG['CHUNK_OVERLAP']):
        """
        Initialisiert den MaterialLoader.
        
        Args:
            data_dir (str): Verzeichnis mit den Dokumenten
            chunk_size (int): Ungefähre Anzahl der Zeichen pro Chunk
            chunk_overlap (int): Überlappung zwischen Chunks in Zeichen
        """
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Lade das deutsche Sprachmodell für spaCy
        try:
            self.nlp = spacy.load("de_core_news_sm")
        except Exception as e:
            print("Bitte installieren Sie das deutsche Sprachmodell:")
            print("python -m spacy download de_core_news_sm")
            raise e

    def chunk_text(self, text: str) -> List[str]:
        """
        Teilt Text in überlappende Chunks basierend auf Satzgrenzen.
        
        Args:
            text (str): Zu teilender Text
            
        Returns:
            List[str]: Liste der Text-Chunks
        """
        # Verarbeite den Text mit spaCy
        doc = self.nlp(text)
        sentences = list(doc.sents)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_text = sentence.text.strip()
            sentence_length = len(sentence_text)
            
            # Wenn der aktuelle Chunk mit diesem Satz zu groß würde
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Speichere aktuellen Chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Starte neuen Chunk mit Überlappung
                # Berechne wie viele Sätze wir für die Überlappung behalten
                overlap_length = 0
                overlap_sentences = []
                for sent in reversed(current_chunk):
                    overlap_length += len(sent)
                    if overlap_length > self.chunk_overlap:
                        break
                    overlap_sentences.insert(0, sent)
                
                current_chunk = overlap_sentences
                current_length = sum(len(s) for s in current_chunk)
            
            # Füge den Satz zum aktuellen Chunk hinzu
            current_chunk.append(sentence_text)
            current_length += sentence_length
        
        # Letzten Chunk hinzufügen, falls vorhanden
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
        
        print(f"\nChunking Ergebnis:")
        print(f"- Anzahl Chunks: {len(chunks)}")
        print(f"- Durchschnittliche Chunk-Länge: {sum(len(c) for c in chunks)/len(chunks):.0f} Zeichen")
        
        return chunks

    def clean_problematic_characters(self, text: str) -> str:
        """
        Bereinigt Text von problematischen Zeichen, die später beim Excel-Export
        zu Fehlern führen könnten.
        
        Args:
            text (str): Zu bereinigender Text
            
        Returns:
            str: Bereinigter Text
        """
        if not text:
            return ""
            
        # Entferne problematische Steuerzeichen
        cleaned_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\uFFFE\uFFFF]', '', text)
        
        # Ersetze bekannte problematische Sonderzeichen
        problematic_chars = ['☺', '☻', '♥', '♦', '♣', '♠']
        for char in problematic_chars:
            cleaned_text = cleaned_text.replace(char, '')
        
        return cleaned_text

    def preprocess_text(self, text: str) -> str:
        """
        Bereinigt und normalisiert Textdaten.
        
        Args:
            text (str): Zu bereinigender Text
            
        Returns:
            str: Bereinigter Text
        """
        if not text:
            return ""
            
        # Entferne überflüssige Whitespaces
        text = ' '.join(text.split())
        
        # Ersetze verschiedene Anführungszeichen durch einheitliche
        text = text.replace('"', '"').replace('"', '"')
        
        # Ersetze verschiedene Bindestriche durch einheitliche
        text = text.replace('–', '-').replace('—', '-')
        
        # Entferne spezielle Steuerzeichen und problematische Zeichen
        text = self.clean_problematic_characters(text)
        
        return text

    def load_documents(self) -> Dict[str, str]:
        """
        Sammelt und lädt alle unterstützten Dateien aus dem Verzeichnis.
        
        Returns:
            dict: Dictionary mit Dateinamen als Schlüssel und Dokumenteninhalt als Wert
        """
        documents = {}
        supported_extensions = {'.txt', '.docx', '.doc', '.pdf'}
        
        try:
            # Prüfe ob Verzeichnis existiert
            if not os.path.exists(self.data_dir):
                print(f"Warnung: Verzeichnis {self.data_dir} existiert nicht")
                os.makedirs(self.data_dir)
                return documents

            # Durchsuche Verzeichnis nach Dateien
            for file in os.listdir(self.data_dir):
                file_path = os.path.join(self.data_dir, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                if not os.path.isfile(file_path):
                    continue
                    
                print(f"Gefunden: {file} ({file_ext})")
                
                try:
                    if file_ext == '.txt':
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                    elif file_ext == '.docx':
                        from docx import Document
                        doc = Document(file_path)
                        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                    elif file_ext == '.doc':
                        # Benötigt antiword oder ähnliches Tool
                        import subprocess
                        text = subprocess.check_output(['antiword', file_path]).decode('utf-8')
                    elif file_ext == '.pdf':
                        import pypdf
                        with open(file_path, 'rb') as f:
                            pdf = pypdf.PdfReader(f)
                            text = '\n'.join([page.extract_text() for page in pdf.pages])
                    else:
                        print(f"Überspringe nicht unterstützte Datei: {file}")
                        continue
                        
                    # Bereinige und speichere Text
                    text = self.preprocess_text(text)
                    if text.strip():  # Nur nicht-leere Texte speichern
                        documents[file] = text
                        print(f"Erfolgreich geladen: {file} ({len(text)} Zeichen)")
                    else:
                        print(f"Warnung: Leerer Text in {file}")
                        
                except Exception as e:
                    print(f"Fehler beim Laden von {file}: {str(e)}")
                    continue
                    
            if not documents:
                print(f"\nKeine Dokumente gefunden in {self.data_dir}")
                print(f"Unterstützte Formate: {', '.join(supported_extensions)}")
            
        except Exception as e:
            print(f"Fehler beim Durchsuchen des Verzeichnisses: {str(e)}")
            
        return documents

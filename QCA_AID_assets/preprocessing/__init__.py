"""
Preprocessing-Module für QCA-AID
=================================
Enthält Klassen und Funktionen für das Laden und Vorbereiten von Analysematerial.
"""

from .material_loader import MaterialLoader
from .document_converter import (
    DocumentConverter,
    ConverterFactory,
    ConvertResult,
    ConvertError,
    StageError,
    ConversionStage,
    StoryStage,
    HtmlStoryStage,
    DocxTextStage,
)
from .document_store import DocumentStore, IngestResult
from .document_index import DocumentIndexer, IndexBuildError
from .page_aware_chunker import PageAwareChunker

__all__ = [
    'MaterialLoader',
    'DocumentConverter',
    'ConverterFactory',
    'ConvertResult',
    'ConvertError',
    'StageError',
    'ConversionStage',
    'StoryStage',
    'HtmlStoryStage',
    'DocxTextStage',
    'DocumentStore',
    'IngestResult',
    'DocumentIndexer',
    'IndexBuildError',
    'PageAwareChunker',
]

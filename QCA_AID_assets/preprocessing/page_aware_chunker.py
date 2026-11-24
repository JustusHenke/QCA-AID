"""
Page-Aware Chunker
==================
Teilt Dokumenttext in Chunks, die ihre Seitenzuordnung behalten
(siehe .kiro/specs/pdf-preview-annotation, Anforderungen 2.3, 2.4).

Statt spaCy-Satzgrenzen (MaterialLoader) arbeitet dieser Chunker leichtgewichtig
auf dem aus dem DocumentIndex rekonstruierten Gesamttext und bildet jeden
Chunk-Offset ueber `DocumentIndex.page_for_offset` auf eine Seite ab.

Ergebnis: TextChunk mit start/end, page_start/page_end und segment_id
(f"{doc_id}_chunk_{i}") - derselbe ID-Schema wie im bestehenden System.
"""

import re
from typing import List

from ..annotation.model import DocumentIndex, TextChunk

DEFAULT_CHUNK_SIZE = 1500
DEFAULT_OVERLAP = 200

_SENTENCE_END = re.compile(r"""[.!?]+["')\]]*\s+|\n{2,}""")


class PageAwareChunker:
    """Erzeugt Chunks mit erhaltener Seitenzuordnung."""

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP):
        if chunk_size <= 0:
            raise ValueError("chunk_size muss > 0 sein")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap muss >= 0 und < chunk_size sein")
        self.chunk_size = chunk_size
        self.overlap = overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_document(self, index: DocumentIndex) -> List[TextChunk]:
        """Chunked den Gesamttext eines DocumentIndex."""
        return self.chunk_text(index.full_text(), index, index.doc_id)

    def chunk_text(self, text: str, index: DocumentIndex, doc_id: str) -> List[TextChunk]:
        """Chunked einen Text und ordnet Offsets den Seiten zu."""
        spans = self._sentence_spans(text)
        chunks: List[TextChunk] = []
        n = len(spans)
        i = 0
        chunk_index = 0

        while i < n:
            start = spans[i][0]
            j = i
            end = spans[j][1]
            while j + 1 < n and (end - start) < self.chunk_size:
                j += 1
                end = spans[j][1]

            chunk_body = text[start:end]
            if chunk_body.strip():
                chunks.append(
                    TextChunk(
                        chunk_index=chunk_index,
                        segment_id=f"{doc_id}_chunk_{chunk_index}",
                        doc_id=doc_id,
                        text=chunk_body,
                        start=start,
                        end=end,
                        page_start=_page_for(index, start),
                        page_end=_page_for(index, max(start, end - 1)),
                    )
                )
                chunk_index += 1

            if j + 1 >= n:
                break

            # Overlap: naechster Start faellt hinter end-overlap zurueck,
            # aber nicht vor die erste Satzgrenze des aktuellen Chunks.
            target = max(start, end - self.overlap)
            k = j + 1
            while k > i + 1 and spans[k - 1][0] > target:
                k -= 1
            i = max(k, i + 1)

        return chunks

    # ------------------------------------------------------------------
    # Satzgrenzen
    # ------------------------------------------------------------------

    @staticmethod
    def _sentence_spans(text: str):
        spans = []
        start = 0
        for match in _SENTENCE_END.finditer(text):
            spans.append((start, match.end()))
            start = match.end()
        if start < len(text):
            spans.append((start, len(text)))
        if not spans:
            spans.append((0, len(text)))
        return spans


def _page_for(index: DocumentIndex, offset: int) -> int:
    """Seite fuer einen Offset; bei Separatoren die naechstgelegene Seite."""
    page = index.page_for_offset(offset)
    if page is not None:
        return page
    # Offset liegt in einem Seitentrenner oder ausserhalb: naechsten Marker nehmen.
    for marker in index.page_markers:
        if offset <= marker.start:
            return marker.page
    return index.page_markers[-1].page if index.page_markers else 1


__all__ = [
    "PageAwareChunker",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_OVERLAP",
]
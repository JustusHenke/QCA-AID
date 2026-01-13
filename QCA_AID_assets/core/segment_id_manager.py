"""
Segment ID Management System
============================
Centralized system for consistent segment ID management across the entire QCA-AID workflow.
"""

import os
import re
from typing import Tuple, Optional, Dict


class SegmentIDManager:
    """
    Central manager for all segment ID operations in QCA-AID.
    Provides consistent ID creation, parsing, and transformation methods.
    """
    
    @staticmethod
    def create_document_id(filename: str) -> str:
        """
        Creates document ID preserving original filename with extension.
        
        Args:
            filename: Original filename (e.g., "Lei_Bera_01.pdf")
            
        Returns:
            Document ID preserving extension (e.g., "Lei_Bera_01.pdf")
        """
        if not filename:
            return 'unknown_document'
        
        # Use the filename as-is to preserve extension
        return filename
    
    @staticmethod
    def create_segment_id(document_id: str, chunk_index: int) -> str:
        """
        Creates segment ID in standardized format.
        
        Args:
            document_id: Document identifier with extension
            chunk_index: Zero-based chunk index
            
        Returns:
            Segment ID in format: document_id_chunk_index
        """
        if not document_id:
            document_id = 'unknown_document'
        
        return f"{document_id}_chunk_{chunk_index}"
    
    @staticmethod
    def add_multiple_coding_suffix(base_segment_id: str, coding_index: int) -> str:
        """
        Adds multiple coding suffix to base segment ID.
        
        Args:
            base_segment_id: Base segment ID without suffix
            coding_index: Coding instance number (starts at 1)
            
        Returns:
            Segment ID with multiple coding suffix: base_id-coding_index
        """
        if not base_segment_id:
            base_segment_id = 'unknown_segment'
        
        return f"{base_segment_id}-{coding_index}"
    
    @staticmethod
    def extract_base_segment_id(segment_id: str) -> str:
        """
        Removes multiple coding suffix to get base segment ID.
        
        Args:
            segment_id: Segment ID potentially with suffix (e.g., "doc_chunk_0-1")
            
        Returns:
            Base segment ID without suffix (e.g., "doc_chunk_0")
        """
        if not segment_id:
            return 'unknown_segment'
        
        # Remove multiple coding suffix if present
        if '-' in segment_id and segment_id.split('-')[-1].isdigit():
            return '-'.join(segment_id.split('-')[:-1])
        
        return segment_id
    
    @staticmethod
    def extract_document_name(segment_id: str) -> str:
        """
        Extracts document name from segment ID.
        
        Args:
            segment_id: Segment ID (e.g., "Lei_Bera_01.pdf_chunk_0" or "Lei_Bera_01.pdf_chunk_0-1")
            
        Returns:
            Document name with extension (e.g., "Lei_Bera_01.pdf")
        """
        if not segment_id:
            return 'unknown_document'
        
        # First remove multiple coding suffix if present
        base_id = SegmentIDManager.extract_base_segment_id(segment_id)
        
        # Extract document name from base segment ID
        if '_chunk_' in base_id:
            return base_id.split('_chunk_')[0]
        
        # Fallback: take everything before the last underscore
        parts = base_id.rsplit('_', 1)
        return parts[0] if len(parts) > 1 else base_id
    
    @staticmethod
    def extract_chunk_index(segment_id: str) -> int:
        """
        Extracts chunk index from segment ID.
        
        Args:
            segment_id: Segment ID (e.g., "Lei_Bera_01.pdf_chunk_5" or "Lei_Bera_01.pdf_chunk_5-1")
            
        Returns:
            Zero-based chunk index (e.g., 5)
        """
        if not segment_id:
            return 0
        
        # First remove multiple coding suffix if present
        base_id = SegmentIDManager.extract_base_segment_id(segment_id)
        
        # Extract chunk index from base segment ID
        if '_chunk_' in base_id:
            try:
                chunk_part = base_id.split('_chunk_')[1]
                return int(chunk_part)
            except (IndexError, ValueError):
                return 0
        
        return 0
    
    @staticmethod
    def extract_multiple_coding_index(segment_id: str) -> Optional[int]:
        """
        Extracts multiple coding index from segment ID.
        
        Args:
            segment_id: Segment ID potentially with suffix (e.g., "doc_chunk_0-2")
            
        Returns:
            Multiple coding index if present, None otherwise
        """
        if not segment_id or '-' not in segment_id:
            return None
        
        suffix = segment_id.split('-')[-1]
        if suffix.isdigit():
            return int(suffix)
        
        return None
    
    @staticmethod
    def is_multiple_coding(segment_id: str) -> bool:
        """
        Checks if segment ID represents a multiple coding instance.
        
        Args:
            segment_id: Segment ID to check
            
        Returns:
            True if segment ID has multiple coding suffix
        """
        return SegmentIDManager.extract_multiple_coding_index(segment_id) is not None
    
    @staticmethod
    def standardize_segment_id(segment_id: str) -> str:
        """
        Standardizes segment ID to ensure consistent format.
        
        Args:
            segment_id: Segment ID in any format
            
        Returns:
            Standardized segment ID
        """
        if not segment_id:
            return 'unknown_segment'
        
        # If already in correct format, return as-is
        if '_chunk_' in segment_id:
            return segment_id
        
        # For legacy or malformed IDs, try to reconstruct
        # This is a fallback - ideally all IDs should be created through this manager
        return segment_id
    
    @staticmethod
    def extract_document_attributes(doc_name: str) -> Tuple[str, str, str]:
        """
        Dynamically extracts document attributes without hardcoded extensions.
        
        Args:
            doc_name: Document name with or without extension
            
        Returns:
            Tuple of (attribut1, attribut2, attribut3) extracted from filename
        """
        if not doc_name:
            return '', '', ''
        
        # Remove file extension dynamically using os.path.splitext
        name_without_ext, extension = os.path.splitext(doc_name)
        
        # Split by underscore for attributes
        parts = name_without_ext.split('_')
        
        attribut1 = parts[0] if len(parts) > 0 else ''
        attribut2 = parts[1] if len(parts) > 1 else ''
        attribut3 = parts[2] if len(parts) > 2 else ''
        
        return attribut1, attribut2, attribut3
    
    @staticmethod
    def create_display_id(segment_id: str) -> str:
        """
        Creates display-friendly ID for export (e.g., "LEI_BERA_01-0-1").
        
        Args:
            segment_id: Internal segment ID
            
        Returns:
            Display-friendly ID for export
        """
        if not segment_id:
            return 'UNKNOWN-0-0'
        
        # Extract components
        doc_name = SegmentIDManager.extract_document_name(segment_id)
        chunk_idx = SegmentIDManager.extract_chunk_index(segment_id)
        coding_idx = SegmentIDManager.extract_multiple_coding_index(segment_id) or 0
        
        # Extract attributes for display format
        attr1, attr2, attr3 = SegmentIDManager.extract_document_attributes(doc_name)
        
        # Create display prefix (first 2 letters of each attribute)
        aa = SegmentIDManager._extract_first_letters(attr1, 2)
        bb = SegmentIDManager._extract_first_letters(attr2, 2)
        cc = SegmentIDManager._extract_first_letters(attr3, 2)
        
        prefix = f"{aa}{bb}{cc}".upper()
        
        # Format: PREFIX-CHUNK-CODING
        return f"{prefix}-{chunk_idx:02d}-{coding_idx:02d}"
    
    @staticmethod
    def _extract_first_letters(text: str, count: int) -> str:
        """
        Helper method to extract first N letters from text.
        
        Args:
            text: Input text
            count: Number of letters to extract
            
        Returns:
            First N letters, padded with 'X' if needed
        """
        if not text:
            return 'X' * count
        
        # Extract only letters
        letters = re.sub(r'[^a-zA-Z]', '', text)
        if not letters:
            return 'X' * count
        
        # Take first N letters and pad if necessary
        result = letters[:count].upper()
        while len(result) < count:
            result += 'X'
        
        return result
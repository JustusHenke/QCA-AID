"""
Reliability Database
===================
Persistent storage and management for intercoder reliability data.
Supports both automatic and manual coding results for reliability analysis.
Stores data as JSON in all_codings.json file.
"""

import json
import logging
import time
import random
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from pathlib import Path
import os

# Optional import for cloud sync detection
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from ..core.data_models import ExtendedCodingResult


def _get_project_root() -> Path:
    """
    Get the project root directory using the same logic as main.py.
    
    Priority:
    1. Environment variable QCA_AID_PROJECT_ROOT
    2. .qca-aid-project.json in repository root
    3. Repository root (fallback)
    """
    # Get repository root (where QCA-AID.py is located)
    repo_root = Path(__file__).resolve().parent.parent.parent
    
    # Check environment variable
    env_root = os.environ.get('QCA_AID_PROJECT_ROOT')
    if env_root:
        candidate = Path(env_root).expanduser()
        if candidate.exists():
            return candidate
    
    # Check .qca-aid-project.json
    settings_path = repo_root / ".qca-aid-project.json"
    if settings_path.exists():
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            project_root_value = data.get("project_root")
            if project_root_value:
                candidate = Path(project_root_value).expanduser()
                if candidate.exists():
                    return candidate
        except Exception:
            pass
    
    # Fallback to repository root
    return repo_root

logger = logging.getLogger(__name__)


class ReliabilityDatabase:
    """
    JSON-based persistent storage for intercoder reliability data.
    
    Features:
    - JSON-based persistent storage in all_codings.json
    - Support for automatic and manual coding results
    - Efficient querying by segment, coder, or analysis mode
    - Data export/import functionality
    - Human-readable format
    """
    
    def __init__(self, db_path: str = "output/all_codings.json", 
                 use_timestamped_filename: bool = False, analysis_mode: str = None):
        """
        Initialize reliability database.
        
        Args:
            db_path: Path to JSON database file (default: output/all_codings.json)
                    Relative paths are resolved relative to the configured project root.
            use_timestamped_filename: If True, use timestamped filename matching Excel export
            analysis_mode: Analysis mode for timestamped filename
        """
        # Ensure the path is relative to configured project root
        if not os.path.isabs(db_path):
            # Get the configured project root (same logic as main.py)
            project_root = _get_project_root()
            
            # If using timestamped filename, generate it to match Excel export
            if use_timestamped_filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                analysis_mode = analysis_mode or 'deductive'
                filename = f"QCA-AID_Analysis_{analysis_mode}_{timestamp}_all_codings.json"
                
                # Extract directory from db_path, use it instead of hardcoded "output"
                if os.path.dirname(db_path):
                    output_dir = os.path.dirname(db_path)
                else:
                    output_dir = "output"
                
                self.db_path = project_root / output_dir / filename
            else:
                self.db_path = project_root / db_path
        else:
            # For absolute paths with timestamped filename
            if use_timestamped_filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                analysis_mode = analysis_mode or 'deductive'
                filename = f"QCA-AID_Analysis_{analysis_mode}_{timestamp}_all_codings.json"
                
                # Use the directory from the absolute path
                output_dir = os.path.dirname(db_path)
                self.db_path = Path(output_dir) / filename
            else:
                self.db_path = Path(db_path)
        
        # Create parent directories if they don't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database structure
        self.data = {
            'metadata': {
                'version': '1.0',
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            },
            'coding_results': []
        }
        
        # For timestamped files, start fresh (don't load existing data)
        if not use_timestamped_filename:
            # Load existing data if file exists
            self._load_data()
        
        logger.info(f"ReliabilityDatabase initialized at {self.db_path}")
        logger.debug(f"Project root: {_get_project_root()}")
    
    def _load_data(self) -> None:
        """Load data from JSON file if it exists."""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                
                # Merge with default structure
                if isinstance(loaded_data, dict):
                    self.data.update(loaded_data)
                    # Ensure metadata exists
                    if 'metadata' not in self.data:
                        self.data['metadata'] = {
                            'version': '1.0',
                            'created_at': datetime.now().isoformat(),
                            'last_updated': datetime.now().isoformat()
                        }
                    # Ensure coding_results exists
                    if 'coding_results' not in self.data:
                        self.data['coding_results'] = []
                
                logger.info(f"Loaded {len(self.data['coding_results'])} existing coding results")
                
            except Exception as e:
                logger.warning(f"Could not load existing data from {self.db_path}: {e}")
                # Keep default empty structure
    
    def _save_data(self) -> None:
        """
        Save data to JSON file with mandatory success requirement.
        
        This method MUST succeed before the analysis can continue.
        If saving fails, the entire analysis will be halted to prevent data loss.
        """
        try:
            # Update last_updated timestamp
            self.data['metadata']['last_updated'] = datetime.now().isoformat()
            
            # Write to temporary file first, then rename (atomic operation)
            temp_path = self.db_path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename with retry mechanism for Windows/Dropbox compatibility
            # This MUST succeed - if it fails, we raise an exception to halt analysis
            self._safe_replace_with_retry(temp_path, self.db_path)
            
            logger.debug(f"✅ Kodierungen erfolgreich gespeichert: {self.db_path}")
            
        except Exception as e:
            logger.error(f"❌ KRITISCH: Speichern der Kodierungen fehlgeschlagen: {self.db_path}: {e}")
            print(f"\n🚨 ANALYSE GESTOPPT: Kodierungen können nicht gespeichert werden!")
            print(f"   📁 Datei: {self.db_path}")
            print(f"   ❌ Fehler: {e}")
            print(f"   \n🔧 Bitte behebe das Speicherproblem und starte die Analyse erneut.")
            raise RuntimeError(f"Kodierungen können nicht gespeichert werden: {e}") from e
    
    def _detect_cloud_sync_processes(self) -> List[str]:
        """
        Detect running cloud synchronization processes that might interfere with file operations.
        
        Returns:
            List of detected cloud sync process names
        """
        if not PSUTIL_AVAILABLE:
            return []
            
        cloud_processes = [
            'dropbox', 'onedrive', 'googledrivesync', 'googledrive', 'icloud',
            'box', 'sync', 'backup', 'carbonite', 'crashplan'
        ]
        
        detected = []
        try:
            for proc in psutil.process_iter(['name']):
                proc_name = proc.info['name'].lower()
                for cloud_proc in cloud_processes:
                    if cloud_proc in proc_name:
                        detected.append(proc.info['name'])
                        break
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
        
        return detected

    def _safe_replace_with_retry(self, temp_path: Path, target_path: Path, max_retries: int = 10) -> None:
        """
        Safely replace target file with temp file using retry mechanism.
        
        This method handles Windows/Dropbox file locking issues by implementing
        exponential backoff with jitter for retry attempts.
        
        Args:
            temp_path: Path to temporary file
            target_path: Path to target file
            max_retries: Maximum number of retry attempts (default: 10)
            
        Raises:
            PermissionError: If all retry attempts fail
        """
        last_error = None
        cloud_sync_warning_shown = False
        
        for attempt in range(max_retries):
            try:
                # Try to remove target file first if it exists (Windows compatibility)
                if target_path.exists():
                    try:
                        target_path.unlink()
                    except PermissionError:
                        pass  # Will be handled by the replace operation
                
                temp_path.replace(target_path)
                
                # Success! Log if we had previous failures
                if attempt > 0:
                    logger.info(f"✅ File successfully saved after {attempt + 1} attempts")
                
                return  # Success!
                
            except PermissionError as e:
                last_error = e
                
                # Show cloud sync warning after first few attempts
                if attempt == 2 and not cloud_sync_warning_shown:
                    detected_processes = self._detect_cloud_sync_processes()
                    
                    print(f"\n⚠️  WARNUNG: Dateispeicherung blockiert!")
                    print(f"   📁 Datei: {target_path}")
                    print(f"   🔄 Mögliche Ursache: Cloud-Synchronisation")
                    
                    if detected_processes:
                        print(f"   🔍 Erkannte Cloud-Prozesse: {', '.join(detected_processes)}")
                        print(f"   💡 Lösung: Pausiere diese Cloud-Synchronisation temporär")
                    else:
                        print(f"   💡 Lösung: Pausiere Cloud-Synchronisation (Dropbox, OneDrive, etc.)")
                    
                    print(f"   ⏳ Versuche weiter automatisch zu speichern...")
                    cloud_sync_warning_shown = True
                
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter (longer delays for cloud sync)
                    base_delay = min(2 ** attempt, 30)  # Cap at 30 seconds
                    jitter = random.uniform(0, 2)  # More jitter for cloud sync
                    delay = base_delay + jitter
                    
                    logger.warning(
                        f"💾 Speichern fehlgeschlagen (Versuch {attempt + 1}/{max_retries}), "
                        f"wiederhole in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    # Final attempt failed - show user guidance
                    print(f"\n❌ KRITISCHER FEHLER: Kodierungen können nicht gespeichert werden!")
                    print(f"   📁 Datei: {target_path}")
                    print(f"   🔄 Nach {max_retries} Versuchen fehlgeschlagen")
                    print(f"   \n🛠️  SOFORTIGE MASSNAHMEN:")
                    print(f"   1. Pausiere Cloud-Synchronisation (Dropbox/OneDrive)")
                    print(f"   2. Schließe alle Programme, die auf Ausgabedateien zugreifen")
                    print(f"   3. Prüfe Schreibrechte im Ausgabeordner")
                    print(f"   4. Starte die Analyse erneut")
                    print(f"   \n⚠️  OHNE SPEICHERN GEHEN KODIERUNGEN VERLOREN!")
                    
                    logger.error(
                        f"File replace failed after {max_retries} attempts. "
                        f"Cloud sync or file access interference detected."
                    )
                    raise last_error
            
            except Exception as e:
                # Non-permission errors should not be retried
                logger.error(f"Unexpected error during file replace: {e}")
                raise
    
    def store_coding_result(self, coding_result: ExtendedCodingResult) -> None:
        """
        Store a coding result in the database.
        
        Args:
            coding_result: Extended coding result to store
        """
        # Convert to dictionary
        result_dict = {
            'segment_id': coding_result.segment_id,
            'coder_id': coding_result.coder_id,
            'category': coding_result.category,
            'subcategories': coding_result.subcategories or [],
            'confidence': coding_result.confidence,
            'justification': coding_result.justification or "",
            'analysis_mode': coding_result.analysis_mode,
            'timestamp': coding_result.timestamp.isoformat(),
            'is_manual': coding_result.is_manual,
            'metadata': coding_result.metadata or {},
            'created_at': datetime.now().isoformat()
        }
        
        # Remove existing result for same segment_id + coder_id combination
        self.data['coding_results'] = [
            r for r in self.data['coding_results'] 
            if not (r['segment_id'] == coding_result.segment_id and r['coder_id'] == coding_result.coder_id)
        ]
        
        # Add new result
        self.data['coding_results'].append(result_dict)
        
        # Save to file
        self._save_data()
        
        logger.debug(f"Stored coding result: segment={coding_result.segment_id}, coder={coding_result.coder_id}")
    
    def store_multiple_results(self, coding_results: List[ExtendedCodingResult]) -> None:
        """
        Store multiple coding results efficiently.
        
        Args:
            coding_results: List of coding results to store
        """
        if not coding_results:
            return
        
        # Convert all results to dictionaries
        new_results = []
        for result in coding_results:
            result_dict = {
                'segment_id': result.segment_id,
                'coder_id': result.coder_id,
                'category': result.category,
                'subcategories': result.subcategories or [],
                'confidence': result.confidence,
                'justification': result.justification or "",
                'analysis_mode': result.analysis_mode,
                'timestamp': result.timestamp.isoformat(),
                'is_manual': result.is_manual,
                'metadata': result.metadata or {},
                'created_at': datetime.now().isoformat()
            }
            new_results.append(result_dict)
        
        # Remove existing results for same segment_id + coder_id combinations
        existing_keys = {(r['segment_id'], r['coder_id']) for r in new_results}
        self.data['coding_results'] = [
            r for r in self.data['coding_results'] 
            if (r['segment_id'], r['coder_id']) not in existing_keys
        ]
        
        # Add new results
        self.data['coding_results'].extend(new_results)
        
        # Save to file - THIS MUST SUCCEED
        try:
            self._save_data()
            logger.info(f"✅ {len(coding_results)} Kodierungen erfolgreich gespeichert")
        except Exception as e:
            logger.error(f"❌ KRITISCH: Batch-Speicherung fehlgeschlagen: {e}")
            print(f"\n🚨 ANALYSE GESTOPPT: {len(coding_results)} Kodierungen können nicht gespeichert werden!")
            print(f"   🔧 Bitte behebe das Speicherproblem und starte die Analyse erneut.")
            # Re-raise to halt the analysis
            raise RuntimeError(f"Batch-Kodierungen können nicht gespeichert werden: {e}") from e
    
    def replace_all_results(self, coding_results: List[ExtendedCodingResult]) -> None:
        """
        Replace all coding results with new ones (clear and store).
        This ensures the JSON file doesn't grow with each run.
        
        Args:
            coding_results: List of coding results to store (replaces all existing)
        """
        if not coding_results:
            # Clear all existing results
            self.data['coding_results'] = []
            self._save_data()
            logger.info("✅ Alle Kodierungen gelöscht (leere Liste)")
            return
        
        # Convert all results to dictionaries
        new_results = []
        for result in coding_results:
            result_dict = {
                'segment_id': result.segment_id,
                'coder_id': result.coder_id,
                'category': result.category,
                'subcategories': result.subcategories or [],
                'confidence': result.confidence,
                'justification': result.justification or "",
                'analysis_mode': result.analysis_mode,
                'timestamp': result.timestamp.isoformat(),
                'is_manual': result.is_manual,
                'metadata': result.metadata or {},
                'created_at': datetime.now().isoformat()
            }
            new_results.append(result_dict)
        
        # Replace all existing results
        self.data['coding_results'] = new_results
        
        # Save to file - THIS MUST SUCCEED
        try:
            self._save_data()
            logger.info(f"✅ {len(coding_results)} Kodierungen erfolgreich ersetzt (alle vorherigen gelöscht)")
        except Exception as e:
            logger.error(f"❌ KRITISCH: Batch-Ersetzung fehlgeschlagen: {e}")
            print(f"\n🚨 ANALYSE GESTOPPT: {len(coding_results)} Kodierungen können nicht gespeichert werden!")
            print(f"   🔧 Bitte behebe das Speicherproblem und starte die Analyse erneut.")
            # Re-raise to halt the analysis
            raise RuntimeError(f"Batch-Kodierungen können nicht gespeichert werden: {e}") from e
    
    def get_coding_results(self, 
                          segment_ids: Optional[List[str]] = None,
                          coder_ids: Optional[List[str]] = None,
                          analysis_modes: Optional[List[str]] = None,
                          include_manual: bool = True,
                          include_automatic: bool = True) -> List[ExtendedCodingResult]:
        """
        Retrieve coding results with optional filtering.
        
        Args:
            segment_ids: Optional list of segment IDs to filter by
            coder_ids: Optional list of coder IDs to filter by
            analysis_modes: Optional list of analysis modes to filter by
            include_manual: Whether to include manual coding results
            include_automatic: Whether to include automatic coding results
            
        Returns:
            List of matching coding results
        """
        results = []
        
        for result_dict in self.data['coding_results']:
            # Apply filters
            if segment_ids and result_dict['segment_id'] not in segment_ids:
                continue
            
            if coder_ids and result_dict['coder_id'] not in coder_ids:
                continue
            
            if analysis_modes and result_dict['analysis_mode'] not in analysis_modes:
                continue
            
            # Filter by manual/automatic
            is_manual = result_dict.get('is_manual', False)
            if not include_manual and is_manual:
                continue
            if not include_automatic and not is_manual:
                continue
            
            # Convert back to ExtendedCodingResult
            try:
                result = ExtendedCodingResult(
                    segment_id=result_dict['segment_id'],
                    coder_id=result_dict['coder_id'],
                    category=result_dict['category'],
                    subcategories=result_dict.get('subcategories', []),
                    confidence=result_dict['confidence'],
                    justification=result_dict.get('justification', ""),
                    analysis_mode=result_dict['analysis_mode'],
                    timestamp=datetime.fromisoformat(result_dict['timestamp']),
                    is_manual=result_dict.get('is_manual', False),
                    metadata=result_dict.get('metadata', {})
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Could not parse coding result: {e}")
                continue
        
        # Sort by segment_id, coder_id, timestamp
        results.sort(key=lambda r: (r.segment_id, r.coder_id, r.timestamp))
        
        return results
    
    def get_reliability_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for reliability data.
        
        Returns:
            Dictionary with summary statistics
        """
        results = self.data['coding_results']
        
        if not results:
            return {
                'total_codings': 0,
                'total_segments': 0,
                'total_coders': 0,
                'manual_codings': 0,
                'automatic_codings': 0,
                'codings_per_coder': {},
                'analysis_modes': [],
                'segments_with_multiple_coders': 0
            }
        
        # Calculate statistics
        total_codings = len(results)
        
        segment_ids = set(r['segment_id'] for r in results)
        total_segments = len(segment_ids)
        
        coder_ids = set(r['coder_id'] for r in results)
        total_coders = len(coder_ids)
        
        manual_codings = sum(1 for r in results if r.get('is_manual', False))
        automatic_codings = total_codings - manual_codings
        
        # Codings per coder
        codings_per_coder = {}
        for r in results:
            coder_id = r['coder_id']
            codings_per_coder[coder_id] = codings_per_coder.get(coder_id, 0) + 1
        
        # Analysis modes
        analysis_modes = list(set(r['analysis_mode'] for r in results))
        
        # Segments with multiple coders
        segment_coders = {}
        for r in results:
            segment_id = r['segment_id']
            if segment_id not in segment_coders:
                segment_coders[segment_id] = set()
            segment_coders[segment_id].add(r['coder_id'])
        
        multi_coder_segments = sum(1 for coders in segment_coders.values() if len(coders) > 1)
        
        return {
            'total_codings': total_codings,
            'total_segments': total_segments,
            'total_coders': total_coders,
            'manual_codings': manual_codings,
            'automatic_codings': automatic_codings,
            'codings_per_coder': codings_per_coder,
            'analysis_modes': analysis_modes,
            'segments_with_multiple_coders': multi_coder_segments
        }
    
    def get_segments_for_reliability_analysis(self) -> List[str]:
        """
        Get list of segment IDs that have multiple coders (suitable for reliability analysis).
        
        Returns:
            List of segment IDs with multiple coders
        """
        segment_coders = {}
        for r in self.data['coding_results']:
            segment_id = r['segment_id']
            if segment_id not in segment_coders:
                segment_coders[segment_id] = set()
            segment_coders[segment_id].add(r['coder_id'])
        
        multi_coder_segments = [
            segment_id for segment_id, coders in segment_coders.items() 
            if len(coders) > 1
        ]
        
        return sorted(multi_coder_segments)
    
    def get_coders_for_segment(self, segment_id: str) -> List[str]:
        """
        Get list of coder IDs that have coded a specific segment.
        
        Args:
            segment_id: Segment ID to query
            
        Returns:
            List of coder IDs
        """
        coder_ids = set()
        for r in self.data['coding_results']:
            if r['segment_id'] == segment_id:
                coder_ids.add(r['coder_id'])
        
        return sorted(list(coder_ids))
    
    def delete_coding_results(self, 
                             segment_ids: Optional[List[str]] = None,
                             coder_ids: Optional[List[str]] = None) -> int:
        """
        Delete coding results with optional filtering.
        
        Args:
            segment_ids: Optional list of segment IDs to delete
            coder_ids: Optional list of coder IDs to delete
            
        Returns:
            Number of deleted records
        """
        original_count = len(self.data['coding_results'])
        
        if segment_ids and coder_ids:
            # Delete specific segment-coder combinations
            self.data['coding_results'] = [
                r for r in self.data['coding_results']
                if not (r['segment_id'] in segment_ids and r['coder_id'] in coder_ids)
            ]
        elif segment_ids:
            # Delete all results for specific segments
            self.data['coding_results'] = [
                r for r in self.data['coding_results']
                if r['segment_id'] not in segment_ids
            ]
        elif coder_ids:
            # Delete all results for specific coders
            self.data['coding_results'] = [
                r for r in self.data['coding_results']
                if r['coder_id'] not in coder_ids
            ]
        else:
            # Delete all results
            self.data['coding_results'] = []
        
        deleted_count = original_count - len(self.data['coding_results'])
        
        if deleted_count > 0:
            self._save_data()
        
        logger.info(f"Deleted {deleted_count} coding results")
        return deleted_count
    
    def export_to_json(self, filepath: str, 
                      segment_ids: Optional[List[str]] = None,
                      coder_ids: Optional[List[str]] = None) -> None:
        """
        Export reliability data to JSON file.
        
        Args:
            filepath: Path to export file
            segment_ids: Optional list of segment IDs to export
            coder_ids: Optional list of coder IDs to export
        """
        results = self.get_coding_results(segment_ids=segment_ids, coder_ids=coder_ids)
        summary = self.get_reliability_summary()
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'database_path': str(self.db_path),
            'summary': summary,
            'coding_results': [result.to_dict() for result in results]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(results)} coding results to {filepath}")
    
    def import_from_json(self, filepath: str, clear_existing: bool = False) -> int:
        """
        Import reliability data from JSON file.
        
        Args:
            filepath: Path to import file
            clear_existing: Whether to clear existing data before import
            
        Returns:
            Number of imported records
        """
        if clear_existing:
            self.delete_coding_results()
        
        with open(filepath, 'r', encoding='utf-8') as f:
            import_data = json.load(f)
        
        # Import coding results
        results_data = import_data.get('coding_results', [])
        results = [ExtendedCodingResult.from_dict(result_data) for result_data in results_data]
        
        self.store_multiple_results(results)
        
        logger.info(f"Imported {len(results)} coding results from {filepath}")
        return len(results)
    
    def backup_database(self, backup_path: str) -> None:
        """
        Create a backup of the database.
        
        Args:
            backup_path: Path for backup file
        """
        import shutil
        
        backup_path = Path(backup_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(self.db_path, backup_path)
        logger.info(f"Database backed up to {backup_path}")
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get information about the database.
        
        Returns:
            Dictionary with database information
        """
        # Get database size
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        
        return {
            'database_path': str(self.db_path),
            'database_size_bytes': db_size,
            'database_size_mb': db_size / (1024 * 1024),
            'version': self.data['metadata'].get('version', 'unknown'),
            'created_at': self.data['metadata'].get('created_at'),
            'last_updated': self.data['metadata'].get('last_updated'),
            'format': 'JSON',
            'summary': self.get_reliability_summary()
        }
    
    def close(self) -> None:
        """Close database connections (cleanup method)."""
        # JSON files don't need explicit closing
        # This method is provided for interface compatibility
        logger.debug("ReliabilityDatabase closed")
"""
Stage 8: Embeddings Generation
Processes Stage 7 summarized content to create embeddings for each chunk.
Self-contained standalone script that loads config from NAS at runtime.

Architecture based on Stage 7 pattern with embeddings generation logic.
Uses per-transcript OAuth refresh, incremental saving, and enhanced error logging.
"""

import os
import tempfile
import logging
import json
import time
from datetime import datetime
from urllib.parse import quote
from typing import Dict, Any, Optional, List, Tuple
import io
import re
import requests

import yaml
from smb.SMBConnection import SMBConnection
from dotenv import load_dotenv
from openai import OpenAI
from collections import defaultdict
import tiktoken
import csv

# Load environment variables
load_dotenv()

# Global variables
config = {}
logger = None
execution_log = []  # Detailed execution log entries
error_log = []  # Error log entries (only if errors occur)

# LLM-specific globals
llm_client = None
oauth_token = None
ssl_cert_path = None
tokenizer = None


def setup_logging() -> logging.Logger:
    """Set up minimal console logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def log_console(message: str, level: str = "INFO"):
    """Log minimal message to console."""
    global logger
    if level == "ERROR":
        logger.error(message)
    elif level == "WARNING":
        logger.warning(message)
    else:
        logger.info(message)


def log_execution(message: str, details: Dict[str, Any] = None):
    """Log detailed execution information for main log file."""
    global execution_log
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "message": message,
        "details": details or {},
    }
    execution_log.append(log_entry)


def log_error(message: str, error_type: str, details: Dict[str, Any] = None):
    """Log error information for error log file."""
    global error_log
    error_entry = {
        "timestamp": datetime.now().isoformat(),
        "error_type": error_type,
        "message": message,
        "details": details or {},
    }
    error_log.append(error_entry)


class EnhancedErrorLogger:
    """Handles separate error logging for different failure types."""

    def __init__(self):
        self.embedding_errors = []
        self.chunking_errors = []
        self.authentication_errors = []
        self.processing_errors = []
        self.validation_errors = []
        self.total_embeddings = 0
        self.total_chunks = 0
        self.using_fallback_tokenizer = False

    def log_embedding_error(self, transcript_id: str, paragraph_id: str, error: str):
        """Log embedding generation errors."""
        self.embedding_errors.append({
            "timestamp": datetime.now().isoformat(),
            "transcript_id": transcript_id,
            "paragraph_id": paragraph_id,
            "error": error,
            "action_required": "Review embedding generation logic"
        })

    def log_chunking_error(self, transcript_id: str, paragraph_id: str, error: str):
        """Log text chunking errors."""
        self.chunking_errors.append({
            "timestamp": datetime.now().isoformat(),
            "transcript_id": transcript_id,
            "paragraph_id": paragraph_id,
            "error": error,
            "action_required": "Review chunking logic"
        })

    def log_authentication_error(self, error: str):
        """Log OAuth/SSL authentication errors."""
        self.authentication_errors.append({
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "action_required": "Check LLM credentials and SSL certificate"
        })

    def log_processing_error(self, transcript_id: str, error: str):
        """Log general processing errors."""
        self.processing_errors.append({
            "timestamp": datetime.now().isoformat(),
            "transcript_id": transcript_id,
            "error": error,
            "action_required": "Review transcript data and processing logic"
        })

    def log_validation_error(self, transcript_id: str, validation_issue: str):
        """Log validation errors."""
        self.validation_errors.append({
            "timestamp": datetime.now().isoformat(),
            "transcript_id": transcript_id,
            "validation_issue": validation_issue,
            "action_required": "Review embedding validation logic"
        })

    def save_error_logs(self, nas_conn: SMBConnection):
        """Save error logs to separate JSON files on NAS."""
        global logger
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logs_path = config["stage_08_embeddings_generation"]["output_logs_path"]
        
        error_types = {
            "embedding": self.embedding_errors,
            "chunking": self.chunking_errors,
            "authentication": self.authentication_errors,
            "processing": self.processing_errors,
            "validation": self.validation_errors
        }
        
        for error_type, errors in error_types.items():
            if errors:
                error_file_path = nas_path_join(logs_path, f"stage_08_{error_type}_errors_{timestamp}.json")
                
                try:
                    error_data = json.dumps(errors, indent=2, default=str)
                    file_obj = io.BytesIO(error_data.encode('utf-8'))
                    
                    if nas_upload_file(nas_conn, file_obj, error_file_path):
                        log_console(f"Saved {len(errors)} {error_type} errors")
                    else:
                        log_console(f"Failed to save {error_type} errors", "WARNING")
                except Exception as e:
                    log_console(f"Error saving {error_type} errors: {e}", "ERROR")


def save_logs_to_nas(nas_conn: SMBConnection, stage_summary: Dict[str, Any], enhanced_error_logger: EnhancedErrorLogger):
    """Save execution logs, error logs, and summary to NAS."""
    global execution_log, error_log
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logs_path = config["stage_08_embeddings_generation"]["output_logs_path"]
    
    # Save execution log
    if execution_log:
        exec_log_path = nas_path_join(logs_path, f"stage_08_execution_{timestamp}.json")
        exec_log_data = json.dumps(execution_log, indent=2, default=str)
        exec_file_obj = io.BytesIO(exec_log_data.encode('utf-8'))
        
        if nas_upload_file(nas_conn, exec_file_obj, exec_log_path):
            log_console(f"Execution log saved: {len(execution_log)} entries")
        else:
            log_console("Failed to save execution log", "WARNING")
    
    # Save error log (if errors occurred)
    if error_log:
        error_log_path = nas_path_join(logs_path, f"stage_08_errors_{timestamp}.json")
        error_log_data = json.dumps(error_log, indent=2, default=str)
        error_file_obj = io.BytesIO(error_log_data.encode('utf-8'))
        
        if nas_upload_file(nas_conn, error_file_obj, error_log_path):
            log_console(f"Error log saved: {len(error_log)} entries")
        else:
            log_console("Failed to save error log", "WARNING")
    
    # Save stage summary
    summary_path = nas_path_join(logs_path, f"stage_08_summary_{timestamp}.json")
    summary_data = json.dumps(stage_summary, indent=2, default=str)
    summary_file_obj = io.BytesIO(summary_data.encode('utf-8'))
    
    if nas_upload_file(nas_conn, summary_file_obj, summary_path):
        log_console("Stage summary saved")
    else:
        log_console("Failed to save stage summary", "WARNING")
    
    # Save enhanced error logs
    enhanced_error_logger.save_error_logs(nas_conn)


def validate_environment_variables() -> None:
    """Validate all required environment variables are present."""
    
    required_env_vars = [
        "API_USERNAME",
        "API_PASSWORD", 
        "PROXY_USER",
        "PROXY_PASSWORD",
        "PROXY_URL",
        "NAS_USERNAME",
        "NAS_PASSWORD",
        "NAS_SERVER_IP",
        "NAS_SERVER_NAME",
        "NAS_SHARE_NAME",
        "NAS_BASE_PATH",
        "NAS_PORT",
        "CONFIG_PATH",
        "CLIENT_MACHINE_NAME",
        "LLM_CLIENT_ID",
        "LLM_CLIENT_SECRET",
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        log_error(
            error_msg,
            "environment_validation",
            {
                "missing_variables": missing_vars,
                "total_required": len(required_env_vars),
                "total_missing": len(missing_vars),
            }
        )
        log_console(f"ERROR: {error_msg}", "ERROR")
        raise ValueError(error_msg)
    else:
        log_execution(
            "Environment variables validated successfully",
            {"total_validated": len(required_env_vars)}
        )


def nas_path_join(*parts) -> str:
    """Join NAS path parts using forward slashes."""
    return "/".join(parts)


def validate_nas_path(path: str) -> bool:
    """Validate NAS path format."""
    if not path or path.startswith("/") or path.startswith("\\"):
        return False
    
    invalid_chars = ["<", ">", ":", '"', "|", "?", "*"]
    for char in invalid_chars:
        if char in path:
            return False
    
    return True




def get_nas_connection() -> Optional[SMBConnection]:
    """Create and return an SMB connection to the NAS."""
    try:
        conn = SMBConnection(
            username=os.getenv("NAS_USERNAME"),
            password=os.getenv("NAS_PASSWORD"),
            my_name=os.getenv("CLIENT_MACHINE_NAME"),
            remote_name=os.getenv("NAS_SERVER_NAME"),
            use_ntlm_v2=True,
            is_direct_tcp=True,
        )
        
        nas_port = int(os.getenv("NAS_PORT", 445))
        if conn.connect(os.getenv("NAS_SERVER_IP"), nas_port):
            log_execution("NAS connection established successfully", {
                "connection_type": "SMB/CIFS",
                "server": os.getenv("NAS_SERVER_NAME"),
                "port": nas_port,
            })
            return conn
        else:
            log_error("Failed to connect to NAS", "nas_connection", {
                "server": os.getenv("NAS_SERVER_NAME"),
                "port": nas_port
            })
            return None
    
    except Exception as e:
        log_error(f"NAS connection error: {str(e)}", "nas_connection", {"error": str(e)})
        return None


def nas_file_exists(conn: SMBConnection, file_path: str) -> bool:
    """Check if a file exists on NAS."""
    try:
        conn.getAttributes(os.getenv("NAS_SHARE_NAME"), file_path)
        return True
    except:
        return False


def nas_download_file(conn: SMBConnection, nas_file_path: str) -> Optional[bytes]:
    """Download a file from NAS and return as bytes."""
    if not validate_nas_path(nas_file_path):
        log_error(f"Invalid NAS path: {nas_file_path}", "path_validation")
        return None

    try:
        file_obj = io.BytesIO()
        conn.retrieveFile(os.getenv("NAS_SHARE_NAME"), nas_file_path, file_obj)
        file_obj.seek(0)
        return file_obj.read()
    except Exception as e:
        log_error(f"Failed to download file from NAS: {nas_file_path}", "nas_download", {"error": str(e)})
        return None


def nas_create_directory_recursive(conn: SMBConnection, dir_path: str) -> bool:
    """Create directory recursively on NAS."""
    try:
        path_parts = dir_path.split('/')
        current_path = ""
        
        for part in path_parts:
            if part:
                current_path = nas_path_join(current_path, part) if current_path else part
                try:
                    conn.createDirectory(os.getenv("NAS_SHARE_NAME"), current_path)
                except:
                    pass
        
        return True
    except Exception as e:
        log_error(f"Failed to create directory: {dir_path}", "nas_mkdir", {"error": str(e)})
        return False


def nas_upload_file(conn: SMBConnection, local_file_obj: io.BytesIO, nas_file_path: str) -> bool:
    """Upload a file object to NAS."""
    if not validate_nas_path(nas_file_path):
        log_error(f"Invalid NAS path: {nas_file_path}", "path_validation")
        return False

    try:
        dir_path = "/".join(nas_file_path.split("/")[:-1])
        if dir_path and not nas_file_exists(conn, nas_file_path):
            nas_create_directory_recursive(conn, dir_path)

        local_file_obj.seek(0)
        conn.storeFile(os.getenv("NAS_SHARE_NAME"), nas_file_path, local_file_obj)
        return True
    except Exception as e:
        log_error(f"Failed to upload file to NAS: {nas_file_path}", "nas_upload", {"error": str(e)})
        return False


def list_nas_directory(conn: SMBConnection, dir_path: str) -> List[str]:
    """List all files in NAS directory."""
    try:
        shared_files = conn.listPath(os.getenv("NAS_SHARE_NAME"), dir_path)
        return [f.filename for f in shared_files if not f.filename.startswith('.')]
    except Exception as e:
        log_error(f"Failed to list directory {dir_path}", "nas_list", {"error": str(e)})
        return []


def nas_rename_file(conn: SMBConnection, old_path: str, new_path: str) -> bool:
    """
    Rename file on NAS (atomic operation where supported).
    SMB doesn't support true atomic rename, so we do: copy → delete.
    """
    try:
        # Download file
        content = nas_download_file(conn, old_path)
        if not content:
            return False

        # Upload to new location
        file_obj = io.BytesIO(content)
        if not nas_upload_file(conn, file_obj, new_path):
            return False

        # Delete old file
        nas_delete_file(conn, old_path)

        return True
    except Exception as e:
        log_error(f"Failed to rename file {old_path} → {new_path}", "nas_rename", {"error": str(e)})
        return False


def nas_delete_file(conn: SMBConnection, file_path: str) -> bool:
    """Delete file from NAS."""
    try:
        conn.deleteFiles(os.getenv("NAS_SHARE_NAME"), file_path)
        return True
    except Exception as e:
        log_error(f"Failed to delete file {file_path}", "nas_delete", {"error": str(e)})
        return False


def load_manifest_with_recovery() -> Dict:
    """Load manifest with crash recovery support."""
    nas_conn = get_nas_connection()
    if not nas_conn:
        return initialize_empty_manifest()

    try:
        manifest_path = nas_path_join(
            config["stage_08_embeddings_generation"]["output_data_path"],
            "manifest.json"
        )
        temp_path = nas_path_join(
            config["stage_08_embeddings_generation"]["output_data_path"],
            "manifest.json.tmp"
        )

        # Try to load main manifest
        manifest_content = nas_download_file(nas_conn, manifest_path)
        if manifest_content:
            try:
                manifest = json.loads(manifest_content.decode('utf-8'))
                nas_conn.close()
                return manifest
            except json.JSONDecodeError:
                log_console("Main manifest corrupted, trying temp...", "WARNING")

        # Try temp manifest
        temp_content = nas_download_file(nas_conn, temp_path)
        if temp_content:
            try:
                manifest = json.loads(temp_content.decode('utf-8'))
                log_console("Recovered from temp manifest", "WARNING")
                nas_conn.close()
                return manifest
            except json.JSONDecodeError:
                log_console("Temp manifest also corrupted", "WARNING")

        # Initialize new manifest
        log_console("No valid manifest found, initializing new one")
        nas_conn.close()
        return initialize_empty_manifest()

    except Exception as e:
        log_error(f"Failed to load manifest: {e}", "manifest_load")
        if nas_conn:
            nas_conn.close()
        return initialize_empty_manifest()


def initialize_empty_manifest() -> Dict:
    """Initialize a new empty manifest."""
    return {
        "version": "8.0.0",
        "created_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "total_transcripts": 0,
        "completed_transcripts": 0,
        "failed_transcripts": 0,
        "consolidation_status": "not_started",
        "transcripts": {}
    }


def update_manifest_atomic(transcript_id: str, updates: Dict) -> bool:
    """Atomically update manifest using temp file pattern."""
    nas_conn = get_nas_connection()
    if not nas_conn:
        log_error("Failed to connect to NAS for manifest update", "manifest_update")
        return False

    try:
        manifest_path = nas_path_join(
            config["stage_08_embeddings_generation"]["output_data_path"],
            "manifest.json"
        )
        temp_path = nas_path_join(
            config["stage_08_embeddings_generation"]["output_data_path"],
            "manifest.json.tmp"
        )

        # Load current manifest
        manifest_content = nas_download_file(nas_conn, manifest_path)
        if manifest_content:
            try:
                manifest = json.loads(manifest_content.decode('utf-8'))
            except json.JSONDecodeError:
                manifest = initialize_empty_manifest()
        else:
            manifest = initialize_empty_manifest()

        # Apply updates
        if transcript_id not in manifest["transcripts"]:
            manifest["transcripts"][transcript_id] = {}

        manifest["transcripts"][transcript_id].update(updates)
        manifest["last_updated"] = datetime.now().isoformat()

        # Recalculate summary statistics
        manifest["total_transcripts"] = len(manifest["transcripts"])
        manifest["completed_transcripts"] = sum(
            1 for t in manifest["transcripts"].values() if t.get("status") == "completed"
        )
        manifest["failed_transcripts"] = sum(
            1 for t in manifest["transcripts"].values() if t.get("status") == "failed"
        )

        # Write to temp file
        temp_obj = io.BytesIO(json.dumps(manifest, indent=2).encode('utf-8'))
        if not nas_upload_file(nas_conn, temp_obj, temp_path):
            nas_conn.close()
            return False

        # Atomic rename (SMB limitation: delete old → rename new)
        if nas_file_exists(nas_conn, manifest_path):
            nas_delete_file(nas_conn, manifest_path)
        nas_rename_file(nas_conn, temp_path, manifest_path)

        nas_conn.close()
        return True

    except Exception as e:
        log_error(f"Failed to update manifest: {e}", "manifest_update", {"error": str(e)})
        if nas_conn:
            nas_conn.close()
        return False


def create_csv_from_records(records: List[Dict]) -> str:
    """Create CSV content from records (matches Stage 8 format)."""
    output = io.StringIO()

    fieldnames = [
        'file_path', 'filename', 'date_last_modified', 'title', 'transcript_type',
        'event_id', 'version_id', 'fiscal_year', 'fiscal_quarter',
        'institution_type', 'institution_id', 'ticker', 'company_name',
        'section_name', 'speaker_block_id', 'qa_group_id',
        'classification_ids', 'classification_names', 'block_summary',
        'chunk_id', 'chunk_tokens', 'chunk_content', 'chunk_paragraph_ids',
        'chunk_embedding'
    ]

    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for record in records:
        # Map fields and convert lists to JSON strings
        row = {}
        row['file_path'] = record.get('file_path', '')
        row['filename'] = record.get('filename', '')
        row['date_last_modified'] = record.get('date_last_modified', '')
        row['title'] = record.get('title', '')
        row['transcript_type'] = record.get('transcript_type', '')
        row['event_id'] = record.get('event_id', '')
        row['version_id'] = record.get('version_id', '')
        row['fiscal_year'] = record.get('fiscal_year', '')
        row['fiscal_quarter'] = record.get('fiscal_quarter', '')
        row['institution_type'] = record.get('institution_type', '')
        row['institution_id'] = record.get('institution_id', '')
        row['ticker'] = record.get('ticker', '')
        row['company_name'] = record.get('company_name', '')
        row['section_name'] = record.get('section_name', '')
        row['speaker_block_id'] = record.get('speaker_block_id', '')
        row['qa_group_id'] = record.get('qa_group_id', '')
        row['classification_ids'] = record.get('classification_ids', [])
        row['classification_names'] = record.get('classification_names', [])
        row['block_summary'] = record.get('block_summary', '')
        row['chunk_id'] = record.get('chunk_id', 0)
        row['chunk_tokens'] = record.get('chunk_tokens', 0)
        row['chunk_content'] = record.get('chunk_text', '')
        row['chunk_paragraph_ids'] = record.get('chunk_paragraph_ids', record.get('block_paragraph_ids', []))
        row['chunk_embedding'] = record.get('embedding', [])

        # Convert lists to JSON strings for CSV storage
        if isinstance(row['chunk_embedding'], list):
            row['chunk_embedding'] = json.dumps(row['chunk_embedding'])
        if isinstance(row['chunk_paragraph_ids'], list):
            row['chunk_paragraph_ids'] = json.dumps(row['chunk_paragraph_ids'])
        if isinstance(row['classification_ids'], list):
            row['classification_ids'] = json.dumps(row['classification_ids'])
        if isinstance(row['classification_names'], list):
            row['classification_names'] = json.dumps(row['classification_names'])

        writer.writerow(row)

    return output.getvalue()


def create_metadata_file(nas_conn: SMBConnection, transcript_id: str, csv_path: str, record_count: int) -> Dict:
    """Create validation metadata after successful write."""
    import hashlib

    try:
        # Download CSV to calculate checksum
        csv_content = nas_download_file(nas_conn, csv_path)
        if not csv_content:
            raise ValueError("Failed to download CSV for metadata creation")

        metadata = {
            "transcript_id": transcript_id,
            "created_at": datetime.now().isoformat(),
            "file_size_bytes": len(csv_content),
            "line_count": csv_content.decode('utf-8').count('\n'),
            "sha256_checksum": hashlib.sha256(csv_content).hexdigest(),
            "expected_records": record_count,
            "stage_version": "8.0.0",
            "processing_host": os.getenv("CLIENT_MACHINE_NAME")
        }

        # Write metadata atomically
        meta_path = f"{csv_path}.meta"
        meta_content = json.dumps(metadata, indent=2)
        meta_obj = io.BytesIO(meta_content.encode('utf-8'))

        if not nas_upload_file(nas_conn, meta_obj, meta_path):
            raise ValueError("Failed to upload metadata file")

        return metadata

    except Exception as e:
        log_error(f"Failed to create metadata for {transcript_id}: {e}", "metadata_create")
        return None


def validate_csv_integrity(nas_conn: SMBConnection, csv_path: str, meta_path: str) -> bool:
    """Validate CSV file against metadata."""
    import hashlib

    try:
        # Load metadata
        meta_content = nas_download_file(nas_conn, meta_path)
        if not meta_content:
            log_console(f"Missing metadata for {csv_path}", "ERROR")
            return False

        metadata = json.loads(meta_content.decode('utf-8'))

        # Download CSV
        csv_content = nas_download_file(nas_conn, csv_path)
        if not csv_content:
            log_console(f"Missing CSV file {csv_path}", "ERROR")
            return False

        # Validation checks
        checks = {
            "file_size": len(csv_content) == metadata["file_size_bytes"],
            "line_count": csv_content.decode('utf-8').count('\n') == metadata["line_count"],
            "checksum": hashlib.sha256(csv_content).hexdigest() == metadata["sha256_checksum"],
        }

        # Parse CSV and count records
        csv_reader = csv.DictReader(io.StringIO(csv_content.decode('utf-8')))
        actual_records = sum(1 for _ in csv_reader)
        checks["record_count"] = actual_records == metadata["expected_records"]

        # Check CSV structure
        csv_reader = csv.DictReader(io.StringIO(csv_content.decode('utf-8')))
        first_row = next(csv_reader, None)
        if first_row:
            required_fields = ['chunk_embedding', 'chunk_content', 'chunk_tokens']
            checks["required_fields"] = all(f in first_row for f in required_fields)

            # Validate embedding is valid JSON array
            try:
                embedding = json.loads(first_row['chunk_embedding'])
                checks["embedding_valid"] = isinstance(embedding, list) and len(embedding) == 3072
            except:
                checks["embedding_valid"] = False
        else:
            checks["required_fields"] = False
            checks["embedding_valid"] = False

        # All checks must pass
        if all(checks.values()):
            return True
        else:
            failed_checks = [k for k, v in checks.items() if not v]
            log_console(f"Validation failed for {csv_path}: {failed_checks}", "ERROR")
            return False

    except Exception as e:
        log_console(f"Validation error for {csv_path}: {e}", "ERROR")
        return False


def cleanup_tmp_file(nas_conn: SMBConnection, transcript_id: str):
    """Remove temporary file after crash or error."""
    output_path = config["stage_08_embeddings_generation"]["output_data_path"]
    tmp_path = nas_path_join(output_path, "individual", f"{transcript_id}.csv.tmp")

    if nas_file_exists(nas_conn, tmp_path):
        nas_delete_file(nas_conn, tmp_path)
        log_console(f"Cleaned up temp file: {tmp_path}")


def quarantine_corrupt_file(nas_conn: SMBConnection, transcript_id: str):
    """Move corrupted file to failed/ directory for investigation."""
    output_path = config["stage_08_embeddings_generation"]["output_data_path"]
    csv_path = nas_path_join(output_path, "individual", f"{transcript_id}.csv")
    meta_path = f"{csv_path}.meta"
    tmp_path = f"{csv_path}.tmp"

    failed_dir = nas_path_join(output_path, "failed")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Move CSV if exists
    if nas_file_exists(nas_conn, csv_path):
        failed_csv = nas_path_join(failed_dir, f"{transcript_id}_{timestamp}.csv.corrupt")
        nas_rename_file(nas_conn, csv_path, failed_csv)

    # Move metadata if exists
    if nas_file_exists(nas_conn, meta_path):
        failed_meta = nas_path_join(failed_dir, f"{transcript_id}_{timestamp}.meta.corrupt")
        nas_rename_file(nas_conn, meta_path, failed_meta)

    # Move temp if exists
    if nas_file_exists(nas_conn, tmp_path):
        failed_tmp = nas_path_join(failed_dir, f"{transcript_id}_{timestamp}.tmp.corrupt")
        nas_rename_file(nas_conn, tmp_path, failed_tmp)

    log_console(f"Quarantined corrupted files for {transcript_id}")


def cleanup_individual_files(nas_conn: SMBConnection, file_list: List[str]):
    """Delete individual files after successful consolidation."""
    output_path = config["stage_08_embeddings_generation"]["output_data_path"]

    for csv_filename in file_list:
        csv_path = nas_path_join(output_path, "individual", csv_filename)
        meta_path = f"{csv_path}.meta"

        nas_delete_file(nas_conn, csv_path)
        nas_delete_file(nas_conn, meta_path)

    log_console(f"Cleaned up {len(file_list)} individual files")


def process_and_save_transcript_resilient(
    transcript_id: str,
    transcript_records: List[Dict],
    enhanced_error_logger: EnhancedErrorLogger,
    max_retries: int = 3
) -> bool:
    """
    Crash-resilient transcript processing with comprehensive error handling.
    Processes transcript and saves to individual CSV file with validation.
    """
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        nas_conn = None
        try:
            # Mark as in-progress in manifest
            update_manifest_atomic(transcript_id, {
                "status": "in_progress",
                "started_at": datetime.now().isoformat(),
                "attempt": attempt + 1
            })

            # Process transcript (embeddings generation)
            log_console(f"  Processing embeddings for {transcript_id} (attempt {attempt + 1}/{max_retries})")
            enhanced_records = process_transcript(transcript_records, transcript_id, enhanced_error_logger)

            if not enhanced_records:
                raise ValueError(f"No records generated for {transcript_id}")

            # Create CSV content
            csv_content = create_csv_from_records(enhanced_records)

            # Get NAS connection
            nas_conn = get_nas_connection()
            if not nas_conn:
                raise RuntimeError("Failed to connect to NAS")

            # Define paths
            output_path = config["stage_08_embeddings_generation"]["output_data_path"]
            individual_path = nas_path_join(output_path, "individual", f"{transcript_id}.csv")
            temp_path = f"{individual_path}.tmp"

            # Write CSV to temp file
            log_console(f"  Writing to temp file...")
            temp_obj = io.BytesIO(csv_content.encode('utf-8'))
            if not nas_upload_file(nas_conn, temp_obj, temp_path):
                raise IOError(f"Failed to upload temp file for {transcript_id}")

            # Atomic rename: .tmp → .csv
            log_console(f"  Finalizing file...")
            if not nas_rename_file(nas_conn, temp_path, individual_path):
                raise IOError(f"Failed to rename temp file for {transcript_id}")

            # Create metadata file
            log_console(f"  Creating metadata...")
            metadata = create_metadata_file(nas_conn, transcript_id, individual_path, len(enhanced_records))
            if not metadata:
                raise ValueError(f"Failed to create metadata for {transcript_id}")

            # Final validation
            log_console(f"  Validating file integrity...")
            if not validate_csv_integrity(nas_conn, individual_path, f"{individual_path}.meta"):
                raise ValueError(f"Final validation failed for {transcript_id}")

            # Update manifest as completed
            update_manifest_atomic(transcript_id, {
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "record_count": len(enhanced_records),
                "file_size_bytes": len(csv_content),
                "individual_file": individual_path,
                "metadata_file": f"{individual_path}.meta"
            })

            log_console(f"✓ Successfully saved {transcript_id} ({len(enhanced_records)} records)")

            if nas_conn:
                nas_conn.close()

            return True

        except Exception as e:
            log_console(f"Error processing {transcript_id} (attempt {attempt + 1}/{max_retries}): {e}", "WARNING")

            if nas_conn:
                # Cleanup temp file on error
                try:
                    cleanup_tmp_file(nas_conn, transcript_id)
                except:
                    pass
                nas_conn.close()

            if attempt < max_retries - 1:
                # Retry with exponential backoff
                time.sleep(retry_delay * (2 ** attempt))
                continue
            else:
                # Max retries exceeded - mark as failed
                log_console(f"✗ Failed to process {transcript_id} after {max_retries} attempts", "ERROR")

                # Quarantine any corrupted files
                nas_conn_cleanup = get_nas_connection()
                if nas_conn_cleanup:
                    try:
                        quarantine_corrupt_file(nas_conn_cleanup, transcript_id)
                    except:
                        pass
                    nas_conn_cleanup.close()

                # Update manifest as failed
                update_manifest_atomic(transcript_id, {
                    "status": "failed",
                    "error": str(e),
                    "failed_at": datetime.now().isoformat(),
                    "retry_count": max_retries
                })

                return False

    return False


def analyze_resume_state(all_transcripts: Dict[str, List[Dict]]) -> Tuple[List[str], List[str], List[str]]:
    """
    Analyze which transcripts need processing based on manifest.
    Returns (to_process, completed, failed) transcript IDs.
    """
    manifest = load_manifest_with_recovery()

    to_process = []
    completed = []
    failed_list = []

    for transcript_id in all_transcripts.keys():
        transcript_info = manifest.get("transcripts", {}).get(transcript_id, {})
        status = transcript_info.get("status", "not_started")

        if status == "completed":
            # Verify the file actually exists and is valid
            nas_conn = get_nas_connection()
            if nas_conn:
                output_path = config["stage_08_embeddings_generation"]["output_data_path"]
                csv_path = nas_path_join(output_path, "individual", f"{transcript_id}.csv")
                meta_path = f"{csv_path}.meta"

                if nas_file_exists(nas_conn, csv_path) and validate_csv_integrity(nas_conn, csv_path, meta_path):
                    completed.append(transcript_id)
                    nas_conn.close()
                else:
                    # File missing or invalid, need to reprocess
                    log_console(f"  {transcript_id}: marked complete but file invalid - will reprocess", "WARNING")
                    to_process.append(transcript_id)
                    nas_conn.close()
            else:
                # Can't verify, assume complete
                completed.append(transcript_id)

        elif status == "failed":
            # Add to failed list, can be reprocessed if desired
            failed_list.append(transcript_id)
            # Optionally reprocess failed transcripts
            retry_failed = config["stage_08_embeddings_generation"].get("retry_failed_on_resume", False)
            if retry_failed:
                log_console(f"  {transcript_id}: previously failed - will retry", "WARNING")
                to_process.append(transcript_id)

        elif status == "in_progress":
            # Crashed during processing, need to reprocess
            log_console(f"  {transcript_id}: was in progress - will reprocess", "WARNING")
            to_process.append(transcript_id)

        else:
            # Not started or unknown status
            to_process.append(transcript_id)

    return to_process, completed, failed_list


def consolidate_individual_files() -> Optional[str]:
    """
    Consolidate all individual transcript CSVs into master file.
    Handles crashes during consolidation with atomic temp file pattern.
    """
    nas_conn = get_nas_connection()
    if not nas_conn:
        log_console("Failed to connect to NAS for consolidation", "ERROR")
        return None

    try:
        output_path = config["stage_08_embeddings_generation"]["output_data_path"]
        individual_dir = nas_path_join(output_path, "individual")
        consolidated_dir = nas_path_join(output_path, "consolidated")

        # List all individual CSV files
        log_console("Scanning for individual transcript files...")
        all_files = list_nas_directory(nas_conn, individual_dir)
        csv_files = [f for f in all_files if f.endswith('.csv') and not f.endswith('.tmp')]

        log_console(f"Found {len(csv_files)} individual CSV files to consolidate")

        if len(csv_files) == 0:
            log_console("No files to consolidate", "WARNING")
            return None

        # Validate all files before consolidation
        valid_files = []
        invalid_files = []

        log_console("Validating individual files...")
        for csv_file in csv_files:
            csv_path = nas_path_join(individual_dir, csv_file)
            meta_path = f"{csv_path}.meta"

            if validate_csv_integrity(nas_conn, csv_path, meta_path):
                valid_files.append(csv_file)
            else:
                invalid_files.append(csv_file)
                # Quarantine invalid file
                transcript_id = csv_file.replace('.csv', '')
                quarantine_corrupt_file(nas_conn, transcript_id)

        if invalid_files:
            log_console(f"WARNING: {len(invalid_files)} files failed validation - quarantined", "WARNING")

        log_console(f"Consolidating {len(valid_files)} validated files...")

        # Create consolidated file with atomic temp pattern
        temp_consolidated_path = nas_path_join(consolidated_dir, "stage_08_embeddings.csv.tmp")
        final_consolidated_path = nas_path_join(consolidated_dir, "stage_08_embeddings.csv")

        # Build consolidated CSV in memory
        output = io.StringIO()
        writer = None
        total_records = 0

        for i, csv_file in enumerate(valid_files):
            csv_path = nas_path_join(individual_dir, csv_file)
            csv_content = nas_download_file(nas_conn, csv_path)

            if not csv_content:
                log_console(f"ERROR: Could not download {csv_file}", "ERROR")
                continue

            reader = csv.DictReader(io.StringIO(csv_content.decode('utf-8')))

            # Initialize writer with first file's fieldnames
            if writer is None:
                fieldnames = reader.fieldnames
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()

            # Write all rows from this file
            file_records = 0
            for row in reader:
                writer.writerow(row)
                file_records += 1
                total_records += 1

            if (i + 1) % 10 == 0 or (i + 1) == len(valid_files):
                log_console(f"  Progress: {i + 1}/{len(valid_files)} files, {total_records} records")

        # Upload consolidated file atomically
        log_console("Uploading consolidated file...")
        consolidated_content = output.getvalue()
        temp_file_obj = io.BytesIO(consolidated_content.encode('utf-8'))

        # Write to .tmp first
        if not nas_upload_file(nas_conn, temp_file_obj, temp_consolidated_path):
            raise RuntimeError("Failed to write consolidated temp file")

        # Atomic rename: .tmp → final
        log_console("Finalizing consolidated file...")
        if nas_file_exists(nas_conn, final_consolidated_path):
            nas_delete_file(nas_conn, final_consolidated_path)
        if not nas_rename_file(nas_conn, temp_consolidated_path, final_consolidated_path):
            raise RuntimeError("Failed to finalize consolidated file")

        log_console(f"✓ Consolidation complete: {total_records} total records in {len(valid_files)} transcripts")

        # Optional: Cleanup individual files
        cleanup_enabled = config["stage_08_embeddings_generation"].get("cleanup_after_consolidation", True)
        if cleanup_enabled:
            log_console("Cleaning up individual files...")
            cleanup_individual_files(nas_conn, valid_files)
            log_console("✓ Cleanup complete")
        else:
            log_console("Individual files preserved (cleanup disabled in config)")

        return final_consolidated_path

    except Exception as e:
        log_console(f"Consolidation failed: {e}", "ERROR")
        log_error(f"Consolidation error: {e}", "consolidation_error")
        return None
    finally:
        # Always close NAS connection
        if nas_conn:
            try:
                nas_conn.close()
            except Exception:
                pass


def load_stage_config(nas_conn: SMBConnection) -> Dict:
    """Load and validate Stage 8 configuration from NAS."""
    try:
        log_execution("Loading shared configuration from NAS...")
        config_data = nas_download_file(nas_conn, os.getenv("CONFIG_PATH"))
        
        if not config_data:
            raise ValueError("Failed to download config file from NAS")
        
        full_config = yaml.safe_load(config_data.decode('utf-8'))
        log_execution("Successfully loaded shared configuration from NAS")
        
        if "stage_08_embeddings_generation" not in full_config:
            raise ValueError("Missing stage_08_embeddings_generation in config")
        
        config = full_config
        
        log_execution("Stage 8 configuration loaded successfully", {
            "config_path": os.getenv("CONFIG_PATH")
        })
        
        return config
    
    except Exception as e:
        log_error(f"Failed to load configuration: {str(e)}", "config_load")
        raise


def setup_ssl_certificate(nas_conn: SMBConnection) -> str:
    """Download SSL certificate from NAS and return temp file path."""
    try:
        ssl_cert_path = config.get("ssl_cert_path")
        if not ssl_cert_path:
            log_console("No SSL certificate path configured")
            return None
        
        cert_data = nas_download_file(nas_conn, ssl_cert_path)
        if not cert_data:
            log_console("Failed to download SSL certificate", "WARNING")
            return None
        
        with tempfile.NamedTemporaryFile(suffix=".cer", delete=False) as temp_cert:
            temp_cert.write(cert_data)
            temp_cert_path = temp_cert.name
        
        # Set SSL environment variables for requests and OpenAI client
        os.environ["REQUESTS_CA_BUNDLE"] = temp_cert_path
        os.environ["SSL_CERT_FILE"] = temp_cert_path
        
        log_execution("SSL certificate setup complete", {"temp_cert_path": temp_cert_path})
        return temp_cert_path
    
    except Exception as e:
        log_error(f"SSL certificate setup failed: {e}", "ssl_setup")
        return None


def get_oauth_token() -> Optional[str]:
    """Obtain OAuth token for LLM API access."""
    try:
        token_endpoint = config["stage_08_embeddings_generation"]["llm_config"]["token_endpoint"]
        
        auth_data = {
            'grant_type': 'client_credentials',
            'client_id': os.getenv("LLM_CLIENT_ID"),
            'client_secret': os.getenv("LLM_CLIENT_SECRET")
        }
        
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        
        # Set up SSL context if certificate available
        verify_ssl = True
        if ssl_cert_path and os.path.exists(ssl_cert_path):
            verify_ssl = ssl_cert_path
        
        response = requests.post(
            token_endpoint,
            data=auth_data,
            headers=headers,
            verify=verify_ssl,
            timeout=30
        )
        
        if response.status_code == 200:
            token_data = response.json()
            return token_data.get('access_token')
        else:
            error_msg = f"OAuth token request failed: {response.status_code}"
            log_error(error_msg, "oauth_token", {"status_code": response.status_code})
            return None
            
    except Exception as e:
        error_msg = f"OAuth token acquisition failed: {e}"
        log_error(error_msg, "oauth_token", {"error": str(e)})
        return None


def setup_llm_client(token: str) -> Optional[OpenAI]:
    """Setup OpenAI client with custom base URL and OAuth token."""
    try:
        client = OpenAI(
            api_key=token,
            base_url=config["stage_08_embeddings_generation"]["llm_config"]["base_url"],
            timeout=config["stage_08_embeddings_generation"]["llm_config"]["timeout"]
        )
        
        log_execution("LLM client setup completed")
        return client
    except Exception as e:
        log_error(f"Failed to setup LLM client: {e}", "llm_setup")
        return None


def refresh_oauth_token_for_transcript(transcript_info: Dict[str, Any]) -> None:
    """Refresh OAuth token per transcript to prevent expiration."""
    global oauth_token, llm_client
    
    log_execution(f"Refreshing OAuth token for transcript", {
        "transcript_id": transcript_info.get("transcript_id", "unknown")
    })
    
    new_token = get_oauth_token()
    if new_token:
        oauth_token = new_token
        llm_client = setup_llm_client(oauth_token)
        log_execution("OAuth token refreshed successfully")
    else:
        raise RuntimeError("Failed to refresh OAuth token")


def save_results_incrementally(results: List[Dict], output_path: str, is_first_batch: bool = False):
    """Save results incrementally after each transcript (following Stage 7 array pattern)."""
    try:
        nas_conn = get_nas_connection()
        if not nas_conn:
            raise RuntimeError("Failed to connect to NAS for incremental save")
        
        # Append records using Stage 7's pattern
        if not append_records_to_json_array(nas_conn, results, output_path, is_first_batch):
            raise RuntimeError("Failed to append records to output file")
        
        log_console(f"Appended {len(results)} records to output file")
        log_execution(f"Incrementally saved {len(results)} records", {
            "output_path": output_path,
            "is_first_batch": is_first_batch
        })
        
        nas_conn.close()
        
    except Exception as e:
        error_msg = f"Failed to save results incrementally: {e}"
        log_error(error_msg, "incremental_save", {"error": str(e)})
        raise


def save_results_incrementally_csv(results: List[Dict], output_path: str, is_first_batch: bool = False):
    """Save results incrementally to CSV format for better handling of large datasets."""
    try:
        nas_conn = get_nas_connection()
        if not nas_conn:
            raise RuntimeError("Failed to connect to NAS for incremental save")
        
        # Append records using CSV format
        if not append_records_to_csv(nas_conn, results, output_path, is_first_batch):
            raise RuntimeError("Failed to append records to output file")
        
        log_console(f"Appended {len(results)} records to CSV output file")
        log_execution(f"Incrementally saved {len(results)} records to CSV", {
            "output_path": output_path,
            "is_first_batch": is_first_batch,
            "format": "CSV"
        })
        
        nas_conn.close()
        
    except Exception as e:
        error_msg = f"Failed to save results incrementally: {e}"
        log_error(error_msg, "incremental_save", {"error": str(e)})
        raise


def append_records_to_csv(nas_conn: SMBConnection, records: List[Dict], file_path: str, is_first: bool = False) -> bool:
    """Append records to a CSV file - much more efficient for large datasets with embeddings."""
    try:
        # Get all field names from records (excluding embedding for header)
        if not records:
            return True
        
        # Define specific field order as per requirements
        fieldnames = [
            'file_path',                    # Full path to file
            'filename',                     # Just the filename
            'date_last_modified',           # File modification date
            'title',                        # Transcript title
            'transcript_type',              # Type from filename (e.g., "Corrected", "Final")
            'event_id',                     # Event identifier
            'version_id',                   # Version identifier
            'fiscal_year',                  # Fiscal year
            'fiscal_quarter',               # Fiscal quarter
            'institution_type',             # Institution type
            'institution_id',               # Institution identifier
            'ticker',                       # Stock ticker
            'company_name',                 # Company name
            'section_name',                 # Section name (MD/Q&A)
            'speaker_block_id',             # Speaker block identifier
            'qa_group_id',                  # Q&A group identifier (for Q&A sections)
            'classification_ids',           # List of classification IDs from Stage 6
            'classification_names',         # List of classification names from Stage 6
            'block_summary',                # Summary from Stage 7
            'chunk_id',                     # Chunk identifier within block/group
            'chunk_tokens',                 # Number of tokens in chunk
            'chunk_content',                # Formatted text with headers that gets embedded
            'chunk_paragraph_ids',          # List of original paragraph IDs in this chunk
            'chunk_embedding'               # 3072-dimensional embedding vector
        ]
        
        # Add any additional fields not in our predefined list (maintain backward compatibility)
        # NOTE: Only use the defined fieldnames - don't add extra fields from records
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        
        # Write header only if first batch
        if is_first:
            writer.writeheader()
        
        # Write records with field mapping and JSON serialization
        for record in records:
            # Map old field names to new ones
            row = {}
            
            # Direct mappings
            row['file_path'] = record.get('file_path', '')
            row['filename'] = record.get('filename', '')
            row['date_last_modified'] = record.get('date_last_modified', '')
            row['title'] = record.get('title', '')
            row['transcript_type'] = record.get('transcript_type', '')  # Add transcript type
            row['event_id'] = record.get('event_id', '')
            row['version_id'] = record.get('version_id', '')
            row['fiscal_year'] = record.get('fiscal_year', '')
            row['fiscal_quarter'] = record.get('fiscal_quarter', '')  # Stage 3 field name
            row['institution_type'] = record.get('institution_type', '')
            row['institution_id'] = record.get('institution_id', '')
            row['ticker'] = record.get('ticker', '')
            row['company_name'] = record.get('company_name', '')
            row['section_name'] = record.get('section_name', '')
            row['speaker_block_id'] = record.get('speaker_block_id', '')  # Stage 3 field name
            row['qa_group_id'] = record.get('qa_group_id', '')  # Stage 5 field name
            row['classification_ids'] = record.get('classification_ids', [])
            row['classification_names'] = record.get('classification_names', [])
            row['block_summary'] = record.get('block_summary', '')
            row['chunk_id'] = record.get('chunk_id', 0)
            row['chunk_tokens'] = record.get('chunk_tokens', 0)
            row['chunk_content'] = record.get('chunk_text', '')  # chunk_text is set during chunking
            row['chunk_paragraph_ids'] = record.get('chunk_paragraph_ids', record.get('block_paragraph_ids', []))
            row['chunk_embedding'] = record.get('embedding', [])  # From embeddings generation
            
            # Convert lists to JSON strings for CSV storage
            if isinstance(row['chunk_embedding'], list):
                row['chunk_embedding'] = json.dumps(row['chunk_embedding'])
            if isinstance(row['chunk_paragraph_ids'], list):
                row['chunk_paragraph_ids'] = json.dumps(row['chunk_paragraph_ids'])
            if isinstance(row['classification_ids'], list):
                row['classification_ids'] = json.dumps(row['classification_ids'])
            if isinstance(row['classification_names'], list):
                row['classification_names'] = json.dumps(row['classification_names'])
            
            writer.writerow(row)
        
        # Get CSV content
        csv_content = output.getvalue()
        
        # If not first batch, download existing file and append
        if not is_first and nas_file_exists(nas_conn, file_path):
            # For CSV, we can truly append without reading the whole file
            # But since SMB doesn't support append mode, we still need to download and re-upload
            # However, this is still more efficient than JSON array manipulation
            existing_content = nas_download_file(nas_conn, file_path)
            if existing_content:
                # Combine existing and new content
                full_content = existing_content.decode('utf-8') + csv_content
                file_obj = io.BytesIO(full_content.encode('utf-8'))
            else:
                file_obj = io.BytesIO(csv_content.encode('utf-8'))
        else:
            file_obj = io.BytesIO(csv_content.encode('utf-8'))
        
        return nas_upload_file(nas_conn, file_obj, file_path)
        
    except Exception as e:
        log_error(f"Failed to append records to CSV: {e}", "csv_append")
        return False


def append_records_to_json_array(nas_conn: SMBConnection, records: List[Dict], file_path: str, is_first: bool = False) -> bool:
    """Append records to a JSON array file (Stage 7 pattern)."""
    try:
        # Prepare JSON content
        if not is_first:
            # Add comma before new records if not the first entry
            content = ","
        else:
            # Start the JSON array
            content = "["
        
        # Add records
        records_json = [json.dumps(record, indent=2, default=str) for record in records]
        content += ",\n".join(records_json)
        
        # Check if file exists and download if appending
        if not is_first and nas_file_exists(nas_conn, file_path):
            existing_content = nas_download_file(nas_conn, file_path)
            if existing_content:
                content = existing_content.decode('utf-8') + "," + ",\n".join(records_json)
        
        # Upload the updated content
        file_obj = io.BytesIO(content.encode('utf-8'))
        return nas_upload_file(nas_conn, file_obj, file_path)
        
    except Exception as e:
        log_error(f"Failed to append records: {e}", "json_append")
        return False


def close_json_array(nas_conn: SMBConnection, file_path: str) -> bool:
    """Close a JSON array file by appending the closing bracket (Stage 7 pattern)."""
    try:
        # Download existing content
        existing_content = nas_download_file(nas_conn, file_path)
        if existing_content is None:
            log_error("No existing file to close", "json_close")
            return False
        
        # Add closing bracket
        content = existing_content.decode('utf-8') + "\n]"
        
        # Upload the updated content
        file_obj = io.BytesIO(content.encode('utf-8'))
        return nas_upload_file(nas_conn, file_obj, file_path)
        
    except Exception as e:
        log_error(f"Failed to close JSON array: {e}", "json_close", {"error": str(e)})
        return False


def setup_tokenizer():
    """Setup tiktoken tokenizer for token counting with fallback support."""
    global tokenizer
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        log_execution("Tokenizer initialized with tiktoken")
        log_console("Tokenizer initialized successfully with tiktoken")
        return True
    except Exception as e:
        log_error(f"Failed to setup tiktoken tokenizer: {e}", "tokenizer_setup")
        log_console(f"WARNING: tiktoken initialization failed: {e}", "WARNING")
        log_console("Using fallback token estimation method", "WARNING")
        tokenizer = None  # Explicitly set to None to trigger fallback
        return False


def estimate_tokens_fallback(text: str) -> int:
    """
    Fallback token estimation when tiktoken is not available.
    
    Uses a hybrid approach combining character and word-based estimation.
    Based on empirical observations:
    - Average ~4 characters per token for English text
    - Average ~1.3 tokens per word
    - Adjusts for punctuation and special characters
    """
    if not text:
        return 0
    
    # Method 1: Character-based estimation (tends to underestimate)
    char_estimate = len(text) / 4.0
    
    # Method 2: Word-based estimation (tends to overestimate for technical text)
    words = text.split()
    word_estimate = len(words) * 1.3
    
    # Method 3: Refined character estimate accounting for whitespace
    # Remove extra whitespace for more accurate count
    compressed_text = ' '.join(text.split())
    refined_char_estimate = len(compressed_text) / 3.5
    
    # Take weighted average favoring refined character method
    # This tends to be most accurate for financial transcripts
    final_estimate = (
        refined_char_estimate * 0.5 +  # 50% weight
        char_estimate * 0.3 +           # 30% weight
        word_estimate * 0.2             # 20% weight
    )
    
    # Add 10% buffer for safety in chunking decisions
    # Better to overestimate and chunk more than underestimate and fail
    final_estimate = int(final_estimate * 1.1)
    
    return final_estimate


def count_tokens(text: str) -> int:
    """
    Count tokens in text using tiktoken with fallback estimation.
    
    Attempts to use tiktoken for accurate counting, but falls back to
    estimation if tiktoken is not available or fails.
    """
    global tokenizer
    
    # Try to initialize tokenizer if not already done
    if tokenizer is None and not hasattr(count_tokens, '_fallback_warned'):
        setup_tokenizer()
    
    # Try tiktoken if available
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception as e:
            # Tiktoken failed on this specific text
            log_error(f"tiktoken encoding failed for text: {e}", "token_encoding")
            log_console(f"WARNING: tiktoken encoding failed, using fallback", "WARNING")
            
            # Set flag to avoid repeated warnings
            count_tokens._fallback_warned = True
            
            # Fall through to fallback method
    
    # Use fallback estimation
    if not hasattr(count_tokens, '_fallback_warned'):
        log_console("Using fallback token estimation (tiktoken unavailable)", "WARNING")
        log_execution("Using fallback token estimation method", {
            "reason": "tiktoken unavailable or failed",
            "text_length": len(text)
        })
        count_tokens._fallback_warned = True
    
    return estimate_tokens_fallback(text)


def find_sentence_boundary(text: str, target_pos: int, window: int = 50) -> int:
    """Find the nearest sentence boundary to target position."""
    # Look for sentence endings within window
    start = max(0, target_pos - window)
    end = min(len(text), target_pos + window)
    
    # Find all sentence endings in window
    endings = []
    for match in re.finditer(r'[.!?]\s+', text[start:end]):
        endings.append(start + match.end())
    
    if endings:
        # Find closest to target
        return min(endings, key=lambda x: abs(x - target_pos))
    
    # Fallback to space if no sentence ending found
    space_pos = text.find(' ', target_pos)
    if space_pos != -1:
        return space_pos
    
    return target_pos


def chunk_text(text: str, max_tokens: int = 500, chunk_threshold: int = 1000, min_final_chunk: int = 300) -> List[Tuple[str, int]]:
    """
    Intelligently chunk text into smaller pieces.
    Returns list of (chunk_text, token_count) tuples.
    """
    total_tokens = count_tokens(text)
    
    # If already under threshold, return as single chunk
    if total_tokens <= chunk_threshold:
        return [(text, total_tokens)]
    
    chunks = []
    remaining_text = text
    
    while remaining_text:
        # Estimate character position for target chunk size
        chars_per_token = len(remaining_text) / count_tokens(remaining_text)
        target_chars = int(max_tokens * chars_per_token)
        
        if len(remaining_text) <= target_chars:
            # Last chunk
            chunk_tokens = count_tokens(remaining_text)
            
            # Check if final chunk is too small
            if chunks and chunk_tokens < min_final_chunk:
                # Merge with previous chunk
                prev_chunk, prev_tokens = chunks[-1]
                merged_chunk = prev_chunk + " " + remaining_text
                merged_tokens = count_tokens(merged_chunk)
                chunks[-1] = (merged_chunk, merged_tokens)
            else:
                chunks.append((remaining_text, chunk_tokens))
            break
        
        # Find good breaking point
        break_pos = find_sentence_boundary(remaining_text, target_chars)
        chunk = remaining_text[:break_pos].strip()
        
        # Validate chunk size
        chunk_tokens = count_tokens(chunk)
        
        # Adjust if chunk is too large
        while chunk_tokens > max_tokens * 1.2:  # Allow 20% overflow
            target_chars = int(target_chars * 0.9)
            break_pos = find_sentence_boundary(remaining_text, target_chars)
            chunk = remaining_text[:break_pos].strip()
            chunk_tokens = count_tokens(chunk)
        
        chunks.append((chunk, chunk_tokens))
        remaining_text = remaining_text[break_pos:].strip()
    
    return chunks


def generate_embeddings_batch(texts: List[str], retry_count: int = 3) -> Optional[List[List[float]]]:
    """Generate embeddings for multiple texts in a single API call."""
    global llm_client
    
    if not texts:
        return []
    
    embed_config = config["stage_08_embeddings_generation"].get("embedding_config", {})
    model = embed_config.get("model", "text-embedding-3-large")
    
    for attempt in range(retry_count):
        try:
            response = llm_client.embeddings.create(
                model=model,
                input=texts  # Pass array of texts for batch processing
            )
            # Extract embeddings in same order as input texts
            embeddings = [data.embedding for data in response.data]
            return embeddings
        except Exception as e:
            log_console(f"Batch embedding attempt {attempt + 1} failed: {e}", "WARNING")
            if attempt < retry_count - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return None
    
    return None


def generate_embedding(text: str, retry_count: int = 3) -> Optional[List[float]]:
    """Generate embedding for single text (kept for backward compatibility)."""
    result = generate_embeddings_batch([text], retry_count)
    return result[0] if result else None


def load_stage7_data(nas_conn: SMBConnection) -> List[Dict]:
    """Load Stage 7 summarized content data from NAS."""
    try:
        # Input path is already a full path from config
        nas_file_path = config["stage_08_embeddings_generation"]["input_data_path"]
        
        input_data = nas_download_file(nas_conn, nas_file_path)
        
        if not input_data:
            raise ValueError(f"Failed to download Stage 7 data from: {nas_file_path}")
        
        # Stage 7 outputs JSON array format
        stage7_records = json.loads(input_data.decode('utf-8'))
        
        log_execution(f"Loaded {len(stage7_records)} records from Stage 7")
        return stage7_records
    
    except Exception as e:
        log_error(f"Failed to load Stage 7 data: {str(e)}", "data_load")
        raise


# Removed aggregate_categories function - no longer needed
# Stage 6 provides block-level classifications that are already consistent within blocks


def format_md_content(speaker: str, paragraphs: List[str]) -> str:
    """Format Management Discussion content with speaker header."""
    formatted = f"[{speaker}]\n\n"
    formatted += "\n\n".join(paragraphs)
    return formatted


def format_qa_content(qa_group_id: str, speaker_blocks: List[Dict]) -> str:
    """Format Q&A content with Q&A group header and speaker subheaders."""
    formatted = f"[Q&A Group {qa_group_id}]\n\n"
    
    for block in speaker_blocks:
        speaker = block['speaker']
        paragraphs = block['paragraphs']
        formatted += f"[{speaker}]\n"
        formatted += "\n\n".join(paragraphs)
        formatted += "\n\n"
    
    return formatted.rstrip()


def process_transcript(transcript_records: List[Dict], transcript_id: str, enhanced_error_logger: EnhancedErrorLogger) -> List[Dict]:
    """Process all records for a single transcript to generate embeddings with hierarchical chunking."""
    
    enhanced_records = []
    
    # Debug: Check what fields are present in the first record
    if transcript_records:
        first_record = transcript_records[0]
        available_fields = list(first_record.keys())
        log_execution(f"Available fields in Stage 7 data: {available_fields}")  # All fields
        
        # Check for paragraph_content specifically
        if 'paragraph_content' not in available_fields:
            log_console(f"ERROR: 'paragraph_content' not found in input data!", "ERROR")
            log_console(f"Available fields: {', '.join(available_fields)}", "ERROR")
            # Check if there's a similar field
            content_fields = [f for f in available_fields if 'content' in f.lower() or 'text' in f.lower()]
            if content_fields:
                log_console(f"Found similar fields: {content_fields}", "WARNING")
        else:
            # Check if paragraph_content is empty
            sample_content = first_record.get('paragraph_content', '')
            if not sample_content:
                log_console(f"WARNING: 'paragraph_content' exists but is empty!", "WARNING")
            else:
                log_console(f"paragraph_content found with {len(sample_content)} characters", "INFO")
    
    # Get embedding config
    embed_config = config["stage_08_embeddings_generation"].get("embedding_config", {})
    target_chunk_size = embed_config.get("target_chunk_size", 500)
    chunk_threshold = embed_config.get("chunk_threshold", 1000)
    min_final_chunk = embed_config.get("min_final_chunk", 300)
    batch_size = embed_config.get("batch_size", 50)  # Default to 50 texts per batch
    
    # Separate Management Discussion and Q&A sections
    # Note: Section names come from Stage 4 validation config (usually all caps)
    md_records = [r for r in transcript_records if r.get('section_name') in ['MANAGEMENT DISCUSSION SECTION', 'Management Discussion Section']]
    qa_records = [r for r in transcript_records if r.get('section_name') == 'Q&A']
    
    log_console(f"Processing {len(md_records)} MD records and {len(qa_records)} Q&A records")
    
    # Process Management Discussion by speaker blocks
    md_blocks = defaultdict(list)
    for record in md_records:
        speaker_block_id = record.get('speaker_block_id')
        if speaker_block_id:
            md_blocks[speaker_block_id].append(record)
    
    # Process Q&A by qa_group_id first, then speaker blocks within
    qa_groups = defaultdict(lambda: defaultdict(list))
    for record in qa_records:
        qa_group_id = record.get('qa_group_id')
        speaker_block_id = record.get('speaker_block_id')
        if qa_group_id and speaker_block_id:
            qa_groups[qa_group_id][speaker_block_id].append(record)
    
    log_console(f"Created {len(md_blocks)} MD speaker blocks and {len(qa_groups)} Q&A groups")
    
    # Collect all chunks first before batching
    chunks_to_process = []
    
    # Process Management Discussion blocks
    for speaker_block_id, block_records in md_blocks.items():
        if not block_records:
            continue
            
        # Extract speaker and paragraphs
        speaker = block_records[0].get('speaker', 'Unknown Speaker')
        paragraphs = [str(r.get('paragraph_content', '')) for r in block_records if r.get('paragraph_content')]
        
        if not paragraphs:
            continue
            
        # Format content with speaker header
        formatted_content = format_md_content(speaker, paragraphs)
        block_tokens = count_tokens(formatted_content)
        
        # Get classifications from first record (all records in block have same classifications)
        classification_ids = block_records[0].get('classification_ids', [])
        classification_names = block_records[0].get('classification_names', [])
        
        # Strategy: Keep as single chunk if possible
        if block_tokens <= chunk_threshold:
            # Single chunk for entire speaker block
            first_record = block_records[0]
            chunk_data = {
                'record': first_record,
                'paragraph_id': f"md_speaker_{speaker_block_id}",
                'block_tokens': block_tokens,
                'chunk_id': 1,
                'total_chunks': 1,
                'chunk_text': formatted_content,
                'chunk_tokens': block_tokens,
                'block_level_chunk': True,
                'paragraphs_in_block': len(block_records),
                'block_paragraph_ids': [r.get('paragraph_id', '') for r in block_records],
                'classification_ids': classification_ids,
                'classification_names': classification_names
            }
            chunks_to_process.append(chunk_data)
            log_console(f"MD speaker block {speaker_block_id} kept as single chunk ({block_tokens} tokens)")
        else:
            # Need to chunk at paragraph boundaries
            current_paragraphs = []
            current_tokens = count_tokens(f"[{speaker}]\n\n")  # Header tokens
            chunk_id = 1
            
            for i, para in enumerate(paragraphs):
                para_tokens = count_tokens(para)
                
                # Check if adding this paragraph exceeds target
                if current_tokens + para_tokens > target_chunk_size and current_paragraphs:
                    # Save current chunk
                    chunk_content = format_md_content(speaker, current_paragraphs)
                    chunk_records = block_records[:len(current_paragraphs)]
                    
                    chunks_to_process.append({
                        'record': block_records[0],
                        'paragraph_id': f"md_speaker_{speaker_block_id}_chunk_{chunk_id}",
                        'block_tokens': block_tokens,
                        'chunk_id': chunk_id,
                        'total_chunks': -1,  # Will update later
                        'chunk_text': chunk_content,
                        'chunk_tokens': count_tokens(chunk_content),
                        'block_level_chunk': False,
                        'paragraphs_in_chunk': len(current_paragraphs),
                        'chunk_paragraph_ids': [chunk_records[j].get('paragraph_id', '') for j in range(len(current_paragraphs))],
                        'classification_ids': classification_ids,
                        'classification_names': classification_names
                    })
                    
                    # Start new chunk
                    current_paragraphs = [para]
                    current_tokens = count_tokens(f"[{speaker}]\n\n") + para_tokens
                    chunk_id += 1
                else:
                    current_paragraphs.append(para)
                    current_tokens += para_tokens
            
            # Save last chunk (with smart handling for small chunks)
            if current_paragraphs:
                chunk_content = format_md_content(speaker, current_paragraphs)
                final_chunk_tokens = count_tokens(chunk_content)
                start_idx = len(paragraphs) - len(current_paragraphs)
                chunk_records = block_records[start_idx:]
                
                # Check if final chunk is too small and we have previous chunks
                if final_chunk_tokens < min_final_chunk and chunk_id > 1 and len(chunks_to_process) > 0:
                    # Merge with previous chunk
                    prev_chunk_idx = len(chunks_to_process) - 1
                    prev_chunk = chunks_to_process[prev_chunk_idx]
                    
                    # Combine paragraphs from both chunks
                    prev_para_count = prev_chunk['paragraphs_in_chunk']
                    combined_start_idx = len(paragraphs) - len(current_paragraphs) - prev_para_count
                    combined_paragraphs = paragraphs[combined_start_idx:]
                    combined_content = format_md_content(speaker, combined_paragraphs)
                    combined_records = block_records[combined_start_idx:]
                    
                    # Update the previous chunk with combined content
                    chunks_to_process[prev_chunk_idx] = {
                        'record': block_records[0],
                        'paragraph_id': f"md_speaker_{speaker_block_id}_chunk_{chunk_id-1}",
                        'block_tokens': block_tokens,
                        'chunk_id': chunk_id - 1,
                        'total_chunks': chunk_id - 1,  # One less chunk now
                        'chunk_text': combined_content,
                        'chunk_tokens': count_tokens(combined_content),
                        'block_level_chunk': False,
                        'paragraphs_in_chunk': len(combined_paragraphs),
                        'chunk_paragraph_ids': [combined_records[j].get('paragraph_id', '') for j in range(len(combined_records))],
                        'classification_ids': classification_ids,
                        'classification_names': classification_names
                    }
                    chunk_id -= 1  # Decrease total chunk count
                    log_console(f"Merged small final chunk ({final_chunk_tokens} tokens) with previous chunk")
                else:
                    # Keep as separate chunk
                    chunks_to_process.append({
                        'record': block_records[0],
                        'paragraph_id': f"md_speaker_{speaker_block_id}_chunk_{chunk_id}",
                        'block_tokens': block_tokens,
                        'chunk_id': chunk_id,
                        'total_chunks': chunk_id,
                        'chunk_text': chunk_content,
                        'chunk_tokens': final_chunk_tokens,
                        'block_level_chunk': False,
                        'paragraphs_in_chunk': len(current_paragraphs),
                        'chunk_paragraph_ids': [chunk_records[j].get('paragraph_id', '') for j in range(len(current_paragraphs))],
                        'classification_ids': classification_ids,
                        'classification_names': classification_names
                    })
            
            # Update total_chunks for this MD block
            for chunk in chunks_to_process[-chunk_id:]:
                if chunk['total_chunks'] == -1:
                    chunk['total_chunks'] = chunk_id
            
            log_console(f"MD speaker block {speaker_block_id} split into {chunk_id} chunks")
    
    # Process Q&A groups
    for qa_group_id, speaker_blocks in qa_groups.items():
        # Collect all records and speaker blocks for this Q&A group
        qa_group_records = []
        qa_speaker_blocks = []
        
        # Sort speaker blocks by ID to maintain order
        for speaker_block_id in sorted(speaker_blocks.keys()):
            block_records = speaker_blocks[speaker_block_id]
            if block_records:
                qa_group_records.extend(block_records)
                speaker = block_records[0].get('speaker', 'Unknown Speaker')
                paragraphs = [str(r.get('paragraph_content', '')) for r in block_records if r.get('paragraph_content')]
                if paragraphs:
                    qa_speaker_blocks.append({
                        'speaker': speaker,
                        'paragraphs': paragraphs,
                        'records': block_records
                    })
        
        if not qa_speaker_blocks:
            continue
        
        # Format the entire Q&A group
        formatted_content = format_qa_content(qa_group_id, qa_speaker_blocks)
        qa_group_tokens = count_tokens(formatted_content)
        
        # Get classifications from first record (all records should have same classifications at block level)
        classification_ids = qa_group_records[0].get('classification_ids', []) if qa_group_records else []
        classification_names = qa_group_records[0].get('classification_names', []) if qa_group_records else []
        
        # Try to keep entire Q&A group together if possible
        if qa_group_tokens <= chunk_threshold:
            # Single chunk for entire Q&A group
            first_record = qa_group_records[0]
            chunk_data = {
                'record': first_record,
                'paragraph_id': f"qa_group_{qa_group_id}",
                'block_tokens': qa_group_tokens,
                'chunk_id': 1,
                'total_chunks': 1,
                'chunk_text': formatted_content,
                'chunk_tokens': qa_group_tokens,
                'block_level_chunk': True,
                'paragraphs_in_block': len(qa_group_records),
                'block_paragraph_ids': [r.get('paragraph_id', '') for r in qa_group_records],
                'classification_ids': classification_ids,
                'classification_names': classification_names,
                'qa_group_id': qa_group_id
            }
            chunks_to_process.append(chunk_data)
            log_console(f"Q&A group {qa_group_id} kept as single chunk ({qa_group_tokens} tokens)")
        else:
            # Need to split Q&A group - try to keep speaker blocks together
            chunk_id = 1
            current_speaker_blocks = []
            current_records = []
            header_tokens = count_tokens(f"[Q&A Group {qa_group_id}]\n\n")
            current_tokens = header_tokens
            
            for speaker_block in qa_speaker_blocks:
                # Calculate tokens for this speaker block
                speaker_text = f"[{speaker_block['speaker']}]\n" + "\n\n".join(speaker_block['paragraphs'])
                speaker_tokens = count_tokens(speaker_text)
                
                # Check if adding this speaker block exceeds target
                if current_tokens + speaker_tokens > target_chunk_size and current_speaker_blocks:
                    # Save current chunk
                    chunk_content = format_qa_content(qa_group_id, current_speaker_blocks)
                    
                    chunks_to_process.append({
                        'record': current_records[0],
                        'paragraph_id': f"qa_group_{qa_group_id}_chunk_{chunk_id}",
                        'block_tokens': qa_group_tokens,
                        'chunk_id': chunk_id,
                        'total_chunks': -1,  # Will update later
                        'chunk_text': chunk_content,
                        'chunk_tokens': count_tokens(chunk_content),
                        'block_level_chunk': False,
                        'paragraphs_in_chunk': len(current_records),
                        'chunk_paragraph_ids': [r.get('paragraph_id', '') for r in current_records],
                        'classification_ids': classification_ids,
                        'classification_names': classification_names,
                        'qa_group_id': qa_group_id
                    })
                    
                    # Start new chunk
                    current_speaker_blocks = [speaker_block]
                    current_records = speaker_block['records']
                    current_tokens = header_tokens + speaker_tokens
                    chunk_id += 1
                else:
                    current_speaker_blocks.append(speaker_block)
                    current_records.extend(speaker_block['records'])
                    current_tokens += speaker_tokens
            
            # Save last chunk if any
            if current_speaker_blocks:
                chunk_content = format_qa_content(qa_group_id, current_speaker_blocks)
                
                chunks_to_process.append({
                    'record': current_records[0],
                    'paragraph_id': f"qa_group_{qa_group_id}_chunk_{chunk_id}",
                    'block_tokens': qa_group_tokens,
                    'chunk_id': chunk_id,
                    'total_chunks': chunk_id,
                    'chunk_text': chunk_content,
                    'chunk_tokens': count_tokens(chunk_content),
                    'block_level_chunk': False,
                    'paragraphs_in_chunk': len(current_records),
                    'chunk_paragraph_ids': [r.get('paragraph_id', '') for r in current_records],
                    'classification_ids': classification_ids,
                    'classification_names': classification_names,
                    'qa_group_id': qa_group_id
                })
            
            # Update total_chunks for this Q&A group
            qa_chunk_start = len(chunks_to_process) - chunk_id
            for i in range(qa_chunk_start, len(chunks_to_process)):
                if chunks_to_process[i].get('qa_group_id') == qa_group_id and chunks_to_process[i]['total_chunks'] == -1:
                    chunks_to_process[i]['total_chunks'] = chunk_id
            
            log_console(f"Q&A group {qa_group_id} split into {chunk_id} chunks")
    
    # Debug: Log chunks_to_process status
    log_console(f"Total chunks created: {len(chunks_to_process)}")
    if len(chunks_to_process) == 0:
        log_console("WARNING: No chunks were created!", "WARNING")
        log_console(f"MD blocks processed: {len(md_blocks)}", "WARNING")
        log_console(f"Q&A groups processed: {len(qa_groups)}", "WARNING")
    
    # Process chunks in batches
    total_chunks = len(chunks_to_process)
    log_console(f"Processing {total_chunks} chunks in batches of {batch_size}")
    
    for batch_start in range(0, total_chunks, batch_size):
        batch_end = min(batch_start + batch_size, total_chunks)
        batch_chunks = chunks_to_process[batch_start:batch_end]
        
        # Extract texts for this batch
        batch_texts = [chunk['chunk_text'] for chunk in batch_chunks]
        
        try:
            # Generate embeddings for the batch
            log_console(f"Generating embeddings for batch {batch_start//batch_size + 1} ({len(batch_texts)} texts)")
            embeddings = generate_embeddings_batch(batch_texts)
            
            if embeddings:
                # Create enhanced records with embeddings
                for i, chunk_info in enumerate(batch_chunks):
                    enhanced_record = {
                        **chunk_info['record'],  # Include all original fields
                        'paragraph_tokens': chunk_info.get('paragraph_tokens', chunk_info['chunk_tokens']),  # Use chunk_tokens if paragraph_tokens not set
                        'block_tokens': chunk_info['block_tokens'],  # Total tokens in the speaker block
                        'chunk_id': chunk_info['chunk_id'],
                        'total_chunks': chunk_info['total_chunks'],
                        'chunk_text': chunk_info['chunk_text'],
                        'chunk_tokens': chunk_info['chunk_tokens'],
                        'embedding': embeddings[i],
                        # Include classification fields
                        'classification_ids': chunk_info.get('classification_ids', []),
                        'classification_names': chunk_info.get('classification_names', []),
                        # Include chunk metadata if present
                        'block_level_chunk': chunk_info.get('block_level_chunk', False),
                        'paragraphs_in_chunk': chunk_info.get('paragraphs_in_chunk', 1),
                        'chunk_paragraph_ids': chunk_info.get('chunk_paragraph_ids', chunk_info.get('block_paragraph_ids', [])),
                        'paragraph_chunk': chunk_info.get('paragraph_chunk', '')
                    }
                    enhanced_records.append(enhanced_record)
                    enhanced_error_logger.total_embeddings += 1
                    enhanced_error_logger.total_chunks += 1
            else:
                # Log errors for failed batch
                for chunk_info in batch_chunks:
                    enhanced_error_logger.log_embedding_error(
                        transcript_id,
                        chunk_info['paragraph_id'],
                        f"Failed to generate embedding in batch for chunk {chunk_info['chunk_id']}"
                    )
                    
        except Exception as e:
            # Log error for entire batch
            log_console(f"Batch embedding failed: {str(e)}", "ERROR")
            for chunk_info in batch_chunks:
                enhanced_error_logger.log_embedding_error(
                    transcript_id,
                    chunk_info['paragraph_id'],
                    f"Batch embedding error: {str(e)}"
                )
    
    log_execution(f"Processed transcript {transcript_id}", {
        "records_in": len(transcript_records),
        "records_out": len(enhanced_records),
        "chunks_created": enhanced_error_logger.total_chunks
    })
    
    return enhanced_records


def save_failed_transcripts(failed_transcripts: List[Dict], nas_conn: SMBConnection):
    """Save list of failed transcripts to a separate JSON file."""
    if not failed_transcripts:
        return
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    failed_summary = {
        "stage": "08_embeddings_generation",
        "timestamp": timestamp,
        "total_failed": len(failed_transcripts),
        "failed_transcripts": failed_transcripts
    }
    
    failed_path = nas_path_join(
        config["stage_08_embeddings_generation"]["output_logs_path"],
        f"stage_08_failed_transcripts_{timestamp}.json"
    )
    
    try:
        failed_data = json.dumps(failed_summary, indent=2, default=str)
        file_obj = io.BytesIO(failed_data.encode('utf-8'))
        
        if nas_upload_file(nas_conn, file_obj, failed_path):
            log_console(f"Saved {len(failed_transcripts)} failed transcripts")
            log_execution(f"Failed transcripts saved", {"count": len(failed_transcripts), "path": failed_path})
    except Exception as e:
        log_error(f"Failed to save failed transcripts: {e}", "save_failed")


def main():
    """Main execution function."""
    global config, logger, ssl_cert_path, llm_client, oauth_token
    
    # Initialize logging
    logger = setup_logging()
    log_console("=" * 50)
    log_console("Stage 8: Embeddings Generation")
    log_console("=" * 50)
    
    start_time = datetime.now()
    
    try:
        # Validate environment variables
        validate_environment_variables()
        
        # Connect to NAS
        nas_conn = get_nas_connection()
        if not nas_conn:
            raise RuntimeError("Failed to connect to NAS")
        
        # Load configuration
        config = load_stage_config(nas_conn)
        
        # Setup SSL certificate
        ssl_cert_path = setup_ssl_certificate(nas_conn)
        
        # Setup tokenizer with fallback support
        tokenizer_success = setup_tokenizer()
        if not tokenizer_success:
            log_console("=" * 50)
            log_console("WARNING: Running with fallback token estimation")
            log_console("Token counts will be approximate")
            log_console("Consider installing tiktoken for accurate counts")
            log_console("=" * 50)
            log_execution("Stage 8 running with fallback token estimation", {
                "reason": "tiktoken unavailable",
                "impact": "Token counts will be approximate"
            })
        
        # Load Stage 7 data
        stage7_data = load_stage7_data(nas_conn)
        
        # Group by transcript (matching Stage 7's grouping logic)
        transcripts = defaultdict(list)
        for record in stage7_data:
            # Use filename as the transcript key (from Stage 7 output)
            filename = record.get("filename", "")
            if filename:
                transcript_id = filename
            else:
                # Fallback to ticker_event_id format
                ticker = record.get("ticker", "UNK")
                event_id = record.get("event_id", "unknown")
                transcript_id = f"{ticker}_{event_id}"
            transcripts[transcript_id].append(record)
        
        log_console(f"Loaded {len(stage7_data)} records from {len(transcripts)} transcripts")
        
        # Check for dev mode
        if config["stage_08_embeddings_generation"].get("dev_mode", False):
            max_transcripts = config["stage_08_embeddings_generation"].get("dev_max_transcripts", 10)
            transcript_items = list(transcripts.items())[:max_transcripts]
            transcripts = dict(transcript_items)
            log_console(f"Development mode: limited to {len(transcripts)} transcripts")
        
        # Initialize enhanced error logger
        enhanced_error_logger = EnhancedErrorLogger()
        enhanced_error_logger.using_fallback_tokenizer = (tokenizer is None)

        # Analyze resume state
        log_console("=" * 50)
        log_console("Analyzing resume state...")
        to_process_ids, completed_ids, failed_ids = analyze_resume_state(transcripts)

        log_console(f"Resume analysis:")
        log_console(f"  To process: {len(to_process_ids)}")
        log_console(f"  Already completed: {len(completed_ids)}")
        log_console(f"  Previously failed: {len(failed_ids)}")
        log_console("=" * 50)

        # Filter transcripts to process
        transcripts_to_process = {
            tid: transcripts[tid] for tid in to_process_ids if tid in transcripts
        }

        if len(transcripts_to_process) == 0:
            log_console("All transcripts already processed. Moving to consolidation...")
        else:
            log_console(f"Processing {len(transcripts_to_process)} transcripts...")

        # Process each transcript with crash-resilient save
        failed_transcripts = []
        successful_count = 0

        for i, (transcript_id, transcript_records) in enumerate(transcripts_to_process.items(), 1):
            transcript_start = datetime.now()

            try:
                log_console(f"\n[{i}/{len(transcripts_to_process)}] {transcript_id} ({len(transcript_records)} records)")

                # Check if records are empty
                if not transcript_records:
                    log_console(f"ERROR: Transcript {transcript_id} has no records!", "ERROR")
                    failed_transcripts.append({
                        "transcript": transcript_id,
                        "timestamp": datetime.now().isoformat(),
                        "error": "No records found"
                    })
                    continue

                # Refresh OAuth token per transcript
                sample_record = transcript_records[0] if transcript_records else {}
                refresh_oauth_token_for_transcript(sample_record)

                # Process and save with resilient error handling
                success = process_and_save_transcript_resilient(
                    transcript_id,
                    transcript_records,
                    enhanced_error_logger
                )

                if success:
                    successful_count += 1
                else:
                    failed_transcripts.append({
                        "transcript": transcript_id,
                        "timestamp": datetime.now().isoformat(),
                        "error": "Processing failed after retries"
                    })

                transcript_time = datetime.now() - transcript_start
                log_console(f"  Time: {transcript_time}")

                # Rate limiting
                time.sleep(1)

            except Exception as e:
                transcript_time = datetime.now() - transcript_start
                enhanced_error_logger.log_processing_error(transcript_id, f"Failed after {transcript_time}: {e}")

                failed_transcripts.append({
                    "transcript": transcript_id,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                    "processing_time": str(transcript_time)
                })

                log_console(f"✗ Failed to process transcript {transcript_id}: {e}", "ERROR")

        # Save failed transcripts summary
        if failed_transcripts:
            save_failed_transcripts(failed_transcripts, nas_conn)

        # Consolidate individual files
        log_console("=" * 50)
        log_console("Starting consolidation of individual files...")
        log_console("=" * 50)

        consolidated_path = consolidate_individual_files()

        if consolidated_path:
            log_console(f"✓ Consolidated file: {consolidated_path}")
        else:
            log_console("✗ Consolidation failed - individual files preserved", "WARNING")
        
        # Calculate final statistics
        end_time = datetime.now()
        total_time = end_time - start_time

        stage_summary = {
            "stage": "08_embeddings_generation",
            "mode": "crash_resilient_individual_files",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_time": str(total_time),
            "total_transcripts": len(transcripts),
            "transcripts_to_process": len(transcripts_to_process),
            "transcripts_completed": len(completed_ids) + successful_count,
            "transcripts_skipped": len(completed_ids),
            "transcripts_newly_processed": successful_count,
            "transcripts_failed": len(failed_transcripts),
            "total_input_records": len(stage7_data),
            "total_embeddings": enhanced_error_logger.total_embeddings,
            "total_chunks": enhanced_error_logger.total_chunks,
            "embedding_errors": len(enhanced_error_logger.embedding_errors),
            "chunking_errors": len(enhanced_error_logger.chunking_errors),
            "processing_errors": len(enhanced_error_logger.processing_errors),
            "tokenizer_method": "fallback_estimation" if enhanced_error_logger.using_fallback_tokenizer else "tiktoken",
            "tokenizer_warning": "Token counts are approximate" if enhanced_error_logger.using_fallback_tokenizer else "Accurate token counts",
            "output_format": "Individual CSV files with consolidation",
            "consolidated_file": consolidated_path if consolidated_path else "consolidation_failed"
        }

        # Save logs
        save_logs_to_nas(nas_conn, stage_summary, enhanced_error_logger)

        # Print summary
        log_console("=" * 50)
        log_console("STAGE 8 SUMMARY")
        log_console("=" * 50)
        log_console(f"Total time: {total_time}")
        log_console(f"Total transcripts: {len(transcripts)}")
        log_console(f"  Already completed (skipped): {len(completed_ids)}")
        log_console(f"  Newly processed: {successful_count}")
        log_console(f"  Failed: {len(failed_transcripts)}")
        log_console(f"Embeddings generated: {enhanced_error_logger.total_embeddings}")
        log_console(f"Total chunks created: {enhanced_error_logger.total_chunks}")
        if consolidated_path:
            log_console(f"Consolidated file: {consolidated_path}")
        log_console("=" * 50)
        
        # Close NAS connection
        nas_conn.close()
        
        # Clean up SSL certificate
        if ssl_cert_path and os.path.exists(ssl_cert_path):
            os.unlink(ssl_cert_path)
        
        return 0 if len(failed_transcripts) == 0 else 1
    
    except Exception as e:
        log_console(f"Stage 8 failed: {e}", "ERROR")
        log_error(f"Stage 8 fatal error: {e}", "fatal", {"traceback": str(e)})
        return 1


if __name__ == "__main__":
    exit(main())
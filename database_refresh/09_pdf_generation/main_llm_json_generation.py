"""
Stage 9: LLM-Optimized JSON Generation
Transforms Stage 7 flat JSON structure into hierarchical format optimized for LLM consumption.
Self-contained standalone script that loads config from NAS at runtime.

Architecture based on Stage 7/8 patterns with hierarchical JSON transformation.
Creates clean, token-efficient JSON structure for direct LLM processing.
"""

import os
import tempfile
import logging
import json
import time
from datetime import datetime
from urllib.parse import quote
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict, OrderedDict

import yaml
from smb.SMBConnection import SMBConnection
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global variables
config = {}
logger = None
execution_log = []  # Detailed execution log entries
error_log = []  # Error log entries (only if errors occur)


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
        self.json_errors = []
        self.transformation_errors = []
        self.processing_errors = []
        self.validation_errors = []
        self.total_transcripts = 0
        self.successful_transcripts = 0
        self.failed_transcripts = []
    
    def log_json_error(self, transcript_id: str, error: str):
        """Log JSON generation error."""
        self.json_errors.append({
            "transcript_id": transcript_id,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
        if transcript_id not in self.failed_transcripts:
            self.failed_transcripts.append(transcript_id)
    
    def log_transformation_error(self, transcript_id: str, error: str):
        """Log transformation error."""
        self.transformation_errors.append({
            "transcript_id": transcript_id,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_processing_error(self, transcript_id: str, error: str):
        """Log general processing error."""
        self.processing_errors.append({
            "transcript_id": transcript_id,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_validation_error(self, transcript_id: str, field: str, issue: str):
        """Log validation error."""
        self.validation_errors.append({
            "transcript_id": transcript_id,
            "field": field,
            "issue": issue,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get error summary statistics."""
        return {
            "total_transcripts": self.total_transcripts,
            "successful_transcripts": self.successful_transcripts,
            "failed_transcripts": len(self.failed_transcripts),
            "json_errors": len(self.json_errors),
            "transformation_errors": len(self.transformation_errors),
            "processing_errors": len(self.processing_errors),
            "validation_errors": len(self.validation_errors),
            "failed_transcript_ids": self.failed_transcripts
        }
    
    def get_detailed_report(self) -> Dict[str, Any]:
        """Get detailed error report."""
        return {
            "summary": self.get_summary(),
            "json_errors": self.json_errors,
            "transformation_errors": self.transformation_errors,
            "processing_errors": self.processing_errors,
            "validation_errors": self.validation_errors
        }


# NAS utility functions
def nas_connect() -> SMBConnection:
    """Create connection to NAS server."""
    share_name = os.getenv("NAS_SHARE_NAME")
    user = os.getenv("NAS_USER")
    password = os.getenv("NAS_PASSWORD")
    domain = os.getenv("NAS_DOMAIN", "")
    
    try:
        # Get credentials from environment (matching Stage 7/8 patterns)
        if not all([share_name, user, password]):
            raise ValueError(
                f"Missing NAS credentials. Check environment variables.\n"
                f"NAS_SHARE_NAME: {'✓' if share_name else '✗'}\n"
                f"NAS_USER: {'✓' if user else '✗'}\n"
                f"NAS_PASSWORD: {'✓' if password else '✗'}"
            )
        
        # Create SMB connection (matching Stage 7/8 patterns)
        conn = SMBConnection(user, password, "client", share_name, domain=domain, use_ntlm_v2=True, is_direct_tcp=True)
        
        # Try multiple server name formats
        server_variants = [
            share_name,
            share_name.split('.')[0] if '.' in share_name else share_name,
            share_name.replace('.local', '') if '.local' in share_name else share_name
        ]
        
        connected = False
        for server in server_variants:
            try:
                if conn.connect(server, 445):
                    connected = True
                    log_execution(f"Connected to NAS using server name: {server}")
                    break
            except Exception:
                continue
        
        if not connected:
            raise ConnectionError(f"Failed to connect to NAS server: {share_name}")
        
        return conn
    
    except Exception as e:
        log_error(f"NAS connection failed: {e}", "nas_connection")
        raise


def nas_path_exists(conn: SMBConnection, path: str) -> bool:
    """Check if path exists on NAS."""
    try:
        share_name = os.getenv("NAS_SHARE_NAME")
        conn.listPath(share_name, path)
        return True
    except Exception:
        return False


def nas_download_file(conn: SMBConnection, nas_path: str) -> bytes:
    """Download file from NAS and return contents."""
    try:
        share_name = os.getenv("NAS_SHARE_NAME")
        file_obj = io.BytesIO()
        conn.retrieveFile(share_name, nas_path, file_obj)
        return file_obj.getvalue()
    except Exception as e:
        log_error(f"Failed to download {nas_path}: {e}", "nas_download")
        return None


def nas_upload_file(conn: SMBConnection, local_content: bytes, nas_path: str) -> bool:
    """Upload content to NAS."""
    try:
        share_name = os.getenv("NAS_SHARE_NAME")
        
        # Ensure parent directory exists
        parent_dir = "/".join(nas_path.split("/")[:-1])
        if parent_dir and not nas_path_exists(conn, parent_dir):
            nas_create_directory(conn, parent_dir)
        
        # Upload file
        file_obj = io.BytesIO(local_content)
        conn.storeFile(share_name, nas_path, file_obj)
        
        log_execution(f"Uploaded file to NAS: {nas_path}", {"size_bytes": len(local_content)})
        return True
    
    except Exception as e:
        log_error(f"Failed to upload to {nas_path}: {e}", "nas_upload")
        return False


def nas_create_directory(conn: SMBConnection, path: str) -> bool:
    """Create directory on NAS if it doesn't exist."""
    try:
        share_name = os.getenv("NAS_SHARE_NAME")
        
        # Build path progressively
        path_parts = path.split('/')
        current_path = ""
        
        for part in path_parts:
            if not part:
                continue
            current_path = f"{current_path}/{part}" if current_path else part
            
            if not nas_path_exists(conn, current_path):
                conn.createDirectory(share_name, current_path)
                log_execution(f"Created directory: {current_path}")
        
        return True
    
    except Exception as e:
        log_error(f"Failed to create directory {path}: {e}", "nas_mkdir")
        return False


def nas_save_logs(conn: SMBConnection, logs_path: str, transcript_count: int, error_logger: EnhancedErrorLogger):
    """Save execution and error logs to NAS."""
    global execution_log, error_log
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save execution log
    if execution_log:
        exec_log_path = f"{logs_path}/stage_09_llm_json_{timestamp}_execution.json"
        exec_log_content = json.dumps(execution_log, indent=2).encode('utf-8')
        nas_upload_file(conn, exec_log_content, exec_log_path)
        log_console(f"Saved execution log: {exec_log_path}")
    
    # Save error log if errors occurred
    if error_log or error_logger.failed_transcripts:
        error_report = error_logger.get_detailed_report()
        error_report["general_errors"] = error_log
        error_log_path = f"{logs_path}/stage_09_llm_json_{timestamp}_errors.json"
        error_log_content = json.dumps(error_report, indent=2).encode('utf-8')
        nas_upload_file(conn, error_log_content, error_log_path)
        log_console(f"Saved error log: {error_log_path}", "WARNING")
    
    # Log summary statistics
    summary = error_logger.get_summary()
    log_console(f"Processing complete: {summary['successful_transcripts']}/{transcript_count} transcripts")
    if summary['failed_transcripts'] > 0:
        log_console(f"Failed transcripts: {summary['failed_transcripts']}", "WARNING")


def load_config(conn: SMBConnection) -> Dict[str, Any]:
    """Load configuration from NAS."""
    global config
    
    try:
        # Use CONFIG_PATH environment variable like Stage 7/8
        config_path = os.getenv("CONFIG_PATH", "Finance Data and Analytics/DSA/Config/config.yaml")
        
        log_console(f"Loading configuration from: {config_path}")
        config_content = nas_download_file(conn, config_path)
        
        if not config_content:
            raise Exception("Failed to download config file from NAS")
        
        full_config = yaml.safe_load(config_content.decode('utf-8'))
        
        # Store the full config (following Stage 7/8 patterns)
        config = full_config.get("stage_09_pdf_generation", {})
        
        log_execution("Stage 9 configuration loaded successfully", {
            "config_path": config_path
        })
        
        return config
    
    except Exception as e:
        log_error(f"Failed to load configuration: {str(e)}", "config_load")
        raise


def transform_to_llm_format(transcript_records: List[Dict], transcript_id: str) -> Dict[str, Any]:
    """
    Transform flat Stage 7 structure to hierarchical LLM-optimized format.
    
    Args:
        transcript_records: List of paragraph-level records from Stage 7
        transcript_id: Unique identifier for the transcript
    
    Returns:
        Hierarchical JSON structure optimized for LLM consumption
    """
    if not transcript_records:
        raise ValueError("No records to transform")
    
    # Get metadata from first record
    first_record = transcript_records[0]
    
    # Build metadata
    metadata = {
        "filename": first_record.get("filename", ""),
        "transcript_type": first_record.get("transcript_type", ""),
        "institution_id": first_record.get("institution_id", ""),
        "institution_type": first_record.get("institution_type", ""),
        "ticker": first_record.get("ticker", ""),
        "company_name": first_record.get("company_name", ""),
        "fiscal_year": first_record.get("fiscal_year", ""),
        "fiscal_quarter": first_record.get("fiscal_quarter", ""),
        "title": first_record.get("title", ""),
        "total_paragraphs": len(transcript_records),
        "total_qa_conversations": 0  # Will be calculated
    }
    
    # Group records by section
    sections_dict = defaultdict(list)
    for record in transcript_records:
        section_id = record.get("section_id")
        sections_dict[section_id].append(record)
    
    # Build sections
    sections = []
    qa_conversation_count = 0
    
    for section_id in sorted(sections_dict.keys()):
        section_records = sections_dict[section_id]
        section_name = section_records[0].get("section_name", "")
        
        section = {
            "section_id": section_id,
            "section_name": section_name
        }
        
        # Check if this is Q&A section
        if "Q&A" in section_name.upper() or section_records[0].get("question_answer_flag"):
            # Process Q&A section
            qa_conversations = []
            qa_groups = defaultdict(list)
            
            # Group by qa_group_id
            for record in section_records:
                qa_group_id = record.get("qa_group_id")
                if qa_group_id is not None:
                    qa_groups[qa_group_id].append(record)
            
            # Build Q&A conversations
            for qa_group_id in sorted(qa_groups.keys()):
                qa_records = qa_groups[qa_group_id]
                qa_conversation_count += 1
                
                # Get classifications for this Q&A group (from first record)
                classification_ids = qa_records[0].get("classification_ids", [])
                classification_names = qa_records[0].get("classification_names", [])
                
                # Build classification mapping (id: name)
                classifications = {}
                if classification_ids and classification_names:
                    for i, class_id in enumerate(classification_ids):
                        if i < len(classification_names):
                            # Convert id to string for JSON key
                            classifications[str(class_id)] = classification_names[i]
                
                # Group by speaker_block_id within this Q&A conversation
                speaker_blocks_dict = defaultdict(list)
                for record in qa_records:
                    speaker_block_id = record.get("speaker_block_id")
                    speaker_blocks_dict[speaker_block_id].append(record)
                
                # Build speaker blocks for this Q&A conversation
                speaker_blocks = []
                for speaker_block_id in sorted(speaker_blocks_dict.keys()):
                    block_records = speaker_blocks_dict[speaker_block_id]
                    
                    speaker_block = {
                        "speaker_block_id": speaker_block_id,
                        "speaker": block_records[0].get("speaker", ""),
                        "paragraphs": []
                    }
                    
                    # Add paragraphs
                    for record in block_records:
                        paragraph = {
                            "paragraph_id": record.get("paragraph_id"),
                            "content": record.get("paragraph_content", "")
                        }
                        speaker_block["paragraphs"].append(paragraph)
                    
                    speaker_blocks.append(speaker_block)
                
                qa_conversation = {
                    "qa_group_id": qa_group_id,
                    "classifications": classifications,
                    "speaker_blocks": speaker_blocks
                }
                
                qa_conversations.append(qa_conversation)
            
            section["qa_conversations"] = qa_conversations
        
        else:
            # Process Management Discussion or other sections
            # Group by speaker_block_id
            speaker_blocks_dict = defaultdict(list)
            for record in section_records:
                speaker_block_id = record.get("speaker_block_id")
                speaker_blocks_dict[speaker_block_id].append(record)
            
            # Build speaker blocks
            speaker_blocks = []
            for speaker_block_id in sorted(speaker_blocks_dict.keys()):
                block_records = speaker_blocks_dict[speaker_block_id]
                
                speaker_block = {
                    "speaker_block_id": speaker_block_id,
                    "speaker": block_records[0].get("speaker", ""),
                    "paragraphs": []
                }
                
                # Add paragraphs
                for record in block_records:
                    paragraph = {
                        "paragraph_id": record.get("paragraph_id"),
                        "content": record.get("paragraph_content", "")
                    }
                    speaker_block["paragraphs"].append(paragraph)
                
                speaker_blocks.append(speaker_block)
            
            section["speaker_blocks"] = speaker_blocks
        
        sections.append(section)
    
    # Update Q&A conversation count in metadata
    metadata["total_qa_conversations"] = qa_conversation_count
    
    # Build final structure
    llm_json = {
        "metadata": metadata,
        "sections": sections
    }
    
    return llm_json


def process_transcript(transcript_id: str, records: List[Dict], error_logger: EnhancedErrorLogger) -> Optional[Dict]:
    """
    Process a single transcript and generate LLM-optimized JSON.
    
    Args:
        transcript_id: Unique identifier for the transcript
        records: List of paragraph records for this transcript
        error_logger: Error logger instance
    
    Returns:
        LLM-optimized JSON structure or None if failed
    """
    try:
        # Transform to LLM format
        llm_json = transform_to_llm_format(records, transcript_id)
        
        # Validate the structure
        if not llm_json.get("metadata"):
            error_logger.log_validation_error(transcript_id, "metadata", "Missing metadata")
            return None
        
        if not llm_json.get("sections"):
            error_logger.log_validation_error(transcript_id, "sections", "No sections found")
            return None
        
        log_execution(f"Successfully transformed transcript: {transcript_id}", {
            "sections": len(llm_json["sections"]),
            "paragraphs": llm_json["metadata"]["total_paragraphs"],
            "qa_conversations": llm_json["metadata"]["total_qa_conversations"]
        })
        
        return llm_json
    
    except Exception as e:
        error_logger.log_transformation_error(transcript_id, str(e))
        log_error(f"Failed to transform transcript {transcript_id}: {e}", "transformation")
        return None


def process_all_transcripts(conn: SMBConnection) -> Tuple[int, int]:
    """
    Process all transcripts from Stage 7 output.
    
    Returns:
        Tuple of (successful_count, failed_count)
    """
    global config
    
    try:
        # Download Stage 7 output (JSON format with paragraph-level data)
        input_path = config.get('input_data_path', 
                               'Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Refresh/stage_07_summarized_content.json')
        
        log_console("Downloading Stage 7 output from NAS...")
        stage_07_content = nas_download_file(conn, input_path)
        
        if not stage_07_content:
            raise Exception("Failed to download Stage 7 output")
        
        # Parse JSON data
        try:
            all_records = json.loads(stage_07_content.decode('utf-8'))
        except json.JSONDecodeError:
            # Try to parse as-is first, only repair if needed
            content_str = stage_07_content.decode('utf-8').strip()
            try:
                all_records = json.loads(content_str)
            except json.JSONDecodeError as e:
                # Only attempt repair for specific known issues
                if 'Expecting value' in str(e) and content_str.endswith(','):
                    # Remove trailing comma before closing bracket
                    content_str = content_str[:-1] + ']'
                    all_records = json.loads(content_str)
                else:
                    raise
        
        log_console(f"Loaded {len(all_records)} records from Stage 7")
        
        # Group records by transcript (using filename as key)
        transcripts = defaultdict(list)
        for record in all_records:
            filename = record.get('filename', 'unknown')
            transcripts[filename].append(record)
        
        log_console(f"Found {len(transcripts)} unique transcripts")
        
        # Check for dev mode
        if config.get("dev_mode", False):
            max_transcripts = config.get("dev_max_transcripts", 5)
            transcript_ids = list(transcripts.keys())[:max_transcripts]
            log_console(f"Dev mode: Processing only {len(transcript_ids)} transcripts")
        else:
            transcript_ids = list(transcripts.keys())
        
        # Initialize error logger
        error_logger = EnhancedErrorLogger()
        error_logger.total_transcripts = len(transcript_ids)
        
        # Get output path
        output_path = config.get("output_data_path", 
                                "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Refresh")
        json_output_dir = f"{output_path}/stage_09_llm_json"
        
        # Create output directory if needed
        nas_create_directory(conn, json_output_dir)
        
        # Process each transcript
        successful = 0
        failed = 0
        
        for idx, transcript_id in enumerate(transcript_ids, 1):
            try:
                # Log progress
                if idx % 10 == 0 or idx == 1:
                    log_console(f"Processing transcript {idx}/{len(transcript_ids)}: {transcript_id}")
                
                # Get records for this transcript
                transcript_records = transcripts[transcript_id]
                
                # Process transcript
                llm_json = process_transcript(transcript_id, transcript_records, error_logger)
                
                if llm_json:
                    # Generate output filename (replace .xml with .json)
                    output_filename = transcript_id.replace('.xml', '.json')
                    output_nas_path = f"{json_output_dir}/{output_filename}"
                    
                    # Convert to JSON and upload
                    json_content = json.dumps(llm_json, indent=2).encode('utf-8')
                    
                    if nas_upload_file(conn, json_content, output_nas_path):
                        successful += 1
                        error_logger.successful_transcripts += 1
                        log_execution(f"Generated LLM JSON: {output_filename}", {
                            "size_bytes": len(json_content),
                            "sections": len(llm_json["sections"]),
                            "paragraphs": llm_json["metadata"]["total_paragraphs"]
                        })
                    else:
                        failed += 1
                        error_logger.log_json_error(transcript_id, "Failed to upload JSON to NAS")
                else:
                    failed += 1
                    error_logger.log_json_error(transcript_id, "Transformation failed")
            
            except Exception as e:
                failed += 1
                error_logger.log_processing_error(transcript_id, str(e))
                log_error(f"Failed to process {transcript_id}: {e}", "processing")
        
        # Save logs
        logs_path = config.get("output_logs_path", 
                              "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Logs")
        nas_save_logs(conn, logs_path, len(transcript_ids), error_logger)
        
        return successful, failed
    
    except Exception as e:
        log_error(f"Critical error in transcript processing: {e}", "critical")
        raise


def main():
    """Main entry point for Stage 9: LLM-Optimized JSON Generation."""
    global logger, config
    
    start_time = time.time()
    
    try:
        # Setup logging
        logger = setup_logging()
        log_console("=" * 80)
        log_console("Stage 9: LLM-Optimized JSON Generation")
        log_console("=" * 80)
        
        # Connect to NAS
        log_console("Connecting to NAS...")
        conn = nas_connect()
        
        # Load configuration
        config = load_config(conn)
        
        # Process transcripts
        log_console("Starting LLM JSON generation...")
        successful, failed = process_all_transcripts(conn)
        
        # Report results
        elapsed_time = time.time() - start_time
        log_console("=" * 80)
        log_console(f"Stage 9 Complete:")
        log_console(f"  Successful: {successful}")
        log_console(f"  Failed: {failed}")
        log_console(f"  Total time: {elapsed_time:.2f} seconds")
        log_console("=" * 80)
        
        # Close NAS connection
        conn.close()
        
        # Exit with appropriate code
        if failed > 0:
            exit(1)
        else:
            exit(0)
    
    except Exception as e:
        log_console(f"Fatal error: {e}", "ERROR")
        log_error(f"Fatal error in main: {e}", "fatal")
        exit(1)


if __name__ == "__main__":
    main()
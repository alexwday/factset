"""
Stage 3: Transcript Content Extraction & Paragraph-Level Breakdown
Processes XML transcripts from Stage 2's processing queue and extracts paragraph-level content.
Self-contained standalone script that loads config from NAS at runtime.
"""

import pandas as pd
import os
from datetime import datetime, timedelta
import json
import tempfile
import io
import xml.etree.ElementTree as ET
from smb.SMBConnection import SMBConnection
from typing import Dict, List, Optional, Set, Tuple, Any
import warnings
from dotenv import load_dotenv
import re
import logging

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Load environment variables
load_dotenv()

# Authentication and connection settings from environment
API_USERNAME = os.getenv("API_USERNAME")
API_PASSWORD = os.getenv("API_PASSWORD")
PROXY_USER = os.getenv("PROXY_USER")
PROXY_PASSWORD = os.getenv("PROXY_PASSWORD")
PROXY_URL = os.getenv("PROXY_URL")
NAS_USERNAME = os.getenv("NAS_USERNAME")
NAS_PASSWORD = os.getenv("NAS_PASSWORD")
NAS_SERVER_IP = os.getenv("NAS_SERVER_IP")
NAS_SERVER_NAME = os.getenv("NAS_SERVER_NAME")
NAS_SHARE_NAME = os.getenv("NAS_SHARE_NAME")
NAS_BASE_PATH = os.getenv("NAS_BASE_PATH")
NAS_PORT = int(os.getenv("NAS_PORT", 445))
CONFIG_PATH = os.getenv("CONFIG_PATH")
CLIENT_MACHINE_NAME = os.getenv("CLIENT_MACHINE_NAME")
PROXY_DOMAIN = os.getenv("PROXY_DOMAIN", "MAPLE")

# Validate required environment variables
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
    "CONFIG_PATH",
    "CLIENT_MACHINE_NAME",
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(
        f"Missing required environment variables: {', '.join(missing_vars)}"
    )

# Global variables for configuration
config = {}
logger = None
error_logger = None


def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    temp_log_file = tempfile.NamedTemporaryFile(
        mode="w+",
        suffix=".log",
        prefix=f'stage_3_content_extraction_log_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_',
        delete=False,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(temp_log_file.name), logging.StreamHandler()],
    )

    logger = logging.getLogger(__name__)
    logger.temp_log_file = temp_log_file.name
    return logger


class EnhancedErrorLogger:
    """Handles separate error logging for different failure types."""

    def __init__(self):
        self.parsing_errors = []
        self.download_errors = []
        self.filesystem_errors = []
        self.validation_errors = []
        self.processing_errors = []

    def log_parsing_error(self, filename: str, error: str):
        """Log XML parsing errors."""
        self.parsing_errors.append(
            {
                "timestamp": datetime.now().isoformat(),
                "filename": filename,
                "error": error,
                "action_required": "Manual review of XML structure needed",
            }
        )

    def log_download_error(self, file_path: str, error: str):
        """Log file download errors."""
        self.download_errors.append(
            {
                "timestamp": datetime.now().isoformat(),
                "file_path": file_path,
                "error": error,
                "action_required": "Check NAS connectivity and file permissions",
            }
        )

    def log_processing_error(self, ticker: str, filename: str, error: str):
        """Log content processing errors."""
        self.processing_errors.append(
            {
                "timestamp": datetime.now().isoformat(),
                "ticker": ticker,
                "filename": filename,
                "error": error,
                "action_required": "Review content extraction logic",
            }
        )

    def save_error_logs(self, nas_conn: SMBConnection):
        """Save error logs to separate JSON files on NAS."""
        global logger
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        error_base_path = nas_path_join(NAS_BASE_PATH, "Outputs", "Logs", "Errors")
        nas_create_directory(nas_conn, error_base_path)

        # Save each error type if errors exist
        error_types = [
            ("parsing_errors", self.parsing_errors),
            ("download_errors", self.download_errors),
            ("filesystem_errors", self.filesystem_errors),
            ("validation_errors", self.validation_errors),
            ("processing_errors", self.processing_errors),
        ]

        summary = {
            "run_timestamp": timestamp,
            "total_errors": sum(len(errors) for _, errors in error_types),
            "errors_by_type": {
                error_type: len(errors) for error_type, errors in error_types
            },
        }

        for error_type, errors in error_types:
            if errors:
                filename = f"stage_3_{error_type}_{timestamp}.json"
                file_path = nas_path_join(error_base_path, filename)
                content = json.dumps({"summary": summary, "errors": errors}, indent=2)
                file_obj = io.BytesIO(content.encode("utf-8"))
                nas_upload_file(nas_conn, file_obj, file_path)
                logger.warning(f"Saved {len(errors)} {error_type} to {filename}")


def sanitize_url_for_logging(url: str) -> str:
    """Sanitize URL for logging by removing query parameters and auth tokens."""
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    except (ValueError, TypeError, AttributeError, ImportError) as e:
        logger.warning(f"URL sanitization failed: {e}")
        return "[URL_SANITIZED]"


def get_nas_connection() -> Optional[SMBConnection]:
    """Create and return an SMB connection to the NAS."""
    global logger
    try:
        conn = SMBConnection(
            username=NAS_USERNAME,
            password=NAS_PASSWORD,
            my_name=CLIENT_MACHINE_NAME,
            remote_name=NAS_SERVER_NAME,
            use_ntlm_v2=True,
            is_direct_tcp=True,
        )

        if conn.connect(NAS_SERVER_IP, NAS_PORT):
            logger.info("Connected to NAS successfully")
            return conn
        else:
            logger.error("Failed to connect to NAS")
            return None

    except Exception as e:
        logger.error(f"Error connecting to NAS: {e}")
        return None


def validate_config_schema(config: Dict[str, Any]) -> None:
    """Validate configuration schema and parameters."""
    global logger

    # Define required configuration structure for Stage 3
    required_structure = {
        "stage_3": {
            "dev_mode": bool,
            "dev_max_files": int,
            "input_source": str,
            "output_file": str,
            "output_path": str,
        },
        "monitored_institutions": dict,
    }

    # Validate top-level structure
    for top_key, top_value in required_structure.items():
        if top_key not in config:
            raise ValueError(f"Missing required configuration section: {top_key}")

        if isinstance(top_value, dict):
            # Validate nested structure
            for nested_key, expected_type in top_value.items():
                if nested_key not in config[top_key]:
                    raise ValueError(
                        f"Missing required configuration parameter: {top_key}.{nested_key}"
                    )

                actual_value = config[top_key][nested_key]
                if not isinstance(actual_value, expected_type):
                    raise ValueError(
                        f"Invalid type for {top_key}.{nested_key}: expected {expected_type}, got {type(actual_value)}"
                    )

    # Validate monitored institutions structure
    if not config["monitored_institutions"]:
        raise ValueError("monitored_institutions cannot be empty")

    logger.info("Configuration validation successful")


def load_stage_config(nas_conn: SMBConnection) -> Dict[str, Any]:
    """Load and validate shared configuration from NAS."""
    global logger
    try:
        logger.info("Loading shared configuration from NAS...")
        config_data = nas_download_file(nas_conn, CONFIG_PATH)

        if config_data:
            stage_config = json.loads(config_data.decode("utf-8"))
            logger.info("Successfully loaded shared configuration from NAS")

            # Validate configuration schema and parameters
            validate_config_schema(stage_config)

            return stage_config
        else:
            logger.error("Config file not found on NAS - script cannot proceed")
            raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        raise
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading config from NAS: {e}")
        raise


def nas_path_join(*parts: str) -> str:
    """Join path parts for NAS paths using forward slashes."""
    # Filter out empty parts and strip slashes
    clean_parts = []
    for part in parts:
        if part:
            clean_part = str(part).strip("/")
            if clean_part:
                clean_parts.append(clean_part)
    return "/".join(clean_parts)


def nas_file_exists(conn: SMBConnection, file_path: str) -> bool:
    """Check if a file exists on the NAS."""
    global logger
    try:
        conn.getAttributes(NAS_SHARE_NAME, file_path)
        return True
    except Exception as e:
        logger.debug(f"File existence check failed for {file_path}: {e}")
        return False


def nas_create_directory(conn: SMBConnection, dir_path: str) -> bool:
    """Create directory on NAS with safe iterative parent creation."""
    global logger

    # Normalize and validate path
    normalized_path = dir_path.strip("/").rstrip("/")
    if not normalized_path:
        logger.error("Cannot create directory with empty path")
        return False

    # Split path into components
    path_parts = [part for part in normalized_path.split("/") if part]
    if not path_parts:
        logger.error("Cannot create directory with invalid path")
        return False

    # Build path incrementally from root
    current_path = ""
    for part in path_parts:
        current_path = f"{current_path}/{part}" if current_path else part

        # Check if directory exists
        if nas_file_exists(conn, current_path):
            continue

        # Try to create directory
        try:
            conn.createDirectory(NAS_SHARE_NAME, current_path)
            logger.debug(f"Created directory: {current_path}")
        except Exception as e:
            # If it fails and doesn't exist, it's a real error
            if not nas_file_exists(conn, current_path):
                logger.error(f"Failed to create directory {current_path}: {e}")
                return False

    return True


def nas_upload_file(
    conn: SMBConnection, local_file_obj: io.BytesIO, nas_file_path: str
) -> bool:
    """Upload a file object to NAS."""
    global logger

    # Validate path for security
    if not validate_nas_path(nas_file_path):
        logger.error(f"Invalid NAS file path: {nas_file_path}")
        return False

    try:
        parent_dir = "/".join(nas_file_path.split("/")[:-1])
        if parent_dir:
            nas_create_directory(conn, parent_dir)

        conn.storeFile(NAS_SHARE_NAME, nas_file_path, local_file_obj)
        return True
    except Exception as e:
        logger.error(f"Failed to upload file to NAS {nas_file_path}: {e}")
        return False


def nas_download_file(conn: SMBConnection, nas_file_path: str) -> Optional[bytes]:
    """Download a file from NAS and return as bytes."""
    global logger

    # Validate path for security
    if not validate_nas_path(nas_file_path):
        logger.error(f"Invalid NAS file path: {nas_file_path}")
        return None

    try:
        file_obj = io.BytesIO()
        conn.retrieveFile(NAS_SHARE_NAME, nas_file_path, file_obj)
        file_obj.seek(0)
        return file_obj.read()
    except Exception as e:
        logger.error(f"Failed to download file from NAS {nas_file_path}: {e}")
        return None


def validate_file_path(path: str) -> bool:
    """Validate file path for security."""
    if not path or not isinstance(path, str):
        return False

    # Check for directory traversal attacks
    if ".." in path or path.startswith("/"):
        return False

    # Check for invalid characters
    invalid_chars = ["<", ">", ":", '"', "|", "?", "*", "\x00"]
    if any(char in path for char in invalid_chars):
        return False

    # Check path length
    if len(path) > 260:  # Windows MAX_PATH limitation
        return False

    return True


def validate_nas_path(path: str) -> bool:
    """Validate NAS path structure."""
    global logger
    
    if not path or not isinstance(path, str):
        logger.debug(f"NAS path validation failed: empty or not string: '{path}'")
        return False

    # Ensure path is relative and safe
    normalized = path.strip("/")
    if not normalized:
        logger.debug(f"NAS path validation failed: empty after normalization: '{path}'")
        return False
        
    parts = normalized.split("/")

    for part in parts:
        if not part or part in [".", ".."]:
            logger.debug(f"NAS path validation failed: invalid part '{part}' in path: '{path}'")
            return False
        if not validate_file_path(part):
            logger.debug(f"NAS path validation failed: file path validation failed for part '{part}' in path: '{path}'")
            return False

    return True


def parse_transcript_xml(xml_content: bytes) -> Dict:
    """Parse FactSet transcript XML and extract structured data."""
    global logger
    
    tree = ET.parse(io.BytesIO(xml_content))
    root = tree.getroot()
    
    # Handle namespace if present
    namespace = ""
    if root.tag.startswith('{'):
        namespace = root.tag.split('}')[0] + '}'
    
    # Helper function to handle namespaced tags
    def ns_tag(tag):
        return f"{namespace}{tag}" if namespace else tag
    
    # Extract metadata
    meta = root.find(ns_tag('meta'))
    if meta is None:
        # Debug: print available elements
        logger.debug(f"Root tag: {root.tag}")
        logger.debug(f"Root children: {[child.tag for child in root]}")
        raise ValueError("No meta section found in XML")
    
    # Extract title and date
    title = meta.find(ns_tag('title'))
    title_text = title.text if title is not None else "Untitled Transcript"
    
    date = meta.find(ns_tag('date'))
    date_text = date.text if date is not None else "Unknown Date"
    
    # Extract companies
    companies = []
    companies_elem = meta.find(ns_tag('companies'))
    if companies_elem is not None:
        for company in companies_elem.findall(ns_tag('company')):
            companies.append(company.text if company.text else "")
    
    # Extract participants and create speaker mapping
    participants = {}
    participants_elem = meta.find(ns_tag('participants'))
    if participants_elem is not None:
        for participant in participants_elem.findall(ns_tag('participant')):
            p_id = participant.get('id')
            if p_id:
                participants[p_id] = {
                    'type': participant.get('type', ''),
                    'affiliation': participant.get('affiliation', ''),
                    'affiliation_entity': participant.get('affiliation_entity', ''),
                    'title': participant.get('title', ''),
                    'entity': participant.get('entity', ''),
                    'name': participant.text.strip() if participant.text else 'Unknown Speaker'
                }
    
    # Extract body content
    body = root.find(ns_tag('body'))
    if body is None:
        raise ValueError("No body section found in XML")
    
    sections = []
    for section in body.findall(ns_tag('section')):
        section_name = section.get('name', 'Unnamed Section')
        speakers = []
        
        for speaker in section.findall(ns_tag('speaker')):
            speaker_id = speaker.get('id')
            speaker_type = speaker.get('type', '')  # 'q' or 'a' for Q&A sections
            
            # Extract paragraphs from plist
            paragraphs = []
            plist = speaker.find(ns_tag('plist'))
            if plist is not None:
                for p in plist.findall(ns_tag('p')):
                    if p.text:
                        paragraphs.append(p.text.strip())
            
            speakers.append({
                'id': speaker_id,
                'type': speaker_type,
                'paragraphs': paragraphs
            })
        
        sections.append({
            'name': section_name,
            'speakers': speakers
        })
    
    return {
        'title': title_text,
        'date': date_text,
        'companies': companies,
        'participants': participants,
        'sections': sections
    }


def format_speaker_string(participant_data: Dict) -> str:
    """
    Format speaker information into a clean string.
    Returns speaker_string
    """
    name = participant_data.get('name', 'Unknown Speaker')
    title = participant_data.get('title', '')
    affiliation = participant_data.get('affiliation', '')
    
    # Build speaker string: "Name, Title, Affiliation"
    parts = [name]
    if title:
        parts.append(title)
    if affiliation:
        parts.append(affiliation)
    
    speaker_string = ", ".join(parts)
    
    return speaker_string


def determine_qa_flag(speaker_type: str) -> Optional[str]:
    """Determine Q&A flag from speaker type attribute."""
    if speaker_type == 'q':
        return "question"
    elif speaker_type == 'a': 
        return "answer"
    else:
        return None


def extract_transcript_paragraphs(
    original_record: Dict[str, Any], 
    xml_content: bytes
) -> List[Dict[str, Any]]:
    """
    Extract all paragraphs from XML with sequential ordering.
    Each paragraph becomes a record with all original fields + new Stage 3 fields.
    """
    global logger, error_logger
    
    try:
        # Parse XML (reuse logic from HTML viewer)
        transcript_data = parse_transcript_xml(xml_content)
        paragraph_records = []
        global_paragraph_order = 1  # Sequential across entire transcript
        
        # Iterate through sections -> speakers -> paragraphs
        for section in transcript_data['sections']:
            section_name = section['name']
            
            for speaker_block in section['speakers']:
                speaker_id = speaker_block['id']
                speaker_type = speaker_block.get('type', '')  # 'q', 'a', or ''
                
                # Get speaker details from participants
                speaker_info = transcript_data['participants'].get(speaker_id, {})
                
                # Format speaker as string
                speaker_string = format_speaker_string(speaker_info)
                
                # Determine Q&A flag
                qa_flag = determine_qa_flag(speaker_type)
                
                # Process each paragraph in this speaker's plist
                for paragraph_text in speaker_block['paragraphs']:
                    if paragraph_text.strip():  # Skip empty paragraphs
                        
                        # Create record with ALL original fields + new Stage 3 fields
                        paragraph_record = {
                            **original_record,  # All fields from Stage 2
                            
                            # New Stage 3 fields (reordered)
                            "section_name": section_name,
                            "paragraph_order": global_paragraph_order,
                            "question_answer_flag": qa_flag,
                            "speaker": speaker_string,
                            "paragraph_content": paragraph_text.strip()
                        }
                        
                        paragraph_records.append(paragraph_record)
                        global_paragraph_order += 1
        
        logger.info(f"Extracted {len(paragraph_records)} paragraphs from {original_record.get('filename', 'unknown')}")
        return paragraph_records
    
    except Exception as e:
        error_logger.log_processing_error(
            original_record.get('ticker', 'unknown'),
            original_record.get('filename', 'unknown'),
            str(e)
        )
        logger.error(f"Failed to extract paragraphs from {original_record.get('filename', 'unknown')}: {e}")
        return []


def load_processing_queue(nas_conn: SMBConnection) -> List[Dict[str, Any]]:
    """Load files to process from Stage 2 output."""
    global logger, config
    
    input_path = nas_path_join(NAS_BASE_PATH, config["stage_3"]["input_source"])
    
    if not nas_file_exists(nas_conn, input_path):
        logger.warning(f"No processing queue found at {input_path}")
        return []
    
    try:
        queue_data = nas_download_file(nas_conn, input_path)
        if queue_data:
            queue_json = json.loads(queue_data.decode("utf-8"))
            files_to_process = queue_json.get("files", [])
            logger.info(f"Loaded {len(files_to_process)} files from processing queue")
            return files_to_process
        else:
            logger.error(f"Failed to download processing queue from {input_path}")
            return []
    
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in processing queue: {e}")
        return []
    except Exception as e:
        logger.error(f"Error loading processing queue: {e}")
        return []


def save_extracted_content(
    nas_conn: SMBConnection,
    all_paragraph_records: List[Dict[str, Any]]
) -> bool:
    """Save extracted content records to NAS."""
    global logger, config
    
    output_data = {
        "schema_version": "1.0",
        "processing_timestamp": datetime.now().isoformat(),
        "total_records": len(all_paragraph_records),
        "total_transcripts_processed": len(set(record.get('filename', '') for record in all_paragraph_records)),
        "records": all_paragraph_records
    }
    
    output_path = nas_path_join(
        NAS_BASE_PATH,
        config["stage_3"]["output_path"],
        config["stage_3"]["output_file"]
    )
    
    try:
        output_content = json.dumps(output_data, indent=2)
        output_file_obj = io.BytesIO(output_content.encode("utf-8"))
        
        if nas_upload_file(nas_conn, output_file_obj, output_path):
            logger.info(f"Saved {len(all_paragraph_records)} content records to {output_path}")
            return True
        else:
            logger.error(f"Failed to upload extracted content to {output_path}")
            return False
    
    except Exception as e:
        logger.error(f"Error saving extracted content: {e}")
        return False


def upload_logs_to_nas(nas_conn: SMBConnection, logger: logging.Logger, error_logger) -> None:
    """Upload logs to NAS immediately (for critical failures)."""
    try:
        # Save error logs first
        error_logger.save_error_logs(nas_conn)
        
        # Close all logging handlers to ensure file is not in use
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        # Force any remaining buffered log data to be written
        logging.shutdown()

        # Upload the log file
        log_file_path = nas_path_join(
            NAS_BASE_PATH,
            "Outputs",
            "Logs",
            f"stage_3_content_extraction_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
        )

        # Read the entire log file and convert to BytesIO for upload
        with open(logger.temp_log_file, "rb") as log_file:
            log_content = log_file.read()

        log_file_obj = io.BytesIO(log_content)
        nas_upload_file(nas_conn, log_file_obj, log_file_path)
        print(f"Emergency log upload completed to: {log_file_path}")
        
    except Exception as e:
        print(f"Emergency log upload failed: {e}")


def main() -> None:
    """Main function to orchestrate Stage 3 transcript content extraction."""
    global config, logger, error_logger

    logger = setup_logging()
    error_logger = EnhancedErrorLogger()
    print(f"Local log file: {logger.temp_log_file}")  # Always show where local log is
    logger.info("STAGE 3: TRANSCRIPT CONTENT EXTRACTION & PARAGRAPH-LEVEL BREAKDOWN")

    # Connect to NAS and load configuration
    nas_conn = get_nas_connection()
    if not nas_conn:
        logger.error("Failed to connect to NAS - aborting content extraction")
        return

    try:
        # Load shared configuration from NAS
        config = load_stage_config(nas_conn)
        logger.info(f"Loaded configuration for Stage 3")
        logger.info(f"Development mode: {config['stage_3']['dev_mode']}")
        
        if config['stage_3']['dev_mode']:
            logger.info(f"Max files in dev mode: {config['stage_3']['dev_max_files']}")

        start_time = datetime.now()

        # Step 1: Load processing queue from Stage 2
        logger.info("Step 1: Loading processing queue from Stage 2...")
        files_to_process = load_processing_queue(nas_conn)
        
        if not files_to_process:
            logger.warning("No files to process - exiting")
            return
        
        # Apply development mode limit if enabled
        if config['stage_3']['dev_mode']:
            original_count = len(files_to_process)
            files_to_process = files_to_process[:config['stage_3']['dev_max_files']]
            logger.info(f"Development mode: Processing {len(files_to_process)} of {original_count} files")

        # Step 2: Process each file and extract paragraph-level content
        logger.info("Step 2: Processing XML files and extracting content...")
        all_paragraph_records = []
        
        for i, file_record in enumerate(files_to_process, 1):
            filename = file_record.get('filename', 'unknown')
            file_path = file_record.get('file_path', '')
            
            logger.info(f"Processing file {i}/{len(files_to_process)}: {filename}")
            
            # Download XML file from NAS (file_path already contains full path from share root)
            xml_content = nas_download_file(nas_conn, file_path)
            
            if not xml_content:
                error_logger.log_download_error(file_path, "Failed to download XML file")
                continue
            
            # Extract paragraph-level content
            paragraph_records = extract_transcript_paragraphs(file_record, xml_content)
            all_paragraph_records.extend(paragraph_records)
        
        # Step 3: Save extracted content records
        logger.info("Step 3: Saving extracted content records...")
        if not save_extracted_content(nas_conn, all_paragraph_records):
            logger.error("Failed to save extracted content")
            # Upload logs immediately on critical failure
            try:
                upload_logs_to_nas(nas_conn, logger, error_logger)
            except:
                pass
            return

        end_time = datetime.now()
        execution_time = end_time - start_time

        # Save error logs
        error_logger.save_error_logs(nas_conn)

        # Final summary
        logger.info("STAGE 3 CONTENT EXTRACTION COMPLETE")
        logger.info(f"Files processed: {len(files_to_process)}")
        logger.info(f"Total paragraph records created: {len(all_paragraph_records)}")
        logger.info(f"Average paragraphs per file: {len(all_paragraph_records) / len(files_to_process) if files_to_process else 0:.1f}")
        logger.info(f"Execution time: {execution_time}")

        # Upload log file to NAS
        try:
            # Close all logging handlers to ensure file is not in use
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

            # Force any remaining buffered log data to be written
            logging.shutdown()

            # Now safely upload the log file
            log_file_path = nas_path_join(
                NAS_BASE_PATH,
                "Outputs",
                "Logs",
                f"stage_3_content_extraction_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
            )

            # Read the entire log file and convert to BytesIO for upload
            with open(logger.temp_log_file, "rb") as log_file:
                log_content = log_file.read()

            log_file_obj = io.BytesIO(log_content)
            nas_upload_file(nas_conn, log_file_obj, log_file_path)

        except Exception as e:
            print(
                f"Error uploading log file: {e}"
            )  # Can't use logger since it's shut down

    except Exception as e:
        logger.error(f"Stage 3 content extraction failed: {e}")
        # Try to save error logs even if main execution failed
        try:
            error_logger.save_error_logs(nas_conn)
        except:
            pass
    finally:
        if nas_conn:
            nas_conn.close()

        try:
            os.unlink(logger.temp_log_file)
        except (OSError, FileNotFoundError) as e:
            print(f"Cleanup failed for log file: {e}")
        except:
            pass


if __name__ == "__main__":
    main()
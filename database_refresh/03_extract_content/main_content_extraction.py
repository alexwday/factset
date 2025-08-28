"""
Stage 3: Transcript Content Extraction & Paragraph-Level Breakdown
Processes XML transcripts from Stage 2's processing queue and extracts paragraph-level content.
Self-contained standalone script that loads config from NAS at runtime.
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
import xml.etree.ElementTree as ET
import requests

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


def save_logs_to_nas(nas_conn: SMBConnection, stage_summary: Dict[str, Any]):
    """Save execution and error logs to NAS at completion."""
    global execution_log, error_log

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logs_path = config["stage_03_extract_content"]["output_logs_path"]

    # Create logs directory
    nas_create_directory_recursive(nas_conn, logs_path)

    # Save main execution log
    main_log_content = {
        "stage": "stage_03_extract_content_extraction",
        "execution_start": (
            execution_log[0]["timestamp"]
            if execution_log
            else datetime.now().isoformat()
        ),
        "execution_end": datetime.now().isoformat(),
        "summary": stage_summary,
        "execution_log": execution_log,
    }

    main_log_filename = f"stage_03_extract_content_extraction_{timestamp}.json"
    main_log_path = nas_path_join(logs_path, main_log_filename)
    main_log_json = json.dumps(main_log_content, indent=2)
    main_log_obj = io.BytesIO(main_log_json.encode("utf-8"))

    if nas_upload_file(nas_conn, main_log_obj, main_log_path):
        log_console(f"Execution log saved: {main_log_filename}")

    # Save error log only if errors exist
    if error_log:
        errors_path = nas_path_join(logs_path, "Errors")
        nas_create_directory_recursive(nas_conn, errors_path)

        error_log_content = {
            "stage": "stage_03_extract_content_extraction",
            "execution_time": datetime.now().isoformat(),
            "total_errors": len(error_log),
            "error_summary": stage_summary.get("errors", {}),
            "errors": error_log,
        }

        error_log_filename = f"stage_03_extract_content_extraction_errors_{timestamp}.json"
        error_log_path = nas_path_join(errors_path, error_log_filename)
        error_log_json = json.dumps(error_log_content, indent=2)
        error_log_obj = io.BytesIO(error_log_json.encode("utf-8"))

        if nas_upload_file(nas_conn, error_log_obj, error_log_path):
            log_console(f"Error log saved: {error_log_filename}", "WARNING")


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
            },
        )
        raise ValueError(error_msg)

    log_execution(
        "Environment variables validated successfully",
        {
            "total_variables": len(required_env_vars),
            "variables_checked": required_env_vars,
        },
    )


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
            log_execution(
                "NAS connection established successfully",
                {
                    "connection_type": "SMB/CIFS",
                    "port": nas_port,
                    "share_name": os.getenv("NAS_SHARE_NAME"),
                },
            )
            return conn
        else:
            log_error(
                "Failed to establish NAS connection",
                "nas_connection",
                {"server_ip": os.getenv("NAS_SERVER_IP"), "port": nas_port},
            )
            return None

    except Exception as e:
        log_error(
            f"Error creating NAS connection: {e}",
            "nas_connection",
            {"exception_type": type(e).__name__, "server_ip": os.getenv("NAS_SERVER_IP")},
        )
        return None


def load_config_from_nas(nas_conn: SMBConnection) -> Dict[str, Any]:
    """Load and validate configuration from NAS."""

    try:
        log_execution("Loading configuration from NAS", {"config_path": os.getenv("CONFIG_PATH")})

        config_data = nas_download_file(nas_conn, os.getenv("CONFIG_PATH"))
        if not config_data:
            error_msg = f"Configuration file not found at {os.getenv('CONFIG_PATH')}"
            log_error(error_msg, "config_load", {"path": os.getenv("CONFIG_PATH")})
            raise FileNotFoundError(error_msg)

        # Parse YAML configuration
        stage_config = yaml.safe_load(config_data.decode("utf-8"))
        log_execution("Configuration loaded successfully", {"sections": list(stage_config.keys())})

        # Load monitored institutions from separate file
        config_path = os.getenv("CONFIG_PATH")
        config_dir = os.path.dirname(config_path) if config_path else ""
        institutions_path = os.path.join(config_dir, "monitored_institutions.yaml")
        log_execution("Loading monitored institutions from separate file", {"institutions_path": institutions_path})
        
        institutions_data = nas_download_file(nas_conn, institutions_path)
        if institutions_data:
            try:
                stage_config["monitored_institutions"] = yaml.safe_load(institutions_data.decode("utf-8"))
                log_execution("Loaded monitored institutions successfully", {"count": len(stage_config["monitored_institutions"])})
            except yaml.YAMLError as e:
                log_error("Failed to load monitored_institutions.yaml, falling back to config.yaml", "config_parse", {"yaml_error": str(e)})
                # Keep monitored_institutions from main config if separate file fails
        else:
            log_execution("monitored_institutions.yaml not found, using main config file")

        # Validate configuration
        validate_config_structure(stage_config)
        return stage_config

    except yaml.YAMLError as e:
        error_msg = f"Invalid YAML in configuration file: {e}"
        log_error(error_msg, "config_parse", {"yaml_error": str(e)})
        raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"Error loading configuration from NAS: {e}"
        log_error(error_msg, "config_load", {"exception_type": type(e).__name__})
        raise


def validate_config_structure(config: Dict[str, Any]) -> None:
    """Validate configuration structure and required parameters."""

    required_sections = [
        "ssl_cert_path",
        "api_settings", 
        "stage_03_extract_content",
        "monitored_institutions"
    ]

    for section in required_sections:
        if section not in config:
            error_msg = f"Missing required configuration section: {section}"
            log_error(error_msg, "config_validation", {"missing_section": section})
            raise ValueError(error_msg)

    # Validate stage_03_extract_content specific parameters
    stage_03_extract_content_config = config["stage_03_extract_content"]
    required_stage_03_extract_content_params = [
        "description", 
        "input_queue_path",
        "output_logs_path",
        "output_data_path",
        "dev_mode",
        "dev_max_files"
    ]

    for param in required_stage_03_extract_content_params:
        if param not in stage_03_extract_content_config:
            error_msg = f"Missing required stage_03_extract_content parameter: {param}"
            log_error(error_msg, "config_validation", {"missing_parameter": f"stage_03_extract_content.{param}"})
            raise ValueError(error_msg)

    # Validate monitored institutions
    if not config["monitored_institutions"]:
        error_msg = "monitored_institutions cannot be empty"
        log_error(error_msg, "config_validation", {"section": "monitored_institutions"})
        raise ValueError(error_msg)

    # Validate institution structure
    for ticker, institution_info in config["monitored_institutions"].items():
        if not isinstance(institution_info, dict):
            error_msg = f"Invalid institution info for {ticker}: must be a dictionary"
            log_error(error_msg, "config_validation", {"ticker": ticker})
            raise ValueError(error_msg)

        required_fields = ["name", "type", "path_safe_name"]
        for field in required_fields:
            if field not in institution_info:
                error_msg = f"Missing required field '{field}' for institution {ticker}"
                log_error(error_msg, "config_validation", {"ticker": ticker, "missing_field": field})
                raise ValueError(error_msg)

    log_execution("Configuration validation successful", {
        "total_institutions": len(config["monitored_institutions"])
    })


def setup_ssl_certificate(nas_conn: SMBConnection) -> str:
    """Download SSL certificate from NAS to temporary file."""

    try:
        log_execution("Setting up SSL certificate", {"ssl_cert_path": config["ssl_cert_path"]})
        
        cert_data = nas_download_file(nas_conn, config["ssl_cert_path"])
        if not cert_data:
            error_msg = f"SSL certificate not found at {config['ssl_cert_path']}"
            log_error(error_msg, "ssl_setup", {"path": config["ssl_cert_path"]})
            raise FileNotFoundError(error_msg)

        # Create temporary certificate file
        temp_cert_file = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".cer", delete=False
        )
        temp_cert_file.write(cert_data)
        temp_cert_file.close()

        # Set SSL environment variable for requests
        os.environ["REQUESTS_CA_BUNDLE"] = temp_cert_file.name
        os.environ["SSL_CERT_FILE"] = temp_cert_file.name

        log_execution("SSL certificate setup complete", {"temp_cert_path": temp_cert_file.name})
        return temp_cert_file.name

    except Exception as e:
        error_msg = f"Error setting up SSL certificate: {e}"
        log_error(error_msg, "ssl_setup", {"exception_type": type(e).__name__})
        raise


def setup_proxy_configuration() -> str:
    """Set up proxy configuration with proper credential escaping."""

    try:
        proxy_domain = os.getenv("PROXY_DOMAIN", "MAPLE")
        proxy_user = os.getenv("PROXY_USER")
        proxy_password = os.getenv("PROXY_PASSWORD")
        proxy_url = os.getenv("PROXY_URL")

        # Escape domain and user for URL
        escaped_domain = quote(proxy_domain + "\\\\" + proxy_user)
        quoted_password = quote(proxy_password)

        # Construct proxy URL
        proxy_with_auth = f"http://{escaped_domain}:{quoted_password}@{proxy_url}"

        log_execution("Proxy configuration setup complete", {
            "proxy_domain": proxy_domain,
            "proxy_url": proxy_url
        })

        return proxy_with_auth

    except Exception as e:
        error_msg = f"Error setting up proxy configuration: {e}"
        log_error(error_msg, "proxy_setup", {"exception_type": type(e).__name__})
        raise


def nas_path_join(*parts: str) -> str:
    """Join path parts for NAS paths using forward slashes."""
    clean_parts = []
    for part in parts:
        if part:
            clean_part = str(part).strip("/")
            if clean_part:
                clean_parts.append(clean_part)
    return "/".join(clean_parts)


def nas_download_file(conn: SMBConnection, nas_file_path: str) -> Optional[bytes]:
    """Download a file from NAS and return as bytes."""

    if not validate_nas_path(nas_file_path):
        log_error(f"Invalid NAS file path: {nas_file_path}", "path_validation", 
                 {"path": nas_file_path})
        return None

    try:
        file_obj = io.BytesIO()
        conn.retrieveFile(os.getenv("NAS_SHARE_NAME"), nas_file_path, file_obj)
        file_obj.seek(0)
        return file_obj.read()
    except Exception as e:
        log_error(f"Failed to download file from NAS: {nas_file_path}", "nas_download", 
                 {"path": nas_file_path, "error": str(e)})
        return None


def nas_upload_file(conn: SMBConnection, local_file_obj: io.BytesIO, nas_file_path: str) -> bool:
    """Upload a file object to NAS."""

    if not validate_nas_path(nas_file_path):
        log_error(f"Invalid NAS file path: {nas_file_path}", "path_validation", 
                 {"path": nas_file_path})
        return False

    try:
        # Ensure parent directory exists
        parent_dir = "/".join(nas_file_path.split("/")[:-1])
        if parent_dir:
            nas_create_directory_recursive(conn, parent_dir)

        conn.storeFile(os.getenv("NAS_SHARE_NAME"), nas_file_path, local_file_obj)
        return True
    except Exception as e:
        log_error(f"Failed to upload file to NAS: {nas_file_path}", "nas_upload", 
                 {"path": nas_file_path, "error": str(e)})
        return False


def nas_file_exists(conn: SMBConnection, file_path: str) -> bool:
    """Check if a file exists on the NAS."""
    try:
        conn.getAttributes(os.getenv("NAS_SHARE_NAME"), file_path)
        return True
    except Exception:
        return False


def nas_create_directory_recursive(conn: SMBConnection, dir_path: str) -> bool:
    """Create directory on NAS with safe iterative parent creation."""

    normalized_path = dir_path.strip("/").rstrip("/")
    if not normalized_path:
        log_error("Cannot create directory with empty path", "directory_creation", {"path": dir_path})
        return False

    path_parts = [part for part in normalized_path.split("/") if part]
    if not path_parts:
        log_error("Cannot create directory with invalid path", "directory_creation", {"path": dir_path})
        return False

    current_path = ""
    for part in path_parts:
        current_path = f"{current_path}/{part}" if current_path else part

        if nas_file_exists(conn, current_path):
            continue

        try:
            conn.createDirectory(os.getenv("NAS_SHARE_NAME"), current_path)
        except Exception as e:
            if not nas_file_exists(conn, current_path):
                log_error(f"Failed to create directory: {current_path}", "directory_creation", 
                         {"path": current_path, "error": str(e)})
                return False

    return True


def validate_file_path(path: str) -> bool:
    """Validate file path for security."""
    if not path or not isinstance(path, str):
        return False

    # Check for directory traversal attacks
    if ".." in path or path.startswith("/"):
        return False

    # Check for invalid characters
    invalid_chars = ["<", ">", ":", '"', "|", "?", "*", "\\x00"]
    if any(char in path for char in invalid_chars):
        return False

    # Check path length
    if len(path) > 260:  # Windows MAX_PATH limitation
        return False

    return True


def validate_nas_path(path: str) -> bool:
    """Validate NAS path structure for security."""
    
    if not path or not isinstance(path, str):
        return False

    normalized = path.strip("/")
    if not normalized:
        return False
        
    parts = normalized.split("/")

    for part in parts:
        if not part or part in [".", ".."]:
            return False
        if not validate_file_path(part):
            return False

    return True


def sanitize_url_for_logging(url: str) -> str:
    """Remove auth tokens from URLs before logging."""
    try:
        sanitized = re.sub(r"(password|token|auth)=[^&]*", r"\\1=***", url, flags=re.IGNORECASE)
        sanitized = re.sub(r"://[^@]*@", "://***:***@", sanitized)
        return sanitized
    except Exception:
        return "[URL_SANITIZED]"


def extract_metadata_from_file_path(file_path: str) -> Dict[str, Any]:
    """Extract metadata from file path structure and filename."""
    
    try:
        # Parse path: Data/YYYY/QX/Type/Company/filename.xml
        path_parts = file_path.split("/")
        
        # Extract from path structure
        fiscal_year = None
        fiscal_quarter = None
        institution_type = None
        
        # Find Data folder and extract following parts
        for i, part in enumerate(path_parts):
            if part == "Data" and i + 3 < len(path_parts):
                fiscal_year = path_parts[i + 1]
                fiscal_quarter = path_parts[i + 2]  
                institution_type = path_parts[i + 3]
                break
        
        # Extract filename
        filename = path_parts[-1] if path_parts else ""
        
        # Parse filename format: ticker_quarter_year_type_eventid_versionid.xml
        # Example: RY-CA_Q1_2024_Corrected_12345_1.xml
        filename_parts = filename.replace(".xml", "").split("_")
        
        ticker = None
        transcript_type = None
        event_id = None
        version_id = None
        
        if len(filename_parts) >= 6:
            ticker = filename_parts[0]
            # quarter = filename_parts[1]  # Already extracted from path
            # year = filename_parts[2]     # Already extracted from path  
            transcript_type = filename_parts[3]
            event_id = filename_parts[4]
            version_id = filename_parts[5]
        
        # Lookup company name from config
        company_name = None
        if ticker and ticker in config.get("monitored_institutions", {}):
            company_name = config["monitored_institutions"][ticker].get("name")
        
        metadata = {
            "filename": filename,
            "fiscal_year": fiscal_year,
            "fiscal_quarter": fiscal_quarter,
            "institution_type": institution_type,
            "ticker": ticker,
            "company_name": company_name,
            "transcript_type": transcript_type,
            "event_id": event_id,
            "version_id": version_id
        }
        
        log_execution(f"Extracted metadata from file path: {file_path}", metadata)
        return metadata
        
    except Exception as e:
        log_error(f"Failed to extract metadata from file path: {file_path}", "metadata_extraction", 
                 {"path": file_path, "error": str(e)})
        return {}


def load_processing_queue(nas_conn: SMBConnection) -> List[Dict[str, Any]]:
    """Load Stage 2 processing queue from NAS."""
    
    try:
        queue_path = config["stage_03_extract_content"]["input_queue_path"]
        log_execution("Loading processing queue from NAS", {"queue_path": queue_path})
        
        queue_data = nas_download_file(nas_conn, queue_path)
        if not queue_data:
            error_msg = f"Processing queue not found at {queue_path}"
            log_error(error_msg, "queue_load", {"path": queue_path})
            raise FileNotFoundError(error_msg)
        
        queue_records = json.loads(queue_data.decode("utf-8"))
        log_execution("Processing queue loaded successfully", {
            "total_files": len(queue_records)
        })
        
        return queue_records
        
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in processing queue: {e}"
        log_error(error_msg, "queue_parse", {"json_error": str(e)})
        raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"Error loading processing queue: {e}"
        log_error(error_msg, "queue_load", {"exception_type": type(e).__name__})
        raise


def parse_transcript_xml(xml_content: bytes) -> Optional[Dict[str, Any]]:
    """Parse transcript XML and extract structured data."""
    
    try:
        # Parse XML with namespace handling
        root = ET.fromstring(xml_content)
        
        # Detect namespace if present
        namespace = ""
        if root.tag.startswith("{"):
            namespace = root.tag.split("}")[0] + "}"
        
        # Helper function to find elements with namespace
        def find_element(parent, tag):
            return parent.find(f"{namespace}{tag}")
        
        def find_all_elements(parent, tag):
            return parent.findall(f"{namespace}{tag}")
        
        # Extract metadata
        meta = find_element(root, "meta")
        if meta is None:
            log_error("XML meta section not found", "xml_parsing", {"xml_size": len(xml_content)})
            return None
        
        # Extract title
        title_elem = find_element(meta, "title")
        title = title_elem.text if title_elem is not None else ""
        
        # Extract participants and create mapping (using original logic)
        participants = {}
        participants_elem = find_element(meta, "participants")
        if participants_elem is not None:
            for participant in find_all_elements(participants_elem, "participant"):
                p_id = participant.get("id")  # Use 'id' not 'pid' like original
                if p_id:
                    participants[p_id] = {
                        "type": participant.get("type", ""),
                        "affiliation": participant.get("affiliation", ""),
                        "affiliation_entity": participant.get("affiliation_entity", ""),
                        "title": participant.get("title", ""),
                        "entity": participant.get("entity", ""),
                        "name": participant.text.strip().replace('"', '\\"').replace('\n', ' ').replace('\r', ' ').replace('\t', ' ') if participant.text else "Unknown Speaker"
                    }
        
        # Extract body content
        body = find_element(root, "body")
        if body is None:
            log_error("XML body section not found", "xml_parsing", {"xml_size": len(xml_content)})
            return None
        
        # Extract sections
        sections = []
        for section in find_all_elements(body, "section"):
            section_name = section.get("name", "")
            section_content = []
            
            for speaker in find_all_elements(section, "speaker"):
                speaker_id = speaker.get("id", "")
                speaker_type = speaker.get("type", "")
                
                # Extract paragraphs
                plist = find_element(speaker, "plist")
                paragraphs = []
                if plist is not None:
                    for p in find_all_elements(plist, "p"):
                        if p.text:
                            # Clean text to prevent JSON breaking characters
                            clean_text = p.text.strip().replace('"', '\\"').replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                            paragraphs.append(clean_text)
                
                if paragraphs:  # Only include speakers with content
                    section_content.append({
                        "speaker_id": speaker_id,
                        "speaker_type": speaker_type,
                        "paragraphs": paragraphs
                    })
            
            if section_content:  # Only include sections with content
                sections.append({
                    "name": section_name,
                    "speakers": section_content
                })
        
        return {
            "title": title,
            "participants": participants,
            "sections": sections
        }
        
    except ET.ParseError as e:
        log_error(f"XML parsing error: {e}", "xml_parsing", {"error": str(e)})
        return None
    except Exception as e:
        log_error(f"Unexpected error parsing XML: {e}", "xml_parsing", {"error": str(e)})
        return None


def format_speaker_string(participant_data: Dict[str, Any]) -> str:
    """Format speaker information into clean readable string (using original logic)."""
    
    name = participant_data.get("name", "").strip()
    title = participant_data.get("title", "").strip() 
    affiliation = participant_data.get("affiliation", "").strip()
    
    # Use original logic - prioritize name, then add title and affiliation
    if not name or name == "Unknown Speaker":
        return "Unknown Speaker"
    
    # Build speaker string starting with name
    parts = [name]
    
    # Add title if available and different from name
    if title and title != name:
        parts.append(title)
        
    # Add affiliation if available and different from previous parts
    if affiliation and affiliation not in parts:
        parts.append(affiliation)
    
    return ", ".join(parts)


def determine_qa_flag(speaker_type: str) -> Optional[str]:
    """Determine Q&A flag from XML speaker type attribute."""
    
    if speaker_type == "q":
        return "question"
    elif speaker_type == "a": 
        return "answer"
    else:
        return None


def extract_transcript_paragraphs(base_record: Dict[str, Any], xml_content: bytes) -> List[Dict[str, Any]]:
    """Extract paragraph-level content from XML transcript."""
    
    # Parse XML content
    parsed_data = parse_transcript_xml(xml_content)
    if not parsed_data:
        log_error("Failed to parse XML content", "content_extraction", {"file_path": base_record.get("file_path", "")})
        return []
    
    # Extract title for base record
    title = parsed_data.get("title", "")
    participants = parsed_data.get("participants", {})
    sections = parsed_data.get("sections", [])
    
    # Create enhanced base record with extracted metadata and title
    enhanced_base_record = {**base_record, "title": title}
    
    paragraph_records = []
    global_paragraph_id = 1
    current_speaker_block_id = 1
    
    # Process each section
    for section_id, section in enumerate(sections, 1):
        section_name = section.get("name", f"Section {section_id}")
        
        # Process each speaker block in section
        for speaker_block in section.get("speakers", []):
            speaker_id = speaker_block.get("speaker_id", "")
            speaker_type = speaker_block.get("speaker_type", "")
            paragraphs = speaker_block.get("paragraphs", [])
            
            # Get speaker information
            participant_data = participants.get(speaker_id, {})
            speaker_string = format_speaker_string(participant_data)
            qa_flag = determine_qa_flag(speaker_type)
            
            # Process each paragraph in speaker block
            for paragraph_text in paragraphs:
                if paragraph_text.strip():  # Only process non-empty paragraphs
                    paragraph_record = {
                        **enhanced_base_record,  # All base fields including title
                        "section_id": section_id,
                        "section_name": section_name,
                        "paragraph_id": global_paragraph_id,
                        "speaker_block_id": current_speaker_block_id,
                        "question_answer_flag": qa_flag,
                        "speaker": speaker_string,
                        "paragraph_content": paragraph_text.strip().replace('"', '\\"').replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                    }
                    
                    paragraph_records.append(paragraph_record)
                    global_paragraph_id += 1
            
            # Increment speaker block ID after processing all paragraphs in block
            if paragraphs:
                current_speaker_block_id += 1
    
    log_execution(f"Extracted {len(paragraph_records)} paragraphs from transcript", {
        "file_path": base_record.get("file_path", ""),
        "total_paragraphs": len(paragraph_records),
        "total_sections": len(sections),
        "title": title
    })
    
    return paragraph_records


def process_transcript_file(nas_conn: SMBConnection, file_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process a single transcript file and return paragraph records."""
    
    file_path = file_record.get("file_path", "")
    date_last_modified = file_record.get("date_last_modified", "")
    institution_id = file_record.get("institution_id")
    ticker = file_record.get("ticker")
    
    try:
        log_execution(f"Processing transcript file: {file_path}")
        
        # Extract metadata from file path
        metadata = extract_metadata_from_file_path(file_path)
        
        # Create enhanced base record combining Stage 2 data and extracted metadata
        base_record = {
            "file_path": file_path,
            "date_last_modified": date_last_modified,
            "institution_id": institution_id,
            "ticker": ticker if ticker else metadata.get("ticker"),  # Use provided ticker or extracted one
            **metadata  # Add all extracted metadata fields
        }
        
        # Download XML content from NAS
        xml_content = nas_download_file(nas_conn, file_path)
        if not xml_content:
            log_error(f"Failed to download XML content: {file_path}", "xml_download", {"path": file_path})
            return []
        
        # Extract paragraph-level content
        paragraph_records = extract_transcript_paragraphs(base_record, xml_content)
        
        log_execution(f"Successfully processed transcript file: {file_path}", {
            "paragraphs_extracted": len(paragraph_records)
        })
        
        return paragraph_records
        
    except Exception as e:
        log_error(f"Error processing transcript file: {file_path}", "content_extraction", {
            "path": file_path,
            "error": str(e),
            "exception_type": type(e).__name__
        })
        return []


def validate_and_preview_json(all_paragraph_records: List[Dict[str, Any]]) -> bool:
    """Validate JSON structure and preview first few records."""
    
    try:
        log_console("🔍 Validating JSON structure...")
        
        # Test JSON serialization
        test_json = json.dumps(all_paragraph_records, indent=2)
        
        # Test JSON deserialization  
        test_data = json.loads(test_json)
        
        if len(test_data) != len(all_paragraph_records):
            log_error("JSON validation failed: Record count mismatch", "json_validation", {
                "original_count": len(all_paragraph_records),
                "parsed_count": len(test_data)
            })
            return False
        
        log_console(f"✅ JSON validation passed: {len(test_data)} records")
        
        # Preview first 3 records
        preview_count = min(3, len(test_data))
        log_console(f"📋 Previewing first {preview_count} records:")
        
        for i in range(preview_count):
            record = test_data[i]
            log_console(f"\n--- Record {i+1} ---")
            log_console(f"File: {record.get('filename', 'N/A')}")
            log_console(f"Ticker: {record.get('ticker', 'N/A')}")
            log_console(f"Company: {record.get('company_name', 'N/A')}")
            log_console(f"Transcript Type: {record.get('transcript_type', 'N/A')}")
            log_console(f"Event ID: {record.get('event_id', 'N/A')}")
            log_console(f"Version ID: {record.get('version_id', 'N/A')}")
            log_console(f"Title: {record.get('title', 'N/A')[:50]}...")
            log_console(f"Speaker: {record.get('speaker', 'N/A')}")
            log_console(f"Section: {record.get('section_name', 'N/A')}")
            log_console(f"Q&A Flag: {record.get('question_answer_flag', 'N/A')}")
            log_console(f"Content: {record.get('paragraph_content', 'N/A')[:100]}...")
        
        # Show field statistics
        sample_record = test_data[0] if test_data else {}
        log_console(f"\n📊 Record structure ({len(sample_record)} fields):")
        for key in sorted(sample_record.keys()):
            log_console(f"  - {key}")
        
        return True
        
    except json.JSONDecodeError as e:
        log_error(f"JSON encoding/decoding error: {e}", "json_validation", {
            "error": str(e),
            "error_position": getattr(e, 'pos', None)
        })
        log_console(f"❌ JSON validation failed: {e}", "ERROR")
        return False
        
    except Exception as e:
        log_error(f"JSON validation error: {e}", "json_validation", {
            "error": str(e),
            "exception_type": type(e).__name__
        })
        log_console(f"❌ JSON validation failed: {e}", "ERROR")
        return False


def save_extracted_content(nas_conn: SMBConnection, all_paragraph_records: List[Dict[str, Any]]) -> bool:
    """Save extracted paragraph records to NAS."""
    
    try:
        # First validate and preview the JSON
        if not validate_and_preview_json(all_paragraph_records):
            log_console("JSON validation failed - not saving file", "ERROR")
            return False
        
        output_path = config["stage_03_extract_content"]["output_data_path"]
        output_filename = "stage_03_extracted_content.json"
        output_file_path = nas_path_join(output_path, output_filename)
        
        # Create output directory
        nas_create_directory_recursive(nas_conn, output_path)
        
        # Convert records to JSON
        output_content = json.dumps(all_paragraph_records, indent=2)
        output_file_obj = io.BytesIO(output_content.encode("utf-8"))
        
        # Upload to NAS
        if nas_upload_file(nas_conn, output_file_obj, output_file_path):
            log_execution("Extracted content saved successfully", {
                "output_path": output_file_path,
                "total_records": len(all_paragraph_records)
            })
            log_console(f"✅ File saved successfully: {output_filename}")
            return True
        else:
            log_error("Failed to save extracted content", "output_save", {"path": output_file_path})
            return False
            
    except Exception as e:
        log_error(f"Error saving extracted content: {e}", "output_save", {
            "error": str(e),
            "exception_type": type(e).__name__
        })
        return False


def main() -> None:
    """Main function to orchestrate Stage 3 content extraction."""
    global config, logger

    # Initialize logging
    logger = setup_logging()
    log_console("=== STAGE 3: TRANSCRIPT CONTENT EXTRACTION & PARAGRAPH-LEVEL BREAKDOWN ===")

    # Initialize stage summary
    stage_summary = {
        "status": "unknown",
        "execution_time_seconds": 0,
        "total_files_queued": 0,
        "files_processed": 0,
        "paragraphs_extracted": 0,
        "errors": {
            "environment_validation": 0,
            "nas_connection": 0,
            "config_load": 0,
            "ssl_setup": 0,
            "queue_load": 0,
            "metadata_extraction": 0,
            "xml_processing": 0,
            "content_extraction": 0
        }
    }

    start_time = datetime.now()
    ssl_cert_path = None
    nas_conn = None

    try:
        # Step 1: Environment validation
        log_console("Step 1: Validating environment variables...")
        validate_environment_variables()

        # Step 2: NAS connection
        log_console("Step 2: Connecting to NAS...")
        nas_conn = get_nas_connection()
        if not nas_conn:
            stage_summary["status"] = "failed"
            log_console("Failed to establish NAS connection", "ERROR")
            return

        # Step 3: Configuration loading
        log_console("Step 3: Loading configuration...")
        config = load_config_from_nas(nas_conn)
        log_console(f"Loaded configuration for {len(config['monitored_institutions'])} institutions")

        # Step 4: SSL certificate setup
        log_console("Step 4: Setting up SSL certificate...")
        ssl_cert_path = setup_ssl_certificate(nas_conn)

        # Step 5: Proxy configuration
        log_console("Step 5: Setting up proxy configuration...")
        proxy_url = setup_proxy_configuration()

        # Step 6: Load processing queue
        log_console("Step 6: Loading processing queue from Stage 2...")
        processing_queue = load_processing_queue(nas_conn)
        stage_summary["total_files_queued"] = len(processing_queue)

        if not processing_queue:
            log_console("No files found in processing queue", "WARNING")
            stage_summary["status"] = "completed_no_files"
            return

        # Step 7: Development mode handling
        dev_mode = config["stage_03_extract_content"].get("dev_mode", False)
        if dev_mode:
            max_files = config["stage_03_extract_content"].get("dev_max_files", 2)
            processing_queue = processing_queue[:max_files]
            log_console(f"Development mode: Processing only {len(processing_queue)} files", "WARNING")

        # Step 8: Process transcript files
        log_console("Step 8: Processing transcript files...")
        log_console(f"Total files to process: {len(processing_queue)}")
        
        all_paragraph_records = []
        
        # Track file processing statistics
        file_stats = {
            "total_files": len(processing_queue),
            "files_by_type": {},
            "files_by_ticker": {},
            "files_by_year": {},
            "files_with_versions": {}
        }
        
        for i, record in enumerate(processing_queue):
            file_path = record.get("file_path", "")
            log_console(f"Processing file {i+1}/{len(processing_queue)}: {file_path}")
            
            # Process single transcript file
            paragraph_records = process_transcript_file(nas_conn, record)
            
            if paragraph_records:
                all_paragraph_records.extend(paragraph_records)
                stage_summary["paragraphs_extracted"] += len(paragraph_records)
                stage_summary["files_processed"] += 1
                log_console(f"Extracted {len(paragraph_records)} paragraphs from {file_path}")
                
                # Track statistics from extracted records
                if paragraph_records:
                    first_record = paragraph_records[0]
                    
                    # Track by transcript type
                    trans_type = first_record.get("transcript_type", "unknown")
                    file_stats["files_by_type"][trans_type] = file_stats["files_by_type"].get(trans_type, 0) + 1
                    
                    # Track by ticker
                    ticker = first_record.get("ticker", "unknown")
                    file_stats["files_by_ticker"][ticker] = file_stats["files_by_ticker"].get(ticker, 0) + 1
                    
                    # Track by year
                    year = first_record.get("fiscal_year", "unknown")
                    file_stats["files_by_year"][year] = file_stats["files_by_year"].get(year, 0) + 1
                    
                    # Track version info
                    version = first_record.get("version_id", "unknown")
                    event_id = first_record.get("event_id", "unknown")
                    version_key = f"{ticker}_{year}_{first_record.get('fiscal_quarter', 'unknown')}"
                    if version_key not in file_stats["files_with_versions"]:
                        file_stats["files_with_versions"][version_key] = []
                    file_stats["files_with_versions"][version_key].append({
                        "filename": first_record.get("filename", "unknown"),
                        "version": version,
                        "event_id": event_id,
                        "type": trans_type
                    })
            else:
                log_console(f"No paragraphs extracted from {file_path}", "WARNING")

        # Step 9: Save extracted content
        log_console("Step 9: Saving extracted content...")
        if all_paragraph_records:
            if not save_extracted_content(nas_conn, all_paragraph_records):
                stage_summary["status"] = "failed"
                log_console("Failed to save extracted content", "ERROR")
                return
        else:
            log_console("No content extracted - nothing to save", "WARNING")

        # Calculate execution time
        end_time = datetime.now()
        execution_time = end_time - start_time
        stage_summary["execution_time_seconds"] = execution_time.total_seconds()
        stage_summary["status"] = "completed_successfully"

        # Count errors by type
        for error_entry in error_log:
            error_type = error_entry.get("error_type", "unknown")
            if error_type in stage_summary["errors"]:
                stage_summary["errors"][error_type] += 1

        # Final summary
        log_console("=== STAGE 3 CONTENT EXTRACTION COMPLETE ===")
        log_console(f"Files processed: {stage_summary['files_processed']}")
        log_console(f"Paragraphs extracted: {stage_summary['paragraphs_extracted']}")
        log_console(f"Total errors: {sum(stage_summary['errors'].values())}")
        log_console(f"Execution time: {execution_time}")
        
        # Add file processing statistics
        if 'file_stats' in locals() and file_stats:
            log_console("\n📊 FILE PROCESSING STATISTICS:")
            log_console(f"  Total files in queue: {file_stats['total_files']}")
            log_console(f"  Successfully processed: {stage_summary['files_processed']}")
            
            # Show files by type
            if file_stats["files_by_type"]:
                log_console("\n  Files by transcript type:")
                for trans_type, count in sorted(file_stats["files_by_type"].items()):
                    log_console(f"    {trans_type}: {count}")
            
            # Count unique transcripts (ticker_year_quarter combinations)
            unique_transcripts = len(file_stats["files_with_versions"])
            if unique_transcripts > 0:
                avg_versions = file_stats["total_files"] / unique_transcripts
                log_console(f"\n  📈 Unique transcripts (ticker_year_quarter): {unique_transcripts}")
                log_console(f"  📈 Average versions per transcript: {avg_versions:.2f}")
                
                # Find transcripts with multiple versions
                multi_version = {k: v for k, v in file_stats["files_with_versions"].items() if len(v) > 1}
                if multi_version:
                    log_console(f"  📈 Transcripts with multiple versions: {len(multi_version)}")
            
            # Show ticker distribution
            if file_stats["files_by_ticker"]:
                log_console(f"\n  Unique tickers processed: {len(file_stats['files_by_ticker'])}")
            
            # Add to stage summary
            stage_summary["file_statistics"] = {
                "total_files": file_stats["total_files"],
                "unique_transcripts": unique_transcripts,
                "files_by_type": file_stats["files_by_type"],
                "unique_tickers": len(file_stats["files_by_ticker"]),
                "multi_version_transcripts": len(multi_version) if 'multi_version' in locals() else 0
            }

    except Exception as e:
        stage_summary["status"] = "failed"
        error_msg = f"Stage 3 content extraction failed: {e}"
        log_console(error_msg, "ERROR")
        log_error(error_msg, "main_execution", {"exception_type": type(e).__name__})

    finally:
        # Save logs to NAS
        if nas_conn:
            try:
                save_logs_to_nas(nas_conn, stage_summary)
            except Exception as e:
                log_console(f"Failed to save logs to NAS: {e}", "WARNING")

        # Cleanup
        if ssl_cert_path:
            try:
                os.unlink(ssl_cert_path)
                log_execution("SSL certificate cleanup completed")
            except Exception as e:
                log_console(f"SSL certificate cleanup failed: {e}", "WARNING")

        if nas_conn:
            try:
                nas_conn.close()
                log_execution("NAS connection closed")
            except Exception as e:
                log_console(f"Error closing NAS connection: {e}", "WARNING")

        log_console(f"Stage 3 content extraction {stage_summary['status']}")


if __name__ == "__main__":
    main()
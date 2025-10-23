"""
Stage 2: Transcript Consolidation & Master Database Synchronization
Synchronizes master database with NAS file system - no selection logic, just file sync.
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
    logs_path = config["stage_02_database_sync"]["output_logs_path"]

    # Create logs directory
    nas_create_directory_recursive(nas_conn, logs_path)

    # Save main execution log
    main_log_content = {
        "stage": "stage_02_database_sync_consolidation",
        "execution_start": (
            execution_log[0]["timestamp"]
            if execution_log
            else datetime.now().isoformat()
        ),
        "execution_end": datetime.now().isoformat(),
        "summary": stage_summary,
        "execution_log": execution_log,
    }

    main_log_filename = f"stage_02_database_sync_consolidation_{timestamp}.json"
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
            "stage": "stage_02_database_sync_consolidation",
            "execution_time": datetime.now().isoformat(),
            "total_errors": len(error_log),
            "error_summary": stage_summary.get("errors", {}),
            "errors": error_log,
        }

        error_log_filename = f"stage_02_database_sync_consolidation_errors_{timestamp}.json"
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
        "stage_02_database_sync",
        "monitored_institutions"
    ]

    for section in required_sections:
        if section not in config:
            error_msg = f"Missing required configuration section: {section}"
            log_error(error_msg, "config_validation", {"missing_section": section})
            raise ValueError(error_msg)

    # Validate stage_02_database_sync specific parameters
    stage_02_database_sync_config = config["stage_02_database_sync"]
    required_stage_02_database_sync_params = [
        "description", 
        "input_data_path",
        "output_logs_path",
        "master_database_path",
        "refresh_output_path"
    ]

    for param in required_stage_02_database_sync_params:
        if param not in stage_02_database_sync_config:
            error_msg = f"Missing required stage_02_database_sync parameter: {param}"
            log_error(error_msg, "config_validation", {"missing_parameter": f"stage_02_database_sync.{param}"})
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
        escaped_domain = quote(proxy_domain + "\\" + proxy_user)
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


def nas_list_files(conn: SMBConnection, directory_path: str) -> List[str]:
    """List XML files in a NAS directory."""
    try:
        files = conn.listPath(os.getenv("NAS_SHARE_NAME"), directory_path)
        return [
            file_info.filename
            for file_info in files
            if not file_info.isDirectory and file_info.filename.endswith(".xml")
        ]
    except Exception as e:
        log_error(f"Failed to list files in directory: {directory_path}", "nas_list_files", 
                 {"directory": directory_path, "error": str(e)})
        return []


def nas_list_directories(conn: SMBConnection, directory_path: str) -> List[str]:
    """List subdirectories in a NAS directory."""
    try:
        files = conn.listPath(os.getenv("NAS_SHARE_NAME"), directory_path)
        return [
            file_info.filename
            for file_info in files
            if file_info.isDirectory and file_info.filename not in [".", ".."]
        ]
    except Exception as e:
        log_error(f"Failed to list directories: {directory_path}", "nas_list_directories", 
                 {"directory": directory_path, "error": str(e)})
        return []


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


def get_file_modified_time(conn: SMBConnection, file_path: str) -> Optional[datetime]:
    """Get the last modified time of a file on NAS."""
    try:
        attrs = conn.getAttributes(os.getenv("NAS_SHARE_NAME"), file_path)
        return datetime.fromtimestamp(attrs.last_write_time)
    except Exception as e:
        log_error(f"Failed to get file modified time: {file_path}", "file_metadata", 
                 {"path": file_path, "error": str(e)})
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
        sanitized = re.sub(r"(password|token|auth)=[^&]*", r"\1=***", url, flags=re.IGNORECASE)
        sanitized = re.sub(r"://[^@]*@", "://***:***@", sanitized)
        return sanitized
    except Exception:
        return "[URL_SANITIZED]"


def parse_quarter_and_year_from_xml(
    xml_content: bytes,
) -> Tuple[Optional[str], Optional[str]]:
    """Parse quarter and fiscal year from transcript XML title.
    
    Returns (quarter, year) if title matches 'Qx 20xx Earnings Call' format,
    or (None, None) if it doesn't match.
    """
    try:
        # Parse only until we find the title
        root = ET.parse(io.BytesIO(xml_content)).getroot()
        namespace = ""
        if root.tag.startswith("{"):
            namespace = root.tag.split("}")[0] + "}"

        meta = root.find(f"{namespace}meta" if namespace else "meta")
        if meta is None:
            return None, None

        title_elem = meta.find(f"{namespace}title" if namespace else "title")
        if title_elem is None or not title_elem.text:
            return None, None

        title = title_elem.text.strip()

        # Only accept exact format: "Qx 20xx Earnings Call"
        pattern = r"^Q([1-4])\s+(20\d{2})\s+Earnings\s+Call$"
        match = re.search(pattern, title, re.IGNORECASE)
        if match:
            quarter = f"Q{match.group(1)}"
            year = match.group(2)
            return quarter, year

        # Title doesn't match required format - log for debugging
        log_execution(
            f"Title validation failed - not an earnings call",
            {
                "actual_title": title,
                "required_format": "Qx 20xx Earnings Call",
                "validation_rule": "Must match exact pattern with no additional text",
            },
        )
        return None, None

    except Exception as e:
        log_error(
            f"Error parsing XML title: {e}", "xml_parsing", {"error_details": str(e)}
        )
        return None, None


def scan_nas_for_all_transcripts(nas_conn: SMBConnection) -> Dict[str, Dict[str, Any]]:
    """Scan NAS for ALL transcript files. Structure: Data/YYYY/QX/Type/Company/files.xml
    
    Returns a dictionary with file_path as key and file_record as value.
    Each file_record now includes an 'is_earnings_call' flag based on title validation.
    """
    
    log_execution("Starting comprehensive NAS file scan with title validation")
    data_base_path = config["stage_02_database_sync"]["input_data_path"]
    nas_inventory = {}
    earnings_call_count = 0
    non_earnings_count = 0
    
    # Scan all years
    years = nas_list_directories(nas_conn, data_base_path)
    log_execution(f"Found fiscal years: {sorted(years)}", {"years_found": len(years)})

    for year in sorted(years):
        if year == "Unknown":
            continue

        year_path = nas_path_join(data_base_path, year)
        quarters = nas_list_directories(nas_conn, year_path)

        for quarter in sorted(quarters):
            if quarter == "Unknown":
                continue

            quarter_path = nas_path_join(year_path, quarter)
            institution_types = nas_list_directories(nas_conn, quarter_path)

            for institution_type in institution_types:
                type_path = nas_path_join(quarter_path, institution_type)
                companies = nas_list_directories(nas_conn, type_path)

                for company in companies:
                    company_path = nas_path_join(type_path, company)
                    files = nas_list_files(nas_conn, company_path)

                    # Process all XML files directly in company folder
                    for filename in files:
                        file_path = nas_path_join(company_path, filename)
                        modified_time = get_file_modified_time(nas_conn, file_path)
                        
                        # Check if this is an earnings call by parsing the XML title
                        is_earnings_call = False
                        try:
                            # Download the file to check its title
                            file_content = nas_download_file(nas_conn, file_path)
                            if file_content:
                                quarter_parsed, year_parsed = parse_quarter_and_year_from_xml(file_content)
                                if quarter_parsed and year_parsed:
                                    is_earnings_call = True
                                    earnings_call_count += 1
                                else:
                                    non_earnings_count += 1
                        except Exception as e:
                            log_execution(f"Could not validate title for {file_path}: {e}")
                            non_earnings_count += 1
                        
                        # Extract ticker from filename to get institution_id
                        institution_id = None
                        ticker = None
                        if filename.endswith('.xml'):
                            # Extract ticker from filename pattern: ticker_quarter_year_type_eventid_versionid.xml
                            parts = filename.replace('.xml', '').split('_')
                            if parts:
                                ticker = parts[0]
                                # Look up institution ID from config
                                if ticker in config.get("monitored_institutions", {}):
                                    institution_id = config["monitored_institutions"][ticker].get("id")
                        
                        # Create file record with earnings call flag and institution_id
                        # FIX #5: Add filename for consistency across stages
                        file_record = {
                            "file_path": file_path,
                            "filename": filename,  # Add filename for downstream consistency
                            "date_last_modified": modified_time.isoformat() if modified_time else datetime.now().isoformat(),
                            "is_earnings_call": is_earnings_call,
                            "institution_id": institution_id,
                            "ticker": ticker
                        }
                        
                        # Use file_path as key for direct comparison
                        nas_inventory[file_path] = file_record

    log_execution(f"NAS file scan complete with title validation", {
        "total_files_found": len(nas_inventory),
        "earnings_calls": earnings_call_count,
        "non_earnings_transcripts": non_earnings_count,
        "earnings_call_percentage": f"{(earnings_call_count / len(nas_inventory) * 100):.1f}%" if nas_inventory else "0%"
    })
    return nas_inventory


def load_master_database(nas_conn: SMBConnection) -> Optional[Dict[str, Any]]:
    """Load existing master database (CSV format) or return None if it doesn't exist."""

    db_path = config["stage_02_database_sync"]["master_database_path"]

    if nas_file_exists(nas_conn, db_path):
        log_execution("Loading existing master database", {"database_path": db_path})
        try:
            db_data = nas_download_file(nas_conn, db_path)
            if db_data:
                # Parse CSV format (matching Stage 8 output format)
                import csv
                import io

                csv_content = io.StringIO(db_data.decode("utf-8"))
                reader = csv.DictReader(csv_content)

                # Extract unique files with their modification dates
                files_dict = {}
                record_count = 0

                for row in reader:
                    record_count += 1
                    # Stage 8 uses 'filename' field (just the filename, not full path)
                    filename = row.get("filename", "")

                    # Fallback: Extract filename from file_path if filename is empty (for old database format)
                    if not filename:
                        file_path = row.get("file_path", "")
                        if file_path:
                            # Extract just the filename from the path
                            filename = file_path.split('/')[-1] if '/' in file_path else file_path

                    date_modified = row.get("date_last_modified", "")

                    # Store only the latest modification date for each file
                    # Note: We store by filename only since that's what Stage 8 provides
                    if filename and (filename not in files_dict or date_modified > files_dict[filename].get("date_last_modified", "")):
                        files_dict[filename] = {
                            "filename": filename,
                            "date_last_modified": date_modified
                        }

                # Convert to expected format
                database = {
                    "files": list(files_dict.values())
                }

                log_execution("Master database loaded successfully",
                             {"total_records": record_count, "unique_files": len(database["files"])})
                return database

        except Exception as e:
            log_error(f"Error loading database: {e}", "database_load",
                     {"database_path": db_path, "error": str(e)})

    # No existing database found
    log_execution("No existing master database found - all NAS files will be marked for processing")
    return None


def database_to_comparison_format(database: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Convert database to comparison format (filename -> record)."""
    comparison_data = {}

    if database is not None:
        for record in database.get("files", []):
            # Use filename as key since that's what the master CSV contains
            filename = record.get("filename", record.get("file_path", ""))
            if filename:
                # Extract just the filename if it's a full path
                if '/' in filename:
                    filename = filename.split('/')[-1]
                comparison_data[filename] = record

    return comparison_data


def extract_transcript_key_and_version(file_path_or_name: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Extract base transcript key and version info from filename.

    Returns:
        Tuple of (base_key, version_info) where:
        - base_key: ticker_quarter_year (e.g., 'JPM_Q1_2024')
        - version_info: dict with version details
    """
    # Get just the filename
    filename = file_path_or_name.split('/')[-1] if '/' in file_path_or_name else file_path_or_name

    # Parse: TICKER_QUARTER_YEAR_TYPE_EVENTID_VERSIONID.xml
    parts = filename.replace('.xml', '').split('_')

    if len(parts) >= 6:
        ticker = parts[0]
        quarter = parts[1]
        year = parts[2]
        transcript_type = parts[3]
        event_id = parts[4]
        version_id = parts[5]

        base_key = f"{ticker}_{quarter}_{year}"

        # Version priority for comparison
        type_priority = {
            "Corrected": 4,
            "Final": 3,
            "Script": 2,
            "Raw": 1
        }

        version_info = {
            'version_id': int(version_id) if version_id.isdigit() else 0,
            'type': transcript_type,
            'priority': type_priority.get(transcript_type, 0),
            'event_id': event_id,
            'filename': filename
        }

        return base_key, version_info

    return None, None


def organize_by_transcript(inventory: Dict[str, Dict[str, Any]], is_nas: bool = True) -> Dict[str, List[Tuple[str, Dict[str, Any], Dict[str, Any]]]]:
    """Organize inventory by base transcript key.

    Args:
        inventory: Either NAS inventory (file_path -> record) or database inventory (filename -> record)
        is_nas: True if NAS inventory, False if database inventory

    Returns:
        Dict of base_key -> list of (identifier, record, version_info)
    """
    organized = {}

    for identifier, record in inventory.items():
        # Skip non-earnings calls if this is NAS inventory
        if is_nas and not record.get("is_earnings_call", False):
            continue

        base_key, version_info = extract_transcript_key_and_version(identifier)

        # FIX #1: Handle malformed filenames
        if not base_key or not version_info:
            log_execution(f"Skipping malformed filename: {identifier}")
            continue

        if base_key not in organized:
            organized[base_key] = []
        organized[base_key].append((identifier, record, version_info))

    return organized


def select_best_from_versions(versions: List[Tuple[str, Dict[str, Any], Dict[str, Any]]]) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Select best version from a list of versions.

    Args:
        versions: List of (identifier, record, version_info) tuples

    Returns:
        Best (identifier, record, version_info) tuple
    """
    if not versions:
        return None, None, None

    # FIX #2: Warn if multiple events detected (unusual situation)
    event_ids = set(v[2]['event_id'] for v in versions)
    if len(event_ids) > 1:
        log_console(f"⚠️ WARNING: Multiple event IDs {list(event_ids)} found for same quarter. Selecting best version across all events.", "WARNING")
        log_execution("Multiple events detected for single quarter", {
            "event_ids": list(event_ids),
            "files": [v[2]['filename'] for v in versions],
            "action": "Selecting single best version across all events"
        })

    # Sort by priority (type), then version_id, then modification date
    sorted_versions = sorted(versions,
                           key=lambda x: (x[2]['priority'],
                                        x[2]['version_id'],
                                        x[1].get('date_last_modified', '')),
                           reverse=True)

    return sorted_versions[0]


def detect_changes_version_aware(nas_inventory: Dict[str, Dict[str, Any]], database_inventory: Dict[str, Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """Version-aware change detection between NAS and database inventories.

    This function groups files by base transcript (ticker_quarter_year) and handles
    version management properly, ensuring old versions are removed when better versions exist.
    """

    files_to_process = []
    files_to_remove = []

    # If no existing database, all NAS files go to processing
    if not database_inventory:
        # Group by transcript and select best version for each
        nas_transcripts = organize_by_transcript(nas_inventory, is_nas=True)
        for base_key, versions in nas_transcripts.items():
            best_path, best_record, _ = select_best_from_versions(versions)
            if best_path:
                files_to_process.append(best_path)

        log_execution("No existing database - processing best version of each transcript",
                     {"unique_transcripts": len(nas_transcripts),
                      "files_to_process": len(files_to_process)})
        return files_to_process, files_to_remove

    # Organize both inventories by base transcript
    nas_transcripts = organize_by_transcript(nas_inventory, is_nas=True)
    db_transcripts = organize_by_transcript(database_inventory, is_nas=False)

    # Get all unique base keys
    all_base_keys = set(nas_transcripts.keys()) | set(db_transcripts.keys())

    log_execution(f"Version-aware comparison starting", {
        "nas_transcripts": len(nas_transcripts),
        "db_transcripts": len(db_transcripts),
        "total_unique_transcripts": len(all_base_keys)
    })

    # Process each transcript
    for base_key in all_base_keys:
        nas_versions = nas_transcripts.get(base_key, [])
        db_versions = db_transcripts.get(base_key, [])

        if not nas_versions:
            # Transcript deleted from NAS - remove all versions from database
            for filename, _, _ in db_versions:
                files_to_remove.append(filename)
            log_execution(f"Transcript deleted from NAS: {base_key}",
                         {"versions_to_remove": len(db_versions)})

        elif not db_versions:
            # New transcript - process best version only
            best_path, best_record, best_version = select_best_from_versions(nas_versions)
            if best_path:
                files_to_process.append(best_path)
                log_execution(f"New transcript: {base_key}",
                             {"selected_version": best_version['filename'],
                              "available_versions": len(nas_versions)})

        else:
            # Transcript exists in both - compare best versions
            best_nas_path, best_nas_record, best_nas_version = select_best_from_versions(nas_versions)
            best_db_name, best_db_record, best_db_version = select_best_from_versions(db_versions)

            needs_update = False
            update_reason = ""

            # Check if we need to update
            if best_nas_version['filename'] != best_db_version['filename']:
                # Different version
                needs_update = True
                update_reason = f"Version change: {best_db_version['filename']} -> {best_nas_version['filename']}"
            elif best_nas_record['date_last_modified'] != best_db_record.get('date_last_modified', ''):
                # Same version, different date
                needs_update = True
                update_reason = f"Date change for {best_nas_version['filename']}"

            if needs_update:
                # Process best NAS version
                files_to_process.append(best_nas_path)

                # Remove ALL database versions of this transcript
                for filename, _, _ in db_versions:
                    files_to_remove.append(filename)

                log_execution(f"Transcript update: {base_key}", {
                    "reason": update_reason,
                    "processing": best_nas_version['filename'],
                    "removing": [v[2]['filename'] for v in db_versions]
                })

    log_execution("Version-aware delta detection complete", {
        "files_to_process": len(files_to_process),
        "files_to_remove": len(files_to_remove),
        "transcripts_analyzed": len(all_base_keys)
    })

    return files_to_process, files_to_remove


def detect_changes(nas_inventory: Dict[str, Dict[str, Any]], database_inventory: Dict[str, Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """Detect changes between NAS and database inventories.

    Note: NAS inventory uses full paths as keys, database inventory uses filenames only.
    THIS IS THE OLD VERSION - KEPT FOR BACKWARDS COMPATIBILITY.
    USE detect_changes_version_aware() INSTEAD.
    """

    files_to_process = []
    files_to_remove = []

    # If no existing database, all NAS files go to processing
    if not database_inventory:
        files_to_process = list(nas_inventory.keys())
        log_execution("No existing database - all NAS files marked for processing",
                     {"total_files": len(files_to_process)})
        return files_to_process, files_to_remove

    # Check for new and modified files
    for file_path, nas_record in nas_inventory.items():
        # Extract filename from full path for comparison with database
        filename = file_path.split('/')[-1] if '/' in file_path else file_path
        db_record = database_inventory.get(filename)

        if not db_record:
            # New file
            files_to_process.append(file_path)
            log_execution(f"New file found: {file_path}")
        else:
            # Check for date modifications
            nas_modified = nas_record["date_last_modified"]
            db_modified = db_record.get("date_last_modified", "")

            if nas_modified != db_modified:
                # Modified file - add to process and mark old version for removal
                files_to_process.append(file_path)
                files_to_remove.append(file_path)  # Full path for modified files
                log_execution(f"Modified file found: {file_path}")

    # Check for deleted files (in database but not on NAS)
    # Build a set of filenames currently on NAS
    nas_filenames = set()
    for file_path in nas_inventory.keys():
        filename = file_path.split('/')[-1] if '/' in file_path else file_path
        nas_filenames.add(filename)

    # Check each database entry
    for filename in database_inventory.keys():
        if filename not in nas_filenames:
            # File no longer exists on NAS
            # We only have the filename, not the full path (since it's deleted)
            files_to_remove.append(filename)  # Just filename for deleted files
            log_execution(f"Deleted file found: {filename}")

    log_execution("Delta detection complete", {
        "files_to_process": len(files_to_process),
        "files_to_remove": len(files_to_remove)
    })

    return files_to_process, files_to_remove


def select_best_version(versions: List[Tuple[str, Dict[str, Any]]]) -> Tuple[str, Dict[str, Any]]:
    """Select the best version from multiple transcript versions.
    
    Selection rules (in priority order):
    1. Highest version_id (newest version)
    2. Transcript type priority: Corrected > Final > Script > Raw
    3. Most recent modification date
    
    Args:
        versions: List of (file_path, nas_record) tuples for the same base transcript
        
    Returns:
        Tuple of (selected_file_path, selected_nas_record)
    """
    
    # Define transcript type priority (higher number = higher priority)
    type_priority = {
        "Corrected": 4,
        "Final": 3,
        "Script": 2,
        "Raw": 1
    }
    
    def get_version_score(version_tuple):
        file_path, nas_record = version_tuple
        
        # Extract version info from filename
        # Format: TICKER_QUARTER_YEAR_TYPE_EVENTID_VERSIONID.xml
        filename = file_path.split('/')[-1] if '/' in file_path else file_path
        parts = filename.replace('.xml', '').split('_')
        
        # Default values if parsing fails
        version_id = 0
        transcript_type = "Raw"
        
        if len(parts) >= 6:
            try:
                transcript_type = parts[3]
                version_id = int(parts[5]) if parts[5].isdigit() else 0
            except (IndexError, ValueError):
                pass
        
        # Calculate score: version_id * 10 + type_priority
        # This ensures version_id is the primary factor
        type_score = type_priority.get(transcript_type, 0)
        score = (version_id * 10) + type_score
        
        # Use modification date as tiebreaker
        mod_date = nas_record.get("date_last_modified", "")
        
        return (score, mod_date)
    
    # Sort versions by score (highest first)
    sorted_versions = sorted(versions, key=get_version_score, reverse=True)
    
    # Log version selection details
    if len(versions) > 1:
        selected_path = sorted_versions[0][0]
        log_execution(f"Selected best version from {len(versions)} options", {
            "selected": selected_path,
            "all_versions": [v[0].split('/')[-1] for v in versions],
            "selection_reason": "Highest version_id and type priority"
        })
    
    return sorted_versions[0]


def save_processing_queues_version_aware(nas_conn: SMBConnection, files_to_process: List[str], files_to_remove: List[str], nas_inventory: Dict[str, Dict[str, Any]]) -> bool:
    """Save processing queues to NAS refresh folder - version-aware edition.

    Since version selection is already done in detect_changes_version_aware,
    this function just formats and saves the queues.
    """

    refresh_path = config["stage_02_database_sync"]["refresh_output_path"]
    nas_create_directory_recursive(nas_conn, refresh_path)

    success = True

    # Create process records from files_to_process
    # Version selection already done, just filter non-earnings calls
    process_records = []
    earnings_calls_count = 0
    non_earnings_filtered = 0

    for file_path in files_to_process:
        nas_record = nas_inventory.get(file_path, {})

        # Skip non-earnings calls
        if not nas_record.get("is_earnings_call", False):
            non_earnings_filtered += 1
            continue

        # Add to processing queue
        # FIX #5: Include filename for consistency
        record = {
            "file_path": file_path,
            "filename": nas_record.get("filename", file_path.split('/')[-1] if '/' in file_path else file_path),
            "date_last_modified": nas_record.get("date_last_modified", datetime.now().isoformat()),
            "institution_id": nas_record.get("institution_id"),
            "ticker": nas_record.get("ticker")
        }
        process_records.append(record)
        earnings_calls_count += 1

    log_execution(f"Processing queue preparation complete", {
        "total_files_to_process": len(files_to_process),
        "earnings_calls_included": earnings_calls_count,
        "non_earnings_filtered_out": non_earnings_filtered
    })

    # Save files to process as simple JSON records
    process_path = nas_path_join(refresh_path, "stage_02_process_queue.json")
    process_content = json.dumps(process_records, indent=2)
    process_file_obj = io.BytesIO(process_content.encode("utf-8"))

    if not nas_upload_file(nas_conn, process_file_obj, process_path):
        log_error("Failed to save stage_02_process_queue.json", "queue_save", {"path": process_path})
        success = False
    else:
        log_console(f"✅ Process queue saved: {earnings_calls_count} files")

    # Convert files_to_remove to JSON records
    remove_records = []
    for file_identifier in files_to_remove:
        # file_identifier could be full path or just filename from database
        # FIX #5: Ensure we have both file_path and filename for consistency
        if '/' in file_identifier:
            # It's a full path
            filename = file_identifier.split('/')[-1]
            file_path = file_identifier
        else:
            # It's just a filename (from database)
            filename = file_identifier
            file_path = file_identifier  # Stage 9 will handle filename-only

        record = {
            "file_path": file_path,
            "filename": filename,  # Always include filename for consistency
            "date_last_modified": datetime.now().isoformat()
        }
        remove_records.append(record)

    # Save files to remove as simple JSON records
    remove_path = nas_path_join(refresh_path, "stage_02_removal_queue.json")
    remove_content = json.dumps(remove_records, indent=2)
    remove_file_obj = io.BytesIO(remove_content.encode("utf-8"))

    if not nas_upload_file(nas_conn, remove_file_obj, remove_path):
        log_error("Failed to save stage_02_removal_queue.json", "queue_save", {"path": remove_path})
        success = False
    else:
        log_console(f"✅ Removal queue saved: {len(remove_records)} files")

    if success:
        log_execution("Processing queues saved successfully", {
            "files_to_process": earnings_calls_count,
            "files_to_remove": len(files_to_remove)
        })

    return success


def save_processing_queues(nas_conn: SMBConnection, files_to_process: List[str], files_to_remove: List[str], nas_inventory: Dict[str, Dict[str, Any]], database_inventory: Dict[str, Dict[str, Any]]) -> bool:
    """Save processing queues to NAS refresh folder as simple JSON records.

    IMPORTANT: Only earnings call transcripts are included in the processing queue.
    Non-earnings transcripts are stored in NAS but not processed by downstream stages.
    THIS IS THE OLD VERSION - KEPT FOR BACKWARDS COMPATIBILITY.
    USE save_processing_queues_version_aware() WITH detect_changes_version_aware() INSTEAD.
    """

    refresh_path = config["stage_02_database_sync"]["refresh_output_path"]
    nas_create_directory_recursive(nas_conn, refresh_path)

    success = True

    # Group files by base transcript (ticker_quarter_year) for version selection
    transcript_groups = {}
    duplicate_count = 0  # For backwards compatibility, though we now handle this differently

    for file_path in files_to_process:
        nas_record = nas_inventory.get(file_path, {})

        # Skip non-earnings calls early
        if not nas_record.get("is_earnings_call", False):
            continue

        # Extract base transcript key from filename
        # Format: TICKER_QUARTER_YEAR_TYPE_EVENTID_VERSIONID.xml
        filename = file_path.split('/')[-1] if '/' in file_path else file_path
        parts = filename.replace('.xml', '').split('_')

        if len(parts) >= 6:
            ticker = parts[0]
            quarter = parts[1]
            year = parts[2]
            base_key = f"{ticker}_{quarter}_{year}"

            if base_key not in transcript_groups:
                transcript_groups[base_key] = []
            transcript_groups[base_key].append((file_path, nas_record))

    # Select best version for each transcript group
    process_records = []
    earnings_calls_count = 0
    non_earnings_filtered = len(files_to_process) - sum(len(v) for v in transcript_groups.values())
    version_selection_count = 0
    versions_dropped_count = 0

    log_console(f"📊 Version selection: {len(transcript_groups)} unique transcripts from {len(files_to_process)} files")

    # Track examples for logging
    version_selection_examples = []

    for base_key, versions in transcript_groups.items():
        if len(versions) > 1:
            # Multiple versions - select the best one
            selected_path, selected_record = select_best_version(versions)
            version_selection_count += 1
            versions_dropped_count += len(versions) - 1

            # Store example for logging (limit to first 5)
            if len(version_selection_examples) < 5:
                version_selection_examples.append({
                    "base_key": base_key,
                    "kept": selected_path.split('/')[-1],
                    "dropped": [path.split('/')[-1] for path, _ in versions if path != selected_path]
                })
        else:
            # Single version - use it
            selected_path, selected_record = versions[0]

        # Add selected version to processing queue
        record = {
            "file_path": selected_path,
            "date_last_modified": selected_record.get("date_last_modified", datetime.now().isoformat()),
            "institution_id": selected_record.get("institution_id"),
            "ticker": selected_record.get("ticker")
        }
        process_records.append(record)
        earnings_calls_count += 1
    
    log_execution(f"Version selection and filtering complete", {
        "total_files_to_process": len(files_to_process),
        "unique_transcripts": len(transcript_groups),
        "version_selection_performed": version_selection_count,
        "versions_dropped": versions_dropped_count,
        "earnings_calls_included": earnings_calls_count,
        "non_earnings_filtered_out": non_earnings_filtered
    })
    
    if version_selection_count > 0:
        log_console(f"✅ Version selection: Selected best version for {version_selection_count} transcripts, dropped {versions_dropped_count} duplicate versions")
        
        # Show examples of version selection
        if version_selection_examples:
            log_console("\n📋 Examples of version selection:")
            for i, example in enumerate(version_selection_examples, 1):
                log_console(f"  {i}. {example['base_key']}:")
                log_console(f"     ✅ Kept: {example['kept']}")
                for dropped_file in example['dropped']:
                    log_console(f"     ❌ Dropped: {dropped_file}")
            
            if version_selection_count > len(version_selection_examples):
                log_console(f"  ... and {version_selection_count - len(version_selection_examples)} more transcripts with version selection")

    # Save files to process as simple JSON records
    process_path = nas_path_join(refresh_path, "stage_02_process_queue.json")
    process_content = json.dumps(process_records, indent=2)
    process_file_obj = io.BytesIO(process_content.encode("utf-8"))

    if not nas_upload_file(nas_conn, process_file_obj, process_path):
        log_error("Failed to save stage_02_process_queue.json", "queue_save", {"path": process_path})
        success = False

    # Convert files_to_remove to JSON records
    remove_records = []
    for file_path in files_to_remove:
        # Get the database record for this file to include date_last_modified
        # Note: file_path could be full path (modified files) or just filename (deleted files)
        # Database inventory uses filename as key
        lookup_key = file_path.split('/')[-1] if '/' in file_path else file_path
        db_record = database_inventory.get(lookup_key, {})

        record = {
            "file_path": file_path,  # Keep original (could be full path or filename)
            "date_last_modified": db_record.get("date_last_modified", datetime.now().isoformat())
        }
        remove_records.append(record)

    # Save files to remove as simple JSON records
    remove_path = nas_path_join(refresh_path, "stage_02_removal_queue.json")
    remove_content = json.dumps(remove_records, indent=2)
    remove_file_obj = io.BytesIO(remove_content.encode("utf-8"))

    if not nas_upload_file(nas_conn, remove_file_obj, remove_path):
        log_error("Failed to save stage_02_removal_queue.json", "queue_save", {"path": remove_path})
        success = False

    if success:
        log_execution("Processing queues saved successfully", {
            "files_to_process": len(files_to_process),
            "files_to_remove": len(files_to_remove)
        })

    return success


def main() -> None:
    """Main function to orchestrate Stage 2 transcript consolidation."""
    global config, logger

    # Initialize logging
    logger = setup_logging()
    log_console("=== STAGE 2: TRANSCRIPT CONSOLIDATION & MASTER DATABASE SYNCHRONIZATION ===")

    # Initialize stage summary
    stage_summary = {
        "status": "unknown",
        "execution_time_seconds": 0,
        "total_nas_files": 0,
        "total_database_files": 0,
        "files_to_process": 0,
        "files_to_remove": 0,
        "errors": {
            "environment_validation": 0,
            "nas_connection": 0,
            "config_load": 0,
            "ssl_setup": 0,
            "nas_download": 0,
            "nas_upload": 0,
            "database_load": 0,
            "queue_save": 0,
            "path_validation": 0
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

        # Step 6: NAS file scanning
        log_console("Step 6: Scanning NAS for all transcript files...")
        nas_inventory = scan_nas_for_all_transcripts(nas_conn)
        stage_summary["total_nas_files"] = len(nas_inventory)

        if not nas_inventory:
            log_console("No transcript files found on NAS", "WARNING")
            stage_summary["status"] = "completed_no_files"
            return

        # Step 7: Load master database
        log_console("Step 7: Loading master database...")
        master_database = load_master_database(nas_conn)
        database_inventory = database_to_comparison_format(master_database)
        stage_summary["total_database_files"] = len(database_inventory) if database_inventory else 0

        # Step 8: Delta detection (VERSION-AWARE)
        log_console("Step 8: Detecting changes between NAS and database (version-aware)...")
        # Use the new version-aware detection
        files_to_process, files_to_remove = detect_changes_version_aware(nas_inventory, database_inventory)

        stage_summary["files_to_process"] = len(files_to_process)
        stage_summary["files_to_remove"] = len(files_to_remove)

        # Step 9: Save processing queues (VERSION-AWARE)
        log_console("Step 9: Saving processing queues (version-aware)...")
        # Use the new version-aware save function (no need for database_inventory now)
        if not save_processing_queues_version_aware(nas_conn, files_to_process, files_to_remove, nas_inventory):
            stage_summary["status"] = "failed"
            log_console("Failed to save processing queues", "ERROR")
            return

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
        log_console("=== STAGE 2 CONSOLIDATION COMPLETE ===")
        log_console(f"Total NAS files: {stage_summary['total_nas_files']}")
        log_console(f"Total database files: {stage_summary['total_database_files']}")
        log_console(f"Files to process: {stage_summary['files_to_process']}")
        log_console(f"Files to remove: {stage_summary['files_to_remove']}")
        log_console(f"Total errors: {sum(stage_summary['errors'].values())}")
        log_console(f"Execution time: {execution_time}")

    except Exception as e:
        stage_summary["status"] = "failed"
        error_msg = f"Stage 2 consolidation failed: {e}"
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

        log_console(f"Stage 2 consolidation {stage_summary['status']}")


if __name__ == "__main__":
    main()
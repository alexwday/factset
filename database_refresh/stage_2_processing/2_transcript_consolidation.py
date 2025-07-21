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
    logs_path = config["stage_2"]["output_logs_path"]

    # Create logs directory
    nas_create_directory_recursive(nas_conn, logs_path)

    # Save main execution log
    main_log_content = {
        "stage": "stage_2_transcript_consolidation",
        "execution_start": (
            execution_log[0]["timestamp"]
            if execution_log
            else datetime.now().isoformat()
        ),
        "execution_end": datetime.now().isoformat(),
        "summary": stage_summary,
        "execution_log": execution_log,
    }

    main_log_filename = f"stage_2_transcript_consolidation_{timestamp}.json"
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
            "stage": "stage_2_transcript_consolidation",
            "execution_time": datetime.now().isoformat(),
            "total_errors": len(error_log),
            "error_summary": stage_summary.get("errors", {}),
            "errors": error_log,
        }

        error_log_filename = f"stage_2_transcript_consolidation_errors_{timestamp}.json"
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
        "stage_2",
        "monitored_institutions"
    ]

    for section in required_sections:
        if section not in config:
            error_msg = f"Missing required configuration section: {section}"
            log_error(error_msg, "config_validation", {"missing_section": section})
            raise ValueError(error_msg)

    # Validate stage_2 specific parameters
    stage_2_config = config["stage_2"]
    required_stage_2_params = [
        "description", 
        "output_data_path",
        "output_logs_path",
        "master_database_path",
        "refresh_output_path"
    ]

    for param in required_stage_2_params:
        if param not in stage_2_config:
            error_msg = f"Missing required stage_2 parameter: {param}"
            log_error(error_msg, "config_validation", {"missing_parameter": f"stage_2.{param}"})
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


def scan_nas_for_all_transcripts(nas_conn: SMBConnection) -> Dict[str, Dict[str, Any]]:
    """Scan NAS for ALL transcript files. Structure: Data/YYYY/QX/Type/Company/files.xml"""
    
    log_execution("Starting comprehensive NAS file scan")
    data_base_path = config["stage_2"]["output_data_path"]
    nas_inventory = {}
    
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
                        
                        # Create file record with minimal required fields
                        file_record = {
                            "file_path": file_path,
                            "date_last_modified": modified_time.isoformat() if modified_time else datetime.now().isoformat()
                        }
                        
                        # Use file_path as key for direct comparison
                        nas_inventory[file_path] = file_record

    log_execution(f"NAS file scan complete", {"total_files_found": len(nas_inventory)})
    return nas_inventory


def load_master_database(nas_conn: SMBConnection) -> Optional[Dict[str, Any]]:
    """Load existing master database or return None if it doesn't exist."""
    
    db_path = config["stage_2"]["master_database_path"]
    
    if nas_file_exists(nas_conn, db_path):
        log_execution("Loading existing master database", {"database_path": db_path})
        try:
            db_data = nas_download_file(nas_conn, db_path)
            if db_data:
                database = json.loads(db_data.decode("utf-8"))
                log_execution("Master database loaded successfully", 
                             {"records_count": len(database.get("files", []))})
                return database
        except json.JSONDecodeError as e:
            log_error(f"Database corruption detected: {e}", "database_load", 
                     {"database_path": db_path, "error": str(e)})
        except Exception as e:
            log_error(f"Error loading database: {e}", "database_load", 
                     {"database_path": db_path, "error": str(e)})

    # No existing database found
    log_execution("No existing master database found - all NAS files will be marked for processing")
    return None


def database_to_comparison_format(database: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Convert database to comparison format (file_path -> record)."""
    comparison_data = {}
    
    if database is not None:
        for record in database.get("files", []):
            file_path = record.get("file_path", "")
            if file_path:
                comparison_data[file_path] = record
    
    return comparison_data


def detect_changes(nas_inventory: Dict[str, Dict[str, Any]], database_inventory: Dict[str, Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """Detect changes between NAS and database inventories."""
    
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
        db_record = database_inventory.get(file_path)

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
                files_to_remove.append(file_path)  # Remove old database entry
                log_execution(f"Modified file found: {file_path}")

    # Check for deleted files (in database but not on NAS)
    for file_path in database_inventory.keys():
        if file_path not in nas_inventory:
            files_to_remove.append(file_path)
            log_execution(f"Deleted file found: {file_path}")

    log_execution("Delta detection complete", {
        "files_to_process": len(files_to_process),
        "files_to_remove": len(files_to_remove)
    })
    
    return files_to_process, files_to_remove


def save_processing_queues(nas_conn: SMBConnection, files_to_process: List[str], files_to_remove: List[str]) -> bool:
    """Save processing queues to NAS refresh folder."""
    
    refresh_path = config["stage_2"]["refresh_output_path"]
    nas_create_directory_recursive(nas_conn, refresh_path)

    timestamp = datetime.now().isoformat()
    success = True

    # Save files to process
    process_data = {
        "timestamp": timestamp,
        "total_files": len(files_to_process),
        "files": files_to_process
    }

    process_path = nas_path_join(refresh_path, "files_to_process.json")
    process_content = json.dumps(process_data, indent=2)
    process_file_obj = io.BytesIO(process_content.encode("utf-8"))

    if not nas_upload_file(nas_conn, process_file_obj, process_path):
        log_error("Failed to save files_to_process.json", "queue_save", {"path": process_path})
        success = False

    # Save files to remove
    remove_data = {
        "timestamp": timestamp,
        "total_files": len(files_to_remove),
        "files": files_to_remove
    }

    remove_path = nas_path_join(refresh_path, "files_to_remove.json")
    remove_content = json.dumps(remove_data, indent=2)
    remove_file_obj = io.BytesIO(remove_content.encode("utf-8"))

    if not nas_upload_file(nas_conn, remove_file_obj, remove_path):
        log_error("Failed to save files_to_remove.json", "queue_save", {"path": remove_path})
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

        # Step 8: Delta detection
        log_console("Step 8: Detecting changes between NAS and database...")
        files_to_process, files_to_remove = detect_changes(nas_inventory, database_inventory)
        
        stage_summary["files_to_process"] = len(files_to_process)
        stage_summary["files_to_remove"] = len(files_to_remove)

        # Step 9: Save processing queues
        log_console("Step 9: Saving processing queues...")
        if not save_processing_queues(nas_conn, files_to_process, files_to_remove):
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
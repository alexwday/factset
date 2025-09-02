"""
Stage 4: Transcript Content Validation
Validates that extracted transcript content has exactly 2 sections: MANAGEMENT DISCUSSION SECTION and Q&A.
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
from collections import defaultdict
import io
import re
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
    logs_path = config["stage_04_validate_structure"]["output_logs_path"]

    # Create logs directory
    nas_create_directory_recursive(nas_conn, logs_path)

    # Save main execution log
    main_log_content = {
        "stage": "stage_04_validate_structure_validation",
        "execution_start": (
            execution_log[0]["timestamp"]
            if execution_log
            else datetime.now().isoformat()
        ),
        "execution_end": datetime.now().isoformat(),
        "summary": stage_summary,
        "execution_log": execution_log,
    }

    main_log_filename = f"stage_04_validate_structure_validation_{timestamp}.json"
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
            "stage": "stage_04_validate_structure_validation",
            "execution_time": datetime.now().isoformat(),
            "total_errors": len(error_log),
            "error_summary": stage_summary.get("errors", {}),
            "errors": error_log,
        }

        error_log_filename = f"stage_04_validate_structure_validation_errors_{timestamp}.json"
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
        "stage_04_validate_structure",
        "monitored_institutions"
    ]

    for section in required_sections:
        if section not in config:
            error_msg = f"Missing required configuration section: {section}"
            log_error(error_msg, "config_validation", {"missing_section": section})
            raise ValueError(error_msg)

    # Validate stage_04_validate_structure specific parameters
    stage_04_validate_structure_config = config["stage_04_validate_structure"]
    required_stage_04_validate_structure_params = [
        "description", 
        "input_data_path",
        "output_logs_path",
        "output_data_path",
        "dev_mode",
        "dev_max_files",
        "expected_sections"
    ]

    for param in required_stage_04_validate_structure_params:
        if param not in stage_04_validate_structure_config:
            error_msg = f"Missing required stage_04_validate_structure parameter: {param}"
            log_error(error_msg, "config_validation", {"missing_parameter": f"stage_04_validate_structure.{param}"})
            raise ValueError(error_msg)

    # Validate expected sections
    expected_sections = stage_04_validate_structure_config.get("expected_sections", [])
    if not expected_sections or len(expected_sections) != 2:
        error_msg = "stage_04_validate_structure.expected_sections must contain exactly 2 section names"
        log_error(error_msg, "config_validation", {"section": "stage_04_validate_structure.expected_sections"})
        raise ValueError(error_msg)

    # Validate monitored institutions
    if not config["monitored_institutions"]:
        error_msg = "monitored_institutions cannot be empty"
        log_error(error_msg, "config_validation", {"section": "monitored_institutions"})
        raise ValueError(error_msg)

    log_execution("Configuration validation successful", {
        "total_institutions": len(config["monitored_institutions"]),
        "expected_sections": expected_sections
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


def load_extracted_content(nas_conn: SMBConnection) -> Dict[str, Any]:
    """Load Stage 3 extracted content from NAS."""
    
    try:
        input_path = config["stage_04_validate_structure"]["input_data_path"]
        log_execution("Loading extracted content from NAS", {"input_path": input_path})
        
        content_data = nas_download_file(nas_conn, input_path)
        if not content_data:
            error_msg = f"Extracted content file not found at {input_path}"
            log_error(error_msg, "content_load", {"path": input_path})
            raise FileNotFoundError(error_msg)
        
        content_records = json.loads(content_data.decode("utf-8"))
        log_execution("Extracted content loaded successfully", {
            "total_records": len(content_records) if isinstance(content_records, list) else content_records.get("total_records", "unknown")
        })
        
        return content_records
        
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in extracted content file: {e}"
        log_error(error_msg, "content_parse", {"json_error": str(e)})
        raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"Error loading extracted content: {e}"
        log_error(error_msg, "content_load", {"exception_type": type(e).__name__})
        raise


def validate_transcript_structure(transcript_records: List[Dict[str, Any]], transcript_key: str) -> Dict[str, Any]:
    """Validate that a transcript has exactly 2 sections with expected names."""
    
    try:
        log_execution(f"Validating transcript structure: {transcript_key}")
        
        # Get expected section names from config
        expected_sections = config["stage_04_validate_structure"]["expected_sections"]
        
        # Group records by section
        sections_found = {}
        for record in transcript_records:
            section_name = record.get("section_name", "").strip()
            if section_name:
                if section_name not in sections_found:
                    sections_found[section_name] = []
                sections_found[section_name].append(record)
        
        # Validation results
        validation_result = {
            "transcript_key": transcript_key,
            "is_valid": True,
            "validation_errors": [],
            "sections_found": list(sections_found.keys()),
            "expected_sections": expected_sections,
            "section_counts": {section: len(records) for section, records in sections_found.items()},
            "total_records": len(transcript_records)
        }
        
        # Check exact number of sections
        if len(sections_found) != 2:
            validation_result["is_valid"] = False
            validation_result["validation_errors"].append(
                f"Expected exactly 2 sections, found {len(sections_found)}: {list(sections_found.keys())}"
            )
        
        # Check section names match expected
        for expected_section in expected_sections:
            if expected_section not in sections_found:
                validation_result["is_valid"] = False
                validation_result["validation_errors"].append(
                    f"Missing expected section: '{expected_section}'"
                )
        
        # Check for unexpected sections
        for found_section in sections_found.keys():
            if found_section not in expected_sections:
                validation_result["is_valid"] = False
                validation_result["validation_errors"].append(
                    f"Unexpected section found: '{found_section}'"
                )
        
        # Log validation results
        if validation_result["is_valid"]:
            log_execution(f"Transcript validation PASSED: {transcript_key}", validation_result)
        else:
            log_error(f"Transcript validation FAILED: {transcript_key}", "validation_failure", validation_result)
        
        return validation_result
        
    except Exception as e:
        error_msg = f"Error validating transcript structure for {transcript_key}: {e}"
        log_error(error_msg, "validation_error", {
            "transcript_key": transcript_key,
            "exception_type": type(e).__name__
        })
        return {
            "transcript_key": transcript_key,
            "is_valid": False,
            "validation_errors": [f"Validation error: {e}"],
            "sections_found": [],
            "expected_sections": expected_sections,
            "section_counts": {},
            "total_records": len(transcript_records) if transcript_records else 0
        }


def process_validation_results(nas_conn: SMBConnection, all_validation_results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process validation results and separate valid from invalid transcripts."""
    
    try:
        valid_transcripts = []
        invalid_transcripts = []
        
        for result in all_validation_results:
            if result["is_valid"]:
                valid_transcripts.append(result)
            else:
                invalid_transcripts.append(result)
        
        log_execution("Validation results processed", {
            "total_transcripts": len(all_validation_results),
            "valid_transcripts": len(valid_transcripts),
            "invalid_transcripts": len(invalid_transcripts)
        })
        
        return valid_transcripts, invalid_transcripts
        
    except Exception as e:
        error_msg = f"Error processing validation results: {e}"
        log_error(error_msg, "result_processing", {"exception_type": type(e).__name__})
        return [], all_validation_results


def save_validation_outputs(nas_conn: SMBConnection, valid_transcripts: List[Dict[str, Any]], invalid_transcripts: List[Dict[str, Any]], original_records: List[Dict[str, Any]]) -> Tuple[bool, bool]:
    """Save validation outputs to NAS."""
    
    try:
        output_path = config["stage_04_validate_structure"]["output_data_path"]
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Create output directory
        nas_create_directory_recursive(nas_conn, output_path)
        
        # Save valid transcript content (main output)
        valid_success = False
        if valid_transcripts:
            # Get records for valid transcripts
            valid_transcript_keys = {result["transcript_key"] for result in valid_transcripts}
            valid_records = [
                record for record in original_records 
                if f"{record.get('ticker', 'unknown')}_{record.get('fiscal_year', 'unknown')}_{record.get('fiscal_quarter', 'unknown')}" in valid_transcript_keys
            ]
            
            # Output valid records in Stage 3 format (direct list)
            valid_output_content = valid_records
            
            valid_filename = "stage_04_validated_content.json"
            valid_file_path = nas_path_join(output_path, valid_filename)
            valid_json = json.dumps(valid_output_content, indent=2)
            valid_file_obj = io.BytesIO(valid_json.encode("utf-8"))
            
            if nas_upload_file(nas_conn, valid_file_obj, valid_file_path):
                log_execution("Valid transcript content saved successfully", {
                    "output_path": valid_file_path,
                    "valid_transcripts": len(valid_transcripts),
                    "total_records": len(valid_records),
                    "format": "Stage 3 format (direct records list)"
                })
                log_console(f"✅ Valid content saved: {valid_filename} ({len(valid_records)} records)")
                valid_success = True
            else:
                log_error("Failed to save valid transcript content", "output_save", {"path": valid_file_path})
        else:
            log_console("No valid transcripts found - no valid content file created", "WARNING")
            valid_success = True  # Not an error if no valid content
        
        # Save invalid transcript records (if any) in Stage 3 format
        invalid_success = False
        if invalid_transcripts:
            # Get records for invalid transcripts  
            invalid_transcript_keys = {result["transcript_key"] for result in invalid_transcripts}
            invalid_records = [
                record for record in original_records 
                if f"{record.get('ticker', 'unknown')}_{record.get('fiscal_year', 'unknown')}_{record.get('fiscal_quarter', 'unknown')}" in invalid_transcript_keys
            ]
            
            # Output invalid records in Stage 3 format (direct list)
            invalid_output_content = invalid_records
            
            invalid_filename = "stage_04_invalid_content.json"
            invalid_file_path = nas_path_join(output_path, invalid_filename)
            invalid_json = json.dumps(invalid_output_content, indent=2)
            invalid_file_obj = io.BytesIO(invalid_json.encode("utf-8"))
            
            if nas_upload_file(nas_conn, invalid_file_obj, invalid_file_path):
                log_execution("Invalid transcript records saved successfully", {
                    "output_path": invalid_file_path,
                    "invalid_transcripts": len(invalid_transcripts),
                    "total_records": len(invalid_records),
                    "format": "Stage 3 format (direct records list)"
                })
                log_console(f"⚠️ Invalid content saved: {invalid_filename} ({len(invalid_records)} records)", "WARNING")
                invalid_success = True
            else:
                log_error("Failed to save invalid transcript details", "output_save", {"path": invalid_file_path})
        else:
            log_console("No invalid transcripts found - no review file created")
            invalid_success = True  # Not an error if no invalid content
        
        return valid_success, invalid_success
        
    except Exception as e:
        error_msg = f"Error saving validation outputs: {e}"
        log_error(error_msg, "output_save", {"exception_type": type(e).__name__})
        return False, False


def main() -> None:
    """Main function to orchestrate Stage 4 content validation."""
    global config, logger

    # Initialize logging
    logger = setup_logging()
    log_console("=== STAGE 4: TRANSCRIPT CONTENT VALIDATION ===")

    # Initialize stage summary
    stage_summary = {
        "status": "unknown",
        "execution_time_seconds": 0,
        "total_transcripts_processed": 0,
        "valid_transcripts": 0,
        "invalid_transcripts": 0,
        "total_records_processed": 0,
        "errors": {
            "environment_validation": 0,
            "nas_connection": 0,
            "config_load": 0,
            "ssl_setup": 0,
            "content_load": 0,
            "validation_error": 0,
            "output_save": 0
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
        log_console(f"Loaded configuration - Expected sections: {config['stage_04_validate_structure']['expected_sections']}")

        # Step 4: SSL certificate setup (for consistency with other stages)
        log_console("Step 4: Setting up SSL certificate...")
        ssl_cert_path = setup_ssl_certificate(nas_conn)

        # Step 5: Proxy configuration (for consistency with other stages)
        log_console("Step 5: Setting up proxy configuration...")
        proxy_url = setup_proxy_configuration()

        # Step 6: Load extracted content from Stage 3
        log_console("Step 6: Loading extracted content from Stage 3...")
        extracted_content = load_extracted_content(nas_conn)
        
        # Handle both old format (direct list) and new format (dict with records)
        if isinstance(extracted_content, list):
            content_records = extracted_content
        else:
            content_records = extracted_content.get("records", [])
        
        if not content_records:
            log_console("No content records found to validate", "WARNING")
            stage_summary["status"] = "completed_no_content"
            return

        stage_summary["total_records_processed"] = len(content_records)

        # Step 7: Development mode handling
        dev_mode = config["stage_04_validate_structure"].get("dev_mode", False)
        if dev_mode:
            max_files = config["stage_04_validate_structure"].get("dev_max_files", 2)
            # Group by transcript first, then limit
            transcripts = {}
            for record in content_records:
                transcript_key = f"{record.get('ticker', 'unknown')}_{record.get('fiscal_year', 'unknown')}_{record.get('fiscal_quarter', 'unknown')}"
                if transcript_key not in transcripts:
                    transcripts[transcript_key] = []
                transcripts[transcript_key].append(record)
            
            # Limit number of transcripts
            limited_transcript_keys = list(transcripts.keys())[:max_files]
            limited_records = []
            for key in limited_transcript_keys:
                limited_records.extend(transcripts[key])
            
            content_records = limited_records
            log_console(f"Development mode: Processing only {len(limited_transcript_keys)} transcripts with {len(content_records)} records", "WARNING")

        # Step 8: Group records by transcript and validate
        log_console("Step 8: Grouping and validating transcript content...")
        
        # First, analyze the input records
        log_console(f"📥 INPUT ANALYSIS:")
        log_console(f"  Total paragraph records from Stage 3: {len(content_records)}")
        
        # Count unique files in the input
        unique_files_in_input = set()
        for record in content_records:
            file_path = record.get('file_path', '')
            if file_path:
                unique_files_in_input.add(file_path)
        log_console(f"  Unique files represented in records: {len(unique_files_in_input)}")
        
        # Track file consolidation details
        transcripts = {}
        file_to_transcript_mapping = {}  # Maps each unique file to its transcript key
        transcript_file_versions = defaultdict(set)  # Track all file versions per transcript
        records_per_file = defaultdict(int)  # Count records per file
        
        for record in content_records:
            # Build transcript key using ALL unique identifiers from filename
            # This ensures we don't incorrectly merge different versions or event IDs
            ticker = record.get('ticker', 'unknown')
            fiscal_year = record.get('fiscal_year', 'unknown')
            fiscal_quarter = record.get('fiscal_quarter', 'unknown')
            transcript_type = record.get('transcript_type', 'unknown')
            event_id = record.get('event_id', 'unknown')
            version_id = record.get('version_id', 'unknown')
            
            # Use the same key structure as the filename for unique identification
            transcript_key = f"{ticker}_{fiscal_quarter}_{fiscal_year}_{transcript_type}_{event_id}_{version_id}"
            
            # Track unique file information
            filename = record.get('filename', 'unknown')
            file_path = record.get('file_path', 'unknown')
            file_identifier = f"{filename} (event:{event_id}, version:{version_id}, type:{transcript_type})"
            
            # Count records per file
            records_per_file[file_path] += 1
            
            # Map file to transcript
            file_to_transcript_mapping[file_identifier] = transcript_key
            transcript_file_versions[transcript_key].add(file_identifier)
            
            # Group records
            if transcript_key not in transcripts:
                transcripts[transcript_key] = []
            transcripts[transcript_key].append(record)
        
        # Log consolidation details
        log_console("=" * 60)
        log_console("TRANSCRIPT GROUPING ANALYSIS")
        log_console("=" * 60)
        
        # Count unique files
        unique_files_count = len(unique_files_in_input)
        unique_transcripts_count = len(transcripts)
        
        log_console(f"📁 Unique files from Stage 3 output: {unique_files_count}")
        log_console(f"📊 Unique transcript keys after grouping: {unique_transcripts_count}")
        log_console(f"📎 Grouping key: ticker_quarter_year_type_eventid_versionid")
        
        # Check for discrepancy
        if unique_files_count != unique_transcripts_count:
            log_console(f"⚠️ DISCREPANCY: {unique_files_count} files resulted in {unique_transcripts_count} transcript groups")
            log_console(f"   This means {unique_files_count - unique_transcripts_count} files are missing from Stage 3 output!")
        else:
            log_console(f"✅ Files and transcript groups match perfectly")
        
        # Group by base transcript (ticker_quarter_year) to find versions
        base_transcript_groups = defaultdict(list)
        for transcript_key in transcripts.keys():
            # Extract base key (ticker_quarter_year) from full key
            key_parts = transcript_key.split('_')
            if len(key_parts) >= 3:
                base_key = f"{key_parts[0]}_{key_parts[1]}_{key_parts[2]}"
                base_transcript_groups[base_key].append(transcript_key)
        
        # Find transcripts with multiple versions
        multi_version_base_transcripts = {k: v for k, v in base_transcript_groups.items() if len(v) > 1}
        if multi_version_base_transcripts:
            log_console(f"\n🔄 Base transcripts with multiple versions/types: {len(multi_version_base_transcripts)}")
            log_console("Detailed breakdown (showing different versions of same base transcript):")
            
            # Show first 5 examples
            for i, (base_key, versions) in enumerate(list(multi_version_base_transcripts.items())[:5], 1):
                log_console(f"  {i}. {base_key}:")
                for version_key in sorted(versions):
                    # Extract type, event_id, version_id from full key
                    key_parts = version_key.split('_')
                    if len(key_parts) >= 6:
                        trans_type = key_parts[3]
                        event_id = key_parts[4]
                        version_id = key_parts[5]
                        log_console(f"     - Type: {trans_type}, Event: {event_id}, Version: {version_id}")
            
            if len(multi_version_base_transcripts) > 5:
                log_console(f"  ... and {len(multi_version_base_transcripts) - 5} more base transcripts with multiple versions")
        
        # Log version distribution by base transcript
        version_distribution = defaultdict(int)
        for base_key, versions in base_transcript_groups.items():
            version_distribution[len(versions)] += 1
        
        if version_distribution:
            log_console("\n📈 Version distribution (by base transcript):")
            for version_count in sorted(version_distribution.keys()):
                transcript_count = version_distribution[version_count]
                log_console(f"  {version_count} version(s) → {transcript_count} base transcript(s)")
        
        # Log transcript type distribution
        transcript_type_counts = defaultdict(int)
        for record in content_records:
            transcript_type = record.get('transcript_type', 'unknown')
            transcript_type_counts[transcript_type] += 1
        
        if transcript_type_counts:
            log_console("\n📝 Transcript type distribution:")
            for trans_type, count in transcript_type_counts.items():
                log_console(f"  {trans_type}: {count} records")
        
        log_console("=" * 60)
        
        # DIAGNOSTIC SUMMARY
        log_console(f"\n🔍 DIAGNOSTIC SUMMARY:")
        log_console(f"  Stage 2 → Stage 3: Expected {unique_files_count} files to produce content")
        log_console(f"  Stage 3 → Stage 4: Received records from {unique_files_count} files")
        log_console(f"  Stage 4 Grouping: Created {unique_transcripts_count} transcript groups")
        
        if unique_files_count != unique_transcripts_count:
            missing_count = unique_files_count - unique_transcripts_count
            log_console(f"\n  ❌ ISSUE IDENTIFIED:")
            log_console(f"     {missing_count} files are missing from Stage 3's output")
            log_console(f"     These files likely produced 0 paragraphs and were dropped")
            log_console(f"     Check Stage 3 logs for 'Files with zero paragraphs' section")
        
        log_console(f"\n✅ Total transcripts to validate: {len(transcripts)}")
        
        # Update stage summary with grouping details
        stage_summary["total_transcripts_processed"] = len(transcripts)
        stage_summary["grouping_details"] = {
            "unique_files_from_stage3": unique_files_count,
            "unique_transcript_versions": unique_transcripts_count,
            "grouping_key": "ticker_quarter_year_type_eventid_versionid",
            "base_transcripts": len(base_transcript_groups),
            "base_transcripts_with_multiple_versions": len(multi_version_base_transcripts),
            "version_distribution": dict(version_distribution),
            "transcript_type_distribution": dict(transcript_type_counts)
        }
        
        # Log grouping details to execution log
        log_execution("Transcript grouping completed", stage_summary["grouping_details"])
        
        # Validate each transcript
        all_validation_results = []
        for i, (transcript_key, transcript_records) in enumerate(transcripts.items(), 1):
            log_console(f"Validating transcript {i}/{len(transcripts)}: {transcript_key}")
            
            validation_result = validate_transcript_structure(transcript_records, transcript_key)
            all_validation_results.append(validation_result)

        # Step 9: Process validation results
        log_console("Step 9: Processing validation results...")
        valid_transcripts, invalid_transcripts = process_validation_results(nas_conn, all_validation_results)
        
        stage_summary["valid_transcripts"] = len(valid_transcripts)
        stage_summary["invalid_transcripts"] = len(invalid_transcripts)

        # Step 10: Save outputs
        log_console("Step 10: Saving validation outputs...")
        valid_success, invalid_success = save_validation_outputs(nas_conn, valid_transcripts, invalid_transcripts, content_records)
        
        if not valid_success or not invalid_success:
            stage_summary["status"] = "failed"
            log_console("Failed to save some validation outputs", "ERROR")
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
        log_console("=== STAGE 4 CONTENT VALIDATION COMPLETE ===")
        log_console(f"Transcripts processed: {stage_summary['total_transcripts_processed']}")
        log_console(f"Valid transcripts: {stage_summary['valid_transcripts']}")
        log_console(f"Invalid transcripts: {stage_summary['invalid_transcripts']}")
        log_console(f"Total records processed: {stage_summary['total_records_processed']}")
        log_console(f"Total errors: {sum(stage_summary['errors'].values())}")
        log_console(f"Execution time: {execution_time}")
        
        # Add grouping summary
        if "grouping_details" in stage_summary:
            log_console("\n📊 GROUPING SUMMARY:")
            log_console(f"  Stage 3 files → Stage 4 transcript versions: {stage_summary['grouping_details']['unique_files_from_stage3']} → {stage_summary['grouping_details']['unique_transcript_versions']}")
            log_console(f"  Grouping key: {stage_summary['grouping_details']['grouping_key']}")
            log_console(f"  Base transcripts (ticker_quarter_year): {stage_summary['grouping_details']['base_transcripts']}")
            log_console(f"  Base transcripts with multiple versions: {stage_summary['grouping_details']['base_transcripts_with_multiple_versions']}")

    except Exception as e:
        stage_summary["status"] = "failed"
        error_msg = f"Stage 4 content validation failed: {e}"
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

        log_console(f"Stage 4 content validation {stage_summary['status']}")


if __name__ == "__main__":
    main()
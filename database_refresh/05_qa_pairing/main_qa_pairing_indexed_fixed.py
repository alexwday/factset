"""
Stage 5: Q&A Boundary Detection & Conversation Pairing - CORRECTED Index-Based Approach
Processes Stage 4 validated content using corrected index-based speaker block analysis.
Self-contained standalone script that loads config from NAS at runtime.

FIXES APPLIED:
- Fixed critical index conversion algorithm bug
- Replaced range-based assignments with explicit block tracking
- Added comprehensive boundary validation and error handling
- Fixed security violations and added missing validation functions
- Added token counting safeguards and improved LLM integration
"""

import os
import tempfile
import logging
import json
import time
from datetime import datetime
from urllib.parse import quote, urlparse
from typing import Dict, Any, Optional, List, Tuple
import io
import re
import requests

import yaml
from smb.SMBConnection import SMBConnection
from dotenv import load_dotenv
from openai import OpenAI
from collections import defaultdict

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
        self.boundary_detection_errors = []
        self.authentication_errors = []
        self.validation_errors = []
        self.processing_errors = []
        self.total_cost = 0.0
        self.total_tokens = 0

    def log_boundary_error(self, transcript_id: str, qa_id: int, error: str):
        """Log Q&A boundary detection errors."""
        self.boundary_detection_errors.append({
            "timestamp": datetime.now().isoformat(),
            "transcript_id": transcript_id,
            "qa_id": qa_id,
            "error": error,
            "action_required": "Review speaker block context and retry"
        })

    def log_authentication_error(self, error: str):
        """Log OAuth/SSL authentication errors."""
        self.authentication_errors.append({
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "action_required": "Check LLM credentials and SSL certificate"
        })

    def log_validation_error(self, transcript_id: str, validation_issue: str):
        """Log Q&A validation errors."""
        self.validation_errors.append({
            "timestamp": datetime.now().isoformat(),
            "transcript_id": transcript_id,
            "validation_issue": validation_issue,
            "action_required": "Review Q&A group assignments"
        })

    def log_processing_error(self, transcript_id: str, error: str):
        """Log general processing errors."""
        self.processing_errors.append({
            "timestamp": datetime.now().isoformat(),
            "transcript_id": transcript_id,
            "error": error,
            "action_required": "Review transcript structure and processing logic"
        })

    def accumulate_costs(self, token_usage: dict = None):
        """Accumulate total cost and token usage for final summary."""
        if token_usage:
            if "cost" in token_usage:
                self.total_cost += token_usage["cost"]["total_cost"]
            if "total_tokens" in token_usage:
                self.total_tokens += token_usage["total_tokens"]

    def save_error_logs(self, nas_conn: SMBConnection):
        """Save error logs to separate JSON files on NAS."""
        global logger, config
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logs_path = config["stage_05_qa_pairing"]["output_logs_path"]
        
        error_types = {
            "boundary_detection": self.boundary_detection_errors,
            "authentication": self.authentication_errors, 
            "validation": self.validation_errors,
            "processing": self.processing_errors
        }
        
        for error_type, errors in error_types.items():
            if errors:
                filename = f"stage_05_qa_pairing_indexed_fixed_{error_type}_errors_{timestamp}.json"
                nas_path = nas_path_join(logs_path, "Errors", filename)
                
                try:
                    error_data = {
                        "error_type": error_type,
                        "timestamp": timestamp,
                        "total_errors": len(errors),
                        "errors": errors
                    }
                    
                    error_json = json.dumps(error_data, indent=2)
                    error_bytes = io.BytesIO(error_json.encode('utf-8'))
                    
                    if nas_upload_file(nas_conn, error_bytes, nas_path):
                        log_execution(f"Uploaded {error_type} error log: {filename}")
                    
                except Exception as e:
                    log_error(f"Failed to upload {error_type} error log: {e}", "error_logging", {})


def save_logs_to_nas(nas_conn: SMBConnection, stage_summary: Dict[str, Any], enhanced_error_logger: EnhancedErrorLogger):
    """Save execution and error logs to NAS at completion."""
    global execution_log, error_log

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logs_path = config["stage_05_qa_pairing"]["output_logs_path"]

    # Create logs directory
    nas_create_directory_recursive(nas_conn, logs_path)

    # Save main execution log
    main_log_content = {
        "stage": "stage_05_qa_pairing_indexed_fixed",
        "execution_start": (
            execution_log[0]["timestamp"]
            if execution_log
            else datetime.now().isoformat()
        ),
        "execution_end": datetime.now().isoformat(),
        "summary": stage_summary,
        "execution_log": execution_log,
    }

    main_log_filename = f"stage_05_qa_pairing_indexed_fixed_{timestamp}.json"
    main_log_path = nas_path_join(logs_path, main_log_filename)
    main_log_json = json.dumps(main_log_content, indent=2)
    main_log_obj = io.BytesIO(main_log_json.encode("utf-8"))

    if nas_upload_file(nas_conn, main_log_obj, main_log_path):
        log_console(f"Execution log saved: {main_log_filename}")

    # Save enhanced error logs
    enhanced_error_logger.save_error_logs(nas_conn)

    # Save basic error log only if errors exist
    if error_log:
        errors_path = nas_path_join(logs_path, "Errors")
        nas_create_directory_recursive(nas_conn, errors_path)

        error_log_content = {
            "stage": "stage_05_qa_pairing_indexed_fixed",
            "execution_time": datetime.now().isoformat(),
            "total_errors": len(error_log),
            "error_summary": stage_summary.get("errors", {}),
            "errors": error_log,
        }

        error_log_filename = f"stage_05_qa_pairing_indexed_fixed_errors_{timestamp}.json"
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
            # SECURITY FIX: Remove server IP from logs
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
                {"port": nas_port},  # SECURITY FIX: Removed server IP
            )
            return None

    except Exception as e:
        log_error(
            f"Error creating NAS connection: {e}",
            "nas_connection",
            {"exception_type": type(e).__name__},  # SECURITY FIX: Removed server IP
        )
        return None


def load_config_from_nas(nas_conn: SMBConnection) -> Dict[str, Any]:
    """Load and validate configuration from NAS."""

    try:
        log_execution("Loading configuration from NAS", {"config_path": "[SANITIZED]"})  # SECURITY FIX

        config_data = nas_download_file(nas_conn, os.getenv("CONFIG_PATH"))
        if not config_data:
            error_msg = "Configuration file not found"
            log_error(error_msg, "config_load", {})  # SECURITY FIX: Removed path
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
    """Validate configuration structure and required parameters with enhanced security."""

    required_sections = [
        "ssl_cert_path",
        "api_settings", 
        "stage_05_qa_pairing",
        "monitored_institutions"
    ]

    for section in required_sections:
        if section not in config:
            error_msg = f"Missing required configuration section: {section}"
            log_error(error_msg, "config_validation", {"missing_section": section})
            raise ValueError(error_msg)

    # Validate stage_05_qa_pairing specific parameters
    stage_05_qa_pairing_config = config["stage_05_qa_pairing"]
    required_stage_05_qa_pairing_params = [
        "description", 
        "input_data_path",
        "output_data_path",
        "output_logs_path",
        "dev_mode",
        "dev_max_transcripts",
        "llm_config",
        "indexed_config"
    ]

    for param in required_stage_05_qa_pairing_params:
        if param not in stage_05_qa_pairing_config:
            error_msg = f"Missing required stage_05_qa_pairing parameter: {param}"
            log_error(error_msg, "config_validation", {"missing_parameter": f"stage_05_qa_pairing.{param}"})
            raise ValueError(error_msg)

    # Validate LLM config structure with security checks
    llm_config = stage_05_qa_pairing_config["llm_config"]
    required_llm_params = [
        "base_url", "model", "temperature", "max_tokens", 
        "timeout", "max_retries", "token_endpoint",
        "cost_per_1k_prompt_tokens", "cost_per_1k_completion_tokens"
    ]

    for param in required_llm_params:
        if param not in llm_config:
            error_msg = f"Missing required LLM config parameter: {param}"
            log_error(error_msg, "config_validation", {"missing_parameter": f"llm_config.{param}"})
            raise ValueError(error_msg)

    # SECURITY FIX: Validate URLs are HTTPS
    if not validate_https_url(llm_config["base_url"]):
        raise ValueError("LLM base_url must use HTTPS")
    if not validate_https_url(llm_config["token_endpoint"]):
        raise ValueError("LLM token_endpoint must use HTTPS")

    # Validate indexed config structure with bounds checking
    indexed_config = stage_05_qa_pairing_config["indexed_config"]
    required_indexed_params = ["initial_window_size", "max_window_size", "window_expansion_step"]

    for param in required_indexed_params:
        if param not in indexed_config:
            error_msg = f"Missing required indexed config parameter: {param}"
            log_error(error_msg, "config_validation", {"missing_parameter": f"indexed_config.{param}"})
            raise ValueError(error_msg)

    # SECURITY FIX: Validate numeric bounds
    if indexed_config["initial_window_size"] < 1 or indexed_config["initial_window_size"] > 50:
        raise ValueError("initial_window_size must be between 1 and 50")
    if indexed_config["max_window_size"] < indexed_config["initial_window_size"]:
        raise ValueError("max_window_size must be >= initial_window_size")
    if indexed_config["window_expansion_step"] < 1:
        raise ValueError("window_expansion_step must be >= 1")

    # Validate monitored institutions
    if not config["monitored_institutions"]:
        error_msg = "monitored_institutions cannot be empty"
        log_error(error_msg, "config_validation", {"section": "monitored_institutions"})
        raise ValueError(error_msg)

    log_execution("Configuration validation successful", {
        "total_institutions": len(config["monitored_institutions"]),
        "llm_model": llm_config["model"],
        "indexed_config": indexed_config
    })


def validate_https_url(url: str) -> bool:
    """SECURITY FIX: Validate URL uses HTTPS protocol."""
    try:
        parsed = urlparse(url)
        return parsed.scheme == 'https'
    except Exception:
        return False


def validate_api_response_structure(response: Any) -> bool:
    """SECURITY FIX: Validate API response structure before processing."""
    try:
        # Validate OpenAI response structure
        if not hasattr(response, 'choices'):
            return False
        if not response.choices or len(response.choices) == 0:
            return False
        if not hasattr(response.choices[0], 'message'):
            return False
        return True
    except Exception:
        return False


def setup_ssl_certificate(nas_conn: SMBConnection) -> str:
    """Download SSL certificate from NAS to temporary file."""
    global config
    
    try:
        log_execution("Setting up SSL certificate", {})  # SECURITY FIX: Removed path
        
        cert_data = nas_download_file(nas_conn, config["ssl_cert_path"])
        if not cert_data:
            error_msg = "SSL certificate not found"
            log_error(error_msg, "ssl_setup", {})  # SECURITY FIX: Removed path
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

        log_execution("SSL certificate setup complete", {"temp_cert_created": True})
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
            "proxy_configured": True
        })  # SECURITY FIX: Removed proxy URL

        return proxy_with_auth

    except Exception as e:
        error_msg = f"Error setting up proxy configuration: {e}"
        log_error(error_msg, "proxy_setup", {"exception_type": type(e).__name__})
        raise


def get_oauth_token() -> Optional[str]:
    """Get OAuth token using client credentials flow with enhanced security."""
    global ssl_cert_path, config
    
    try:
        token_endpoint = config["stage_05_qa_pairing"]["llm_config"]["token_endpoint"]
        
        # SECURITY FIX: Validate endpoint before use
        if not validate_https_url(token_endpoint):
            error_msg = "OAuth token endpoint must use HTTPS"
            log_error(error_msg, "authentication", {})
            return None
        
        # Prepare OAuth request
        auth_data = {
            'grant_type': 'client_credentials',
            'client_id': os.getenv("LLM_CLIENT_ID"),
            'client_secret': os.getenv("LLM_CLIENT_SECRET")
        }
        
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        
        # Use SSL certificate if available
        verify_ssl = ssl_cert_path if ssl_cert_path else True
        
        response = requests.post(
            token_endpoint,
            data=auth_data,
            headers=headers,
            verify=verify_ssl,
            timeout=30
        )
        
        if response.status_code == 200:
            token_data = response.json()
            access_token = token_data.get('access_token')
            if access_token:
                log_execution("Successfully obtained OAuth token")
                return access_token
            else:
                error_msg = "No access token in OAuth response"
                log_error(error_msg, "authentication", {})
                return None
        else:
            # SECURITY FIX: Sanitize error message
            error_msg = f"OAuth token request failed: {response.status_code}"
            log_error(error_msg, "authentication", {})
            return None
            
    except requests.RequestException as e:
        # SECURITY FIX: Specific exception handling
        error_msg = f"OAuth token network error: {type(e).__name__}"
        log_error(error_msg, "authentication", {"exception_type": type(e).__name__})
        return None
    except Exception as e:
        error_msg = f"OAuth token acquisition failed: {type(e).__name__}"
        log_error(error_msg, "authentication", {"exception_type": type(e).__name__})
        return None


def setup_llm_client() -> Optional[OpenAI]:
    """Setup OpenAI client with custom base URL and OAuth token."""
    global oauth_token, config
    
    try:
        # Get OAuth token
        oauth_token = get_oauth_token()
        if not oauth_token:
            return None
        
        # Setup OpenAI client with custom configuration
        llm_config = config["stage_05_qa_pairing"]["llm_config"]
        
        client = OpenAI(
            api_key=oauth_token,  # Use OAuth token as API key
            base_url=llm_config["base_url"],
            timeout=llm_config.get("timeout", 60),
            max_retries=llm_config.get("max_retries", 3)
        )
        
        log_execution("Successfully configured LLM client")
        return client
        
    except Exception as e:
        error_msg = f"LLM client setup failed: {e}"
        log_error(error_msg, "authentication", {"exception_type": type(e).__name__})
        return None


def refresh_oauth_token_for_transcript() -> bool:
    """Refresh OAuth token for each new transcript."""
    global oauth_token, llm_client, config
    
    # Get fresh token for each transcript
    new_token = get_oauth_token()
    if new_token:
        oauth_token = new_token
        # Update client with new token
        llm_client = setup_llm_client()
        return llm_client is not None
    return False


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
        log_error(f"Invalid NAS file path", "path_validation", {})  # SECURITY FIX: Don't log path
        return None

    try:
        file_obj = io.BytesIO()
        conn.retrieveFile(os.getenv("NAS_SHARE_NAME"), nas_file_path, file_obj)
        file_obj.seek(0)
        return file_obj.read()
    except Exception as e:
        log_error(f"Failed to download file from NAS", "nas_download", 
                 {"error": str(e)})  # SECURITY FIX: Don't log path
        return None


def nas_upload_file(conn: SMBConnection, local_file_obj: io.BytesIO, nas_file_path: str) -> bool:
    """Upload a file object to NAS."""

    if not validate_nas_path(nas_file_path):
        log_error(f"Invalid NAS file path", "path_validation", {})  # SECURITY FIX: Don't log path
        return False

    try:
        # Ensure parent directory exists
        parent_dir = "/".join(nas_file_path.split("/")[:-1])
        if parent_dir:
            nas_create_directory_recursive(conn, parent_dir)

        conn.storeFile(os.getenv("NAS_SHARE_NAME"), nas_file_path, local_file_obj)
        return True
    except Exception as e:
        log_error(f"Failed to upload file to NAS", "nas_upload", 
                 {"error": str(e)})  # SECURITY FIX: Don't log path
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
        log_error("Cannot create directory with empty path", "directory_creation", {})
        return False

    path_parts = [part for part in normalized_path.split("/") if part]
    if not path_parts:
        log_error("Cannot create directory with invalid path", "directory_creation", {})
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
                log_error(f"Failed to create directory", "directory_creation", 
                         {"error": str(e)})  # SECURITY FIX: Don't log path
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


def load_stage_4_content(nas_conn: SMBConnection) -> Dict[str, Any]:
    """Load Stage 4 validated content from NAS."""
    global config
    
    try:
        input_path = config["stage_05_qa_pairing"]["input_data_path"]
        
        log_execution("Loading Stage 4 validated content from NAS", {})  # SECURITY FIX: Don't log path
        
        content_data = nas_download_file(nas_conn, input_path)
        if not content_data:
            error_msg = "Stage 4 content file not found"
            log_error(error_msg, "content_load", {})  # SECURITY FIX: Don't log path
            raise FileNotFoundError(error_msg)
        
        content = json.loads(content_data.decode("utf-8"))
        
        # Handle both old format (direct records list) and new format (dict with records)
        if isinstance(content, list):
            records = content
            log_execution("Loaded Stage 4 content (legacy format)", {"total_records": len(records)})
        else:
            records = content.get("records", [])
            log_execution("Loaded Stage 4 content successfully", {
                "total_records": len(records),
                "validation_summary": content.get("validation_summary", {})
            })
        
        return records
        
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in Stage 4 content file: {e}"
        log_error(error_msg, "content_parse", {"json_error": str(e)})
        raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"Error loading Stage 4 content: {e}"
        log_error(error_msg, "content_load", {"exception_type": type(e).__name__})
        raise


def group_records_by_transcript(records: List[Dict]) -> Dict[str, List[Dict]]:
    """Group records by transcript for processing."""
    transcripts = defaultdict(list)
    
    for record in records:
        # Create transcript key using filename or event identifiers
        transcript_key = record.get("filename", f"{record.get('ticker', 'unknown')}_{record.get('event_id', 'unknown')}")
        transcripts[transcript_key].append(record)
    
    return dict(transcripts)


def group_records_by_speaker_block(records: List[Dict]) -> List[Dict]:
    """Group records by speaker blocks within a transcript with validation."""
    speaker_blocks = defaultdict(list)
    
    # Group paragraphs by speaker block with validation
    for record in records:
        # VALIDATION FIX: Check required fields exist
        if "speaker_block_id" not in record:
            log_error(f"Record missing speaker_block_id", "validation", 
                     {"record_keys": list(record.keys())})
            continue
            
        speaker_block_id = record["speaker_block_id"]
        if not isinstance(speaker_block_id, int) or speaker_block_id < 1:
            log_error(f"Invalid speaker_block_id: {speaker_block_id}", "validation", {})
            continue
            
        speaker_blocks[speaker_block_id].append(record)
    
    # Create speaker block objects with metadata validation
    speaker_block_list = []
    for block_id in sorted(speaker_blocks.keys()):
        paragraphs = speaker_blocks[block_id]
        
        if not paragraphs:
            continue
        
        # Use first paragraph for speaker block metadata
        first_paragraph = paragraphs[0]
        
        # VALIDATION FIX: Check for required fields
        if "speaker" not in first_paragraph:
            log_error(f"Speaker block {block_id} missing speaker field", "validation", {})
            continue
        
        speaker_block = {
            "speaker_block_id": block_id,
            "speaker": first_paragraph["speaker"],
            "question_answer_flag": first_paragraph.get("question_answer_flag"),
            "section_name": first_paragraph.get("section_name"),
            "paragraphs": paragraphs
        }
        
        speaker_block_list.append(speaker_block)
    
    return speaker_block_list


def estimate_token_count(text: str) -> int:
    """IMPROVEMENT: Rough token count estimation for context management."""
    # Rough approximation: 1 token ≈ 4 characters for English text
    return len(text) // 4


def create_indexed_speaker_context(qa_speaker_blocks: List[Dict], 
                                 start_index: int, 
                                 current_qa_id: int,
                                 window_size: int,
                                 transcript_metadata: Dict) -> Dict:
    """
    Create indexed context window for LLM analysis with fixed indexing.
    
    ALGORITHM FIX: Uses consistent 1-based indexing for LLM consumption.
    """
    
    # Validate input parameters
    if start_index < 0 or start_index >= len(qa_speaker_blocks):
        raise ValueError(f"Invalid start_index: {start_index}")
    if window_size < 1:
        raise ValueError(f"Invalid window_size: {window_size}")
    
    # Get the window of speaker blocks
    end_index = min(start_index + window_size, len(qa_speaker_blocks))
    window_blocks = qa_speaker_blocks[start_index:end_index]
    
    # ALGORITHM FIX: Create indexed blocks with 1-based indexing for LLM
    indexed_blocks = []
    total_tokens = 0
    
    for i, block in enumerate(window_blocks):
        # 1-based index for LLM (starts at 1 for each window)
        llm_index = i + 1
        
        # Format content and estimate tokens
        content = format_speaker_block_content(block)
        tokens = estimate_token_count(content)
        total_tokens += tokens
        
        indexed_block = {
            "index": llm_index,  # 1-based for LLM
            "speaker_block_id": block["speaker_block_id"],
            "speaker": block["speaker"],
            "qa_id": current_qa_id if i == 0 else None,  # Only first block gets qa_id
            "content": content,
            "question_answer_flag": block.get("question_answer_flag"),  # IMPROVEMENT: Include for better LLM decisions
            "section_name": block.get("section_name")
        }
        indexed_blocks.append(indexed_block)
    
    return {
        "transcript_metadata": transcript_metadata,
        "current_qa_id": current_qa_id,
        "window_start_index": start_index,  # 0-based for internal use
        "window_size": len(window_blocks),
        "total_blocks": len(qa_speaker_blocks),
        "indexed_blocks": indexed_blocks,
        "can_expand": end_index < len(qa_speaker_blocks),
        "estimated_tokens": total_tokens
    }


def format_speaker_block_content(block: Dict) -> str:
    """Format speaker block content for LLM analysis."""
    paragraphs_text = []
    
    for para in block["paragraphs"]:
        content = para['paragraph_content'].strip()
        paragraphs_text.append(content)
    
    return "\n".join(paragraphs_text)


def format_indexed_context(context_data: Dict) -> str:
    """Format indexed context for LLM analysis with improved structure."""
    context_sections = []
    
    # Company and transcript context
    transcript_metadata = context_data.get("transcript_metadata", {})
    company_name = transcript_metadata.get("company_name", "Unknown Company")
    transcript_title = transcript_metadata.get("transcript_title", "Earnings Call Transcript")
    
    context_sections.append(f"<company>{company_name}</company>")
    context_sections.append(f"<transcript_title>{transcript_title}</transcript_title>")
    context_sections.append("")
    
    # Current QA ID and window info
    current_qa_id = context_data["current_qa_id"]
    window_size = context_data["window_size"]
    total_blocks = context_data["total_blocks"]
    
    context_sections.append(f"<context_info>")
    context_sections.append(f"Current QA ID: {current_qa_id}")
    context_sections.append(f"Window Size: {window_size} speaker blocks")
    context_sections.append(f"Total Blocks in Transcript: {total_blocks}")
    context_sections.append(f"Estimated Tokens: {context_data.get('estimated_tokens', 'unknown')}")
    context_sections.append(f"</context_info>")
    context_sections.append("")
    
    # Indexed speaker blocks with enhanced formatting
    context_sections.append("<speaker_blocks>")
    for block in context_data["indexed_blocks"]:
        qa_assignment = f" (QA ID: {block['qa_id']})" if block['qa_id'] else ""
        context_sections.append(f"Speaker Block {block['index']}{qa_assignment}:")
        context_sections.append(f"Speaker: {block['speaker']}")
        
        # IMPROVEMENT: Include metadata for better decisions
        if block.get('question_answer_flag'):
            context_sections.append(f"Type: {block['question_answer_flag']}")
        if block.get('section_name'):
            context_sections.append(f"Section: {block['section_name']}")
            
        context_sections.append(f"Content: {block['content']}")
        context_sections.append("")
    context_sections.append("</speaker_blocks>")
    
    # Expansion possibility
    if context_data["can_expand"]:
        context_sections.append("<expansion_note>")
        context_sections.append("Additional speaker blocks are available if you need more context.")
        context_sections.append("</expansion_note>")
    
    return "\n".join(context_sections)


def create_indexed_boundary_prompt(formatted_context: str, current_qa_id: int) -> str:
    """Create improved prompt for index-based boundary detection."""
    
    return f"""<task>
You are analyzing an earnings call Q&A transcript to identify where the current analyst's session ends and the next analyst begins.
</task>

<context>
You are looking at a numbered list of speaker blocks (numbered 1, 2, 3, etc.). Speaker Block 1 is assigned to QA ID {current_qa_id}. 
Your task is to determine the index number where the NEXT analyst's session begins.
</context>

<objective>
Find the speaker block index where a NEW analyst session should start (meaning the current analyst session ends at the previous block).

Return either:
1. The index number (1-based) where the next analyst session begins
2. A request for more speaker blocks if you cannot determine the boundary within the current window
</objective>

<instructions>
Analyze the conversation flow to identify when one analyst's complete session ends.

<decision_criteria>
Next analyst session starts when:
- A clearly different analyst begins asking questions
- Operator introduces a new analyst ("next question from...")
- Current analyst's questions are completely answered and conversation moves to new topic/person
- There's a clear transition from one analyst's complete interaction to another's

DO NOT start new session for:
- Follow-up questions from the same analyst
- Executive responses to the current analyst's questions
- Brief operator comments within the same analyst session
- Clarifications or continued discussion of same analyst's topics
</decision_criteria>

<examples>
Good boundary detection:
- Block 1-3: Analyst A asks questions and gets responses
- Block 4: Operator says "Next question from John Smith at XYZ Bank"
- Return: next_analyst_index = 4

Poor boundary detection:
- Block 1: Analyst asks question
- Block 2: Executive gives partial answer
- Block 3: Same analyst asks follow-up
- DO NOT split at block 2 or 3
</examples>

<key_principles>
1. **Complete Sessions**: Each QA ID should capture one analyst's full interaction from start to finish
2. **Clear Boundaries**: Only create new sessions when there's a definitive transition to a new analyst
3. **Index Accuracy**: Use the exact 1-based index numbers shown in the speaker blocks
4. **Conservative Approach**: When uncertain, prefer to continue current session rather than create premature breaks
</key_principles>
</instructions>

{formatted_context}

<output_format>
Call the appropriate function:
- boundary_decision: If you can identify where the next analyst session begins
- request_more_blocks: If you need additional speaker blocks to make a decision
</output_format>"""


def create_indexed_boundary_tools():
    """Create improved function calling tools for indexed boundary detection."""
    
    return [
        {
            "type": "function",
            "function": {
                "name": "boundary_decision",
                "description": "Identify the speaker block index where the next analyst session begins",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "next_analyst_index": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 50,
                            "description": "The speaker block index number (1-based) where the next analyst session should begin. Must be a positive integer."
                        },
                        "reasoning": {
                            "type": "string",
                            "minLength": 5,  # IMPROVEMENT: Reduced minimum for flexibility
                            "maxLength": 500,
                            "description": "Brief explanation of why this index marks the start of a new analyst session"
                        }
                    },
                    "required": ["next_analyst_index", "reasoning"],
                    "additionalProperties": false  # IMPROVEMENT: Prevent unexpected fields
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "request_more_blocks",
                "description": "Request additional speaker blocks to make boundary decision",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "minLength": 5,  # IMPROVEMENT: Reduced minimum for flexibility
                            "maxLength": 200,
                            "description": "Explanation of why more blocks are needed to determine the boundary"
                        }
                    },
                    "required": ["reason"],
                    "additionalProperties": false  # IMPROVEMENT: Prevent unexpected fields
                }
            }
        }
    ]


def calculate_token_cost(prompt_tokens: int, completion_tokens: int) -> dict:
    """Calculate cost based on token usage and configured rates."""
    global config
    
    prompt_cost_per_1k = config["stage_05_qa_pairing"]["llm_config"]["cost_per_1k_prompt_tokens"]
    completion_cost_per_1k = config["stage_05_qa_pairing"]["llm_config"]["cost_per_1k_completion_tokens"]
    
    prompt_cost = (prompt_tokens / 1000) * prompt_cost_per_1k
    completion_cost = (completion_tokens / 1000) * completion_cost_per_1k
    total_cost = prompt_cost + completion_cost
    
    return {
        "prompt_cost": round(prompt_cost, 6),
        "completion_cost": round(completion_cost, 6),
        "total_cost": round(total_cost, 6)
    }


def make_indexed_boundary_decision(context_data: Dict,
                                 current_qa_id: int,
                                 transcript_id: str,
                                 enhanced_error_logger: EnhancedErrorLogger) -> Optional[Dict]:
    """
    Make LLM decision for indexed boundary detection with improved error handling.
    Returns dict with decision type and details.
    """
    global llm_client, config
    
    try:
        # IMPROVEMENT: Check token count before sending to LLM
        estimated_tokens = context_data.get("estimated_tokens", 0)
        max_context_tokens = 8000  # Conservative limit for GPT-4
        
        if estimated_tokens > max_context_tokens:
            log_error(f"Context too large: {estimated_tokens} tokens", "token_limit", 
                     {"qa_id": current_qa_id, "estimated_tokens": estimated_tokens})
            return {"type": "context_too_large", "estimated_tokens": estimated_tokens}
        
        # Format context for LLM analysis
        formatted_context = format_indexed_context(context_data)
        prompt = create_indexed_boundary_prompt(formatted_context, current_qa_id)
        
        # Retry logic for LLM calls with improved error handling
        max_retries = 3 
        retry_delay = 1  # seconds
        decision = None  # Initialize decision variable to prevent NameError
        
        for attempt in range(1, max_retries + 1):
            try:
                log_execution(f"QA ID {current_qa_id} indexed boundary analysis attempt {attempt}/{max_retries}")
                
                # Make LLM API call with enhanced configuration
                try:
                    response = llm_client.chat.completions.create(
                        model=config["stage_05_qa_pairing"]["llm_config"]["model"],
                        messages=[{"role": "user", "content": prompt}],
                        tools=create_indexed_boundary_tools(),
                        tool_choice="required", 
                        temperature=config["stage_05_qa_pairing"]["llm_config"]["temperature"],
                        max_tokens=config["stage_05_qa_pairing"]["llm_config"]["max_tokens"],
                        timeout=config["stage_05_qa_pairing"]["llm_config"].get("timeout", 60)
                    )
                except Exception as llm_api_error:
                    error_msg = f"LLM API call failed for QA ID {current_qa_id}: {type(llm_api_error).__name__}"
                    if attempt == max_retries:
                        log_error(f"{error_msg} (final attempt)", "boundary_detection", {
                            "error_type": type(llm_api_error).__name__,
                            "attempt": attempt,
                            "qa_id": current_qa_id
                        })
                        enhanced_error_logger.log_boundary_error(transcript_id, current_qa_id, f"{error_msg} after {max_retries} attempts")
                        return None
                    else:
                        log_execution(f"QA ID {current_qa_id}: {error_msg}, retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        continue
                
                # SECURITY FIX: Validate response structure
                if not validate_api_response_structure(response):
                    error_msg = f"Invalid API response structure for QA ID {current_qa_id}"
                    if attempt == max_retries:
                        log_error(f"{error_msg} (final attempt)", "boundary_detection", {})
                        enhanced_error_logger.log_boundary_error(transcript_id, current_qa_id, f"{error_msg} after {max_retries} attempts")
                        return None
                    else:
                        log_execution(f"QA ID {current_qa_id}: {error_msg}, retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        continue
                
                # Parse response with enhanced validation
                if not response.choices[0].message.tool_calls:
                    error_msg = f"No tool call response for QA ID {current_qa_id}"
                    if attempt == max_retries:
                        log_error(f"{error_msg} (final attempt)", "boundary_detection", {})
                        enhanced_error_logger.log_boundary_error(transcript_id, current_qa_id, f"{error_msg} after {max_retries} attempts")
                        return None
                    else:
                        log_execution(f"QA ID {current_qa_id}: {error_msg}, retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        continue
                
                # Extract decision with robust validation
                tool_call = None
                function_name = "unknown"
                try:
                    tool_call = response.choices[0].message.tool_calls[0]
                    function_name = tool_call.function.name
                    
                    # IMPROVEMENT: Enhanced JSON parsing with validation
                    try:
                        result = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as json_err:
                        # Try to clean the JSON string
                        cleaned_args = tool_call.function.arguments.strip()
                        if not cleaned_args.startswith('{'):
                            cleaned_args = '{' + cleaned_args
                        if not cleaned_args.endswith('}'):
                            cleaned_args = cleaned_args + '}'
                        try:
                            result = json.loads(cleaned_args)
                        except json.JSONDecodeError:
                            raise ValueError(f"LLM returned invalid JSON: {json_err}")
                    
                    if function_name == "boundary_decision":
                        # ROBUST VALIDATION: Check field existence and type coercion
                        if "next_analyst_index" not in result:
                            raise ValueError("Missing 'next_analyst_index' field in LLM response")
                        if "reasoning" not in result:
                            raise ValueError("Missing 'reasoning' field in LLM response")
                        
                        # IMPROVEMENT: Type coercion with validation
                        raw_index = result["next_analyst_index"]
                        try:
                            if isinstance(raw_index, str) and raw_index.isdigit():
                                next_analyst_index = int(raw_index)
                            elif isinstance(raw_index, (int, float)):
                                next_analyst_index = int(raw_index)
                            else:
                                raise ValueError(f"Cannot convert index to integer: {raw_index}")
                        except (ValueError, TypeError):
                            raise ValueError(f"Invalid boundary index format: {raw_index}")
                        
                        # IMPROVEMENT: Range validation with context awareness
                        window_size = context_data.get("window_size", 10)
                        if next_analyst_index < 1:
                            raise ValueError(f"Boundary index must be >= 1, got: {next_analyst_index}")
                        if next_analyst_index > window_size + 5:  # Allow some buffer for edge cases
                            raise ValueError(f"Boundary index {next_analyst_index} exceeds reasonable range (window size: {window_size})")
                        
                        # Validate reasoning field
                        reasoning = result.get("reasoning", "").strip()
                        if not reasoning or len(reasoning) < 5:
                            reasoning = f"LLM boundary decision for index {next_analyst_index}"
                        
                        decision = {
                            "type": "boundary_found",
                            "next_analyst_index": next_analyst_index,
                            "reasoning": reasoning
                        }
                        
                    elif function_name == "request_more_blocks":
                        # ROBUST VALIDATION: Check reason field
                        if "reason" not in result:
                            raise ValueError("Missing 'reason' field in LLM response")
                        
                        reason = result.get("reason", "").strip()
                        if not reason or len(reason) < 5:
                            reason = "LLM requested more context blocks"
                        
                        decision = {
                            "type": "request_expansion",
                            "reason": reason
                        }
                    else:
                        raise ValueError(f"Unknown function name: {function_name}")
                        
                except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                    # IMPROVEMENT: Enhanced error logging with debugging info
                    error_details = {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "function_name": function_name,
                        "has_tool_calls": bool(getattr(response, 'choices', [{}])[0].message.tool_calls if hasattr(response, 'choices') and response.choices else False),
                        "attempt": attempt,
                        "qa_id": current_qa_id
                    }
                    
                    # Log the raw arguments for debugging (sanitized)
                    if tool_call and hasattr(tool_call, 'function'):
                        raw_args = tool_call.function.arguments[:200] + "..." if len(tool_call.function.arguments) > 200 else tool_call.function.arguments
                        error_details["raw_arguments_preview"] = raw_args
                    
                    error_msg = f"Failed to parse LLM response for QA ID {current_qa_id}: {type(e).__name__} - {str(e)}"
                    
                    if attempt == max_retries:
                        log_error(f"{error_msg} (final attempt)", "boundary_detection", error_details)
                        enhanced_error_logger.log_boundary_error(transcript_id, current_qa_id, f"{error_msg} after {max_retries} attempts")
                        return None
                    else:
                        log_execution(f"QA ID {current_qa_id}: {error_msg}, retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        continue
                
                # Log token usage and cost
                if hasattr(response, 'usage') and response.usage:
                    cost_info = calculate_token_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
                    token_usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                        "cost": cost_info
                    }
                    log_execution(f"QA ID {current_qa_id} tokens - input: {response.usage.prompt_tokens}, output: {response.usage.completion_tokens}, total: {response.usage.total_tokens}, cost: ${cost_info['total_cost']:.4f}")
                    
                    # Accumulate costs for final summary
                    enhanced_error_logger.accumulate_costs(token_usage)
                
                # Log successful decision
                attempt_info = f" (attempt {attempt})" if attempt > 1 else ""
                log_execution(f"QA ID {current_qa_id} decision{attempt_info}: {decision['type']}")
                
                return decision
                
            except requests.RequestException as e:
                # SECURITY FIX: Specific network error handling
                error_msg = f"Network error for QA ID {current_qa_id}: {type(e).__name__}"
                if attempt == max_retries:
                    log_error(f"{error_msg} (final attempt)", "boundary_detection", {})
                    enhanced_error_logger.log_boundary_error(transcript_id, current_qa_id, f"{error_msg} after {max_retries} attempts")
                    return None
                else:
                    log_execution(f"QA ID {current_qa_id}: {error_msg}, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    
            except Exception as e:
                error_msg = f"LLM analysis failed for QA ID {current_qa_id}: {type(e).__name__}"
                if attempt == max_retries:
                    log_error(f"{error_msg} (final attempt)", "boundary_detection", {})
                    enhanced_error_logger.log_boundary_error(transcript_id, current_qa_id, f"{error_msg} after {max_retries} attempts")
                    return None
                else:
                    log_execution(f"QA ID {current_qa_id}: {error_msg}, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
        
        # Should never reach here, but safety fallback
        if decision is None:
            log_error(f"Decision variable remained None for QA ID {current_qa_id}", "boundary_detection", {})
            enhanced_error_logger.log_boundary_error(transcript_id, current_qa_id, "Decision variable was None after all retry attempts")
        return decision
        
    except Exception as e:
        error_msg = f"Context preparation failed for QA ID {current_qa_id}: {type(e).__name__}"
        log_error(error_msg, "boundary_detection", {})
        enhanced_error_logger.log_boundary_error(transcript_id, current_qa_id, error_msg)
        return None


def process_qa_boundaries_indexed(speaker_blocks: List[Dict], 
                                transcript_id: str, 
                                transcript_metadata: Dict, 
                                enhanced_error_logger: EnhancedErrorLogger) -> List[Dict]:
    """
    CORRECTED: Process Q&A boundaries using fixed index-based approach.
    
    Algorithm Fixes:
    1. Proper index conversion between LLM 1-based and Python 0-based
    2. Comprehensive boundary validation
    3. Explicit block assignment tracking (no range-based assignments)
    4. Improved error handling and fallback strategies
    """
    global config
    
    try:
        # Filter to only Q&A sections
        qa_speaker_blocks = [block for block in speaker_blocks 
                           if block.get("section_name") == "Q&A"]
        
        if not qa_speaker_blocks:
            log_execution(f"No Q&A sections found in transcript {transcript_id}")
            return []
        
        log_execution(f"Processing {len(qa_speaker_blocks)} Q&A speaker blocks using corrected indexed approach")
        
        # Initialize processing state
        current_index = 0  # 0-based index in qa_speaker_blocks
        current_qa_id = 1
        block_assignments = []  # ALGORITHM FIX: Track explicit assignments, not ranges
        
        indexed_config = config["stage_05_qa_pairing"]["indexed_config"]
        initial_window_size = indexed_config["initial_window_size"]
        max_window_size = indexed_config["max_window_size"]
        window_expansion_step = indexed_config["window_expansion_step"]
        
        # Safety check to prevent infinite loops
        max_iterations = len(qa_speaker_blocks) * 2
        iteration_count = 0
        
        while current_index < len(qa_speaker_blocks) and iteration_count < max_iterations:
            iteration_count += 1
            log_execution(f"Processing QA ID {current_qa_id} starting at block index {current_index} (iteration {iteration_count})")
            
            # Start with initial window size
            window_size = initial_window_size
            boundary_found = False
            
            while not boundary_found and window_size <= max_window_size:
                # VALIDATION FIX: Ensure we don't exceed available blocks
                effective_window_size = min(window_size, len(qa_speaker_blocks) - current_index)
                
                if effective_window_size <= 0:
                    log_execution(f"No blocks remaining for QA ID {current_qa_id}")
                    boundary_found = True
                    break
                
                # Create context window
                try:
                    context_data = create_indexed_speaker_context(
                        qa_speaker_blocks,
                        current_index,
                        current_qa_id,
                        effective_window_size,
                        transcript_metadata
                    )
                except ValueError as e:
                    log_error(f"Context creation failed: {e}", "processing", {"qa_id": current_qa_id})
                    boundary_found = True
                    break
                
                # Make LLM decision
                decision = make_indexed_boundary_decision(
                    context_data,
                    current_qa_id,
                    transcript_id,
                    enhanced_error_logger
                )
                
                if not decision:
                    # LLM failed, assign remaining blocks in current window to current QA ID
                    log_execution(f"LLM failed for QA ID {current_qa_id}, assigning current window blocks")
                    for i in range(current_index, min(current_index + effective_window_size, len(qa_speaker_blocks))):
                        block = qa_speaker_blocks[i]
                        block_assignments.append({
                            "speaker_block_id": block["speaker_block_id"],
                            "qa_group_id": current_qa_id,
                            "method": "llm_failure_fallback"
                        })
                    current_index = min(current_index + effective_window_size, len(qa_speaker_blocks))
                    current_qa_id += 1
                    boundary_found = True
                    
                elif decision["type"] == "boundary_found":
                    # ALGORITHM FIX: Proper index conversion
                    llm_next_index = decision["next_analyst_index"]  # 1-based from LLM
                    
                    # VALIDATION FIX: Comprehensive boundary validation
                    if not isinstance(llm_next_index, int) or llm_next_index < 1:
                        log_execution(f"Invalid LLM index {llm_next_index}, using fallback")
                        llm_next_index = min(2, effective_window_size)  # Safe fallback
                    
                    if llm_next_index > effective_window_size:
                        log_execution(f"LLM index {llm_next_index} exceeds window size {effective_window_size}, using fallback")
                        llm_next_index = effective_window_size
                    
                    # Convert to 0-based array index relative to current_index
                    # LLM returns 1-based index within the window (1, 2, 3, ...)
                    # Convert to absolute 0-based array index
                    next_analyst_array_index = current_index + (llm_next_index - 1)
                    
                    # VALIDATION FIX: Ensure forward progress
                    if next_analyst_array_index <= current_index:
                        log_execution(f"Boundary would not advance, forcing progress")
                        next_analyst_array_index = current_index + 1
                    
                    # VALIDATION FIX: Ensure within bounds
                    next_analyst_array_index = min(next_analyst_array_index, len(qa_speaker_blocks))
                    
                    # ALGORITHM FIX: Assign specific blocks (not ranges) to current QA ID
                    for i in range(current_index, next_analyst_array_index):
                        if i < len(qa_speaker_blocks):
                            block = qa_speaker_blocks[i]
                            block_assignments.append({
                                "speaker_block_id": block["speaker_block_id"],
                                "qa_group_id": current_qa_id,
                                "method": "indexed_llm_decision"
                            })
                    
                    blocks_assigned = next_analyst_array_index - current_index
                    log_execution(f"QA ID {current_qa_id} assigned {blocks_assigned} blocks (indices {current_index} to {next_analyst_array_index-1}), reasoning: {decision.get('reasoning', 'N/A')}")
                    
                    # Move to next QA ID
                    current_index = next_analyst_array_index
                    current_qa_id += 1
                    boundary_found = True
                    
                elif decision["type"] == "request_expansion":
                    # LLM wants more blocks
                    log_execution(f"QA ID {current_qa_id} requested window expansion from {window_size} blocks: {decision['reason']}")
                    window_size += window_expansion_step
                    
                    if window_size > max_window_size:
                        # Reached max window size, assign reasonable number of blocks
                        log_execution(f"Reached max window size for QA ID {current_qa_id}, using fallback assignment")
                        fallback_blocks = min(initial_window_size, len(qa_speaker_blocks) - current_index)
                        for i in range(current_index, current_index + fallback_blocks):
                            if i < len(qa_speaker_blocks):
                                block = qa_speaker_blocks[i]
                                block_assignments.append({
                                    "speaker_block_id": block["speaker_block_id"],
                                    "qa_group_id": current_qa_id,
                                    "method": "max_window_fallback"
                                })
                        current_index += fallback_blocks
                        current_qa_id += 1
                        boundary_found = True
                        
                elif decision["type"] == "context_too_large":
                    # Context too large, use smaller window
                    log_execution(f"Context too large for QA ID {current_qa_id}, using reduced window")
                    fallback_blocks = min(5, len(qa_speaker_blocks) - current_index)  # Smaller fallback
                    for i in range(current_index, current_index + fallback_blocks):
                        if i < len(qa_speaker_blocks):
                            block = qa_speaker_blocks[i]
                            block_assignments.append({
                                "speaker_block_id": block["speaker_block_id"],
                                "qa_group_id": current_qa_id,
                                "method": "context_size_fallback"
                            })
                    current_index += fallback_blocks
                    current_qa_id += 1
                    boundary_found = True
        
        # Safety check for infinite loop
        if iteration_count >= max_iterations:
            log_error(f"Maximum iterations reached, may indicate infinite loop", "processing", 
                     {"transcript_id": transcript_id, "iterations": iteration_count})
        
        # ALGORITHM FIX: Convert explicit assignments to QA groups
        qa_groups = convert_explicit_assignments_to_qa_groups(block_assignments)
        
        log_execution(f"Successfully identified {len(qa_groups)} Q&A groups using corrected indexed approach")
        return qa_groups
        
    except Exception as e:
        error_msg = f"Indexed Q&A boundary processing failed: {e}"
        log_error(error_msg, "processing", {"exception_type": type(e).__name__})
        enhanced_error_logger.log_processing_error(transcript_id, error_msg)
        return []


def convert_explicit_assignments_to_qa_groups(assignments: List[Dict]) -> List[Dict]:
    """ALGORITHM FIX: Convert explicit block assignments to QA groups without range errors."""
    if not assignments:
        return []
    
    # Group assignments by QA ID
    qa_groups_dict = defaultdict(list)
    for assignment in assignments:
        qa_id = assignment["qa_group_id"]
        qa_groups_dict[qa_id].append(assignment)
    
    # Create QA group objects
    qa_groups = []
    for qa_id in sorted(qa_groups_dict.keys()):
        group_assignments = qa_groups_dict[qa_id]
        block_ids = [a["speaker_block_id"] for a in group_assignments]
        
        # ALGORITHM FIX: Track explicit assignments, not ranges
        # This prevents the range expansion bug
        
        # Determine primary method for the group
        methods = [a["method"] for a in group_assignments]
        if "indexed_llm_decision" in methods:
            primary_method = "indexed_llm_decision"
            confidence = 0.9
        elif "llm_failure_fallback" in methods:
            primary_method = "llm_failure_fallback"
            confidence = 0.6
        elif "context_size_fallback" in methods:
            primary_method = "context_size_fallback"
            confidence = 0.5
        else:
            primary_method = "max_window_fallback"
            confidence = 0.4
        
        qa_group = {
            "qa_group_id": qa_id,
            "start_block_id": min(block_ids),
            "end_block_id": max(block_ids),
            "confidence": confidence,
            "method": primary_method,
            "assigned_block_ids": sorted(block_ids),  # ALGORITHM FIX: Explicit list
            "block_assignments": group_assignments
        }
        qa_groups.append(qa_group)
    
    return qa_groups


def extract_transcript_metadata(transcript_records: List[Dict]) -> Dict:
    """Extract company name and transcript title from records."""
    if not transcript_records:
        return {"company_name": "Unknown Company", "transcript_title": "Earnings Call Transcript"}
    
    # Try to extract from first record
    first_record = transcript_records[0]
    
    # Extract company name (try multiple fields)
    company_name = (
        first_record.get("company_name") or 
        first_record.get("ticker", "").replace("-", " ") or
        "Unknown Company"
    )
    
    # Extract transcript title (try multiple fields)
    transcript_title = (
        first_record.get("transcript_title") or
        first_record.get("event_title") or
        "Earnings Call Transcript"
    )
    
    return {
        "company_name": company_name,
        "transcript_title": transcript_title
    }


def apply_qa_assignments_to_records(records: List[Dict], qa_groups: List[Dict]) -> List[Dict]:
    """
    ALGORITHM FIX: Apply Q&A group assignments using explicit block lists, not ranges.
    Only adds: qa_group_id, qa_group_confidence, qa_group_method
    """
    
    # ALGORITHM FIX: Create mapping from explicit assignments, not ranges
    block_to_qa_map = {}
    
    for group in qa_groups:
        assigned_block_ids = group.get("assigned_block_ids", [])
        if not assigned_block_ids:
            # Fallback to range if explicit list not available
            for block_id in range(group["start_block_id"], group["end_block_id"] + 1):
                assigned_block_ids.append(block_id)
        
        # Map only explicitly assigned blocks
        for block_id in assigned_block_ids:
            block_to_qa_map[block_id] = {
                "qa_group_id": group["qa_group_id"],
                "qa_group_confidence": group["confidence"],
                "qa_group_method": group["method"]
            }
    
    # Apply to records
    enhanced_records = []
    
    for record in records:
        enhanced_record = record.copy()
        
        # Only apply Q&A assignments to Q&A section records
        if record.get("section_name") == "Q&A":
            speaker_block_id = record.get("speaker_block_id")
            
            if speaker_block_id and speaker_block_id in block_to_qa_map:
                qa_info = block_to_qa_map[speaker_block_id]
                enhanced_record.update(qa_info)
            else:
                # No Q&A group assignment
                enhanced_record.update({
                    "qa_group_id": None,
                    "qa_group_confidence": None,
                    "qa_group_method": None
                })
        else:
            # Non-Q&A sections don't get Q&A assignments
            enhanced_record.update({
                "qa_group_id": None,
                "qa_group_confidence": None,
                "qa_group_method": None
            })
        
        enhanced_records.append(enhanced_record)
    
    return enhanced_records


def process_transcript_qa_pairing(transcript_records: List[Dict], transcript_id: str, enhanced_error_logger: EnhancedErrorLogger) -> Tuple[List[Dict], int]:
    """
    Process a single transcript for Q&A pairing using corrected indexed approach.
    """
    
    try:
        # Refresh OAuth token for each transcript (but only if needed)
        if not refresh_oauth_token_for_transcript():
            log_error(f"Failed to refresh OAuth token for transcript", "authentication", 
                     {"transcript_id": transcript_id})
            enhanced_error_logger.log_authentication_error(f"Token refresh failed for {transcript_id}")
            return transcript_records, 0  # Return original records without Q&A assignments
        
        log_execution(f"Processing Q&A pairing for transcript: {transcript_id}")
        
        # Extract transcript metadata for context
        transcript_metadata = extract_transcript_metadata(transcript_records)
        
        # Group records by speaker blocks with validation
        speaker_blocks = group_records_by_speaker_block(transcript_records)
        
        if not speaker_blocks:
            log_execution(f"No valid speaker blocks found in transcript {transcript_id}")
            return transcript_records, 0
        
        # Process Q&A boundaries using corrected indexed approach
        qa_groups = process_qa_boundaries_indexed(
            speaker_blocks, 
            transcript_id, 
            transcript_metadata, 
            enhanced_error_logger
        )
        
        # Apply Q&A assignments to records using corrected algorithm
        enhanced_records = apply_qa_assignments_to_records(transcript_records, qa_groups)
        
        log_execution(f"Completed Q&A pairing for transcript {transcript_id}: {len(qa_groups)} groups identified")
        
        return enhanced_records, len(qa_groups)
        
    except Exception as e:
        error_msg = f"Transcript Q&A pairing failed: {type(e).__name__}"
        log_error(error_msg, "processing", {"transcript_id": transcript_id, "exception_type": type(e).__name__})
        enhanced_error_logger.log_processing_error(transcript_id, error_msg)
        return transcript_records, 0  # Return original records on failure


def main() -> None:
    """Main function to orchestrate Stage 5 Q&A pairing using corrected indexed approach."""
    global config, logger, llm_client, ssl_cert_path

    # Initialize logging
    logger = setup_logging()
    enhanced_error_logger = EnhancedErrorLogger()
    log_console("=== STAGE 5: Q&A BOUNDARY DETECTION (CORRECTED INDEXED APPROACH) ===")

    # Initialize stage summary
    stage_summary = {
        "status": "unknown",
        "execution_time_seconds": 0,
        "total_transcripts_processed": 0,
        "total_records_processed": 0,
        "total_qa_groups_identified": 0,
        "total_llm_cost": 0.0,
        "total_tokens_used": 0,
        "errors": {
            "environment_validation": 0,
            "nas_connection": 0,
            "config_load": 0,
            "ssl_setup": 0,
            "content_load": 0,
            "authentication": 0,
            "processing": 0,
            "output_save": 0
        }
    }

    start_time = datetime.now()
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
        log_console(f"Loaded corrected indexed Q&A pairing configuration - Model: {config['stage_05_qa_pairing']['llm_config']['model']}")

        # Step 4: SSL certificate setup
        log_console("Step 4: Setting up SSL certificate...")
        ssl_cert_path = setup_ssl_certificate(nas_conn)

        # Step 5: Proxy configuration
        log_console("Step 5: Setting up proxy configuration...")
        proxy_url = setup_proxy_configuration()

        # Step 6: Setup LLM client
        log_console("Step 6: Setting up LLM client...")
        llm_client = setup_llm_client()
        if not llm_client:
            stage_summary["status"] = "failed"
            log_console("Failed to setup LLM client", "ERROR")
            return

        # Step 7: Load Stage 4 content
        log_console("Step 7: Loading Stage 4 validated content...")
        records = load_stage_4_content(nas_conn)
        if not records:
            log_console("No records found to process", "WARNING")
            stage_summary["status"] = "completed_no_content"
            return

        stage_summary["total_records_processed"] = len(records)

        # Step 8: Group records by transcript
        log_console("Step 8: Grouping records by transcript...")
        transcripts = group_records_by_transcript(records)
        log_console(f"Found {len(transcripts)} transcripts to process")

        # Step 9: Development mode handling
        dev_mode = config["stage_05_qa_pairing"].get("dev_mode", False)
        if dev_mode:
            max_transcripts = config["stage_05_qa_pairing"].get("dev_max_transcripts", 2)
            transcript_items = list(transcripts.items())[:max_transcripts]
            transcripts = dict(transcript_items)
            log_console(f"Development mode: Limited to {len(transcripts)} transcripts", "WARNING")

        stage_summary["total_transcripts_processed"] = len(transcripts)

        # Step 10: Process each transcript
        log_console("Step 10: Processing Q&A pairing using corrected indexed approach...")
        all_enhanced_records = []
        total_qa_groups_count = 0

        for i, (transcript_id, transcript_records) in enumerate(transcripts.items(), 1):
            log_console(f"Processing transcript {i}/{len(transcripts)}: {transcript_id}")
            
            enhanced_records, qa_groups_count = process_transcript_qa_pairing(
                transcript_records, transcript_id, enhanced_error_logger
            )
            
            all_enhanced_records.extend(enhanced_records)
            total_qa_groups_count += qa_groups_count

        stage_summary["total_qa_groups_identified"] = total_qa_groups_count
        stage_summary["total_llm_cost"] = enhanced_error_logger.total_cost
        stage_summary["total_tokens_used"] = enhanced_error_logger.total_tokens

        # Step 11: Save output
        log_console("Step 11: Saving Q&A paired content...")
        output_data = {
            "schema_version": "1.1",  # Updated version for corrected approach
            "processing_timestamp": datetime.now().isoformat(),
            "processing_method": "corrected_indexed_approach",
            "total_records": len(all_enhanced_records),
            "total_transcripts_processed": len(transcripts),
            "qa_pairing_summary": {
                "transcripts_with_qa_groups": len([t for t in transcripts.keys() 
                                                 if any(r.get("qa_group_id") is not None 
                                                       for r in all_enhanced_records
                                                       if r.get("filename", "").startswith(t.split("_")[0]))]),
                "total_qa_groups": total_qa_groups_count
            },
            "records": all_enhanced_records
        }

        # Save to NAS following Stage 4 pattern
        output_path = config["stage_05_qa_pairing"]["output_data_path"]
        output_filename = "stage_05_qa_paired_content_indexed_fixed.json"
        output_file_path = nas_path_join(output_path, output_filename)

        nas_create_directory_recursive(nas_conn, output_path)

        output_json = json.dumps(output_data, indent=2)
        output_bytes = io.BytesIO(output_json.encode("utf-8"))

        if nas_upload_file(nas_conn, output_bytes, output_file_path):
            log_execution("Q&A paired content saved successfully", {
                "output_filename": output_filename,
                "total_records": len(all_enhanced_records)
            })
            log_console(f"✅ Output saved successfully: {output_filename}")
        else:
            stage_summary["status"] = "failed"
            log_console("Failed to save output file", "ERROR")
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
        log_console("=== STAGE 5 CORRECTED INDEXED Q&A PAIRING COMPLETE ===")
        log_console(f"Transcripts processed: {stage_summary['total_transcripts_processed']}")
        log_console(f"Total records processed: {stage_summary['total_records_processed']}")
        log_console(f"Q&A groups identified: {stage_summary['total_qa_groups_identified']}")
        log_console(f"Total LLM cost: ${stage_summary['total_llm_cost']:.4f}")
        log_console(f"Total tokens used: {stage_summary['total_tokens_used']:,}")
        log_console(f"Total errors: {sum(stage_summary['errors'].values())}")
        log_console(f"Execution time: {execution_time}")
        log_console(f"Output file: {output_filename}")

    except Exception as e:
        stage_summary["status"] = "failed"
        error_msg = f"Stage 5 corrected indexed Q&A pairing failed: {e}"
        log_console(error_msg, "ERROR")
        log_error(error_msg, "main_execution", {"exception_type": type(e).__name__})

    finally:
        # Save logs to NAS
        if nas_conn:
            try:
                save_logs_to_nas(nas_conn, stage_summary, enhanced_error_logger)
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

        log_console(f"Stage 5 corrected indexed Q&A pairing {stage_summary['status']}")


if __name__ == "__main__":
    main()
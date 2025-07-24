"""
Stage 5: Q&A Boundary Detection & Conversation Pairing
Processes Stage 4 validated content to identify and group question-answer conversation boundaries.
Self-contained standalone script that loads config from NAS at runtime.

Configuration Parameters (in stage_05_qa_pairing section):
- window_size: Number of speaker blocks to analyze at once (default: 10)
- max_held_blocks: Maximum blocks to accumulate before forcing a breakpoint (default: 50)
- max_consecutive_skips: Maximum consecutive skip decisions before forcing a breakpoint (default: 5)
- max_validation_retries_per_position: Retries for validation at same position (default: 2)
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

    def log_boundary_error(self, transcript_id: str, speaker_block_id: int, error: str):
        """Log Q&A boundary detection errors."""
        self.boundary_detection_errors.append({
            "timestamp": datetime.now().isoformat(),
            "transcript_id": transcript_id,
            "speaker_block_id": speaker_block_id,
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
        global logger
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
                filename = f"stage_05_qa_pairing_{error_type}_errors_{timestamp}.json"
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
        "stage": "stage_05_qa_pairing",
        "execution_start": (
            execution_log[0]["timestamp"]
            if execution_log
            else datetime.now().isoformat()
        ),
        "execution_end": datetime.now().isoformat(),
        "summary": stage_summary,
        "execution_log": execution_log,
    }

    main_log_filename = f"stage_05_qa_pairing_{timestamp}.json"
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
            "stage": "stage_05_qa_pairing",
            "execution_time": datetime.now().isoformat(),
            "total_errors": len(error_log),
            "error_summary": stage_summary.get("errors", {}),
            "errors": error_log,
        }

        error_log_filename = f"stage_05_qa_pairing_errors_{timestamp}.json"
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
        "window_size"
    ]

    for param in required_stage_05_qa_pairing_params:
        if param not in stage_05_qa_pairing_config:
            error_msg = f"Missing required stage_05_qa_pairing parameter: {param}"
            log_error(error_msg, "config_validation", {"missing_parameter": f"stage_05_qa_pairing.{param}"})
            raise ValueError(error_msg)

    # Validate LLM config structure
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

    # Validate monitored institutions
    if not config["monitored_institutions"]:
        error_msg = "monitored_institutions cannot be empty"
        log_error(error_msg, "config_validation", {"section": "monitored_institutions"})
        raise ValueError(error_msg)

    log_execution("Configuration validation successful", {
        "total_institutions": len(config["monitored_institutions"]),
        "llm_model": llm_config["model"],
        "approach": "sliding_window",
        "window_size": stage_05_qa_pairing_config["window_size"]
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


def get_oauth_token(retry_count: int = 3, retry_delay: float = 1.0) -> Optional[str]:
    """Get OAuth token using client credentials flow with retry logic."""
    global ssl_cert_path
    
    for attempt in range(1, retry_count + 1):
        try:
            token_endpoint = config["stage_05_qa_pairing"]["llm_config"]["token_endpoint"]
            
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
                    log_error(error_msg, "authentication", {"attempt": attempt})
                    if attempt < retry_count:
                        time.sleep(retry_delay * attempt)  # Exponential backoff
                        continue
                    return None
            else:
                error_msg = f"OAuth token request failed: {response.status_code}"
                log_error(error_msg, "authentication", {"attempt": attempt, "status_code": response.status_code})
                if attempt < retry_count:
                    time.sleep(retry_delay * attempt)  # Exponential backoff
                    continue
                return None
                
        except (requests.ConnectionError, requests.Timeout) as e:
            error_msg = f"Network error during OAuth token acquisition: {e}"
            log_error(error_msg, "authentication", {"attempt": attempt, "exception_type": type(e).__name__})
            if attempt < retry_count:
                time.sleep(retry_delay * attempt)
                continue
            return None
        except Exception as e:
            error_msg = f"OAuth token acquisition failed: {e}"
            log_error(error_msg, "authentication", {"attempt": attempt, "exception_type": type(e).__name__})
            return None
    
    return None


def setup_llm_client() -> Optional[OpenAI]:
    """Setup OpenAI client with custom base URL and OAuth token."""
    global oauth_token
    
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
    global oauth_token, llm_client
    
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
    
    try:
        input_path = config["stage_05_qa_pairing"]["input_data_path"]
        
        log_execution("Loading Stage 4 validated content from NAS", {"input_path": input_path})
        
        content_data = nas_download_file(nas_conn, input_path)
        if not content_data:
            error_msg = f"Stage 4 content file not found at {input_path}"
            log_error(error_msg, "content_load", {"path": input_path})
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
    """Group records by speaker blocks within a transcript."""
    speaker_blocks = defaultdict(list)
    
    # Group paragraphs by speaker block
    for record in records:
        speaker_block_id = record["speaker_block_id"]
        speaker_blocks[speaker_block_id].append(record)
    
    # Create speaker block objects
    speaker_block_list = []
    for block_id in sorted(speaker_blocks.keys()):
        paragraphs = speaker_blocks[block_id]
        
        # Use first paragraph for speaker block metadata
        first_paragraph = paragraphs[0]
        
        speaker_block = {
            "speaker_block_id": block_id,
            "speaker": first_paragraph["speaker"],
            "question_answer_flag": first_paragraph.get("question_answer_flag"),
            "section_name": first_paragraph.get("section_name"),
            "paragraphs": paragraphs
        }
        
        speaker_block_list.append(speaker_block)
    
    return speaker_block_list


def create_breakpoint_detection_prompt(indexed_blocks: List[Dict], company_name: str, transcript_title: str, current_qa_id: int) -> str:
    """
    Create prompt for analyst breakpoint detection in configurable-size sliding windows.
    Instructs LLM to find the LAST block of current analyst turn or skip to next batch.
    """
    
    # Format the indexed blocks for the prompt
    formatted_blocks = []
    for i, block_data in enumerate(indexed_blocks, 1):
        block = block_data["block"]
        formatted_blocks.append(f"""Index {i}:
Speaker: {block['speaker']}
Content: {format_speaker_block_content(block['paragraphs'])}""")
    
    blocks_text = "\n\n".join(formatted_blocks)
    
    return f"""<task>
You are analyzing a {company_name} earnings call Q&A transcript titled "{transcript_title}".
Your task is to find the LAST block that belongs to the current analyst's turn (QA ID {current_qa_id}).
</task>

<context>
You are currently processing QA ID {current_qa_id}. You need to find where this analyst's turn ends - the LAST block that belongs to their conversation.
</context>

<objective>
Examine the {len(indexed_blocks)} indexed speaker blocks below IN SEQUENTIAL ORDER (1, 2, 3, etc.) and determine:

1. SKIP: If the current analyst's turn continues beyond ALL these blocks (no endpoint found in this batch)
2. BREAKPOINT: If you find where the current analyst's turn ENDS - return the index of the LAST block belonging to this analyst

🚨 CRITICAL: You are looking for the LAST block of the current analyst's turn. This includes:
- The analyst's final follow-up question
- The executive's final response to this analyst
- Any operator comment that concludes this analyst's session (e.g., "Thank you")

When you choose SKIP: These {len(indexed_blocks)} blocks will be held and combined with the next batch to find the endpoint.
When you choose BREAKPOINT: All blocks up to AND INCLUDING your chosen index belong to the current analyst.
</objective>

<instructions>
🔍 SEQUENTIAL ANALYSIS PROCESS:
1. Start at Index 1 and read through each block in order
2. Identify where the current analyst's conversation naturally ends
3. Return the index of the LAST block that belongs to the current analyst
4. If the current analyst's turn extends beyond ALL blocks, choose SKIP

<endpoint_indicators>
STRONG endpoint signals (the LAST block of current analyst's turn):
- Executive's final response to the current analyst's questions
- Operator thanking the current analyst (e.g., "Thank you, Mr. Smith")
- Operator transitioning to next analyst (this belongs to CURRENT analyst as their conclusion)
- Current analyst's final thank you or closing remarks

CONTINUE signals (current analyst's turn is still ongoing):
- Analyst asking another follow-up question
- Executive still answering the current analyst
- Back-and-forth dialogue continuing
- Analyst saying "just one more question" or similar
</endpoint_indicators>

<critical_reminders>
🚨 LAST BLOCK: You're finding the LAST block of the current analyst's turn
🚨 INCLUDE TRANSITIONS: Operator comments like "Next question from..." that conclude the current analyst belong to the CURRENT analyst
🚨 COMPLETE SESSIONS: Make sure all of the current analyst's questions have been answered
🚨 BE INCLUSIVE: When in doubt, include transitional blocks with the current analyst
</critical_reminders>
</instructions>

<indexed_speaker_blocks>
{blocks_text}
</indexed_speaker_blocks>

<output_format>
Call the analyst_breakpoint function with:
- "skip" if the current analyst's turn continues beyond these {len(indexed_blocks)} blocks
- "breakpoint" with the index of the LAST block belonging to the current analyst (1-{len(indexed_blocks)})

🚨 REMEMBER: Return the index of the LAST block that belongs to the current analyst's turn. This includes any operator transitions that conclude their session.
</output_format>"""


def create_validation_prompt(proposed_qa_blocks: List[Dict], remaining_blocks: List[Dict], company_name: str, transcript_title: str, qa_id: int) -> str:
    """
    Create prompt for validating analyst qa_id assignments.
    Shows proposed assignment and asks for accept/reject decision.
    """
    
    # Format proposed blocks
    proposed_text = []
    for i, block in enumerate(proposed_qa_blocks, 1):
        proposed_text.append(f"""Block {i}:
Speaker: {block['speaker']}
Content: {format_speaker_block_content(block['paragraphs'])}""")
    
    # Format remaining blocks (up to 5 for context)
    remaining_text = []
    for i, block_data in enumerate(remaining_blocks[:5], 1):
        block = block_data["block"] if isinstance(block_data, dict) and "block" in block_data else block_data
        remaining_text.append(f"""Remaining Block {i}:
Speaker: {block['speaker']}
Content: {format_speaker_block_content(block['paragraphs'])}""")
    
    proposed_blocks_text = "\n\n".join(proposed_text)
    remaining_blocks_text = "\n\n".join(remaining_text) if remaining_text else "No remaining blocks shown."
    
    return f"""<task>
You are validating analyst turn boundaries for {company_name} earnings call "{transcript_title}".
Validate whether the proposed QA ID {qa_id} correctly captures ONE analyst's complete turn.
</task>

<context>
A breakpoint was detected and these speaker blocks have been proposed as QA ID {qa_id}.
You need to validate whether this grouping correctly captures one analyst's complete conversation.
</context>

<objective>
Review the proposed QA ID assignment and determine:
- ACCEPT: The assignment correctly captures one analyst's complete turn from start to finish
- REJECT: The assignment is incorrect (too much, too little, or wrong boundaries)
</objective>

<validation_criteria>
ACCEPT if:
- Contains one analyst's complete question-answer session
- Includes all follow-up questions and responses for this analyst
- Ends at a natural conversation boundary
- Next blocks clearly belong to a different analyst or conversation

REJECT if:
- Cuts off mid-conversation (analyst questions not fully addressed)
- Includes parts of different analysts' sessions
- Stops too early (more questions from same analyst follow)
- Breakpoint seems arbitrary or unnatural
</validation_criteria>

<proposed_qa_id_{qa_id}>
{proposed_blocks_text}
</proposed_qa_id_{qa_id}>

<remaining_context>
{remaining_blocks_text}
</remaining_context>

<output_format>
Call the validate_assignment function with:
- "accept" if QA ID {qa_id} correctly captures the analyst's complete turn
- "reject" if the assignment needs to be reconsidered
</output_format>"""


def format_speaker_block_content(paragraphs: List[Dict], max_length: Optional[int] = None) -> str:
    """Format paragraph content from a speaker block for prompt display."""
    if not paragraphs:
        return "[No content]"
    
    if max_length:
        # For previews, just use first paragraph
        first_content = paragraphs[0].get('paragraph_content', '').strip()
        if len(first_content) > max_length:
            return first_content[:max_length] + "..."
        return first_content
    
    # Full content concatenation
    content_parts = []
    for paragraph in paragraphs:
        content = paragraph.get('paragraph_content', '').strip()
        if content:
            content_parts.append(content)
    return "\n".join(content_parts)


def create_breakpoint_detection_tool():
    """Create function calling tool for analyst endpoint detection in configurable-size windows."""
    
    return [
        {
            "type": "function",  
            "function": {
                "name": "analyst_breakpoint",
                "description": "Find the LAST block that belongs to the current analyst's turn. Return the index of the LAST block of their conversation, or skip if their turn extends beyond this batch.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["skip", "breakpoint"],
                            "description": "skip: The current analyst's turn continues beyond these blocks, examine next batch. breakpoint: Found the LAST block of current analyst's turn."
                        },
                        "index": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Required when action is 'breakpoint'. The 1-based index of the LAST block belonging to the current analyst. This includes any operator transitions that conclude their session."
                        }
                    },
                    "required": ["action"]
                }
            }
        }
    ]


def create_validation_tool():
    """Create function calling tool for validating analyst qa_id assignments."""
    
    return [
        {
            "type": "function",  
            "function": {
                "name": "validate_assignment",
                "description": "Validate whether the proposed qa_id assignment correctly captures one analyst's complete turn from start to finish",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "validation": {
                            "type": "string",
                            "enum": ["accept", "reject"],
                            "description": "accept: The qa_id assignment correctly captures the analyst's complete turn. reject: The assignment is incorrect and breakpoint detection should be repeated."
                        }
                    },
                    "required": ["validation"]
                }
            }
        }
    ]


def calculate_token_cost(prompt_tokens: int, completion_tokens: int) -> dict:
    """Calculate cost based on token usage and configured rates."""
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


def get_blocks_for_speaker_block_ids(qa_speaker_blocks: List[Dict], speaker_block_ids: List[int]) -> List[Dict]:
    """
    Get all speaker blocks that match the specified speaker_block_ids.
    Returns blocks in the order they appear in qa_speaker_blocks.
    """
    matching_blocks = []
    target_ids = set(speaker_block_ids)
    
    for block in qa_speaker_blocks:
        if block.get("speaker_block_id") in target_ids:
            matching_blocks.append(block)
    
    return matching_blocks


def create_indexed_speaker_blocks_from_ids(qa_speaker_blocks: List[Dict], speaker_block_ids: List[int]) -> List[Dict]:
    """
    Create 1-based indexed speaker blocks for LLM processing from speaker_block_ids.
    Maps speaker_block_ids to 1-based indices for LLM comprehension.
    Returns list of dicts with 'index' (1-based), 'speaker_block_id', and 'block' data.
    """
    # Get all blocks for the specified speaker_block_ids
    matching_blocks = get_blocks_for_speaker_block_ids(qa_speaker_blocks, speaker_block_ids)
    
    # Group blocks by speaker_block_id to maintain speaker block boundaries
    blocks_by_speaker_id = defaultdict(list)
    for block in matching_blocks:
        speaker_id = block.get("speaker_block_id")
        blocks_by_speaker_id[speaker_id].append(block)
    
    # Create indexed blocks maintaining 1-based indexing for LLM
    indexed_blocks = []
    for i, speaker_block_id in enumerate(speaker_block_ids, 1):
        if speaker_block_id in blocks_by_speaker_id:
            # All paragraphs for this speaker block get the same index
            for block in blocks_by_speaker_id[speaker_block_id]:
                indexed_block = {
                    "index": i,  # 1-based indexing for LLM
                    "speaker_block_id": speaker_block_id,  # Original speaker block ID
                    "block": block
                }
                indexed_blocks.append(indexed_block)
    
    return indexed_blocks


def detect_analyst_breakpoint(indexed_blocks: List[Dict], 
                            company_name: str, 
                            transcript_title: str, 
                            current_qa_id: int,
                            transcript_id: str,
                            enhanced_error_logger: EnhancedErrorLogger) -> Optional[Dict]:
    """
    Use LLM to detect the LAST block of current analyst's turn in configurable-size window.
    Returns dict with action ("skip" or "breakpoint") and optional index.
    Breakpoint index indicates the LAST block belonging to current analyst.
    """
    global llm_client
    
    try:
        # Create breakpoint detection prompt
        prompt = create_breakpoint_detection_prompt(indexed_blocks, company_name, transcript_title, current_qa_id)
        
        # Retry logic for LLM calls
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(1, max_retries + 1):
            try:
                log_execution(f"Breakpoint detection attempt {attempt}/{max_retries} for QA ID {current_qa_id}")
                
                # Make LLM API call
                response = llm_client.chat.completions.create(
                    model=config["stage_05_qa_pairing"]["llm_config"]["model"],
                    messages=[{"role": "user", "content": prompt}],
                    tools=create_breakpoint_detection_tool(),
                    tool_choice="required", 
                    temperature=config["stage_05_qa_pairing"]["llm_config"]["temperature"],
                    max_tokens=config["stage_05_qa_pairing"]["llm_config"]["max_tokens"]
                )
                
                # Parse response
                
                # Parse response
                if not response.choices[0].message.tool_calls:
                    error_msg = f"No tool call response for breakpoint detection (QA ID {current_qa_id})"
                    log_execution(f"ERROR: {error_msg}")
                    if attempt == max_retries:
                        log_error(f"{error_msg} (final attempt)", "boundary_detection", {})
                        enhanced_error_logger.log_boundary_error(transcript_id, current_qa_id, f"{error_msg} after {max_retries} attempts")
                        return None
                    else:
                        log_execution(f"{error_msg}, retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        continue
                
                # Extract decision
                try:
                    tool_call = response.choices[0].message.tool_calls[0]
                    result = json.loads(tool_call.function.arguments)
                    action = result["action"]
                    
                    if action not in ["skip", "breakpoint"]:
                        raise ValueError(f"Invalid action: {action}")
                    
                    # Validate breakpoint index if provided
                    if action == "breakpoint":
                        if "index" not in result:
                            raise ValueError("Breakpoint action requires index")
                        
                        breakpoint_index = result["index"]
                        if not (1 <= breakpoint_index <= len(indexed_blocks)):
                            raise ValueError(f"Invalid breakpoint index: {breakpoint_index} (must be 1-{len(indexed_blocks)})")
                        
                        parsed_result = {
                            "action": "breakpoint",
                            "index": breakpoint_index
                        }
                    else:
                        parsed_result = {
                            "action": "skip"
                        }
                        
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    error_msg = f"Failed to parse breakpoint detection response: {e}"
                    if attempt == max_retries:
                        log_error(f"{error_msg} (final attempt)", "boundary_detection", {})
                        enhanced_error_logger.log_boundary_error(transcript_id, current_qa_id, f"{error_msg} after {max_retries} attempts")
                        return None
                    else:
                        log_execution(f"{error_msg}, retrying in {retry_delay}s...")
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
                    log_execution(f"Breakpoint detection tokens - input: {response.usage.prompt_tokens}, output: {response.usage.completion_tokens}, cost: ${cost_info['total_cost']:.4f}")
                    
                    # Accumulate costs for final summary
                    enhanced_error_logger.accumulate_costs(token_usage)
                
                # Log successful decision concisely
                attempt_info = f" (attempt {attempt})" if attempt > 1 else ""
                if parsed_result["action"] == "breakpoint":
                    breakpoint_index = parsed_result['index']
                    log_execution(f"Breakpoint detected{attempt_info} at index {breakpoint_index} for QA ID {current_qa_id}")
                else:
                    log_execution(f"Skip decision{attempt_info} for QA ID {current_qa_id}")
                
                return parsed_result
                
            except Exception as e:
                error_msg = f"Breakpoint detection failed: {e}"
                if attempt == max_retries:
                    log_error(f"{error_msg} (final attempt)", "boundary_detection", {})
                    enhanced_error_logger.log_boundary_error(transcript_id, current_qa_id, f"{error_msg} after {max_retries} attempts")
                    return None
                else:
                    log_execution(f"{error_msg}, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
        
        # Should never reach here, but safety fallback
        return None
        
    except Exception as e:
        error_msg = f"Context preparation failed for breakpoint detection: {e}"
        log_error(error_msg, "boundary_detection", {})
        enhanced_error_logger.log_boundary_error(transcript_id, current_qa_id, error_msg)
        return None


def validate_analyst_assignment(proposed_qa_blocks: List[Dict], 
                              remaining_blocks: List[Dict],
                              company_name: str, 
                              transcript_title: str, 
                              qa_id: int,
                              transcript_id: str,
                              enhanced_error_logger: EnhancedErrorLogger) -> bool:
    """
    Use LLM to validate analyst qa_id assignment with 3-attempt cycle.
    Returns True if validated, False if rejected after all attempts.
    """
    global llm_client
    
    max_validation_attempts = 3
    
    for validation_attempt in range(1, max_validation_attempts + 1):
        try:
            log_execution(f"Validation attempt {validation_attempt}/{max_validation_attempts} for QA ID {qa_id}")
            
            # Log validation summary only
            log_execution(f"Validating QA ID {qa_id}: {len(proposed_qa_blocks)} blocks")
            
            # Create validation prompt
            prompt = create_validation_prompt(proposed_qa_blocks, remaining_blocks, company_name, transcript_title, qa_id)
            
            # Create validation prompt
            prompt = create_validation_prompt(proposed_qa_blocks, remaining_blocks, company_name, transcript_title, qa_id)
            
            # Make LLM API call
            response = llm_client.chat.completions.create(
                model=config["stage_05_qa_pairing"]["llm_config"]["model"],
                messages=[{"role": "user", "content": prompt}],
                tools=create_validation_tool(),
                tool_choice="required", 
                temperature=config["stage_05_qa_pairing"]["llm_config"]["temperature"],
                max_tokens=config["stage_05_qa_pairing"]["llm_config"]["max_tokens"]
            )
            
            # Parse validation response
            
            # Parse response
            if not response.choices[0].message.tool_calls:
                error_msg = f"No tool call response for validation (QA ID {qa_id})"
                log_execution(f"ERROR: {error_msg}, attempt {validation_attempt}/{max_validation_attempts}")
                if validation_attempt == max_validation_attempts:
                    log_execution(f"⚠️ Validation failed after {max_validation_attempts} attempts, accepting assignment")
                    return True  # Accept on final failure
                continue
            
            # Extract validation decision
            try:
                tool_call = response.choices[0].message.tool_calls[0]
                result = json.loads(tool_call.function.arguments)
                validation = result["validation"]
                
                if validation not in ["accept", "reject"]:
                    raise ValueError(f"Invalid validation: {validation}")
                
                # Log token usage
                if hasattr(response, 'usage') and response.usage:
                    cost_info = calculate_token_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
                    token_usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                        "cost": cost_info
                    }
                    enhanced_error_logger.accumulate_costs(token_usage)
                
                # Log and return validation result with detailed feedback
                attempt_info = f" (attempt {validation_attempt})" if validation_attempt > 1 else ""
                
                if validation == "accept":
                    log_execution(f"Validation accepted{attempt_info} for QA ID {qa_id}")
                    return True
                else:
                    log_execution(f"Validation rejected{attempt_info} for QA ID {qa_id}")
                    if validation_attempt == max_validation_attempts:
                        log_execution(f"Accepting assignment after {max_validation_attempts} failed attempts")
                        return True
                    else:
                        return False
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                error_msg = f"Failed to parse validation response: {e}"
                log_execution(f"{error_msg}, attempt {validation_attempt}/{max_validation_attempts}")
                if validation_attempt == max_validation_attempts:
                    log_execution(f"Validation parsing failed after {max_validation_attempts} attempts, accepting assignment")
                    return True  # Accept on final failure
                continue
                
        except Exception as e:
            error_msg = f"Validation failed: {e}"
            log_execution(f"{error_msg}, attempt {validation_attempt}/{max_validation_attempts}")
            if validation_attempt == max_validation_attempts:
                log_execution(f"Validation error after {max_validation_attempts} attempts, accepting assignment")
                enhanced_error_logger.log_validation_error(transcript_id, f"QA ID {qa_id} validation failed: {error_msg}")
                return True  # Accept on final failure
            continue
    
    # Fallback - should not reach here
    return True


def process_qa_boundaries_sliding_window(speaker_blocks: List[Dict], transcript_id: str, transcript_metadata: Dict, enhanced_error_logger: EnhancedErrorLogger) -> List[Dict]:
    """
    Sliding window Q&A boundary processing with configurable window size and skip/hold logic.
    
    Process:
    1. Create sliding windows with configurable size (1-based indexing)
    2. LLM decides "skip" (hold blocks) or "breakpoint" (create qa_id)
    3. If breakpoint: validate assignment, advance window from breakpoint
    4. If skip: hold blocks, advance window, combine with next batch
    """
    try:
        # Filter to only Q&A sections and sort by speaker_block_id, then paragraph_id
        qa_speaker_blocks = [block for block in speaker_blocks 
                           if block.get("section_name") == "Q&A"]
        
        if not qa_speaker_blocks:
            log_execution(f"No Q&A sections found in transcript {transcript_id}")
            return []
        
        # Sort by speaker_block_id
        qa_speaker_blocks.sort(key=lambda x: x.get("speaker_block_id", 0))
        
        # Extract unique speaker_block_ids for tracking (maintains original transcript order)
        speaker_block_id_sequence = []
        seen_block_ids = set()
        for block in qa_speaker_blocks:
            block_id = block.get("speaker_block_id")
            if block_id not in seen_block_ids:
                speaker_block_id_sequence.append(block_id)
                seen_block_ids.add(block_id)
        
        log_execution(f"Processing {len(qa_speaker_blocks)} Q&A speaker blocks using sliding window approach")
        log_execution(f"Speaker block ID range: {min(speaker_block_id_sequence)} to {max(speaker_block_id_sequence)} ({len(speaker_block_id_sequence)} unique blocks)")
        
        # Get configuration limits
        window_size = config["stage_05_qa_pairing"].get("window_size", 10)
        max_held_blocks = config["stage_05_qa_pairing"].get("max_held_blocks", 50)  # Memory limit
        max_validation_retries = config["stage_05_qa_pairing"].get("max_validation_retries_per_position", 2)
        max_consecutive_skips = config["stage_05_qa_pairing"].get("max_consecutive_skips", 5)  # Prevent infinite skips
        
        # Initialize sliding window state using speaker_block_ids
        sliding_state = {
            "current_qa_id": 1,
            "current_block_id_index": 0,  # Index in speaker_block_id_sequence
            "held_blocks": [],  # Accumulated blocks from "skip" decisions
            "processing_complete": False,
            "all_qa_groups": [],
            "speaker_block_id_sequence": speaker_block_id_sequence,  # For tracking
            "validation_retries_for_current_position": 0,  # Track validation retries per window position
            "max_validation_retries_per_position": max_validation_retries,
            "consecutive_skips": 0,  # Track consecutive skip decisions
            "stuck_position_count": 0  # Track if we're stuck at same position
        }
        
        # Extract metadata for prompts
        company_name = transcript_metadata.get("company_name", "Unknown Company")
        transcript_title = transcript_metadata.get("transcript_title", "Earnings Call Transcript")
        
        # Main processing loop with enhanced infinite loop protection
        max_iterations = len(speaker_block_id_sequence) * 3  # Allow 3x for retries
        iteration_count = 0
        last_position = -1
        
        while not sliding_state["processing_complete"] and sliding_state["current_block_id_index"] < len(speaker_block_id_sequence):
            iteration_count += 1
            
            # Infinite loop protection
            if iteration_count > max_iterations:
                log_execution(f"INFINITE LOOP DETECTED! Breaking after {iteration_count} iterations")
                enhanced_error_logger.log_processing_error(transcript_id, f"Infinite loop detected after {iteration_count} iterations")
                break
            
            # Check if we're stuck at same position
            if sliding_state["current_block_id_index"] == last_position:
                sliding_state["stuck_position_count"] += 1
                if sliding_state["stuck_position_count"] > 3:
                    log_execution(f"Stuck at position {last_position} for {sliding_state['stuck_position_count']} iterations, force advancing")
                    sliding_state["current_block_id_index"] += 1
                    sliding_state["stuck_position_count"] = 0
            else:
                sliding_state["stuck_position_count"] = 0
                last_position = sliding_state["current_block_id_index"]
            
            # Create current window using speaker_block_ids
            end_block_index = min(sliding_state["current_block_id_index"] + window_size, len(speaker_block_id_sequence))
            current_window_block_ids = speaker_block_id_sequence[sliding_state["current_block_id_index"]:end_block_index]
            
            if not current_window_block_ids:
                # No more blocks to process
                sliding_state["processing_complete"] = True
                break
            
            # Create indexed blocks for LLM (maps speaker_block_ids to 1-based indices)
            current_window = create_indexed_speaker_blocks_from_ids(qa_speaker_blocks, current_window_block_ids)
            
            if not current_window:
                # No blocks found for these IDs
                sliding_state["processing_complete"] = True
                break
            
            log_execution(f"Processing window for QA ID {sliding_state['current_qa_id']} (iteration {iteration_count})")
            
            # Track window advancement to detect infinite loops
            previous_block_id_index = sliding_state['current_block_id_index']
            
            # Phase 1: Breakpoint Detection
            breakpoint_result = detect_analyst_breakpoint(
                current_window,
                company_name,
                transcript_title, 
                sliding_state["current_qa_id"],
                transcript_id,
                enhanced_error_logger
            )
            
            if breakpoint_result is None:
                # LLM call failed - treat as skip to be conservative with consistent structure
                log_execution(f"Breakpoint detection failed: Treating as skip for QA ID {sliding_state['current_qa_id']}")
                
                # Get actual blocks using same method as breakpoint logic for consistency
                current_failed_blocks = get_blocks_for_speaker_block_ids(qa_speaker_blocks, current_window_block_ids)
                sliding_state["held_blocks"].extend(current_failed_blocks)
                
                new_held_count = len(sliding_state["held_blocks"])
                new_block_id_index = sliding_state["current_block_id_index"] + len(current_window_block_ids)
                sliding_state["current_block_id_index"] = new_block_id_index
                
                next_block_id = speaker_block_id_sequence[new_block_id_index] if new_block_id_index < len(speaker_block_id_sequence) else "END"
                log_execution(f"Total held blocks: {new_held_count}, Next speaker block: {next_block_id}")
                continue
            
            if breakpoint_result["action"] == "skip":
                sliding_state["consecutive_skips"] += 1
                
                # Check for too many consecutive skips or memory limit
                if sliding_state["consecutive_skips"] >= max_consecutive_skips:
                    log_execution(f"Reached {max_consecutive_skips} consecutive skips, forcing breakpoint")
                    # Force create a QA group with all held blocks
                    if sliding_state["held_blocks"]:
                        forced_qa_group = {
                            "qa_group_id": sliding_state["current_qa_id"],
                            "start_block_id": sliding_state["held_blocks"][0]["speaker_block_id"],
                            "end_block_id": sliding_state["held_blocks"][-1]["speaker_block_id"],
                            "confidence": 0.7,
                            "method": "sliding_window_forced_skip_limit",
                            "speaker_blocks": sliding_state["held_blocks"]
                        }
                        sliding_state["all_qa_groups"].append(forced_qa_group)
                        log_execution(f"Created forced QA ID {sliding_state['current_qa_id']} with {len(sliding_state['held_blocks'])} blocks")
                        sliding_state["current_qa_id"] += 1
                        sliding_state["held_blocks"] = []
                        sliding_state["consecutive_skips"] = 0
                
                # Get actual blocks using same method as breakpoint logic for consistency
                current_skip_blocks = get_blocks_for_speaker_block_ids(qa_speaker_blocks, current_window_block_ids)
                
                # Check memory limit before adding more blocks
                if len(sliding_state["held_blocks"]) + len(current_skip_blocks) > max_held_blocks:
                    log_execution(f"Held blocks would exceed limit ({max_held_blocks}), forcing breakpoint")
                    # Create QA group with current held blocks
                    if sliding_state["held_blocks"]:
                        memory_limit_qa_group = {
                            "qa_group_id": sliding_state["current_qa_id"],
                            "start_block_id": sliding_state["held_blocks"][0]["speaker_block_id"],
                            "end_block_id": sliding_state["held_blocks"][-1]["speaker_block_id"],
                            "confidence": 0.8,
                            "method": "sliding_window_memory_limit",
                            "speaker_blocks": sliding_state["held_blocks"]
                        }
                        sliding_state["all_qa_groups"].append(memory_limit_qa_group)
                        log_execution(f"Created QA ID {sliding_state['current_qa_id']} with {len(sliding_state['held_blocks'])} blocks (memory limit)")
                        sliding_state["current_qa_id"] += 1
                        sliding_state["held_blocks"] = current_skip_blocks  # Start fresh with current blocks
                        sliding_state["consecutive_skips"] = 1
                else:
                    sliding_state["held_blocks"].extend(current_skip_blocks)
                
                new_held_count = len(sliding_state["held_blocks"])
                new_block_id_index = sliding_state["current_block_id_index"] + len(current_window_block_ids)
                sliding_state["current_block_id_index"] = new_block_id_index
                
                next_block_id = speaker_block_id_sequence[new_block_id_index] if new_block_id_index < len(speaker_block_id_sequence) else "END"
                log_execution(f"Skip action: Held blocks: {new_held_count}/{max_held_blocks}, Consecutive skips: {sliding_state['consecutive_skips']}/{max_consecutive_skips}")
                continue
                
            elif breakpoint_result["action"] == "breakpoint":
                # Found breakpoint - reset consecutive skip counter
                sliding_state["consecutive_skips"] = 0
                
                # Found breakpoint - create qa_id assignment
                breakpoint_index = breakpoint_result["index"]  # 1-based index within current window
                
                # Map breakpoint index to speaker_block_id
                if breakpoint_index <= len(current_window_block_ids):
                    breakpoint_speaker_block_id = current_window_block_ids[breakpoint_index - 1]
                    breakpoint_block_id_index = sliding_state["current_block_id_index"] + breakpoint_index - 1
                else:
                    # Invalid breakpoint index
                    log_execution(f"Invalid breakpoint index {breakpoint_index} > window size {len(current_window_block_ids)}")
                    enhanced_error_logger.log_boundary_error(transcript_id, current_window_block_ids[0], f"Invalid breakpoint index {breakpoint_index}")
                    sliding_state["processing_complete"] = True
                    continue
                
                log_execution(f"Current analyst ends at index {breakpoint_index} (speaker block {breakpoint_speaker_block_id})")
                log_execution(f"Current window speaker blocks: {current_window_block_ids}, held blocks count: {len(sliding_state['held_blocks'])}")
                
                # Get speaker_block_ids up to AND including breakpoint for current qa_id
                # breakpoint_index is 1-based, so if it's 3, we want indices 0,1,2 (blocks 1,2,3)
                current_qa_block_ids = current_window_block_ids[:breakpoint_index]
                current_window_blocks = get_blocks_for_speaker_block_ids(qa_speaker_blocks, current_qa_block_ids)
                proposed_qa_blocks = sliding_state["held_blocks"] + current_window_blocks
                
                # Remaining speaker_block_ids in window (for validation context)
                remaining_block_ids = current_window_block_ids[breakpoint_index:]
                remaining_window_blocks = get_blocks_for_speaker_block_ids(qa_speaker_blocks, remaining_block_ids)
                
                # Prepare for validation
                
                if not proposed_qa_blocks:
                    # Edge case: no blocks for this qa_id 
                    # This can happen when breakpoint is at index 1 with no held blocks
                    log_execution(f"WARNING: No blocks proposed for QA ID {sliding_state['current_qa_id']} (breakpoint at index {breakpoint_index}, held blocks: {len(sliding_state['held_blocks'])})")
                    # Don't create an empty group - just advance to after the breakpoint
                    sliding_state["current_block_id_index"] = breakpoint_block_id_index + 1
                    # Don't increment QA ID since we didn't create a group
                    continue
                else:
                    # Phase 2: Validation (3 attempts)
                    validated = validate_analyst_assignment(
                        proposed_qa_blocks,
                        remaining_window_blocks,
                        company_name,
                        transcript_title,
                        sliding_state["current_qa_id"],
                        transcript_id,
                        enhanced_error_logger
                    )
                
                if validated:
                    # Validation succeeded - create qa_group and advance
                    if proposed_qa_blocks:  # Only create group if we have blocks
                        qa_group = {
                            "qa_group_id": sliding_state["current_qa_id"],
                            "start_block_id": proposed_qa_blocks[0]["speaker_block_id"],
                            "end_block_id": proposed_qa_blocks[-1]["speaker_block_id"],
                            "confidence": 1.0,  # High confidence in sliding window + validation
                            "method": "sliding_window_validated",
                            "speaker_blocks": proposed_qa_blocks
                        }
                        sliding_state["all_qa_groups"].append(qa_group)
                        log_execution(f"Created QA ID {sliding_state['current_qa_id']} with {len(proposed_qa_blocks)} blocks")
                    
                    # Reset validation retry counter and advance to next qa_id
                    sliding_state["validation_retries_for_current_position"] = 0
                    sliding_state["current_qa_id"] += 1
                    sliding_state["held_blocks"] = []  # Clear held blocks
                    
                    # Advance window to after the breakpoint (next analyst starts after current one ends)
                    sliding_state["current_block_id_index"] = breakpoint_block_id_index + 1
                    
                    # Window advanced
                    
                else:
                    # Validation failed - check retry limit before retrying
                    sliding_state["validation_retries_for_current_position"] += 1
                    
                    if sliding_state["validation_retries_for_current_position"] >= sliding_state["max_validation_retries_per_position"]:
                        # Exceeded retry limit - create qa_group anyway with lower confidence and advance
                        log_execution(f"Validation failed {sliding_state['validation_retries_for_current_position']} times, creating group with lower confidence")
                        
                        if proposed_qa_blocks:  # Only create group if we have blocks
                            qa_group = {
                                "qa_group_id": sliding_state["current_qa_id"],
                                "start_block_id": proposed_qa_blocks[0]["speaker_block_id"],
                                "end_block_id": proposed_qa_blocks[-1]["speaker_block_id"],
                                "confidence": 0.5,  # Lower confidence due to validation failure
                                "method": "sliding_window_unvalidated",
                                "speaker_blocks": proposed_qa_blocks
                            }
                            sliding_state["all_qa_groups"].append(qa_group)
                            log_execution(f"Created QA ID {sliding_state['current_qa_id']} with {len(proposed_qa_blocks)} blocks (unvalidated)")
                        
                        # Reset validation retry counter and advance to next qa_id
                        sliding_state["validation_retries_for_current_position"] = 0
                        sliding_state["current_qa_id"] += 1
                        sliding_state["held_blocks"] = []  # Clear held blocks
                        
                        # Advance window to after the breakpoint (next analyst starts after current one ends)
                        sliding_state["current_block_id_index"] = breakpoint_block_id_index + 1
                        
                        # Force advanced window
                    
                    else:
                        # Still have retries left - retry breakpoint detection on same window
                        log_execution(f"Validation failed (retry {sliding_state['validation_retries_for_current_position']}/{sliding_state['max_validation_retries_per_position']}), retrying")
                        # Continue with same window state - will retry breakpoint detection
                    
            else:
                # Invalid action (shouldn't happen with validation)
                log_execution(f"Invalid breakpoint action: {breakpoint_result.get('action')}")
                sliding_state["processing_complete"] = True
            
            # Check for window advancement - critical for preventing infinite loops
            if sliding_state['current_block_id_index'] == previous_block_id_index and not sliding_state["processing_complete"]:
                current_block_id = speaker_block_id_sequence[previous_block_id_index] if previous_block_id_index < len(speaker_block_id_sequence) else "END"
                log_execution(f"WARNING: Window did not advance from index {previous_block_id_index}")
                
                # Force advancement to prevent infinite loop
                if sliding_state['current_block_id_index'] + len(current_window_block_ids) >= len(speaker_block_id_sequence):
                    log_execution("Reached end of speaker blocks")
                    sliding_state["processing_complete"] = True
                else:
                    force_advance_index = sliding_state['current_block_id_index'] + len(current_window_block_ids)
                    next_block_id = speaker_block_id_sequence[force_advance_index] if force_advance_index < len(speaker_block_id_sequence) else "END"
                    log_execution(f"Force advancing window to prevent infinite loop")
                    sliding_state['current_block_id_index'] = force_advance_index
        
        # Handle any remaining held blocks as final qa_id
        if sliding_state["held_blocks"]:
            final_qa_group = {
                "qa_group_id": sliding_state["current_qa_id"],
                "start_block_id": sliding_state["held_blocks"][0]["speaker_block_id"],
                "end_block_id": sliding_state["held_blocks"][-1]["speaker_block_id"],
                "confidence": 1.0,
                "method": "sliding_window_final",
                "speaker_blocks": sliding_state["held_blocks"]
            }
            sliding_state["all_qa_groups"].append(final_qa_group)
            log_execution(f"Created final QA ID {sliding_state['current_qa_id']} with {len(sliding_state['held_blocks'])} remaining blocks")
        
        log_execution(f"Sliding window processing complete - Created {len(sliding_state['all_qa_groups'])} Q&A groups in {iteration_count} iterations")
        
        # Check for gaps in speaker block coverage
        all_assigned_blocks = set()
        for group in sliding_state["all_qa_groups"]:
            for block in group.get("speaker_blocks", []):
                all_assigned_blocks.add(block.get("speaker_block_id"))
        
        # Find gaps
        all_qa_block_ids = set(b.get("speaker_block_id") for b in qa_speaker_blocks)
        missing_blocks = all_qa_block_ids - all_assigned_blocks
        
        if missing_blocks:
            log_execution(f"WARNING: {len(missing_blocks)} speaker blocks not assigned to any QA group: {sorted(missing_blocks)[:10]}{'...' if len(missing_blocks) > 10 else ''}")
            # Log details about missing blocks
            for block_id in sorted(missing_blocks)[:5]:
                # Find the speaker block
                for block in qa_speaker_blocks:
                    if block.get("speaker_block_id") == block_id:
                        log_execution(f"  Missing block {block_id}: Speaker={block.get('speaker')}, Paragraphs={len(block.get('paragraphs', []))}")
                        break
            
        log_execution(f"Successfully identified {len(sliding_state['all_qa_groups'])} Q&A groups covering {len(all_assigned_blocks)} of {len(all_qa_block_ids)} speaker blocks")
        return sliding_state["all_qa_groups"]
        
    except Exception as e:
        error_msg = f"Sliding window Q&A boundary processing failed: {e}"
        log_error(error_msg, "processing", {})
        enhanced_error_logger.log_processing_error(transcript_id, error_msg)
        return []


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
    Apply Q&A group assignments to paragraph records.
    Only adds: qa_group_id, qa_group_confidence, qa_group_method
    """
    
    # Create mapping from speaker block ID to Q&A group info
    block_to_qa_map = {}
    
    log_execution(f"Applying Q&A assignments from {len(qa_groups)} groups to {len(records)} records")
    
    # Log QA groups summary for debugging missing assignments
    all_covered_speaker_ids = set()
    for group in qa_groups:
        speaker_blocks = group.get("speaker_blocks", [])
        block_ids = [b.get("speaker_block_id") for b in speaker_blocks]
        all_covered_speaker_ids.update(block_ids)
        if block_ids:
            log_execution(f"QA Group {group['qa_group_id']}: {len(speaker_blocks)} blocks, speaker IDs {min(block_ids)}-{max(block_ids)}")
    
    for group in qa_groups:
        # Get all speaker block IDs from the group's speaker_blocks
        speaker_blocks = group.get("speaker_blocks", [])
        qa_id = group["qa_group_id"]
        
        # Process all speaker blocks in this QA group
        
        block_ids_in_group = []
        for speaker_block in speaker_blocks:
            block_id = speaker_block.get("speaker_block_id")
            if block_id is not None:
                block_to_qa_map[block_id] = {
                    "qa_group_id": group["qa_group_id"],
                    "qa_group_confidence": group["confidence"],
                    "qa_group_method": group["method"]
                }
                block_ids_in_group.append(block_id)
            else:
                log_execution(f"WARNING: Speaker block missing speaker_block_id in QA Group {qa_id}")
    
    # Apply to records
    enhanced_records = []
    qa_records_count = 0
    assigned_records_count = 0
    unassigned_speaker_block_ids = []
    
    log_execution(f"Total speaker block IDs covered by QA groups: {len(all_covered_speaker_ids)}")
    
    for record in records:
        enhanced_record = record.copy()
        
        # Only apply Q&A assignments to Q&A section records
        if record.get("section_name") == "Q&A":
            qa_records_count += 1
            speaker_block_id = record.get("speaker_block_id")
            
            if speaker_block_id is not None and speaker_block_id in block_to_qa_map:
                qa_info = block_to_qa_map[speaker_block_id]
                enhanced_record.update(qa_info)
                assigned_records_count += 1
            else:
                # No Q&A group assignment
                enhanced_record.update({
                    "qa_group_id": None,
                    "qa_group_confidence": None,
                    "qa_group_method": None
                })
                if speaker_block_id is not None and speaker_block_id not in block_to_qa_map:
                    if speaker_block_id not in unassigned_speaker_block_ids:
                        unassigned_speaker_block_ids.append(speaker_block_id)
        else:
            # Non-Q&A sections don't get Q&A assignments
            enhanced_record.update({
                "qa_group_id": None,
                "qa_group_confidence": None,
                "qa_group_method": None
            })
        
        enhanced_records.append(enhanced_record)
    
    if qa_records_count > 0:
        log_execution(f"Q&A assignment complete - Records: {qa_records_count}, Assigned: {assigned_records_count}, Unassigned: {qa_records_count - assigned_records_count}")
        if unassigned_speaker_block_ids:
            log_execution(f"WARNING: {len(unassigned_speaker_block_ids)} speaker blocks have no QA assignment: {sorted(unassigned_speaker_block_ids)[:10]}{'...' if len(unassigned_speaker_block_ids) > 10 else ''}")
    
    return enhanced_records


def process_transcript_qa_pairing(transcript_records: List[Dict], transcript_id: str, enhanced_error_logger: EnhancedErrorLogger) -> Tuple[List[Dict], int]:
    """
    Process a single transcript for Q&A pairing using state-based approach with per-transcript OAuth refresh.
    """
    
    try:
        # Refresh OAuth token for each transcript
        if not refresh_oauth_token_for_transcript():
            log_error(f"Failed to refresh OAuth token for transcript {transcript_id}", "authentication", {"transcript_id": transcript_id})
            enhanced_error_logger.log_authentication_error(f"Token refresh failed for {transcript_id}")
            # Track authentication error in stage summary
            global error_log
            error_log.append({
                "timestamp": datetime.now().isoformat(),
                "error_type": "authentication",
                "message": f"OAuth token refresh failed for transcript {transcript_id}",
                "details": {"transcript_id": transcript_id}
            })
            return transcript_records, 0  # Return original records without Q&A assignments
        
        log_execution(f"Processing Q&A pairing for transcript: {transcript_id}")
        
        # Extract transcript metadata for context
        transcript_metadata = extract_transcript_metadata(transcript_records)
        
        # Group records by speaker blocks
        speaker_blocks = group_records_by_speaker_block(transcript_records)
        
        # Process Q&A boundaries using sliding window approach
        qa_groups = process_qa_boundaries_sliding_window(
            speaker_blocks, 
            transcript_id, 
            transcript_metadata, 
            enhanced_error_logger
        )
        
        # Apply Q&A assignments to records
        enhanced_records = apply_qa_assignments_to_records(transcript_records, qa_groups)
        
        log_execution(f"Completed Q&A pairing for transcript {transcript_id}: {len(qa_groups)} groups identified")
        
        return enhanced_records, len(qa_groups)
        
    except Exception as e:
        error_msg = f"Transcript Q&A pairing failed: {e}"
        log_error(error_msg, "processing", {"transcript_id": transcript_id, "exception_type": type(e).__name__})
        enhanced_error_logger.log_processing_error(transcript_id, error_msg)
        return transcript_records, 0  # Return original records on failure


def main() -> None:
    """Main function to orchestrate Stage 5 Q&A pairing."""
    global config, logger, llm_client, ssl_cert_path

    # Initialize logging
    logger = setup_logging()
    enhanced_error_logger = EnhancedErrorLogger()
    log_console("=== STAGE 5: Q&A BOUNDARY DETECTION & CONVERSATION PAIRING ===")

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
        log_console(f"Loaded Q&A pairing configuration - Model: {config['stage_05_qa_pairing']['llm_config']['model']}")

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
        log_console("Step 10: Processing Q&A pairing...")
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

        # Step 11: Save output (just records array, no metadata wrapper)
        log_console("Step 11: Saving Q&A paired content...")

        # Save to NAS following Stage 4 pattern
        output_path = config["stage_05_qa_pairing"]["output_data_path"]
        output_filename = "stage_05_qa_paired_content.json"
        output_file_path = nas_path_join(output_path, output_filename)

        nas_create_directory_recursive(nas_conn, output_path)

        # Output just the records array, no metadata wrapper
        output_json = json.dumps(all_enhanced_records, indent=2)
        output_bytes = io.BytesIO(output_json.encode("utf-8"))

        if nas_upload_file(nas_conn, output_bytes, output_file_path):
            log_execution("Q&A paired content saved successfully", {
                "output_path": output_file_path,
                "total_records": len(all_enhanced_records)
            })
            log_console(f"Output saved successfully: {output_filename}")
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
        log_console("=== STAGE 5 Q&A PAIRING COMPLETE ===")
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
        error_msg = f"Stage 5 Q&A pairing failed: {e}"
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

        log_console(f"Stage 5 Q&A pairing {stage_summary['status']}")


if __name__ == "__main__":
    main()
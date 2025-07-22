"""
Stage 5: Q&A Boundary Detection & Conversation Pairing
Processes Stage 4 validated content to identify and group question-answer conversation boundaries.
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
                filename = f"stage_05_qa_pairing_pairing_{error_type}_errors_{timestamp}.json"
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
        "stage": "stage_05_qa_pairing_pairing",
        "execution_start": (
            execution_log[0]["timestamp"]
            if execution_log
            else datetime.now().isoformat()
        ),
        "execution_end": datetime.now().isoformat(),
        "summary": stage_summary,
        "execution_log": execution_log,
    }

    main_log_filename = f"stage_05_qa_pairing_pairing_{timestamp}.json"
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
            "stage": "stage_05_qa_pairing_pairing",
            "execution_time": datetime.now().isoformat(),
            "total_errors": len(error_log),
            "error_summary": stage_summary.get("errors", {}),
            "errors": error_log,
        }

        error_log_filename = f"stage_05_qa_pairing_pairing_errors_{timestamp}.json"
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
        "window_config"
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

    # Validate window config structure
    window_config = stage_05_qa_pairing_config["window_config"]
    required_window_params = ["context_blocks_before", "context_blocks_after"]

    for param in required_window_params:
        if param not in window_config:
            error_msg = f"Missing required window config parameter: {param}"
            log_error(error_msg, "config_validation", {"missing_parameter": f"window_config.{param}"})
            raise ValueError(error_msg)

    # Validate monitored institutions
    if not config["monitored_institutions"]:
        error_msg = "monitored_institutions cannot be empty"
        log_error(error_msg, "config_validation", {"section": "monitored_institutions"})
        raise ValueError(error_msg)

    log_execution("Configuration validation successful", {
        "total_institutions": len(config["monitored_institutions"]),
        "llm_model": llm_config["model"],
        "window_config": window_config
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


def get_oauth_token() -> Optional[str]:
    """Get OAuth token using client credentials flow."""
    global ssl_cert_path
    
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
                log_error(error_msg, "authentication", {})
                return None
        else:
            error_msg = f"OAuth token request failed: {response.status_code} - {response.text}"
            log_error(error_msg, "authentication", {})
            return None
            
    except Exception as e:
        error_msg = f"OAuth token acquisition failed: {e}"
        log_error(error_msg, "authentication", {"exception_type": type(e).__name__})
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


# Placeholder for all the Q&A processing functions from original script
# This is a simplified version - the full implementation would include all the
# LLM boundary detection, speaker block analysis, etc. from the original

def create_speaker_block_window(current_block_index: int, 
                               speaker_blocks: List[Dict],
                               qa_state: Dict = None) -> List[Dict]:
    """
    Create optimized context window for Q&A boundary detection.
    
    Expands window strategically to capture complete question-answer exchanges.
    """
    window_config = config["stage_05_qa_pairing"]["window_config"]
    base_context_before = window_config["context_blocks_before"]
    base_context_after = window_config["context_blocks_after"]
    
    # For active Q&A groups, expand context to see full exchange
    if qa_state and qa_state.get("group_active"):
        question_start_index = qa_state.get("question_start_index", 0)
        # Always include from question start for better context
        start_index = max(0, question_start_index - 1)  # Include one block before question start
        
        # Expand forward context to see potential exchange continuation
        context_after = base_context_after + 1  # Look ahead for exchange completion
    else:
        # Standard window for new exchange detection
        start_index = max(0, current_block_index - base_context_before)
        context_after = base_context_after
    
    end_index = min(len(speaker_blocks), current_block_index + context_after + 1)
    
    return speaker_blocks[start_index:end_index]


def format_speaker_block_context(window_blocks: List[Dict], current_block_index: int) -> str:
    """
    Format speaker blocks with enhanced focus on Q&A exchange flow.
    """
    context_sections = []
    current_block_id = window_blocks[current_block_index]["speaker_block_id"]
    
    # 1. CONVERSATION HISTORY
    prior_blocks = [b for b in window_blocks if b["speaker_block_id"] < current_block_id]
    if prior_blocks:
        context_sections.append("=== CONVERSATION HISTORY ===")
        for block in prior_blocks:
            context_sections.append(format_single_speaker_block(block))
    
    # 2. CURRENT DECISION POINT
    current_block = next(b for b in window_blocks if b["speaker_block_id"] == current_block_id)
    context_sections.append("=== 🎯 DECISION POINT: CURRENT SPEAKER BLOCK ===")
    context_sections.append("**MAKE BOUNDARY DECISION FOR THIS BLOCK:**")
    context_sections.append(format_single_speaker_block(current_block))
    
    # 3. UPCOMING SPEAKERS (for context)
    future_blocks = [b for b in window_blocks if b["speaker_block_id"] > current_block_id]
    if future_blocks:
        context_sections.append("=== UPCOMING SPEAKERS (for context) ===")
        for block in future_blocks:
            context_sections.append(format_single_speaker_block(block))
    
    return "\n\n".join(context_sections)


def format_single_speaker_block(block: Dict) -> str:
    """Format a single speaker block using actual speaker information from the data."""
    paragraphs_text = []
    
    for para in block["paragraphs"]:
        # Clean formatting with clear paragraph separation
        content = para['paragraph_content'].strip()
        paragraphs_text.append(f"    • {content}")
    
    # Use the actual speaker field as provided in the data
    speaker = block['speaker']
    
    return f"""
SPEAKER BLOCK {block['speaker_block_id']}:
  Speaker: {speaker}
  Content:
{chr(10).join(paragraphs_text)}"""


def create_qa_boundary_prompt(formatted_context: str, current_block_id: int, qa_state: Dict = None) -> str:
    """
    Create focused prompt for Q&A boundary detection: analyst questions → complete executive responses.
    """
    # Extract state information
    current_group_active = qa_state.get("group_active", False) if qa_state else False
    current_group_id = qa_state.get("current_qa_group_id") if qa_state else None
    
    # Determine current objective
    if current_group_active:
        objective = f"Determine if current block CONTINUES or ENDS Q&A group {current_group_id}"
        your_options = "CONTINUE (same Q&A exchange) or END (Q&A exchange complete)"
    else:
        objective = "Determine if current block STARTS a new Q&A exchange"  
        your_options = "START (analyst begins question) or STANDALONE (operator/filler)"
    
    return f"""**PURPOSE**: Group sequential Q&A exchanges where analyst questions get complete executive responses.

**CURRENT STATE**: Group {"ACTIVE" if current_group_active else "INACTIVE"} | Group ID: {current_group_id or "None"}
**YOUR TASK**: {objective}
**YOUR OPTIONS**: {your_options}

{formatted_context}

**CORE LOGIC** (Priority Order):

1. **OPERATORS/MODERATORS** → Always STANDALONE (ignore call management)
   - Content like: "Next question comes from...", "Our next caller is...", "Thank you, next question..."

2. **Q&A GROUP STARTS** → When someone begins asking a question:
   - Question content: "Can you walk us through...", "What's your outlook on...", "My question is about..."
   - Multiple speakers can contribute to same question
   - START even if operator transitions occurred just before

3. **Q&A GROUP CONTINUES** → When the same exchange is ongoing:
   - Response content: "Absolutely. Let me break this down...", "To answer your question..."
   - Follow-up content: "And also...", "Just to clarify...", "Let me add to that..."
   - Multiple speakers can contribute to same response
   - Same exchange continues until fully answered

4. **Q&A GROUP ENDS** → When the exchange is complete:
   - Response concludes AND next speaker starts different question
   - Response concludes AND operator transitions to next question
   - Brief closing: "Thank you", "Got it", "That's helpful"

**EXAMPLES**:
- Block 10: "Could you provide more color on margins?" → START (question begins)
- Block 11: "Absolutely. Let me break this down..." → CONTINUE (response begins)  
- Block 12: "First quarter margins expanded 200bp..." → CONTINUE (response continues)
- Block 13: "And we expect this trend to persist..." → CONTINUE (response continues)
- Block 14: "Thank you" → END (question fully answered)
- Block 15: "Next question comes from John at Bank ABC..." → STANDALONE (ignore operator)
- Block 16: "My question is about revenue guidance..." → START (new question)

**KEY INSIGHT**: Each Q&A group = ONE complete question-answer cycle. Focus on CONTENT patterns, not speaker titles."""


def create_qa_boundary_detection_tools(qa_state: Dict = None):
    """Create focused function calling tools for Q&A boundary detection."""
    
    # Extract state information
    current_group_active = qa_state.get("group_active", False) if qa_state else False
    current_group_id = qa_state.get("current_qa_group_id") if qa_state else None
    
    # Create dynamic enum and description based on current state
    if current_group_active:
        status_enum = ["group_continue", "group_end", "standalone"]
        description = f"Q&A Group {current_group_id} is ACTIVE. Decide: CONTINUE (same exchange), END (exchange complete), or STANDALONE (operator)"
    else:
        status_enum = ["group_start", "standalone"]
        description = "No active Q&A group. Decide: START (analyst begins question) or STANDALONE (operator/filler)"
    
    return [
        {
            "type": "function",
            "function": {
                "name": "determine_qa_group_boundary",
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "qa_group_decision": {
                            "type": "object",
                            "properties": {
                                "current_block_id": {"type": "integer"},
                                "group_status": {
                                    "type": "string",
                                    "enum": status_enum
                                },
                                "confidence_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "reasoning": {
                                    "type": "string", 
                                    "maxLength": 150,
                                    "description": "Concise reasoning focusing on: speaker role, question/answer pattern, exchange status"
                                }
                            },
                            "required": ["current_block_id", "group_status", "confidence_score", "reasoning"]
                        }
                    },
                    "required": ["qa_group_decision"]
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


def is_operator_block(speaker_block: Dict) -> bool:
    """
    Detect if a speaker block contains operator/moderator content that should be excluded from Q&A assignments.
    Focuses on content patterns rather than speaker name inference.
    """
    # Get all content from the speaker block
    all_content = ""
    for para in speaker_block.get("paragraphs", []):
        content = para.get("paragraph_content", "")
        all_content += " " + content.lower()
    
    # Operator/moderator content patterns - these indicate call management rather than Q&A content
    operator_content_patterns = [
        "next question", "our next question", "next caller", "next participant",
        "question comes from", "question is from", "caller is from",
        "please hold", "please stand by", "one moment please",
        "we have no further questions", "no more questions", "end of q&a",
        "concludes our question", "concludes the q&a", "end of our q&a",
        "thank you for joining", "this concludes", "that concludes",
        "we will now take", "we'll take the next"
    ]
    
    # If content contains operator patterns, mark as operator
    for pattern in operator_content_patterns:
        if pattern in all_content:
            return True
    
    # Special case: "thank you" + transition patterns indicate operator
    if "thank you" in all_content and any(transition in all_content for transition in ["next question", "next caller", "question comes from", "question is from"]):
        return True
    
    return False


def assign_group_id_to_decision(decision: Dict, qa_state: Dict, current_block_id: int) -> Dict:
    """
    Automatically assign appropriate group ID based on decision status and current state.
    """
    status = decision.get("group_status")
    current_group_active = qa_state.get("group_active", False)
    current_group_id = qa_state.get("current_qa_group_id")
    next_group_id = qa_state.get("next_group_id", 1)
    
    # Create enhanced decision with proper group ID
    enhanced_decision = decision.copy()
    
    if status == "group_start":
        # Assign next available group ID
        enhanced_decision["qa_group_id"] = next_group_id
        qa_state["next_group_id"] = next_group_id + 1
        qa_state["assigned_group_ids"].add(next_group_id)
        
    elif status in ["group_continue", "group_end"]:
        # Use current active group ID
        if current_group_active and current_group_id:
            enhanced_decision["qa_group_id"] = current_group_id
        else:
            # This shouldn't happen with proper validation, but handle gracefully
            enhanced_decision["qa_group_id"] = None
            
    elif status == "standalone":
        # Operator/non-Q&A blocks get null ID
        enhanced_decision["qa_group_id"] = None
    
    return enhanced_decision


def validate_llm_decision(decision: Dict, qa_state: Dict, current_block_id: int) -> str:
    """
    Validate LLM decision against current state. Returns error message if invalid, None if valid.
    Note: We no longer validate group IDs since they're auto-assigned.
    """
    current_group_active = qa_state.get("group_active", False)
    status = decision.get("group_status")
    
    # Validate status based on current state
    if current_group_active:
        # Active group: can only continue, end, or standalone
        if status not in ["group_continue", "group_end", "standalone"]:
            return f"Invalid status '{status}' for active group state"
    else:
        # No active group: can only start or standalone
        if status not in ["group_start", "standalone"]:
            return f"Invalid status '{status}' for no active group state"
    
    return None  # Valid decision


def analyze_speaker_block_boundary(current_block_index: int, 
                                 speaker_blocks: List[Dict],
                                 transcript_id: str,
                                 qa_state: Dict = None,
                                 enhanced_error_logger: EnhancedErrorLogger = None) -> Optional[Dict]:
    """
    Analyze a single speaker block for Q&A boundary decisions using LLM.
    Handles operator blocks with special exclusion logic.
    """
    global llm_client
    
    try:
        current_block = speaker_blocks[current_block_index]
        current_block_id = current_block["speaker_block_id"]
        
        # Check if this is an operator/moderator block - exclude from Q&A assignments
        if is_operator_block(current_block):
            log_execution(f"Block {current_block_id} identified as operator/moderator block - excluding from Q&A groups")
            return {
                "current_block_id": current_block_id,
                "qa_group_id": None,  # No Q&A group assignment
                "qa_group_start_block": None,
                "qa_group_end_block": None,
                "group_status": "standalone",  # Operator blocks are standalone
                "confidence_score": 1.0,  # High confidence in operator detection
                "reasoning": f"Operator/moderator content detected - excluded from Q&A assignments"
            }
        
        # Create context window for non-operator blocks
        window_blocks = create_speaker_block_window(current_block_index, speaker_blocks, qa_state)
        
        # Format context for LLM
        formatted_context = format_speaker_block_context(window_blocks, 
                                                       next(i for i, b in enumerate(window_blocks) 
                                                           if b["speaker_block_id"] == current_block_id))
        
        # Create prompt with state awareness
        prompt = create_qa_boundary_prompt(formatted_context, current_block_id, qa_state)
        
        # Make LLM API call with dynamic tools
        response = llm_client.chat.completions.create(
            model=config["stage_05_qa_pairing"]["llm_config"]["model"],
            messages=[{"role": "user", "content": prompt}],
            tools=create_qa_boundary_detection_tools(qa_state),
            tool_choice="required",
            temperature=config["stage_05_qa_pairing"]["llm_config"]["temperature"],
            max_tokens=config["stage_05_qa_pairing"]["llm_config"]["max_tokens"]
        )
        
        # Parse and validate response
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            result = json.loads(tool_call.function.arguments)
            decision = result["qa_group_decision"]
            
            # Validate decision against current state
            validation_error = validate_llm_decision(decision, qa_state, current_block_id)
            if validation_error:
                log_error(f"Block {current_block_id}: Invalid LLM decision - {validation_error}", "boundary_detection", {})
                enhanced_error_logger.log_boundary_error(transcript_id, current_block_id, f"Invalid LLM decision: {validation_error}")
                return None
            
            # Automatically assign appropriate group ID
            decision = assign_group_id_to_decision(decision, qa_state, current_block_id)
            
            # Log token usage and cost
            if hasattr(response, 'usage') and response.usage:
                cost_info = calculate_token_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "cost": cost_info
                }
                log_execution(f"Block {current_block_id} tokens - input: {response.usage.prompt_tokens}, output: {response.usage.completion_tokens}, total: {response.usage.total_tokens}, cost: ${cost_info['total_cost']:.4f}")
                
                # Accumulate costs for final summary
                enhanced_error_logger.accumulate_costs(token_usage)
            
            log_execution(f"Block {current_block_id} boundary decision: {decision['group_status']} (group: {decision.get('qa_group_id')}, confidence: {decision['confidence_score']:.2f}) | Reasoning: {decision.get('reasoning', 'No reasoning provided')}")
            
            return decision
        else:
            error_msg = f"No tool call response for block {current_block_id}"
            log_error(error_msg, "boundary_detection", {})
            enhanced_error_logger.log_boundary_error(transcript_id, current_block_id, error_msg)
            return None
            
    except Exception as e:
        error_msg = f"LLM boundary analysis failed for block {current_block_id}: {e}"
        log_error(error_msg, "boundary_detection", {})
        enhanced_error_logger.log_boundary_error(transcript_id, current_block_id, error_msg)
        return None


def calculate_qa_group_confidence(block_decisions: List[Dict]) -> float:
    """
    Calculate aggregated confidence for Q&A group spanning multiple blocks.
    
    Strategy: Use weighted average based on decision criticality
    - Group start/end decisions: Higher weight (0.4 each)
    - Group continuation decisions: Lower weight (0.2 / num_continuations)
    """
    if len(block_decisions) == 1:
        return block_decisions[0]["confidence_score"]
    
    weighted_confidence = 0.0
    total_weight = 0.0
    
    continuation_count = len([d for d in block_decisions if d["group_status"] == "group_continue"])
    
    for decision in block_decisions:
        if decision["group_status"] in ["group_start", "group_end"]:
            weight = 0.4  # Critical boundary decisions
        else:
            weight = 0.2 / max(1, continuation_count)  # Distribute continuation weight
        
        weighted_confidence += decision["confidence_score"] * weight
        total_weight += weight
    
    return round(weighted_confidence / total_weight, 3)


def determine_qa_group_method(block_decisions: List[Dict]) -> str:
    """Determine method based on decision quality and confidence."""
    avg_confidence = calculate_qa_group_confidence(block_decisions)
    
    if avg_confidence >= 0.8:
        return "llm_detection"
    elif avg_confidence >= 0.6:
        return "llm_detection_medium_confidence"
    else:
        return "xml_fallback"


def apply_xml_fallback_grouping(speaker_blocks: List[Dict]) -> List[Dict]:
    """
    Fallback Q&A grouping using XML type attributes when LLM analysis fails.
    """
    log_execution("Applying XML fallback grouping for Q&A boundary detection")
    
    qa_groups = []
    current_group = []
    group_id = 1
    
    for block in speaker_blocks:
        xml_type = block.get("question_answer_flag")
        
        if xml_type == "question":
            # Start new group if we have an existing group
            if current_group:
                qa_groups.append({
                    "qa_group_id": group_id,
                    "start_block_id": current_group[0]["speaker_block_id"],
                    "end_block_id": current_group[-1]["speaker_block_id"],
                    "confidence": 0.5,  # Lower confidence for fallback
                    "method": "xml_fallback",
                    "blocks": current_group
                })
                group_id += 1
            
            current_group = [block]
        
        elif xml_type == "answer" and current_group:
            # Add to current group
            current_group.append(block)
        
        else:
            # Handle blocks without clear XML type
            if current_group:
                current_group.append(block)
    
    # Add final group if exists
    if current_group:
        qa_groups.append({
            "qa_group_id": group_id,
            "start_block_id": current_group[0]["speaker_block_id"],
            "end_block_id": current_group[-1]["speaker_block_id"],
            "confidence": 0.5,
            "method": "xml_fallback",
            "blocks": current_group
        })
    
    return qa_groups


def process_qa_boundaries_with_fallbacks(speaker_blocks: List[Dict], transcript_id: str, enhanced_error_logger: EnhancedErrorLogger) -> List[Dict]:
    """
    Main Q&A boundary processing with comprehensive fallback handling.
    """
    try:
        # Filter to only Q&A sections
        qa_speaker_blocks = [block for block in speaker_blocks 
                           if block.get("section_name") == "Q&A"]
        
        if not qa_speaker_blocks:
            log_execution(f"No Q&A sections found in transcript {transcript_id}")
            return []
        
        log_execution(f"Processing {len(qa_speaker_blocks)} Q&A speaker blocks")
        
        # Process each speaker block for boundary decisions with enhanced state tracking
        all_decisions = []
        qa_state = {
            "current_qa_group_id": None,
            "group_active": False,
            "last_decision_status": None,
            "last_block_id": None,
            "question_start_index": None,
            "extends_question_start": False,
            "next_group_id": 1,  # Auto-increment group ID counter
            "assigned_group_ids": set()  # Track used IDs for validation
        }
        
        for i, block in enumerate(qa_speaker_blocks):
            # Add look-ahead context for better end decisions
            next_block = qa_speaker_blocks[i + 1] if i + 1 < len(qa_speaker_blocks) else None
            qa_state["next_block"] = next_block
            qa_state["current_block_index"] = i
            
            decision = analyze_speaker_block_boundary(i, qa_speaker_blocks, transcript_id, qa_state, enhanced_error_logger)
            
            if decision:
                all_decisions.append(decision)
                
                # Update comprehensive QA state tracking
                block_id = decision["current_block_id"]
                status = decision["group_status"]
                group_id = decision.get("qa_group_id")
                
                # Real-time state validation and updates
                if status == "group_start":
                    # Validate: can't start if group already active
                    if qa_state["group_active"]:
                        log_error(f"Block {block_id}: Cannot start group {group_id} - group {qa_state['current_qa_group_id']} already active", "boundary_detection", {})
                        # Force end previous group first
                        qa_state["group_active"] = False
                        log_execution(f"Block {block_id}: Force-ended previous group {qa_state['current_qa_group_id']}")
                    
                    qa_state["current_qa_group_id"] = group_id
                    qa_state["group_active"] = True
                    qa_state["question_start_index"] = i
                    qa_state["extends_question_start"] = True
                    log_execution(f"Block {block_id}: Started Q&A group {group_id}")
                    
                elif status == "group_continue":
                    # Validate: must have active group
                    if not qa_state["group_active"]:
                        log_error(f"Block {block_id}: Cannot continue group {group_id} - no active group", "boundary_detection", {})
                        # Convert to group start
                        qa_state["current_qa_group_id"] = group_id
                        qa_state["group_active"] = True
                        log_execution(f"Block {block_id}: Converted continue to start for group {group_id}")
                    elif qa_state["current_qa_group_id"] != group_id:
                        log_error(f"Block {block_id}: Group ID mismatch - continuing {group_id} but active is {qa_state['current_qa_group_id']}", "boundary_detection", {})
                    
                elif status == "group_end":
                    # Validate: must have active group to end
                    if not qa_state["group_active"]:
                        log_execution(f"Block {block_id}: Cannot end group {group_id} - no active group")
                    elif qa_state["current_qa_group_id"] != group_id:
                        log_execution(f"Block {block_id}: Ending group {group_id} but active is {qa_state['current_qa_group_id']}")
                    
                    qa_state["group_active"] = False
                    qa_state["extends_question_start"] = False
                    log_execution(f"Block {block_id}: Ended Q&A group {group_id}")
                    
                elif status == "standalone":
                    # Operator blocks don't change group state
                    log_execution(f"Block {block_id}: Standalone block (operator/non-Q&A)")
                
                # Update last decision tracking
                qa_state["last_decision_status"] = status
                qa_state["last_block_id"] = block_id
                
            else:
                # LLM analysis failed, use XML fallback
                log_execution(f"LLM analysis failed for block {block['speaker_block_id']}, falling back to XML grouping")
                return apply_xml_fallback_grouping(qa_speaker_blocks)
        
        # Group decisions into Q&A groups
        qa_groups = []
        current_group_decisions = []
        current_group_id = None
        
        for decision in all_decisions:
            if decision["group_status"] == "group_start":
                # Finalize previous group
                if current_group_decisions:
                    qa_groups.append(finalize_qa_group(current_group_decisions, qa_speaker_blocks))
                
                # Start new group
                current_group_decisions = [decision]
                current_group_id = decision["qa_group_id"]
                
            elif decision["group_status"] == "standalone":
                # Skip standalone blocks (operators) - they don't belong to any group
                continue
                
            elif decision["qa_group_id"] == current_group_id:
                current_group_decisions.append(decision)
            
            else:
                # Handle group ID mismatch (real issues only)
                decision_group_id = decision.get("qa_group_id")
                decision_status = decision.get("group_status")
                log_execution(f"Group ID mismatch in decision sequence: decision has group {decision_group_id} (status: {decision_status}) but current group is {current_group_id}")
                current_group_decisions.append(decision)
        
        # Finalize last group
        if current_group_decisions:
            qa_groups.append(finalize_qa_group(current_group_decisions, qa_speaker_blocks))
        
        log_execution(f"Successfully identified {len(qa_groups)} Q&A groups")
        return qa_groups
        
    except Exception as e:
        error_msg = f"Q&A boundary processing failed: {e}"
        log_error(error_msg, "processing", {})
        enhanced_error_logger.log_processing_error(transcript_id, error_msg)
        return apply_xml_fallback_grouping(speaker_blocks)


def finalize_qa_group(group_decisions: List[Dict], speaker_blocks: List[Dict]) -> Dict:
    """Create final Q&A group from boundary decisions."""
    
    # Get block range
    block_ids = [d["current_block_id"] for d in group_decisions]
    start_block_id = min(block_ids)
    end_block_id = max(block_ids)
    
    # Calculate aggregated confidence and method
    confidence = calculate_qa_group_confidence(group_decisions)
    method = determine_qa_group_method(group_decisions)
    
    # Get group ID (should be consistent across decisions)
    qa_group_id = group_decisions[0]["qa_group_id"]
    
    return {
        "qa_group_id": qa_group_id,
        "start_block_id": start_block_id,
        "end_block_id": end_block_id,
        "confidence": confidence,
        "method": method,
        "block_decisions": group_decisions
    }


def apply_qa_assignments_to_records(records: List[Dict], qa_groups: List[Dict]) -> List[Dict]:
    """
    Apply Q&A group assignments to paragraph records.
    Only adds: qa_group_id, qa_group_confidence, qa_group_method
    """
    
    # Create mapping from speaker block to Q&A group
    block_to_qa_map = {}
    
    for group in qa_groups:
        for block_id in range(group["start_block_id"], group["end_block_id"] + 1):
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
            speaker_block_id = record["speaker_block_id"]
            
            if speaker_block_id in block_to_qa_map:
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
    Process a single transcript for Q&A pairing with per-transcript OAuth refresh.
    """
    
    try:
        # Refresh OAuth token for each transcript
        if not refresh_oauth_token_for_transcript():
            log_error(f"Failed to refresh OAuth token for transcript {transcript_id}", "authentication", {})
            enhanced_error_logger.log_authentication_error(f"Token refresh failed for {transcript_id}")
            return transcript_records, 0  # Return original records without Q&A assignments
        
        log_execution(f"Processing Q&A pairing for transcript: {transcript_id}")
        
        # Group records by speaker blocks
        speaker_blocks = group_records_by_speaker_block(transcript_records)
        
        # Process Q&A boundaries
        qa_groups = process_qa_boundaries_with_fallbacks(speaker_blocks, transcript_id, enhanced_error_logger)
        
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

        # Step 11: Save output
        log_console("Step 11: Saving Q&A paired content...")
        output_data = {
            "schema_version": "1.0",
            "processing_timestamp": datetime.now().isoformat(),
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
        output_filename = "stage_05_qa_paired_content.json"
        output_file_path = nas_path_join(output_path, output_filename)

        nas_create_directory_recursive(nas_conn, output_path)

        output_json = json.dumps(output_data, indent=2)
        output_bytes = io.BytesIO(output_json.encode("utf-8"))

        if nas_upload_file(nas_conn, output_bytes, output_file_path):
            log_execution("Q&A paired content saved successfully", {
                "output_path": output_file_path,
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
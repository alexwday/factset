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
        "state_config"
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

    # Validate state config structure
    state_config = stage_05_qa_pairing_config["state_config"]
    required_state_params = ["lookahead_blocks", "default_decision"]

    for param in required_state_params:
        if param not in state_config:
            error_msg = f"Missing required state config parameter: {param}"
            log_error(error_msg, "config_validation", {"missing_parameter": f"state_config.{param}"})
            raise ValueError(error_msg)

    # Validate monitored institutions
    if not config["monitored_institutions"]:
        error_msg = "monitored_institutions cannot be empty"
        log_error(error_msg, "config_validation", {"section": "monitored_institutions"})
        raise ValueError(error_msg)

    log_execution("Configuration validation successful", {
        "total_institutions": len(config["monitored_institutions"]),
        "llm_model": llm_config["model"],
        "state_config": state_config
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

def create_analyst_session_context(qa_speaker_blocks: List[Dict], 
                                  current_block_index: int,
                                  qa_start_index: int,
                                  transcript_metadata: Dict) -> Dict:
    """
    Create focused context window for analyst session boundary detection.
    
    Returns structured context with:
    1. All previous speaker blocks of current QA ID (in full)
    2. Current speaker block being examined
    3. Next 4 speaker blocks as lookahead context
    """
    current_block = qa_speaker_blocks[current_block_index]
    
    # 1. All previous blocks of current QA ID (from qa_start_index to current_block_index - 1)
    previous_blocks = qa_speaker_blocks[qa_start_index:current_block_index]
    
    # 2. Next blocks for lookahead context (configurable)
    lookahead_count = config["stage_05_qa_pairing"]["state_config"]["lookahead_blocks"]
    next_blocks = qa_speaker_blocks[current_block_index + 1:current_block_index + 1 + lookahead_count]
    
    return {
        "transcript_metadata": transcript_metadata,
        "previous_qa_blocks": previous_blocks,
        "current_block": current_block,
        "next_blocks": next_blocks,
        "current_block_index": current_block_index,
        "qa_start_index": qa_start_index
    }


def format_analyst_session_context(context_data: Dict) -> str:
    """
    Format structured context for analyst session boundary detection.
    Creates numbered speaker blocks with full content for LLM analysis.
    """
    context_sections = []
    block_counter = 1
    
    # Company and transcript context
    transcript_metadata = context_data.get("transcript_metadata", {})
    company_name = transcript_metadata.get("company_name", "Unknown Company")
    transcript_title = transcript_metadata.get("transcript_title", "Earnings Call Transcript")
    
    context_sections.append(f"<company>{company_name}</company>")
    context_sections.append(f"<transcript_title>{transcript_title}</transcript_title>")
    context_sections.append("")
    
    # 1. Previous blocks of current QA ID (if any)
    previous_blocks = context_data.get("previous_qa_blocks", [])
    if previous_blocks:
        context_sections.append("<previous_qa_blocks>")
        context_sections.append("All previous speaker blocks of the current analyst session:")
        for block in previous_blocks:
            context_sections.append(format_numbered_speaker_block(block, block_counter))
            block_counter += 1
        context_sections.append("</previous_qa_blocks>")
        context_sections.append("")
    
    # 2. Current block being examined
    current_block = context_data["current_block"]
    context_sections.append("<current_decision_block>")
    context_sections.append("🎯 CURRENT BLOCK - Make decision for this block:")
    context_sections.append(format_numbered_speaker_block(current_block, block_counter))
    context_sections.append("</current_decision_block>")
    block_counter += 1
    context_sections.append("")
    
    # 3. Next blocks for lookahead context (configurable count)
    next_blocks = context_data.get("next_blocks", [])
    if next_blocks:
        context_sections.append("<lookahead_context>")
        context_sections.append("Next speaker blocks for context (to help determine boundaries):")
        for block in next_blocks:
            context_sections.append(format_numbered_speaker_block(block, block_counter))
            block_counter += 1
        context_sections.append("</lookahead_context>")
    
    return "\n".join(context_sections)


def format_numbered_speaker_block(block: Dict, block_number: int) -> str:
    """Format a speaker block with clear numbering and full content for LLM analysis."""
    paragraphs_text = []
    
    for para in block["paragraphs"]:
        # Include full paragraph content without modification
        content = para['paragraph_content'].strip()
        paragraphs_text.append(content)
    
    # Use the actual speaker field as provided in the data
    speaker = block['speaker']
    
    # Join paragraphs with clear separation
    full_content = "\n".join(paragraphs_text)
    
    return f"""Speaker Block {block_number}:
Speaker: {speaker}
Content: {full_content}"""


def create_analyst_session_prompt(formatted_context: str, current_qa_id: int) -> str:
    """
    Create research-backed system prompt for analyst session boundary detection using XML structure.
    Based on Anthropic 2024 best practices for Claude prompting.
    """
    return f"""<task>
You are analyzing an earnings call Q&A transcript to determine when analyst sessions end. Each analyst session (QA ID) captures one analyst's complete interaction from start to finish.
</task>

<context>
You are examining Speaker Block in position "🎯 CURRENT BLOCK" within the transcript context below. This block is currently part of QA ID {current_qa_id}.
</context>

<objective>
Determine whether the current speaker block should CONTINUE the current analyst session (QA ID {current_qa_id}) or END it.

When you choose END:
- The current block becomes the final block of QA ID {current_qa_id}
- The very next speaker block will automatically start QA ID {current_qa_id + 1}
</objective>

<instructions>
Analyze the conversation flow to identify when one analyst's complete session ends and a new analyst's session should begin.

<decision_criteria>
CONTINUE if:
- Same analyst is still speaking/asking follow-up questions
- Executive is still responding to the current analyst's questions
- There's ongoing back-and-forth between current analyst and executives
- Operator comments are brief transitions within the same analyst session

END if:
- Current analyst's questions are completely answered
- Next speaker block clearly introduces a NEW analyst
- Operator transitions to "next question from [different analyst]"
- Current exchange has naturally concluded before new topic/analyst
</decision_criteria>

<key_principles>
1. **Analyst Session Focus**: Each QA ID represents ONE analyst's complete interaction
2. **Sequential Order**: Analysts take turns - one finishes completely before next begins  
3. **Content Over Names**: Focus on conversation patterns, not just speaker names
4. **Operator Integration**: Operators introducing new analysts signal session transitions
5. **Complete Exchanges**: Ensure current analyst's questions are fully addressed before ending
</key_principles>

<critical_reminder>
Review the lookahead context carefully. When you choose END:
- Verify the current block truly concludes the analyst's session
- Verify the next block appropriately starts a new analyst's session
- Remember that operators often introduce new analysts with phrases like "next question from..."
</critical_reminder>
</instructions>

{formatted_context}

<output_format>
Make your decision by calling the boundary_decision function with either "continue" or "end".
</output_format>"""


def create_analyst_session_boundary_tool():
    """Create simplified function calling tool for analyst session boundary detection."""
    
    return [
        {
            "type": "function",  
            "function": {
                "name": "boundary_decision",
                "description": "Decide whether the current speaker block continues or ends the current analyst session",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "decision": {
                            "type": "string",
                            "enum": ["continue", "end"],
                            "description": "continue: Current speaker block continues the current analyst session. end: Current speaker block ends the current analyst session and next block starts new session."
                        }
                    },
                    "required": ["decision"]
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


# Removed old validation and state management functions - replaced with simplified state-based approach


def make_qa_boundary_decision(context_data: Dict,
                            current_block_id: int,
                            current_qa_id: int,
                            transcript_id: str,
                            enhanced_error_logger: EnhancedErrorLogger) -> Optional[str]:
    """
    Make simplified LLM decision for analyst session boundary detection.
    Returns "continue" or "end" based on context analysis.
    """
    global llm_client
    
    try:
        # Format context for LLM analysis
        formatted_context = format_analyst_session_context(context_data)
        prompt = create_analyst_session_prompt(formatted_context, current_qa_id)
        
        # Retry logic for LLM calls
        max_retries = 3 
        retry_delay = 1  # seconds
        
        for attempt in range(1, max_retries + 1):
            try:
                log_execution(f"Block {current_block_id} LLM analysis attempt {attempt}/{max_retries}")
                
                # Make LLM API call
                response = llm_client.chat.completions.create(
                    model=config["stage_05_qa_pairing"]["llm_config"]["model"],
                    messages=[{"role": "user", "content": prompt}],
                    tools=create_analyst_session_boundary_tool(),
                    tool_choice="required", 
                    temperature=config["stage_05_qa_pairing"]["llm_config"]["temperature"],
                    max_tokens=config["stage_05_qa_pairing"]["llm_config"]["max_tokens"]
                )
                
                # Parse response
                if not response.choices[0].message.tool_calls:
                    error_msg = f"No tool call response for block {current_block_id}"
                    if attempt == max_retries:
                        log_error(f"{error_msg} (final attempt)", "boundary_detection", {})
                        enhanced_error_logger.log_boundary_error(transcript_id, current_block_id, f"{error_msg} after {max_retries} attempts")
                        return None
                    else:
                        log_execution(f"Block {current_block_id}: {error_msg}, retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        continue
                
                # Extract decision
                try:
                    tool_call = response.choices[0].message.tool_calls[0]
                    result = json.loads(tool_call.function.arguments)
                    decision = result["decision"]
                    
                    if decision not in ["continue", "end"]:
                        raise ValueError(f"Invalid decision: {decision}")
                        
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    error_msg = f"Failed to parse LLM response for block {current_block_id}: {e}"
                    if attempt == max_retries:
                        log_error(f"{error_msg} (final attempt)", "boundary_detection", {})
                        enhanced_error_logger.log_boundary_error(transcript_id, current_block_id, f"{error_msg} after {max_retries} attempts")
                        return None
                    else:
                        log_execution(f"Block {current_block_id}: {error_msg}, retrying in {retry_delay}s...")
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
                    log_execution(f"Block {current_block_id} tokens - input: {response.usage.prompt_tokens}, output: {response.usage.completion_tokens}, total: {response.usage.total_tokens}, cost: ${cost_info['total_cost']:.4f}")
                    
                    # Accumulate costs for final summary
                    enhanced_error_logger.accumulate_costs(token_usage)
                
                # Log successful decision
                attempt_info = f" (attempt {attempt})" if attempt > 1 else ""
                log_execution(f"Block {current_block_id} decision{attempt_info}: {decision} for QA ID {current_qa_id}")
                
                return decision
                
            except Exception as e:
                error_msg = f"LLM analysis failed for block {current_block_id}: {e}"
                if attempt == max_retries:
                    log_error(f"{error_msg} (final attempt)", "boundary_detection", {})
                    enhanced_error_logger.log_boundary_error(transcript_id, current_block_id, f"{error_msg} after {max_retries} attempts")
                    return None
                else:
                    log_execution(f"Block {current_block_id}: {error_msg}, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
        
        # Should never reach here, but safety fallback
        return None
        
    except Exception as e:
        error_msg = f"Context preparation failed for block {current_block_id}: {e}"
        log_error(error_msg, "boundary_detection", {})
        enhanced_error_logger.log_boundary_error(transcript_id, current_block_id, error_msg)
        return None


# Removed old confidence calculation functions - replaced with state-based approach




def process_qa_boundaries_state_based(speaker_blocks: List[Dict], transcript_id: str, transcript_metadata: Dict, enhanced_error_logger: EnhancedErrorLogger) -> List[Dict]:
    """
    State-based Q&A boundary processing focused on analyst sessions.
    
    State 1: Auto-assign first block to QA ID 1
    State 2: LLM decides continue vs end current QA ID  
    State 3: Auto-assign next block to next QA ID, return to State 2
    """
    try:
        # Filter to only Q&A sections
        qa_speaker_blocks = [block for block in speaker_blocks 
                           if block.get("section_name") == "Q&A"]
        
        if not qa_speaker_blocks:
            log_execution(f"No Q&A sections found in transcript {transcript_id}")
            return []
        
        log_execution(f"Processing {len(qa_speaker_blocks)} Q&A speaker blocks using state-based approach")
        
        # Initialize state machine
        qa_state = {
            "current_qa_id": 1,
            "current_qa_start_index": 0,  # Index where current QA ID started
            "state": "auto_assign_start",  # States: auto_assign_start, llm_decision, auto_assign_next
            "all_decisions": []
        }
        
        # State 1: Auto-assign first block
        first_block = qa_speaker_blocks[0]
        first_decision = {
            "current_block_id": first_block["speaker_block_id"],
            "qa_group_id": qa_state["current_qa_id"],
            "decision_type": "auto_assigned_start",
            "state": "auto_assign_start"
        }
        qa_state["all_decisions"].append(first_decision)
        log_execution(f"State 1: Auto-assigned block {first_block['speaker_block_id']} to QA ID {qa_state['current_qa_id']}")
        
        # Process remaining blocks using while loop to handle skipped iterations
        i = 1
        while i < len(qa_speaker_blocks):
            current_block = qa_speaker_blocks[i]
            current_block_id = current_block["speaker_block_id"]
            
            # State 2: LLM Decision (continue vs end current QA ID)
            qa_state["state"] = "llm_decision"
            
            # Create focused context window for LLM
            context_data = create_analyst_session_context(
                qa_speaker_blocks, 
                i, 
                qa_state["current_qa_start_index"],
                transcript_metadata
            )
            
            # Make LLM decision
            llm_decision = make_qa_boundary_decision(
                context_data, 
                current_block_id, 
                qa_state["current_qa_id"],
                transcript_id,
                enhanced_error_logger
            )
            
            if llm_decision == "continue":
                # Continue current QA ID
                decision = {
                    "current_block_id": current_block_id,
                    "qa_group_id": qa_state["current_qa_id"],
                    "decision_type": "llm_continue",
                    "state": "llm_decision"
                }
                qa_state["all_decisions"].append(decision)
                log_execution(f"State 2: Block {current_block_id} continues QA ID {qa_state['current_qa_id']}")
                i += 1  # Move to next block
                
            elif llm_decision == "end":
                # End current QA ID and auto-assign next block to next QA ID
                
                # First, add the current block as the end of current QA ID
                end_decision = {
                    "current_block_id": current_block_id,
                    "qa_group_id": qa_state["current_qa_id"],
                    "decision_type": "llm_end",
                    "state": "llm_decision"
                }
                qa_state["all_decisions"].append(end_decision)
                log_execution(f"State 2: Block {current_block_id} ends QA ID {qa_state['current_qa_id']}")
                
                # State 3: Auto-assign next block (if exists) to next QA ID
                if i + 1 < len(qa_speaker_blocks):
                    qa_state["current_qa_id"] += 1
                    qa_state["current_qa_start_index"] = i + 1
                    qa_state["state"] = "auto_assign_next"
                    
                    next_block = qa_speaker_blocks[i + 1]
                    next_decision = {
                        "current_block_id": next_block["speaker_block_id"],
                        "qa_group_id": qa_state["current_qa_id"],
                        "decision_type": "auto_assigned_next",
                        "state": "auto_assign_next"
                    }
                    qa_state["all_decisions"].append(next_decision)
                    log_execution(f"State 3: Auto-assigned block {next_block['speaker_block_id']} to QA ID {qa_state['current_qa_id']}")
                    
                    # Skip the next block since we've already processed it, move to i+2
                    i += 2
                else:
                    # No more blocks, end processing
                    i += 1
                    
            else:
                # LLM decision failed - treat as continue to be conservative
                log_execution(f"LLM decision failed for block {current_block_id}, defaulting to continue")
                decision = {
                    "current_block_id": current_block_id,
                    "qa_group_id": qa_state["current_qa_id"],
                    "decision_type": "default_continue",
                    "state": "llm_decision_failed"
                }
                qa_state["all_decisions"].append(decision)
                i += 1  # Move to next block
        
        # Convert decisions to QA groups
        qa_groups = convert_decisions_to_qa_groups(qa_state["all_decisions"])
        
        log_execution(f"Successfully identified {len(qa_groups)} Q&A groups using state-based approach")
        return qa_groups
        
    except Exception as e:
        error_msg = f"State-based Q&A boundary processing failed: {e}"
        log_error(error_msg, "processing", {})
        enhanced_error_logger.log_processing_error(transcript_id, error_msg)
        return []


def convert_decisions_to_qa_groups(all_decisions: List[Dict]) -> List[Dict]:
    """Convert state machine decisions to QA groups for final output."""
    qa_groups = []
    current_group_decisions = []
    current_qa_id = None
    
    for decision in all_decisions:
        qa_id = decision["qa_group_id"]
        
        if qa_id != current_qa_id:
            # Finalize previous group
            if current_group_decisions:
                qa_groups.append(create_qa_group_from_decisions(current_group_decisions))
            
            # Start new group
            current_group_decisions = [decision]
            current_qa_id = qa_id
        else:
            # Continue current group
            current_group_decisions.append(decision)
    
    # Finalize last group
    if current_group_decisions:
        qa_groups.append(create_qa_group_from_decisions(current_group_decisions))
    
    return qa_groups


def create_qa_group_from_decisions(decisions: List[Dict]) -> Dict:
    """Create QA group object from list of decisions."""
    block_ids = [d["current_block_id"] for d in decisions]
    qa_id = decisions[0]["qa_group_id"]
    
    # Determine method based on decision types
    decision_types = [d["decision_type"] for d in decisions]
    if any("auto_assigned" in dt for dt in decision_types):
        method = "state_based_auto_assignment"
    else:
        method = "state_based_llm_decision"
    
    return {
        "qa_group_id": qa_id,
        "start_block_id": min(block_ids),
        "end_block_id": max(block_ids),
        "confidence": 1.0,  # High confidence in state-based approach
        "method": method,
        "block_decisions": decisions
    }


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
    Process a single transcript for Q&A pairing using state-based approach with per-transcript OAuth refresh.
    """
    
    try:
        # Refresh OAuth token for each transcript
        if not refresh_oauth_token_for_transcript():
            log_error(f"Failed to refresh OAuth token for transcript {transcript_id}", "authentication", {})
            enhanced_error_logger.log_authentication_error(f"Token refresh failed for {transcript_id}")
            return transcript_records, 0  # Return original records without Q&A assignments
        
        log_execution(f"Processing Q&A pairing for transcript: {transcript_id}")
        
        # Extract transcript metadata for context
        transcript_metadata = extract_transcript_metadata(transcript_records)
        
        # Group records by speaker blocks
        speaker_blocks = group_records_by_speaker_block(transcript_records)
        
        # Process Q&A boundaries using state-based approach
        qa_groups = process_qa_boundaries_state_based(
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
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

    # Note: state_config is no longer used in sliding window approach
    # Legacy validation removed - sliding window approach uses direct LLM calls without lookahead configuration

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


def create_breakpoint_detection_prompt(indexed_blocks: List[Dict], company_name: str, transcript_title: str, current_qa_id: int) -> str:
    """
    Create prompt for analyst breakpoint detection in configurable-size sliding windows.
    Instructs LLM to find where the next analyst turn begins or skip to next batch.
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
Your task is to identify where the NEXT analyst's turn begins in these indexed speaker blocks.
</task>

<context>
You are currently processing QA ID {current_qa_id}. You need to find where this analyst's turn ends and the next analyst's turn begins.
</context>

<objective>
Examine the {len(indexed_blocks)} indexed speaker blocks below and determine:
1. SKIP: If the current analyst's turn continues beyond these blocks (breakpoint not found)
2. BREAKPOINT: If you find where the next analyst's turn begins (provide the index 1-{len(indexed_blocks)})

When you choose SKIP: These {len(indexed_blocks)} blocks will be held and combined with the next batch of blocks to find the true breakpoint.
When you choose BREAKPOINT: The blocks up to (but not including) your chosen index belong to the current analyst.
</objective>

<instructions>
Look for clear signals that indicate when one analyst's complete session ends and a new analyst begins:

<breakpoint_indicators>
STRONG breakpoint signals:
- Operator: "Next question from [New Analyst Name], [Firm]"
- New speaker introduction after analyst questions are fully answered
- Clear topic shift with new analyst acknowledgment
- Operator transitioning to different analyst

CONTINUE signals (choose SKIP):
- Same analyst asking follow-up questions
- Executive still responding to current analyst's questions
- Ongoing back-and-forth within same analyst session
- Brief operator comments within same analyst conversation
</breakpoint_indicators>

<decision_logic>
1. **Analyst Session Focus**: Each QA ID should capture ONE analyst's complete interaction
2. **Complete Conversations**: Ensure current analyst's questions are fully addressed before breakpoint
3. **Clear Transitions**: Look for explicit operator transitions or new analyst introductions
4. **Natural Boundaries**: Breakpoints should feel natural, not mid-conversation
</decision_logic>
</instructions>

<indexed_speaker_blocks>
{blocks_text}
</indexed_speaker_blocks>

<output_format>
Call the analyst_breakpoint function with:
- "skip" if the breakpoint is not in these {len(indexed_blocks)} blocks
- "breakpoint" with the index (1-{len(indexed_blocks)}) where the next analyst turn begins
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


def format_speaker_block_content(paragraphs: List[Dict]) -> str:
    """Format paragraph content from a speaker block for prompt display."""
    content_parts = []
    for paragraph in paragraphs:
        content = paragraph.get('paragraph_content', '').strip()
        if content:
            content_parts.append(content)
    return "\n".join(content_parts)


def create_breakpoint_detection_tool():
    """Create function calling tool for analyst breakpoint detection in configurable-size windows."""
    
    return [
        {
            "type": "function",  
            "function": {
                "name": "analyst_breakpoint",
                "description": "Identify where the next analyst turn begins in the indexed speaker blocks, or skip to examine the next batch. When you skip, the current blocks will be held and combined with blocks from the next window to find the true breakpoint.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["skip", "breakpoint"],
                            "description": "skip: The end of current analyst turn is not in these blocks, examine next batch and combine with current blocks. breakpoint: Found where next analyst turn begins."
                        },
                        "index": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Required when action is 'breakpoint'. The 1-based index where the next analyst turn begins."
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


def create_indexed_speaker_blocks(speaker_blocks: List[Dict], start_index: int, window_size: int) -> List[Dict]:
    """
    Create 1-based indexed speaker blocks for sliding window processing.
    Returns list of dicts with 'index' (1-based) and 'block' (speaker block data).
    """
    indexed_blocks = []
    end_index = min(start_index + window_size, len(speaker_blocks))
    
    for i in range(start_index, end_index):
        indexed_block = {
            "index": len(indexed_blocks) + 1,  # 1-based indexing
            "absolute_index": i,  # Original position in full speaker_blocks array
            "block": speaker_blocks[i]
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
    Use LLM to detect analyst breakpoint in configurable-size window.
    Returns dict with action ("skip" or "breakpoint") and optional index.
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
                
                # Log the prompt being sent (for debugging)
                log_execution("=== LLM BREAKPOINT DETECTION PROMPT ===")
                log_execution(f"Window: {len(indexed_blocks)} blocks (indices 1-{len(indexed_blocks)})")
                
                # Show speaker preview for each index
                speaker_preview = []
                for i, block_data in enumerate(indexed_blocks, 1):
                    block = block_data["block"]
                    # Get first 100 chars of content for preview
                    first_para = block['paragraphs'][0] if block['paragraphs'] else {}
                    content_preview = first_para.get('paragraph_content', '')[:100] + "..." if first_para.get('paragraph_content', '') else "[No content]"
                    speaker_preview.append(f"Index {i}: {block['speaker']} - {content_preview}")
                
                for preview in speaker_preview:
                    log_execution(preview)
                
                log_execution("=== FULL PROMPT SENT TO LLM ===")
                log_execution(prompt)
                log_execution("=== END PROMPT ===")
                
                # Make LLM API call
                response = llm_client.chat.completions.create(
                    model=config["stage_05_qa_pairing"]["llm_config"]["model"],
                    messages=[{"role": "user", "content": prompt}],
                    tools=create_breakpoint_detection_tool(),
                    tool_choice="required", 
                    temperature=config["stage_05_qa_pairing"]["llm_config"]["temperature"],
                    max_tokens=config["stage_05_qa_pairing"]["llm_config"]["max_tokens"]
                )
                
                # Log the raw response from LLM
                log_execution("=== LLM RESPONSE RECEIVED ===")
                if response.choices[0].message.tool_calls:
                    for i, tool_call in enumerate(response.choices[0].message.tool_calls):
                        log_execution(f"Tool Call {i+1}: {tool_call.function.name}")
                        log_execution(f"Raw Arguments: {tool_call.function.arguments}")
                else:
                    log_execution("No tool calls in response")
                    if hasattr(response.choices[0].message, 'content') and response.choices[0].message.content:
                        log_execution(f"Text Content: {response.choices[0].message.content}")
                log_execution("=== END LLM RESPONSE ===")
                
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
                
                # Log successful decision with detailed context
                attempt_info = f" (attempt {attempt})" if attempt > 1 else ""
                if parsed_result["action"] == "breakpoint":
                    breakpoint_index = parsed_result['index']
                    log_execution(f"🎯 BREAKPOINT DETECTED{attempt_info}: INDEX {breakpoint_index} for QA ID {current_qa_id}")
                    
                    # Show what's at the breakpoint for context
                    if breakpoint_index <= len(indexed_blocks):
                        breakpoint_block = indexed_blocks[breakpoint_index-1]["block"]  # Convert to 0-based
                        first_para = breakpoint_block['paragraphs'][0] if breakpoint_block['paragraphs'] else {}
                        breakpoint_content = first_para.get('paragraph_content', '')[:200] + "..." if first_para.get('paragraph_content', '') else "[No content]"
                        log_execution(f"🎯 Breakpoint at Index {breakpoint_index}: {breakpoint_block['speaker']} - {breakpoint_content}")
                    
                    log_execution(f"✅ QA ID {current_qa_id} will include indices 1-{breakpoint_index-1} ({breakpoint_index-1} blocks)")
                    log_execution(f"🔄 Next QA ID {current_qa_id+1} will start at index {breakpoint_index}")
                else:
                    log_execution(f"⏭️ SKIP DECISION{attempt_info} for QA ID {current_qa_id}")
                    log_execution(f"🔄 Will hold all {len(indexed_blocks)} blocks and examine next window")
                
                log_execution("=" * 80)  # Separator for readability
                
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
            log_execution(f"🔍 VALIDATION ATTEMPT {validation_attempt}/{max_validation_attempts} for QA ID {qa_id}")
            
            # Log what we're validating
            log_execution(f"📋 Proposed QA ID {qa_id}: {len(proposed_qa_blocks)} blocks")
            for i, block in enumerate(proposed_qa_blocks, 1):
                first_para = block['paragraphs'][0] if block['paragraphs'] else {}
                content_preview = first_para.get('paragraph_content', '')[:100] + "..." if first_para.get('paragraph_content', '') else "[No content]"
                log_execution(f"  Block {i}: {block['speaker']} - {content_preview}")
            
            log_execution(f"📋 Remaining blocks for context: {len(remaining_blocks[:5])}")
            for i, block_data in enumerate(remaining_blocks[:3], 1):  # Show first 3 for context
                block = block_data["block"] if isinstance(block_data, dict) and "block" in block_data else block_data
                first_para = block['paragraphs'][0] if block['paragraphs'] else {}
                content_preview = first_para.get('paragraph_content', '')[:100] + "..." if first_para.get('paragraph_content', '') else "[No content]"
                log_execution(f"  Remaining {i}: {block['speaker']} - {content_preview}")
            
            # Create validation prompt
            prompt = create_validation_prompt(proposed_qa_blocks, remaining_blocks, company_name, transcript_title, qa_id)
            
            log_execution("=== VALIDATION PROMPT SENT TO LLM ===")
            log_execution(prompt)
            log_execution("=== END VALIDATION PROMPT ===")
            
            # Make LLM API call
            response = llm_client.chat.completions.create(
                model=config["stage_05_qa_pairing"]["llm_config"]["model"],
                messages=[{"role": "user", "content": prompt}],
                tools=create_validation_tool(),
                tool_choice="required", 
                temperature=config["stage_05_qa_pairing"]["llm_config"]["temperature"],
                max_tokens=config["stage_05_qa_pairing"]["llm_config"]["max_tokens"]
            )
            
            # Log validation response
            log_execution("=== VALIDATION RESPONSE RECEIVED ===")
            if response.choices[0].message.tool_calls:
                for i, tool_call in enumerate(response.choices[0].message.tool_calls):
                    log_execution(f"Tool Call {i+1}: {tool_call.function.name}")
                    log_execution(f"Raw Arguments: {tool_call.function.arguments}")
            else:
                log_execution("No tool calls in validation response")
                if hasattr(response.choices[0].message, 'content') and response.choices[0].message.content:
                    log_execution(f"Text Content: {response.choices[0].message.content}")
            log_execution("=== END VALIDATION RESPONSE ===")
            
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
                    log_execution(f"✅ VALIDATION ACCEPTED{attempt_info} for QA ID {qa_id}")
                    log_execution(f"📋 QA ID {qa_id} confirmed with {len(proposed_qa_blocks)} blocks")
                    log_execution("=" * 80)  # Separator
                    return True
                else:
                    log_execution(f"❌ VALIDATION REJECTED{attempt_info} for QA ID {qa_id}")
                    # Reject - try again if attempts remaining
                    if validation_attempt == max_validation_attempts:
                        log_execution(f"⚠️ Validation rejected after {max_validation_attempts} attempts, accepting assignment anyway")
                        log_execution("=" * 80)  # Separator
                        return True  # Accept on final failure
                    else:
                        log_execution(f"🔄 Validation rejected, will retry breakpoint detection")
                        log_execution("=" * 80)  # Separator
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
        # Filter to only Q&A sections
        qa_speaker_blocks = [block for block in speaker_blocks 
                           if block.get("section_name") == "Q&A"]
        
        if not qa_speaker_blocks:
            log_execution(f"No Q&A sections found in transcript {transcript_id}")
            return []
        
        log_execution(f"Processing {len(qa_speaker_blocks)} Q&A speaker blocks using sliding window approach")
        
        # Initialize sliding window state
        sliding_state = {
            "current_qa_id": 1,
            "current_window_start": 0,  # Absolute index in qa_speaker_blocks
            "held_blocks": [],  # Accumulated blocks from "skip" decisions
            "processing_complete": False,
            "all_qa_groups": []
        }
        
        # Extract metadata for prompts
        company_name = transcript_metadata.get("company_name", "Unknown Company")
        transcript_title = transcript_metadata.get("transcript_title", "Earnings Call Transcript")
        
        # Main processing loop
        while not sliding_state["processing_complete"] and sliding_state["current_window_start"] < len(qa_speaker_blocks):
            
            # Create current window (configurable size)
            window_size = config["stage_05_qa_pairing"]["window_size"]
            current_window = create_indexed_speaker_blocks(
                qa_speaker_blocks, 
                sliding_state["current_window_start"], 
                window_size
            )
            
            if not current_window:
                # No more blocks to process
                sliding_state["processing_complete"] = True
                break
            
            log_execution("🔄" + "="*79)
            log_execution(f"🎯 PROCESSING WINDOW for QA ID {sliding_state['current_qa_id']}")
            log_execution(f"📍 Window start: block {sliding_state['current_window_start']}")
            log_execution(f"📊 Current window: {len(current_window)} blocks")
            log_execution(f"💼 Held blocks: {len(sliding_state['held_blocks'])} blocks")
            
            if sliding_state['held_blocks']:
                log_execution(f"📝 Total context if skip: {len(sliding_state['held_blocks']) + len(current_window)} blocks")
            
            log_execution("🔄" + "="*79)
            
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
                # LLM call failed - treat as skip to be conservative
                log_execution(f"⚠️ BREAKPOINT DETECTION FAILED: Treating as skip for QA ID {sliding_state['current_qa_id']}")
                sliding_state["held_blocks"].extend([block_data["block"] for block_data in current_window])
                new_held_count = len(sliding_state["held_blocks"])
                new_window_start = sliding_state["current_window_start"] + len(current_window)
                sliding_state["current_window_start"] = new_window_start
                
                log_execution(f"📝 Total held blocks now: {new_held_count}")
                log_execution(f"📍 Next window will start at: {new_window_start}")
                log_execution("⚠️" + "="*79)
                continue
            
            if breakpoint_result["action"] == "skip":
                # Hold current blocks and advance window
                log_execution(f"⏭️ SKIP ACTION: Holding {len(current_window)} blocks for QA ID {sliding_state['current_qa_id']}")
                sliding_state["held_blocks"].extend([block_data["block"] for block_data in current_window])
                new_held_count = len(sliding_state["held_blocks"])
                new_window_start = sliding_state["current_window_start"] + len(current_window)
                sliding_state["current_window_start"] = new_window_start
                
                log_execution(f"📝 Total held blocks now: {new_held_count}")
                log_execution(f"📍 Next window will start at: {new_window_start}")
                log_execution("⏭️" + "="*79)
                continue
                
            elif breakpoint_result["action"] == "breakpoint":
                # Found breakpoint - create qa_id assignment
                breakpoint_index = breakpoint_result["index"]  # 1-based index within current window
                
                log_execution(f"Breakpoint detected at index {breakpoint_index} for QA ID {sliding_state['current_qa_id']}")
                
                # Blocks for current qa_id: held_blocks + blocks up to (but not including) breakpoint
                current_window_blocks = [block_data["block"] for block_data in current_window[:breakpoint_index-1]]
                proposed_qa_blocks = sliding_state["held_blocks"] + current_window_blocks
                
                # Remaining blocks in window (for validation context)
                remaining_window_blocks = current_window[breakpoint_index-1:]
                
                if not proposed_qa_blocks:
                    # Edge case: no blocks for this qa_id (shouldn't happen with proper logic)
                    log_execution(f"No blocks proposed for QA ID {sliding_state['current_qa_id']}, skipping validation")
                    validated = True
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
                    # Create and store qa_group
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
                    
                    # Advance state for next qa_id
                    sliding_state["current_qa_id"] += 1
                    sliding_state["held_blocks"] = []  # Clear held blocks
                    
                    # Advance window to breakpoint (where next analyst starts)
                    breakpoint_absolute_index = sliding_state["current_window_start"] + breakpoint_index - 1
                    sliding_state["current_window_start"] = breakpoint_absolute_index
                    
                else:
                    # Validation failed - retry breakpoint detection on same window
                    log_execution(f"Validation failed for QA ID {sliding_state['current_qa_id']}, retrying breakpoint detection")
                    # Continue with same window state - will retry breakpoint detection
                    
            else:
                # Invalid action (shouldn't happen with validation)
                log_execution(f"Invalid breakpoint action: {breakpoint_result.get('action')}")
                sliding_state["processing_complete"] = True
        
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
        
        log_execution(f"Successfully identified {len(sliding_state['all_qa_groups'])} Q&A groups using sliding window approach")
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
    
    for group in qa_groups:
        # Get all speaker block IDs from the group's speaker_blocks
        speaker_blocks = group.get("speaker_blocks", [])
        for speaker_block in speaker_blocks:
            block_id = speaker_block["speaker_block_id"]
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